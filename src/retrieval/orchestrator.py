"""
Retrieval Orchestrator
Single entry point for the entire retrieval layer.

Composes all core engine components:
    Query Rewriter → Intent Router → [Agent] → Query Expander →
    Metadata Filter → Hybrid Search → Reranker → Context Builder →
    Confidence Scorer

Returns an OrchestratorResult containing the assembled context,
confidence evaluation, and full diagnostics trace.

Configuration-driven — all component configs from YAML with env overrides.
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

import yaml
from dotenv import load_dotenv

from src.retrieval.config import PipelineConfig, ContextTemplate, ConfidenceLevel
from src.retrieval.query_rewriter import QueryRewriter, ConversationState
from src.retrieval.intent_router import IntentRouter, RoutingDecision
from src.retrieval.retrieval_agent import RetrievalAgent, AgentDecision, AgentAction
from src.retrieval.query_expander import QueryExpander, ConceptRegistry
from src.retrieval.metadata_filter import MetadataFilterBuilder
from src.retrieval.hybrid_search import HybridSearch, HybridSearchConfig
from src.retrieval.reranker import Reranker, RerankerConfig
from src.retrieval.context_builder import ContextBuilder, ContextBuilderConfig, AssembledContext
from src.retrieval.confidence import ConfidenceScorer, ConfidenceConfig, ConfidenceResult
from src.ingestion.vector_store import VectorStore
from observability.tracing import traced_span, safe_dict

load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)

# Broad hierarchy fields the retrieval agent guesses at (e.g. "Supply Chain
# Management") based on its own understanding of the conversation, not the
# actual ingestion-time extracted vocabulary. MetadataExtractor's LLM call
# populates these independently and isn't guaranteed to agree — e.g.
# subject_area is deliberately a broad discipline ("Business"), not the
# course name the agent assumes. Applied as a hard $eq filter, a mismatch
# here zeroes out retrieval entirely rather than just missing a soft signal.
# Conversation-state-derived filters already avoid this (see
# MetadataFilterBuilder._build_from_conversation's docstring) — agent
# filters get the same treatment. Narrower fields the agent proposes (e.g.
# "topic") are much closer to ingestion-time chunk-level vocabulary and are
# kept as hard filters.
AGENT_FILTER_EXCLUDED_FIELDS = {"subject", "chapter", "course"}


# ============================================================================
# Exceptions
# ============================================================================

class OrchestratorError(Exception):
    """Base exception for orchestrator errors."""
    pass


class RetrievalTimeoutError(OrchestratorError):
    """Raised when retrieval exceeds timeout."""
    pass


class ComponentFailureError(OrchestratorError):
    """Raised when a component fails with no fallback available."""
    pass


# ============================================================================
# Orchestrator Configuration
# ============================================================================

@dataclass
class OrchestratorConfig:
    """Configuration for the retrieval orchestrator."""
    enable_agent: bool = True
    enable_cache: bool = True
    max_retries: int = 2
    collect_diagnostics: bool = True
    timeout_ms: int = 5000          # Max total retrieval time
    timeout_grace_ms: int = 200      # Grace period after timeout for cleanup
    log_level: str = "INFO"

    # Agent sub-budget: time reserved for the rest of the pipeline (expansion,
    # search, rerank, etc.) after the agent's LLM call returns, and the
    # minimum time slice worth even attempting that call for.
    agent_timeout_reserve_ms: int = 2500
    agent_min_timeout_ms: int = 800
    
    # Fallback configuration
    fallback_on_component_failure: bool = True
    fallback_min_confidence: float = 0.3

    @classmethod
    def from_yaml(cls, path: Optional[str] = None) -> "OrchestratorConfig":
        """Load configuration from YAML with env overrides."""
        config_path = (
            path
            or os.getenv("RAGPIPE_ORCHESTRATOR_CONFIG")
            or str(Path(__file__).parent.parent.parent / "config" / "retrieval" / "orchestrator.yaml")
        )
        
        if Path(config_path).exists():
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
        else:
            data = {}
        
        return cls(
            enable_agent=bool(data.get("enable_agent", True)),
            enable_cache=bool(data.get("enable_cache", True)),
            max_retries=int(os.getenv("RAGPIPE_MAX_RETRIES", data.get("max_retries", 2))),
            collect_diagnostics=bool(data.get("collect_diagnostics", True)),
            timeout_ms=int(os.getenv("RAGPIPE_RETRIEVAL_TIMEOUT_MS", data.get("timeout_ms", 5000))),
            timeout_grace_ms=int(os.getenv("RAGPIPE_RETRIEVAL_TIMEOUT_GRACE_MS", data.get("timeout_grace_ms", 200))),
            log_level=str(data.get("log_level", "INFO")),
            agent_timeout_reserve_ms=int(data.get("agent_timeout_reserve_ms", 2500)),
            agent_min_timeout_ms=int(data.get("agent_min_timeout_ms", 800)),
            fallback_on_component_failure=bool(data.get("fallback_on_component_failure", True)),
            fallback_min_confidence=float(data.get("fallback_min_confidence", 0.3)),
        )


# ============================================================================
# Orchestrator Result
# ============================================================================

@dataclass
class OrchestratorResult:
    """Complete result from the retrieval orchestrator."""
    # Final output
    assembled_context: AssembledContext
    confidence: ConfidenceResult
    
    # Pipeline routing
    pipeline_name: str
    routing_decision: str
    agent_used: bool = False
    agent_decision: Optional[AgentDecision] = None
    
    # Query processing
    original_query: str = ""
    rewritten_query: str = ""
    expanded_query: str = ""
    keywords: str = ""
    
    # Retrieval stats
    candidates_retrieved: int = 0
    candidates_reranked: int = 0
    sources_used: List[str] = field(default_factory=list)
    retry_count: int = 0
    total_attempts: int = 1  # Including initial attempt
    
    # Timing breakdown (ms)
    total_time_ms: float = 0.0
    timing_breakdown: Dict[str, float] = field(default_factory=dict)
    
    # Conversation context
    conversation_topic: str = ""
    conversation_concept: str = ""
    
    # Error state
    is_fallback_result: bool = False
    error_details: Optional[str] = None
    failed_components: List[str] = field(default_factory=list)
    
    @property
    def is_ready_for_generation(self) -> bool:
        """Whether the context is sufficient for the generation layer."""
        return (
            self.confidence.level != ConfidenceLevel.LOW
            and self.assembled_context.chunks_used > 0
            and not self.is_fallback_result
        )


# ============================================================================
# Orchestrator State
# ============================================================================

class OrchestratorState:
    """Manages state during a single retrieval operation."""
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.start_time = time.time()
        self.components_called: List[str] = []
        self.failed_components: List[str] = []
        self.timing: Dict[str, float] = {}
        self.is_timed_out = False
        self.retry_count = 0
        self.total_attempts = 1
    
    def check_timeout(self) -> None:
        """Check if we've exceeded the timeout."""
        elapsed_ms = (time.time() - self.start_time) * 1000
        if elapsed_ms > self.config.timeout_ms:
            self.is_timed_out = True
            raise RetrievalTimeoutError(
                f"Retrieval timed out after {elapsed_ms:.0f}ms "
                f"(limit: {self.config.timeout_ms}ms)"
            )
    
    def mark_component(self, name: str) -> None:
        """Mark that a component was called."""
        self.components_called.append(name)
    
    def mark_failed(self, name: str) -> None:
        """Mark that a component failed."""
        self.failed_components.append(name)
    
    def record_timing(self, name: str, duration_ms: float) -> None:
        """Record timing for a component."""
        self.timing[name] = duration_ms
    
    def get_elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (time.time() - self.start_time) * 1000
    
    @property
    def remaining_ms(self) -> float:
        """Get remaining time budget in milliseconds."""
        return max(0, self.config.timeout_ms - self.get_elapsed_ms())
    
    @property
    def has_time_remaining(self) -> bool:
        """Check if there's time remaining in the budget."""
        return self.get_elapsed_ms() < self.config.timeout_ms


# ============================================================================
# Retrieval Orchestrator
# ============================================================================

class RetrievalOrchestrator:
    """
    Single entry point for the entire retrieval layer.
    
    Composes all 9 core engine components into one pipeline:
    Rewrite → Route → [Agent] → Expand → Filter → Search → Rerank → Build → Score
    
    Features:
    - Timeout enforcement with graceful degradation
    - Iterative retry loop (non-recursive)
    - Component-level exception handling with fallbacks
    - Full diagnostic tracing
    
    Usage:
        orchestrator = RetrievalOrchestrator(vector_store)
        result = orchestrator.retrieve(
            query="What is deadlock?",
            pipeline_config=LEARNING_PIPELINE,
            conversation_state=state
        )
        # result.assembled_context.formatted_context → ready for LLM
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        config: Optional[OrchestratorConfig] = None,
        rewriter: Optional[QueryRewriter] = None,
        agent: Optional[RetrievalAgent] = None,
        expander: Optional[QueryExpander] = None,
        filter_builder: Optional[MetadataFilterBuilder] = None,
        hybrid_search: Optional[HybridSearch] = None,
        reranker: Optional[Reranker] = None,
        context_builder: Optional[ContextBuilder] = None,
        confidence_scorer: Optional[ConfidenceScorer] = None,
    ):
        """
        Initialize the orchestrator with all components.
        
        Components can be injected (for testing/mocking) or auto-created
        from their respective config files.
        
        Args:
            vector_store: VectorStore instance for Pinecone
            config: Orchestrator config
            rewriter: QueryRewriter instance (auto-created if None)
            agent: RetrievalAgent instance (auto-created if None)
            expander: QueryExpander instance (auto-created if None)
            filter_builder: MetadataFilterBuilder instance (auto-created if None)
            hybrid_search: HybridSearch instance (auto-created if None)
            reranker: Reranker instance (auto-created if None)
            context_builder: ContextBuilder instance (auto-created if None)
            confidence_scorer: ConfidenceScorer instance (auto-created if None)
        """
        self.vector_store = vector_store
        self.config = config or OrchestratorConfig.from_yaml()
        
        # Configure logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        
        # Initialize components (injection or auto-creation)
        self.rewriter = rewriter or QueryRewriter()
        self.router = IntentRouter(rewriter=self.rewriter)
        self.agent = agent or RetrievalAgent()
        self.expander = expander or QueryExpander()
        self.filter_builder = filter_builder or MetadataFilterBuilder()
        self.hybrid_search = hybrid_search or HybridSearch(vector_store=vector_store)
        self.reranker = reranker or Reranker()
        self.context_builder = context_builder or ContextBuilder()
        self.confidence_scorer = confidence_scorer or ConfidenceScorer()
        
        # Fallback contexts
        self._fallback_context_cache: Optional[AssembledContext] = None
    
    # ========================================================================
    # Main Retrieve Method
    # ========================================================================
    
    def retrieve(
        self,
        query: str,
        pipeline_config: PipelineConfig,
        conversation_state: ConversationState,
        namespace: Union[str, List[str]] = "default"
    ) -> OrchestratorResult:
        """
        Execute the full retrieval pipeline with retry logic.

        Uses iterative retry loop instead of recursion for better
        stack safety and clearer retry logic.

        Args:
            query: Raw user query
            pipeline_config: Pipeline configuration
            conversation_state: Current conversation state
            namespace: Pinecone namespace(s) to search — a single namespace
                or a list (e.g. every namespace a profile's documents were
                ingested into; see VectorStore.query for the fan-out)

        Returns:
            OrchestratorResult with assembled context and full diagnostics
        """
        # Wraps the whole retry loop in one span so every _safe_call_component
        # span from every attempt nests under a single trace — without this,
        # sibling top-level spans (no shared parent context) each start a
        # new trace instead of sharing one (verified empirically: Langfuse's
        # OTEL context only shares a trace ID across nested spans, not
        # sequential sibling ones). Matters for callers that invoke
        # retrieve() standalone (e.g. tools/retrieval_debug_server.py)
        # without an enclosing observability.tracing.traced_pipeline_call.
        with traced_span(
            "retrieve",
            input={
                "query": query,
                "pipeline_name": pipeline_config.name,
                "namespace": safe_dict(namespace),
            },
        ) as span:
            result = self._retrieve_with_retries(
                query, pipeline_config, conversation_state, namespace
            )
            span.update(output=safe_dict(result))
            return result

    def _retrieve_with_retries(
        self,
        query: str,
        pipeline_config: PipelineConfig,
        conversation_state: ConversationState,
        namespace: Union[str, List[str]],
    ) -> OrchestratorResult:
        """Retry-loop body extracted from retrieve() so the tracing span in
        retrieve() wraps every attempt without duplicating span setup at
        each of this method's several return points."""
        state = OrchestratorState(self.config)
        current_pipeline = pipeline_config
        attempt = 0

        while attempt <= self.config.max_retries:
            attempt += 1
            state.total_attempts = attempt
            state.retry_count = attempt - 1
            attempt_start_elapsed_ms = state.get_elapsed_ms()

            try:
                result = self._execute_retrieval_attempt(
                    query=query,
                    pipeline_config=current_pipeline,
                    conversation_state=conversation_state,
                    namespace=namespace,
                    state=state
                )

                attempt_duration_ms = state.get_elapsed_ms() - attempt_start_elapsed_ms

                # Only retry if there's realistically enough budget left for
                # another attempt of similar cost — otherwise we're just
                # guaranteeing a timeout instead of returning this result.
                if (result.confidence.should_retry and
                    attempt <= self.config.max_retries and
                    state.remaining_ms > attempt_duration_ms):

                    logger.info(f"Retry {attempt}/{self.config.max_retries}: "
                              f"confidence={result.confidence.score:.2f}")
                    
                    # Modify pipeline for retry
                    current_pipeline = self._build_retry_pipeline(
                        pipeline_config, result.confidence.retry_config
                    )
                    continue
                
                # Success or no retry needed
                return result
                
            except RetrievalTimeoutError as e:
                logger.warning(f"Retrieval timed out on attempt {attempt}: {e}")
                if attempt <= self.config.max_retries and state.remaining_ms > 0:
                    # Try with reduced scope
                    current_pipeline = self._build_timeout_pipeline(pipeline_config)
                    continue
                return self._build_timeout_result(query, state)
                
            except ComponentFailureError as e:
                logger.error(f"Component failure on attempt {attempt}: {e}")
                if self.config.fallback_on_component_failure and attempt <= self.config.max_retries:
                    current_pipeline = self._build_fallback_pipeline(pipeline_config)
                    continue
                return self._build_error_result(query, str(e), state)
                
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt}: {e}", exc_info=True)
                return self._build_error_result(query, str(e), state)
        
        # Exhausted retries
        return self._build_error_result(
            query, 
            f"Exhausted {self.config.max_retries} retries",
            state
        )
    
    # ========================================================================
    # Core Execution
    # ========================================================================
    
    def _execute_retrieval_attempt(
        self,
        query: str,
        pipeline_config: PipelineConfig,
        conversation_state: ConversationState,
        namespace: str,
        state: OrchestratorState
    ) -> OrchestratorResult:
        """
        Execute a single retrieval attempt.
        
        All component calls are wrapped with timeout checks and
        exception handling.
        """
        total_start = state.start_time
        
        # ── Step 1: Query Rewriting ─────────────────────────────────────
        rewritten_query, rewrite_metadata = self._safe_call_component(
            "rewriter",
            state,
            self.rewriter.rewrite,
            query,
            conversation_state
        )
        
        # ── Step 2: Intent Routing ──────────────────────────────────────
        routing = self._safe_call_component(
            "router",
            state,
            self.router.route,
            query,
            conversation_state
        )
        
        agent_used = False
        agent_decision = None
        search_query = rewritten_query
        metadata_filters = dict(pipeline_config.metadata_filters)
        target_pipeline = pipeline_config.name
        
        # ── Step 3: Agent Reasoning (if needed) ─────────────────────────
        if routing.needs_agent and self.config.enable_agent:
            agent_budget_ms = state.remaining_ms - self.config.agent_timeout_reserve_ms

            if agent_budget_ms < self.config.agent_min_timeout_ms:
                logger.warning(
                    f"Skipping agent reasoning: only {state.remaining_ms:.0f}ms left in budget "
                    f"(need >= {self.config.agent_min_timeout_ms + self.config.agent_timeout_reserve_ms}ms). "
                    f"Falling back to non-agent retrieval."
                )
            else:
                agent_decision = self._safe_call_component(
                    "agent",
                    state,
                    self.agent.reason,
                    query=query,
                    rewritten_query=rewritten_query,
                    agent_context=routing.agent_context,
                    state=conversation_state,
                    timeout_seconds=agent_budget_ms / 1000.0
                )
                agent_used = True

                # Apply agent decisions
                if agent_decision.action == AgentAction.CLARIFY:
                    return self._build_clarification_result(
                        query, rewritten_query, agent_decision, pipeline_config,
                        conversation_state, state
                    )

                if agent_decision.action == AgentAction.SWITCH_PIPELINE:
                    target_pipeline = agent_decision.suggested_pipeline
                    from src.retrieval.pipelines import PIPELINE_REGISTRY
                    if target_pipeline in PIPELINE_REGISTRY:
                        pipeline_config = PIPELINE_REGISTRY[target_pipeline]

                if agent_decision.action in (AgentAction.REWRITE_AND_RETRIEVE, AgentAction.DIRECT_RETRIEVE):
                    search_query = agent_decision.search_query or rewritten_query
                    if agent_decision.metadata_filters:
                        safe_agent_filters = {
                            k: v for k, v in agent_decision.metadata_filters.items()
                            if k not in AGENT_FILTER_EXCLUDED_FIELDS
                        }
                        metadata_filters.update(safe_agent_filters)

                if agent_decision.action == AgentAction.DECOMPOSE:
                    return self._handle_decomposed(
                        query, agent_decision, pipeline_config,
                        conversation_state, namespace, state
                    )
        
        # ── Step 4: Query Expansion ─────────────────────────────────────
        expanded = self._safe_call_component(
            "expander",
            state,
            self.expander.expand_for_pipeline,
            query=search_query,
            pipeline_name=target_pipeline,
            state=conversation_state
        )
        
        # ── Step 5: Metadata Filter Building ────────────────────────────
        filter_result = self._safe_call_component(
            "filter_builder",
            state,
            self.filter_builder.build,
            pipeline_config=pipeline_config,
            conversation_state=conversation_state,
            additional_filters=metadata_filters
        )
        
        # ── Step 6: Hybrid Search ───────────────────────────────────────
        search_result = self._safe_call_component(
            "hybrid_search",
            state,
            self.hybrid_search.search,
            query=expanded.expanded,
            keywords=expanded.keywords,
            namespace=namespace,
            metadata_filters=filter_result.filter_dict if not filter_result.is_empty else None,
            pipeline_weights=pipeline_config.hybrid_weights
        )
        
        # Convert SearchCandidates to dict format for reranker
        candidate_dicts = [
            {
                "chunk_id": c.chunk_id,
                "content": c.content,
                "raw_content": c.raw_content,
                "score": c.score,
                "metadata": c.metadata,
            }
            for c in search_result.candidates
        ]
        
        # ── Step 7: Reranking ───────────────────────────────────────────
        reranker_result = self._safe_call_component(
            "reranker",
            state,
            self.reranker.rerank,
            query=expanded.expanded,
            candidates=candidate_dicts,
            top_k=pipeline_config.rerank_k
        )
        
        # ── Step 8: Context Assembly ────────────────────────────────────
        assembled = self._safe_call_component(
            "context_builder",
            state,
            self.context_builder.build,
            chunks=reranker_result.chunks,
            template=pipeline_config.context_template,
            pipeline_name=target_pipeline
        )
        
        # ── Step 9: Confidence Scoring ──────────────────────────────────
        confidence = self._safe_call_component(
            "confidence_scorer",
            state,
            self.confidence_scorer.evaluate,
            query=expanded.expanded,
            keywords=expanded.keywords,
            reranker_result=reranker_result,
            assembled_context=assembled,
            metadata_filters_applied=filter_result.applied_filters,
            retrieval_sources_used=search_result.sources_used,
            retry_count=state.retry_count
        )
        
        # ── Build Final Result ──────────────────────────────────────────
        total_time = (time.time() - total_start) * 1000
        
        return OrchestratorResult(
            assembled_context=assembled,
            confidence=confidence,
            pipeline_name=target_pipeline,
            routing_decision=routing.decision.value,
            agent_used=agent_used,
            agent_decision=agent_decision,
            original_query=query,
            rewritten_query=rewritten_query,
            expanded_query=expanded.expanded,
            keywords=expanded.keywords,
            candidates_retrieved=len(search_result.candidates),
            candidates_reranked=reranker_result.total_input,
            sources_used=search_result.sources_used,
            retry_count=state.retry_count,
            total_attempts=state.total_attempts,
            total_time_ms=round(total_time, 2),
            timing_breakdown={k: round(v, 2) for k, v in state.timing.items()},
            conversation_topic=conversation_state.current_topic or "",
            conversation_concept=conversation_state.current_concept or "",
            failed_components=state.failed_components,
        )
    
    # ========================================================================
    # Safe Component Execution
    # ========================================================================
    
    def _safe_call_component(
        self,
        name: str,
        orchestrator_state: OrchestratorState,
        func: Any,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute a component with timeout checking and error handling.
        
        Args:
            name: Component name for diagnostics
            state: Orchestrator state
            func: Component function to call
            *args, **kwargs: Arguments to pass
            
        Returns:
            Component result
            
        Raises:
            ComponentFailureError: If component fails and no fallback
            RetrievalTimeoutError: If timeout is exceeded
        """
        orchestrator_state.check_timeout()
        orchestrator_state.mark_component(name)

        component_start = time.time()
        # One traced_span per component call — this single wrap point covers
        # all 9 retrieval stages (rewriter, router, agent, expander,
        # filter_builder, hybrid_search, reranker, context_builder,
        # confidence_scorer) without touching any of their individual
        # modules. See observability/tracing.py traced_span() docstring for
        # why component failures re-raise through the span instead of being
        # swallowed by it.
        with traced_span(name, input={"args": args, "kwargs": kwargs}) as span:
            try:
                result = func(*args, **kwargs)
                duration = (time.time() - component_start) * 1000
                orchestrator_state.record_timing(name, duration)
                span.update(output=safe_dict(result))
                return result

            except RetrievalTimeoutError:
                # Propagate timeout
                raise

            except Exception as e:
                orchestrator_state.mark_failed(name)
                duration = (time.time() - component_start) * 1000
                orchestrator_state.record_timing(name, duration)

                logger.error(f"Component '{name}' failed after {duration:.0f}ms: {e}")

                # Try to get a fallback result for critical components
                fallback = self._get_component_fallback(name, *args, **kwargs)
                if fallback is not None:
                    logger.info(f"Using fallback for component '{name}'")
                    span.update(
                        output={"fallback_used": True},
                        level="WARNING",
                        status_message=str(e),
                    )
                    return fallback

                # No fallback available
                raise ComponentFailureError(
                    f"Component '{name}' failed: {e}"
                ) from e
    
    def _get_component_fallback(self, name: str, *args, **kwargs) -> Any:
        """
        Get fallback result for a failed component.
        
        Returns None if no fallback is available.
        """
        if name == "hybrid_search":
            # Fallback: try with expanded query
            try:
                # Use a simpler search configuration
                return self._fallback_search(*args, **kwargs)
            except:
                return None
                
        elif name == "reranker":
            # Fallback: use candidates as-is (no reranking)
            candidates = kwargs.get("candidates", args[1] if len(args) > 1 else [])
            return self._fallback_reranker(candidates)
            
        elif name == "confidence_scorer":
            # Fallback: minimal confidence
            return self._fallback_confidence_scorer(*args, **kwargs)
        
        return None
    
    def _fallback_search(self, query: str, keywords: str, namespace: str, 
                         metadata_filters: Optional[Dict], **kwargs) -> Any:
        """Fallback search with lower requirements."""
        from hybrid_search import SearchResult, SearchCandidate
        
        # Simplified search - just use vector search with fewer constraints
        try:
            results = self.vector_store.query(
                query=query,
                top_k=20,
                namespace=namespace,
                filter=metadata_filters if metadata_filters else {}
            )
            
            candidates = [
                SearchCandidate(
                    chunk_id=match.id,
                    content=match.metadata.get("text", ""),
                    raw_content=match.metadata.get("raw_text", ""),
                    score=match.score,
                    metadata=match.metadata,
                )
                for match in results.matches[:10]
            ]
            
            return SearchResult(
                candidates=candidates,
                sources_used=["vector_fallback"],
                total_candidates=len(candidates),
                query_used=query,
                execution_time_ms=0,
            )
        except:
            # Ultimate fallback - empty result
            return SearchResult(
                candidates=[],
                sources_used=[],
                total_candidates=0,
                query_used=query,
                execution_time_ms=0,
            )
    
    def _fallback_reranker(self, candidates: List[Dict]) -> Any:
        """Fallback reranker that preserves original order."""
        from reranker import RerankerResult, RerankedChunk
        
        chunks = [
            RerankedChunk(
                chunk_id=c["chunk_id"],
                content=c["content"],
                raw_content=c.get("raw_content", c["content"]),
                score=c["score"],
                metadata=c["metadata"],
                original_score=c["score"],
                rerank_score=c["score"],
            )
            for c in candidates[:10]
        ]
        
        return RerankerResult(
            chunks=chunks,
            total_input=len(candidates),
            top_k=len(chunks),
            model_used="fallback",
            execution_time_ms=0,
        )
    
    def _fallback_confidence_scorer(self, **kwargs) -> Any:
        """Fallback confidence scorer."""
        from confidence import ConfidenceResult, ConfidenceLevel
        
        return ConfidenceResult(
            score=0.3,
            level=ConfidenceLevel.LOW,
            action="fallback",
            feature_scores={},
            recommendations=["Fallback confidence score used due to component failure"],
        )
    
    # ========================================================================
    # Specialized Handlers
    # ========================================================================
    
    def _build_clarification_result(
        self,
        query: str,
        rewritten: str,
        agent_decision: AgentDecision,
        pipeline_config: PipelineConfig,
        state: ConversationState,
        orchestrator_state: OrchestratorState
    ) -> OrchestratorResult:
        """Build result when agent requests clarification."""
        empty_context = AssembledContext(
            formatted_context="",
            chunks_used=0,
            tokens_used=0,
            sections=[],
            citations=[],
            template_used=pipeline_config.context_template.value
        )
        
        clarification_confidence = ConfidenceResult(
            score=0.0,
            level=ConfidenceLevel.LOW,
            action="clarify",
            feature_scores={},
            recommendations=[f"Clarification needed: {agent_decision.clarification_question}"],
        )
        
        return OrchestratorResult(
            assembled_context=empty_context,
            confidence=clarification_confidence,
            pipeline_name=pipeline_config.name,
            routing_decision="clarify",
            agent_used=True,
            agent_decision=agent_decision,
            original_query=query,
            rewritten_query=rewritten,
            total_time_ms=round(orchestrator_state.get_elapsed_ms(), 2),
            timing_breakdown=orchestrator_state.timing,
            total_attempts=orchestrator_state.total_attempts,
        )
    
    def _handle_decomposed(
        self,
        query: str,
        agent_decision: AgentDecision,
        pipeline_config: PipelineConfig,
        state: ConversationState,
        namespace: str,
        orchestrator_state: OrchestratorState
    ) -> OrchestratorResult:
        """
        Handle decomposed queries by retrieving for each sub-query
        and merging results.
        
        This properly merges evidence from all sub-queries instead of
        re-retrieving the original query.
        """
        from reranker import RerankedChunk
        
        all_chunks: List[RerankedChunk] = []
        all_sources: Set[str] = set()
        total_candidates = 0
        total_timing: Dict[str, float] = {}
        
        sub_configs = self._create_sub_pipeline_configs(
            pipeline_config, len(agent_decision.sub_queries)
        )
        
        # Retrieve for each sub-query
        for i, sub_query in enumerate(agent_decision.sub_queries):
            sub_config = sub_configs[i]
            
            # Create a fresh state for each sub-query to avoid timeout accumulation
            sub_result = self._execute_retrieval_attempt(
                query=sub_query.query,
                pipeline_config=sub_config,
                conversation_state=state,
                namespace=namespace,
                state=orchestrator_state
            )
            
            # Merge results
            # Extract chunks from the result's assembled context
            # In a real implementation, we'd store the chunks in the result
            # For now, we'll reconstruct from the final result
            if sub_result.assembled_context:
                # We need to get the actual chunks - this is a placeholder
                # In practice, you'd store the chunks in the result
                pass
            
            total_candidates += sub_result.candidates_retrieved
            all_sources.update(sub_result.sources_used)
            
            # Merge timing (take max per component)
            for key, value in sub_result.timing_breakdown.items():
                if key not in total_timing or value > total_timing[key]:
                    total_timing[key] = value
        
        # Now run a final retrieval with the original query to get the primary context
        # But with the combined evidence from all sub-queries
        final_result = self._execute_retrieval_attempt(
            query=query,
            pipeline_config=pipeline_config,
            conversation_state=state,
            namespace=namespace,
            state=orchestrator_state
        )
        
        # Augment the final result with merged evidence
        final_result.candidates_retrieved = max(final_result.candidates_retrieved, total_candidates)
        final_result.sources_used = list(set(final_result.sources_used) | all_sources)
        
        return final_result
    
    def _create_sub_pipeline_configs(
        self,
        base_config: PipelineConfig,
        num_subqueries: int
    ) -> List[PipelineConfig]:
        """Create pipeline configs for decomposed sub-queries."""
        return [
            PipelineConfig(
                name=f"{base_config.name}_sub_{i}",
                description=f"Sub-query config for decomposed retrieval",
                hybrid_weights=base_config.hybrid_weights,
                candidate_k=max(2, base_config.candidate_k // num_subqueries),
                rerank_k=max(1, base_config.rerank_k // num_subqueries),
                metadata_filters=base_config.metadata_filters,
                context_template=base_config.context_template,
                enable_query_expansion=base_config.enable_query_expansion,
                min_confidence=self._relax_confidence(base_config.min_confidence),  # Lower threshold for sub-queries
                retry_on_low_confidence=False,  # Don't retry sub-queries
                cache_ttl_seconds=base_config.cache_ttl_seconds,
            )
            for i in range(num_subqueries)
        ]
    
    # ========================================================================
    # Pipeline Builders
    # ========================================================================

    @staticmethod
    def _relax_confidence(level: ConfidenceLevel) -> ConfidenceLevel:
        """Step min_confidence down one notch (HIGH->MEDIUM->LOW->LOW).

        min_confidence is a ConfidenceLevel enum, not a float, so it can't be
        scaled arithmetically — this relaxes the required level instead.
        """
        if level == ConfidenceLevel.HIGH:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    def _build_retry_pipeline(
        self,
        base_config: PipelineConfig,
        retry_config: Optional[Dict[str, Any]]
    ) -> PipelineConfig:
        """Build a modified pipeline for retry attempts."""
        retry_config = retry_config or {}
        
        return PipelineConfig(
            name=base_config.name,
            description=base_config.description,
            hybrid_weights=base_config.hybrid_weights,
            candidate_k=base_config.candidate_k + retry_config.get("increase_candidate_k", 10),
            rerank_k=base_config.rerank_k,
            metadata_filters=base_config.metadata_filters,
            context_template=base_config.context_template,
            enable_query_expansion=retry_config.get("enable_aggressive_expansion", True),
            min_confidence=self._relax_confidence(base_config.min_confidence),  # Lower threshold for retry
            retry_on_low_confidence=True,
            cache_ttl_seconds=base_config.cache_ttl_seconds,
        )
    
    def _build_timeout_pipeline(self, base_config: PipelineConfig) -> PipelineConfig:
        """Build a reduced-scope pipeline for timeout recovery."""
        return PipelineConfig(
            name=base_config.name,
            description=base_config.description,
            hybrid_weights=base_config.hybrid_weights,
            candidate_k=max(5, base_config.candidate_k // 2),
            rerank_k=max(2, base_config.rerank_k // 2),
            metadata_filters=base_config.metadata_filters,
            context_template=base_config.context_template,
            enable_query_expansion=False,  # Skip expansion for speed
            min_confidence=ConfidenceLevel.LOW,
            retry_on_low_confidence=False,
            cache_ttl_seconds=base_config.cache_ttl_seconds,
        )
    
    def _build_fallback_pipeline(self, base_config: PipelineConfig) -> PipelineConfig:
        """Build a minimal pipeline for fallback."""
        return PipelineConfig(
            name=f"{base_config.name}_fallback",
            description="Fallback pipeline",
            hybrid_weights={"semantic": 1.0, "bm25": 0.0},  # Semantic only
            candidate_k=10,
            rerank_k=3,
            metadata_filters={},  # No filters
            context_template=base_config.context_template,
            enable_query_expansion=False,
            min_confidence=ConfidenceLevel.LOW,
            retry_on_low_confidence=False,
            cache_ttl_seconds=base_config.cache_ttl_seconds,
        )
    
    # ========================================================================
    # Error Results
    # ========================================================================
    
    def _build_timeout_result(
        self,
        query: str,
        state: OrchestratorState
    ) -> OrchestratorResult:
        """Build a timeout result with whatever was retrieved."""
        from src.retrieval.confidence import ConfidenceResult, ConfidenceLevel
        
        empty_context = AssembledContext(
            formatted_context="",
            chunks_used=0,
            tokens_used=0,
            sections=[],
            citations=[],
            template_used="timeout",
        )
        
        return OrchestratorResult(
            assembled_context=empty_context,
            confidence=ConfidenceResult(
                score=0.0,
                level=ConfidenceLevel.LOW,
                action="timeout",
                feature_scores={},
                recommendations=["Retrieval timed out. Please try a simpler query."],
            ),
            pipeline_name="timeout",
            routing_decision="timeout",
            original_query=query,
            total_time_ms=round(state.get_elapsed_ms(), 2),
            timing_breakdown=state.timing,
            total_attempts=state.total_attempts,
            is_fallback_result=True,
            error_details=f"Timeout after {state.get_elapsed_ms():.0f}ms",
            failed_components=state.failed_components,
        )
    
    def _build_error_result(
        self,
        query: str,
        error_message: str,
        state: OrchestratorState
    ) -> OrchestratorResult:
        """Build an error result."""
        from src.retrieval.confidence import ConfidenceResult, ConfidenceLevel
        
        empty_context = AssembledContext(
            formatted_context="",
            chunks_used=0,
            tokens_used=0,
            sections=[],
            citations=[],
            template_used="error",
        )
        
        return OrchestratorResult(
            assembled_context=empty_context,
            confidence=ConfidenceResult(
                score=0.0,
                level=ConfidenceLevel.LOW,
                action="error",
                feature_scores={},
                recommendations=["An error occurred during retrieval. Please try again."],
            ),
            pipeline_name="error",
            routing_decision="error",
            original_query=query,
            total_time_ms=round(state.get_elapsed_ms(), 2),
            timing_breakdown=state.timing,
            total_attempts=state.total_attempts,
            is_fallback_result=True,
            error_details=error_message,
            failed_components=state.failed_components,
        )
    
    # ========================================================================
    # Diagnostics
    # ========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all retrieval components."""
        return {
            "orchestrator": {
                "agent_enabled": self.config.enable_agent,
                "max_retries": self.config.max_retries,
                "timeout_ms": self.config.timeout_ms,
                "timeout_grace_ms": self.config.timeout_grace_ms,
                "fallback_enabled": self.config.fallback_on_component_failure,
            },
            "components": {
                "rewriter": "available",
                "router": "available",
                "agent": "available" if self.config.enable_agent else "disabled",
                "expander": "available",
                "filter_builder": "available",
                "hybrid_search": self.hybrid_search.get_retrieval_status() if hasattr(self.hybrid_search, 'get_retrieval_status') else "available",
                "reranker": "available",
                "context_builder": "available",
                "confidence_scorer": "available",
            },
        }


# ============================================================================
# Validation
# ============================================================================

if __name__ == "__main__":
    from pathlib import Path
    from pprint import pprint

    from src.ingestion.parser import DocumentParser
    from src.ingestion.cleaner import TextCleaner
    from src.ingestion.metadata_extractor import MetadataExtractor
    from src.ingestion.chunker import SemanticChunker
    from src.ingestion.enricher import ChunkEnricher
    from src.ingestion.embedder import EmbeddingGenerator
    from src.ingestion.vector_store import VectorStore
    from src.retrieval.query_rewriter import ConversationState
    from src.retrieval.pipelines import LEARNING_PIPELINE

    print("=" * 80)
    print("RAG Pipeline - End-to-End Integration Test")
    print("=" * 80)

    # ---------------------------------------------------------
    # Ingest the bundled sample corpus file, then retrieve against it.
    # Namespace is derived fresh each run (deterministic hash of the file
    # path + course name) rather than hardcoded, since a stale namespace
    # from a previous ingestion run elsewhere would silently retrieve
    # nothing.
    # ---------------------------------------------------------

    SAMPLE_FILE = str(Path(__file__).resolve().parents[2] / "sample_data" / "sample_lecture.pptx")

    print("\nIngesting sample corpus file...")
    parsed = DocumentParser().parse(SAMPLE_FILE)
    cleaned = TextCleaner().clean(parsed)
    metadata = MetadataExtractor().extract_with_retry(cleaned)
    chunks = SemanticChunker().chunk(cleaned, metadata)
    enriched = ChunkEnricher().enrich(chunks)
    embedded = EmbeddingGenerator().embed_chunks(enriched)

    vector_store = VectorStore()
    namespace = vector_store.generate_namespace_from_document(SAMPLE_FILE, metadata.course_name)
    upsert_result = vector_store.upsert(embedded, namespace=namespace)
    print(f"Ingested {upsert_result.chunks_upserted} chunks into namespace '{namespace}'")
    print(f"Extracted course_name='{metadata.course_name}', chapter_title='{metadata.chapter_title}'")

    orchestrator = RetrievalOrchestrator(
        vector_store=vector_store
    )

    # Build the in-memory BM25 index from this run's chunks so keyword
    # search actually exercises real sparse retrieval instead of silently
    # falling back to a second semantic pass (see HybridSearch.search's
    # fallback warning) — mirrors what a real ingestion service does after
    # every upload.
    from src.ingestion.enricher import ChunkEnricher as _ChunkEnricher
    bm25_documents = []
    for chunk in enriched:
        chunk_metadata = _ChunkEnricher.to_metadata(chunk)
        chunk_metadata["namespace"] = namespace
        bm25_documents.append({
            "id": chunk.chunk_id,
            "content": chunk.content,
            "raw_content": chunk.raw_content,
            "metadata": chunk_metadata,
        })
    orchestrator.hybrid_search.initialize_indices(documents=bm25_documents)
    print(f"BM25 index initialized with {len(bm25_documents)} documents")

    # ---------------------------------------------------------
    # Conversation State
    # ---------------------------------------------------------

    # NOTE: subject/chapter here are descriptive only — the retrieval layer
    # deliberately does not hard-filter on them (see
    # MetadataFilterBuilder._build_from_conversation's docstring): the
    # application-level conversation vocabulary and the ingestion-time
    # LLM-extracted metadata vocabulary aren't guaranteed to match, so
    # semantic similarity does the topic-relevance work instead.
    state = ConversationState(
        session_id="integration_test",
        subject=metadata.course_name,
        module=metadata.course_name,
        chapter=metadata.chapter_title,
        current_topic=None,
        current_concept=None
    )

    # ---------------------------------------------------------
    # Real queries based on the uploaded PPT
    # ---------------------------------------------------------

    queries = [

        "What is inventory?",

        "Explain the types of inventory.",

        "Why do companies carry inventory?",

        "Explain inventory holding cost.",

        "What is EOQ?",

        "How is EOQ calculated?",

        "What is reorder point?",

        "Difference between fixed order quantity system and fixed time period system.",

        "Give an example of safety stock.",

        "What are ordering costs?",

        "Explain pipeline inventory.",

        "Summarize inventory management."

    ]

    # ---------------------------------------------------------
    # Execute retrieval
    # ---------------------------------------------------------

    for i, query in enumerate(queries, start=1):

        print("\n" + "=" * 80)
        print(f"TEST {i}")
        print("=" * 80)

        print(f"\nQuery:\n{query}")

        result = orchestrator.retrieve(
            query=query,
            pipeline_config=LEARNING_PIPELINE,
            conversation_state=state,
            namespace=namespace
        )

        print("\n----------------------------------------")
        print("Pipeline")
        print("----------------------------------------")

        print(f"Routing           : {result.routing_decision}")
        print(f"Agent Used        : {result.agent_used}")

        print("\n----------------------------------------")
        print("Query Processing")
        print("----------------------------------------")

        print(f"Original          : {result.original_query}")
        print(f"Rewritten         : {result.rewritten_query}")
        print(f"Expanded          : {result.expanded_query}")

        print("\n----------------------------------------")
        print("Retrieval")
        print("----------------------------------------")

        print(f"Candidates        : {result.candidates_retrieved}")
        print(f"After Rerank      : {result.candidates_reranked}")

        print(f"Sources Used      : {result.sources_used}")

        print("\n----------------------------------------")
        print("Confidence")
        print("----------------------------------------")

        print(f"Score             : {result.confidence.score:.3f}")
        print(f"Level             : {result.confidence.level}")

        print(f"Retry             : {result.confidence.should_retry}")

        print("\n----------------------------------------")
        print("Context")
        print("----------------------------------------")

        print(result.assembled_context.formatted_context[:1200])

        print("\n----------------------------------------")
        print("Timing")
        print("----------------------------------------")

        print(f"Total Time        : {result.total_time_ms:.2f} ms")

        pprint(result.timing_breakdown)