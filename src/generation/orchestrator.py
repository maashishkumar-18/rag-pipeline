"""
Generation Orchestrator
Single entry point for the generation layer.

Composes:
    PromptBuilder → LLMClient → PostProcessor → GenerationResponse

Owns:
- Request lifecycle (ID generation, timing, error handling)
- Configuration loading
- Mode-specific prompt assembly strategy selection
- Citation formatting

Architecture:
    GenerationRequest → GenerationOrchestrator.generate() → GenerationResponse

Usage:
    orchestrator = GenerationOrchestrator()
    response = orchestrator.generate(request)
"""

import os
import time
import uuid
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from dotenv import load_dotenv

from src.generation.config import (
    GenerationConfig,
    GenerationRequest,
    GenerationResponse,
    GenerationMode,
    ModeConfig,
    ModelConfig,
    AnswerFormat,
    CitationStyle,
)

from src.generation.prompt_builder import (
    PromptBuilder,
    ContextAssembler,
    StandardContextAssembly,
    MinimalContextAssembly,
    ChronologicalContextAssembly,
    ExamAnswerContextAssembly,
)

from src.common.llm_client import LLMClient

from src.generation.post_processor import PostProcessor, PostProcessingConfig

from observability.tracing import traced_span, safe_dict
from src.common.pricing import estimate_deepseek_cost

load_dotenv()

logger = logging.getLogger(__name__)


# ============================================================================
# Strategy Selector
# ============================================================================

class AssemblyStrategySelector:
    """
    Selects the appropriate ContextAssemblyStrategy based on GenerationMode.
    
    Maps generation modes to assembly strategies:
    - CONTEXT_AWARE → StandardContextAssembly (section-grouped)
    - SIMPLE_EXPLANATION → StandardContextAssembly (section-grouped)
    
    Future modes (revision, qa, quiz) can be mapped to other strategies.
    """
    
    def __init__(self):
        self._strategies = {
            GenerationMode.CONTEXT_AWARE: StandardContextAssembly(),
            GenerationMode.SIMPLE_EXPLANATION: StandardContextAssembly(),
            GenerationMode.TOPIC_EXTRACTION: MinimalContextAssembly(),
            GenerationMode.QUIZ_GENERATION: MinimalContextAssembly(),
        }
    
    def select(self, mode: GenerationMode):
        """
        Select the assembly strategy for a generation mode.
        
        Args:
            mode: Generation mode
            
        Returns:
            ContextAssemblyStrategy instance
        """
        strategy = self._strategies.get(mode)
        if strategy is None:
            logger.warning(
                f"No assembly strategy configured for mode '{mode.value}'. "
                f"Using StandardContextAssembly."
            )
            return StandardContextAssembly()
        return strategy
    
    def register(self, mode: GenerationMode, strategy):
        """Register a custom strategy for a mode."""
        self._strategies[mode] = strategy


# ============================================================================
# Generation Orchestrator
# ============================================================================

class GenerationOrchestrator:
    """
    Single entry point for the generation layer.
    
    Composes PromptBuilder, LLMClient, and PostProcessor into
    a single generate() call.
    
    Owns:
    - Request lifecycle
    - ID generation
    - Timing
    - Error handling
    - Configuration
    
    Usage:
        orchestrator = GenerationOrchestrator()
        response = orchestrator.generate(request)
    """
    
    def __init__(
        self,
        config: Optional[GenerationConfig] = None,
        config_dir: Optional[str] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        llm_client: Optional[LLMClient] = None,
        post_processor: Optional[PostProcessor] = None,
    ):
        """
        Initialize the generation orchestrator.
        
        Args:
            config: GenerationConfig (loads from YAML if None)
            config_dir: Path to generation config directory
            prompt_builder: PromptBuilder instance (auto-created if None)
            llm_client: LLMClient instance (auto-created if None)
            post_processor: PostProcessor instance (auto-created if None)
        """
        # Load configuration
        self.config = config or GenerationConfig.from_yaml(config_dir)
        
        # Initialize strategy selector
        self.strategy_selector = AssemblyStrategySelector()
        
        # Create context assembler with strategy selection
        context_assembler = ContextAssembler(
            strategy=StandardContextAssembly()
        )
        
        # Initialize components (injection or auto-creation)
        self.prompt_builder = prompt_builder or PromptBuilder(
            assembler=context_assembler
        )
        self.llm_client = llm_client or LLMClient()
        self.post_processor = post_processor or PostProcessor(
            config=self.config.post_processing
        )
    
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate an answer from retrieved context.

        This is the single public method for the generation layer.

        Args:
            request: GenerationRequest with query, chunks, and mode

        Returns:
            GenerationResponse with answer, citations, and metadata
        """
        # Wraps the whole method in one span so prompt_builder/llm_generate/
        # post_processor (below) nest under a single trace instead of each
        # starting its own — sibling spans with no shared parent context
        # don't share a trace id (see src/retrieval/orchestrator.py
        # retrieve()'s equivalent wrap for the empirical check). Matters for
        # callers that invoke generate() standalone, without an enclosing
        # observability.tracing.traced_pipeline_call.
        with traced_span(
            "generate",
            input=safe_dict({
                "request_id": request.request_id,
                "query": request.query,
                "mode": request.mode,
                "chunk_count": len(request.chunks),
            }),
        ) as span:
            response = self._generate_impl(request)
            if response.error_type:
                span.update(output=safe_dict(response), level="ERROR", status_message=response.error_type)
            elif response.warnings:
                span.update(output=safe_dict(response), level="WARNING", status_message="; ".join(response.warnings))
            else:
                span.update(output=safe_dict(response))
            return response

    def _generate_impl(self, request: GenerationRequest) -> GenerationResponse:
        """generate()'s body, extracted so the tracing span in generate()
        wraps every phase without duplicating span setup at each of this
        method's several early-return error paths."""
        total_start = time.time()
        generation_id = str(uuid.uuid4())[:8]
        warnings: List[str] = []

        # ── Step 1: Get mode and model configuration ─────────────────
        try:
            mode_config = self.config.get_mode_config(request.mode)
            model_config = self.config.get_model_config(mode_config.model_key)
        except Exception as e:
            logger.error(f"Configuration resolution failed: {e}")
            return self._build_error_response(
                request=request,
                generation_id=generation_id,
                model_config=None,
                warnings=[f"Configuration resolution failed: {str(e)}"],
                total_start=total_start,
            )

        # ── Step 2: Select and apply assembly strategy ───────────────
        strategy = self.strategy_selector.select(request.mode)
        self.prompt_builder.set_assembly_strategy(strategy)

        # ── Step 3: Build prompt ────────────────────────────────────
        prompt_start = time.time()
        with traced_span(
            "prompt_builder",
            input=safe_dict({"query": request.query, "mode": request.mode}),
        ) as prompt_span:
            try:
                prompt = self.prompt_builder.build(request, mode_config)
            except Exception as e:
                logger.error(f"Prompt building failed: {e}")
                prompt_span.update(level="ERROR", status_message=str(e))
                return self._build_error_response(
                    request=request,
                    generation_id=generation_id,
                    model_config=model_config,
                    warnings=[f"Prompt building failed: {str(e)}"],
                    total_start=total_start,
                )
            prompt_span.update(output=safe_dict(prompt))
        prompt_time = (time.time() - prompt_start) * 1000

        # ── Step 4: Generate answer ──────────────────────────────────
        gen_start = time.time()
        with traced_span(
            "llm_generate",
            as_type="generation",
            input=safe_dict({"system_prompt": prompt.system_prompt, "user_prompt": prompt.user_prompt}),
            model=model_config.model_name,
            metadata={
                "prompt_id": prompt.template_id,
                "prompt_version": prompt.template_version,
                "provider": model_config.provider,
            },
        ) as gen_span:
            try:
                generated = self.llm_client.generate(prompt, model_config)
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                gen_span.update(level="ERROR", status_message=str(e))
                return self._build_error_response(
                    request=request,
                    generation_id=generation_id,
                    model_config=model_config,
                    warnings=[f"LLM generation failed: {str(e)}"],
                    total_start=total_start,
                )

            cost_details = None
            if model_config.provider == "deepseek":
                cost_details = {
                    "total": estimate_deepseek_cost(
                        generated.usage.input_tokens, generated.usage.output_tokens
                    )
                }
            gen_span.update(
                output=safe_dict(generated.content),
                usage_details={
                    "input": generated.usage.input_tokens,
                    "output": generated.usage.output_tokens,
                    "total": generated.usage.total_tokens,
                },
                cost_details=cost_details,
            )
            if generated.finish_reason == "error":
                gen_span.update(level="ERROR", status_message=generated.error_type)
        gen_time = (time.time() - gen_start) * 1000

        # Check for empty response
        if not generated.content or generated.finish_reason == "error":
            warnings.append(f"LLM returned empty or error response: {generated.finish_reason}")
            return self._build_error_response(
                request=request,
                generation_id=generation_id,
                model_config=model_config,
                warnings=warnings,
                total_start=total_start,
                generated=generated,
            )

        # ── Step 5: Post-process ─────────────────────────────────────
        pp_start = time.time()
        with traced_span(
            "post_processor",
            input=safe_dict({"chunk_count": len(request.chunks), "mode": request.mode}),
        ) as pp_span:
            try:
                response = self.post_processor.process(
                    generated=generated,
                    chunks=request.chunks,
                    retrieval_metadata=request.retrieval_metadata,
                    mode=request.mode,
                    request_id=request.request_id,
                    generation_id=generation_id,
                    warnings=warnings,
                    sources_used=[],  # Will be populated by retrieval orchestrator
                )
            except Exception as e:
                logger.error(f"Post-processing failed: {e}")
                pp_span.update(level="ERROR", status_message=str(e))
                return self._build_error_response(
                    request=request,
                    generation_id=generation_id,
                    model_config=model_config,
                    warnings=warnings + [f"Post-processing failed: {str(e)}"],
                    total_start=total_start,
                    generated=generated,
                )
            pp_span.update(output=safe_dict({
                "is_grounded": response.is_grounded,
                "grounding_confidence": response.grounding_confidence,
                "citations_count": len(response.citations),
                "warnings": response.warnings,
            }))
        pp_time = (time.time() - pp_start) * 1000

        # ── Step 6: Set timing ───────────────────────────────────────
        total_time = (time.time() - total_start) * 1000

        response.generation_time_ms = generated.generation_time_ms
        response.total_time_ms = round(total_time, 2)
        response.prompt_version = prompt.template_version
        response.config_version = self.config.config_version

        # ── Step 7: Log diagnostics ─────────────────────────────────
        logger.info(
            f"Generation complete: mode={request.mode.value}, "
            f"model={model_config.model_name}, "
            f"tokens={generated.usage.total_tokens}, "
            f"grounded={response.is_grounded}, "
            f"citations={len(response.citations)}, "
            f"total_time={total_time:.0f}ms "
            f"(prompt={prompt_time:.0f}ms, gen={gen_time:.0f}ms, pp={pp_time:.0f}ms)"
        )

        return response
    
    def _build_error_response(
        self,
        request: GenerationRequest,
        generation_id: str,
        model_config: Optional[ModelConfig],
        warnings: List[str],
        total_start: float,
        generated: Optional[Any] = None,
    ) -> GenerationResponse:
        """
        Build a GenerationResponse for error cases.

        Returns a valid response object with empty answer and error details.
        model_config may be None if the error occurred before mode/model
        configuration could be resolved (e.g. unknown mode or model key).
        """
        from src.generation.config import UsageStats, ModelInfo

        total_time = (time.time() - total_start) * 1000

        if model_config is not None:
            model_info = model_config.to_model_info()
        else:
            model_info = ModelInfo(provider="unknown", model_name="unknown")

        return GenerationResponse(
            answer="",
            mode=request.mode,
            answer_format=AnswerFormat.PLAIN_TEXT,
            citations=[],
            is_grounded=False,
            grounding_confidence=0.0,
            retrieval_metadata=request.retrieval_metadata,
            model_info=model_info,
            usage=generated.usage if generated else UsageStats(),
            generation_time_ms=generated.generation_time_ms if generated else 0,
            total_time_ms=round(total_time, 2),
            request_id=request.request_id,
            generation_id=generation_id,
            warnings=warnings,
            sources_used=[],
            error_type=generated.error_type if generated else None,
            retry_after_seconds=generated.retry_after_seconds if generated else None,
            is_daily_quota=generated.is_daily_quota if generated else False,
        )


# ============================================================================
# Validation
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Generation Orchestrator - Validation")
    print("=" * 60)
    
    # Load configuration
    config = GenerationConfig.from_yaml()
    
    print(f"\n📋 Configuration:")
    print(f"   Default mode: {config.default_mode.value}")
    print(f"   Default model: {config.default_model_key}")
    print(f"   Available models: {list(config.models.keys())}")
    print(f"   Available modes: {list(config.modes.keys())}")
    print(f"   Post-processing: citations={config.post_processing.extract_citations}, "
          f"grounding={config.post_processing.validate_grounding}")
    
    # Initialize orchestrator
    orchestrator = GenerationOrchestrator(config=config)
    
    print(f"\n🔧 Orchestrator initialized:")
    print(f"   Components: PromptBuilder → LLMClient → PostProcessor")
    print(f"   Strategy selector: {list(orchestrator.strategy_selector._strategies.keys())}")
    
    # Build a sample request
    from src.generation.config import (
        GenerationRequest,
        RetrievedChunk,
        RetrievalMetadata,
        CitationLocationType,
        ConfidenceLevel,
    )
    
    sample_chunks = [
        RetrievedChunk(
            chunk_id="doc1_0001",
            content="Inventory refers to the stock of goods, materials, and supplies that a business holds for the purpose of resale, production, or use in providing services.",
            score=0.92,
            course_name="Supply Chain Management Strategy",
            chapter_title="Inventory Management",
            location_type=CitationLocationType.SLIDE,
            location_start=5,
            location_end=5,
            metadata={"chunk_type": "definition", "topic": "Inventory"}
        ),
        RetrievedChunk(
            chunk_id="doc1_0002",
            content="The main types of inventory include: raw materials, work-in-progress (WIP), finished goods, and MRO supplies.",
            score=0.88,
            course_name="Supply Chain Management Strategy",
            chapter_title="Inventory Management",
            location_type=CitationLocationType.SLIDE,
            location_start=6,
            location_end=6,
            metadata={"chunk_type": "explanation", "topic": "Inventory Types"}
        ),
        RetrievedChunk(
            chunk_id="doc1_0003",
            content="For example, a car manufacturer holds raw materials (steel, rubber), WIP (partially assembled vehicles), and finished goods (completed cars ready for sale).",
            score=0.82,
            course_name="Supply Chain Management Strategy",
            chapter_title="Inventory Management",
            location_type=CitationLocationType.SLIDE,
            location_start=7,
            location_end=7,
            metadata={"chunk_type": "example", "topic": "Inventory Example"}
        ),
    ]
    
    request = GenerationRequest(
        request_id="test-orch-001",
        query="What is inventory and what are its types?",
        mode=GenerationMode.CONTEXT_AWARE,
        chunks=sample_chunks,
        retrieval_metadata=RetrievalMetadata(
            confidence_score=0.87,
            confidence_level=ConfidenceLevel.HIGH,
            retrieval_time_ms=150.0,
            total_chunks_retrieved=3,
            namespace="supply_chain_test",
        ),
    )
    
    # Check if DeepSeek credentials are available
    deepseek_config = config.models.get("deepseek_chat")
    from src.common.llm_client import ProviderRegistry

    if deepseek_config and ProviderRegistry().get("deepseek").validate_credentials():
        print(f"\n{'─' * 50}")
        print("Running full generation pipeline...")
        
        response = orchestrator.generate(request)
        
        print(f"\n📊 Generation Results:")
        print(f"   Mode: {response.mode.value}")
        print(f"   Model: {response.model_info.provider}/{response.model_info.model_name}")
        print(f"   Tokens: {response.usage.input_tokens} in / {response.usage.output_tokens} out")
        print(f"   Time: generation={response.generation_time_ms:.0f}ms, total={response.total_time_ms:.0f}ms")
        print(f"   Citations: {len(response.citations)}")
        print(f"   Grounded: {response.is_grounded} (confidence: {response.grounding_confidence:.2f})")
        print(f"   Warnings: {response.warnings}")
        
        if response.answer:
            print(f"\n📝 Answer Preview:")
            print(response.answer[:500])
        else:
            print(f"\n❌ No answer generated — check warnings above")
    else:
        print(f"\n⚠️  DeepSeek credentials not found — skipping live generation test")
        print(f"   Set DEEPSEEK_API_KEY environment variable to run full test")
    
    print(f"\n✅ Generation Orchestrator ready for integration")