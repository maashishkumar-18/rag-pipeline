"""
Core configuration types for the retrieval layer.
Defines PipelineConfig, retrieval result types, and shared enums.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum

from src.common.types import RetrievedChunk, CitationLocationType  # noqa: F401 — re-exported


# ============================================================================
# Enums
# ============================================================================

class RequestType(str, Enum):
    """Classification of user requests."""
    LEARNING = "learning"
    REVISION = "revision"
    QUESTION_ANSWERING = "question_answering"
    STUDY_PLANNING = "study_planning"
    QUIZ_GENERATION = "quiz_generation"
    TOPIC_NAVIGATION = "topic_navigation"
    SEARCH = "search"
    COMPARISON = "comparison"
    FOLLOW_UP = "follow_up"
    GENERAL = "general"


class ConfidenceLevel(str, Enum):
    """Retrieval confidence levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ContextTemplate(str, Enum):
    """Predefined context assembly templates."""
    LEARNING = "learning"           # definition → explanation → example → notes
    REVISION = "revision"           # keyword → formula → tip
    QA = "qa"                       # question → answer → marks
    PLANNER = "planner"             # topic → weightage → difficulty
    QUIZ = "quiz"                   # concept → distractors
    COMPARISON = "comparison"       # item_a → item_b → differences
    RAW = "raw"                     # no reordering, just concatenate


# ============================================================================
# Pipeline Configuration
# ============================================================================

@dataclass
class HybridWeights:
    """Weights for combining semantic, keyword, and metadata search results."""
    semantic: float = 0.6
    keyword: float = 0.2
    metadata: float = 0.2
    
    def __post_init__(self):
        total = self.semantic + self.keyword + self.metadata
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Hybrid weights must sum to 1.0, got {total}")


@dataclass
class PipelineConfig:
    """
    Complete configuration for a retrieval pipeline.
    
    Each pipeline (learning, revision, QA, etc.) is just a preset of this config.
    The orchestrator reads this config and composes the appropriate building blocks.
    """
    # Identity
    name: str
    description: str = ""
    
    # Search strategy
    hybrid_weights: HybridWeights = field(default_factory=HybridWeights)
    
    # Retrieval depth
    candidate_k: int = 20          # Number of candidates from hybrid search
    rerank_k: int = 5              # Number after cross-encoder reranking
    
    # Metadata filtering (Pinecone filter dict)
    metadata_filters: Dict[str, Any] = field(default_factory=dict)
    
    # Context assembly
    context_template: ContextTemplate = ContextTemplate.RAW
    max_chunks_in_context: int = 5
    
    # Query processing
    enable_query_expansion: bool = True
    expansion_terms: int = 5        # Max terms to add during expansion
    enable_query_rewriting: bool = True
    
    # Confidence & retry
    min_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    retry_on_low_confidence: bool = True
    retry_increase_k: int = 10     # Additional candidates on retry
    
    # Caching
    cache_ttl_seconds: int = 3600   # 1 hour default
    enable_cache: bool = True
    
    # Metadata fields to include in context
    include_source_prefix: bool = True
    include_page_numbers: bool = True


# ============================================================================
# Retrieval Result Types
# ============================================================================
#
# RetrievedChunk lives in src.common.types (imported above) — shared with
# the generation layer so no adapter is needed between the two.

@dataclass
class AssembledContext:
    """
    Final assembled context ready for the generation layer.
    This is what gets injected into the LLM prompt.
    """
    chunks: List[RetrievedChunk]
    formatted_context: str          # Full context string for the prompt
    confidence: ConfidenceLevel
    confidence_score: float         # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Diagnostics
    query_used: str = ""
    expanded_queries: List[str] = field(default_factory=list)
    total_candidates_retrieved: int = 0
    retrieval_time_ms: float = 0.0
    
    @property
    def is_sufficient(self) -> bool:
        """Check if context quality is sufficient for generation."""
        return self.confidence != ConfidenceLevel.LOW


@dataclass
class RetrievalResult:
    """
    Complete retrieval result including assembled context and diagnostics.
    """
    context: AssembledContext
    pipeline_name: str
    request_type: RequestType
    conversation_topic: str = ""
    cache_hit: bool = False
    
    # For follow-up questions
    current_topic: Optional[str] = None
    current_concept: Optional[str] = None