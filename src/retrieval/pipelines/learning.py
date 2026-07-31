"""Learning pipeline configuration."""
from ..config import PipelineConfig, HybridWeights, ContextTemplate, ConfidenceLevel

LEARNING_PIPELINE = PipelineConfig(
    name="learning",
    description="Explains concepts using study materials with high semantic fidelity",
    hybrid_weights=HybridWeights(semantic=0.7, keyword=0.2, metadata=0.1),
    candidate_k=20,
    rerank_k=5,
    context_template=ContextTemplate.LEARNING,
    max_chunks_in_context=5,
    enable_query_expansion=True,
    expansion_terms=3,
    enable_query_rewriting=True,
    min_confidence=ConfidenceLevel.MEDIUM,
    retry_on_low_confidence=True,
    retry_increase_k=10,
    cache_ttl_seconds=3600,
    include_source_prefix=True,
    include_page_numbers=True,
)