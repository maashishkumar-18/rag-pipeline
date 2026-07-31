"""Revision pipeline configuration."""
from ..config import PipelineConfig, HybridWeights, ContextTemplate, ConfidenceLevel

REVISION_PIPELINE = PipelineConfig(
    name="revision",
    description="Retrieves concise, high-importance content for last-minute review",
    hybrid_weights=HybridWeights(semantic=0.5, keyword=0.4, metadata=0.1),
    candidate_k=15,
    rerank_k=5,
    context_template=ContextTemplate.REVISION,
    max_chunks_in_context=5,
    enable_query_expansion=False,
    enable_query_rewriting=True,
    min_confidence=ConfidenceLevel.MEDIUM,
    retry_on_low_confidence=True,
    retry_increase_k=10,
    cache_ttl_seconds=1800,           # 30 minutes
    include_source_prefix=False,      # Revision doesn't need full citations
    include_page_numbers=False,
)