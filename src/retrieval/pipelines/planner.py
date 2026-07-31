"""Study planner pipeline configuration."""
from ..config import PipelineConfig, HybridWeights, ContextTemplate, ConfidenceLevel

PLANNER_PIPELINE = PipelineConfig(
    name="planner",
    description="Builds study roadmap from syllabus, topics, and weightage",
    hybrid_weights=HybridWeights(semantic=0.3, keyword=0.1, metadata=0.6),
    candidate_k=40,
    rerank_k=15,
    context_template=ContextTemplate.PLANNER,
    max_chunks_in_context=15,
    enable_query_expansion=False,     # Planner uses metadata, not semantic expansion
    enable_query_rewriting=False,
    min_confidence=ConfidenceLevel.MEDIUM,
    retry_on_low_confidence=True,
    retry_increase_k=20,
    cache_ttl_seconds=86400,          # 24 hours — study plans change slowly
    include_source_prefix=False,
    include_page_numbers=False,
)