"""Quiz generation pipeline configuration."""
from ..config import PipelineConfig, HybridWeights, ContextTemplate, ConfidenceLevel

QUIZ_PIPELINE = PipelineConfig(
    name="quiz",
    description="Retrieves concepts and generates adaptive quiz questions",
    hybrid_weights=HybridWeights(semantic=0.6, keyword=0.1, metadata=0.3),
    candidate_k=12,
    rerank_k=5,
    context_template=ContextTemplate.QUIZ,
    max_chunks_in_context=5,
    enable_query_expansion=True,
    expansion_terms=4,
    enable_query_rewriting=False,
    min_confidence=ConfidenceLevel.MEDIUM,
    retry_on_low_confidence=False,    # Quiz can work with medium confidence
    cache_ttl_seconds=1800,
    include_source_prefix=False,
    include_page_numbers=False,
)