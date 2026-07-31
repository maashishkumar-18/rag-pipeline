"""Question answering pipeline configuration."""
from ..config import PipelineConfig, HybridWeights, ContextTemplate, ConfidenceLevel

QA_PIPELINE = PipelineConfig(
    name="qa",
    description="Retrieves PYQs and generates exam-style answers with mark schemes",
    hybrid_weights=HybridWeights(semantic=0.4, keyword=0.2, metadata=0.4),
    candidate_k=20,
    rerank_k=5,
    context_template=ContextTemplate.QA,
    max_chunks_in_context=5,
    enable_query_expansion=True,
    expansion_terms=5,
    enable_query_rewriting=True,
    min_confidence=ConfidenceLevel.MEDIUM,
    retry_on_low_confidence=True,
    retry_increase_k=10,
    cache_ttl_seconds=3600,
    include_source_prefix=True,
    include_page_numbers=True,
)