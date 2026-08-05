"""
DeepSeek Cost Estimation — Shared

No pricing constant exists anywhere else in this repo for the generation
LLM (only EmbeddingGenerator.estimate_cost() exists, and only for embedding
providers). These are deepseek-chat's published cache-miss rates at
https://api-docs.deepseek.com/quick_start/pricing at the time this module
was written — approximate, update if DeepSeek's pricing changes.

Kept in src/common/ (not eval/run_ragas_eval.py, where this originated) so
both the eval harness and the observability layer (observability/tracing.py,
src/generation/orchestrator.py's Langfuse generation spans) can import cost
calculation without eval/'s heavy transitive deps (unstructured, ragas, ...)
loading on every pipeline call.
"""

DEEPSEEK_PRICE_PER_1M_INPUT_USD = 0.27
DEEPSEEK_PRICE_PER_1M_OUTPUT_USD = 1.10


def estimate_deepseek_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        (input_tokens / 1_000_000) * DEEPSEEK_PRICE_PER_1M_INPUT_USD
        + (output_tokens / 1_000_000) * DEEPSEEK_PRICE_PER_1M_OUTPUT_USD
    )
