"""
Simulated Traffic Generator — for the observability dashboard

Fires a mix of golden-set questions and randomized/adversarial (out-of-
corpus, ambiguous, malformed) queries through the real retrieval+generation
pipeline, tagged env="simulated_production" in both Langfuse and
observability/metrics.db, so the Streamlit dashboard (dashboard/app.py) has
enough data to show trends rather than a single flat point.

For a chunk of the run (--incident-start-fraction through
--incident-start-fraction + --incident-duration-fraction), retrieval
deliberately runs against a degraded pipeline config (DEGRADED_PIPELINE
below: BM25 disabled, rerank_k slashed to 1, retry-on-low-confidence off) —
a plausible "someone misconfigured the reranker" incident, reproducible on
demand for the README/demo walkthrough ("day X, faithfulness dropped, here's
the dashboard, here's the root cause, here's the fix").

WHY NO BACKDATED TIMESTAMPS: the roadmap this script implements originally
called for compressing a multi-day traffic trend into one run by writing
fabricated past timestamps into the metrics store. Checked the Langfuse
Python SDK (4.14.2) directly — its span/trace creation APIs
(start_as_current_observation, start_observation) don't accept a custom
start_time, so Langfuse traces always get stamped with real wall-clock time.
Backdating only the local SQLite timestamps would make the dashboard's time
axis disagree with what a reviewer sees after clicking a trace link out to
Langfuse — exactly the confusing mismatch flagged during planning. Every
timestamp here is real; the incident window is identified by request
sequence and the retrieval_pipeline_name field/trace tags (not calendar
date), and the dashboard plots by request order alongside real time.

Usage:
    python -m scripts.simulate_traffic
    python -m scripts.simulate_traffic --num-requests 100 --incident-start-fraction 0.5
"""

import argparse
import logging
import random
import sys
import time
import uuid
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv()

import dataclasses

from src.retrieval.pipelines import LEARNING_PIPELINE
from src.retrieval.config import HybridWeights

from src.generation.config import (
    GenerationRequest,
    GenerationMode,
    RetrievalMetadata,
    ConfidenceLevel as GenConfidenceLevel,
)

from eval.run_ragas_eval import (
    bootstrap_pipeline,
    load_golden_set,
    score_unanswerable_item,
    build_judge_llm,
    GOLDEN_SET_PATH,
)

from src.common.pricing import estimate_deepseek_cost
from observability.tracing import traced_pipeline_call, current_trace_id, current_trace_url, score_trace, flush
from observability.metrics_store import MetricsStore, PipelineCallMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("simulate_traffic")

# ============================================================================
# Degraded pipeline config for the injected incident.
#
# First version of this only cut rerank_k to 1 while leaving candidate_k at
# LEARNING_PIPELINE's default (20) — the reranker still picked the single
# best chunk out of 20 good semantic candidates, so faithfulness (which only
# checks whether the answer is grounded in *whatever context it was given*,
# not whether that context was complete) barely moved: a real run showed
# 3/3 in-window faithfulness samples still scoring 1.0. Honest finding, but
# not a useful incident demo — the fix is to also starve the candidate pool
# itself (candidate_k slashed to 3), not just the final selection, so the
# reranker has too little to work with and a genuinely weak/irrelevant chunk
# is more likely to be the one that survives. Combined with BM25 disabled
# (semantic-only hybrid search) and retry-on-low-confidence off (so the
# orchestrator's own retry logic can't quietly compensate and mask it).
# ============================================================================

DEGRADED_PIPELINE = dataclasses.replace(
    LEARNING_PIPELINE,
    name="learning_degraded_incident",
    hybrid_weights=HybridWeights(semantic=1.0, keyword=0.0, metadata=0.0),
    candidate_k=3,
    rerank_k=1,
    enable_query_expansion=False,
    retry_on_low_confidence=False,
)

# Out-of-corpus / ambiguous / malformed queries — exercise the refusal path
# and edge cases a hand-picked golden set won't cover on its own.
ADVERSARIAL_QUERIES = [
    "What is the boiling point of nitrogen?",
    "How do I file my taxes in California?",
    "Explain the plot of Hamlet.",
    "What's the capital of Mongolia?",
    "asdkjfh aslkdjf qwoieuroiu",
    "",
    "Explain quantum entanglement in the context of supply chains.",
    "What did the CEO say in yesterday's earnings call?",
    "Summarize chapter 47.",
    "Tell me a joke about inventory.",
]


def compute_faithfulness(question: str, answer: str, contexts: List[str], judge_llm) -> Optional[float]:
    """
    Single-metric RAGAS faithfulness check — cheaper than the eval harness's
    full 4-metric evaluate() call, and doesn't need ground_truth (faithfulness
    only checks whether the answer's claims are supported by contexts), so
    it can score adversarial/ad-hoc queries the golden set has no ground
    truth for, not just golden-set items.
    """
    if not contexts or not answer:
        return None
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness
        from ragas.run_config import RunConfig

        dataset = Dataset.from_list([{"question": question, "answer": answer, "contexts": contexts}])
        result = evaluate(
            dataset, metrics=[faithfulness], llm=judge_llm,
            run_config=RunConfig(max_workers=1, timeout=60), raise_exceptions=False,
        )
        value = result.to_pandas().iloc[0].get("faithfulness")
        if value is None or (isinstance(value, float) and value != value):  # NaN check
            return None
        return float(value)
    except Exception:
        logger.exception("Faithfulness scoring failed for this sampled request; skipping.")
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate traffic against the real RAG pipeline for the observability dashboard.")
    parser.add_argument("--num-requests", type=int, default=50, help="Total number of simulated requests.")
    parser.add_argument("--adversarial-rate", type=float, default=0.15, help="Probability per request of drawing from ADVERSARIAL_QUERIES instead of the golden set.")
    parser.add_argument("--incident-start-fraction", type=float, default=0.4, help="Fraction of requests (0-1) into the run where the degraded-config incident begins.")
    parser.add_argument("--incident-duration-fraction", type=float, default=0.2, help="Fraction of total requests (0-1) affected by the incident.")
    parser.add_argument("--faithfulness-sample-rate", type=int, default=4, help="Run RAGAS faithfulness on every Nth non-refused answered request.")
    parser.add_argument("--delay-seconds", type=float, default=0.0, help="Optional sleep between requests.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--golden-set", type=Path, default=GOLDEN_SET_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    logger.info("Bootstrapping pipeline against the sample corpus...")
    pipeline = bootstrap_pipeline()
    golden_items = load_golden_set(args.golden_set)
    judge_llm = build_judge_llm()
    store = MetricsStore()

    incident_start_idx = int(args.num_requests * args.incident_start_fraction)
    incident_end_idx = int(incident_start_idx + args.num_requests * args.incident_duration_fraction)
    logger.info(
        "Simulating %d requests. Incident window: requests [%d, %d) run against '%s' "
        "instead of '%s'.",
        args.num_requests, incident_start_idx, incident_end_idx,
        DEGRADED_PIPELINE.name, LEARNING_PIPELINE.name,
    )

    faithfulness_eligible_count = 0

    for i in range(1, args.num_requests + 1):
        in_incident = incident_start_idx <= i < incident_end_idx
        pipeline_config = DEGRADED_PIPELINE if in_incident else LEARNING_PIPELINE

        if rng.random() < args.adversarial_rate:
            query = rng.choice(ADVERSARIAL_QUERIES)
            source, category = "adversarial", "adversarial"
        else:
            item = rng.choice(golden_items)
            query, category = item["question"], item["category"]
            source = "golden"

        request_id = str(uuid.uuid4())
        item_start = time.time()

        try:
            with traced_pipeline_call(
                request_id=request_id,
                query=query,
                env="simulated_production",
                tags=[source, category, "incident" if in_incident else "normal"],
                metadata={"pipeline_config": pipeline_config.name},
            ) as trace:
                result = pipeline.retrieval_orchestrator.retrieve(
                    query=query,
                    pipeline_config=pipeline_config,
                    conversation_state=pipeline.conversation_state,
                    namespace=pipeline.namespace,
                )
                retrieval_metadata = RetrievalMetadata(
                    confidence_score=result.confidence.score,
                    confidence_level=GenConfidenceLevel(result.confidence.level.value),
                    retrieval_time_ms=result.total_time_ms,
                    total_chunks_retrieved=result.candidates_retrieved,
                    namespace=pipeline.namespace,
                    top_k=len(result.assembled_context.chunks),
                    retrieval_method=result.routing_decision,
                )
                gen_request = GenerationRequest(
                    request_id=request_id,
                    query=query,
                    mode=GenerationMode.CONTEXT_AWARE,
                    chunks=result.assembled_context.chunks,
                    retrieval_metadata=retrieval_metadata,
                )
                response = pipeline.generation_orchestrator.generate(gen_request)
                trace.update(output={"answer": response.answer, "is_grounded": response.is_grounded})

                # Must capture inside the `with` block — no active span once it exits.
                trace_id = current_trace_id()
                trace_url = current_trace_url()

            _, refusal_signals = score_unanswerable_item(result, response)
            refused = bool(refusal_signals["refused"])

            faithfulness_score = None
            if not refused:
                if faithfulness_eligible_count % args.faithfulness_sample_rate == 0:
                    faithfulness_score = compute_faithfulness(
                        question=query,
                        answer=response.answer,
                        contexts=[c.content for c in result.assembled_context.chunks],
                        judge_llm=judge_llm,
                    )
                    if faithfulness_score is not None and trace_id:
                        score_trace(trace_id, "faithfulness", faithfulness_score)
                faithfulness_eligible_count += 1

            cost_usd = estimate_deepseek_cost(response.usage.input_tokens, response.usage.output_tokens)

            store.record(PipelineCallMetrics(
                request_id=request_id,
                env="simulated_production",
                query=query,
                total_time_ms=(time.time() - item_start) * 1000,
                retrieval_time_ms=result.total_time_ms,
                generation_time_ms=response.generation_time_ms,
                stage_timings=result.timing_breakdown,
                confidence_score=result.confidence.score,
                confidence_level=result.confidence.level.value,
                retrieval_hit=result.confidence.level.value != "low",
                candidates_retrieved=result.candidates_retrieved,
                is_grounded=response.is_grounded,
                citations_count=len(response.citations),
                refused=refused,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cost_usd=cost_usd,
                prompt_version=response.prompt_version,
                model_name=response.model_info.model_name,
                retrieval_pipeline_name=result.pipeline_name,
                faithfulness_score=faithfulness_score,
                langfuse_trace_id=trace_id,
                langfuse_trace_url=trace_url,
            ))

            logger.info(
                "[%d/%d]%s %s/%s confidence=%s refused=%s faithfulness=%s",
                i, args.num_requests, " [INCIDENT]" if in_incident else "",
                source, category, result.confidence.level.value, refused,
                f"{faithfulness_score:.2f}" if faithfulness_score is not None else "n/a",
            )

        except Exception:
            # A single bad request (e.g. the empty-string adversarial query
            # hitting a component that doesn't expect it) must not kill the
            # rest of the simulated run — record it as a failed call and
            # keep going, same as real traffic would need to.
            logger.exception("[%d/%d] request failed unexpectedly — recording as failed and continuing", i, args.num_requests)
            store.record(PipelineCallMetrics(
                request_id=request_id,
                env="simulated_production",
                query=query,
                total_time_ms=(time.time() - item_start) * 1000,
                refused=True,
                retrieval_pipeline_name=pipeline_config.name,
            ))

        if args.delay_seconds:
            time.sleep(args.delay_seconds)

    flush()
    logger.info(
        "Done. %d requests simulated (incident window=[%d, %d)).",
        args.num_requests, incident_start_idx, incident_end_idx,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
