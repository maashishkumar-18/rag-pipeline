"""
RAGAS Evaluation Harness
Runs the golden QA set (eval/golden_qa_set.json) through the full
retrieval → generation pipeline and scores it with RAGAS.

Judge model: DeepSeek (via langchain_openai.ChatOpenAI pointed at
DeepSeek's OpenAI-compatible endpoint) — a separate call path from the
pipeline's own LLMClient/DeepSeekAdapter, even though both currently
resolve to the same "deepseek-chat" model per config/generation/models.yaml.
Keeping the judge on its own client means it stays independent if the
pipeline's default model ever changes, and the pipeline is never grading
itself through the exact code path being evaluated.

Judge embeddings: a local sentence-transformers model (see
"WHY LOCAL EMBEDDINGS" below), needed only for answer_correctness's
semantic-similarity component — faithfulness/context_precision/
context_recall are LLM-only in ragas 0.1.x.

Scoring:
- Answerable items: mean of faithfulness, context_precision,
  context_recall, answer_correctness (RAGAS, batched through one
  evaluate() call for efficiency).
- Unanswerable items: no ground_truth exists to score RAGAS metrics
  against by design, so these are scored 1.0 (pipeline correctly
  hedged/refused) or 0.0 (pipeline confidently answered anyway) using
  structured signals already produced by the pipeline — see
  score_unanswerable_item().
- Aggregate score: mean of all per-question composite scores.

Usage:
    python -m eval.run_ragas_eval
    python -m eval.run_ragas_eval --sample-size 15 --threshold 0.7
See eval/README.md for full docs.
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv()

from src.ingestion.parser import DocumentParser
from src.ingestion.cleaner import TextCleaner
from src.ingestion.metadata_extractor import MetadataExtractor
from src.ingestion.chunker import SemanticChunker
from src.ingestion.enricher import ChunkEnricher
from src.ingestion.embedder import EmbeddingGenerator
from src.ingestion.vector_store import VectorStore

from src.retrieval.orchestrator import RetrievalOrchestrator, OrchestratorResult
from src.retrieval.query_rewriter import ConversationState
from src.retrieval.pipelines import LEARNING_PIPELINE

from src.generation.orchestrator import GenerationOrchestrator
from src.generation.config import (
    GenerationRequest,
    GenerationResponse,
    GenerationMode,
    RetrievalMetadata,
    ConfidenceLevel as GenConfidenceLevel,
)

from observability.tracing import traced_pipeline_call, current_trace_id, current_trace_url, score_trace, flush
from observability.metrics_store import MetricsStore, PipelineCallMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ragas_eval")

SAMPLE_FILE = str(REPO_ROOT / "sample_data" / "sample_lecture.pptx")
GOLDEN_SET_PATH = REPO_ROOT / "eval" / "golden_qa_set.json"
RESULTS_DIR = REPO_ROOT / "eval" / "results"

# ============================================================================
# Refusal detection for unanswerable items
#
# config/generation/prompts/context_aware.yaml's system prompt explicitly
# instructs the LLM to emit this exact phrase when the context is
# insufficient (rule #2). Checking for it is not a heuristic invented for
# this script — it's the structured signal the pipeline already produces
# on purpose. Combined with the retrieval-layer's own confidence.action
# (src/retrieval/confidence.py) and OrchestratorResult.is_ready_for_generation.
# ============================================================================

# The instructed phrase is "...does not cover this topic in sufficient
# detail", but the LLM sometimes (correctly, more naturally) substitutes the
# actual topic name for "this topic" — e.g. "does not cover Just-In-Time
# (JIT) inventory management in sufficient detail" — which breaks an exact
# substring match. Match the two anchor fragments allowing arbitrary text
# (the substituted topic name) in between, within a reasonable span so it
# doesn't false-positive on two unrelated occurrences far apart.
REFUSAL_PATTERNS = (
    re.compile(r"does not cover .{0,120}?in sufficient detail", re.IGNORECASE),
)

# confidence.py's ConfidenceScorer._determine_action / orchestrator.py's
# fallback/clarify/timeout/error builders — any of these mean the retrieval
# layer itself flagged the request as not confidently answerable.
LOW_CONFIDENCE_ACTIONS = {"retry_or_clarify", "fallback", "clarify", "timeout", "error"}


def _contains_refusal_phrase(answer: str) -> bool:
    text = answer or ""
    return any(pattern.search(text) for pattern in REFUSAL_PATTERNS)


def score_unanswerable_item(result: OrchestratorResult, response: GenerationResponse) -> "tuple[float, Dict[str, Any]]":
    """
    1.0 if the pipeline hedged/refused (correct — the info isn't in the
    corpus), 0.0 if it generated a confident unhedged answer anyway.
    """
    not_ready = not result.is_ready_for_generation
    low_confidence_action = result.confidence.action in LOW_CONFIDENCE_ACTIONS
    refusal_phrase = _contains_refusal_phrase(response.answer)
    generation_failed = response.error_type is not None

    refused = not_ready or low_confidence_action or refusal_phrase or generation_failed
    signals = {
        "is_ready_for_generation": result.is_ready_for_generation,
        "confidence_action": result.confidence.action,
        "confidence_level": result.confidence.level.value,
        "refusal_phrase_detected": refusal_phrase,
        "generation_error": response.error_type,
        # Exposed so callers outside this scoring function (e.g.
        # observability metrics recording, which needs "did the pipeline
        # refuse" for every item, not just unanswerable ones) can reuse this
        # exact signal instead of re-deriving it.
        "refused": refused,
    }
    return (1.0 if refused else 0.0), signals


# ============================================================================
# DeepSeek cost estimate — see src/common/pricing.py for the constants and
# rationale (moved there so observability/tracing.py can reuse it without
# pulling in this module's heavy transitive deps).
# ============================================================================

from src.common.pricing import estimate_deepseek_cost  # noqa: E402


# ============================================================================
# Pipeline bootstrap — mirrors src/retrieval/orchestrator.py's __main__
# integration test: ingest the sample deck into a fresh Pinecone namespace,
# build the in-memory BM25 index, and construct both orchestrators.
# ============================================================================

@dataclass
class Pipeline:
    retrieval_orchestrator: RetrievalOrchestrator
    generation_orchestrator: GenerationOrchestrator
    namespace: str
    conversation_state: ConversationState


def bootstrap_pipeline() -> Pipeline:
    logger.info("Ingesting sample corpus (%s)...", SAMPLE_FILE)
    parsed = DocumentParser().parse(SAMPLE_FILE)
    cleaned = TextCleaner().clean(parsed)
    metadata = MetadataExtractor().extract_with_retry(cleaned)
    chunks = SemanticChunker().chunk(cleaned, metadata)
    enriched = ChunkEnricher().enrich(chunks)
    embedded = EmbeddingGenerator().embed_chunks(enriched)

    vector_store = VectorStore()
    namespace = vector_store.generate_namespace_from_document(SAMPLE_FILE, metadata.course_name)
    upsert_result = vector_store.upsert(embedded, namespace=namespace)
    logger.info(
        "Ingested %d chunks into namespace '%s' (course=%s, chapter=%s)",
        upsert_result.chunks_upserted, namespace, metadata.course_name, metadata.chapter_title,
    )

    retrieval_orchestrator = RetrievalOrchestrator(vector_store=vector_store)

    bm25_documents = []
    for chunk in enriched:
        chunk_metadata = ChunkEnricher.to_metadata(chunk)
        chunk_metadata["namespace"] = namespace
        bm25_documents.append({
            "id": chunk.chunk_id,
            "content": chunk.content,
            "raw_content": chunk.raw_content,
            "metadata": chunk_metadata,
        })
    retrieval_orchestrator.hybrid_search.initialize_indices(documents=bm25_documents)
    logger.info("BM25 index initialized with %d documents", len(bm25_documents))

    conversation_state = ConversationState(
        session_id="ragas_eval",
        subject=metadata.course_name,
        module=metadata.course_name,
        chapter=metadata.chapter_title,
    )

    generation_orchestrator = GenerationOrchestrator()

    return Pipeline(
        retrieval_orchestrator=retrieval_orchestrator,
        generation_orchestrator=generation_orchestrator,
        namespace=namespace,
        conversation_state=conversation_state,
    )


# ============================================================================
# Golden set loading / sampling
# ============================================================================

def load_golden_set(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["items"]


def sample_items(items: List[Dict[str, Any]], sample_size: Optional[int], seed: int) -> List[Dict[str, Any]]:
    if sample_size is None or sample_size >= len(items):
        return items
    rng = random.Random(seed)
    return rng.sample(items, sample_size)


# ============================================================================
# Phase A — run each item through the real pipeline
# ============================================================================

@dataclass
class PipelineRecord:
    id: str
    question: str
    category: str
    answerable: bool
    ground_truth: Optional[str]
    generated_answer: str
    contexts: List[str]
    result: OrchestratorResult
    response: GenerationResponse
    pipeline_time_ms: float
    langfuse_trace_id: Optional[str] = None
    langfuse_trace_url: Optional[str] = None


def run_pipeline_phase(
    pipeline: Pipeline, items: List[Dict[str, Any]], env: str = "ci"
) -> "tuple[List[PipelineRecord], int, int]":
    records: List[PipelineRecord] = []
    running_input_tokens = 0
    running_output_tokens = 0

    for idx, item in enumerate(items, start=1):
        item_start = time.time()
        request_id = str(uuid.uuid4())

        # One trace per golden-set item, tagged env=ci — RAGAS/composite
        # scores get attached after the fact (see record_observability_metrics),
        # once Phase B has scored this batch, since score computation and
        # trace lifetime don't overlap (RAGAS runs as one batched evaluate()
        # call across all answerable items, not interleaved per item).
        with traced_pipeline_call(
            request_id=request_id,
            query=item["question"],
            env=env,
            tags=[item["category"], "answerable" if item["answerable"] else "unanswerable"],
            metadata={"golden_set_id": item["id"]},
        ) as trace:
            result = pipeline.retrieval_orchestrator.retrieve(
                query=item["question"],
                pipeline_config=LEARNING_PIPELINE,
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

            request = GenerationRequest(
                request_id=request_id,
                query=item["question"],
                mode=GenerationMode.CONTEXT_AWARE,
                chunks=result.assembled_context.chunks,
                retrieval_metadata=retrieval_metadata,
            )
            response = pipeline.generation_orchestrator.generate(request)
            trace.update(output={"answer": response.answer, "is_grounded": response.is_grounded})

            # Must capture inside the `with` block — current_trace_id()/
            # current_trace_url() need an active span; there isn't one once
            # traced_pipeline_call's context manager has exited.
            trace_id = current_trace_id()
            trace_url = current_trace_url()

        running_input_tokens += response.usage.input_tokens
        running_output_tokens += response.usage.output_tokens
        running_cost = estimate_deepseek_cost(running_input_tokens, running_output_tokens)

        pipeline_time_ms = (time.time() - item_start) * 1000
        answerability = "answerable" if item["answerable"] else "unanswerable"
        logger.info(
            "[%d/%d] %s (%s, %s) retrieval=%.0fms gen=%.0fms tokens=%d running_pipeline_cost=$%.4f",
            idx, len(items), item["id"], item["category"], answerability,
            result.total_time_ms, response.generation_time_ms,
            response.usage.total_tokens, running_cost,
        )

        records.append(PipelineRecord(
            id=item["id"],
            question=item["question"],
            category=item["category"],
            answerable=item["answerable"],
            ground_truth=item.get("ground_truth"),
            generated_answer=response.answer,
            contexts=[c.content for c in result.assembled_context.chunks],
            result=result,
            response=response,
            pipeline_time_ms=pipeline_time_ms,
            langfuse_trace_id=trace_id,
            langfuse_trace_url=trace_url,
        ))

    return records, running_input_tokens, running_output_tokens


# ============================================================================
# Observability — push metrics + scores now that both the pipeline run
# (Phase A, with trace ids captured per item) and RAGAS scoring (Phase B,
# batched) have completed.
# ============================================================================

def record_observability_metrics(
    records: List["PipelineRecord"],
    ragas_scores: Dict[str, Dict[str, float]],
    env: str = "ci",
) -> None:
    store = MetricsStore()

    for r in records:
        _, refusal_signals = score_unanswerable_item(r.result, r.response)
        refused = bool(refusal_signals["refused"])

        item_metrics = ragas_scores.get(r.id, {}) if r.answerable else {}
        faithfulness = item_metrics.get("faithfulness")
        composite = (
            (sum(item_metrics.values()) / len(item_metrics)) if item_metrics
            else (1.0 if refused else 0.0)
        )

        cost_usd = estimate_deepseek_cost(r.response.usage.input_tokens, r.response.usage.output_tokens)

        store.record(PipelineCallMetrics(
            request_id=r.response.request_id,
            env=env,
            query=r.question,
            total_time_ms=r.pipeline_time_ms,
            retrieval_time_ms=r.result.total_time_ms,
            generation_time_ms=r.response.generation_time_ms,
            stage_timings=r.result.timing_breakdown,
            confidence_score=r.result.confidence.score,
            confidence_level=r.result.confidence.level.value,
            retrieval_hit=r.result.confidence.level.value != "low",
            candidates_retrieved=r.result.candidates_retrieved,
            is_grounded=r.response.is_grounded,
            citations_count=len(r.response.citations),
            refused=refused,
            input_tokens=r.response.usage.input_tokens,
            output_tokens=r.response.usage.output_tokens,
            cost_usd=cost_usd,
            prompt_version=r.response.prompt_version,
            model_name=r.response.model_info.model_name,
            retrieval_pipeline_name=r.result.pipeline_name,
            faithfulness_score=faithfulness,
            langfuse_trace_id=r.langfuse_trace_id,
            langfuse_trace_url=r.langfuse_trace_url,
        ))

        if r.langfuse_trace_id:
            score_trace(r.langfuse_trace_id, "composite_score", composite)
            if faithfulness is not None:
                score_trace(r.langfuse_trace_id, "faithfulness", faithfulness)


# ============================================================================
# Phase B — RAGAS judge (answerable items only)
# ============================================================================

def build_judge_llm():
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required to run the RAGAS judge LLM.")
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com",
        temperature=0,
    )


def build_judge_embeddings():
    """
    WHY LOCAL EMBEDDINGS:
    The pipeline's own EmbeddingGenerator (src/ingestion/embedder.py) is
    not a drop-in LangChain Embeddings implementation — it exposes
    embed_query()/embed_queries(), not the embed_query()/embed_documents()
    pair RAGAS's Embeddings interface expects, and it's backed by a live
    paid provider (Gemini/OpenAI) whose cost would then get conflated with
    judge cost on every eval run. RAGAS only needs embeddings for
    answer_correctness's semantic-similarity term (faithfulness /
    context_precision / context_recall are LLM-only in ragas 0.1.x), so a
    small local sentence-transformers model is a lightweight, free,
    deterministic fallback — and sentence-transformers is already a
    pipeline dependency (used by the cross-encoder reranker).
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def run_ragas_phase(records: List[PipelineRecord]) -> "tuple[Dict[str, Dict[str, float]], Dict[str, int]]":
    answerable = [r for r in records if r.answerable]
    if not answerable:
        return {}, {"input_tokens": 0, "output_tokens": 0}

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, context_precision, context_recall, answer_correctness
    from ragas.run_config import RunConfig

    try:
        from langchain_community.callbacks import get_openai_callback
    except ImportError:  # pragma: no cover - optional cost tracking only
        get_openai_callback = None

    logger.info("Running RAGAS judge over %d answerable items...", len(answerable))

    dataset = Dataset.from_list([
        {
            "question": r.question,
            "answer": r.generated_answer,
            "contexts": r.contexts,
            "ground_truth": r.ground_truth,
        }
        for r in answerable
    ])

    judge_llm = build_judge_llm()
    judge_embeddings = build_judge_embeddings()
    metrics = [faithfulness, context_precision, context_recall, answer_correctness]

    judge_usage = {"input_tokens": 0, "output_tokens": 0}
    cb_ctx = get_openai_callback() if get_openai_callback else None
    cb = cb_ctx.__enter__() if cb_ctx else None
    try:
        ragas_result = evaluate(
            dataset,
            metrics=metrics,
            llm=judge_llm,
            embeddings=judge_embeddings,
            run_config=RunConfig(max_workers=4, timeout=180),
            raise_exceptions=False,
        )
    finally:
        if cb_ctx:
            cb_ctx.__exit__(None, None, None)
            judge_usage["input_tokens"] = cb.prompt_tokens
            judge_usage["output_tokens"] = cb.completion_tokens

    df = ragas_result.to_pandas()
    metric_names = ["faithfulness", "context_precision", "context_recall", "answer_correctness"]

    scores_by_id: Dict[str, Dict[str, float]] = {}
    for record, (_, row) in zip(answerable, df.iterrows()):
        metric_scores = {}
        for name in metric_names:
            value = row.get(name)
            metric_scores[name] = 0.0 if value is None or (isinstance(value, float) and value != value) else float(value)
        scores_by_id[record.id] = metric_scores

    return scores_by_id, judge_usage


# ============================================================================
# Phase C — assemble report
# ============================================================================

def build_report(
    records: List[PipelineRecord],
    ragas_scores: Dict[str, Dict[str, float]],
    args: argparse.Namespace,
    duration_seconds: float,
    pipeline_tokens: "tuple[int, int]",
    judge_tokens: Dict[str, int],
) -> Dict[str, Any]:
    skip_ragas = getattr(args, "skip_ragas", False)
    items_report = []
    for r in records:
        if r.answerable:
            if skip_ragas:
                composite, metrics, refusal_signals = None, None, None
            else:
                metrics = ragas_scores.get(r.id, {})
                composite = sum(metrics.values()) / len(metrics) if metrics else 0.0
                refusal_signals = None
        else:
            composite, refusal_signals = score_unanswerable_item(r.result, r.response)
            metrics = {}

        items_report.append({
            "id": r.id,
            "question": r.question,
            "category": r.category,
            "answerable": r.answerable,
            "ground_truth": r.ground_truth,
            "generated_answer": r.generated_answer,
            "is_ready_for_generation": r.result.is_ready_for_generation,
            "metrics": metrics or None,
            "refusal_signals": refusal_signals,
            "composite_score": round(composite, 4) if composite is not None else None,
            "flagged_low_score": (composite is not None) and (composite < args.flag_threshold),
            "pipeline_time_ms": round(r.pipeline_time_ms, 1),
            # Diagnostics for separating "genuinely refused/low-scored" from
            # "retrieval found nothing because of a routing/filter bug" (the
            # agent-path metadata-filter mismatch — see src/retrieval/
            # orchestrator.py's AGENT_FILTER_EXCLUDED_FIELDS and
            # MetadataFilterBuilder._build_from_dict). A 0-candidate item
            # scored as a "correct refusal" or "low faithfulness" would be a
            # false signal, not a true one — kept here as a permanent
            # diagnostic in case of regressions, not because the bug is
            # currently open.
            "agent_used": r.result.agent_used,
            "routing_decision": r.result.routing_decision,
            "candidates_retrieved": r.result.candidates_retrieved,
            "candidates_reranked": r.result.candidates_reranked,
            "citations_count": len(r.response.citations),
        })

    scored_items = [it for it in items_report if it["composite_score"] is not None]
    all_scores = [it["composite_score"] for it in scored_items]
    aggregate_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    unscored_count = len(items_report) - len(scored_items)

    # Citation-count gate — added after scripts/simulate_traffic.py's
    # incident simulation found that RAGAS faithfulness alone doesn't catch
    # a starved-retrieval regression: an answer from 1 narrow chunk instead
    # of 5 well-chosen ones is still "faithful" to what little it got, but
    # cites a third fewer sources. Computed over answerable items only —
    # correctly-refused unanswerable items legitimately have 0 citations
    # and would bias the average down for the wrong reason. See
    # eval/README.md "Citation-count baseline" for the threshold rationale.
    answerable_items = [it for it in items_report if it["answerable"]]
    avg_citations = (
        sum(it["citations_count"] for it in answerable_items) / len(answerable_items)
        if answerable_items else 0.0
    )

    def _breakdown(key_fn):
        buckets: Dict[str, List[float]] = {}
        counts: Dict[str, int] = {}
        for it in items_report:
            key = key_fn(it)
            counts[key] = counts.get(key, 0) + 1
            if it["composite_score"] is not None:
                buckets.setdefault(key, []).append(it["composite_score"])
        result = {}
        for k in sorted(counts.keys()):
            v = buckets.get(k, [])
            result[k] = {
                "mean_score": round(sum(v) / len(v), 4) if v else None,
                "n": counts[k],
                "n_scored": len(v),
            }
        return result

    breakdown_by_category = _breakdown(lambda it: it["category"])
    breakdown_by_answerability = _breakdown(lambda it: "answerable" if it["answerable"] else "unanswerable")
    low_scoring = [it for it in items_report if it["flagged_low_score"]]

    # Items where the agent path fired AND retrieval came back empty — a
    # zero-candidate result here scores as "correct refusal" (unanswerable)
    # or "low faithfulness" (answerable) for the wrong reason: a routing/
    # metadata-filter bug, not a genuine "info isn't in the deck" signal.
    # Surfaced separately so a real gap in the corpus isn't confused with
    # this known failure mode.
    suspect_agent_zero_candidates = [
        {"id": it["id"], "answerable": it["answerable"], "category": it["category"], "composite_score": it["composite_score"]}
        for it in items_report
        if it["agent_used"] and it["candidates_retrieved"] == 0
    ]

    pipeline_input_tokens, pipeline_output_tokens = pipeline_tokens
    pipeline_cost = estimate_deepseek_cost(pipeline_input_tokens, pipeline_output_tokens)
    judge_cost = estimate_deepseek_cost(judge_tokens.get("input_tokens", 0), judge_tokens.get("output_tokens", 0))

    return {
        "run_metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(duration_seconds, 1),
            "sample_size": len(records),
            "total_items_in_golden_set": args.total_items_in_golden_set,
            "seed": args.seed,
            "threshold": args.threshold,
            "flag_threshold": args.flag_threshold,
            "ragas_skipped": skip_ragas,
            "unscored_items": unscored_count,
            "judge_llm": "skipped (--skip-ragas)" if skip_ragas else "deepseek-chat (langchain_openai.ChatOpenAI, DeepSeek endpoint)",
            "judge_embeddings": "skipped (--skip-ragas)" if skip_ragas else "sentence-transformers/all-MiniLM-L6-v2 (local fallback)",
            "pipeline_model": "deepseek_chat (config/generation/models.yaml default)",
            "cost_estimate_usd": {
                "pipeline_generation": round(pipeline_cost, 4),
                "ragas_judge": round(judge_cost, 4),
                "total": round(pipeline_cost + judge_cost, 4),
                "note": "Approximate (see DEEPSEEK_PRICE_PER_1M_* constants). Excludes one-time ingestion cost (chunking/metadata extraction/embedding of the sample deck).",
            },
        },
        "aggregate_score": round(aggregate_score, 4),
        "aggregate_score_note": (
            f"Computed over {len(scored_items)}/{len(items_report)} items — "
            f"{unscored_count} answerable item(s) unscored due to --skip-ragas."
            if skip_ragas else None
        ),
        "threshold": args.threshold,
        "avg_citations_answerable": round(avg_citations, 3),
        "min_avg_citations": args.min_avg_citations if args.min_avg_citations > 0 else None,
        "citations_gate_passed": (
            True if (skip_ragas or args.min_avg_citations <= 0)
            else avg_citations >= args.min_avg_citations
        ),
        "passed": (
            (aggregate_score >= args.threshold)
            and (skip_ragas or args.min_avg_citations <= 0 or avg_citations >= args.min_avg_citations)
        ),
        "breakdown_by_category": breakdown_by_category,
        "breakdown_by_answerability": breakdown_by_answerability,
        "low_scoring_items": [
            {"id": it["id"], "question": it["question"], "category": it["category"], "composite_score": it["composite_score"]}
            for it in low_scoring
        ],
        "suspect_agent_zero_candidates": suspect_agent_zero_candidates,
        "items": items_report,
    }


def _fmt_score(value: Optional[float]) -> str:
    return f"{value:.4f}" if value is not None else "n/a"


def render_markdown(report: Dict[str, Any]) -> str:
    meta = report["run_metadata"]
    lines = [
        "# RAGAS Evaluation Report",
        "",
        f"**Aggregate score: {report['aggregate_score']:.4f}** "
        f"(threshold {report['threshold']:.2f} — "
        f"{'PASSED' if report['aggregate_score'] >= report['threshold'] else 'FAILED'})",
    ]
    if report.get("aggregate_score_note"):
        lines.append(f"*{report['aggregate_score_note']}*")
    if report.get("min_avg_citations") is not None:
        lines.append(
            f"**Avg citations/answer (answerable items): {report['avg_citations_answerable']:.2f}** "
            f"(min {report['min_avg_citations']:.2f} — "
            f"{'PASSED' if report['citations_gate_passed'] else 'FAILED'})"
        )
    lines += [
        "",
        f"**Overall: {'PASSED' if report['passed'] else 'FAILED'}**",
        "",
        f"- Run at: {meta['timestamp']}",
        f"- Sample size: {meta['sample_size']} / {meta['total_items_in_golden_set']} golden items (seed={meta['seed']})",
        f"- Duration: {meta['duration_seconds']}s",
        f"- Judge LLM: {meta['judge_llm']}",
        f"- Judge embeddings: {meta['judge_embeddings']}",
        f"- Estimated cost: ${meta['cost_estimate_usd']['total']:.4f} "
        f"(pipeline generation ${meta['cost_estimate_usd']['pipeline_generation']:.4f} + "
        f"RAGAS judge ${meta['cost_estimate_usd']['ragas_judge']:.4f})",
        "",
        "## Breakdown by question_type",
        "",
        "| Category | Mean score | N |",
        "|---|---|---|",
    ]
    for cat, stats in report["breakdown_by_category"].items():
        lines.append(f"| {cat} | {_fmt_score(stats['mean_score'])} | {stats['n']} ({stats['n_scored']} scored) |")

    lines += ["", "## Breakdown by answerability", "", "| Type | Mean score | N |", "|---|---|---|"]
    for kind, stats in report["breakdown_by_answerability"].items():
        lines.append(f"| {kind} | {_fmt_score(stats['mean_score'])} | {stats['n']} ({stats['n_scored']} scored) |")

    lines += ["", "## Suspect agent-path zero-candidate items", "",
               "Items where the agent path fired and retrieval returned 0 "
               "candidates — these scored as \"refused\"/low for a routing/"
               "metadata-filter reason, not necessarily a true grounding gap. "
               "See known issue in eval/README.md.", ""]
    if report["suspect_agent_zero_candidates"]:
        lines.append("| ID | Answerable | Category | Score |")
        lines.append("|---|---|---|---|")
        for it in report["suspect_agent_zero_candidates"]:
            lines.append(f"| {it['id']} | {it['answerable']} | {it['category']} | {_fmt_score(it['composite_score'])} |")
    else:
        lines.append("None.")

    lines += ["", f"## Flagged low-scoring items (< {report['run_metadata']['flag_threshold']})", ""]
    if report["low_scoring_items"]:
        lines.append("| ID | Category | Score | Question |")
        lines.append("|---|---|---|---|")
        for it in report["low_scoring_items"]:
            q = it["question"].replace("|", "\\|")
            lines.append(f"| {it['id']} | {it['category']} | {_fmt_score(it['composite_score'])} | {q} |")
    else:
        lines.append("None.")

    lines += ["", "## Per-question detail", ""]
    for it in report["items"]:
        flag = " ⚠️ LOW SCORE" if it["flagged_low_score"] else ""
        lines.append(f"### {it['id']} — {it['category']} ({'answerable' if it['answerable'] else 'unanswerable'}){flag}")
        lines.append("")
        lines.append(f"**Q:** {it['question']}")
        lines.append("")
        lines.append(f"**Answer:** {it['generated_answer']}")
        lines.append("")
        if it["metrics"]:
            metric_str = ", ".join(f"{k}={v:.3f}" for k, v in it["metrics"].items())
            lines.append(f"**RAGAS metrics:** {metric_str}")
        if it["refusal_signals"]:
            sig = it["refusal_signals"]
            lines.append(
                f"**Refusal signals:** is_ready_for_generation={sig['is_ready_for_generation']}, "
                f"confidence_action={sig['confidence_action']}, "
                f"refusal_phrase_detected={sig['refusal_phrase_detected']}"
            )
        lines.append(f"**Composite score:** {_fmt_score(it['composite_score'])}")
        agent_flag = " ⚠️ agent path + 0 candidates" if it["agent_used"] and it["candidates_retrieved"] == 0 else ""
        lines.append(
            f"**Retrieval:** routing={it['routing_decision']}, agent_used={it['agent_used']}, "
            f"candidates_retrieved={it['candidates_retrieved']}, citations={it['citations_count']}{agent_flag}"
        )
        lines.append("")

    return "\n".join(lines)


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAGAS evaluation harness for the RAG pipeline.")
    parser.add_argument("--sample-size", type=int, default=None, help="Run against a random subset of N items instead of the full golden set (fast iteration).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --sample-size sampling.")
    parser.add_argument("--threshold", type=float, default=0.75, help="Aggregate score CI gate — exit non-zero if the aggregate falls below this.")
    parser.add_argument(
        "--min-avg-citations", type=float, default=2.5,
        help="Average citations-per-answer CI gate (answerable items only) — exit non-zero if "
             "the average falls below this. Pass 0 (or negative) to disable. Added after "
             "scripts/simulate_traffic.py's incident simulation found that RAGAS faithfulness "
             "alone doesn't catch a starved-retrieval regression (a narrow-but-grounded answer "
             "still scores well); citation count does. Default derived from a real baseline of "
             "3.07 (see eval/README.md 'Citation-count baseline' for the full rationale, "
             "including a caveat about the run this baseline came from).",
    )
    parser.add_argument("--flag-threshold", type=float, default=0.5, help="Per-question score below which an item is flagged as a notable failure in the report.")
    parser.add_argument("--golden-set", type=Path, default=GOLDEN_SET_PATH, help="Path to golden_qa_set.json.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR, help="Directory to write the JSON/Markdown report to.")
    parser.add_argument(
        "--skip-ragas", action="store_true",
        help="Run the pipeline over the golden set without RAGAS judge scoring. "
             "Answerable items are reported (answer, retrieval diagnostics) but left "
             "unscored (composite_score=null) and excluded from the aggregate; "
             "unanswerable items still get the refusal-signal score (no RAGAS needed "
             "for those). Useful for validating pipeline health across the full "
             "golden set without paying for judge calls.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    all_items = load_golden_set(args.golden_set)
    args.total_items_in_golden_set = len(all_items)
    items = sample_items(all_items, args.sample_size, args.seed)
    logger.info(
        "Loaded %d golden items, evaluating %d (%s).",
        len(all_items), len(items),
        f"sample_size={args.sample_size}, seed={args.seed}" if args.sample_size else "full set",
    )

    run_start = time.time()

    pipeline = bootstrap_pipeline()

    logger.info("=" * 80)
    logger.info("Phase 1/2: running pipeline for %d items", len(items))
    logger.info("=" * 80)
    records, pipeline_input_tokens, pipeline_output_tokens = run_pipeline_phase(pipeline, items)

    if args.skip_ragas:
        logger.info("=" * 80)
        logger.info("Phase 2/2: RAGAS judge scoring — SKIPPED (--skip-ragas)")
        logger.info("=" * 80)
        ragas_scores, judge_tokens = {}, {"input_tokens": 0, "output_tokens": 0}
    else:
        logger.info("=" * 80)
        logger.info("Phase 2/2: RAGAS judge scoring")
        logger.info("=" * 80)
        ragas_scores, judge_tokens = run_ragas_phase(records)

    duration_seconds = time.time() - run_start

    record_observability_metrics(records, ragas_scores, env="ci")
    flush()  # short-lived script — force-flush buffered Langfuse spans/scores before exit

    report = build_report(
        records=records,
        ragas_scores=ragas_scores,
        args=args,
        duration_seconds=duration_seconds,
        pipeline_tokens=(pipeline_input_tokens, pipeline_output_tokens),
        judge_tokens=judge_tokens,
    )
    markdown = render_markdown(report)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = args.output_dir / f"eval_{timestamp}.json"
    md_path = args.output_dir / f"eval_{timestamp}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    logger.info("Wrote %s and %s", json_path, md_path)

    if args.skip_ragas:
        # Aggregate here only reflects the unanswerable/refusal-signal items
        # (see aggregate_score_note) — not a real quality score, so it isn't
        # meaningful to gate CI on. Exit 0 regardless; this mode is for
        # pipeline-health validation, not scoring.
        logger.info(
            "Aggregate score (partial, unanswerable items only): %.4f — %s",
            report["aggregate_score"], report["aggregate_score_note"],
        )
        return 0

    score_passed = report["aggregate_score"] >= args.threshold
    logger.info(
        "Aggregate score: %.4f (threshold %.2f) — %s",
        report["aggregate_score"], args.threshold, "PASSED" if score_passed else "FAILED",
    )
    if args.min_avg_citations > 0:
        logger.info(
            "Avg citations/answer: %.3f (min %.2f) — %s",
            report["avg_citations_answerable"], args.min_avg_citations,
            "PASSED" if report["citations_gate_passed"] else "FAILED",
        )

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
