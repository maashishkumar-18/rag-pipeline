# RAG Pipeline

A standalone, config-driven Retrieval-Augmented Generation pipeline: document
ingestion → hybrid search → cross-encoder reranking → confidence-gated
generation with citation round-tripping. Extracted from a larger
exam-prep application, generalized to work over any document corpus (PDF /
PPTX / DOCX).

This is not a toy RAG demo. Every stage below is a real, independently
configurable component — not a single 50-line "load docs, embed, ask GPT"
script.

## What's real here (and what to check for yourself)

- **Hybrid search is a genuine three-way merge**, not semantic-search-with-
  extra-steps: dense vector search (Pinecone) + real BM25 sparse retrieval
  (`rank_bm25`) + metadata-field boosting, combined via configurable
  weighted-merge or reciprocal-rank-fusion. See `src/retrieval/hybrid_search.py`
  and `config/retrieval/hybrid_search.yaml`.
- **Reranking is a real cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`
  via `sentence-transformers`), genuinely re-scoring (query, chunk) pairs —
  not a heuristic wearing a reranker's name. An LLM reranker and a
  parallel-hybrid mode (weighted combination of both) are also available.
  See `src/retrieval/reranker.py`.
- **Generation is confidence-gated.** A 7-feature weighted confidence score
  (reranker scores, score distribution, chunk diversity, context
  completeness, semantic coverage, metadata match, retrieval depth) decides
  whether to proceed, retry with expanded parameters, or flag that the
  corpus doesn't cover the question — see `src/retrieval/confidence.py` and
  `OrchestratorResult.is_ready_for_generation`.
- **Citations are traceable, not decorative.** Ingestion bakes a source
  prefix (`[Course | Chapter | Slide N]`) into each chunk; generation is
  instructed to reproduce it; post-processing extracts it back out of the
  generated text with regex and matches it to the originating chunk ID. Ask
  the pipeline something and you get back exactly which chunk backed each
  claim — see `src/generation/post_processor.py`.
- **Everything is YAML-configured with env var overrides** — chunk size,
  hybrid search weights, reranker backend/thresholds, confidence weights,
  context assembly rules, retry/timeout budgets. No hardcoded magic numbers
  buried in application code.

## What's honestly not here yet

There's a golden QA dataset (`eval/golden_qa_set.json` — 101 hand-built,
programmatically-verified pairs against the sample corpus), a RAGAS
scoring harness that runs it (`eval/run_ragas_eval.py`), and CI wired to
run it (`.github/workflows/eval.yml` — sampled on every PR, full run on
push to main and weekly) — see `eval/README.md` for how to run it, the
observed baseline (0.82) and threshold rationale (0.75), and known
limitations the harness surfaced along the way. Every
`if __name__ == "__main__":` block at the bottom of each module remains a
manual smoke test a human runs and eyeballs — useful for development, not
a substitute for the eval harness or CI.

Grounding validation is keyword-overlap only (not an entailment/NLI model),
so it catches unsupported answers but not fluent answers that lexically
overlap with the context while misstating the actual claim.

## Architecture

```
Ingestion:  Parse → Clean → Extract Metadata → Chunk → Enrich → Embed → Store
            (unstructured)  (ftfy)   (LLM)      (LLM)  (prefix) (Gemini/  (Pinecone)
                                                                  OpenAI)

Retrieval:  Query Rewriter → Intent Router → [Agent] → Query Expander →
            Metadata Filter → Hybrid Search → Reranker → Context Builder →
            Confidence Scorer

Generation: PromptBuilder → LLMClient → PostProcessor
            (YAML templates)  (DeepSeek/       (citations, grounding,
                                Gemini/OpenAI)   artifact cleanup)
```

`RetrievedChunk` (`src/common/types.py`) is the single shared contract
between retrieval and generation — the retrieval orchestrator's assembled
context chunks are passed directly into a `GenerationRequest`, no adapter
layer in between.

`src/common/llm_client.py` is genuinely shared infrastructure, not
generation-only: ingestion's chunker and metadata extractor, and
retrieval's LLM reranker and retrieval agent, all call `simple_generate()`
from it directly.

## Project structure

```
src/
  ingestion/     parser, cleaner, metadata_extractor, chunker, enricher,
                 embedder, vector_store
  retrieval/     orchestrator + 8 components (hybrid_search, reranker,
                 confidence, context_builder, query_rewriter,
                 query_expander, intent_router, retrieval_agent,
                 metadata_filter) + pipelines/ (per-use-case presets)
  generation/    orchestrator, prompt_builder, post_processor, config
  common/        types.py (shared RetrievedChunk), llm_client.py
                 (shared provider-agnostic LLM wrapper)
config/
  retrieval/     hybrid_search.yaml, reranker.yaml, confidence.yaml,
                 context_builder.yaml, metadata_filter.yaml, orchestrator.yaml
  generation/    generation.yaml, models.yaml, post_processing.yaml,
                 prompts/*.yaml
sample_data/     bundled sample lecture deck for the quickstart below
tools/           retrieval_debug_server.py — local FastAPI wrapper for
                 manually exercising retrieval quality
eval/            golden_qa_set.json (101 pairs) + run_ragas_eval.py (RAGAS
                 scoring harness) — see eval/README.md
observability/   tracing.py (Langfuse client + span helpers), metrics_store.py
                 (local SQLite store) — see "Observability" below
scripts/         simulate_traffic.py — traffic generator with an injectable
                 incident, for the observability dashboard
dashboard/       app.py — Streamlit observability dashboard
tests/           (empty — see Roadmap)
```

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env
# fill in DEEPSEEK_API_KEY, one of GEMINI_API_KEY/OPENAI_API_KEY, and
# PINECONE_API_KEY in .env
```

Run the bundled end-to-end example — it ingests `sample_data/sample_lecture.pptx`
into a fresh Pinecone namespace, then runs a set of real queries against it
through the full retrieval pipeline, printing routing decisions, confidence
scores, and timing breakdowns per query:

```bash
python -m src.retrieval.orchestrator
```

Each ingestion pipeline stage can also be run and inspected independently
— `python -m src.ingestion.parser`, `.cleaner`, `.metadata_extractor`,
`.chunker`, `.enricher`, `.embedder`, `.vector_store` — each has its own
`__main__` smoke test against the same sample file.

To exercise retrieval from a frontend or `curl`/Postman instead of the
CLI script:

```bash
uvicorn tools.retrieval_debug_server:app --reload --port 8500
```

## Configuration

Every component loads its config from `config/retrieval/*.yaml` or
`config/generation/*.yaml` by default, with per-value environment variable
overrides (prefix `RAGPIPE_`, e.g. `RAGPIPE_SEMANTIC_K=40`,
`RAGPIPE_RERANKER_BACKEND=llm`, `RAGPIPE_CONFIDENCE_HIGH=0.8`). See each
config class's `from_yaml()`/`_apply_env_overrides()` for the full mapping
— `src/retrieval/hybrid_search.py`, `reranker.py`, `confidence.py`,
`context_builder.py`, `metadata_filter.py`, `orchestrator.py`, and
`src/generation/config.py`, `prompt_builder.py`.

## Adapting to a different corpus

The ingestion metadata schema (`course_name`, `chapter_title`, `topic`) and
citation vocabulary (`Slide`/`Page`) are tuned for academic PDF/PPTX/DOCX
material. To point this at a different kind of corpus:

- Rewrite the extraction prompt in `src/ingestion/metadata_extractor.py`
  for your domain's fields.
- Extend `CitationLocationType` (`src/common/types.py`) and
  `src/ingestion/parser.py`'s `SUPPORTED_FORMATS` for other document types
  and location vocabularies (paragraph, URL anchor, line number, ...).
- Replace the example seed data in `src/retrieval/query_expander.py`'s
  `ConceptRegistry` with your own acronym/synonym set, or point it at a
  JSON file via `synonyms_path`.

## Observability

An extension of the pipeline above, not a separate project: tracing
(Langfuse) plus quality/cost/latency metrics over CI and simulated traffic,
built non-invasively — no component in `src/retrieval/` or `src/generation/`
was rewritten to add it.

- **`observability/tracing.py`** — a Langfuse client wrapper that's a
  complete no-op if `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` aren't set,
  so the pipeline behaves identically with or without tracing configured.
  The single integration point for retrieval is
  `RetrievalOrchestrator._safe_call_component` — one wrap there covers all
  9 retrieval stages (rewriter, router, agent, expander, filter_builder,
  hybrid_search, reranker, context_builder, confidence_scorer) without
  touching any of their individual modules. Generation is wrapped at its 3
  existing phase boundaries in `GenerationOrchestrator.generate()` (prompt
  build, LLM call, post-process).
- **`observability/metrics_store.py`** — a local SQLite store (WAL mode) for
  the aggregate queries a dashboard needs (latency percentiles, cost, quality
  trends) that would be slow/rate-limited to compute by repeatedly querying
  Langfuse's API. Every row also carries the Langfuse trace URL for drill-down.
- **`scripts/simulate_traffic.py`** — fires golden-set + adversarial/
  out-of-corpus queries at the real pipeline, deliberately degrading the
  retrieval config (BM25 disabled, `rerank_k` cut to 1, retry-on-low-
  confidence off — see `DEGRADED_PIPELINE`) for a configurable window to
  produce a reproducible "misconfigured reranker" incident, and samples a
  subset of answered requests through RAGAS faithfulness scoring.
- **`dashboard/app.py`** — a Streamlit app (not just a Langfuse screenshot)
  reading `metrics_store`: latency P50/P95 + per-stage breakdown, faithfulness
  over time with the incident window shaded, refusal/retrieval-hit rate,
  cost per-request and cumulative, and a recent-requests table linking out
  to each request's full trace waterfall in Langfuse.

### Setup

Optional — everything above degrades to a no-op without it. Create a free
project at [cloud.langfuse.com](https://cloud.langfuse.com), then set
`LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`/`LANGFUSE_HOST` in `.env`.

```bash
python -m scripts.simulate_traffic          # populate observability/metrics.db
streamlit run dashboard/app.py              # view at localhost:8501
```

`eval/run_ragas_eval.py` also pushes a trace per golden-set item (tagged
`env:ci`) and attaches the RAGAS/composite score to it after judging, so CI
runs are visible in both places too — see `record_observability_metrics()`.

### Environment bugs found by actually running this, not just writing it

Building this required, for the first time, actually running the full
ingestion → retrieval → generation pipeline end-to-end from a clean,
isolated `.venv` — previously it had only ever run against a developer
machine's existing global Python install, which silently had gaps papered
over by whatever had accumulated there from unrelated work. A clean venv
surfaced all of them:

- **`google-generativeai==0.7.2` → `0.8.6`**: the old pin's
  `google-ai-generativelanguage` transitive dependency caps `protobuf<5.0dev`,
  which conflicts with `streamlit==1.61.0`'s `protobuf>=5.26.1` floor.
- **`fastapi==0.111.0` unpinned to `>=0.115.0`**: the old pin's `starlette<0.38`
  ceiling conflicts with streamlit's `starlette>=0.46.0` floor.
- **`starlette` pinned `<1.0`**: streamlit 1.61.0's own declared range
  (`<2,>=0.46.0`) is wide enough for pip to resolve starlette 1.4.0, but
  streamlit's vendored GZip middleware isn't actually compatible with 1.x's
  internals — every request crashed with `GZipResponder.__init__() missing
  1 required keyword-only argument`. Found by running `streamlit run
  dashboard/app.py` and hitting a real 500, not by reading changelogs.
- **Missing `python-magic-bin` on Windows**: this one looks unrelated to
  observability at first glance, but `scripts/simulate_traffic.py` and
  `eval/run_ragas_eval.py` both call the *existing* ingestion pipeline's
  `bootstrap_pipeline()` to get real chunks to retrieve against — so
  exercising either one for the first time in an isolated venv is also the
  first time `unstructured`'s pptx parsing (`DocumentParser`) ran without
  whatever had made it work globally before. It segfaulted natively rather
  than raising a catchable Python exception; `requirements.txt` now installs
  the Windows-only wheel explicitly instead of relying on it being present
  by accident.
- **Dashboard incident-window shading spanned min-to-max across two
  non-contiguous simulation runs**, falsely shading untouched requests in
  between. `scripts/simulate_traffic.py` computes its incident window as a
  fraction of *that run's* request count, so two separate invocations landing
  in the same `metrics.db` produce two disjoint incident-tagged clusters, not
  one — `dashboard/app.py` now shades each contiguous run of incident-tagged
  requests separately instead of one band spanning the earliest to the latest.

### Two non-obvious things found by actually running this, not just writing it

- **Sibling spans don't share a trace unless something wraps them.** Verified
  empirically: two sequential `traced_span()` calls with no shared parent
  context get two different trace IDs, not one. That's why both
  `RetrievalOrchestrator.retrieve()` and `GenerationOrchestrator.generate()`
  each open their own outer span around their whole body — otherwise the 9
  retrieval-component spans (and the 3 generation-phase spans) would land as
  9 separate top-level traces instead of one waterfall.
- **No backdated timestamps.** The original plan called for compressing a
  multi-day traffic trend into one run via fabricated past timestamps.
  Checked the Langfuse SDK directly first: its span/trace creation APIs
  don't accept a custom start time, so traces are always stamped with real
  wall-clock time. Backdating only the local SQLite rows would make the
  dashboard's time axis disagree with what a reviewer sees after clicking a
  trace link — so every timestamp here is real, and the incident window is
  identified by request sequence and `retrieval_pipeline_name`, not calendar
  date.

### Honest result from the current sample run

`scripts/simulate_traffic.py` was run twice against the sample corpus (58
requests total, 14 tagged with the degraded config). The degradation
mechanism itself is verified working — incident-tagged requests are
correctly routed through `DEGRADED_PIPELINE` and tagged as such in both
Langfuse and `metrics.db` — but at this sample size and faithfulness-sample
rate, the 3 faithfulness samples that happened to fall inside the incident
window all still scored 1.0; the one low score observed (0.67) fell
*outside* the incident window. That's a real small-sample-size result, not
a hidden success — a larger `--num-requests` and/or a denser
`--faithfulness-sample-rate` would be needed for a demo run that reliably
shows the drop. The refusal-rate and latency trends (visible in the
dashboard regardless of sample size) are unaffected by this.

## Roadmap

The retrieval architecture is intentionally the mature part of this
project; evaluation and CI are the deliberately-unbuilt next layer:

1. ~~**Golden eval set**~~ — done: `eval/golden_qa_set.json`, 101
   pairs (84 answerable + 17 deliberately-unanswerable, including hard
   negatives) against the sample corpus, every answerable item's
   `source_quote` programmatically verified as a genuine substring of the
   deck's real extracted text on its claimed slide. See `eval/README.md`.
2. ~~**RAGAS-style automated scoring**~~ — done: `eval/run_ragas_eval.py`
   scores faithfulness, context precision/recall, and answer correctness
   for answerable items, and a structured refusal-signal check for
   deliberately-unanswerable ones. See `eval/README.md`.
3. ~~**CI**~~ — done: `.github/workflows/eval.yml` runs
   `eval/run_ragas_eval.py` (sampled on PRs, full run on push to main and
   weekly on schedule) and fails the build on regression via the script's
   own threshold-gated exit code.
4. **Replace keyword-overlap grounding** with an actual entailment/NLI
   check in `src/generation/post_processor.py`'s `GroundingValidator`.
5. **Fix the two known limitations documented in `eval/README.md`**:
   reranker cold-start latency (warm it at service startup) and
   multi-topic chunk dilution hurting single-term queries (finer-grained
   chunking for multi-item topical spans) — both root-caused, neither
   fixed yet, see that doc for why.
6. ~~**Cost/latency dashboard**~~ — done: Langfuse tracing +
   `observability/metrics_store.py` + `dashboard/app.py`, see
   "Observability" above.
7. **CI regression gating on quality metrics** — extend
   `.github/workflows/eval.yml` to fail the build on a faithfulness/cost/
   latency regression against a committed baseline, the same way
   `eval/run_ragas_eval.py`'s aggregate-score threshold already gates on
   RAGAS score. Not built yet; the observability layer above only traces
   and dashboards CI/simulated runs, it doesn't gate on them.
