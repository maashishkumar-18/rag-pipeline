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
6. **Cost/latency dashboard** — per-request token/cost tracking already
   exists in `UsageStats`/`EmbeddingGenerator.estimate_cost()`; nothing
   currently aggregates it.
