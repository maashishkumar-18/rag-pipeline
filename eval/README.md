# Golden Evaluation QA Set

`golden_qa_set.json` — 101 hand-built QA pairs against the bundled sample
corpus (`sample_data/sample_lecture.pptx`, a Supply Chain Management
"Inventory Management" lecture deck, slides 2-18).

## What "verified" means here

Every answerable item's `source_quote` is a **verbatim substring** of the
real, parsed text of the deck — not a paraphrase, not general domain
knowledge about inventory management, not an LLM's guess at what the slide
probably says. Nothing in any `ground_truth` exists that isn't traceable to
its `source_quote`. This was checked programmatically, not just asserted:

```bash
python -m src.ingestion.parser   # or see the verification script below
```

Verification script (reproduce the check yourself):

```python
import sys, json, re
sys.path.insert(0, ".")
from src.ingestion.parser import DocumentParser
from src.ingestion.cleaner import TextCleaner

parsed = DocumentParser().parse("sample_data/sample_lecture.pptx")
cleaned = TextCleaner().clean(parsed)

def norm(s):
    return re.sub(r"\s+", " ", s).strip()

slide_text = {}
for el in cleaned.elements:
    slide_text.setdefault(el.page_number, []).append(el.content)
slide_text = {k: norm(" ".join(v)) for k, v in slide_text.items()}

with open("eval/golden_qa_set.json", encoding="utf-8") as f:
    data = json.load(f)

for item in data["items"]:
    if not item["answerable"]:
        continue
    quotes = item["source_quote"] if isinstance(item["source_quote"], list) else [item["source_quote"]]
    for q in quotes:
        assert any(norm(q) in slide_text.get(s, "") for s in item["expected_slides"]), item["id"]
print("All quotes verified.")
```

Two things were checked, both passing on the current file:
1. Every `source_quote` (or, for multi-span answers, every element of the
   `source_quote` array) is a genuine substring of the deck's real
   extracted text.
2. Every quote actually appears on the specific slide(s) listed in
   `expected_slides` — not just somewhere in the deck.

Unanswerable items (`answerable: false`) were checked by hand against the
full extracted text of every slide (2-18) to confirm the topic genuinely
never appears in the corpus. Several are deliberate **hard negatives** —
questions that sound like a natural follow-up to a real, covered fact but
whose specific answer is not actually in the deck (e.g. q089 asks for a
safety-stock formula when the deck only ever discusses safety stock
conceptually; q099 asks for a cost comparison the deck states both halves
of but never actually computes). These are the cases most likely to trip
up a RAG pipeline into hallucinating a plausible-sounding but ungrounded
answer, which is exactly what the confidence-gating design is supposed to
catch — see the main README's "What's real here" section.

## Schema

```jsonc
{
  "id": "q001",                 // unique
  "question": "...",
  "ground_truth": "..." | null, // null for unanswerable items
  "category": "definition | explanation | calculation | comparison | multi_hop | reference | unanswerable",
  "answerable": true | false,   // does the corpus actually contain the answer
  "expected_slides": [3],       // 1-indexed slide numbers; [] for unanswerable
  "source_quote": "..." | ["...", "..."] | null,  // verbatim deck text; array when the answer spans multiple quotes; null for unanswerable
  "notes": "..." | null         // context, e.g. why a hard negative is hard
}
```

## Breakdown

| Category | Count | Notes |
|---|---|---|
| definition | 26 | "What is X?" |
| explanation | 16 | "Why/how does X work?" |
| calculation | 28 | Worked-example numbers (EOQ, ROP, fixed-time-period) and formula components — every number is one actually stated in the deck, never invented |
| comparison | 6 | X vs Y (fixed-order vs fixed-time-period, holding vs ordering cost, etc.) |
| multi_hop | 6 | Requires combining facts from 2+ slides |
| reference | 2 | Deck metadata (course/chapter/textbook, slide 2) |
| unanswerable | 17 | Topically-adjacent concepts genuinely absent from this corpus, plus 1 fully out-of-domain control question |

84 answerable + 17 unanswerable = 101 total.

## What this is (and isn't) yet

This is a **dataset** — `run_ragas_eval.py` (same directory) is the
scoring harness that runs it. The `expected_slides` field is what RAGAS's
`context_precision`/`context_recall` key off indirectly: they check
whether the retrieved context chunks actually support `ground_truth`,
which in turn was written against the slide(s) listed here.

If you swap in your own corpus, this file becomes invalid (it's written
against this specific deck's specific facts) — treat it as a worked
example of how to build a golden set, not a reusable fixture.

## Running the evaluation harness

```bash
# Full 101-item run (what CI gates on)
python -m eval.run_ragas_eval

# Fast iteration: random 15-item subset, lower threshold
python -m eval.run_ragas_eval --sample-size 15 --threshold 0.7
```

Requires `DEEPSEEK_API_KEY`, one embedding provider key (`GEMINI_API_KEY`
or `OPENAI_API_KEY`), and `PINECONE_API_KEY` in `.env` — every run
re-ingests `sample_data/sample_lecture.pptx` into a fresh Pinecone
namespace (same as `python -m src.retrieval.orchestrator`'s integration
test), then runs each golden-set question through the real
`RetrievalOrchestrator` → `GenerationOrchestrator` pipeline. Expect
~100+ live LLM calls on a full run (pipeline generation + RAGAS judge) —
use `--sample-size` while iterating.

### Flags

| Flag | Default | Meaning |
|---|---|---|
| `--sample-size N` | full set (101) | Evaluate a random subset instead of everything. |
| `--seed N` | 42 | RNG seed for `--sample-size` sampling (reproducible subsets). |
| `--threshold F` | 0.75 | Aggregate score CI gate — process exits non-zero if the run's aggregate falls below this. |
| `--flag-threshold F` | 0.5 | Per-question score below which an item is flagged as a notable failure in the report. |
| `--golden-set PATH` | `eval/golden_qa_set.json` | Override the input dataset. |
| `--output-dir PATH` | `eval/results/` | Where the timestamped JSON/Markdown reports are written. |

### Judge configuration

RAGAS needs a judge LLM and, for `answer_correctness`'s semantic-similarity
term, an embeddings model — both intentionally **separate from the
pipeline's own generation call**:

- **Judge LLM**: `langchain_openai.ChatOpenAI` pointed at DeepSeek's
  OpenAI-compatible endpoint (`base_url="https://api.deepseek.com"`,
  `DEEPSEEK_API_KEY`). This is a different client instance than the
  pipeline's `LLMClient`/`DeepSeekAdapter` — even though both currently
  resolve to `deepseek-chat` per `config/generation/models.yaml`, the
  judge is never the same code path grading its own output, and stays
  independent if the pipeline's default model changes later.
- **Judge embeddings**: a local `sentence-transformers/all-MiniLM-L6-v2`
  model via `langchain_community.embeddings.HuggingFaceEmbeddings`. The
  pipeline's own `EmbeddingGenerator` (`src/ingestion/embedder.py`) isn't
  a drop-in LangChain `Embeddings` implementation (it exposes
  `embed_query`/`embed_queries`, not the `embed_query`/`embed_documents`
  pair RAGAS expects) and is backed by a paid provider — reusing it would
  conflate ingestion cost/provider with judge cost on every eval run. A
  small local model avoids that and costs nothing per run;
  `sentence-transformers` is already a pipeline dependency (the
  cross-encoder reranker uses it).

### Scoring

- **Answerable items** (84): `faithfulness`, `context_precision`,
  `context_recall`, and `answer_correctness` are computed via RAGAS
  against `ground_truth`, then averaged into one per-question score.
- **Unanswerable items** (17): there's no `ground_truth` to score RAGAS
  metrics against by design, so these are scored 1.0 (pipeline correctly
  hedged) or 0.0 (pipeline confidently answered anyway) using signals the
  pipeline already produces — `OrchestratorResult.is_ready_for_generation`,
  `confidence.action` (`src/retrieval/confidence.py`), and whether the
  answer contains the exact refusal phrase
  `config/generation/prompts/context_aware.yaml`'s system prompt
  instructs the model to emit when context is insufficient
  ("...does not cover this topic in sufficient detail").
- **Aggregate score**: the mean of all per-question composite scores —
  this is the headline number CI gates on via `--threshold`.

### Output

Each run writes two timestamped files to `eval/results/`:

- `eval_<timestamp>.json` — machine-readable: aggregate score, pass/fail,
  breakdown by `category` (definition/explanation/calculation/comparison/
  multi_hop/reference/unanswerable), breakdown by answerable vs.
  unanswerable, a `low_scoring_items` list (composite score below
  `--flag-threshold`), a rough DeepSeek cost estimate, and full per-question
  detail (question, generated answer, individual metric scores, refusal
  signals for unanswerable items).
- `eval_<timestamp>.md` — the same data as a human-readable summary.

The process exit code is non-zero when the aggregate score is below
`--threshold` — that's the hook for CI gating (see main README's
"Roadmap" item 3).

## Baseline run and threshold rationale

Full 101-item run, real RAGAS judge scoring (`eval/results/eval_20260731T210201Z.json`):
**aggregate 0.8203**, PASSED against `--threshold 0.75`. Breakdown:

| Category | Mean | N |
|---|---|---|
| unanswerable | 1.0000 | 17 |
| multi_hop | 0.9051 | 6 |
| reference | 0.8558 | 2 |
| comparison | 0.8238 | 6 |
| explanation | 0.7819 | 16 |
| calculation | 0.7934 | 28 |
| definition | 0.7323 | 26 |

**Why the default threshold is 0.75, not tuned closer to 0.82**: this run and
an earlier identical 12-query retrieval test both showed real run-to-run
non-determinism (the same query routing through `direct_retrieval` vs.
`agent_reasoning` on different runs, with corresponding confidence-score
swings). A ~9% margin below the observed baseline is there to absorb that
ordinary stochasticity without masking a genuine regression — moving the
gate closer to 0.82 would risk the CI check flagging normal variance as a
failure, which trains you to distrust (and eventually ignore) the gate.

## Known limitations found via this eval harness

Building and running this harness surfaced four distinct issues — one fixed
and verified at scale, two intentionally documented rather than patched, and
one bug in the harness's own scoring logic (caught and corrected before it
could distort the baseline). Recording all four here, not just the fixed
one, because a harness that only reports "everything passed" is less
trustworthy than one that shows its own limits.

### Fixed: agent-path metadata-filter zero-candidate bug

**Root cause**: `RetrievalAgent`'s LLM call proposes metadata filters like
`{"subject": "Supply Chain Management", "chapter": "Inventory Management"}`.
`chapter` resolves correctly (`chapter_title` matches), but `subject`
resolves to `subject_area`, which `MetadataExtractor` populates with a
*broad academic discipline* ("Business"), not the course name the agent
assumes — confirmed by direct Pinecone metadata inspection. Applied as a
hard `$eq` AND filter, the mismatch zeroes out retrieval entirely. Separately,
the agent also routinely proposed filter keys (`module`, `concept`,
`current_topic`) with no schema mapping at all; `MetadataFilterBuilder`
logged a warning ("Field 'X' not defined in schema") but applied them as
literal filters anyway — a guaranteed zero-match against a nonexistent
Pinecone key, silently, since the warning was never surfaced by the caller.

**Fix**: `MetadataFilterBuilder._build_from_dict` now skips (rather than
applies) fields absent from the schema. `RetrievalOrchestrator` no longer
merges agent-proposed `subject`/`chapter`/`course` into the hard filter at
all — matching the design already used for conversation-state filters (see
`MetadataFilterBuilder._build_from_conversation`'s docstring), which
avoids exactly this vocabulary-mismatch class of bug for a different filter
source. Narrower agent-proposed fields (e.g. `topic`) are unaffected and
still applied.

**Before/after, same 12-query retrieval integration test**
(`src/retrieval/orchestrator.py`'s own `__main__`): before the fix, 2
queries routed through the agent path and both returned 0 candidates
(100% failure on that path). After the fix, a rerun of the same 12 queries
routed 3 through the agent path (routing itself is non-deterministic
between runs) and all 3 returned a full candidate set. Verified again at
scale across the full golden set: 0 zero-candidate agent-path failures in
both the 101-item pipeline-only run and the 101-item RAGAS-scored run
(`suspect_agent_zero_candidates: []` in both reports).

### Documented, not fixed: reranker cold-start latency

The cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) loads
its model lazily on first use. The very first retrieval call in a freshly
started process pays a one-time ~10s model-load cost on top of normal
inference time, which can exceed the retrieval layer's 15s timeout budget
— observed directly (q001, the first item in a 101-item run, timed out at
20.6s with 0 candidates as a result). Not a logic bug; a known
ML-serving pattern (cold start). Left undocumented-as-fixed rather than
patched: the fix is straightforward (warm the reranker at service startup,
e.g. one dummy `rerank()` call before accepting traffic) but is an infra
change, not a retrieval-logic change, so it's out of scope for this harness
to apply on its own.

### Documented, not fixed: single-term queries against multi-topic chunks (q014)

**Root cause**: `SemanticChunker` grouped five distinct inventory sub-types
(Cycle Stock, Safety Stock, Anticipation, Pipeline, MRO) into one chunk
under the broad topic "Types of Inventory" (slides 6-7). For a query asking
specifically about one of those five ("What does MRO stand for, and what
does it include?"), the cross-encoder reranker scores the *whole chunk's*
relevance against the query — and a chunk whose semantic content is spread
across five sub-topics scores worse than chunks singularly focused on one
topic (e.g. the deck's EOQ chunks), even though it contains the correct
answer. The chunk doesn't rank in the pipeline's default top 5 at all.

**Fix directions considered**:
1. Finer-grained chunking for multi-item topical spans (ingestion-level
   change, corpus-wide impact — not attempted; see rationale below).
2. Increase `rerank_k` / `ContextBuilderConfig.max_chunks` (tested, see
   below).

**Tested (direction 2)**: swept `k` from 5 to 13 (both the reranker's
output size and `ContextBuilder`'s own — separately configured — chunk
cap, which is what actually enforces the final context size regardless of
`PipelineConfig`). Result:

| k | MRO chunk present | Rank | Tokens used |
|---|---|---|---|
| 5 (default) | No | — | 1044 |
| 6, 7 | No | — | 1279, 1416 |
| 8 | Yes | last (8/8) | 1618 (+55%) |
| 10, 13 | Yes | last (10/10) | 1937 (+85%) |

Raising `k` does eventually surface the chunk, but only at nearly double
the token cost per query, and even with the entire candidate pool
available (k=13) it never rises above dead last — this isn't a marginal
near-miss at the k=5 cutoff, the chunk's cross-encoder relevance score is
genuinely low relative to the rest of the pool. That result points at
direction 1 (chunking granularity) as the real fix, not direction 2 — but
re-chunking is a corpus-wide ingestion change that would need its own
validation pass against the rest of the golden set to confirm it doesn't
regress other questions, which is real scope beyond documenting this one
finding. Left as a known limitation.

### Self-caught: eval harness's own refusal-phrase detection bug

The unanswerable-item scoring checks generated answers for the exact
refusal phrase the system prompt instructs (`config/generation/prompts/
context_aware.yaml`: "...does not cover this topic in sufficient detail").
One item (q086, "What is JIT inventory management?") scored 0.0 despite the
model correctly refusing — it had naturally substituted the specific topic
name for "this topic" ("...does not cover Just-In-Time (JIT) inventory
management in sufficient detail"), which broke an exact substring match.
Caught by inspecting every flagged low-scoring item rather than trusting
the aggregate number, confirmed via the other 16/17 unanswerable items that
it was an isolated case (not a systemic detection failure), fixed to a
regex tolerant of the substitution, and **the existing report was
regenerated from the already-captured pipeline answers and RAGAS metrics —
no re-running of paid API calls, since only the scoring logic was wrong,
not the underlying data.** Corrected aggregate: 0.8104 → 0.8203.
