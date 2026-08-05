"""
Langfuse Tracing — Shared Client + Span Helpers

Fully optional: every function here degrades to a no-op if
LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY aren't set (checked once, cached),
so the pipeline behaves identically with or without tracing configured —
no call site needs an `if tracing_enabled` guard.

Defensive by design, not just at client-init: a Langfuse API hiccup mid-run
must never fail the actual pipeline call it's trying to observe. Every
span-creation/update/close call is wrapped so tracing failures are logged
and swallowed. The one thing that must NOT be swallowed is an exception
raised by the caller's own wrapped code (a real pipeline component
failure) — traced_span() re-raises those after best-effort marking the
span as errored. See traced_span()'s docstring for why this needs manual
context-manager enter/exit instead of one big try/except.

Usage — component-level span (see src/retrieval/orchestrator.py
_safe_call_component, src/generation/orchestrator.py generate()):

    with traced_span("hybrid_search", input={"query": q}) as span:
        result = hybrid_search.search(...)
        span.update(output=safe_dict(result))

Usage — one trace spanning retrieval + generation for a single request
(see eval/run_ragas_eval.py, scripts/simulate_traffic.py):

    with traced_pipeline_call(request_id=rid, query=q, env="ci") as trace:
        retrieval_result = orchestrator.retrieve(...)
        generation_response = gen_orchestrator.generate(...)
        trace.update(output={"answer": generation_response.answer})

traced_span() calls made by retrieve()/generate() internals automatically
nest under the currently-open trace via Langfuse's OTEL context propagation
— no explicit parent-passing needed.
"""

import logging
import os
import sys
import threading
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MAX_TEXT_CHARS = 2000  # truncate long text fields (context, prompts, answers) in span payloads
_MAX_LIST_ITEMS = 20

_client_lock = threading.Lock()
_client = None
_client_init_attempted = False


# ============================================================================
# Client singleton
# ============================================================================

def is_enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY")) and bool(os.getenv("LANGFUSE_SECRET_KEY"))


def get_langfuse():
    """
    Lazily create the process-wide Langfuse client.

    Returns None if LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY aren't set, or
    if client construction itself fails — every caller in this module
    already treats None as "tracing disabled" rather than special-casing it.
    """
    global _client, _client_init_attempted
    if _client_init_attempted:
        return _client

    with _client_lock:
        if _client_init_attempted:
            return _client
        _client_init_attempted = True

        if not is_enabled():
            logger.info("Langfuse tracing disabled (LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY not set).")
            return None

        try:
            from langfuse import Langfuse

            _client = Langfuse(
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            )
        except Exception:
            logger.exception("Failed to initialize Langfuse client; tracing disabled for this process.")
            _client = None

    return _client


# ============================================================================
# Payload serialization
# ============================================================================

def _truncate(value: Any) -> Any:
    if isinstance(value, str) and len(value) > _MAX_TEXT_CHARS:
        return value[:_MAX_TEXT_CHARS] + f"... [truncated, {len(value)} chars total]"
    return value


def safe_dict(obj: Any, max_list_items: int = _MAX_LIST_ITEMS) -> Any:
    """
    Best-effort JSON-safe conversion for span input/output payloads —
    dataclasses (the pipeline's OrchestratorResult, SearchResult,
    RerankerResult, ConfidenceResult, GenerationResponse, etc. are all
    dataclasses), enums, and nested combinations, with long text fields
    truncated so spans stay small.

    Never raises: a span payload isn't worth crashing the pipeline over, so
    anything unrecognized falls back to str(obj).
    """
    try:
        if obj is None or isinstance(obj, (bool, int, float)):
            return obj
        if isinstance(obj, str):
            return _truncate(obj)
        if isinstance(obj, Enum):
            return obj.value
        if is_dataclass(obj) and not isinstance(obj, type):
            return {k: safe_dict(v, max_list_items) for k, v in asdict(obj).items()}
        if isinstance(obj, dict):
            return {str(k): safe_dict(v, max_list_items) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            items = list(obj)
            result = [safe_dict(v, max_list_items) for v in items[:max_list_items]]
            if len(items) > max_list_items:
                result.append(f"... [{len(items) - max_list_items} more]")
            return result
        return str(obj)
    except Exception:
        return f"<unserializable {type(obj).__name__}>"


# ============================================================================
# No-op fallback
# ============================================================================

class _NullSpan:
    """
    Stand-in returned whenever tracing is disabled or a span failed to
    start. Accepts any method call (.update(), .score(), ...) as a no-op
    so call sites never need to check for None.
    """

    def __getattr__(self, _name: str):
        return lambda *args, **kwargs: None


_NULL_SPAN = _NullSpan()


# ============================================================================
# Span context managers
# ============================================================================

@contextmanager
def traced_span(
    name: str,
    *,
    as_type: str = "span",
    input: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
    **extra: Any,
):
    """
    Generic child-span context manager for instrumenting pipeline internals.

    Manually drives the underlying Langfuse context manager's __enter__/
    __exit__ (instead of one wrapping try/except) so two very different
    failure modes are handled correctly:
      1. Langfuse itself fails (network hiccup, span-start error) — caught,
         logged, degrades to a no-op span. The caller's code hasn't run yet.
      2. The caller's wrapped code raises (a real pipeline component
         failure) — the span is best-effort marked as errored, then the
         exception is always re-raised. A single broad try/except around
         the whole `yield` would silently swallow this case instead
         (the exception surfaces at the yield statement, inside the try),
         defeating the orchestrator's own error handling/fallback logic.
    """
    client = get_langfuse()
    if client is None:
        yield _NULL_SPAN
        return

    cm = None
    span = None
    try:
        cm = client.start_as_current_observation(
            name=name,
            as_type=as_type,
            input=safe_dict(input) if input is not None else None,
            metadata=safe_dict(metadata) if metadata else None,
            **extra,
        )
        span = cm.__enter__()
    except Exception:
        logger.exception("Langfuse span '%s' failed to start; continuing without tracing.", name)
        yield _NULL_SPAN
        return

    try:
        yield span
    except Exception:
        try:
            span.update(level="ERROR", status_message=str(sys.exc_info()[1]))
        except Exception:
            pass
        try:
            cm.__exit__(*sys.exc_info())
        except Exception:
            pass
        raise
    else:
        try:
            cm.__exit__(None, None, None)
        except Exception:
            logger.exception("Langfuse span '%s' failed to close cleanly.", name)


@contextmanager
def traced_pipeline_call(
    *,
    request_id: str,
    query: str,
    env: str,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Opens one Langfuse trace spanning both the retrieval and generation
    orchestrator calls for a single pipeline request. Component-level spans
    from RetrievalOrchestrator._safe_call_component and
    GenerationOrchestrator.generate()'s phases nest under this trace
    automatically via Langfuse's OTEL context propagation.

    env is tagged both as a trace tag ("env:<env>") and via Langfuse's
    dedicated environment field, so CI runs (env="ci"), simulated traffic
    (env="simulated_production"), etc. can be filtered either way.
    """
    client = get_langfuse()
    if client is None:
        yield _NULL_SPAN
        return

    try:
        from langfuse import propagate_attributes
    except Exception:
        logger.exception("Failed to import langfuse.propagate_attributes; continuing without tracing.")
        yield _NULL_SPAN
        return

    all_tags = list(tags or []) + [f"env:{env}"]
    trace_metadata = {"request_id": request_id, "env": env, **(metadata or {})}

    prop_cm = None
    try:
        prop_cm = propagate_attributes(
            trace_name="pipeline_call",
            tags=all_tags,
            metadata=safe_dict(trace_metadata),
            environment=env,
        )
        prop_cm.__enter__()
    except Exception:
        logger.exception("Langfuse propagate_attributes failed to start; continuing without tracing.")
        yield _NULL_SPAN
        return

    try:
        with traced_span(
            "pipeline_call",
            input={"query": query, "request_id": request_id},
        ) as span:
            yield span
    finally:
        try:
            prop_cm.__exit__(None, None, None)
        except Exception:
            logger.exception("Langfuse propagate_attributes failed to close cleanly.")


# ============================================================================
# Scoring / trace lookup
# ============================================================================

def score_current_trace(name: str, value: float, comment: Optional[str] = None) -> None:
    """Attach a score (e.g. sampled RAGAS faithfulness) to the currently-open trace."""
    client = get_langfuse()
    if client is None:
        return
    try:
        client.score_current_trace(name=name, value=value, comment=comment)
    except Exception:
        logger.exception("Failed to attach Langfuse score '%s'.", name)


def score_trace(trace_id: str, name: str, value: float, comment: Optional[str] = None) -> None:
    """
    Attach a score to a trace by ID, after the fact — for scores computed
    later than the trace itself (e.g. eval/run_ragas_eval.py's RAGAS judge
    runs as a separate batched phase, once all per-item traces have already
    closed; score_current_trace() wouldn't have an active span to attach
    to by then).
    """
    client = get_langfuse()
    if client is None or not trace_id:
        return
    try:
        client.create_score(trace_id=trace_id, name=name, value=value, comment=comment)
    except Exception:
        logger.exception("Failed to attach Langfuse score '%s' to trace %s.", name, trace_id)


def current_trace_id() -> Optional[str]:
    client = get_langfuse()
    if client is None:
        return None
    try:
        return client.get_current_trace_id()
    except Exception:
        return None


def current_trace_url() -> Optional[str]:
    """Deep link into the Langfuse UI for the currently-open trace — used by
    metrics_store rows / the dashboard's recent-requests table."""
    client = get_langfuse()
    if client is None:
        return None
    try:
        return client.get_trace_url()
    except Exception:
        return None


def flush() -> None:
    """Force-flush buffered spans. Call at the end of short-lived scripts
    (eval harness, simulate_traffic.py) so traces aren't lost on exit —
    long-running servers don't need this, Langfuse flushes on its own
    schedule."""
    client = get_langfuse()
    if client is None:
        return
    try:
        client.flush()
    except Exception:
        logger.exception("Langfuse flush failed.")
