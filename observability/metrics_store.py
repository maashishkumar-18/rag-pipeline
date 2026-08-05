"""
Local SQLite Metrics Store — for the observability dashboard

Complements Langfuse (which owns the full per-request trace waterfall —
see observability/tracing.py) with a lightweight local store purpose-built
for the aggregate queries the Streamlit dashboard needs: latency
percentiles, cost over time, faithfulness trend, refusal rate, retrieval
hit rate. Repeatedly querying Langfuse's API for these would be slower and
rate-limit-bound; this is a plain SQLite file the dashboard can query
directly. Each row also carries the Langfuse trace URL so the dashboard can
link out to the full waterfall for any individual request.

Domain-agnostic by design: this module doesn't decide what counts as
"refused" or a "retrieval hit" — callers (eval/run_ragas_eval.py,
scripts/simulate_traffic.py) compute those from OrchestratorResult /
GenerationResponse using logic that already lives with the rest of the
pipeline's refusal-detection code, and just pass the booleans in.
"""

import json
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_DB_PATH = Path(__file__).parent / "metrics.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pipeline_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,           -- ISO 8601 UTC
    env TEXT NOT NULL,                 -- "ci" | "simulated_production" | ...
    query TEXT,

    total_time_ms REAL,
    retrieval_time_ms REAL,
    generation_time_ms REAL,
    stage_timings_json TEXT,           -- JSON dict, per-component ms (OrchestratorResult.timing_breakdown)

    confidence_score REAL,
    confidence_level TEXT,
    retrieval_hit INTEGER,             -- 0/1 -- confidence_level != LOW
    candidates_retrieved INTEGER,

    is_grounded INTEGER,               -- 0/1
    citations_count INTEGER,
    refused INTEGER,                   -- 0/1 -- caller-computed refusal signal

    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,

    prompt_version TEXT,
    model_name TEXT,
    retrieval_pipeline_name TEXT,

    faithfulness_score REAL,           -- nullable -- only set for sampled requests

    langfuse_trace_id TEXT,
    langfuse_trace_url TEXT
);

CREATE INDEX IF NOT EXISTS idx_pipeline_calls_timestamp ON pipeline_calls(timestamp);
CREATE INDEX IF NOT EXISTS idx_pipeline_calls_env ON pipeline_calls(env);
"""

_NUMERIC_COLUMNS = {
    "total_time_ms", "retrieval_time_ms", "generation_time_ms",
    "confidence_score", "candidates_retrieved", "citations_count",
    "input_tokens", "output_tokens", "cost_usd", "faithfulness_score",
}


@dataclass
class PipelineCallMetrics:
    """
    One row's worth of data — built by the caller from OrchestratorResult +
    GenerationResponse (+ observability.tracing's current_trace_id()/
    current_trace_url(), captured *inside* the traced_pipeline_call `with`
    block — no active span afterward) after a single retrieve()+generate()
    call.
    """
    request_id: str
    env: str
    query: str = ""
    timestamp: Optional[str] = None  # defaults to now (UTC) if unset

    total_time_ms: float = 0.0
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    stage_timings: Dict[str, float] = field(default_factory=dict)

    confidence_score: float = 0.0
    confidence_level: str = ""
    retrieval_hit: bool = False
    candidates_retrieved: int = 0

    is_grounded: bool = False
    citations_count: int = 0
    refused: bool = False

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    prompt_version: Optional[str] = None
    model_name: Optional[str] = None
    retrieval_pipeline_name: Optional[str] = None

    faithfulness_score: Optional[float] = None

    langfuse_trace_id: Optional[str] = None
    langfuse_trace_url: Optional[str] = None


class MetricsStore:
    """
    Thin SQLite wrapper. WAL mode is on so the Streamlit dashboard can read
    concurrently while a simulator/eval run is actively writing, without
    lock contention — the failure mode without it is a dashboard refresh
    raising "database is locked" mid-simulation. No persistent connection
    is held open; each call opens/closes its own (SQLite connections are
    cheap, and this avoids any cross-thread connection-sharing issues
    between a simulator process and a Streamlit dashboard process).
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or DEFAULT_DB_PATH)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(_SCHEMA)

    def record(self, metrics: PipelineCallMetrics) -> None:
        row = asdict(metrics)
        row["timestamp"] = metrics.timestamp or datetime.now(timezone.utc).isoformat()
        row["stage_timings_json"] = json.dumps(row.pop("stage_timings"))
        row["retrieval_hit"] = int(bool(row["retrieval_hit"]))
        row["is_grounded"] = int(bool(row["is_grounded"]))
        row["refused"] = int(bool(row["refused"]))

        columns = list(row.keys())
        placeholders = ", ".join(["?"] * len(columns))
        sql = f"INSERT INTO pipeline_calls ({', '.join(columns)}) VALUES ({placeholders})"

        with self._lock, self._connect() as conn:
            conn.execute(sql, [row[c] for c in columns])

    def query_recent(self, limit: int = 100, env: Optional[str] = None) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM pipeline_calls"
        params: List[Any] = []
        if env:
            sql += " WHERE env = ?"
            params.append(env)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def query_all(self, env: Optional[str] = None) -> List[Dict[str, Any]]:
        """All rows, oldest first — for dashboard time-series charts."""
        sql = "SELECT * FROM pipeline_calls"
        params: List[Any] = []
        if env:
            sql += " WHERE env = ?"
            params.append(env)
        sql += " ORDER BY timestamp ASC"

        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def percentile(self, column: str, p: float, env: Optional[str] = None) -> Optional[float]:
        """
        p in [0, 100]. Computed in Python (SQLite has no built-in
        percentile function) — fine at this scale (thousands, not millions,
        of rows for a portfolio demo).
        """
        if column not in _NUMERIC_COLUMNS:
            raise ValueError(f"Unknown/non-numeric column: {column}")

        sql = f"SELECT {column} FROM pipeline_calls WHERE {column} IS NOT NULL"
        params: List[Any] = []
        if env:
            sql += " AND env = ?"
            params.append(env)

        with self._lock, self._connect() as conn:
            values = [r[0] for r in conn.execute(sql, params).fetchall()]

        if not values:
            return None
        values.sort()
        k = (len(values) - 1) * (p / 100.0)
        lo, hi = int(k), min(int(k) + 1, len(values) - 1)
        if lo == hi:
            return float(values[lo])
        return float(values[lo] + (values[hi] - values[lo]) * (k - lo))
