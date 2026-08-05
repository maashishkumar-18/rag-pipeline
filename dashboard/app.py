"""
Observability Dashboard — Streamlit

Reads observability/metrics.db (see observability/metrics_store.py) and
renders latency, cost, and quality trends across simulated/CI traffic, plus
a recent-requests table linking out to the full Langfuse trace waterfall
for any individual request. Complements Langfuse's own UI (per-request
detail) rather than duplicating it — this dashboard is for aggregate trend
questions ("is quality degrading over time?"), Langfuse is for "what
happened in this one request?".

Run:
    streamlit run dashboard/app.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from observability.metrics_store import MetricsStore

# ============================================================================
# Palette — validated colorblind-safe values from the dataviz skill's
# references/palette.md, copied verbatim (categorical slot order is itself
# the safety mechanism — never reorder or substitute individual hues).
# ============================================================================

CATEGORICAL = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark":  ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
SEQUENTIAL_BLUE = {"light": "#2a78d6", "dark": "#3987e5"}
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
INK = {
    "light": {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"},
    "dark":  {"primary": "#ffffff", "secondary": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835"},
}


def get_theme() -> str:
    base = st.get_option("theme.base")
    return "dark" if base == "dark" else "light"


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def style_fig(fig: go.Figure, theme: str, show_legend: bool = False) -> go.Figure:
    ink = INK[theme]
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=ink["secondary"], size=13),
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=ink["secondary"])),
        margin=dict(l=10, r=10, t=36 if show_legend else 10, b=10),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, linecolor=ink["axis"], color=ink["muted"])
    fig.update_yaxes(showgrid=True, gridcolor=ink["grid"], zeroline=False, linecolor=ink["axis"], color=ink["muted"])
    return fig


# ============================================================================
# Page + data load
# ============================================================================

st.set_page_config(page_title="RAG Pipeline — Observability", layout="wide")
theme = get_theme()
ink = INK[theme]
cat = CATEGORICAL[theme]
seq_blue = SEQUENTIAL_BLUE[theme]

st.title("RAG Pipeline — Observability Dashboard")
st.caption(
    "Extension of Project 1 — tracing (Langfuse) + quality/cost/latency metrics "
    "over CI and simulated traffic. Click any request's trace link to see the "
    "full retrieval → rerank → generation waterfall in Langfuse."
)

store = MetricsStore()
probe_rows = store.query_recent(limit=5000)
if not probe_rows:
    st.info(
        "No data yet. Run `python -m eval.run_ragas_eval` (env=ci) or "
        "`python -m scripts.simulate_traffic` (env=simulated_production) to populate this dashboard."
    )
    st.stop()

all_envs = sorted({r["env"] for r in probe_rows})

filter_col1, filter_col2, _ = st.columns([1, 2, 3])
with filter_col1:
    default_env = "simulated_production" if "simulated_production" in all_envs else "All"
    options = ["All"] + all_envs
    env_filter = st.selectbox("Environment", options=options, index=options.index(default_env))

rows = store.query_all(env=None if env_filter == "All" else env_filter)
df = pd.DataFrame(rows)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
df["request_index"] = range(1, len(df) + 1)
df["stage_timings"] = df["stage_timings_json"].apply(lambda s: json.loads(s) if s else {})

with filter_col2:
    min_date, max_date = df["timestamp"].min().date(), df["timestamp"].max().date()
    if min_date != max_date:
        date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start, end = date_range
            mask = (df["timestamp"].dt.date >= start) & (df["timestamp"].dt.date <= end)
            df = df[mask].reset_index(drop=True)
            df["request_index"] = range(1, len(df) + 1)

if df.empty:
    st.warning("No requests match the current filters.")
    st.stop()

# ============================================================================
# KPI row
# ============================================================================

n = len(df)
p50 = df["total_time_ms"].quantile(0.5)
p95 = df["total_time_ms"].quantile(0.95)
total_cost = df["cost_usd"].sum()
refusal_rate = df["refused"].mean() * 100
hit_rate = df["retrieval_hit"].mean() * 100
faith_vals = df["faithfulness_score"].dropna()
avg_faith = faith_vals.mean() if len(faith_vals) else None

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Requests", n)
k2.metric("Latency P50 / P95", f"{p50:.0f} / {p95:.0f} ms")
k3.metric("Total cost", f"${total_cost:.4f}")
k4.metric("Refusal rate", f"{refusal_rate:.1f}%")
k5.metric("Retrieval hit rate", f"{hit_rate:.1f}%")
k6.metric("Avg faithfulness (sampled)", f"{avg_faith:.2f}" if avg_faith is not None else "n/a")

st.divider()

# ============================================================================
# Latency
# ============================================================================

st.subheader("Latency")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["request_index"], y=df["total_time_ms"], mode="lines+markers",
    line=dict(color=seq_blue, width=2), marker=dict(size=6),
    hovertemplate="Request %{x}<br>%{y:.0f} ms<extra></extra>",
))
fig.add_hline(y=p50, line_dash="dash", line_color=ink["muted"], annotation_text=f"P50 {p50:.0f}ms", annotation_position="top left")
fig.add_hline(y=p95, line_dash="dash", line_color=STATUS["warning"], annotation_text=f"P95 {p95:.0f}ms", annotation_position="top left")
fig.update_layout(yaxis_title="ms", xaxis_title="Request #")
st.plotly_chart(style_fig(fig, theme), use_container_width=True)

st.subheader("Per-stage latency (average)")
stage_means: dict = {}
for stages in df["stage_timings"]:
    for k, v in stages.items():
        stage_means.setdefault(k, []).append(v)

if stage_means:
    stage_avg = {k: sum(v) / len(v) for k, v in stage_means.items()}
    stage_series = pd.Series(stage_avg).sort_values(ascending=True)
    # Cap at 8 categorical slots (the validated palette's series limit) —
    # fold anything past that into "other" rather than generating a 9th hue.
    if len(stage_series) > 8:
        top = stage_series.iloc[-7:]
        other = stage_series.iloc[:-7].sum()
        stage_series = pd.concat([pd.Series({"other": other}), top]).sort_values()

    fig = go.Figure(go.Bar(
        x=stage_series.values, y=stage_series.index, orientation="h",
        marker_color=[cat[i % len(cat)] for i in range(len(stage_series))],
        hovertemplate="%{y}: %{x:.0f} ms<extra></extra>",
    ))
    fig.update_layout(xaxis_title="ms")
    st.plotly_chart(style_fig(fig, theme), use_container_width=True)
else:
    st.caption("No per-stage timing data (all requests in this filter failed before any stage completed).")

st.divider()

# ============================================================================
# Quality — faithfulness (with incident window), refusal rate, hit rate
# ============================================================================

st.subheader("Faithfulness (sampled)")
faith_df = df.dropna(subset=["faithfulness_score"])
if len(faith_df):
    fig = go.Figure()
    incident_mask = df["retrieval_pipeline_name"] == "learning_degraded_incident"
    if incident_mask.any():
        # Shade each contiguous run of incident-tagged requests separately —
        # incident rows aren't necessarily one contiguous block (e.g. two
        # separate scripts/simulate_traffic.py invocations, each with its
        # own incident window, landing in the same metrics store): a single
        # min-to-max band would falsely span untouched requests in between.
        incident_indices = df.loc[incident_mask, "request_index"].tolist()
        runs = [[incident_indices[0], incident_indices[0]]]
        for idx in incident_indices[1:]:
            if idx == runs[-1][1] + 1:
                runs[-1][1] = idx
            else:
                runs.append([idx, idx])
        for i, (s, e) in enumerate(runs):
            vrect_kwargs = {}
            if i == 0:
                # Plotly renders a literal placeholder ("new text") if
                # annotation_text=None is passed explicitly — omit the kwarg
                # entirely for the unlabeled runs instead.
                vrect_kwargs = dict(annotation_text="incident: degraded retrieval config", annotation_position="top left")
            fig.add_vrect(
                x0=s - 0.5, x1=e + 0.5,
                fillcolor=STATUS["critical"], opacity=0.12, line_width=0,
                **vrect_kwargs,
            )
    fig.add_hline(y=0.75, line_dash="dash", line_color=ink["muted"], annotation_text="threshold 0.75", annotation_position="bottom left")
    fig.add_trace(go.Scatter(
        x=faith_df["request_index"], y=faith_df["faithfulness_score"], mode="lines+markers",
        line=dict(color=seq_blue, width=2), marker=dict(size=8),
        hovertemplate="Request %{x}<br>faithfulness %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(yaxis_title="faithfulness", xaxis_title="Request #", yaxis_range=[0, 1.05])
    st.plotly_chart(style_fig(fig, theme), use_container_width=True)
else:
    st.caption("No sampled faithfulness scores yet — run scripts/simulate_traffic.py or the eval harness without --skip-ragas.")

st.subheader("Refusal rate & retrieval hit rate (rolling window)")
window = max(3, len(df) // 10)
df["refusal_rate_roll"] = df["refused"].rolling(window, min_periods=1).mean() * 100
df["hit_rate_roll"] = df["retrieval_hit"].rolling(window, min_periods=1).mean() * 100

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["request_index"], y=df["hit_rate_roll"], mode="lines",
    line=dict(color=STATUS["good"], width=2), name="Retrieval hit rate",
    hovertemplate="Request %{x}<br>hit rate %{y:.0f}%<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=df["request_index"], y=df["refusal_rate_roll"], mode="lines",
    line=dict(color=STATUS["critical"], width=2), name="Refusal rate",
    hovertemplate="Request %{x}<br>refusal rate %{y:.0f}%<extra></extra>",
))
# Padded range, not [0, 100]: a flat line sitting exactly on the plot
# border (e.g. a constant 100% hit rate) would otherwise blend into the
# axis and read as missing data instead of a real flat trend.
fig.update_layout(yaxis_title="%", xaxis_title="Request #", yaxis_range=[-4, 104])
st.plotly_chart(style_fig(fig, theme, show_legend=True), use_container_width=True)
st.caption(f"Rolling window: {window} requests.")

st.divider()

# ============================================================================
# Cost — per-request and cumulative are different scales, so two charts
# rather than a dual-axis combo (dual-axis charts are never the right call).
# ============================================================================

st.subheader("Cost")
cost_col1, cost_col2 = st.columns(2)
with cost_col1:
    fig = go.Figure(go.Bar(
        x=df["request_index"], y=df["cost_usd"], marker_color=seq_blue,
        hovertemplate="Request %{x}<br>$%{y:.5f}<extra></extra>",
    ))
    fig.update_layout(yaxis_title="$ per request", xaxis_title="Request #")
    st.plotly_chart(style_fig(fig, theme), use_container_width=True)
with cost_col2:
    df["cumulative_cost"] = df["cost_usd"].cumsum()
    fig = go.Figure(go.Scatter(
        x=df["request_index"], y=df["cumulative_cost"], mode="lines",
        line=dict(color=seq_blue, width=2), fill="tozeroy", fillcolor=hex_to_rgba(seq_blue, 0.15),
        hovertemplate="Request %{x}<br>cumulative $%{y:.4f}<extra></extra>",
    ))
    fig.update_layout(yaxis_title="$ cumulative", xaxis_title="Request #")
    st.plotly_chart(style_fig(fig, theme), use_container_width=True)

st.divider()

# ============================================================================
# Recent requests — table view, with a link out to each trace's Langfuse waterfall
# ============================================================================

st.subheader("Recent requests")
recent = df.sort_values("timestamp", ascending=False).head(30).copy()
display_cols = [
    "timestamp", "env", "query", "confidence_level", "refused", "is_grounded",
    "citations_count", "faithfulness_score", "cost_usd", "langfuse_trace_url",
]
st.dataframe(
    recent[display_cols],
    column_config={
        "langfuse_trace_url": st.column_config.LinkColumn("Trace", display_text="View in Langfuse"),
        "timestamp": st.column_config.DatetimeColumn("Time"),
        "cost_usd": st.column_config.NumberColumn("Cost", format="$%.5f"),
        "faithfulness_score": st.column_config.NumberColumn("Faithfulness", format="%.2f"),
        "query": st.column_config.TextColumn("Query", width="large"),
    },
    use_container_width=True,
    hide_index=True,
)

with st.expander("Raw data"):
    st.dataframe(df.drop(columns=["stage_timings"]), use_container_width=True, hide_index=True)
