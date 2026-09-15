"""Monte Carlo — does random selection among tied names matter?

Precomputed by scripts/monte_carlo_random_book.py and
scripts/monte_carlo_managed_book.py (research artifacts, not wired into the
live production composite/scoring pipeline -- see
docs/monte_carlo_book_construction.md). This page only reads those cached
JSONs; it does not recompute the simulations live.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if sys.path[:1] != [_ROOT]:
    if _ROOT in sys.path:
        sys.path.remove(_ROOT)
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components import empty_state, page_header, sidebar

st.set_page_config(page_title="Monte Carlo", layout="wide", page_icon="🎲")
sidebar()

ROOT = Path(__file__).resolve().parents[2]

CONFIGS = {
    "sleeves": {
        "label": "Staggered sleeves — n=5 / hold=4mo / refresh=5 / 4 sleeves",
        "path": ROOT / "output" / "monte_carlo_random_book" / "n5_hold4_refresh5_sleeves4_sims2500.json",
        "color": "#2563eb",
        "blurb": (
            "4 staggered 5-name sleeves, each reviewed every 4 months, evicting "
            "5 names at random from that sleeve and refilling from the composite "
            "score==100 pool (instead of the deterministic worst-momentum-decline "
            "eviction rule)."
        ),
    },
}
BENCH_COLOR = {"SPY": "#64748b", "QQQ": "#94a3b8"}


@st.cache_data(ttl=3600, show_spinner=False)
def load_config(path_str: str) -> dict | None:
    path = Path(path_str)
    if not path.exists():
        return None
    with path.open() as fh:
        return json.load(fh)


page_header("Monte Carlo", "Random-entry simulations vs SPY and QQQ")

any_loaded = False
for key, meta in CONFIGS.items():
    data = load_config(str(meta["path"]))
    if data is None:
        continue
    any_loaded = True

    st.markdown(f"### {meta['label']}")
    st.caption(meta["blurb"])

    dates = data["dates"]
    p10 = np.array(data["p10_curve"]) * 1000.0
    p90 = np.array(data["p90_curve"]) * 1000.0
    median = np.array(data["median_curve"]) * 1000.0
    spy = np.array(data["spy_curve"]) * 1000.0
    qqq = np.array(data["qqq_curve"]) * 1000.0

    BAND_COLOR = "rgba(96, 165, 250, 0.28)"   # light blue
    MEDIAN_COLOR = "#1e3a8a"                  # dark blue
    LINE_COLORS = {"SPY": "#ea580c", "QQQ": "#059669"}   # distinct, dashed

    fig = go.Figure()
    fig.add_scatter(x=dates, y=p90, mode="lines", line=dict(width=0),
                    showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=dates, y=p10, mode="lines", line=dict(width=0),
                    fill="tonexty", fillcolor=BAND_COLOR,
                    name="10th–90th percentile", hoverinfo="skip")
    fig.add_scatter(x=dates, y=median, mode="lines",
                    line=dict(color=MEDIAN_COLOR, width=3.5), name="Median")
    fig.add_scatter(x=dates, y=spy, mode="lines",
                    line=dict(color=LINE_COLORS["SPY"], width=1.75, dash="dash"),
                    name="SPY")
    fig.add_scatter(x=dates, y=qqq, mode="lines",
                    line=dict(color=LINE_COLORS["QQQ"], width=1.75, dash="dash"),
                    name="QQQ")

    end_markers = [(median[-1], MEDIAN_COLOR), (spy[-1], LINE_COLORS["SPY"]),
                  (qqq[-1], LINE_COLORS["QQQ"])]
    for y_end, color in end_markers:
        fig.add_annotation(x=dates[-1], y=y_end, text=f"${y_end:,.0f}",
                           showarrow=False, xanchor="left", xshift=8,
                           font=dict(color=color, size=12))

    fig.update_layout(
        height=420, margin=dict(l=10, r=70, t=10, b=10),
        yaxis_title="Portfolio Value ($1,000 invested)",
        yaxis=dict(tickprefix="$", gridcolor="rgba(148, 163, 184, 0.15)"),
        xaxis=dict(gridcolor="rgba(148, 163, 184, 0.15)"),
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True, key=f"{key}_chart")

    params = data["params"]
    summary = data["summary"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sims", summary["n_sims"])
    m2.metric("Median final return", f"{summary['median_final_return']:.2f}x")
    m3.metric("% sims beating SPY", f"{summary['pct_sims_beating_spy']:.0%}")
    m4.metric("% sims beating QQQ", f"{summary['pct_sims_beating_qqq']:.0%}")

    rows = []
    for name, label in [("portfolio", meta["label"]), ("spy", "SPY"), ("qqq", "QQQ")]:
        s = data["stats"][name]
        rows.append({
            "": label, "CAGR": s["cagr"], "Sharpe": s["sharpe"],
            "Sortino": s["sortino"], "Max DD": s["max_dd"],
            "Beta (vs SPY)": s["beta"], "Alpha (vs SPY, ann.)": s["alpha"],
        })
    table = pd.DataFrame(rows).set_index("")
    st.dataframe(
        table.style.format({
            "CAGR": "{:+.1%}", "Sharpe": "{:.2f}", "Sortino": "{:.2f}",
            "Max DD": "{:.1%}", "Beta (vs SPY)": "{:.2f}",
            "Alpha (vs SPY, ann.)": "{:+.1%}",
        }),
        use_container_width=True, height=42 + 35 * len(table))
    st.caption(f"Params: `{json.dumps(params)}`. Portfolio stats are the mean "
              f"across all {summary['n_sims']} random-draw sims.")
    st.divider()

INSIDER_REV_PATH = ROOT / "output" / "loop_research" / "insider_revisions_min_perturbation.json"
insider_rev_data = load_config(str(INSIDER_REV_PATH))
if insider_rev_data is not None:
    any_loaded = True
    st.markdown("### Top-20-pool champion — min(insider, revisions), 0.5x noise")
    st.caption(
        "Within the top-20-by-composite-score pool, rank by min(insider parent "
        "score, revisions parent score) descending — both signals must "
        "independently clear a bar (floor/AND logic), not a sum or average. "
        f"Gaussian noise ({insider_rev_data['noise_mult']}x the date's "
        "cross-sectional std) is added to the ranking score before each sim's "
        "top-5 selection so the band reflects genuine ranking-sensitivity, not "
        "just a single deterministic path. Research finding only "
        "(research/loop_research champion.md, promoted 2026-09-12), not wired "
        "into production selection.")

    dates_r = insider_rev_data["dates"]
    p10_r = np.array(insider_rev_data["p10_curve"]) * 1000.0
    p90_r = np.array(insider_rev_data["p90_curve"]) * 1000.0
    median_r = np.array(insider_rev_data["median_curve"]) * 1000.0
    spy_r = np.array(insider_rev_data["spy_curve"]) * 1000.0
    qqq_r = np.array(insider_rev_data["qqq_curve"]) * 1000.0

    fig_r = go.Figure()
    fig_r.add_scatter(x=dates_r, y=p90_r, mode="lines", line=dict(width=0),
                      showlegend=False, hoverinfo="skip")
    fig_r.add_scatter(x=dates_r, y=p10_r, mode="lines", line=dict(width=0),
                      fill="tonexty", fillcolor=BAND_COLOR,
                      name="10th–90th percentile", hoverinfo="skip")
    fig_r.add_scatter(x=dates_r, y=median_r, mode="lines",
                      line=dict(color=MEDIAN_COLOR, width=3.5), name="Median")
    fig_r.add_scatter(x=dates_r, y=spy_r, mode="lines",
                      line=dict(color=LINE_COLORS["SPY"], width=1.75, dash="dash"),
                      name="SPY")
    fig_r.add_scatter(x=dates_r, y=qqq_r, mode="lines",
                      line=dict(color=LINE_COLORS["QQQ"], width=1.75, dash="dash"),
                      name="QQQ")

    for y_end, color in [(median_r[-1], MEDIAN_COLOR), (spy_r[-1], LINE_COLORS["SPY"]),
                         (qqq_r[-1], LINE_COLORS["QQQ"])]:
        fig_r.add_annotation(x=dates_r[-1], y=y_end, text=f"${y_end:,.0f}",
                             showarrow=False, xanchor="left", xshift=8,
                             font=dict(color=color, size=12))

    fig_r.update_layout(
        height=420, margin=dict(l=10, r=70, t=10, b=10),
        yaxis_title="Portfolio Value ($1,000 invested)",
        yaxis=dict(tickprefix="$", gridcolor="rgba(148, 163, 184, 0.15)"),
        xaxis=dict(gridcolor="rgba(148, 163, 184, 0.15)"),
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig_r, use_container_width=True, key="insider_rev_chart")

    summary_r = insider_rev_data["summary"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sims", summary_r["n_sims"])
    m2.metric("Median final return", f"{summary_r['median_final_return']:.2f}x")
    m3.metric("% sims beating SPY", f"{summary_r['pct_sims_beating_spy']:.0%}")
    m4.metric("% sims beating QQQ", f"{summary_r['pct_sims_beating_qqq']:.0%}")

    rows_r = []
    for name, label in [("portfolio", "min(insider, revisions)"), ("spy", "SPY"), ("qqq", "QQQ")]:
        s = insider_rev_data["stats"][name]
        rows_r.append({
            "": label, "CAGR": s["cagr"], "Sharpe": s["sharpe"],
            "Sortino": s["sortino"], "Max DD": s["max_dd"],
            "Beta (vs SPY)": s.get("beta"), "Alpha (vs SPY, ann.)": s.get("alpha"),
        })
    table_r = pd.DataFrame(rows_r).set_index("")
    st.dataframe(
        table_r.style.format({
            "CAGR": "{:+.1%}", "Sharpe": "{:.2f}", "Sortino": "{:.2f}",
            "Max DD": "{:.1%}", "Beta (vs SPY)": "{:.2f}",
            "Alpha (vs SPY, ann.)": "{:+.1%}",
        }),
        use_container_width=True, height=42 + 35 * len(table_r))
    st.caption(f"Params: `{json.dumps(insider_rev_data['params'])}`. Portfolio "
              f"stats are the mean across all {summary_r['n_sims']} noise sims.")
    st.divider()

if not any_loaded:
    empty_state(
        "No Monte Carlo results yet.",
        "Run `python scripts/monte_carlo_random_book.py --n 5 --hold 2 --refresh 5 "
        "--sims 500` and `python scripts/monte_carlo_managed_book.py --mode random "
        "--k 5 --sims 200` first — see docs/monte_carlo_book_construction.md.")
    st.stop()

st.caption("See docs/monte_carlo_book_construction.md for the full writeup, "
          "other saved configs, and caveats. Not wired into the live "
          "production composite/scoring pipeline.")
