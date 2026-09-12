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
    "managed": {
        "label": "Managed book — k=5, cap9/trail10%/rankfloor75%",
        "path": ROOT / "output" / "monte_carlo_managed_book" / "k5_results.json",
        "color": "#d97706",
        "blurb": (
            "A single 5-name book with the v3 loop-engineering exit rule (9-month "
            "time cap OR 10% trailing stop OR falling into the worst quartile of "
            "the scored universe, 1-week minimum hold), but entries are a uniform "
            "random draw from the score==100 pool instead of top-k-by-score."
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

    insider_curve, insider_stats = None, None
    if key == "sleeves":
        pool_path = ROOT / "output" / "loop_research" / "pool_comparison.json"
        pool_data = load_config(str(pool_path))
        if pool_data is not None:
            top100 = pool_data["top100"]
            insider_curve = np.array(top100["insider_curve"]) * 1000.0
            insider_stats = top100["insider_stats"]

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

    INSIDER_COLOR = "#e87ba4"   # magenta, distinct from median/SPY/QQQ
    if insider_curve is not None and len(insider_curve) == len(dates):
        fig.add_scatter(x=dates, y=insider_curve, mode="lines",
                        line=dict(color=INSIDER_COLOR, width=3),
                        name="Top-5 by insider score (deterministic)")

    end_markers = [(median[-1], MEDIAN_COLOR), (spy[-1], LINE_COLORS["SPY"]),
                  (qqq[-1], LINE_COLORS["QQQ"])]
    if insider_curve is not None and len(insider_curve) == len(dates):
        end_markers.append((insider_curve[-1], INSIDER_COLOR))
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
        if insider_stats is not None and name == "portfolio":
            rows.append({
                "": "Top-5 by insider score (deterministic)",
                "CAGR": insider_stats["cagr"], "Sharpe": insider_stats["sharpe"],
                "Sortino": insider_stats["sortino"], "Max DD": insider_stats["max_dd"],
                "Beta (vs SPY)": insider_stats["beta"],
                "Alpha (vs SPY, ann.)": insider_stats["alpha"],
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
    if insider_stats is not None:
        st.caption("Magenta line: instead of a uniform random draw, deterministically "
                  "pick the top 5 by insider parent score (ins_no_selling_flag / "
                  "ins_cluster_buyers_180d / ins_sell_pressure_inv) from the score==100 "
                  "pool each review. Found via research/loop_research (93-candidate "
                  "batch, 2026-09-09): beats the random baseline on 98.4% of paired "
                  "seeds, 95% CI on the paired Sharpe difference entirely positive. "
                  "Research finding only, not wired into production selection — see "
                  "research/loop_research/session_log.md and champion.md for the full "
                  "writeup, including the scope-limited caveat (works best as a "
                  "tie-breaker within this already-elite pool, weaker on a less "
                  "pre-screened pool).")
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
