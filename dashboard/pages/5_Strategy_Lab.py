"""Strategy Lab — two loop-engineering research strategies vs SPY/QQQ.

Precomputed by scripts/generate_strategy_lab_comparison.py (research
artifacts, not wired into the live production composite/scoring pipeline --
see research/loop_engineering/README.md and research/autoresearch/
session_log.md). This page only reads that cached JSON; it does not
recompute the backtests live.
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

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import strategy_lab_live as live
from dashboard.components import empty_state, page_header, sidebar

st.set_page_config(page_title="Strategy Lab", layout="wide", page_icon="🧪")
sidebar()

COMPARISON_JSON = (Path(__file__).resolve().parents[2]
                   / "output" / "loop_engineering_v3" / "strategy_lab_comparison.json")

STRATEGIES = {
    "v3_loopeng_cap9_minhold1wk_quartile": {
        "label": "v3 loop-eng: cap9 + min-hold + quartile exit",
        "color": "#2563eb",
        "blurb": (
            "A single top-10 book selected by the production EQEFF composite score, "
            "held with a 10% trailing stop and a 9-month time cap, plus two extra exit "
            "rules layered on afterward: a one-week minimum hold before any soft exit "
            "can fire, and a forced sale if a name's rank falls into the worst quartile "
            "of the book at a monthly review."
        ),
    },
    "21_loopeng_book4_hold4_evict3": {
        "label": "autoresearch: book4 / hold4 / evict3",
        "color": "#d97706",
        "blurb": (
            "Three staggered 4-name sleeves on the same composite score, each reviewed "
            "every 4 months and evicting its 3 worst-momentum names (biggest score "
            "decline since the prior review) at each review, refilling with the next-"
            "best-ranked names. A small, high-turnover, high-conviction variant of the "
            "autoresearch momentum-eviction family (the live production recipe there "
            "instead uses an 11-name book with only 1 eviction per review)."
        ),
    },
}
BENCH_COLOR = {"SPY": "#64748b", "QQQ": "#94a3b8"}


@st.cache_data(ttl=3600, show_spinner=False)
def load_comparison() -> dict | None:
    if not COMPARISON_JSON.exists():
        return None
    with COMPARISON_JSON.open() as fh:
        return json.load(fh)


page_header("Strategy Lab",
           "Loop-engineering research strategies vs SPY/QQQ — equity curves, "
           "mechanics, and risk/return stats")

data = load_comparison()
if data is None:
    empty_state(
        "No strategy comparison data yet.",
        "Run `python scripts/generate_strategy_lab_comparison.py` first "
        "(rebuilds output/loop_engineering_v3/strategy_lab_comparison.json).")
    st.stop()

st.caption(
    "Research artifacts, not the live production strategy — see "
    "research/loop_engineering/README.md and "
    "research/autoresearch/session_log.md. Both strategies run on the same "
    "production EQEFF composite score and price panel, 2020-01 through "
    "2026-06 (78 monthly reviews), benchmarked against the same same-day "
    "SPY/QQQ monthly returns.")

st.divider()

# ---------------------------------------------------------------------------
# Equity curves
# ---------------------------------------------------------------------------
st.markdown("### Equity curves")
dates = data["dates"]
fig = go.Figure()
for name, meta in STRATEGIES.items():
    fig.add_scatter(x=dates, y=data["equity"][name], mode="lines",
                    name=meta["label"], line=dict(color=meta["color"], width=2.5))
for bench in ("SPY", "QQQ"):
    fig.add_scatter(x=dates, y=data["equity"][bench], mode="lines",
                    name=bench, line=dict(color=BENCH_COLOR[bench], width=1.5, dash="dot"))
fig.update_layout(height=440, margin=dict(l=10, r=10, t=10, b=10),
                  yaxis_title="Growth of $1", yaxis_tickformat=".1f",
                  legend=dict(orientation="h", y=1.12))
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Mechanics blurbs
# ---------------------------------------------------------------------------
st.markdown("### How each strategy works")
cols = st.columns(2)
for col, (name, meta) in zip(cols, STRATEGIES.items()):
    with col:
        st.markdown(f"**{meta['label']}**")
        st.caption(meta["blurb"])

st.divider()

# ---------------------------------------------------------------------------
# Results summary table
# ---------------------------------------------------------------------------
st.markdown("### Results summary")
order = list(STRATEGIES) + ["SPY", "QQQ"]
labels = {**{k: v["label"] for k, v in STRATEGIES.items()}, "SPY": "SPY", "QQQ": "QQQ"}
rows = []
for name in order:
    m = data["metrics"][name]
    rows.append({
        "strategy": labels[name],
        "CAGR": m["cagr"],
        "Sharpe": m["sharpe"],
        "Beta (vs SPY)": m["beta_vs_spy"],
        "Max DD": m["max_dd"],
        "Alpha (vs SPY, ann.)": m["alpha_vs_spy"],
        "R² vs SPY": m["r2_vs_spy"],
        "R² vs QQQ": m["r2_vs_qqq"],
    })
table = pd.DataFrame(rows).set_index("strategy")
st.dataframe(
    table.style.format({
        "CAGR": "{:+.1%}", "Sharpe": "{:.2f}", "Beta (vs SPY)": "{:.2f}",
        "Max DD": "{:.1%}", "Alpha (vs SPY, ann.)": "{:+.1%}",
        "R² vs SPY": "{:.2f}", "R² vs QQQ": "{:.2f}",
    }),
    use_container_width=True, height=42 + 35 * len(table))
st.caption("Sharpe/CAGR/max-DD computed on monthly returns, 2020-01 through 2026-06 "
          "(78 reviews). Alpha is the annualized CAPM intercept vs SPY monthly "
          "returns; R² is the squared Pearson correlation of monthly returns. Not "
          "wired into the live production composite/scoring pipeline.")

st.divider()

# ---------------------------------------------------------------------------
# Live snapshot: what would you buy right now
# ---------------------------------------------------------------------------
st.markdown("## Live snapshot: what would you buy")
st.caption(
    "Uses the LIVE production composite scores and prices (not the frozen "
    "2020-2026 research panel above), so this reflects the actual current "
    "date. Both baskets use plain equal weighting (1/N per name), the same "
    "position-sizing convention each strategy uses in its own backtest.")

today_baskets = live.form_today()
month_ago = live.month_ago_scenarios()

if not today_baskets or not month_ago:
    empty_state("No live composite scores found in the warehouse yet.",
               "Run `python run_scoring.py` first.")
    st.stop()

st.markdown("### If you started today")
cols = st.columns(2)
for col, name in zip(cols, STRATEGIES):
    basket = today_baskets[name]
    with col:
        st.markdown(f"**{STRATEGIES[name]['label']}** — as of {basket.as_of}")
        rows = sorted(basket.weights.items(), key=lambda kv: -kv[1])
        st.dataframe(
            pd.DataFrame(rows, columns=["ticker", "weight"])
            .style.format({"weight": "{:.1%}"}),
            use_container_width=True, hide_index=True,
            height=42 + 35 * min(len(rows), 12))
        if basket.note:
            st.caption(basket.note)

st.markdown("### If you started a month ago")

for name in STRATEGIES:
    res = month_ago[name]
    st.markdown(f"**{STRATEGIES[name]['label']}** — bought {res.formed_on}, "
               f"as of {res.as_of_today}")

    fig = go.Figure()
    fig.add_scatter(x=list(res.equity.index), y=res.equity.values, mode="lines",
                    name=STRATEGIES[name]["label"],
                    line=dict(color=STRATEGIES[name]["color"], width=2.5))
    fig.add_scatter(x=list(res.spy_equity.index), y=res.spy_equity.values, mode="lines",
                    name="SPY", line=dict(color=BENCH_COLOR["SPY"], width=1.5, dash="dot"))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_title="Growth of $1", legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("CAGR (annualized)", f"{res.cagr:+.0%}" if res.cagr == res.cagr else "—")
    m2.metric("Beta (vs SPY)", f"{res.beta:.2f}" if res.beta == res.beta else "—")
    m3.metric("Alpha (vs SPY, ann.)", f"{res.alpha:+.0%}" if res.alpha == res.alpha else "—")
    st.caption("Annualized off ~1 month of daily returns — small-sample, noisy by "
              "construction; a single bad or good week swings these a lot.")

    t1, t2, t3 = st.columns(3)
    with t1:
        st.caption(f"Bought on {res.formed_on}:")
        rows = sorted(res.initial_weights.items(), key=lambda kv: -kv[1])
        st.dataframe(pd.DataFrame(rows, columns=["ticker", "weight"])
                    .style.format({"weight": "{:.1%}"}),
                    use_container_width=True, hide_index=True,
                    height=42 + 35 * min(len(rows), 8))
    with t2:
        if res.events:
            st.caption("What happened since:")
            ev = pd.DataFrame(res.events)[["date", "ticker", "reason"]]
            st.dataframe(ev, use_container_width=True, hide_index=True,
                        height=42 + 35 * min(len(ev), 8))
        else:
            st.caption("No exits — buy-and-hold, no review due within a month.")
    with t3:
        st.caption(f"Holding today:")
        rows = sorted(res.current_weights.items(), key=lambda kv: -kv[1])
        st.dataframe(pd.DataFrame(rows, columns=["ticker", "weight"])
                    .style.format({"weight": "{:.1%}"}),
                    use_container_width=True, hide_index=True,
                    height=42 + 35 * min(len(rows), 8))
    if res.note:
        st.caption(res.note)
    st.divider()
