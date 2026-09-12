"""Scoring — inside the composite: weights, crowding, and subfactors.

Shows each parent's engine weight next to its *effective* exposure
(C·w — own weight plus what flows in through correlated parents), the
parent correlation heatmap those effective numbers come from, and the
production subfactor formulas with their research-battery statistics.
"""
from __future__ import annotations

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

from dashboard import data as ddata
from dashboard.components import empty_state, kpi_row, page_header, sidebar

st.set_page_config(page_title="Scoring", layout="wide", page_icon="⚖️")
sidebar()

weights = pd.Series(ddata.load_engine_weights(), name="weight")
C = ddata.load_parent_correlation()
r2 = ddata.load_factor_composite_r2()
as_of = ddata.latest_score_date()

page_header("Scoring",
            "Engine weights · effective exposure · subfactor formulas", as_of)

if C.empty:
    empty_state("No parent factor scores in the warehouse yet.",
                "Run `python run_scoring.py` first.")
    st.stop()

parents = [p for p in weights.index if p in C.index]
w = weights.reindex(parents)
Cm = C.loc[parents, parents]
eff = pd.Series(Cm.values @ w.values, index=parents, name="effective")
eff_bets = 1.0 / float(w.values @ Cm.values @ w.values)

kpi_row([
    ("Parents", f"{len(parents)}"),
    ("Effective bets", f"{eff_bets:.1f}", "1 / w'Cw"),
    ("Effective exposure spread", f"{eff.max() - eff.min():.3f}",
     f"target 0 (equal exposure); {eff.idxmax()} highest, {eff.idxmin()} lowest"),
    ("Corr window", "last 30 score dates"),
])
st.caption("Weights are the 2026-09-01 method-EQEFF ship: solved so every parent's "
           "EFFECTIVE exposure (C·w — own weight plus what correlated parents "
           "import) comes out equal, rather than capping the crowded ones "
           "(method B) or capping+flooring the outliers (method E). No IC/IR "
           "information is used — pure correlation-structure risk parity. Beat "
           "10 IC-informed hybrids on every metric of a 10-metric portfolio-"
           "agnostic scorecard (output/crowding/weight_config_study/); a "
           "concentrated top-25% costed portfolio backtest still favors the "
           "prior method E specifically, a deliberate tradeoff — see "
           "factors/parent_selection_v4.py for the full evidence trail.")

st.divider()

# ---------------------------------------------------------------------------
# Nominal vs effective weight
# ---------------------------------------------------------------------------
left, right = st.columns([1.15, 1])

with left:
    st.markdown("### Weight vs effective exposure vs realized R²")
    order = w.sort_values(ascending=False).index
    r2o = r2.reindex(order)
    fig = go.Figure()
    fig.add_bar(name="Engine weight", x=list(order), y=w.reindex(order),
                marker_color="#2563eb")
    fig.add_bar(name="Effective exposure (C·w)", x=list(order),
                y=eff.reindex(order), marker_color="#d97706")
    if not r2o.dropna().empty:
        fig.add_bar(name="R² vs composite score", x=list(order), y=r2o,
                    marker_color="#16a34a")
    fig.add_hline(y=float(eff.mean()), line_dash="dash", line_color="#dc2626",
                  annotation_text=f"equal-exposure target ({eff.mean():.1%})")
    fig.update_layout(barmode="group", height=380, yaxis_tickformat=".0%",
                      margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Effective exposure = a parent's own weight plus what correlated "
               "parents import. Method EQEFF solves for the weight vector that "
               "makes every orange bar equal (the dashed line) — the resulting "
               "common value is a property of the current correlation "
               "structure, not a fixed target like 1/8. R² vs composite score "
               "is measured on the realized, latest-date output (after "
               "normalization and the sector re-rank) — the honest answer to "
               "'how much does this factor actually move the ranking,' as "
               "opposed to weight or effective exposure, which describe the "
               "blend inputs, not the output.")

with right:
    st.markdown("### Parent correlation")
    hm = go.Figure(go.Heatmap(
        z=Cm.values, x=parents, y=parents,
        zmin=-1, zmax=1, colorscale="RdBu_r",
        text=[[f"{v:.2f}" for v in row] for row in Cm.values],
        texttemplate="%{text}", textfont=dict(size=10)))
    hm.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                     yaxis_autorange="reversed")
    st.plotly_chart(hm, use_container_width=True)
    st.caption("Mean per-date cross-sectional Pearson correlation of parent "
               "scores, last 30 score dates.")

st.divider()

# ---------------------------------------------------------------------------
# Subfactor formulas
# ---------------------------------------------------------------------------
st.markdown("### Subfactors in production")
subs = ddata.load_selected_subs()
stats = ddata.load_subfactor_stats()
stats_by = stats.set_index("candidate") if not stats.empty else pd.DataFrame()

rows = []
for parent in w.sort_values(ascending=False).index:
    for sub, sw in subs.get(parent, {}).items():
        row = {"parent": parent, "subfactor": sub, "sub weight": sw,
               "parent weight": w[parent],
               "combined weight": sw * w[parent]}
        if sub in stats_by.index:
            s = stats_by.loc[sub]
            row |= {"mean IC (3M/6M)": s.get("mean_ic_3m6m"),
                    "IR": s.get("information_ratio"),
                    "coverage": s.get("coverage"),
                    "hit rate": s.get("hit_rate")}
        rows.append(row)
sub_df = pd.DataFrame(rows)

st.dataframe(
    sub_df.style.format({
        "sub weight": "{:.3f}", "parent weight": "{:.4f}",
        "combined weight": "{:.4f}", "mean IC (3M/6M)": "{:+.4f}",
        "IR": "{:.2f}", "coverage": "{:.0%}", "hit rate": "{:.0%}",
    }, na_rep="—"),
    use_container_width=True, hide_index=True,
    height=42 + 35 * len(sub_df))
st.caption("Sub weights renormalize within each parent at scoring time "
           "(missing subs fall to neutral 50). IC/IR/coverage come from the "
           "latest research-battery validation run "
           "(output/subfactor_expansion/summary_3M.csv), not live returns.")

st.divider()

# ---------------------------------------------------------------------------
# Rolling IC by parent
# ---------------------------------------------------------------------------
st.markdown("### Rolling 5-year IC by parent")

ic_hist = ddata.load_parent_ic_history(horizon="3M")
if ic_hist.empty:
    st.caption("No IC history yet — run `python run_factor_research.py` first.")
else:
    ROLL_WINDOW = 12  # trailing rebalances (~1yr of monthly points) to smooth
    cutoff = ic_hist.index.max() - pd.DateOffset(years=5)
    windowed = ic_hist[ic_hist.index >= cutoff]
    rolling = windowed.rolling(ROLL_WINDOW, min_periods=max(3, ROLL_WINDOW // 2)).mean()

    # dataviz-skill validated 8-slot categorical palette, fixed hue order
    PALETTE = ["#2a78d6", "#008300", "#e87ba4", "#eda100",
               "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]

    ic_fig = go.Figure()
    for i, parent in enumerate(rolling.columns):
        ic_fig.add_scatter(x=rolling.index, y=rolling[parent], mode="lines",
                           name=parent, line=dict(color=PALETTE[i % len(PALETTE)], width=2))
    ic_fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
    ic_fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title=f"rolling {ROLL_WINDOW}-period mean IC (3M fwd return)",
        legend=dict(orientation="h", y=1.12),
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="rgba(148, 163, 184, 0.15)"))
    st.plotly_chart(ic_fig, use_container_width=True)

    latest = rolling.iloc[-1].dropna().sort_values(ascending=False)
    full_mean = windowed[latest.index].mean()
    ic_table = pd.DataFrame({
        "Latest rolling IC": latest,
        "Full-window mean IC": full_mean,
    })
    st.dataframe(
        ic_table.style.format("{:+.4f}"),
        use_container_width=True, height=42 + 35 * len(ic_table))

    st.markdown("#### Current 5-year IC by parent")
    bar_order = full_mean.sort_values().index  # ascending so best is at top, horizontal
    bar_vals = full_mean.reindex(bar_order)
    bar_colors = ["#e34948" if v < 0 else "#2a78d6" for v in bar_vals]
    bar_fig = go.Figure(go.Bar(
        x=bar_vals, y=list(bar_order), orientation="h",
        marker_color=bar_colors,
        text=[f"{v:+.3f}" for v in bar_vals], textposition="outside"))
    bar_fig.add_vline(x=0, line_color="#94a3b8")
    bar_fig.update_layout(
        height=80 + 34 * len(bar_vals), margin=dict(l=10, r=40, t=10, b=10),
        xaxis_title="full-window mean IC (3M fwd return)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(148, 163, 184, 0.15)"), showlegend=False)
    st.plotly_chart(bar_fig, use_container_width=True)
    st.caption("A pie chart isn't shown here — IC is signed (some parents are "
               "currently negative, e.g. quality) and isn't a share of a fixed "
               "total, so slices would misrepresent rather than clarify. Blue = "
               "positive full-window mean IC, red = negative.")

    span = f"{rolling.index.min():%Y-%m} to {rolling.index.max():%Y-%m}"
    st.caption(
        f"Trailing {ROLL_WINDOW}-rebalance (~1yr) rolling mean of each parent's "
        f"point-in-time Spearman IC vs 3-month forward return, monthly "
        f"rebalances, {span}. Labeled '5-year' as the target window, but the "
        f"earliest reliable point-in-time parent scores only go back to "
        f"mid-2022 (see adj_close contamination fix), so history here is "
        f"capped at what's actually available (~3.75 years) rather than a "
        f"true 5y span. Source: output/factor_research/ic_monthly.csv "
        f"(`python run_factor_research.py` to refresh) — a research artifact, "
        f"not live production scores, so it can lag the current date.")

st.divider()

# ---------------------------------------------------------------------------
# Decile-spread validation (research/loop_research/decile_spread.py,
# 2026-09-09) -- does the composite score actually rank the universe, or is
# a high score no better than a low one?
# ---------------------------------------------------------------------------
st.markdown("### Does the score actually work? Decile spread")
st.caption("Full universe ranked by composite score each date into 10 "
          "equal-sized rank deciles (not raw score-value bins -- the score "
          "is a per-sector percentile with a large tied mass at 100, so an "
          "equal-COUNT split is the fair way to test this). A random "
          "10-stock book is drawn from each decile (same staggered-sleeve "
          "mechanics as the Monte Carlo page: 4 sleeves, 4-month hold, full "
          "turnover), 500 independent draws per decile. If the score is "
          "doing real work, decile 1 (highest scores) should clearly beat "
          "decile 10 (lowest). Research artifact "
          "(research/loop_research/decile_spread.py) -- not wired into the "
          "live production pipeline.")

decile_path = Path(__file__).resolve().parents[2] / "output" / "loop_research" / "decile_spread.json"
if decile_path.exists():
    import json as _json
    with decile_path.open() as fh:
        decile_data = _json.load(fh)
    deciles = sorted(decile_data, key=int)
    sharpe_vals = [decile_data[k]["sharpe"] for k in deciles]
    cagr_vals = [decile_data[k]["cagr"] for k in deciles]
    alpha_vals = [decile_data[k]["alpha"] for k in deciles]
    pct_beat_vals = [decile_data[k]["pct_sims_beating_spy"] for k in deciles]

    decile_fig = go.Figure()
    decile_fig.add_bar(x=deciles, y=sharpe_vals, name="Sharpe (median of 500 sims)",
                       marker_color="#2563eb")
    decile_fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="decile (1 = highest composite score, 10 = lowest)",
        yaxis_title="Sharpe (median across 500 random-draw sims)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="rgba(148, 163, 184, 0.15)"))
    st.plotly_chart(decile_fig, use_container_width=True)

    decile_table = pd.DataFrame({
        "Decile": [f"{k} (highest)" if k == "1" else f"{k} (lowest)" if k == "10" else k
                  for k in deciles],
        "CAGR": cagr_vals, "Sharpe": sharpe_vals, "Alpha (vs SPY, ann.)": alpha_vals,
        "% sims beating SPY": pct_beat_vals,
    }).set_index("Decile")
    st.dataframe(
        decile_table.style.format({
            "CAGR": "{:+.1%}", "Sharpe": "{:.3f}",
            "Alpha (vs SPY, ann.)": "{:+.1%}", "% sims beating SPY": "{:.0%}",
        }),
        use_container_width=True, height=42 + 35 * len(decile_table))
    st.caption("Read: deciles 1-2 (top 20%) clearly separate from the rest "
              "(positive alpha, beats SPY on 82-93% of sims); decile 10 "
              "(bottom 10%) is clearly worst on every metric. Deciles 3-9 "
              "are compressed and non-monotonic -- the score discriminates "
              "the extremes well but doesn't finely rank the middle of the "
              "universe. See research/loop_research/session_log.md for the "
              "full writeup.")
else:
    st.caption("No decile-spread results yet — run "
              "`python -m research.loop_research.decile_spread` first.")
