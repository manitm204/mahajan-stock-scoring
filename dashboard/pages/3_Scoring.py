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
