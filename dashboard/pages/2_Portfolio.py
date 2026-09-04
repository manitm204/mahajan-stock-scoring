"""Portfolio — the ratified model book, computed live from today's scores.

Construction: top-25% of the scored universe by composite, cap-weighted
with the 5% waterfill cap, then sector-scaled to the 50/50 SPY/QQQ blend
targets (engine mode ``blend_match``, ratified 2026-08-07). Charts show
how the book's sector mix sits against the blend / SPY / QQQ targets and
what the individual holdings look like.
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

st.set_page_config(page_title="Portfolio", layout="wide", page_icon="📊")
sidebar()

model = ddata.load_model_book()
book: pd.DataFrame = model["book"]
targets: pd.DataFrame = model["targets"]

if book.empty:
    page_header("Portfolio")
    empty_state("No composite scores yet — the model book needs a scoring run.",
                "Run `python run_scoring.py` first.")
    st.stop()

page_header("Portfolio",
            "Model book · top-25% composite · cap5 weights · 50/50 SPY-QQQ sector blend",
            model["as_of"])

eff_n = 1.0 / float((book["weight"] ** 2).sum())
kpi_row([
    ("Holdings", f"{len(book)}"),
    ("Top position", f"{book.iloc[0]['ticker']} · {book.iloc[0]['weight']:.1%}"),
    ("Top-10 weight", f"{book['weight'].head(10).sum():.1%}"),
    ("Effective names", f"{eff_n:.0f}"),
    ("Sectors held", f"{book['sector'].nunique()}"),
])

st.divider()

# ---------------------------------------------------------------------------
# Sector allocation vs targets
# ---------------------------------------------------------------------------
st.markdown("### Sector allocation")
show_refs = st.toggle("Show SPY / QQQ reference targets", value=False)

fig = go.Figure()
fig.add_bar(name="Book", x=targets.index, y=targets["book"],
            marker_color="#2563eb")
fig.add_bar(name="Blend target (50/50)", x=targets.index,
            y=targets["blend_target"], marker_color="#16a34a")
if show_refs:
    fig.add_bar(name="SPY-style target", x=targets.index,
                y=targets["spy_target"], marker_color="#94a3b8")
    fig.add_bar(name="QQQ-style target", x=targets.index,
                y=targets["qqq_target"], marker_color="#d97706")
fig.update_layout(barmode="group", height=380, yaxis_tickformat=".0%",
                  margin=dict(l=10, r=10, t=10, b=10),
                  legend=dict(orientation="h", y=1.08))
st.plotly_chart(fig, use_container_width=True)
st.caption("The book tracks the blend target by construction; residual gaps are "
           "sectors where the top-25% selection holds no (or few) names.")

st.divider()

# ---------------------------------------------------------------------------
# Holdings
# ---------------------------------------------------------------------------
st.markdown("### Holdings")
left, right = st.columns([1.4, 1])

with left:
    top_n = st.slider("Chart top N", 10, len(book), min(30, len(book)), step=5)
    top = book.head(top_n).iloc[::-1]
    hfig = go.Figure(go.Bar(
        x=top["weight"], y=top["ticker"], orientation="h",
        marker_color="#2563eb",
        customdata=top[["company_name", "sector"]],
        hovertemplate="<b>%{y}</b> %{customdata[0]}<br>"
                      "%{customdata[1]} · %{x:.2%}<extra></extra>"))
    hfig.update_layout(height=max(340, 16 * top_n),
                       xaxis_tickformat=".1%",
                       margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(hfig, use_container_width=True)

with right:
    sfig = go.Figure(go.Pie(
        labels=book["sector"], values=book["weight"], hole=0.45,
        textinfo="label+percent", textposition="inside"))
    sfig.update_layout(height=340, showlegend=False,
                       margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(sfig, use_container_width=True)
    st.caption("Book weight by sector")

table = book.assign(weight_pct=book["weight"] * 100)[
    ["ticker", "company_name", "sector", "weight_pct", "composite_score", "price"]]
st.dataframe(
    table.style.format({"weight_pct": "{:.2f}%", "composite_score": "{:.1f}",
                        "price": "${:,.2f}"}),
    use_container_width=True, height=420, hide_index=True)

st.download_button("Download holdings CSV",
                   table.to_csv(index=False).encode(),
                   file_name=f"model_book_{model['as_of']}.csv",
                   mime="text/csv")
