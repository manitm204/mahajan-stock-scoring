"""Stock Detail — everything the engine knows about one ticker.

Composite + parent + sub-factor scores, the score trend, the price chart,
and the full LLM research overlay (if one exists). Reached by clicking
"Details →" on a Stocks card, or by picking a ticker directly below.
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
from dashboard.candidates import PARENT_FACTORS
from dashboard.components import (
    STATUS_SEVERITY, empty_state, kpi_row, page_header, pill_row, sidebar,
    status_pill,
)

st.set_page_config(page_title="Stock Detail", layout="wide", page_icon="🔎")
sidebar()

FACTOR_COLORS = {
    "growth":        "#2563eb",
    "quality":       "#7c3aed",
    "value":         "#16a34a",
    "momentum":      "#d97706",
    "revisions":     "#0891b2",
    "insider":       "#db2777",
    "institutional": "#65a30d",
    "short":         "#dc2626",
}


def _fmt(x, digits=1):
    return "—" if x is None or pd.isna(x) else f"{x:.{digits}f}"


# ---------------------------------------------------------------------------
# Resolve which ticker to show
# ---------------------------------------------------------------------------
frame = ddata.load_screener()
if frame.empty:
    page_header("Stock Detail")
    empty_state("No composite scores in the warehouse yet.",
                "Run `python run_scoring.py` first.")
    st.stop()

all_tickers = sorted(frame["ticker"].unique())
name_by_ticker = (frame.drop_duplicates("ticker")
                  .set_index("ticker")["company_name"].fillna(""))

# The selectbox's session-state key is stable across reruns/navigations, so
# a plain `index=` default is only honored on the very first render ever
# (Streamlit keeps whatever the widget last held otherwise). We adopt the
# query param into session state on a genuinely NEW navigation (e.g. the
# Stocks page's "Full detail" button) — but only then. Every other rerun on
# this page (picking the dropdown, typing a search) already writes the
# query param to match the fresh selection below, so re-adopting it here
# unconditionally would just snap the widget straight back to whatever the
# URL held *before* that click and silently undo it.
SELECT_KEY = "stock_detail_ticker"
LAST_QP_KEY = "stock_detail_last_seen_qp"
SEARCH_KEY = "stock_detail_search"
qp_ticker = st.query_params.get("ticker")
if qp_ticker:
    qp_ticker = qp_ticker.upper()

if (qp_ticker and qp_ticker in all_tickers
        and qp_ticker != st.session_state.get(LAST_QP_KEY)):
    st.session_state[SELECT_KEY] = qp_ticker
    st.session_state[LAST_QP_KEY] = qp_ticker
    # Leftover search text would otherwise re-match on the next rerun and
    # immediately fight this navigation back to whatever it was searching.
    st.session_state[SEARCH_KEY] = ""
elif SELECT_KEY not in st.session_state:
    st.session_state[SELECT_KEY] = all_tickers[0]
    st.session_state[LAST_QP_KEY] = st.session_state[SELECT_KEY]

search = st.text_input("Search", placeholder="🔎 Search ticker or company name, then press Enter…",
                       key=SEARCH_KEY, label_visibility="collapsed")
if search.strip():
    q = search.strip().upper()
    matches = [t for t in all_tickers
              if q in t or q in name_by_ticker.get(t, "").upper()]
    if matches:
        # Best match first: exact ticker > ticker prefix > substring/name hit.
        matches.sort(key=lambda t: (t != q, not t.startswith(q), t))
        # Jump straight to the top match if the search no longer matches
        # whatever was previously selected — that's the whole point of
        # searching from here: type a ticker, land on its detail page.
        if st.session_state[SELECT_KEY] not in matches:
            st.session_state[SELECT_KEY] = matches[0]
    else:
        matches = all_tickers
else:
    matches = all_tickers


def _label(t: str) -> str:
    name = name_by_ticker.get(t, "")
    return f"{t} — {name}" if name else t


picked = st.selectbox("Ticker", matches, key=SELECT_KEY,
                      format_func=_label, label_visibility="collapsed")
if st.query_params.get("ticker") != picked:
    st.query_params["ticker"] = picked
st.session_state[LAST_QP_KEY] = picked
ticker = picked

row = frame[frame["ticker"] == ticker].iloc[0]
as_of = row["as_of_date"]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
page_header(f"{ticker} — {row['company_name'] or ''}",
            f"{row['sector'] or '—'} · {row['industry'] or '—'}", as_of)

pills = []
status = row["research_status"]
if pd.notna(status):
    pills.append((str(status), STATUS_SEVERITY.get(str(status), "info")))
else:
    pills.append(("NO LLM", "muted"))
flag = row.get("long_short_flag")
if pd.notna(flag) and flag:
    pills.append((str(flag), "info" if flag == "LONG" else "warn"))
st.markdown(pill_row(pills), unsafe_allow_html=True)
st.write("")

kpi_row([
    ("Price", f"${_fmt(row['price'], 2)}"),
    ("Composite score", _fmt(row["composite_score"])),
    ("Sector rank", "—" if pd.isna(row["sector_rank"]) else f"{int(row['sector_rank'])}"),
    ("LLM confidence", _fmt(row.get("overlay_confidence"), 2)),
])

st.divider()

# ---------------------------------------------------------------------------
# Price chart — candlestick, 5y history
# ---------------------------------------------------------------------------
st.markdown("### Price")
prices = ddata.load_price_history(ticker, n_days=1260)
if prices.empty:
    empty_state("No price history for this ticker.")
else:
    fig = go.Figure()
    fig.add_candlestick(x=prices["date"], open=prices["open"], high=prices["high"],
                        low=prices["low"], close=prices["close"], name=ticker,
                        increasing_line_color="#16a34a", decreasing_line_color="#dc2626",
                        increasing_fillcolor="#16a34a", decreasing_fillcolor="#dc2626")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_title=None, xaxis_title=None, showlegend=False,
                      xaxis_rangeslider_visible=False, hovermode="x unified")
    fig.update_xaxes(
        rangeselector=dict(buttons=[
            dict(count=1, label="1M", step="month", stepmode="backward"),
            dict(count=3, label="3M", step="month", stepmode="backward"),
            dict(count=6, label="6M", step="month", stepmode="backward"),
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(count=3, label="3Y", step="year", stepmode="backward"),
            dict(count=5, label="5Y", step="year", stepmode="backward"),
            dict(step="all", label="All"),
        ]))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# P/E ratio — trailing (TTM), 5y history
# ---------------------------------------------------------------------------
st.markdown("### P/E ratio (trailing twelve months)")
pe_hist = ddata.load_pe_history(ticker, n_days=1260)
if pe_hist.empty:
    empty_state("Not enough quarterly EPS history to compute a P/E series.",
               "Needs 4+ reported quarters of diluted EPS.")
else:
    fig = go.Figure()
    fig.add_scatter(x=pe_hist["date"], y=pe_hist["pe"], mode="lines",
                    line=dict(color="#7c3aed", width=2), name="P/E",
                    fill="tozeroy", fillcolor="rgba(124,58,237,0.08)")
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_title=None, xaxis_title=None, showlegend=False,
                      hovermode="x unified")
    fig.update_xaxes(
        rangeselector=dict(buttons=[
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(count=3, label="3Y", step="year", stepmode="backward"),
            dict(count=5, label="5Y", step="year", stepmode="backward"),
            dict(step="all", label="All"),
        ]))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Close ÷ trailing-twelve-month diluted EPS. Each quarter's EPS is "
              f"admitted {ddata.REPORTING_LAG_DAYS} days after its fiscal period "
              "end (conservative filing-lag convention) so the series never "
              "looks ahead to unreported earnings.")

st.divider()

# ---------------------------------------------------------------------------
# Composite score trend
# ---------------------------------------------------------------------------
st.markdown("### Composite score trend")
score_hist = ddata.load_score_history(ticker)
if score_hist.empty:
    empty_state("No score history for this ticker.")
else:
    fig = go.Figure()
    fig.add_scatter(x=score_hist["as_of_date"], y=score_hist["composite_score"],
                    mode="lines", line=dict(color="#d97706", width=2),
                    name="Composite")
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_range=[0, 100], showlegend=False,
                      hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Parent factor scores — current snapshot + history
# ---------------------------------------------------------------------------
st.markdown("### Parent factors")
left, right = st.columns([1, 1.3])

facs = {f: row.get(f) for f in PARENT_FACTORS}
fac_series = pd.Series(facs).dropna().sort_values()

with left:
    if fac_series.empty:
        empty_state("No parent factor scores for this ticker.")
    else:
        fig = go.Figure()
        fig.add_bar(x=fac_series.values, y=list(fac_series.index),
                    orientation="h",
                    marker_color=[FACTOR_COLORS.get(f, "#2563eb")
                                  for f in fac_series.index],
                    text=[f"{v:.1f}" for v in fac_series.values],
                    textposition="outside")
        fig.update_layout(height=40 + 38 * len(fac_series),
                          margin=dict(l=10, r=30, t=10, b=10),
                          xaxis_range=[0, 105], showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with right:
    parent_hist = ddata.load_parent_factor_history(ticker)
    if parent_hist.empty or len(parent_hist) < 2:
        empty_state("Not enough history to chart parent-factor trends yet.")
    else:
        fig = go.Figure()
        for f in PARENT_FACTORS:
            if f not in parent_hist.columns:
                continue
            fig.add_scatter(x=parent_hist["as_of_date"], y=parent_hist[f],
                            mode="lines", name=f.title(),
                            line=dict(color=FACTOR_COLORS.get(f, "#2563eb"),
                                      width=2))
        fig.update_layout(height=40 + 38 * len(fac_series) if not fac_series.empty
                          else 320,
                          margin=dict(l=10, r=10, t=10, b=10),
                          yaxis_range=[0, 100], hovermode="x unified",
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Sub-factor scores — the production formula, broken down
# ---------------------------------------------------------------------------
st.markdown("### Sub-factors")
sub_scores = ddata.load_sub_factor_scores(ticker, as_of)
selected_subs = ddata.load_selected_subs()

if sub_scores.empty:
    empty_state("No sub-factor scores for this ticker at this date.")
else:
    sub_by_factor = {f: g for f, g in sub_scores.groupby("factor")}
    for f in PARENT_FACTORS:
        g = sub_by_factor.get(f)
        if g is None or g.empty:
            continue
        sub_w = selected_subs.get(f, {})
        g = g.assign(sub_weight=g["sub_factor"].map(sub_w)).sort_values(
            "sub_weight", ascending=False, na_position="last")
        with st.expander(f"{f.title()}  ·  score {_fmt(facs.get(f))}"):
            st.dataframe(
                g[["sub_factor", "score", "sub_weight", "raw_value"]]
                .rename(columns={"sub_factor": "subfactor",
                                 "sub_weight": "weight in parent"})
                .style.format({"score": "{:.1f}", "weight in parent": "{:.3f}",
                              "raw_value": "{:.4f}"}, na_rep="—")
                .background_gradient(cmap="RdYlGn", subset=["score"],
                                    vmin=0, vmax=100),
                use_container_width=True, hide_index=True,
                height=42 + 35 * len(g))

st.divider()

# ---------------------------------------------------------------------------
# LLM research overlay
# ---------------------------------------------------------------------------
st.markdown("### LLM analysis")
overlay = ddata.load_overlay_full(ticker)
if overlay is None:
    empty_state("This ticker hasn't been through the LLM overlay yet.",
               "Run `python run_analysis.py` to cover it.")
else:
    st.caption(f"{overlay.get('model') or '—'} · as of {overlay.get('as_of_date')} "
              f"· computed {overlay.get('computed_at') or '—'}")

    qual_pills = []
    if overlay.get("qualitative_risk_level"):
        qual_pills.append((f"risk: {overlay['qualitative_risk_level']}",
                           STATUS_SEVERITY.get(overlay["qualitative_risk_level"], "warn")))
    if overlay.get("quant_signal_review"):
        qual_pills.append((f"signal: {overlay['quant_signal_review']}", "info"))
    if qual_pills:
        st.markdown(pill_row(qual_pills), unsafe_allow_html=True)
        st.write("")

    fields = [
        ("Thesis alignment", "thesis_alignment"),
        ("Business quality", "business_quality"),
        ("Management tone", "management_tone"),
        ("Competitive position", "competitive_position"),
        ("Accounting risk", "accounting_risk"),
        ("Filing risk", "filing_risk"),
        ("Insider signal interpretation", "insider_signal_interpretation"),
    ]
    populated = [(label, overlay.get(key)) for label, key in fields if overlay.get(key)]
    if populated:
        cols = st.columns(2)
        for i, (label, val) in enumerate(populated):
            with cols[i % 2]:
                st.markdown(f"**{label}**")
                st.write(val)

    ev_l, ev_r = st.columns(2)
    with ev_l:
        if overlay.get("confirming_evidence"):
            st.markdown("**Confirming evidence**")
            for e in overlay["confirming_evidence"]:
                st.markdown(f"- {e}")
        if overlay.get("red_flags"):
            st.markdown("**Red flags**")
            for f in overlay["red_flags"]:
                st.markdown(f"- {f}")
    with ev_r:
        if overlay.get("contradicting_evidence"):
            st.markdown("**Contradicting evidence**")
            for e in overlay["contradicting_evidence"]:
                st.markdown(f"- {e}")
        if overlay.get("open_questions"):
            st.markdown("**Open questions**")
            for q in overlay["open_questions"]:
                st.markdown(f"- {q}")
