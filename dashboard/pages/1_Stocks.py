"""Stocks — card-grid screener over every scored ticker.

Best composite scores first (default), with sector / LLM-coverage /
side filters, a ticker search box, and a cards-per-row selector. Each
card shows the composite, the LLM overlay verdict if one exists, and an
expander with parent-factor scores + overlay detail.
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
import streamlit as st

from dashboard import data as ddata
from dashboard.candidates import PARENT_FACTORS
from dashboard.components import (
    STATUS_SEVERITY, empty_state, page_header, pill_row, sidebar, status_pill,
)

st.set_page_config(page_title="Stocks", layout="wide", page_icon="🃏")
sidebar()

frame = ddata.load_screener()
if frame.empty:
    page_header("Stocks")
    empty_state("No composite scores in the warehouse yet.",
                "Run `python run_scoring.py` first.")
    st.stop()

as_of = frame["as_of_date"].iloc[0]
n_covered = int(frame["research_status"].notna().sum())
page_header("Stocks",
            f"{len(frame)} scored · {n_covered} with LLM analysis", as_of)

# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns([2.2, 2.2, 1.6, 1.4, 1.1])
with c1:
    query = st.text_input("Search", placeholder="Ticker or company name…",
                          label_visibility="collapsed")
with c2:
    sectors = st.multiselect("Sectors", sorted(frame["sector"].dropna().unique()),
                             placeholder="All sectors",
                             label_visibility="collapsed")
with c3:
    coverage = st.selectbox("Coverage", ["All stocks", "LLM-analyzed only",
                                         "Not yet analyzed"],
                            label_visibility="collapsed")
with c4:
    sort = st.selectbox("Sort", ["Best score", "Worst score", "LLM first",
                                 "Market cap (high→low)",
                                 "Market cap (low→high)"],
                        label_visibility="collapsed")
with c5:
    per_row = st.selectbox("Per row", [2, 3, 4, 5], index=1,
                           label_visibility="collapsed")

MCAP_BUCKETS = {
    "Mega (≥$200B)": (200e9, float("inf")),
    "Large ($10B–$200B)": (10e9, 200e9),
    "Mid ($2B–$10B)": (2e9, 10e9),
    "Small (<$2B)": (0, 2e9),
}
d1, d2 = st.columns([2.2, 2.2])
with d1:
    statuses = st.multiselect(
        "LLM verdict", ["PASS", "REVIEW", "WATCHLIST", "AVOID_RED_FLAG"],
        placeholder="All LLM verdicts", label_visibility="collapsed")
with d2:
    mcap_buckets = st.multiselect(
        "Market cap", list(MCAP_BUCKETS), placeholder="All market caps",
        label_visibility="collapsed")

view = frame.copy()
if query:
    q = query.strip().upper()
    view = view[view["ticker"].str.upper().str.contains(q, na=False)
                | view["company_name"].fillna("").str.upper().str.contains(q)]
if sectors:
    view = view[view["sector"].isin(sectors)]
if coverage == "LLM-analyzed only":
    view = view[view["research_status"].notna()]
elif coverage == "Not yet analyzed":
    view = view[view["research_status"].isna()]
if statuses:
    view = view[view["research_status"].isin(statuses)]
if mcap_buckets:
    ranges = [MCAP_BUCKETS[b] for b in mcap_buckets]
    mask = pd.Series(False, index=view.index)
    for lo, hi in ranges:
        mask |= view["market_cap"].between(lo, hi, inclusive="left")
    view = view[mask]

if sort == "Worst score":
    view = view.sort_values("composite_score")
elif sort == "LLM first":
    view = view.assign(_c=view["research_status"].notna()).sort_values(
        ["_c", "composite_score"], ascending=[False, False]).drop(columns="_c")
elif sort == "Market cap (high→low)":
    view = view.sort_values("market_cap", ascending=False)
elif sort == "Market cap (low→high)":
    view = view.sort_values("market_cap", ascending=True)
else:
    view = view.sort_values("composite_score", ascending=False)

# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------
PAGE_SIZE = 60
n_pages = max(1, -(-len(view) // PAGE_SIZE))
if n_pages > 1:
    page = st.selectbox(
        f"{len(view)} matches",
        range(1, n_pages + 1),
        format_func=lambda p: f"Page {p} · ranks {(p-1)*PAGE_SIZE+1}–{min(p*PAGE_SIZE, len(view))}",
    )
else:
    page = 1
    st.caption(f"{len(view)} matches")
view = view.iloc[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------
def _fmt(x, digits=1):
    return "—" if x is None or pd.isna(x) else f"{x:.{digits}f}"


def _fmt_mcap(x):
    if x is None or pd.isna(x):
        return "—"
    if x >= 1e12:
        return f"${x / 1e12:.2f}T"
    if x >= 1e9:
        return f"${x / 1e9:.1f}B"
    return f"${x / 1e6:.0f}M"


def render_card(row: pd.Series) -> None:
    with st.container(border=True):
        top_l, top_r = st.columns([3, 1.2])
        with top_l:
            name = row["company_name"] or ""
            st.markdown(f"**{row['ticker']}** &nbsp; "
                        f"<span style='color:#94a3b8;font-size:0.85rem;'>{name}</span>",
                        unsafe_allow_html=True)
            st.caption(f"{row['sector'] or '—'} · ${_fmt(row['price'], 2)} · "
                       f"{_fmt_mcap(row.get('market_cap'))}")
        with top_r:
            st.markdown(
                f"<div style='text-align:right;font-size:1.45rem;font-weight:700;'>"
                f"{_fmt(row['composite_score'])}</div>"
                f"<div style='text-align:right;color:#94a3b8;font-size:0.7rem;'>"
                f"composite</div>", unsafe_allow_html=True)

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

        if st.button("🔎 Full detail", key=f"detail_{row['ticker']}",
                     use_container_width=True):
            # st.switch_page clears all query params unless given its own
            # query_params=, so set-then-switch would lose the ticker.
            st.switch_page("pages/4_Stock_Detail.py",
                           query_params={"ticker": row["ticker"]})

        with st.expander("Details"):
            facs = {f: row.get(f) for f in PARENT_FACTORS}
            fac_df = (pd.Series(facs, name="score").rename_axis("factor")
                      .to_frame().dropna())
            if not fac_df.empty:
                st.dataframe(
                    fac_df.style.format("{:.1f}").background_gradient(
                        cmap="RdYlGn", vmin=0, vmax=100),
                    use_container_width=True, height=min(320, 40 + 35 * len(fac_df)))
            if pd.notna(status):
                st.caption(
                    f"LLM view · {row.get('quant_signal_review') or '—'} · "
                    f"risk {row.get('qualitative_risk_level') or '—'} · "
                    f"confidence {_fmt(row.get('overlay_confidence'), 2)}")
                detail = ddata.load_overlay_detail(row["ticker"]) or {}
                flags = detail.get("red_flags") or []
                if flags:
                    st.markdown("**Red flags**")
                    for f in flags[:5]:
                        st.markdown(f"- {f}")
                qs = detail.get("open_questions") or []
                if qs:
                    st.markdown("**Open questions**")
                    for q in qs[:3]:
                        st.markdown(f"- {q}")


rows = [view.iloc[i:i + per_row] for i in range(0, len(view), per_row)]
for chunk in rows:
    cols = st.columns(per_row)
    for col, (_, row) in zip(cols, chunk.iterrows()):
        with col:
            render_card(row)
