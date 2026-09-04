"""Shared widgets used on every page.

* :func:`page_header` - title + subtitle + as-of stamp
* :func:`kpi_row` - row of "metric card" tiles
* :func:`status_pill` - colourful tag (PASS / REVIEW / ...)
* :func:`sidebar` - sidebar nav (refresh + 3-page links)
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import streamlit as st

from . import data as ddata

SEVERITY_COLOURS = {
    "ok":      "#16a34a",
    "info":    "#2563eb",
    "warn":    "#d97706",
    "warning": "#d97706",
    "error":   "#dc2626",
    "muted":   "#475569",
}

# research_status → pill severity
STATUS_SEVERITY = {
    "PASS": "ok",
    "WATCHLIST": "info",
    "REVIEW": "warn",
    "AVOID_RED_FLAG": "error",
}


def page_header(title: str, subtitle: str = "", as_of: str | None = None) -> None:
    st.markdown(f"## {title}")
    sub_bits = []
    if subtitle:
        sub_bits.append(subtitle)
    if as_of:
        sub_bits.append(f"as of {as_of}")
    if sub_bits:
        st.caption("  ·  ".join(sub_bits))


def kpi_row(metrics: list[tuple]) -> None:
    """Render ``len(metrics)`` metric tiles in a single row.

    Each item is either ``(label, value)`` or ``(label, value, delta)``.
    """
    if not metrics:
        return
    cols = st.columns(len(metrics))
    for col, item in zip(cols, metrics):
        if len(item) == 2:
            label, value = item
            caption = None
        else:
            label, value, caption = item[0], item[1], item[2]
        with col:
            st.metric(label=label, value=value, delta=caption or None)


def status_pill(label: str, severity: str = "info") -> str:
    colour = SEVERITY_COLOURS.get(severity.lower(), "#2563eb")
    return (
        f'<span style="background:{colour};color:#fff;padding:3px 10px;'
        f'border-radius:999px;font-size:0.78rem;font-weight:600;'
        f'letter-spacing:0.04em;">{label}</span>'
    )


def pill_row(items: Iterable[tuple[str, str]]) -> str:
    """``items``: (label, severity) — returns joined pill HTML."""
    return " ".join(status_pill(label, sev) for label, sev in items)


# ---------------------------------------------------------------------------
# Responsive / mobile styling
# ---------------------------------------------------------------------------
_RESPONSIVE_CSS = """
<style>
@media (max-width: 640px) {
  .block-container {
    padding: 2.5rem 0.9rem 3rem 0.9rem !important;
  }
  div[data-testid="stHorizontalBlock"] {
    flex-wrap: wrap !important;
  }
  div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
    flex: 1 1 100% !important;
    width: 100% !important;
    min-width: 100% !important;
  }
  div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:has(div[data-testid="stMetric"]) {
    flex: 1 1 42% !important;
    width: auto !important;
    min-width: 42% !important;
  }
  div[data-testid="stMetricValue"] { font-size: 1.3rem !important; }
  div[data-testid="stMetricLabel"] { font-size: 0.75rem !important; }
  div[data-testid="stDataFrame"],
  div[data-testid="stTable"] { overflow-x: auto !important; }
  h1 { font-size: 1.6rem !important; }
  h2 { font-size: 1.3rem !important; }
  h3 { font-size: 1.1rem !important; }
}
</style>
"""


def inject_responsive_css() -> None:
    st.markdown(_RESPONSIVE_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def sidebar() -> None:
    inject_responsive_css()
    with st.sidebar:
        st.markdown("### Mahajan Hedge Fund")
        st.caption("Read-only dashboard")

        if "_last_refresh" not in st.session_state:
            st.session_state["_last_refresh"] = _now()

        if st.button("Refresh data", use_container_width=True):
            ddata.refresh_all()
            st.session_state["_last_refresh"] = _now()
            st.rerun()
        st.caption(f"Last refresh: {st.session_state['_last_refresh']}")

        st.divider()
        st.markdown("**Pages**")
        try:
            st.page_link("pages/1_Stocks.py",       label="🃏  Stocks")
            st.page_link("pages/2_Portfolio.py",    label="📊  Portfolio")
            st.page_link("pages/3_Scoring.py",      label="⚖️  Scoring")
            st.page_link("pages/4_Stock_Detail.py", label="🔎  Stock Detail")
        except KeyError:
            # AppTest runs a page file standalone, so the multipage registry
            # (url_pathname) isn't populated — links only exist in a real run.
            st.caption("Stocks · Portfolio · Scoring · Stock Detail")

        st.divider()
        st.markdown(
            "<small style='color:#94a3b8;'>Engines decide.  Dashboard explains."
            "</small>",
            unsafe_allow_html=True,
        )


def empty_state(message: str, hint: str = "") -> None:
    st.info(message)
    if hint:
        st.caption(hint)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
