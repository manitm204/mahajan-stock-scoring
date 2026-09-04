"""Streamlit entry point.

Run with::

    streamlit run dashboard/app.py

The home view simply forwards to the Stocks screener — the dashboard is
three pages (Stocks / Portfolio / Scoring), nothing else.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
# Force project root ahead of Streamlit's main-script dir on sys.path, else
# `dashboard/data.py` shadows the top-level `data` package on cold page loads.
if sys.path[:1] != [_ROOT]:
    if _ROOT in sys.path:
        sys.path.remove(_ROOT)
    sys.path.insert(0, _ROOT)

import streamlit as st

st.set_page_config(
    page_title="Mahajan Hedge Fund",
    layout="wide",
    page_icon="📡",
    initial_sidebar_state="expanded",
)

st.switch_page("pages/1_Stocks.py")
