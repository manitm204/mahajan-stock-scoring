"""Assemble the weekly panel and define the buckets under test."""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from spystudy import signals as sg
from spystudy import stats as st
from spystudy.data import FWD_DAYS, forward_return, load_panels, weekly_dates

CACHE = Path("cache/spystudy_panel.pkl")
DEEP_CACHE = Path("cache/spystudy_panel_deep.pkl")
DEEP_START = "2005-01-01"


def _deep_panels() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Close, total-return and VIX panels from the 2004→ deep history."""
    from relvalue.prices import _adjust_ticker, load_dividends

    from spystudy import deep

    close, vix = deep.build()
    divs = load_dividends()
    divs = divs[divs["ticker"].isin(close.columns)].copy()
    divs["ex_date"] = pd.to_datetime(divs["ex_date"])
    tr = {}
    for t in close.columns:
        d = divs[divs["ticker"] == t].sort_values("ex_date")
        tr[t], _ = _adjust_ticker(close[t], d)
    return close, pd.DataFrame(tr), vix


def build_panel(rebuild: bool = False, deep: bool = False
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (weekly panel with signals + fwd return, daily close panel).

    ``deep=True`` uses the 2004→ FMP/FRED history and adds the VIX-derived
    signals; the default uses the production DB (2014→) and omits them.
    """
    cache = DEEP_CACHE if deep else CACHE
    if cache.exists() and not rebuild:
        return pickle.loads(cache.read_bytes())
    if deep:
        close, tr, vix = _deep_panels()
    else:
        close, tr = load_panels()
        vix = None
    sig = sg.build(close, vix)
    sig["fwd_1m"] = forward_return(tr["SPY"], FWD_DAYS)
    weekly = sig.reindex(weekly_dates(close.index))
    weekly.index.name = "date"
    if deep:
        # 2004 is warmup only; the study starts once the slowest signal is live
        weekly = weekly.loc[DEEP_START:]
    cache.write_bytes(pickle.dumps((weekly, close)))
    return weekly, close


# (key, column, predicate, ON label, OFF label, group, short label)
SPECS = [
    ("rsi_70", "rsi14", lambda s: s > 70, "RSI > 70", "RSI ≤ 70", "RSI",
     "RSI\n> 70"),
    ("corr21_045", "sector_corr_21d", lambda s: s < 0.45,
     "21d sector corr < 0.45", "≥ 0.45", "Sector correlation", "corr 21d\n< 0.45"),
    ("corr60_045", "sector_corr_60d", lambda s: s < 0.45,
     "60d sector corr < 0.45", "≥ 0.45", "Sector correlation", "corr 60d\n< 0.45"),
    ("trend_10", "trend_4m", lambda s: s >= 0.10, "4m trend ≥ 10%", "< 10%",
     "4-month trend", "trend\n≥ 10%"),
    ("trend_15", "trend_4m", lambda s: s >= 0.15, "4m trend ≥ 15%", "< 15%",
     "4-month trend", "trend\n≥ 15%"),
    ("trend_20", "trend_4m", lambda s: s >= 0.20, "4m trend ≥ 20%", "< 20%",
     "4-month trend", "trend\n≥ 20%"),
    ("ma200_5", "ma200_dist", lambda s: s >= 0.05, "Price ≥ 5% over MA200",
     "< 5%", "MA200 extension", "MA200\n+5%"),
    ("ma200_10", "ma200_dist", lambda s: s >= 0.10, "Price ≥ 10% over MA200",
     "< 10%", "MA200 extension", "MA200\n+10%"),
    ("ma200_15", "ma200_dist", lambda s: s >= 0.15, "Price ≥ 15% over MA200",
     "< 15%", "MA200 extension", "MA200\n+15%"),
    ("absorb_raw", "absorption_chg", lambda s: s < 0,
     "Absorption falling (raw 15d)", "rising", "Absorption", "absorb ↓\n(raw)"),
    ("absorb_std", "absorption_shift", lambda s: s <= -1.0,
     "Absorption shift ≤ −1σ", "> −1σ", "Absorption", "absorb ↓\n(≤ −1σ)"),
]

# only defined when the deep panel supplies VIX
IV_SPECS = [
    ("vrp_pos", "vrp", lambda s: s > 0, "IV > realised vol (VRP > 0)",
     "VRP ≤ 0", "Implied vol", "VRP\n> 0"),
    ("vrp_hi", "vrp", lambda s: s > 0.04, "VRP > 4 vol points", "≤ 4pts",
     "Implied vol", "VRP\n> 4pts"),
    ("ivr_30", "ivr", lambda s: s >= 30, "IV Rank ≥ 30", "< 30",
     "Implied vol", "IVR\n≥ 30"),
    ("ivr_50", "ivr", lambda s: s >= 50, "IV Rank ≥ 50", "< 50",
     "Implied vol", "IVR\n≥ 50"),
]


def specs_for(weekly: pd.DataFrame) -> list:
    """SPECS plus the IV buckets when the panel carries VIX."""
    return SPECS + (IV_SPECS if "vrp" in weekly.columns else [])


def run(weekly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, col, pred, on, off, group, short in specs_for(weekly):
        mask = pred(weekly[col]).where(weekly[col].notna())
        row = st.compare(mask, weekly["fwd_1m"], key, on, off)
        row["column"] = col
        row["group"] = group
        row["short"] = short
        rows.append(row)
    return st.table(rows)


def masks(weekly: pd.DataFrame) -> dict[str, pd.Series]:
    return {key: pred(weekly[col]).where(weekly[col].notna())
            for key, col, pred, _, _, _, _ in specs_for(weekly)}
