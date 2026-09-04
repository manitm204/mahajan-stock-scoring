"""Per-fund panels for the SPY / QQQ / IWM comparison.

Each fund is scored with its own price signals (RSI, trend, MA200), its own
volatility index (VIX / VXN / RVX), its rolling beta to SPY, and the shared
market-wide regime signals (sector correlation, absorption) which are the same
series for every fund by construction.

Two forward returns are carried:
  ``fwd_abs``  the fund's own next-month total return
  ``fwd_rel``  that minus SPY's, i.e. the rotation question
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from spystudy import signals as sg
from spystudy.data import FWD_DAYS, forward_return, weekly_dates
from spystudy.deep import FUNDS, build_multi

CACHE = Path("cache/spystudy_multi_panels.pkl")
START = "2005-01-01"


def _tr_panel(close: pd.DataFrame) -> pd.DataFrame:
    from relvalue.prices import _adjust_ticker, load_dividends
    divs = load_dividends()
    divs = divs[divs["ticker"].isin(close.columns)].copy()
    divs["ex_date"] = pd.to_datetime(divs["ex_date"])
    out = {}
    for t in close.columns:
        d = divs[divs["ticker"] == t].sort_values("ex_date")
        out[t], _ = _adjust_ticker(close[t], d)
    return pd.DataFrame(out)


def build_panels(rebuild: bool = False) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Return ({fund: weekly panel}, daily sector close panel)."""
    if CACHE.exists() and not rebuild:
        return pickle.loads(CACHE.read_bytes())
    close, vols = build_multi()
    tr = _tr_panel(close)
    wk = weekly_dates(close.index)

    # market-wide signals are identical across funds; compute them once
    shared = pd.DataFrame({
        "sector_corr_21d": sg.sector_correlation(close, 21),
        "sector_corr_60d": sg.sector_correlation(close, 60),
    })
    ar = sg.absorption_ratio(close)
    shared["absorption"] = ar
    shared["absorption_shift"] = sg.absorption_shift(ar)
    shared["absorption_chg"] = sg.absorption_raw_change(ar)

    spy_fwd = forward_return(tr["SPY"], FWD_DAYS)
    panels = {}
    for fund in FUNDS:
        px, v = close[fund], vols[fund]
        d = shared.copy()
        d["rsi14"] = sg.rsi(px)
        d["trend_4m"] = sg.trend(px)
        d["ma200_dist"] = sg.ma_distance(px)
        d["vix"] = v
        d["realized_vol"] = sg.realized_vol(px)
        d["vrp"] = sg.variance_risk_premium(v, px)
        d["ivr"] = sg.iv_rank(v)
        d["beta_60"] = sg.beta_to(px, close["SPY"])
        d["fwd_abs"] = forward_return(tr[fund], FWD_DAYS)
        d["fwd_rel"] = d["fwd_abs"] - spy_fwd
        w = d.reindex(wk).loc[START:]
        w.index.name = "date"
        panels[fund] = w
    CACHE.write_bytes(pickle.dumps((panels, close)))
    return panels, close
