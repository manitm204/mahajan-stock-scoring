"""Price data for the SPY signal study.

Two series per ticker, deliberately:

* ``close``  — retroactively split-adjusted, price-only. This is the chart a
  trader actually looks at, so all *signals* (RSI, MA200, trend) are computed
  on it.
* ``tr``     — close backward-adjusted for cash dividends. This is what you
  actually earn, so all *forward returns* are computed on it.

Both come from ``relvalue.prices`` primitives rather than ``adj_close``, which
mixes dividend conventions by source era in this DB (see that module's header).
The cached relvalue panel covers 728 tickers and goes stale; we rebuild just
the 12 tickers we need straight from the DB so the study is always current.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from relvalue.prices import DB_PATH, _adjust_ticker, load_close_panel, load_dividends

SPY = "SPY"

# The 11 GICS sector SPDRs that partition the S&P 500.
SECTORS = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLC", "XLY", "XLP", "XLB", "XLRE", "XLU"]

SECTOR_NAMES = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLC": "Comm. Services",
    "XLY": "Cons. Discretionary",
    "XLP": "Cons. Staples",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
}

TICKERS = [SPY] + SECTORS

FWD_DAYS = 21  # one month of trading days


def load_panels(db_path: Path = DB_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (close, total_return) panels indexed by date, columns = TICKERS."""
    close = load_close_panel(db_path)[TICKERS]
    close.index = pd.to_datetime(close.index)
    # the pivot spans every date any of the DB's 700+ tickers traded; keep only
    # real SPY sessions so the trading-day grid is the US equity calendar
    close = close.loc[close[SPY].notna()]

    divs = load_dividends(db_path)
    divs = divs[divs["ticker"].isin(TICKERS)].copy()
    divs["ex_date"] = pd.to_datetime(divs["ex_date"])

    tr = {}
    for t in TICKERS:
        d = divs[divs["ticker"] == t].sort_values("ex_date")
        tr[t], _ = _adjust_ticker(close[t], d)
    return close, pd.DataFrame(tr)


def weekly_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each calendar week present in ``index``."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.resample("W-FRI").last().dropna().values)


def forward_return(tr: pd.Series, days: int = FWD_DAYS) -> pd.Series:
    """Total return over the next ``days`` trading days, NaN where truncated."""
    return tr.shift(-days) / tr - 1.0
