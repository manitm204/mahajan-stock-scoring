"""Price/calendar access for the backtester (separate from factor scoring).

Two concerns live here:

* the **trading calendar** and rebalance-date schedule, both derived from real
  ``daily_prices`` dates so every rebalance and every return is measured on a day
  the market was actually open; and
* a single **adjusted-close price matrix** (date x ticker) used to realize the
  forward returns the strategies earn between rebalances.

Scoring itself is *not* done here — the engine calls
:func:`factors.pipeline.score_universe` per rebalance date so the point-in-time
discipline stays in one place.
"""
from __future__ import annotations

import pandas as pd

from data.db import Database

SPY = "SPY"
QQQ = "QQQ"
BENCHMARKS = (SPY, QQQ)


def trading_calendar(db: Database) -> list[str]:
    """All distinct trading dates in ``daily_prices``, ascending."""
    df = db.query_df("SELECT DISTINCT date FROM daily_prices ORDER BY date")
    return df["date"].tolist()


def global_sectors(db: Database) -> "pd.Series":
    """GICS sector for **every** ticker in the ``universe`` table (current + departed),
    indexed by ticker, missing → ``Unknown``.

    Unlike :meth:`DataContext.sectors` (which reindexes to a single as-of universe), this
    covers the full point-in-time union — so sector-neutral construction over a
    survivorship-free backtest can place since-departed names in their real sector.
    """
    df = db.query_df("SELECT ticker, gics_sector FROM universe")
    s = df.set_index("ticker")["gics_sector"]
    return s.fillna("Unknown")


def load_price_matrix(
    db: Database, tickers: list[str], start: str, end: str
) -> pd.DataFrame:
    """Adjusted-close matrix (date x ticker) over ``[start, end]``.

    ``adj_close`` falls back to ``close`` (mirrors the scoring layer) so a name is
    only NaN on a date when it genuinely had no print. SPY and QQQ are always included
    (benchmarks for the walk-forward report).
    """
    want = sorted(set(tickers) | set(BENCHMARKS))
    placeholders = ",".join("?" * len(want))
    sql = (
        f"SELECT ticker, date, adj_close, close FROM daily_prices "
        f"WHERE ticker IN ({placeholders}) AND date >= ? AND date <= ?"
    )
    df = db.query_df(sql, (*want, start, end))
    if df.empty:
        return pd.DataFrame()
    df["px"] = df["adj_close"].fillna(df["close"])
    matrix = df.pivot_table(index="date", columns="ticker", values="px")
    return matrix.sort_index()


def generate_rebalance_dates(
    trading_dates: list[str], start: str, end: str, freq: str
) -> list[str]:
    """First trading date of each calendar period within ``[start, end]``.

    ``freq`` is ``weekly`` | ``monthly`` | ``quarterly``. Anchoring on the first
    open day of each period (rather than a fixed calendar day) guarantees every
    rebalance lands on a real trading date with prices available.
    """
    idx = pd.to_datetime(pd.Series(trading_dates))
    idx = idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
    if idx.empty:
        return []

    if freq == "weekly":
        iso = idx.dt.isocalendar()
        groups = list(zip(iso["year"], iso["week"]))
    elif freq == "monthly":
        groups = list(zip(idx.dt.year, idx.dt.month))
    elif freq == "quarterly":
        groups = list(zip(idx.dt.year, idx.dt.quarter))
    else:
        raise ValueError(f"Unknown rebalance frequency: {freq!r}")

    frame = pd.DataFrame({"date": idx.values, "grp": groups})
    firsts = frame.groupby("grp", sort=False)["date"].min().sort_values()
    return [pd.Timestamp(d).date().isoformat() for d in firsts]


def last_trading_on_or_before(trading_dates: list[str], date: str) -> str | None:
    """The latest trading date <= ``date`` (terminal settlement date)."""
    eligible = [d for d in trading_dates if d <= date]
    return eligible[-1] if eligible else None
