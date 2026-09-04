"""Read-only data access for the VIX-relative overlay study.

Everything here *loads* existing artefacts — the cached PIT candidate panel
(built by the prior walk-forward studies), adjusted-close prices, GICS sectors
and the VIX history from the Layer-1 database. Nothing in production or
``research/`` is mutated. The price matrix is cached locally under
``cache/vix_relative_overlay/`` so repeat runs skip the DB scan.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data.db import get_db

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache" / "vix_relative_overlay"
PANEL_PICKLE = (ROOT / "cache" / "subfactor_expansion" /
                "cand_panel_2015-06-30_2026-06-30_monthly_v2.pkl")
BENCHMARK = "SPY"


@dataclass
class StudyPanel:
    """The slice of the cached PIT candidate panel this study needs.

    ``scores`` maps rebalance date → (ticker × sub-factor) frame on the 0-100
    sector-relative scale; the frame index is the point-in-time membership of
    that date. ``universe`` is the PIT union across all rebalances.
    """

    rebal_dates: list[str]
    scores: dict[str, pd.DataFrame]
    parent_keys: list[str]
    sub_by_parent: dict[str, list[str]]
    universe: list[str]


def load_panel(path: Path = PANEL_PICKLE) -> StudyPanel:
    """Unpickle the cached PIT candidate panel (data access only)."""
    with path.open("rb") as fh:
        cand = pickle.load(fh)
    parents = [p for p, subs in cand.candidates_by_parent.items() if subs]
    return StudyPanel(
        rebal_dates=list(cand.rebal_dates),
        scores=cand.scores,
        parent_keys=parents,
        sub_by_parent={p: list(cand.candidates_by_parent[p]) for p in parents},
        universe=list(cand.universe),
    )


def load_price_matrix(tickers: list[str], start: str, end: str,
                      *, refresh: bool = False) -> pd.DataFrame:
    """Adjusted-close matrix (ISO-date index × ticker), benchmark included.

    ``adj_close`` falls back to ``close`` (mirrors the scoring layer). Cached
    to a local pickle keyed by span + universe size.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    want = sorted(set(tickers) | {BENCHMARK})
    key = CACHE_DIR / f"prices_{start}_{end}_{len(want)}.pkl"
    if key.exists() and not refresh:
        return pd.read_pickle(key)
    with get_db() as db:
        placeholders = ",".join("?" * len(want))
        df = db.query_df(
            f"SELECT ticker, date, adj_close, close FROM daily_prices "
            f"WHERE ticker IN ({placeholders}) AND date >= ? AND date <= ?",
            (*want, start, end))
    df["px"] = df["adj_close"].fillna(df["close"])
    matrix = df.pivot_table(index="date", columns="ticker", values="px").sort_index()
    matrix.to_pickle(key)
    return matrix


def load_vix() -> pd.Series:
    """Daily VIX close, Timestamp-indexed, ascending."""
    with get_db() as db:
        df = db.query_df("SELECT date, adj_close, close FROM daily_prices "
                         "WHERE ticker = 'VIX' ORDER BY date")
    px = df["adj_close"].fillna(df["close"]).astype(float)
    return pd.Series(px.values, index=pd.to_datetime(df["date"])).dropna()


def load_sectors() -> pd.Series:
    """GICS sector per ticker over the full universe table (current + departed)."""
    with get_db() as db:
        df = db.query_df("SELECT ticker, gics_sector FROM universe")
    return df.set_index("ticker")["gics_sector"].fillna("Unknown")
