"""Layer-7 stock-view assembly (read-only).

Builds the screener frame: the latest row in ``composite_scores`` for every
scored ticker joined with everything the Stocks page shows next to it:

* ``research_overlays`` (Layer 3 qualitative overlay, if present)
* ``parent_factor_scores`` pivoted into one column per factor
* ``universe`` for company name + GICS sub-industry
* ``daily_prices`` for the latest close

This module returns *DataFrames* — it does not render and it does not write.

The overlay join carries the MOST RECENT overlay row forward
(``as_of_date <= score date``) because overlays are re-generated less often
than scores; an exact-date join would hide a memo the moment the universe is
re-scored.
"""
from __future__ import annotations

import sqlite3
from typing import Iterable, Sequence

import pandas as pd

from data.db import Database

# Canonical factor order — every view shows the same columns in the same
# order regardless of which factors happen to be populated on a given date.
PARENT_FACTORS: tuple[str, ...] = (
    "growth", "quality", "value", "momentum",
    "revisions", "insider", "institutional", "short",
)

# Overlay columns carried through every screener frame.
OVERLAY_COLS: tuple[str, ...] = (
    "research_status", "quant_signal_review", "qualitative_risk_level",
    "overlay_confidence",
)


def latest_as_of(db: Database) -> str | None:
    return db.scalar("SELECT MAX(as_of_date) FROM composite_scores")


def all_scored(db: Database, as_of: str | None = None) -> pd.DataFrame:
    """Every scored ticker at ``as_of`` (default: latest score date).

    Columns: ticker, company_name, sector, industry, price, market_cap,
    composite_score, research_status, quant_signal_review,
    qualitative_risk_level, overlay_confidence, total_score, sector_rank,
    long_short_flag, as_of_date, plus one column per parent factor.
    """
    if as_of is None:
        as_of = latest_as_of(db)
    if as_of is None:
        return _empty_frame()
    base = _load_scored_frame(db, as_of)
    if base.empty:
        return base
    factors = _factor_pivot(db, as_of, base["ticker"].tolist())
    out = base.merge(factors, on="ticker", how="left")
    for f in PARENT_FACTORS:
        if f not in out.columns:
            out[f] = pd.NA
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _empty_frame() -> pd.DataFrame:
    cols = ["ticker", "company_name", "sector", "industry", "price",
            "market_cap", "composite_score", *OVERLAY_COLS, "total_score",
            "sector_rank", "long_short_flag", "as_of_date",
            *PARENT_FACTORS]
    return pd.DataFrame(columns=cols)


def _load_scored_frame(db: Database, as_of: str,
                       tickers: Iterable[str] | None = None) -> pd.DataFrame:
    """Join composite_scores ⋈ research_overlays ⋈ universe ⋈ latest price.

    Falls back gracefully if research_overlays isn't populated yet.
    ``total_score`` is always the Layer 2 composite — the overlay never
    moves a number.
    """
    base_sql = (
        "SELECT cs.ticker, "
        "       u.company_name, "
        "       cs.sector, "
        "       u.gics_sub_industry AS industry, "
        "       cs.composite_score, "
        "       ro.research_status, "
        "       ro.quant_signal_review, "
        "       ro.qualitative_risk_level, "
        "       ro.confidence AS overlay_confidence, "
        "       cs.composite_score AS total_score, "
        "       cs.sector_rank, "
        "       cs.long_short_flag, "
        "       cs.as_of_date "
        "FROM composite_scores cs "
        "LEFT JOIN research_overlays ro "
        "  ON ro.ticker = cs.ticker "
        " AND ro.as_of_date = ("
        "       SELECT MAX(r2.as_of_date) FROM research_overlays r2 "
        "       WHERE r2.ticker = cs.ticker AND r2.as_of_date <= cs.as_of_date) "
        "LEFT JOIN universe u ON u.ticker = cs.ticker "
        "WHERE cs.as_of_date = ?"
    )
    params: list = [as_of]
    if tickers:
        placeholders = ", ".join(["?"] * len(list(tickers)))
        base_sql += f" AND cs.ticker IN ({placeholders})"
        params.extend(t.upper() for t in tickers)

    try:
        df = pd.read_sql_query(base_sql, db._conn, params=params)
    except (pd.errors.DatabaseError, sqlite3.OperationalError):
        # research_overlays absent — degrade to composite-only.
        fallback = (
            "SELECT cs.ticker, u.company_name, cs.sector, "
            "       u.gics_sub_industry AS industry, "
            "       cs.composite_score, "
            "       NULL AS research_status, NULL AS quant_signal_review, "
            "       NULL AS qualitative_risk_level, NULL AS overlay_confidence, "
            "       cs.composite_score AS total_score, "
            "       cs.sector_rank, cs.long_short_flag, cs.as_of_date "
            "FROM composite_scores cs "
            "LEFT JOIN universe u ON u.ticker = cs.ticker "
            "WHERE cs.as_of_date = ?"
        )
        fb_params: list = [as_of]
        if tickers:
            placeholders = ", ".join(["?"] * len(list(tickers)))
            fallback += f" AND cs.ticker IN ({placeholders})"
            fb_params.extend(t.upper() for t in tickers)
        df = pd.read_sql_query(fallback, db._conn, params=fb_params)

    if df.empty:
        return df

    prices = _latest_prices(db, as_of)
    df["price"] = df["ticker"].map(prices)
    shares = _latest_shares_outstanding(db, as_of)
    df["market_cap"] = df["price"] * df["ticker"].map(shares)
    return df


def _factor_pivot(db: Database, as_of: str,
                  tickers: Sequence[str]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=["ticker", *PARENT_FACTORS])
    placeholders = ", ".join(["?"] * len(tickers))
    rows = pd.read_sql_query(
        f"SELECT ticker, LOWER(factor) AS factor, score "
        f"FROM parent_factor_scores "
        f"WHERE as_of_date = ? AND ticker IN ({placeholders})",
        db._conn, params=[as_of, *[t.upper() for t in tickers]],
    )
    if rows.empty:
        return pd.DataFrame({"ticker": list(tickers)})
    pivot = (rows.pivot_table(index="ticker", columns="factor",
                              values="score", aggfunc="first")
                 .reset_index())
    pivot.columns.name = None
    return pivot


def _latest_prices(db: Database, as_of: str) -> pd.Series:
    """Latest close on/before ``as_of`` for every ticker, one query."""
    rows = pd.read_sql_query(
        "SELECT ticker, close FROM daily_prices dp "
        "WHERE date = (SELECT MAX(date) FROM daily_prices d2 "
        "              WHERE d2.ticker = dp.ticker AND d2.date <= ?)",
        db._conn, params=[as_of],
    )
    return rows.set_index("ticker")["close"]


def _latest_shares_outstanding(db: Database, as_of: str) -> pd.Series:
    """Most recent quarterly shares-outstanding on/before ``as_of``.

    ``fundamentals.shares_outstanding`` is almost entirely NULL, so this
    pulls ``weightedAverageShsOut`` out of the FMP ``raw_json`` blob instead
    (populated for every scored ticker).
    """
    rows = pd.read_sql_query(
        "SELECT ticker, "
        "       json_extract(raw_json, '$.weightedAverageShsOut') AS shares "
        "FROM fundamentals f "
        "WHERE period_type = 'quarterly' AND raw_json IS NOT NULL "
        "  AND fiscal_date = ("
        "       SELECT MAX(f2.fiscal_date) FROM fundamentals f2 "
        "       WHERE f2.ticker = f.ticker AND f2.period_type = 'quarterly' "
        "         AND f2.raw_json IS NOT NULL AND f2.fiscal_date <= ?)",
        db._conn, params=[as_of],
    )
    return rows.set_index("ticker")["shares"]
