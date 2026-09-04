"""Cached read-only data loaders for the 3-page dashboard.

Every accessor is:

* **read-only** — no INSERT/UPDATE/DELETE anywhere
* **cached** — wrapped in :func:`streamlit.cache_data` so a page render
  does not hit SQLite on every widget interaction

The Portfolio page's model book reuses the research ablation engine's own
``select_book`` / ``base_weights`` / ``sector_overlay`` helpers, so what the
dashboard shows is byte-identical to the ratified construction
(top-25% / cap5 / 50-50 SPY-QQQ ``blend_match`` sector targets).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import streamlit as st

from data.db import Database, get_db
from dashboard import candidates as dcand

DEFAULT_TTL = 300      # seconds
ROOT = Path(__file__).resolve().parents[1]

TOP_PCT = 0.25         # ratified book construction
SUMMARY_CSV = ROOT / "output" / "subfactor_expansion" / "summary_3M.csv"


# ---------------------------------------------------------------------------
# DB handle
# ---------------------------------------------------------------------------
# NOTE: we intentionally do NOT cache the Database across calls. Streamlit
# runs each page render on its own ScriptRunner thread, and sqlite3
# connections cannot be shared across threads.
def get_database() -> Database:
    return get_db()


@st.cache_data(ttl=DEFAULT_TTL, show_spinner=False)
def latest_score_date() -> str | None:
    return dcand.latest_as_of(get_database())


def refresh_all() -> None:
    st.cache_data.clear()


# ---------------------------------------------------------------------------
# Stocks page
# ---------------------------------------------------------------------------
@st.cache_data(ttl=DEFAULT_TTL, show_spinner=False)
def load_screener(as_of: str | None = None) -> pd.DataFrame:
    """Every scored ticker with overlay + parent factors + latest price."""
    return dcand.all_scored(get_database(), as_of)


@st.cache_data(ttl=DEFAULT_TTL, show_spinner=False)
def load_overlay_detail(ticker: str) -> dict[str, Any] | None:
    """Latest research overlay JSON payload for one ticker (or None)."""
    db = get_database()
    try:
        row = db.query_df(
            "SELECT overlay_json, model, computed_at, as_of_date "
            "FROM research_overlays WHERE ticker = ? "
            "ORDER BY as_of_date DESC LIMIT 1", (ticker.upper(),))
    except (pd.errors.DatabaseError, sqlite3.OperationalError):
        return None
    if row.empty:
        return None
    out = json.loads(row.iloc[0]["overlay_json"] or "{}")
    out["_model"] = row.iloc[0]["model"]
    out["_as_of"] = row.iloc[0]["as_of_date"]
    return out


# ---------------------------------------------------------------------------
# Stock detail page
# ---------------------------------------------------------------------------
@st.cache_data(ttl=DEFAULT_TTL, show_spinner=False)
def load_overlay_full(ticker: str) -> dict[str, Any] | None:
    """Full latest research_overlays row for one ticker, list columns parsed."""
    db = get_database()
    try:
        row = db.query_df(
            "SELECT * FROM research_overlays WHERE ticker = ? "
            "ORDER BY as_of_date DESC LIMIT 1", (ticker.upper(),))
    except (pd.errors.DatabaseError, sqlite3.OperationalError):
        return None
    if row.empty:
        return None
    out = row.iloc[0].to_dict()
    for col in ("red_flags", "open_questions", "confirming_evidence",
                "contradicting_evidence"):
        try:
            out[col] = json.loads(out.get(col) or "[]")
        except (TypeError, json.JSONDecodeError):
            out[col] = []
    return out


@st.cache_data(ttl=DEFAULT_TTL, show_spinner=False)
def load_sub_factor_scores(ticker: str, as_of: str | None = None) -> pd.DataFrame:
    """Every sub-factor score for one ticker at ``as_of`` (default latest)."""
    db = get_database()
    if as_of is None:
        as_of = latest_score_date()
    if as_of is None:
        return pd.DataFrame(columns=["factor", "sub_factor", "score", "raw_value"])
    return pd.read_sql_query(
        "SELECT factor, sub_factor, score, raw_value FROM sub_factor_scores "
        "WHERE ticker = ? AND as_of_date = ? ORDER BY factor, sub_factor",
        db._conn, params=[ticker.upper(), as_of])


@st.cache_data(ttl=DEFAULT_TTL, show_spinner=False)
def load_score_history(ticker: str) -> pd.DataFrame:
    """Composite score + rank/flag history for one ticker, oldest first."""
    db = get_database()
    return pd.read_sql_query(
        "SELECT as_of_date, composite_score, sector_rank, long_short_flag "
        "FROM composite_scores WHERE ticker = ? ORDER BY as_of_date",
        db._conn, params=[ticker.upper()])


@st.cache_data(ttl=DEFAULT_TTL, show_spinner=False)
def load_parent_factor_history(ticker: str) -> pd.DataFrame:
    """Parent-factor scores over time for one ticker, one column per factor."""
    db = get_database()
    rows = pd.read_sql_query(
        "SELECT as_of_date, LOWER(factor) AS factor, score "
        "FROM parent_factor_scores WHERE ticker = ? ORDER BY as_of_date",
        db._conn, params=[ticker.upper()])
    if rows.empty:
        return rows
    pivot = rows.pivot_table(index="as_of_date", columns="factor",
                              values="score", aggfunc="first").reset_index()
    pivot.columns.name = None
    return pivot


@st.cache_data(ttl=DEFAULT_TTL, show_spinner=False)
def load_price_history(ticker: str, n_days: int = 1260) -> pd.DataFrame:
    """Last ``n_days`` OHLC rows for one ticker, oldest first (default ~5y)."""
    db = get_database()
    df = pd.read_sql_query(
        "SELECT date, open, high, low, close, volume FROM daily_prices "
        "WHERE ticker = ? ORDER BY date DESC LIMIT ?",
        db._conn, params=[ticker.upper(), n_days])
    return df.iloc[::-1].reset_index(drop=True)


REPORTING_LAG_DAYS = 45  # matches factors.utils.DataContext.lag_quarterly_days


@st.cache_data(ttl=DEFAULT_TTL, show_spinner=False)
def load_pe_history(ticker: str, n_days: int = 1260) -> pd.DataFrame:
    """Trailing P/E (close / TTM diluted EPS) over the last ``n_days``.

    Point-in-time: a quarter's EPS is only admitted ``REPORTING_LAG_DAYS``
    after its fiscal period end, same conservative filing-lag convention
    :class:`factors.utils.DataContext` uses for backtests.
    """
    db = get_database()
    fund = pd.read_sql_query(
        "SELECT fiscal_date, eps_diluted FROM fundamentals "
        "WHERE ticker = ? AND period_type = 'quarterly' AND eps_diluted IS NOT NULL "
        "ORDER BY fiscal_date", db._conn, params=[ticker.upper()])
    if len(fund) < 4:
        return pd.DataFrame(columns=["date", "pe"])

    fund["fiscal_date"] = pd.to_datetime(fund["fiscal_date"])
    # A handful of tickers carry the same real quarter twice under two
    # fiscal_dates a few days apart (raw vs. source-normalized period end,
    # e.g. AAPL 2025-12-27 & 2025-12-31 with identical EPS). A rolling-4
    # window would double-count that quarter and drop a real one, so
    # collapse near-duplicate reports (<30d apart — real quarters are
    # always ~90d apart) into the latest one before computing TTM.
    keep: list[int] = []
    for idx in fund.index:
        if keep and (fund.loc[idx, "fiscal_date"]
                    - fund.loc[keep[-1], "fiscal_date"]).days < 30:
            keep[-1] = idx
        else:
            keep.append(idx)
    fund = fund.loc[keep].reset_index(drop=True)

    fund["ttm_eps"] = fund["eps_diluted"].rolling(4, min_periods=4).sum()
    fund = fund.dropna(subset=["ttm_eps"])
    if fund.empty:
        return pd.DataFrame(columns=["date", "pe"])
    fund["available_date"] = fund["fiscal_date"] + pd.Timedelta(days=REPORTING_LAG_DAYS)
    fund = fund.sort_values("available_date")

    prices = load_price_history(ticker, n_days)[["date", "close"]].copy()
    if prices.empty:
        return pd.DataFrame(columns=["date", "pe"])
    prices["date"] = pd.to_datetime(prices["date"])

    merged = pd.merge_asof(prices, fund[["available_date", "ttm_eps"]],
                           left_on="date", right_on="available_date",
                           direction="backward")
    merged["pe"] = merged["close"] / merged["ttm_eps"].where(merged["ttm_eps"] > 0)
    return merged[["date", "pe"]].dropna()


# ---------------------------------------------------------------------------
# Portfolio page — the model book
# ---------------------------------------------------------------------------
@st.cache_data(ttl=DEFAULT_TTL, show_spinner=False)
def load_caps(as_of: str) -> pd.Series:
    """Market-cap proxy per ticker: latest float shares × latest close
    (same proxy the research pipeline's PIT caps use)."""
    db = get_database()
    fs = pd.read_sql_query(
        "SELECT ticker, float_shares FROM short_interest si "
        "WHERE date = (SELECT MAX(date) FROM short_interest s2 "
        "              WHERE s2.ticker = si.ticker AND s2.date <= ?) "
        "AND float_shares IS NOT NULL", db._conn, params=[as_of])
    px = pd.read_sql_query(
        "SELECT ticker, close FROM daily_prices dp "
        "WHERE date = (SELECT MAX(date) FROM daily_prices d2 "
        "              WHERE d2.ticker = dp.ticker AND d2.date <= ?)",
        db._conn, params=[as_of])
    caps = (fs.set_index("ticker")["float_shares"]
            * px.set_index("ticker")["close"]).dropna()
    return caps[caps > 0]


@st.cache_data(ttl=DEFAULT_TTL, show_spinner=False)
def load_model_book(as_of: str | None = None) -> dict[str, Any]:
    """Compute the ratified model book from the latest composite scores.

    Returns {"book": DataFrame, "targets": DataFrame, "as_of": str}.
    ``book``: ticker, company_name, sector, weight, composite_score, price.
    ``targets``: sector × [book, blend_target, spy_target, qqq_target].
    """
    from research.ablation.engine import (
        AblationConfig, _fill_median, _qqq_members, base_weights,
        sector_overlay, select_book,
    )

    frame = load_screener(as_of)
    if frame.empty:
        return {"book": pd.DataFrame(), "targets": pd.DataFrame(), "as_of": None}
    d = frame["as_of_date"].iloc[0]
    scores = frame.set_index("ticker")["composite_score"].dropna()
    sectors = frame.set_index("ticker")["sector"].fillna("Unknown")
    caps = load_caps(d).reindex(scores.index)
    data = SimpleNamespace(caps=pd.DataFrame([caps], index=[d]), sectors=sectors)
    cfg = AblationConfig(name="model_book", top_pct=TOP_PCT, weighting="cap5",
                         sector="blend_match", hold_months=1)

    names = select_book(scores, TOP_PCT, None, [])
    w = base_weights(cfg, names, scores, d, data)
    w = sector_overlay(w, cfg, scores.index.tolist(), d, data)
    w = w / w.sum()

    book = (frame.set_index("ticker")
            .loc[w.index, ["company_name", "sector", "composite_score", "price"]]
            .assign(weight=w)
            .sort_values("weight", ascending=False)
            .reset_index())

    # Sector targets for the allocation chart (same math as blend_match).
    ucaps = _fill_median(data.caps.loc[d].reindex(scores.index))
    uni_sec = sectors.reindex(scores.index).fillna("Unknown")
    spy_tgt = ucaps.groupby(uni_sec).sum()
    spy_tgt = spy_tgt / spy_tgt.sum()
    members = [t for t in _qqq_members() if t in caps.index]
    mc = caps.reindex(members).dropna()
    qqq_tgt = mc.groupby(sectors.reindex(mc.index)).sum()
    qqq_tgt = qqq_tgt / qqq_tgt.sum()
    idx = spy_tgt.index.union(qqq_tgt.index)
    spy_tgt = spy_tgt.reindex(idx).fillna(0.0)
    qqq_tgt = qqq_tgt.reindex(idx).fillna(0.0)
    blend = 0.5 * spy_tgt + 0.5 * qqq_tgt
    held = w.groupby(sectors.reindex(w.index)).sum().reindex(idx).fillna(0.0)
    targets = pd.DataFrame({"book": held, "blend_target": blend,
                            "spy_target": spy_tgt, "qqq_target": qqq_tgt})
    targets = targets[targets.max(axis=1) > 0.001].sort_values(
        "blend_target", ascending=False)
    return {"book": book, "targets": targets, "as_of": d}


# ---------------------------------------------------------------------------
# Scoring page — factor internals
# ---------------------------------------------------------------------------
@st.cache_data(ttl=DEFAULT_TTL, show_spinner=False)
def load_engine_weights() -> dict[str, float]:
    from factors.parent_selection_v4 import V4_PARENT_WEIGHTS
    return dict(V4_PARENT_WEIGHTS)


@st.cache_data(ttl=DEFAULT_TTL, show_spinner=False)
def load_selected_subs() -> dict[str, dict[str, float]]:
    from factors.parent_selection_v4 import SELECTED_SUBS
    return {p: dict(s) for p, s in SELECTED_SUBS.items()}


@st.cache_data(ttl=DEFAULT_TTL, show_spinner=False)
def load_parent_correlation(n_dates: int = 30) -> pd.DataFrame:
    """Mean per-date cross-sectional Pearson correlation of the parent
    scores over the last ``n_dates`` score dates.

    Per-date zscore normalization is linear per column, and Pearson
    correlation is invariant to linear transforms — so the raw parent
    percentiles give exactly the correlation the composite blend sees.
    """
    db = get_database()
    dates = pd.read_sql_query(
        "SELECT DISTINCT as_of_date FROM parent_factor_scores "
        "ORDER BY as_of_date DESC LIMIT ?", db._conn, params=[n_dates])
    mats = []
    for d in dates["as_of_date"]:
        rows = pd.read_sql_query(
            "SELECT ticker, LOWER(factor) AS factor, score "
            "FROM parent_factor_scores WHERE as_of_date = ?",
            db._conn, params=[d])
        if rows.empty:
            continue
        pivot = rows.pivot_table(index="ticker", columns="factor",
                                 values="score", aggfunc="first")
        if len(pivot) >= 30:
            mats.append(pivot.corr(method="pearson"))
    if not mats:
        return pd.DataFrame()
    C = pd.concat(mats).groupby(level=0, sort=False).mean()
    order = [f for f in dcand.PARENT_FACTORS if f in C.index]
    return C.loc[order, order]


@st.cache_data(ttl=DEFAULT_TTL, show_spinner=False)
def load_factor_composite_r2() -> pd.Series:
    """Pearson R² of each parent factor score against the final composite
    score, on the latest scored date. Answers "how much does this factor
    actually move the ranking" directly — unlike engine weight (the blend
    input) or effective exposure (correlation-adjusted blend input), this is
    measured against the realized output after normalization, cap/floor
    trimming, and the sector-percentile re-rank all apply.
    """
    db = get_database()
    d = pd.read_sql_query(
        "SELECT MAX(as_of_date) AS d FROM composite_scores", db._conn
    )["d"].iloc[0]
    if not d:
        return pd.Series(dtype=float)
    comp = pd.read_sql_query(
        "SELECT ticker, composite_score FROM composite_scores WHERE as_of_date = ?",
        db._conn, params=[d]).set_index("ticker")["composite_score"]
    par = pd.read_sql_query(
        "SELECT ticker, LOWER(factor) AS factor, score FROM parent_factor_scores "
        "WHERE as_of_date = ?", db._conn, params=[d])
    if par.empty or comp.empty:
        return pd.Series(dtype=float)
    piv = par.pivot_table(index="ticker", columns="factor", values="score", aggfunc="first")
    df = piv.join(comp, how="inner").dropna()
    order = [f for f in dcand.PARENT_FACTORS if f in piv.columns]
    r2 = {f: float(df[f].corr(df["composite_score"]) ** 2) for f in order}
    return pd.Series(r2, name="r2")


@st.cache_data(ttl=DEFAULT_TTL, show_spinner=False)
def load_subfactor_stats() -> pd.DataFrame:
    """Research-battery stats (mean IC, IR, coverage, hit rate) per
    candidate subfactor, from the latest validation run's summary CSV."""
    if not SUMMARY_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(SUMMARY_CSV)
    keep = ["candidate", "parent", "coverage", "mean_ic_3m6m",
            "information_ratio", "hit_rate", "monotonicity"]
    return df[[c for c in keep if c in df.columns]]
