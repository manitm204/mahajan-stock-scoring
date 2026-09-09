"""Candidate subfactor library — 8-10 candidates per parent.

Each parent bucket exposes a ``build_<parent>(ctx)`` returning a list of
:class:`Candidate` records (name, raw Series, direction). The panel layer
scores every raw with the production 0-100 GICS-sector-relative percentile
rule (:func:`factors.utils.sector_percentile`), so ICs are directly comparable
to the incumbent ``factors/`` output. NaNs pre-percentile map to a neutral 50.
Heavy-tailed raws are winsorized within sector before ranking.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from data.db import get_db
from factors.utils import DataContext, col

# =============================================================================
# Candidate record + helpers
# =============================================================================


@dataclass
class Candidate:
    """One candidate subfactor at one as-of.

    ``name`` — canonical subfactor identifier used everywhere (matches the
    proposal doc). ``raw`` — per-ticker Series. ``higher_is_better`` — the
    direction the sector-percentile step applies. ``parent`` — parent bucket
    label. ``family`` — coarse strategy tag used only for the report grouping.
    """

    name: str
    raw: pd.Series
    higher_is_better: bool
    parent: str
    family: str = ""


ParentBuilder = Callable[[DataContext], list[Candidate]]


def _wins(series: pd.Series, sectors: pd.Series, lo: float = 0.05,
          hi: float = 0.95) -> pd.Series:
    """Winsorize ``series`` at ``[lo, hi]`` percentiles *within each sector*.

    Heavy-tailed raw ratios (gross margin, FCF margin, interest coverage) let
    a handful of outliers dominate the percentile rank in a small sector. This
    trims those tails before ranking without shifting the overall order.
    Missing values pass through unchanged.
    """
    if series.empty:
        return series
    out = series.copy().astype(float)
    for sec in sectors.dropna().unique():
        mask = sectors.reindex(out.index) == sec
        vals = out[mask]
        finite = vals.dropna()
        if len(finite) < 5:
            continue
        low_q, high_q = finite.quantile([lo, hi])
        out.loc[mask] = vals.clip(lower=low_q, upper=high_q)
    return out


def _price_history_days(ctx: DataContext, lookback_days: int) -> pd.DataFrame:
    """Adjusted-close matrix (date x ticker) for at least ``lookback_days``.

    Prefer the DataContext's cached matrix; fall back to a fresh query so we
    always have enough history for the requested window.
    """
    px = ctx.price_matrix(lookback_days=max(lookback_days + 30, 420))
    return px


def _annual_series(ctx: DataContext, metric: str) -> pd.DataFrame:
    """Wide (ticker x fiscal_date) annual metric from raw fundamentals."""
    df = ctx.fund_annual()
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    return df.pivot_table(index="fiscal_date", columns="ticker",
                          values=metric, aggfunc="last").sort_index()


def _annual_yoy(ctx: DataContext, metric: str) -> pd.Series:
    """YoY change of ``metric`` between each ticker's two latest annual periods.

    Growth is ``(curr - prior) / |prior|``. Because different tickers have
    different fiscal-year ends, the calculation is per-ticker on that ticker's
    own last two annual rows — a global .tail(2) of the pivot would miss any
    ticker whose fiscal calendar is offset from the majority.
    """
    df = ctx.fund_annual()
    if df.empty or metric not in df.columns:
        return pd.Series(dtype=float, index=ctx.universe)

    def _pct(g: pd.DataFrame) -> float:
        vals = g.sort_values("fiscal_date")[metric].dropna().values
        if len(vals) < 2:
            return float("nan")
        prior = float(vals[-2])
        if prior == 0.0:
            return float("nan")
        return float((vals[-1] - prior) / abs(prior))

    return df[["ticker", "fiscal_date", metric]].groupby("ticker").apply(
        _pct).reindex(ctx.universe)


# =============================================================================
# Momentum
# =============================================================================

_MONTH = 21
_QUARTER = 63
_HALF = 126
_YEAR = 252


def build_momentum(ctx: DataContext) -> list[Candidate]:
    pf = ctx.price_features()
    px = _price_history_days(ctx, _YEAR + _MONTH)

    ret_12_1 = ctx.horizon_return(_YEAR - _MONTH, offset=_MONTH)
    ret_6m = ctx.horizon_return(_HALF)
    ret_3m_ex_1m = ctx.horizon_return(_QUARTER - _MONTH, offset=_MONTH)
    high_prox = col(pf, "distance_from_52w_high")
    vol_20 = col(pf, "volatility_20d")
    vol_adj = ret_12_1 / vol_20.replace(0, np.nan)
    rel_vol = col(pf, "relative_volume")

    # Volume-confirmed: 12-1 return weighted by whether trailing volume expanded.
    # rel_volume ~1 = normal, >1 = expansion. The weight (1 + log(rel_vol)) is
    # clipped to stay positive: multiplying by a raw log(rel_vol) < 0 would
    # *flip the sign* of the return, ranking a crash on shrinking volume above
    # a rally on shrinking volume. Clipped, shrinking volume only dampens.
    vol_conf = ret_12_1 * np.clip(1.0 + np.log(rel_vol.clip(lower=0.1)), 0.25, 2.0)

    # Consistency: share of last 60 daily returns > 0.
    consistency = _positive_share(px, 60).reindex(ctx.universe)
    # Max drawdown last 252 sessions (negative-return metric).
    mdd = _max_drawdown(px, 252).reindex(ctx.universe)

    return [
        Candidate("mom_12_1", ret_12_1, True, "momentum", "trend"),
        Candidate("mom_6m", ret_6m, True, "momentum", "trend"),
        Candidate("mom_3m_ex_1m", ret_3m_ex_1m, True, "momentum", "trend"),
        Candidate("mom_52w_high_prox", high_prox, True, "momentum", "anchor"),
        Candidate("mom_vol_adjusted", vol_adj, True, "momentum", "risk_adj"),
        Candidate("mom_volume_confirmed", vol_conf, True, "momentum", "confirmation"),
        Candidate("mom_consistency_60d", consistency, True, "momentum", "quality_of_trend"),
        # Drawdowns are <= 0 with 0 = no drawdown, so *larger* raw = milder
        # drawdown = better trend quality. (Was False, which ranked the most
        # severe drawdown best.)
        Candidate("mom_max_drawdown_252d", mdd, True, "momentum", "quality_of_trend"),
    ]


def _positive_share(px: pd.DataFrame, window: int) -> pd.Series:
    if px.empty or len(px) <= window + 1:
        return pd.Series(dtype=float)
    rets = px.pct_change(fill_method=None).tail(window)
    if rets.empty:
        return pd.Series(dtype=float)
    return (rets > 0).mean()


def _max_drawdown(px: pd.DataFrame, window: int) -> pd.Series:
    if px.empty or len(px) < window:
        return pd.Series(dtype=float)
    tail = px.tail(window)
    running_max = tail.cummax()
    dd = tail / running_max - 1.0
    # The deepest drawdown in the window (a value <= 0; 0 = never drew down).
    return dd.min()


# =============================================================================
# Value
# =============================================================================


def build_value(ctx: DataContext) -> list[Candidate]:
    f = ctx.fund_annual_latest()
    mcap = ctx.market_cap()
    ev = ctx.enterprise_value()
    sectors = ctx.sectors()

    equity = col(f, "shareholder_equity")
    book_to_price = equity / mcap.replace(0, np.nan)

    fcf = col(f, "free_cash_flow")
    fcf_yield_raw = fcf / mcap.replace(0, np.nan)
    fcf_yield_clean = _wins(fcf_yield_raw, sectors, 0.02, 0.98)

    # EV multiples are ranked in *yield* form (metric / EV, higher = cheaper).
    # The naive inverted multiple (EV / metric, lower-is-better) sign-flips when
    # the denominator is negative: a negative-EBITDA name gets a negative
    # EV/EBITDA and ranks as the cheapest stock in its sector. Yield form sorts
    # negatives to the bottom naturally. Non-positive EV (reconstructed EV is
    # meaningless for cash-heavy names and banks) is masked to NaN -> neutral 50.
    # Candidate names keep their historical "_inv" identifiers because the V4
    # production allowlist references them.
    ev_pos = ev.where(ev > 0)
    ebitda = col(f, "ebitda")
    ebitda_yield = ebitda / ev_pos
    fcf_to_ev = fcf / ev_pos
    revenue = col(f, "revenue")
    revenue_to_ev = revenue / ev_pos

    # Earnings yield (net income / mcap): the direct P/E inverse; robust when
    # profits are volatile because the raw values are already in dollar units.
    ni = col(f, "net_income")
    earnings_yield = ni / mcap.replace(0, np.nan)

    # Buyback yield: buybacks are stored as negative cash outflows, so negate
    # to make a positive-yield reading. Where FMP's dividends_paid is populated
    # (rare in the current pull), the shareholder yield sub folds it in.
    buybacks = col(f, "buybacks").fillna(0.0)
    buyback_yield = (-buybacks) / mcap.replace(0, np.nan)

    # Dividend yield from the new historical_dividends table. Trailing-12M sum of
    # per-share cash dividends divided by the latest price.
    div_yield = _dividend_yield(ctx)

    # Shareholder yield rebuilt with the real dividend series (falls back to
    # buybacks-only if the dividend table has no rows for a ticker).
    dividends_dollars = div_yield * mcap
    total_return_flow = (-buybacks).fillna(0.0) + dividends_dollars.fillna(0.0)
    both_missing = buybacks.isna() & dividends_dollars.isna()
    shareholder_yield = (total_return_flow / mcap.replace(0, np.nan)).mask(
        both_missing.reindex(mcap.index, fill_value=True))

    return [
        Candidate("val_book_to_price", book_to_price, True, "value", "book"),
        Candidate("val_earnings_yield", earnings_yield, True, "value", "earnings"),
        Candidate("val_fcf_yield_clean", fcf_yield_clean, True, "value", "cash"),
        Candidate("val_ev_ebitda_inv", ebitda_yield, True, "value", "operating"),
        Candidate("val_ev_fcf_inv", fcf_to_ev, True, "value", "operating"),
        Candidate("val_ev_revenue_inv", revenue_to_ev, True, "value", "operating"),
        # val_sales_to_ev dropped: identical to the fixed val_ev_revenue_inv.
        Candidate("val_dividend_yield", div_yield, True, "value", "shareholder_return"),
        Candidate("val_buyback_yield", buyback_yield, True, "value", "shareholder_return"),
        Candidate("val_shareholder_yield", shareholder_yield, True, "value", "shareholder_return"),
    ]


def _dividend_yield(ctx: DataContext) -> pd.Series:
    """Trailing-12M cash dividend per share / current price.

    Returns an all-NaN Series when the ``historical_dividends`` table is empty
    (e.g. dividend backfill has not run yet). ``NaN`` for non-dividend payers
    is the correct behaviour: sector percentile will score them a neutral 50,
    which fairly reflects "no yield" for a factor that rewards yield.
    """
    db = ctx.db
    row = db.query_one("SELECT COUNT(*) AS n FROM historical_dividends")
    if row is None or row["n"] == 0:
        return pd.Series(dtype=float, index=ctx.universe)

    upper = ctx.as_of
    lower = (pd.Timestamp(upper) - pd.Timedelta(days=365)).date().isoformat()
    df = db.query_df(
        "SELECT ticker, SUM(COALESCE(adj_dividend, dividend)) AS ttm_div "
        "FROM historical_dividends WHERE ex_date <= ? AND ex_date > ? "
        "GROUP BY ticker",
        (upper, lower),
    )
    if df.empty:
        return pd.Series(dtype=float, index=ctx.universe)
    ttm = df.set_index("ticker")["ttm_div"].astype(float)
    px = ctx.prices()
    yld = ttm / px.replace(0, np.nan)
    return yld.reindex(ctx.universe)


# =============================================================================
# Quality
# =============================================================================


def build_quality(ctx: DataContext) -> list[Candidate]:
    from factors.quality import _piotroski, _altman  # reuse production logic

    ff = ctx.fund_features_latest()
    f = ctx.fund_annual_latest()
    prior = ctx.fund_annual_prior()
    sectors = ctx.sectors()

    piotroski = _piotroski(f, prior)
    altman = _altman(f, ctx.market_cap(), sectors)

    # ROIC's denominator (debt + equity) can go negative for buyback-heavy
    # names, flipping the sign of the ratio; mask those to NaN -> neutral.
    invested = col(f, "debt").fillna(0.0) + col(f, "shareholder_equity")
    roic = col(ff, "roic").mask(invested <= 0)
    gross_margin = _wins(col(ff, "gross_margin"), sectors, 0.05, 0.95)
    operating_margin = _wins(col(ff, "operating_margin"), sectors, 0.05, 0.95)
    fcf = col(f, "free_cash_flow")
    revenue = col(f, "revenue").replace(0, np.nan)
    fcf_margin = _wins(fcf / revenue, sectors, 0.05, 0.95)
    # Leverage ranked as net_debt / |EBITDA| instead of Layer 1's
    # net_debt_to_ebitda (net_debt / EBITDA): dividing by a *negative* EBITDA
    # flips the sign, so an indebted money-loser ranked as the least-levered
    # name in its sector. Scaling by |EBITDA| keeps the net-debt sign — net
    # cash stays negative (good) whether or not EBITDA is positive, debt stays
    # positive (bad) — and Altman-Z separately penalizes the distressed cases.
    debt_to_ebitda = col(f, "net_debt") / col(f, "ebitda").abs().replace(0, np.nan)
    interest_cov = _wins(col(ff, "interest_coverage"), sectors, 0.02, 0.98)
    asset_turn = col(ff, "asset_turnover")
    earnings_stab = _earnings_stability_3y(ctx)

    return [
        Candidate("qual_altman_z", altman, True, "quality", "distress"),
        Candidate("qual_piotroski_f", piotroski, True, "quality", "composite"),
        Candidate("qual_roic", roic, True, "quality", "profitability"),
        Candidate("qual_gross_margin_wins", gross_margin, True, "quality", "profitability"),
        Candidate("qual_operating_margin", operating_margin, True, "quality", "profitability"),
        Candidate("qual_fcf_margin", fcf_margin, True, "quality", "cash_conversion"),
        Candidate("qual_debt_to_ebitda_inv", debt_to_ebitda, False, "quality", "leverage"),
        Candidate("qual_interest_coverage", interest_cov, True, "quality", "leverage"),
        Candidate("qual_asset_turnover", asset_turn, True, "quality", "efficiency"),
        Candidate("qual_earnings_stability_3y", earnings_stab, True, "quality", "stability"),
    ]


def _earnings_stability_3y(ctx: DataContext) -> pd.Series:
    """Inverse std-dev of each ticker's own last 3 annual EPS reports.

    Higher = smoother earnings. Names with fewer than 3 annual EPS reports
    return NaN (neutral score downstream). Std of zero (constant EPS) maps to
    ``+inf`` — clipped to a large finite number to keep percentiles stable.
    """
    df = ctx.fund_annual()
    if df.empty or "eps_diluted" not in df.columns:
        return pd.Series(dtype=float, index=ctx.universe)

    def _stab(g: pd.DataFrame) -> float:
        vals = g.sort_values("fiscal_date")["eps_diluted"].dropna().values
        if len(vals) < 3:
            return float("nan")
        s = float(np.std(vals[-3:], ddof=0))
        return float("inf") if s == 0 else 1.0 / s

    stab = df.groupby("ticker").apply(_stab)
    # Cap absurd values so a duplicate-EPS row cannot rig the percentile.
    finite = stab.replace([np.inf, -np.inf], np.nan).dropna()
    if not finite.empty:
        cap = float(finite.quantile(0.99)) * 10
        stab = stab.replace([np.inf, -np.inf], cap).clip(upper=cap)
    return stab.reindex(ctx.universe)


# =============================================================================
# Growth
# =============================================================================


def build_growth(ctx: DataContext) -> list[Candidate]:
    ff = ctx.fund_features_latest()

    rev_yoy = col(ff, "revenue_growth_yoy")
    eps_yoy = col(ff, "eps_growth_yoy")
    rev_cagr = col(ff, "revenue_cagr_3y")

    ffa = ctx.fund_features_annual()
    rev_accel = _yoy_delta(ffa, "revenue_growth_yoy").reindex(ctx.universe)
    margin_exp = _yoy_delta(ffa, "operating_margin").reindex(ctx.universe)

    gross_profit_growth = _annual_yoy(ctx, "gross_profit")
    op_income_growth = _annual_yoy(ctx, "operating_income")
    ebitda_growth = _annual_yoy(ctx, "ebitda")
    fcf_growth_smoothed = _smoothed_yoy(ctx, "free_cash_flow", years=2)

    surprise = _latest_earnings_surprise(ctx)

    return [
        Candidate("grw_revenue_yoy", rev_yoy, True, "growth", "level"),
        Candidate("grw_earnings_yoy", eps_yoy, True, "growth", "level"),
        Candidate("grw_revenue_cagr_3y", rev_cagr, True, "growth", "level"),
        Candidate("grw_revenue_acceleration", rev_accel, True, "growth", "acceleration"),
        Candidate("grw_gross_profit_growth", gross_profit_growth, True, "growth", "quality_of_growth"),
        Candidate("grw_operating_income_growth", op_income_growth, True, "growth", "quality_of_growth"),
        Candidate("grw_ebitda_growth", ebitda_growth, True, "growth", "quality_of_growth"),
        Candidate("grw_fcf_growth_smoothed", fcf_growth_smoothed, True, "growth", "cash"),
        Candidate("grw_margin_expansion_1y", margin_exp, True, "growth", "acceleration"),
        Candidate("grw_earnings_surprise", surprise, True, "growth", "surprise"),
    ]


def _yoy_delta(ffa: pd.DataFrame, metric: str) -> pd.Series:
    """Change in an annual-frequency metric between the two latest points."""
    if ffa.empty or metric not in ffa.columns:
        return pd.Series(dtype=float)

    def _delta(g: pd.Series) -> float:
        vals = g.dropna().values
        if len(vals) < 2:
            return float("nan")
        return float(vals[-1] - vals[-2])

    return ffa.groupby("ticker")[metric].apply(_delta)


def _smoothed_yoy(ctx: DataContext, metric: str, years: int = 2) -> pd.Series:
    """Average YoY growth of ``metric`` over each ticker's last ``years`` annuals.

    Dampens single-year noise in FCF-like metrics. Per-ticker so heterogeneous
    fiscal calendars don't leave anyone missing.
    """
    df = ctx.fund_annual()
    if df.empty or metric not in df.columns:
        return pd.Series(dtype=float, index=ctx.universe)

    def _avg(g: pd.DataFrame) -> float:
        vals = g.sort_values("fiscal_date")[metric].dropna().values
        if len(vals) < years + 1:
            return float("nan")
        tail = vals[-(years + 1):]
        growth = np.diff(tail) / np.abs(tail[:-1])
        growth = growth[np.isfinite(growth)]
        return float(np.mean(growth)) if growth.size else float("nan")

    return df.groupby("ticker").apply(_avg).reindex(ctx.universe)


def _latest_earnings_surprise(ctx: DataContext) -> pd.Series:
    """Most recently reported earnings surprise per ticker.

    Uses ``earnings_calendar.eps_actual`` and ``eps_estimate`` — the last row
    on/before ``ctx.cutoff`` where both are non-null. Ratio is
    ``(actual - estimate) / |estimate|`` capped to ±5x.
    """
    sql = (
        "SELECT ticker, earnings_date, eps_estimate, eps_actual FROM earnings_calendar "
        "WHERE earnings_date <= ? AND eps_estimate IS NOT NULL AND eps_actual IS NOT NULL"
    )
    df = ctx.db.query_df(sql, (ctx.cutoff,))
    if df.empty:
        return pd.Series(dtype=float, index=ctx.universe)
    df = df.sort_values("earnings_date").drop_duplicates("ticker", keep="last")
    denom = df["eps_estimate"].abs().replace(0, np.nan)
    surp = (df["eps_actual"] - df["eps_estimate"]) / denom
    surp = surp.clip(lower=-5.0, upper=5.0)
    surp.index = df["ticker"]
    return surp.reindex(ctx.universe)


# =============================================================================
# Registry — parents in the order they appear in the report.
# =============================================================================

# The revisions/institutional/insider/short builders live in `library_flow.py`
# to keep every module under 500 lines. They are imported here so
# CANDIDATE_BUILDERS is the single lookup consumers use.
from .library_flow import (  # noqa: E402
    build_insider,
    build_institutional,
    build_revisions,
    build_short,
)
# `activist` (congressional trading + 13D beneficial ownership) was tested
# 2026-09-07/08 and rejected: net-negative standalone IC/IR over 2020-2026,
# and folding its subs into `institutional` degrades that parent's realized
# IC/IR despite ranking well on the selector's own in-sample metric (see
# output/activist_eqeff_study/). Builder kept in library_activist.py — not
# registered here so it no longer enters any battery/selection run. The
# ingestion pipeline (data/beneficial_ownership.py, data/congressional_trades.py)
# stays in place in case data quality/coverage improves enough to re-test.

CANDIDATE_BUILDERS: dict[str, ParentBuilder] = {
    "momentum": build_momentum,
    "value": build_value,
    "quality": build_quality,
    "growth": build_growth,
    "revisions": build_revisions,
    "institutional": build_institutional,
    "insider": build_insider,
    "short": build_short,
}


def iter_parents() -> list[str]:
    """The canonical parent order used by every consumer + the report."""
    return list(CANDIDATE_BUILDERS.keys())


def all_candidates(ctx: DataContext) -> list[Candidate]:
    """Every candidate across every parent for one ``as_of``."""
    out: list[Candidate] = []
    for parent, builder in CANDIDATE_BUILDERS.items():
        try:
            out.extend(builder(ctx))
        except Exception as exc:  # noqa: BLE001 — factor failures are individual
            # Emit an empty Series so downstream code sees the name but the raw
            # is uniformly missing; the report will surface this via coverage.
            from data.utils import get_logger
            get_logger("subfactor_expansion.library").warning(
                "candidate build failed for %s: %s", parent, exc)
    return out


__all__ = [
    "Candidate", "CANDIDATE_BUILDERS", "iter_parents", "all_candidates",
    "build_momentum", "build_value", "build_quality", "build_growth",
    "build_revisions", "build_institutional", "build_insider", "build_short",
]


if __name__ == "__main__":
    # Smoke test: run every builder against a live DataContext and print counts.
    ctx = DataContext(get_db())
    try:
        print(f"built {len(all_candidates(ctx))} candidates across "
              f"{len(iter_parents())} parents")
    finally:
        ctx.close()
