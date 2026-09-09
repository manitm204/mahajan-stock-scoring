"""Candidate subfactors for the new "activist" parent: congressional trading
(STOCK Act disclosures) and 13D activist beneficial-ownership stakes.

Two independent "informed trader" populations, treated like the insider
parent (raw window aggregation, no percentile step here — the panel handles
that). See ``data/congressional_trades.py`` and ``data/beneficial_ownership.py``
for the ingestion + PIT-gating rationale.

Known limitation carried into these candidates: FMP's beneficial-ownership
endpoint mixes Schedule 13D (activist, "intent to influence") and 13G
(passive, e.g. index-fund 5% crossings) filings with no schema field
distinguishing them. `filing_type` is parsed from the filing URL and is only
reliably populated for filings the SEC's XSL viewer served with a schedule
marker in the path — this favors recent years and is NOT a complete pre-2019
history. Every candidate here filters to `filing_type == '13D'` and accepts
that survivorship rather than mixing in unfiltered 13G noise.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from factors.utils import DataContext

# Late import to avoid the circular from library.py (same pattern as library_flow.py).
def _Candidate(*args, **kwargs):  # pragma: no cover — trivial passthrough
    from .library import Candidate
    return Candidate(*args, **kwargs)


_AMOUNT_RE = re.compile(r"[\d,]+")


def _amount_midpoint(amount_range: object) -> float:
    """Midpoint dollar estimate from FMP's disclosed range string.

    Handles "$1,001 - $15,000" (two numbers -> midpoint) and single-sided
    forms like "Over $50,000,000" (one number -> that number). Unparsable
    input returns NaN.
    """
    if not isinstance(amount_range, str):
        return np.nan
    nums = [float(n.replace(",", "")) for n in _AMOUNT_RE.findall(amount_range)]
    if len(nums) >= 2:
        return (nums[0] + nums[1]) / 2.0
    if len(nums) == 1:
        return nums[0]
    return np.nan


def _window_bounds(ctx: DataContext, days: int) -> tuple[str, str]:
    today = pd.Timestamp.today().date().isoformat()
    upper = min(ctx.as_of, today)
    start = (pd.Timestamp(upper) - pd.Timedelta(days=days)).date().isoformat()
    return upper, max(start, "1990-01-01")


# =============================================================================
# Congressional trading (STOCK Act)
# =============================================================================


def _congress_window_rows(ctx: DataContext, days: int) -> pd.DataFrame:
    upper, start = _window_bounds(ctx, days)
    # PIT: gate on disclosure_date, never transaction_date (up to 45d lag).
    df = ctx.db.query_df(
        "SELECT ticker, chamber, member_id, transaction_type, transaction_date, "
        "disclosure_date, amount_range FROM congressional_trades "
        "WHERE disclosure_date <= ? AND disclosure_date >= ?", (upper, start))
    return df[df["ticker"].isin(ctx.universe)] if not df.empty else df


def _congress_agg(df: pd.DataFrame, ctx: DataContext) -> dict[str, pd.Series]:
    empty = pd.Series(dtype=float, index=ctx.universe)
    keys = ("net_buy_ratio", "buyer_count", "purchase_dollar_vol", "purchase_count")
    if df.empty:
        return {k: empty.copy() for k in keys}

    d = df.copy()
    d["mid"] = d["amount_range"].apply(_amount_midpoint).fillna(0.0)
    is_buy = d["transaction_type"] == "Purchase"
    is_sell = d["transaction_type"] == "Sale"

    buy_usd = d[is_buy].groupby("ticker")["mid"].sum()
    sell_usd = d[is_sell].groupby("ticker")["mid"].sum()
    traded = d[is_buy | is_sell]
    active = pd.Index(traded["ticker"].unique())
    buy_a = buy_usd.reindex(active, fill_value=0.0)
    sell_a = sell_usd.reindex(active, fill_value=0.0)
    gross = buy_a + sell_a
    net_buy_ratio = ((buy_a - sell_a) / gross.where(gross > 0)).reindex(active)

    buys = d[is_buy]
    buyer_count = buys.groupby("ticker")["member_id"].nunique()
    purchase_dollar_vol = buy_usd.reindex(active, fill_value=0.0)
    purchase_count = buys.groupby("ticker").size().reindex(active, fill_value=0).astype(float)

    def _idx(s: pd.Series) -> pd.Series:
        return s.reindex(ctx.universe)

    return {
        "net_buy_ratio": _idx(net_buy_ratio),
        "buyer_count": _idx(buyer_count),
        "purchase_dollar_vol": _idx(purchase_dollar_vol),
        "purchase_count": _idx(purchase_count),
    }


# =============================================================================
# 13D activist beneficial ownership
# =============================================================================


def _beneficial_13d_rows(ctx: DataContext, days: int | None = None) -> pd.DataFrame:
    if days is None:
        upper, start = min(ctx.as_of, pd.Timestamp.today().date().isoformat()), "1990-01-01"
    else:
        upper, start = _window_bounds(ctx, days)
    df = ctx.db.query_df(
        "SELECT ticker, filing_date, percent_of_class FROM beneficial_ownership "
        "WHERE filing_type = '13D' AND filing_date <= ? AND filing_date >= ?",
        (upper, start))
    return df[df["ticker"].isin(ctx.universe)] if not df.empty else df


def _latest_13d_stake(ctx: DataContext) -> pd.Series:
    """Most recent 13D percent_of_class per ticker (any history), PIT-gated."""
    df = _beneficial_13d_rows(ctx, days=None)
    empty = pd.Series(dtype=float, index=ctx.universe)
    if df.empty:
        return empty
    df = df.sort_values(["ticker", "filing_date"]).drop_duplicates("ticker", keep="last")
    return df.set_index("ticker")["percent_of_class"].reindex(ctx.universe)


# =============================================================================
# Combined parent
# =============================================================================


def build_activist(ctx: DataContext) -> list:
    cong_180d = _congress_window_rows(ctx, days=180)
    cong_agg = _congress_agg(cong_180d, ctx)

    d13d_90 = _beneficial_13d_rows(ctx, days=90)
    d13d_365 = _beneficial_13d_rows(ctx, days=365)

    new_13d_flag = pd.Series(0.0, index=ctx.universe)
    if not d13d_90.empty:
        new_13d_flag.loc[new_13d_flag.index.isin(d13d_90["ticker"].unique())] = 1.0

    filing_count_365 = pd.Series(0.0, index=ctx.universe)
    if not d13d_365.empty:
        cnt = d13d_365.groupby("ticker").size()
        filing_count_365.loc[cnt.index] = cnt.astype(float)

    stake_pct = _latest_13d_stake(ctx)

    return [
        _Candidate("act_cong_net_buy_ratio_180d", cong_agg["net_buy_ratio"], True,
                   "activist", "congress_flow"),
        _Candidate("act_cong_buyer_count_180d", cong_agg["buyer_count"], True,
                   "activist", "congress_cluster"),
        _Candidate("act_cong_purchase_dollar_vol_180d", cong_agg["purchase_dollar_vol"], True,
                   "activist", "congress_flow"),
        _Candidate("act_cong_purchase_count_180d", cong_agg["purchase_count"], True,
                   "activist", "congress_flow"),
        _Candidate("act_new_13d_flag_90d", new_13d_flag, True, "activist", "13d_event"),
        _Candidate("act_13d_filing_count_365d", filing_count_365, True, "activist", "13d_event"),
        _Candidate("act_13d_stake_pct", stake_pct, True, "activist", "13d_level"),
    ]
