"""Insider Activity factor (management conviction).

Aggregates Form 4 transactions over a trailing window with the prescribed code
treatment: open-market purchases (P) are positive, sales (S) mildly negative,
and option/award mechanics (A, M, F) neutral.

Construction note — the earlier version keyed almost entirely on *open-market
buys*, which are episodic: in any trailing window only ~25% of names have one, so
three of four sub-factors sat pinned at neutral and the parent barely
differentiated the universe. This build leads with **continuous, widely-populated
signals** — signed dollar flow and a buy/sell balance ratio — which are defined
for any name with *any* insider activity (~90% of the universe), and folds the
rare high-conviction buy patterns into a single secondary flag instead of three.
Names with no insider data in the window stay missing -> neutral 50.
"""
from __future__ import annotations

import pandas as pd

from .base import Factor, SubFactor
from .utils import DataContext

_SALE_WEIGHT = 0.5          # sales count, but less than purchases
_CEO_CFO_TERMS = ("CEO", "CHIEF EXECUTIVE", "CFO", "CHIEF FINANCIAL")
_LARGE_PURCHASE_USD = 1_000_000
_CLUSTER_MIN_BUYERS = 3


class InsiderFactor(Factor):
    name = "insider"

    def compute(self, ctx: DataContext) -> list[SubFactor]:
        agg = _aggregate(ctx)
        return [
            SubFactor("ins_net_dollar_flow", agg["net_flow"], higher_is_better=True),
            SubFactor("ins_buy_sell_ratio", agg["buy_sell_ratio"], higher_is_better=True),
            SubFactor("ins_high_conviction_buy", agg["high_conviction"], higher_is_better=True),
        ]


def _aggregate(ctx: DataContext) -> dict[str, pd.Series]:
    df = ctx.insider_window().copy()
    empty = pd.Series(dtype=float)
    keys = ("net_flow", "buy_sell_ratio", "high_conviction")
    if df.empty:
        return {k: empty for k in keys}

    df["code"] = df["transaction_code"].fillna("").str.upper().str.strip()
    df["value"] = pd.to_numeric(df["value"], errors="coerce").abs().fillna(0.0)
    df["title"] = df["insider_title"].fillna("").str.upper()
    is_buy = df["code"] == "P"
    is_sell = df["code"] == "S"

    # Signed dollar flow: purchases add, sales subtract (weighted), rest neutral.
    df["signed"] = 0.0
    df.loc[is_buy, "signed"] = df.loc[is_buy, "value"]
    df.loc[is_sell, "signed"] = -_SALE_WEIGHT * df.loc[is_sell, "value"]
    net_flow = df.groupby("ticker")["signed"].sum()

    # Buy/sell balance in [0, 1]: 1 = all open-market buying, 0 = all selling.
    # Defined for any name with open-market activity (P or S) — the wide-coverage
    # signal that de-compresses the factor.
    buy_usd = df[is_buy].groupby("ticker")["value"].sum()
    sell_usd = df[is_sell].groupby("ticker")["value"].sum()
    traded = df[is_buy | is_sell]
    active_ps = pd.Index(traded["ticker"].unique())
    buy_usd = buy_usd.reindex(active_ps, fill_value=0.0)
    sell_usd = sell_usd.reindex(active_ps, fill_value=0.0)
    gross = buy_usd + sell_usd
    buy_sell_ratio = (buy_usd / gross.where(gross > 0)).reindex(active_ps)

    # High-conviction buy: any CEO/CFO purchase, a buyer cluster, or a >$1M buy.
    # One combined flag (0/1) instead of three sparse ones. Reindexed to every
    # name with *any* insider activity so "activity but no conviction buy" scores
    # the low end (0), while names absent from the window stay missing (neutral).
    buys = df[is_buy]
    ceo_cfo = set(buys[buys["title"].str.contains("|".join(_CEO_CFO_TERMS), regex=True)]["ticker"])
    large = set(buys[buys["value"] >= _LARGE_PURCHASE_USD]["ticker"])
    cluster = set(buys.groupby("ticker")["insider_name"].nunique()
                  .loc[lambda s: s >= _CLUSTER_MIN_BUYERS].index)
    conviction_tickers = ceo_cfo | large | cluster
    active_any = pd.Index(df["ticker"].unique())
    high_conviction = pd.Series(
        [1.0 if t in conviction_tickers else 0.0 for t in active_any],
        index=active_any, dtype=float)

    return {
        "net_flow": net_flow,
        "buy_sell_ratio": buy_sell_ratio,
        "high_conviction": high_conviction,
    }
