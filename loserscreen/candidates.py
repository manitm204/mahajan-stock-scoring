"""PIT candidate veto scores for the v5 battery (see package docstring).

All scores are oriented as GOODNESS (low = flagged), so the standard
bottom-percentile veto rule applies unchanged. Fundamental candidates use
annual reports with the production 90-day availability lag; price candidates
use trailing windows ending at the formation date (inclusive), the same
convention as composite scoring.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAG_DAYS = 90
PRIOR_GAP = (300, 430)      # fiscal-date gap defining "the prior annual report"
IDIO_WIN, IDIO_MIN = 252, 126
MAX5_WIN, MAX5_MIN, MAX5_K = 21, 15, 5
SPY = "SPY"


def _implied_shares(sub: pd.DataFrame) -> pd.Series:
    eps = sub["eps_diluted"].where(sub["eps_diluted"].abs() >= 0.01)
    sh = sub["net_income"] / eps
    return sh.where((sh > 1e6) & (sh < 1e11))


def fundamental_scores(db, dates: list[str],
                       universe: list[str]) -> dict[str, pd.DataFrame]:
    """{date: frame[net_issuance, asset_growth, accruals]} (goodness scores)."""
    df = db.query_df(
        "SELECT ticker, fiscal_date, net_income, eps_diluted, total_assets, "
        "operating_cash_flow FROM fundamentals WHERE period_type='annual'")
    df = df[df["ticker"].isin(set(universe))].copy()
    df["avail"] = (pd.to_datetime(df["fiscal_date"])
                   + pd.Timedelta(days=LAG_DAYS)).dt.strftime("%Y-%m-%d")
    df = df.sort_values(["ticker", "fiscal_date"], kind="mergesort")

    out: dict[str, pd.DataFrame] = {}
    for d in dates:
        av = df[df["avail"] <= d]
        if av.empty:
            continue
        t2 = av.groupby("ticker").tail(2)
        last = t2.groupby("ticker").last()
        first = t2.groupby("ticker").first()
        gap = (pd.to_datetime(last["fiscal_date"])
               - pd.to_datetime(first["fiscal_date"])).dt.days
        has_prior = gap.between(*PRIOR_GAP)

        sh0, sh1 = _implied_shares(last), _implied_shares(first)
        issuance = (sh0 / sh1 - 1.0).where(has_prior)

        ta0 = last["total_assets"].where(last["total_assets"] > 0)
        ta1 = first["total_assets"].where(first["total_assets"] > 0)
        agrow = (ta0 / ta1 - 1.0).where(has_prior)

        accr = (last["net_income"] - last["operating_cash_flow"]) / ta0

        out[d] = pd.DataFrame({"net_issuance": -issuance,
                               "asset_growth": -agrow,
                               "accruals": -accr})
    return out


def price_scores(matrix: pd.DataFrame,
                 dates: list[str]) -> dict[str, pd.DataFrame]:
    """{date: frame[idio_vol, max5]} (goodness scores)."""
    rets = matrix.pct_change()
    cols = [c for c in matrix.columns if c != SPY]
    out: dict[str, pd.DataFrame] = {}
    for d in dates:
        if d not in rets.index:
            continue
        pos = rets.index.get_loc(d)

        # idio vol: pairwise-complete regression on SPY over the 252d window
        w = rets.iloc[max(0, pos - IDIO_WIN + 1): pos + 1]
        w = w[w[SPY].notna()]
        s = w[SPY]
        x = w[cols]
        m = x.notna()
        n = m.sum()
        es = m.mul(s, axis=0).sum() / n
        es2 = m.mul(s * s, axis=0).sum() / n
        ex = x.mean()
        exs = x.mul(s, axis=0).mean()
        var_s = es2 - es ** 2
        beta = (exs - ex * es) / var_s
        resid_var = x.var(ddof=0) - beta ** 2 * var_s
        idio = np.sqrt(resid_var.clip(lower=0.0)) * np.sqrt(252)
        idio[n < IDIO_MIN] = np.nan

        # MAX5: mean of the 5 largest daily returns in the trailing 21 days
        w21 = rets.iloc[max(0, pos - MAX5_WIN + 1): pos + 1][cols]
        arr = w21.to_numpy()
        cnt = (~np.isnan(arr)).sum(axis=0)
        srt = np.sort(arr, axis=0)            # NaNs sort to the end
        max5 = np.full(len(cols), np.nan)
        for j in range(len(cols)):
            if cnt[j] >= MAX5_MIN:
                max5[j] = srt[cnt[j] - MAX5_K: cnt[j], j].mean()

        out[d] = pd.DataFrame({"idio_vol": -idio,
                               "max5": pd.Series(-max5, index=cols)})
    return out


def candidate_scores(db, matrix: pd.DataFrame, dates: list[str],
                     universe: list[str]) -> dict[str, pd.DataFrame]:
    fund = fundamental_scores(db, dates, universe)
    px = price_scores(matrix, dates)
    out: dict[str, pd.DataFrame] = {}
    for d in dates:
        parts = [f for f in (fund.get(d), px.get(d)) if f is not None]
        if parts:
            out[d] = pd.concat(parts, axis=1)
    return out
