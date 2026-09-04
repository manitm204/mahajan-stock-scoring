"""Point-in-time market caps for the cap-weighted books.

Mirrors production ``DataContext.implied_shares`` / ``market_cap``: per rebalance
date, take each ticker's latest annual report *available* on that date (fiscal
year end + ``lag_days`` reporting lag), prefer reported shares_outstanding, else
back shares out of net_income / diluted EPS with the same guards, and price at
the rebalance date. Names failing every guard get no cap weight (reported as
coverage in the study outputs).
"""
from __future__ import annotations

import pandas as pd

LAG_DAYS = 90


def market_caps(db, dates: list[str], universe: list[str],
                matrix: pd.DataFrame) -> dict[str, pd.Series]:
    df = db.query_df(
        "SELECT ticker, fiscal_date, net_income, eps_diluted, shares_outstanding "
        "FROM fundamentals WHERE period_type='annual'")
    df = df[df["ticker"].isin(set(universe))].copy()
    df["avail"] = (pd.to_datetime(df["fiscal_date"])
                   + pd.Timedelta(days=LAG_DAYS)).dt.strftime("%Y-%m-%d")
    df = df.sort_values(["ticker", "fiscal_date"], kind="mergesort")

    out: dict[str, pd.Series] = {}
    for d in dates:
        if d not in matrix.index:
            continue
        latest = df[df["avail"] <= d].groupby("ticker").last()
        if latest.empty:
            continue
        eps = latest["eps_diluted"]
        implied = latest["net_income"] / eps.where(eps.abs() >= 0.01)
        implied = implied.where((implied > 1e6) & (implied < 1e11))
        reported = latest["shares_outstanding"].where(
            latest["shares_outstanding"] > 1e6)
        shares = reported.fillna(implied)
        mc = (matrix.loc[d].reindex(shares.index) * shares).dropna()
        mc = mc[mc > 0].sort_index()
        if not mc.empty:
            out[d] = mc
    return out
