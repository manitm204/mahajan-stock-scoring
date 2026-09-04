"""Quality factor.

Identifies profitable, financially healthy businesses with durable economics.
Combines point-in-time ratios (ROE, margins, cash conversion, leverage) with two
classic composite diagnostics computed here from raw annual statements:

* **Piotroski F-Score** (0–9): nine year-over-year fundamental health tests.
  7–9 strong / 4–6 neutral / 1–3 weak.
* **Altman Z-Score**: distress model. >2.99 safe / 1.81–2.99 grey / <1.81
  distress. It is *not* meaningful for balance-sheet-driven sectors, so it is
  set neutral (NaN -> 50) for Financials and Real Estate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor, SubFactor
from .utils import ALTMAN_EXCLUDED_SECTORS, DataContext, col


class QualityFactor(Factor):
    name = "quality"

    def compute(self, ctx: DataContext) -> list[SubFactor]:
        ff = ctx.fund_features_latest()
        f = ctx.fund_annual_latest()
        prior = ctx.fund_annual_prior()

        # Ratio guards: ROE goes *positive* when net income and equity are both
        # negative, CFO/NI flips sign when net income is negative, and D/E flips
        # when equity is negative — in each case the broken sign would reward
        # the distressed name. Mask the meaningless cases to NaN (neutral 50)
        # and rank D/E against |equity| so negative-equity names stay penalized.
        equity = col(f, "shareholder_equity")
        ni = col(f, "net_income")
        roe = col(ff, "roe").mask(equity <= 0)
        gross_margin = col(ff, "gross_margin")
        cfo_to_ni = col(ff, "cfo_to_net_income").mask(ni <= 0)
        debt_to_equity = col(f, "debt") / equity.abs().replace(0, np.nan)

        roe_stability = _annual_dispersion(ctx, "roe")            # lower std better
        gross_margin_trend = _annual_trend(ctx, "gross_margin")   # rising better

        # Accruals: low (net income not backed by cash) is a red flag.
        accruals = (col(f, "net_income") - col(f, "operating_cash_flow")) / col(f, "total_assets").replace(0, np.nan)

        piotroski = _piotroski(f, prior)
        altman = _altman(f, ctx.market_cap(), ctx.sectors())

        return [
            SubFactor("qual_roe", roe, higher_is_better=True),
            SubFactor("qual_roe_stability", roe_stability, higher_is_better=False),
            SubFactor("qual_gross_margin", gross_margin, higher_is_better=True),
            SubFactor("qual_gross_margin_trend", gross_margin_trend, higher_is_better=True),
            SubFactor("qual_cfo_to_ni", cfo_to_ni, higher_is_better=True),
            SubFactor("qual_accruals_inv", accruals, higher_is_better=False),
            SubFactor("qual_debt_to_equity_inv", debt_to_equity, higher_is_better=False),
            SubFactor("qual_piotroski_f", piotroski, higher_is_better=True),
            SubFactor("qual_altman_z", altman, higher_is_better=True),
        ]


def _annual_dispersion(ctx: DataContext, metric: str) -> pd.Series:
    """Std-dev of an annual fundamental-feature metric over its history."""
    ffa = ctx.fund_features_annual()
    if ffa.empty or metric not in ffa.columns:
        return pd.Series(dtype=float)
    return ffa.groupby("ticker")[metric].std().reindex(ctx.universe)


def _annual_trend(ctx: DataContext, metric: str, years: int = 4) -> pd.Series:
    """Latest-minus-earliest change of a metric over the last ``years`` annuals."""
    ffa = ctx.fund_features_annual()
    if ffa.empty or metric not in ffa.columns:
        return pd.Series(dtype=float)

    def _trend(g: pd.Series) -> float:
        vals = g.dropna().tail(years).values
        return float(vals[-1] - vals[0]) if len(vals) >= 2 else np.nan

    return ffa.groupby("ticker")[metric].apply(_trend).reindex(ctx.universe)


def _piotroski(cur: pd.DataFrame, prior: pd.DataFrame) -> pd.Series:
    """Nine-test Piotroski F-Score from current vs prior annual statements."""
    if cur.empty or prior.empty:
        return pd.Series(dtype=float)
    c, p = cur, prior.reindex(cur.index)

    def g(df: pd.DataFrame, name: str) -> pd.Series:
        return col(df, name)

    assets_c, assets_p = g(c, "total_assets"), g(p, "total_assets")
    ni_c, ni_p = g(c, "net_income"), g(p, "net_income")
    cfo_c = g(c, "operating_cash_flow")
    roa_c = ni_c / assets_c.replace(0, np.nan)
    roa_p = ni_p / assets_p.replace(0, np.nan)
    cr_c = g(c, "current_assets") / g(c, "current_liabilities").replace(0, np.nan)
    cr_p = g(p, "current_assets") / g(p, "current_liabilities").replace(0, np.nan)
    lev_c = g(c, "debt") / assets_c.replace(0, np.nan)
    lev_p = g(p, "debt") / assets_p.replace(0, np.nan)
    gm_c = g(c, "gross_profit") / g(c, "revenue").replace(0, np.nan)
    gm_p = g(p, "gross_profit") / g(p, "revenue").replace(0, np.nan)
    turn_c = g(c, "revenue") / assets_c.replace(0, np.nan)
    turn_p = g(p, "revenue") / assets_p.replace(0, np.nan)
    sh_c = ni_c / g(c, "eps_diluted").replace(0, np.nan)
    sh_p = ni_p / g(p, "eps_diluted").replace(0, np.nan)

    tests = [
        roa_c > 0,
        cfo_c > 0,
        roa_c > roa_p,
        cfo_c / assets_c.replace(0, np.nan) > roa_c,   # accruals
        lev_c < lev_p,
        cr_c > cr_p,
        sh_c <= sh_p * 1.01,                           # no meaningful dilution
        gm_c > gm_p,
        turn_c > turn_p,
    ]
    score = sum(t.astype(float) for t in tests)
    # Require a usable current statement; rows lacking prior comparatives still
    # yield a (lower) score but are not invented out of nothing.
    score = score.where(assets_c.notna() & ni_c.notna())
    return score.reindex(cur.index)


def _altman(f: pd.DataFrame, market_cap: pd.Series, sectors: pd.Series) -> pd.Series:
    """Altman Z-Score; neutral (NaN) for sectors where the model is unreliable."""
    if f.empty:
        return pd.Series(dtype=float)
    assets = col(f, "total_assets").replace(0, np.nan)
    # working_capital is sparsely populated upstream; derive it from current
    # assets minus current liabilities (both fully populated) when absent.
    working_capital = col(f, "working_capital").fillna(
        col(f, "current_assets") - col(f, "current_liabilities"))
    x1 = working_capital / assets
    x2 = col(f, "retained_earnings") / assets
    x3 = col(f, "ebit") / assets
    x4 = market_cap.reindex(f.index) / col(f, "total_liabilities").replace(0, np.nan)
    x5 = col(f, "revenue") / assets
    z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
    excluded = sectors.reindex(f.index).isin(ALTMAN_EXCLUDED_SECTORS)
    return z.mask(excluded)
