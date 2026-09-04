"""Growth factor.

Identifies improving businesses with sustained, and ideally accelerating,
top- and bottom-line growth plus reinvestment intensity. Level and CAGR metrics
reuse Layer 1 fundamental features; FCF growth and acceleration are derived from
the annual statement history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor, SubFactor
from .utils import DataContext, col, yoy_change


class GrowthFactor(Factor):
    name = "growth"

    def compute(self, ctx: DataContext) -> list[SubFactor]:
        ff = ctx.fund_features_latest()
        f = ctx.fund_annual_latest()
        prior = ctx.fund_annual_prior().reindex(f.index)

        rev_growth = col(ff, "revenue_growth_yoy")
        eps_growth = col(ff, "eps_growth_yoy")
        rev_cagr_3y = col(ff, "revenue_cagr_3y")

        fcf_c, fcf_p = col(f, "free_cash_flow"), col(prior, "free_cash_flow")
        fcf_growth = (fcf_c - fcf_p) / fcf_p.abs().replace(0, np.nan)

        rev_accel = _yoy_acceleration(ctx, "revenue_growth_yoy")

        rnd = col(f, "rnd_expense") / col(f, "revenue").replace(0, np.nan)

        return [
            SubFactor("grw_revenue_yoy", rev_growth, higher_is_better=True),
            SubFactor("grw_earnings_yoy", eps_growth, higher_is_better=True),
            SubFactor("grw_fcf_yoy", fcf_growth, higher_is_better=True),
            SubFactor("grw_revenue_acceleration", rev_accel, higher_is_better=True),
            SubFactor("grw_revenue_cagr_3y", rev_cagr_3y, higher_is_better=True),
            SubFactor("grw_rd_intensity", rnd, higher_is_better=True),
        ]


def _yoy_acceleration(ctx: DataContext, metric: str) -> pd.Series:
    """Change in an annual growth-rate metric (this year's rate minus last)."""
    ffa = ctx.fund_features_annual()
    if ffa.empty or metric not in ffa.columns:
        return pd.Series(dtype=float)
    return ffa.groupby("ticker")[metric].apply(yoy_change).reindex(ctx.universe)
