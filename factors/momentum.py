"""Momentum factor.

Captures persistent stock-specific momentum while damping the two effects that
make raw momentum fragile: single-name volatility and sector beta. Horizon
returns are measured from point-in-time prices; 52-week-high proximity and
sector relative strength reuse Layer 1 price features.
"""
from __future__ import annotations

import numpy as np

from .base import Factor, SubFactor
from .utils import DataContext, col

# Trading-day horizons (approximate calendar months).
_MONTH = 21
_QUARTER = 63
_HALF = 126
_YEAR = 252


class MomentumFactor(Factor):
    name = "momentum"

    def compute(self, ctx: DataContext) -> list[SubFactor]:
        pf = ctx.price_features()

        # 12-1 month: full-year return skipping the most recent month, the
        # classic momentum construct that drops short-term mean reversion.
        ret_12_1 = ctx.horizon_return(_YEAR - _MONTH, offset=_MONTH)
        ret_6m = ctx.horizon_return(_HALF)
        ret_3m = ctx.horizon_return(_QUARTER)

        # Acceleration: most recent quarter return minus the prior quarter's.
        accel = ctx.horizon_return(_QUARTER) - ctx.horizon_return(_QUARTER, offset=_QUARTER)

        high_prox = col(pf, "distance_from_52w_high")     # 0 = at high (best)
        sector_rs = col(pf, "sector_relative_strength")
        vol = col(pf, "volatility_20d")

        # Volatility-adjusted momentum: reward smooth trends over jumpy ones.
        vol_adj = ret_12_1 / vol.replace(0, np.nan)

        return [
            SubFactor("mom_12_1", ret_12_1, higher_is_better=True),
            SubFactor("mom_6m", ret_6m, higher_is_better=True),
            SubFactor("mom_3m", ret_3m, higher_is_better=True),
            SubFactor("mom_acceleration", accel, higher_is_better=True),
            SubFactor("mom_52w_high_proximity", high_prox, higher_is_better=True),
            SubFactor("mom_sector_rel_strength", sector_rs, higher_is_better=True),
            SubFactor("mom_vol_adjusted", vol_adj, higher_is_better=True),
        ]
