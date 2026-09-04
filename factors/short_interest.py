"""Short Interest factor (sentiment / risk confirmation).

For long candidates, less short exposure and *declining* short interest are
favorable, so every sub-factor is inverted (lower raw -> higher score). The
factor is confirmation, not a primary signal — hence its small composite weight.

Separately, :func:`squeeze_warnings` flags names where heavy short interest
coincides with strong momentum (the classic squeeze setup). That is an
informational risk flag surfaced to the operator; it never alters scores.
"""
from __future__ import annotations

import pandas as pd

from .base import Factor, SubFactor
from .utils import DataContext, col


class ShortInterestFactor(Factor):
    name = "short"

    def compute(self, ctx: DataContext) -> list[SubFactor]:
        si = ctx.short_interest()
        return [
            SubFactor("si_short_pct_float", col(si, "short_percent_of_float"), higher_is_better=False),
            SubFactor("si_days_to_cover", col(si, "days_to_cover"), higher_is_better=False),
            SubFactor("si_short_interest_change", col(si, "short_interest_change"), higher_is_better=False),
        ]


def squeeze_warnings(
    ctx: DataContext,
    momentum_score: pd.Series,
    short_pct_threshold: float = 0.20,
    momentum_score_threshold: float = 70.0,
) -> list[str]:
    """Tickers with high short interest *and* strong momentum (squeeze risk)."""
    si = ctx.short_interest()
    if si.empty:
        return []
    short_pct = col(si, "short_percent_of_float").reindex(ctx.universe)
    flag = col(si, "short_squeeze_risk_flag").reindex(ctx.universe).fillna(0) > 0
    heavy = (short_pct >= short_pct_threshold) | flag
    strong = momentum_score.reindex(ctx.universe) >= momentum_score_threshold
    return sorted(short_pct.index[(heavy & strong).fillna(False)])
