"""Factor-score overlays for the RV engine (PIT: latest rebalance ≤ entry day).

Wraps the production-faithful composite caches used by pairtrading.study; the
same carry-forward + look-ahead assertion applies (study.score_asof).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pairtrading.study import score_asof


class RVGate:
    """rule ∈ {tilt, mindiff, veto, none}; ETF legs have no score → treated as
    missing, and missing scores never block (counted by the caller)."""

    def __init__(self, composites: dict[str, pd.Series], rule: str,
                 min_diff: float = 10.0, long_min: float = 70.0,
                 short_max: float = 50.0):
        self.composites, self.rule, self.min_diff = composites, rule, min_diff
        self.long_min, self.short_max = long_min, short_max

    def __call__(self, day: str) -> pd.Series | None:
        return score_asof(self.composites, day)

    def allows(self, day: str) -> bool:      # regime hook (default: always)
        return True

    def passes(self, long_score: float, short_score: float,
               sc: pd.Series) -> bool:
        if self.rule == "none":
            return True
        if np.isnan(long_score) or np.isnan(short_score):
            return True
        if self.rule == "tilt":
            return long_score >= short_score
        if self.rule == "mindiff":
            return long_score - short_score >= self.min_diff
        if self.rule == "veto":
            lo, hi = sc.quantile(0.20), sc.quantile(0.80)
            return long_score > lo and short_score < hi
        if self.rule == "levels":
            return long_score >= self.long_min and short_score <= self.short_max
        raise ValueError(self.rule)


class HotStreakGate(RVGate):
    """Round-6 replicated rule: no fresh entries while the book's own trailing
    ``lookback``-day return exceeds ``thresh`` (PIT via shift)."""

    def __init__(self, composites, rule: str, daily: pd.Series,
                 lookback: int = 42, thresh: float = 0.01):
        super().__init__(composites, rule)
        trail = daily.sort_index().rolling(lookback).sum().shift(1)
        self.allow_s = ~(trail > thresh)

    def allows(self, day: str) -> bool:
        a = self.allow_s.loc[:day]
        return bool(a.iloc[-1]) if len(a) else True
