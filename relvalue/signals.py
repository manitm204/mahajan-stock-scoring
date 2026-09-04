"""Spread definitions → standardized z-series over a trading span.

Every builder consumes ONLY formation-estimated parameters (frozen in the
PairSpec) plus trailing prices; rolling variants use only past data by
construction. Signals are read at close t and executed at close t+1 by the
engine, so same-day z uses no future information.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PairSpec:
    a: str
    b: str
    family: str          # ssd | coint | setf | etf | dual
    sector: str = ""
    # formation-frozen parameters
    alpha: float = 0.0
    beta: float = 1.0    # hedge ratio of log(A) on log(B)
    mu: float = 0.0      # formation mean of the spread measure
    sigma: float = 1.0   # formation std of the spread measure
    diag: dict = field(default_factory=dict, hash=False, compare=False)


def z_distance(pa: pd.Series, pb: pd.Series, anchor: str, spec: PairSpec) -> pd.Series:
    """GGR: normalized-price difference / formation sigma (sigma in spec)."""
    na = pa / pa.loc[anchor]
    nb = pb / pb.loc[anchor]
    return (na - nb) / spec.sigma


def z_residual(pa: pd.Series, pb: pd.Series, anchor: str, spec: PairSpec) -> pd.Series:
    """OLS residual log(A) - (alpha + beta·log(B)), standardized by formation."""
    resid = np.log(pa) - (spec.alpha + spec.beta * np.log(pb))
    return (resid - spec.mu) / spec.sigma


def z_logspread(pa: pd.Series, pb: pd.Series, anchor: str, spec: PairSpec) -> pd.Series:
    """log(A) - log(B), standardized by formation mean/std."""
    return (np.log(pa) - np.log(pb) - spec.mu) / spec.sigma


def z_ratio(pa: pd.Series, pb: pd.Series, anchor: str, spec: PairSpec,
            ma: int = 20, ma2: int = 20, vol: int = 20) -> pd.Series:
    """User baseline spec: ratio − MA(20), de-meaned by MA(20) of the diff,
    scaled by rolling std(20) of the diff. Fully rolling → PIT; the input
    series must include ≥ ma+ma2+vol days before the trading span."""
    ratio = pa / pb
    diff = ratio - ratio.rolling(ma).mean()
    z = (diff - diff.rolling(ma2).mean()) / diff.rolling(vol).std()
    return z.replace([np.inf, -np.inf], np.nan)


BUILDERS = {
    "distance": z_distance,
    "residual": z_residual,
    "logspread": z_logspread,
    "ratio": z_ratio,
}
