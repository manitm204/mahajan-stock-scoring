"""Smooth VIX regime probabilities, shrinkage, and the Expected IC blend.

Pure functions only — no ScorePanel/DB dependency, so every function here takes
plain floats/Series and is testable in isolation. See
docs/superpowers/specs/2026-07-11-hierarchical-regime-aware-factor-model-design.md
Section 1 for the derivation.

Replaces the hard vix < 15 / 15-25 / > 25 cutoffs in vix_regime_study.py with a
soft 3-way split: two logistic sigmoids centered at the same boundaries (15, 25)
give a probability vector that always sums to 1, instead of one hard label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.walkforward.vix_regime_study import REGIME_ORDER

REGIME_LOW = 15.0
REGIME_HIGH = 25.0
TRANSITION_WIDTH = 4.0     # sigmoid width; larger = smoother regime transition
SHRINKAGE_K = 24.0         # default shrinkage constant (spec Section 2)


def sigmoid(x: float) -> float:
    """Standard logistic function, 1/(1+e^-x)."""
    return 1.0 / (1.0 + np.exp(-x))


def regime_probabilities(
    vix: float, *, low: float = REGIME_LOW, high: float = REGIME_HIGH,
    width: float = TRANSITION_WIDTH,
) -> dict[str, float]:
    """Soft (Low, Medium, High) membership for one VIX level. Always sums to 1.

    p_low = 1 - sigmoid((vix-low)/width); p_high = sigmoid((vix-high)/width);
    p_med = the remainder. p_med >= 0 always because sigmoid is monotonic and
    low < high, so sigmoid((vix-low)/width) >= sigmoid((vix-high)/width).
    """
    if not np.isfinite(vix):
        return {r: float("nan") for r in REGIME_ORDER}
    s1 = sigmoid((vix - low) / width)
    s2 = sigmoid((vix - high) / width)
    return {REGIME_ORDER[0]: 1.0 - s1, REGIME_ORDER[1]: s1 - s2, REGIME_ORDER[2]: s2}


def shrink_regime_ic(
    regime_ic: float, long_run_ic: float, n_eff: float, k: float = SHRINKAGE_K,
) -> float:
    """lambda = n_eff/(n_eff+k); shrunk = lambda*regime_ic + (1-lambda)*long_run_ic.

    Falls back to long_run_ic when there's no usable regime observation (n_eff<=0
    or regime_ic is NaN) — an empty/thin regime should never inject noise.
    """
    if n_eff <= 0 or not np.isfinite(regime_ic):
        return long_run_ic
    lam = n_eff / (n_eff + k)
    return lam * regime_ic + (1.0 - lam) * long_run_ic


def effective_regime_stats(
    vix_by_date: pd.Series, ic_by_date: pd.Series, *, width: float = TRANSITION_WIDTH,
) -> dict[str, dict[str, float]]:
    """Probability-weighted (n_eff, regime_ic) per regime over the dates given.

    Both Series are indexed by date and must already be filtered by the caller to
    date <= cutoff (this function has no notion of "now" — it aggregates whatever
    it's handed). n_eff is the probability-weighted effective sample size; regime_ic
    is the probability-weighted mean IC. NaN ICs are skipped.
    """
    out = {r: {"n_eff": 0.0, "regime_ic": float("nan")} for r in REGIME_ORDER}
    idx = vix_by_date.index.intersection(ic_by_date.index)
    if idx.empty:
        return out
    weighted_ic = {r: 0.0 for r in REGIME_ORDER}
    n_eff = {r: 0.0 for r in REGIME_ORDER}
    for d in idx:
        ic = ic_by_date.loc[d]
        if pd.isna(ic):
            continue
        probs = regime_probabilities(float(vix_by_date.loc[d]), width=width)
        for r in REGIME_ORDER:
            p = probs[r]
            if not np.isfinite(p):
                continue
            n_eff[r] += p
            weighted_ic[r] += p * float(ic)
    for r in REGIME_ORDER:
        out[r]["n_eff"] = n_eff[r]
        out[r]["regime_ic"] = weighted_ic[r] / n_eff[r] if n_eff[r] > 1e-9 else float("nan")
    return out


def expected_ic(
    long_run_ic: float, recent_ic: float, vix_now: float,
    regime_stats: dict[str, dict[str, float]], *,
    k: float = SHRINKAGE_K, width: float = TRANSITION_WIDTH,
) -> float:
    """expected_ic = 0.50*long_run + 0.25*recent + 0.25*prob-weighted shrunk regime IC.

    ``regime_stats`` is the output of :func:`effective_regime_stats` (already
    computed over date <= cutoff). ``recent_ic`` falls back to ``long_run_ic`` when
    NaN (e.g. fewer than 24 months of history so far — see spec Section 1).
    """
    if not np.isfinite(long_run_ic):
        return float("nan")
    recent = recent_ic if np.isfinite(recent_ic) else long_run_ic
    probs_now = regime_probabilities(vix_now, width=width)
    regime_component = 0.0
    total_p = 0.0
    for r in REGIME_ORDER:
        p = probs_now[r]
        if not np.isfinite(p):
            continue
        stats = regime_stats.get(r, {"n_eff": 0.0, "regime_ic": float("nan")})
        shrunk = shrink_regime_ic(stats["regime_ic"], long_run_ic, stats["n_eff"], k=k)
        regime_component += p * shrunk
        total_p += p
    if total_p <= 1e-9:
        regime_component = long_run_ic
    else:
        regime_component /= total_p
    return 0.50 * long_run_ic + 0.25 * recent + 0.25 * regime_component
