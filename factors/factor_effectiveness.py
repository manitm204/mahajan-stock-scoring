"""Per-factor predictive-power metrics over multiple lookback windows.

For each completed holding period the engine pairs a factor's cross-sectional
parent score at the *start* of the period with the realized forward return over
it, then measures three complementary things:

* **Information Coefficient (IC)** — the Spearman rank correlation of score vs
  forward return. The cleanest single read on "did high scores lead to high
  returns"; it is the primary signal.
* **hit rate** — the fraction of periods whose IC was positive. A factor can
  have a respectable mean IC driven by one lucky month; a high hit rate says the
  edge showed up *often*.
* **factor spread** — mean forward return of the top score quintile minus the
  bottom quintile. Unlike IC (a rank correlation) this is in return units, so it
  captures whether the tails — the names we actually trade — separated.

Each metric is summarized over trailing windows of 1, 3, 6 and 12 periods and
then collapsed with **recency weights** (shorter windows count more) so recent
behaviour dominates without throwing away the longer-horizon view. Nothing here
looks ahead: a caller only ever feeds periods that completed on/before the date
being weighted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

WINDOWS = (1, 3, 6, 12)
# Recency-tilted window weights: the most recent month appears in every window,
# so it is implicitly up-weighted; the 12-month bucket is the only one that sees
# year-old data and is weighted least. Renormalized over available windows.
DEFAULT_WINDOW_WEIGHTS = {1: 0.15, 3: 0.30, 6: 0.30, 12: 0.25}
DEFAULT_MIN_NAMES = 20
DEFAULT_QUINTILE = 0.20

# Combination of the three signals into one effectiveness number. IC leads;
# hit rate and spread confirm. Applied to cross-sectional z-scores so the mix of
# units (correlation / probability / return) is unit-free and comparable.
_W_IC, _W_HIT, _W_SPREAD = 0.60, 0.20, 0.20


@dataclass
class PeriodObs:
    """One completed period's predictive read for a single factor."""

    end_date: str
    ic: float
    spread: float


@dataclass
class FactorMetrics:
    """Windowed predictive-power summary for one factor."""

    key: str
    n_obs: int
    ic_by_window: dict[int, float] = field(default_factory=dict)
    hit_by_window: dict[int, float] = field(default_factory=dict)
    spread_by_window: dict[int, float] = field(default_factory=dict)
    n_by_window: dict[int, int] = field(default_factory=dict)
    ic_recency: float = float("nan")
    hit_recency: float = float("nan")
    spread_recency: float = float("nan")
    ic_consistency: float = float("nan")   # sign agreement across windows [0,1]
    effectiveness: float = float("nan")    # filled by compute_effectiveness


def spearman_ic(
    scores: pd.Series | None, fwd: pd.Series, min_names: int = DEFAULT_MIN_NAMES
) -> float | None:
    """Cross-sectional Spearman rank IC of ``scores`` vs forward returns."""
    if scores is None:
        return None
    df = pd.DataFrame({"s": scores, "f": fwd}).dropna()
    if len(df) < min_names or df["s"].nunique() < 2:
        return None
    ic = df["s"].corr(df["f"], method="spearman")
    return None if pd.isna(ic) else float(ic)


def quintile_spread(
    scores: pd.Series | None,
    fwd: pd.Series,
    q: float = DEFAULT_QUINTILE,
    min_names: int = DEFAULT_MIN_NAMES,
) -> float | None:
    """Mean forward return of the top score quantile minus the bottom one.

    Positive means high-scoring names out-returned low-scoring names — the
    tradable edge in the tails, expressed in return units.
    """
    if scores is None:
        return None
    df = pd.DataFrame({"s": scores, "f": fwd}).dropna()
    if len(df) < min_names or df["s"].nunique() < 2:
        return None
    k = max(1, int(round(len(df) * q)))
    ordered = df.sort_values("s")
    bottom = ordered["f"].iloc[:k].mean()
    top = ordered["f"].iloc[-k:].mean()
    if pd.isna(top) or pd.isna(bottom):
        return None
    return float(top - bottom)


def _trailing(values: list[float], w: int) -> list[float]:
    """Last ``w`` finite values (fewer if history is shorter)."""
    finite = [v for v in values if v is not None and not np.isnan(v)]
    return finite[-w:]


def summarize_factor(
    key: str,
    obs: list[PeriodObs],
    *,
    windows: tuple[int, ...] = WINDOWS,
    window_weights: dict[int, float] | None = None,
) -> FactorMetrics:
    """Collapse a factor's period observations into windowed + recency metrics."""
    ww = window_weights or DEFAULT_WINDOW_WEIGHTS
    ics = [o.ic for o in obs]
    spreads = [o.spread for o in obs]
    m = FactorMetrics(key=key, n_obs=len([i for i in ics if not np.isnan(i)]))

    for w in windows:
        ic_w = _trailing(ics, w)
        sp_w = _trailing(spreads, w)
        m.n_by_window[w] = len(ic_w)
        m.ic_by_window[w] = float(np.mean(ic_w)) if ic_w else float("nan")
        m.hit_by_window[w] = (
            float(np.mean([1.0 if v > 0 else 0.0 for v in ic_w])) if ic_w else float("nan")
        )
        m.spread_by_window[w] = float(np.mean(sp_w)) if sp_w else float("nan")

    m.ic_recency = _recency_blend(m.ic_by_window, ww, windows)
    m.hit_recency = _recency_blend(m.hit_by_window, ww, windows)
    m.spread_recency = _recency_blend(m.spread_by_window, ww, windows)
    m.ic_consistency = _sign_agreement(m.ic_by_window, windows)
    return m


def _recency_blend(
    by_window: dict[int, float], ww: dict[int, float], windows: tuple[int, ...]
) -> float:
    num = 0.0
    den = 0.0
    for w in windows:
        v = by_window.get(w, float("nan"))
        weight = ww.get(w, 0.0)
        if weight > 0 and not np.isnan(v):
            num += weight * v
            den += weight
    return num / den if den > 0 else float("nan")


def _sign_agreement(by_window: dict[int, float], windows: tuple[int, ...]) -> float:
    signs = [np.sign(by_window[w]) for w in windows
             if w in by_window and not np.isnan(by_window[w]) and by_window[w] != 0.0]
    if not signs:
        return float("nan")
    return float(abs(np.mean(signs)))


def compute_effectiveness(
    metrics: dict[str, FactorMetrics],
    *,
    weights: tuple[float, float, float] = (_W_IC, _W_HIT, _W_SPREAD),
) -> dict[str, float]:
    """Cross-sectional effectiveness score per factor (mutates ``.effectiveness``).

    Each of the three recency signals is z-scored *across the factor cross
    section* so it is unit-free, then blended. The result is a relative measure —
    "which factors have been predicting better than the others lately" — which is
    exactly what a tilt around a fixed baseline needs. Factors with no usable
    history get effectiveness 0 (no tilt) rather than dragging the z-scores.
    """
    keys = list(metrics.keys())
    ic = np.array([metrics[k].ic_recency for k in keys], dtype=float)
    hit = np.array([metrics[k].hit_recency for k in keys], dtype=float)
    spread = np.array([metrics[k].spread_recency for k in keys], dtype=float)

    z_ic = _zscore(ic)
    z_hit = _zscore(hit - 0.5)          # center hit rate on coin-flip
    z_spread = _zscore(spread)

    w_ic, w_hit, w_spread = weights
    out: dict[str, float] = {}
    for i, k in enumerate(keys):
        eff = w_ic * z_ic[i] + w_hit * z_hit[i] + w_spread * z_spread[i]
        eff = 0.0 if np.isnan(eff) else float(eff)
        metrics[k].effectiveness = eff
        out[k] = eff
    return out


def _zscore(arr: np.ndarray) -> np.ndarray:
    """Population z-score with NaNs mapped to 0 (neutral) and a 0-std guard."""
    mask = ~np.isnan(arr)
    out = np.zeros_like(arr, dtype=float)
    if mask.sum() < 2:
        return out
    vals = arr[mask]
    mu = vals.mean()
    sd = vals.std(ddof=0)
    if sd <= 1e-12:
        return out
    out[mask] = (vals - mu) / sd
    return out
