"""Factor Weight Engine — robust, regime-aware composite factor weights.

The job is to decide *how much* each parent factor (momentum, value, quality,
growth, …) should count in the composite at a given rebalance — without
overfitting to the recent past. Naively maximizing trailing returns would pour
weight into whatever just worked (momentum, in a bull market) and then get
whipsawed. This engine is built around the opposite philosophy:

    final = stable strategic baseline  +  small, confidence-scaled overlay

* The **baseline** is the philosophy-driven static weight vector (config
  ``factors.default_weights``). It always dominates, so the strategy keeps a
  coherent identity even when the data is noisy.
* The **overlay** is a *small* adaptive tilt toward factors that have recently
  shown more predictive power — measured by IC, hit rate and factor spread over
  multiple recency-weighted windows (see :mod:`factors.factor_effectiveness`),
  and learned *separately by market regime* (see :mod:`factors.market_regime`).
  Its size is bounded by ``overlay_strength`` and further shrunk by a
  **confidence** score, so when the evidence is thin the engine simply stays at
  the baseline.

Everything is point-in-time: a weight vector for date *d* is built only from
periods that completed on/before *d*. Final touches — min/max weight bounds and
EWMA smoothing across rebalances — keep the vector diversified and prevent the
abrupt month-to-month swings that would otherwise churn the book.

The engine is stateful, mirroring :class:`backtesting.weighting.ICWeighter`:
``record(date, parent_scores, regime)`` after each rebalance, ``weights_for(
date, regime)`` before scoring the next one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .factor_effectiveness import (
    DEFAULT_WINDOW_WEIGHTS,
    WINDOWS,
    FactorMetrics,
    PeriodObs,
    compute_effectiveness,
    quintile_spread,
    spearman_ic,
    summarize_factor,
)
from .market_regime import RegimeState

# IC magnitude that we treat as "a genuinely strong signal" when scoring
# confidence. ~0.05 trailing IC is already respectable for a monthly equity
# factor, so that is the point where the strength term saturates.
_STRONG_IC = 0.05


@dataclass
class WeightDecision:
    """Everything the engine decided for one date — weights plus the why."""

    date: str
    weights: dict[str, float]            # final: smoothed, bounded, sums to 1
    target: dict[str, float]             # pre-smoothing target (bounded)
    baseline: dict[str, float]
    overlay: dict[str, float]            # additive tilt actually applied (sums ~0)
    tilt: dict[str, float]               # unit-L1 relative effectiveness tilt
    regime: str
    regime_state: RegimeState
    confidence: float
    overlay_strength_effective: float    # overlay_strength * confidence
    n_periods: int
    n_regime_periods: int
    metrics: dict[str, FactorMetrics]    # global (all-regime) windowed metrics
    effectiveness: dict[str, float]      # regime-blended effectiveness per factor
    applied: str                         # baseline:warmup | overlay | overlay:flat
    confidence_terms: dict[str, float] = field(default_factory=dict)


def project_to_simplex_box(
    weights: dict[str, float], lo: float, hi: float, *, max_iter: int = 100
) -> dict[str, float]:
    """Project a weight dict onto {sum=1, lo<=w<=hi} by water-filling.

    Clip to the box, then push the leftover mass (``1 - sum``) onto the factors
    that still have room in the needed direction, proportional to that room.
    Repeats until the residual vanishes. Feasibility (``lo*n <= 1 <= hi*n``) is
    the caller's responsibility — checked once in the engine constructor.
    """
    keys = list(weights)
    x = {k: float(weights[k]) for k in keys}
    for _ in range(max_iter):
        for k in keys:
            x[k] = min(max(x[k], lo), hi)
        resid = 1.0 - sum(x.values())
        if abs(resid) < 1e-12:
            break
        room = ({k: hi - x[k] for k in keys} if resid > 0
                else {k: x[k] - lo for k in keys})
        base = sum(room.values())
        if base <= 1e-15:
            break
        for k in keys:
            x[k] += resid * room[k] / base
    total = sum(x.values())
    if total > 0:
        x = {k: v / total for k, v in x.items()}
    return x


class FactorWeightEngine:
    def __init__(
        self,
        factor_keys,
        baseline_weights: dict[str, float],
        matrix: pd.DataFrame,
        *,
        overlay_strength: float = 0.30,
        min_weight: float = 0.02,
        max_weight: float = 0.40,
        smoothing: float = 0.5,
        min_periods: int = 4,
        min_names: int = 20,
        regime_blend: float = 0.5,
        window_weights: dict[int, float] | None = None,
    ) -> None:
        self.keys = list(factor_keys)
        n = len(self.keys)
        if n == 0:
            raise ValueError("factor_keys must be non-empty.")
        if not (0.0 <= overlay_strength <= 1.0):
            raise ValueError(f"overlay_strength must be in [0,1], got {overlay_strength}")
        if not (0.0 <= smoothing < 1.0):
            raise ValueError(f"smoothing must be in [0,1), got {smoothing}")
        if not (0.0 <= regime_blend <= 1.0):
            raise ValueError(f"regime_blend must be in [0,1], got {regime_blend}")
        if min_weight < 0.0 or max_weight <= 0.0:
            raise ValueError("min_weight>=0 and max_weight>0 required.")
        if min_weight * n > 1.0 + 1e-9 or max_weight * n < 1.0 - 1e-9:
            raise ValueError(
                f"weight bounds infeasible for {n} factors: need "
                f"min_weight<={1.0/n:.3f} and max_weight>={1.0/n:.3f}.")
        # Baseline restricted to the active keys and renormalized to sum 1.
        base = {k: max(float(baseline_weights.get(k, 0.0)), 0.0) for k in self.keys}
        s = sum(base.values())
        if s <= 0:
            base = {k: 1.0 / n for k in self.keys}
        else:
            base = {k: v / s for k, v in base.items()}
        self.baseline = project_to_simplex_box(base, min_weight, max_weight)
        self.matrix = matrix
        self.overlay_strength = float(overlay_strength)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.smoothing = float(smoothing)
        self.min_periods = int(min_periods)
        self.min_names = int(min_names)
        self.regime_blend = float(regime_blend)
        self.window_weights = window_weights or DEFAULT_WINDOW_WEIGHTS
        self._history: list[tuple[str, pd.DataFrame, str]] = []
        self._prev_applied: dict[str, float] | None = None

    # -- public API ---------------------------------------------------------
    def record(self, date: str, parent_scores: pd.DataFrame, regime: RegimeState) -> None:
        """Store the parent ``*_score`` frame + regime label observed at ``date``."""
        missing = [k for k in self.keys if f"{k}_score" not in parent_scores.columns]
        if missing:
            raise ValueError(f"parent_scores missing columns for: {missing}")
        cols = [f"{k}_score" for k in self.keys]
        self._history.append((date, parent_scores[cols].copy(), regime.label))

    def weights_for(self, current_date: str, regime: RegimeState) -> WeightDecision:
        """Weight vector to use when scoring ``current_date`` (history < date only)."""
        obs_all, obs_reg = self._build_observations(current_date, regime.label)
        n_periods = max((len(v) for v in obs_all.values()), default=0)
        n_regime = max((len(v) for v in obs_reg.values()), default=0)

        if n_periods < self.min_periods:
            return self._baseline_decision(
                current_date, regime, n_periods, n_regime, "baseline:warmup")

        metrics = {k: summarize_factor(k, obs_all[k], windows=WINDOWS,
                                       window_weights=self.window_weights)
                   for k in self.keys}
        metrics_reg = {k: summarize_factor(k, obs_reg[k], windows=WINDOWS,
                                           window_weights=self.window_weights)
                       for k in self.keys}
        eff_all = compute_effectiveness(metrics)
        eff_reg = compute_effectiveness(metrics_reg)

        # Blend the all-regime view with the current-regime view, weighting the
        # regime view by how much same-regime history we actually have.
        rb = self.regime_blend * min(1.0, n_regime / max(self.min_periods, 1))
        effectiveness = {k: (1.0 - rb) * eff_all[k] + rb * eff_reg[k] for k in self.keys}

        tilt = self._unit_tilt(effectiveness)
        confidence, terms = self._confidence(metrics, n_periods, n_regime)
        eff_strength = self.overlay_strength * confidence

        overlay = {k: eff_strength * tilt[k] for k in self.keys}
        target = project_to_simplex_box(
            {k: self.baseline[k] + overlay[k] for k in self.keys},
            self.min_weight, self.max_weight)

        prev = self._prev_applied or self.baseline
        blended = {k: self.smoothing * prev[k] + (1.0 - self.smoothing) * target[k]
                   for k in self.keys}
        applied_w = project_to_simplex_box(blended, self.min_weight, self.max_weight)
        self._prev_applied = applied_w

        applied = "overlay" if any(abs(t) > 1e-9 for t in tilt.values()) else "overlay:flat"
        return WeightDecision(
            date=current_date, weights=applied_w, target=target,
            baseline=dict(self.baseline), overlay=overlay, tilt=tilt,
            regime=regime.label, regime_state=regime, confidence=confidence,
            overlay_strength_effective=eff_strength, n_periods=n_periods,
            n_regime_periods=n_regime, metrics=metrics, effectiveness=effectiveness,
            applied=applied, confidence_terms=terms)

    # -- internals ----------------------------------------------------------
    def _build_observations(
        self, current_date: str, regime_label: str
    ) -> tuple[dict[str, list[PeriodObs]], dict[str, list[PeriodObs]]]:
        """Per-factor period IC/spread series, globally and for ``regime_label``.

        Pairs each recorded date with the next endpoint (up to ``current_date``)
        and tags the period with the regime that was in force at its *start* —
        the information an allocator would actually have had.
        """
        obs_all = {k: [] for k in self.keys}
        obs_reg = {k: [] for k in self.keys}
        endpoints = [d for d, _, _ in self._history] + [current_date]
        for j, (d0, scores0, reg0) in enumerate(self._history):
            fwd = self._forward_return(d0, endpoints[j + 1])
            if fwd is None:
                continue
            for k in self.keys:
                s = scores0.get(f"{k}_score")
                ic = spearman_ic(s, fwd, self.min_names)
                if ic is None:
                    continue
                spread = quintile_spread(s, fwd, min_names=self.min_names)
                ob = PeriodObs(endpoints[j + 1], ic,
                               float("nan") if spread is None else spread)
                obs_all[k].append(ob)
                if reg0 == regime_label:
                    obs_reg[k].append(ob)
        return obs_all, obs_reg

    def _forward_return(self, d0: str, d1: str) -> pd.Series | None:
        m = self.matrix
        if m is None or d0 not in m.index or d1 not in m.index or d1 <= d0:
            return None
        return (m.loc[d1] / m.loc[d0]) - 1.0

    def _unit_tilt(self, effectiveness: dict[str, float]) -> dict[str, float]:
        """Demean effectiveness, then L1-normalize so sum|tilt| == 1.

        Demeaning makes the overlay sum to zero (it reallocates, never changes
        the gross). L1-normalizing gives ``overlay_strength`` a crisp meaning:
        the fraction of total weight the overlay shifts from the weakest factors
        to the strongest. Returns all-zero when no factor stands out.
        """
        vals = np.array([effectiveness[k] for k in self.keys], dtype=float)
        vals = np.nan_to_num(vals, nan=0.0)
        centered = vals - vals.mean()
        l1 = np.abs(centered).sum()
        if l1 <= 1e-12:
            return {k: 0.0 for k in self.keys}
        # Scale so the positive side sums to +0.5 and negative to -0.5 (sum|.|==1).
        scaled = centered / l1
        return {k: float(scaled[i]) for i, k in enumerate(self.keys)}

    def _confidence(
        self, metrics: dict[str, FactorMetrics], n_periods: int, n_regime: int
    ) -> tuple[float, dict[str, float]]:
        """Blend four explainable terms into a [0,1] confidence on the overlay."""
        depth = _clip01(n_periods / (2.0 * max(self.min_periods, 1)))
        regime_support = _clip01(n_regime / max(self.min_periods, 1))
        consistency_vals = [m.ic_consistency for m in metrics.values()
                            if not np.isnan(m.ic_consistency)]
        consistency = float(np.mean(consistency_vals)) if consistency_vals else 0.0
        ic_vals = [abs(m.ic_recency) for m in metrics.values()
                   if not np.isnan(m.ic_recency)]
        strength = _clip01((float(np.mean(ic_vals)) if ic_vals else 0.0) / _STRONG_IC)
        confidence = _clip01(
            0.40 * depth + 0.20 * regime_support + 0.20 * consistency + 0.20 * strength)
        terms = {"depth": depth, "regime_support": regime_support,
                 "consistency": consistency, "strength": strength}
        return confidence, terms

    def _baseline_decision(
        self, date: str, regime: RegimeState, n_periods: int, n_regime: int, applied: str
    ) -> WeightDecision:
        prev = self._prev_applied or self.baseline
        # Even in warm-up we smooth toward the baseline so the very first live
        # vector does not jump if a prior applied vector exists.
        blended = {k: self.smoothing * prev[k] + (1.0 - self.smoothing) * self.baseline[k]
                   for k in self.keys}
        applied_w = project_to_simplex_box(blended, self.min_weight, self.max_weight)
        self._prev_applied = applied_w
        return WeightDecision(
            date=date, weights=applied_w, target=dict(self.baseline),
            baseline=dict(self.baseline), overlay={k: 0.0 for k in self.keys},
            tilt={k: 0.0 for k in self.keys}, regime=regime.label, regime_state=regime,
            confidence=0.0, overlay_strength_effective=0.0, n_periods=n_periods,
            n_regime_periods=n_regime, metrics={}, effectiveness={k: 0.0 for k in self.keys},
            applied=applied, confidence_terms={})


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))
