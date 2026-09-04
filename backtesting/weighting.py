"""Walk-forward Information Coefficient (IC) factor weighting.

At each rebalance the composite can weight factors by how well each one's score
has predicted forward returns *in the past*. The trailing predictive signal is
the cross-sectional Spearman rank correlation between the parent score at the
start of a period and the realized forward return over that period (the
Information Coefficient). Only periods that completed on/before the current
rebalance feed the estimate, so the weights are point-in-time by construction
(no look-ahead, like the rest of the backtest).

Two scoring modes:
    * ``mean``: rank by trailing mean IC (the original behaviour).
    * ``ir``:   rank by IC information ratio (mean / std of the IC time series).
      This penalizes factors whose IC is high on average but unstable, which is
      what causes a single noisy factor to dominate the composite for a few
      months and then collapse.

A ``max_weight`` cap (e.g. 0.4) prevents any one factor from running away with
the composite. Excess weight above the cap is redistributed proportionally to
the remaining uncapped factors; the loop repeats until the vector is feasible
or every factor is at the cap. With the default ``max_weight=1.0`` the cap is a
no-op so the original behaviour is preserved.

Negative scores are clipped at zero (a factor that has not predicted is dropped
rather than bet against) and the result is normalized to sum to 1. Until
``min_periods`` of history accrue — or if no factor has a positive score — the
weighter falls back to the configured static weights so the strategy is always
well-defined.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

_MEAN = "mean"
_IR = "ir"


class ICWeighter:
    def __init__(
        self,
        factor_keys: Iterable[str],
        default_weights: dict[str, float],
        matrix: pd.DataFrame,
        *,
        min_periods: int = 6,
        min_names: int = 20,
        metric: str = _MEAN,
        max_weight: float = 1.0,
        min_weight: float = 0.0,
        min_std_floor: float = 1e-4,
    ) -> None:
        if metric not in (_MEAN, _IR):
            raise ValueError(f"metric must be 'mean' or 'ir', got {metric!r}")
        if not (0.0 < max_weight <= 1.0):
            raise ValueError(f"max_weight must be in (0, 1], got {max_weight}")
        if min_weight < 0.0:
            raise ValueError(f"min_weight must be >= 0, got {min_weight}")
        self.keys = list(factor_keys)
        # A cap below 1/n_factors is infeasible (the vector cannot sum to 1).
        if max_weight * len(self.keys) < 1.0 - 1e-9:
            raise ValueError(
                f"max_weight={max_weight} is infeasible for {len(self.keys)} "
                f"factors (needs >= {1.0/len(self.keys):.3f}).")
        # A floor above 1/n_factors is infeasible (the floors alone overshoot 1).
        if min_weight * len(self.keys) > 1.0 + 1e-9:
            raise ValueError(
                f"min_weight={min_weight} is infeasible for {len(self.keys)} "
                f"factors (needs <= {1.0/len(self.keys):.3f}).")
        if min_weight > max_weight + 1e-9:
            raise ValueError(
                f"min_weight={min_weight} cannot exceed max_weight={max_weight}.")
        self.default = {k: float(default_weights.get(k, 0.0)) for k in self.keys}
        self.matrix = matrix
        self.min_periods = int(min_periods)
        self.min_names = int(min_names)
        self.metric = metric
        self.max_weight = float(max_weight)
        self.min_weight = float(min_weight)
        self.min_std_floor = float(min_std_floor)
        self._history: list[tuple[str, pd.DataFrame]] = []  # (date, parent *_score frame)

    def record(self, date: str, parent_scores: pd.DataFrame) -> None:
        """Store the parent factor scores observed at ``date`` for future ICs."""
        self._history.append((date, parent_scores))

    def weights_for(self, current_date: str) -> dict:
        """Learned weights to use when scoring ``current_date``.

        Consumes only history recorded before this call (dates < current_date),
        pairing each stored date with the next endpoint up to and including
        ``current_date``. Returns the weight vector plus diagnostics for logging.
        """
        ic_samples: dict[str, list[float]] = {k: [] for k in self.keys}
        endpoints = [d for d, _ in self._history] + [current_date]
        for j, (d0, scores0) in enumerate(self._history):
            fwd = self._forward_return(d0, endpoints[j + 1])
            if fwd is None:
                continue
            for k in self.keys:
                ic = self._spearman(scores0.get(f"{k}_score"), fwd)
                if ic is not None:
                    ic_samples[k].append(ic)

        mean_ic = {k: (float(np.mean(v)) if v else np.nan) for k, v in ic_samples.items()}
        std_ic = {k: (float(np.std(v, ddof=1)) if len(v) >= 2 else np.nan)
                  for k, v in ic_samples.items()}
        score = self._score(mean_ic, std_ic)
        n_periods = max((len(v) for v in ic_samples.values()), default=0)
        weights, applied = self._to_weights(score, n_periods)
        return {"weights": weights, "mean_ic": mean_ic, "std_ic": std_ic,
                "score": score, "n_ic_periods": n_periods, "applied": applied}

    # -- internals ----------------------------------------------------------
    def _forward_return(self, d0: str, d1: str) -> pd.Series | None:
        m = self.matrix
        if d0 not in m.index or d1 not in m.index or d1 <= d0:
            return None
        return (m.loc[d1] / m.loc[d0]) - 1.0

    def _spearman(self, scores: pd.Series | None, fwd: pd.Series) -> float | None:
        if scores is None:
            return None
        df = pd.DataFrame({"s": scores, "f": fwd}).dropna()
        if len(df) < self.min_names or df["s"].nunique() < 2:
            return None
        ic = df["s"].corr(df["f"], method="spearman")
        return None if pd.isna(ic) else float(ic)

    def _score(self, mean_ic: dict[str, float],
               std_ic: dict[str, float]) -> dict[str, float]:
        """Per-factor ranking score; ``mean`` is the IC mean, ``ir`` is IC/std.

        In IR mode a factor with fewer than two IC samples has no defined std
        and is treated as missing (NaN). It will be skipped by the warm-up /
        flat-fallback path until enough history accrues.
        """
        out: dict[str, float] = {}
        for k in self.keys:
            m, s = mean_ic.get(k, np.nan), std_ic.get(k, np.nan)
            if self.metric == _MEAN:
                out[k] = m
                continue
            if np.isnan(m) or np.isnan(s):
                out[k] = np.nan
                continue
            denom = max(s, self.min_std_floor)
            out[k] = m / denom
        return out

    def _to_weights(self, score: dict[str, float],
                    n_periods: int) -> tuple[dict, str]:
        if n_periods < self.min_periods:
            return self._apply_floor(dict(self.default)), "warmup:config"
        pos = {k: max(score[k], 0.0) for k in self.keys if not np.isnan(score[k])}
        total = sum(pos.values())
        if total <= 0:
            return self._apply_floor(dict(self.default)), "ic:flat->config"
        raw = {k: pos.get(k, 0.0) / total for k in self.keys}
        capped, was_capped = self._apply_cap(raw)
        floored, was_floored = self._apply_floor(capped, return_changed=True)
        if was_floored:
            applied = "ic:capped+floored" if was_capped else "ic:floored"
        else:
            applied = "ic:capped" if was_capped else "ic"
        return floored, applied

    def _apply_floor(self, weights: dict[str, float],
                     return_changed: bool = False):
        """Lift any weight below ``min_weight`` to the floor and rebalance.

        The deficit is taken from factors that are above the floor in
        proportion to their excess over the floor, so a strongly-positive
        factor still dominates — it just can't push the others to zero.
        Iterates because applying the floor can push another factor over the
        cap; the loop re-applies the cap each pass until both bounds hold or
        every weight has converged to the floor.
        """
        floor = self.min_weight
        if floor <= 0.0:
            return (weights, False) if return_changed else weights
        w = {k: float(weights.get(k, 0.0)) for k in self.keys}
        changed = False
        for _ in range(2 * len(self.keys) + 1):
            below = {k: v for k, v in w.items() if v < floor - 1e-12}
            if not below:
                break
            changed = True
            deficit = sum(floor - v for v in below.values())
            for k in below:
                w[k] = floor
            # Donors: weights strictly above the floor. Allocate proportionally
            # to the excess over the floor so the largest weights donate most.
            donors = {k: w[k] - floor for k in self.keys
                      if k not in below and w[k] > floor + 1e-12}
            base = sum(donors.values())
            if base <= 0:
                # Every weight is already at the floor; nothing more to take.
                break
            for k, excess in donors.items():
                w[k] -= deficit * (excess / base)
            # Re-cap so a donor that absorbed too little can't violate the cap
            # boundary on the next pass; usually a no-op since we only reduced.
            w, _ = self._apply_cap(w)
        # Final normalize against floating-point drift, then re-floor once if
        # the renormalize pushed a borderline weight back below the floor.
        total = sum(w.values())
        if total > 0:
            w = {k: v / total for k, v in w.items()}
        if any(v < floor - 1e-9 for v in w.values()):
            below = {k: v for k, v in w.items() if v < floor - 1e-9}
            deficit = sum(floor - v for v in below.values())
            for k in below:
                w[k] = floor
            donors = {k: w[k] - floor for k in self.keys
                      if k not in below and w[k] > floor + 1e-12}
            base = sum(donors.values())
            if base > 0:
                for k, excess in donors.items():
                    w[k] -= deficit * (excess / base)
        return (w, changed) if return_changed else w

    def _apply_cap(self, weights: dict[str, float]) -> tuple[dict[str, float], bool]:
        """Cap every weight at ``self.max_weight`` and redistribute the excess.

        Iterates: any factor whose weight currently exceeds the cap is fixed at
        the cap; the remaining factors absorb the excess. The redistribution
        basis is the other positive (uncapped) factors when there are any;
        otherwise — typically when the IC selected only a single factor in an
        early walk-forward period — the excess is allocated to the *default*
        weight vector restricted to non-capped factors. The latter is the
        critical guard against single-factor flight: without a fallback basis,
        the final sum-normalization would just push the lone factor straight
        back through the cap to 100%.

        Terminates when nothing is over the cap, or when every factor is at it.
        """
        w = dict(weights)
        cap = self.max_weight
        if cap >= 1.0:
            return w, False
        was_capped = False
        for _ in range(len(self.keys) + 1):
            over = {k: v for k, v in w.items() if v > cap + 1e-12}
            if not over:
                break
            was_capped = True
            excess = sum(v - cap for v in over.values())
            for k in over:
                w[k] = cap

            non_over = [k for k in self.keys if k not in over]
            free = {k: w[k] for k in non_over
                    if w[k] > 0.0 and w[k] < cap - 1e-12}
            base = sum(free.values())
            if base > 0:
                for k, v in free.items():
                    w[k] = v + excess * (v / base)
                continue

            # No other positive factor to absorb: fall back to the default
            # weight vector restricted to factors not at the cap. This keeps
            # the sum at 1.0 *and* respects the cap.
            default_pool = {k: self.default.get(k, 0.0) for k in non_over
                            if w[k] < cap - 1e-12}
            default_sum = sum(default_pool.values())
            if default_sum > 0:
                for k, v in default_pool.items():
                    w[k] = w.get(k, 0.0) + excess * (v / default_sum)
                continue

            # Last resort: even split across all non-capped factors. Only
            # reachable in degenerate configs (every default weight is 0 on
            # every non-capped factor).
            spread = excess / max(len(non_over), 1) if non_over else 0.0
            for k in non_over:
                w[k] = w.get(k, 0.0) + spread
            break

        total = sum(w.values())
        if total > 0:
            w = {k: v / total for k, v in w.items()}
        # A final renormalization can lift a capped factor above the cap if
        # the redistribution rounded short. One more pass tames that.
        if cap < 1.0 and any(v > cap + 1e-9 for v in w.values()):
            w, _ = self._apply_cap_once(w, cap)
        return w, was_capped

    @staticmethod
    def _apply_cap_once(w: dict[str, float], cap: float) -> tuple[dict[str, float], bool]:
        """One-shot cap + proportional redistribute among any positive others."""
        over = {k: v for k, v in w.items() if v > cap + 1e-12}
        if not over:
            return w, False
        excess = sum(v - cap for v in over.values())
        for k in over:
            w[k] = cap
        free = {k: v for k, v in w.items()
                if v > 0.0 and v < cap - 1e-12 and k not in over}
        base = sum(free.values())
        if base > 0:
            for k, v in free.items():
                w[k] = v + excess * (v / base)
        return w, True
