"""Parent utility score, 70/30 base/adaptive weight blend, monthly/quarterly
change caps, and the subfactor-membership hysteresis state machine. See
docs/superpowers/specs/2026-07-11-hierarchical-regime-aware-factor-model-design.md
Section 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from research.walkforward.regime_aware_scoring import _add_predictive_reliability_scores
from research.walkforward.vix_overlay import _water_fill

# _water_fill's cap is best-effort: with n eligible parents, cap*n < 1 makes the cap
# mathematically infeasible at sum=1 and it breaches the cap to preserve the sum
# instead (see adaptive_weight/blend_parent_weights callers and their tests). At the
# realistic production parent count (~8), cap=0.25 is always feasible (0.25*8=2.0).
PARENT_CAP = 0.25
MONTHLY_CAP = 0.02
QUARTERLY_CAP = 0.05
ENTRY_MONTHS = 2
EXIT_MONTHS = 3
MAX_SUBS = 3


def parent_utility_table(parent_evidence: pd.DataFrame) -> pd.DataFrame:
    """Same Predictive x Reliability construction as regime_aware_scoring.score_table,
    but ranked across ALL parents at once (no sub-grouping at this level -- every
    parent competes against every other parent). ``parent_evidence`` is the output
    of regime_aware_evidence.as_of() run on a parent-composite panel (one row per
    parent, with "sub_factor" == "parent" == the parent's own name).
    """
    df = parent_evidence.copy()
    df["ic_ir"] = (df["long_run_mean_ic"] / df["long_run_std_ic"]).replace(
        [np.inf, -np.inf], np.nan)
    return _add_predictive_reliability_scores(df, "utility_score")


def adaptive_weight(utility: pd.Series, expected_ic: pd.Series, *,
                    cap: float = PARENT_CAP) -> dict[str, float]:
    """Water-fill by positive Parent Utility, restricted to parents with positive
    expected_ic -- the "30%" adaptive component of the parent-weight blend."""
    eligible = {p: max(float(utility.get(p, 0.0)), 0.0) for p in utility.index
               if expected_ic.get(p, float("nan")) > 0}
    return _water_fill(eligible, cap)


def blend_parent_weights(
    base: dict[str, float], adaptive: dict[str, float], expected_ic: dict[str, float], *,
    base_weight_frac: float = 0.70, cap: float = PARENT_CAP,
) -> dict[str, float]:
    """raw = base_weight_frac*base + (1-base_weight_frac)*adaptive; any parent with
    expected_ic <= 0 is zeroed; water-fill renormalise to ``cap``."""
    all_p = set(base) | set(adaptive)
    raw = {p: base_weight_frac * base.get(p, 0.0)
          + (1 - base_weight_frac) * adaptive.get(p, 0.0) for p in all_p}
    raw = {p: (0.0 if expected_ic.get(p, float("nan")) <= 0 else v) for p, v in raw.items()}
    return _water_fill(raw, cap)


def apply_change_caps(
    target: dict[str, float], prev_month_w: dict[str, float], quarter_start_w: dict[str, float],
    *, monthly_cap: float = MONTHLY_CAP, quarterly_cap: float = QUARTERLY_CAP,
) -> dict[str, float]:
    """Clips ``target`` toward [prev_month_w +/- monthly_cap] intersected with
    [quarter_start_w +/- quarterly_cap], then redistributes any sum-to-1 shortfall/
    excess only among parents still strictly inside their own band -- a single clip-
    then-uniformly-renormalise pass can push a parent back outside its own band
    whenever the clipped values don't already sum to exactly 1 (e.g. 3+ asymmetric
    parents); iterating keeps every parent inside [lo, hi] as the binding constraint.
    The 25% parent cap is already enforced upstream in blend_parent_weights, so it is
    not re-applied here.
    """
    all_p = set(target) | set(prev_month_w) | set(quarter_start_w)
    lo: dict[str, float] = {}
    hi: dict[str, float] = {}
    w: dict[str, float] = {}
    for p in all_p:
        v = target.get(p, 0.0)
        pm = prev_month_w.get(p, 0.0)
        qs = quarter_start_w.get(p, 0.0)
        lo[p] = max(pm - monthly_cap, qs - quarterly_cap, 0.0)
        hi[p] = max(min(pm + monthly_cap, qs + quarterly_cap), lo[p])
        w[p] = float(np.clip(v, lo[p], hi[p]))

    for _ in range(20):
        diff = sum(w.values()) - 1.0
        if abs(diff) < 1e-9:
            break
        if diff > 0:
            movable = [p for p in w if w[p] > lo[p] + 1e-12]
            room = sum(w[p] - lo[p] for p in movable)
            if not movable or room <= 1e-12:
                break
            for p in movable:
                w[p] = max(w[p] - diff * (w[p] - lo[p]) / room, lo[p])
        else:
            deficit = -diff
            movable = [p for p in w if w[p] < hi[p] - 1e-12]
            room = sum(hi[p] - w[p] for p in movable)
            if not movable or room <= 1e-12:
                break
            for p in movable:
                w[p] = min(w[p] + deficit * (hi[p] - w[p]) / room, hi[p])
    return w


@dataclass
class HysteresisState:
    """Per-parent subfactor-membership hysteresis, carried month-over-month.

    ``members`` is the currently active (quarter-frozen) selected-sub set.
    ``above_streak``/``below_streak`` count consecutive months each sub has been
    eligible-and-selectable / not, reset whenever the direction flips.
    """
    members: set[str] = field(default_factory=set)
    above_streak: dict[str, int] = field(default_factory=dict)
    below_streak: dict[str, int] = field(default_factory=dict)


def update_streaks(state: HysteresisState, would_select: list[str], eligible: set[str],
                   immediate_exit: set[str]) -> None:
    """Call every month. ``would_select`` is this month's one-shot greedy selection
    (regime_aware_selection.select_parent_subfactors' ``selected``, priority order);
    ``eligible`` is subs with expected_ic > 0 and not EXCLUDED; ``immediate_exit`` is
    subs with a sign-inversion/coverage-collapse flag -- removed from ``members``
    immediately, bypassing hysteresis entirely (spec Section 3).
    """
    state.members -= immediate_exit
    would_select_set = set(would_select)
    all_seen = (would_select_set | eligible | state.members
               | set(state.above_streak) | set(state.below_streak))
    for sub in all_seen:
        selectable = sub in would_select_set and sub in eligible
        if selectable:
            state.above_streak[sub] = state.above_streak.get(sub, 0) + 1
            state.below_streak[sub] = 0
        else:
            state.below_streak[sub] = state.below_streak.get(sub, 0) + 1
            state.above_streak[sub] = 0


def apply_quarterly_membership(state: HysteresisState, would_select: list[str], *,
                               max_subs: int = MAX_SUBS) -> None:
    """Call only at a calendar-quarter boundary (Jan/Apr/Jul/Oct). Exits are
    evaluated for every current member first (freeing a slot); entries are then
    considered in ``would_select`` priority order, capped at ``max_subs`` members.
    """
    for sub in list(state.members):
        if state.below_streak.get(sub, 0) >= EXIT_MONTHS:
            state.members.discard(sub)
    for sub in would_select:
        if len(state.members) >= max_subs:
            break
        if sub not in state.members and state.above_streak.get(sub, 0) >= ENTRY_MONTHS:
            state.members.add(sub)
