"""Per-parent subfactor selection: reuses parent_selection.py's R²-diversification
loop (fed Production Score instead of the old blended Sub-factor Score, gated on
Expected IC via an eligibility pre-filter), adds a "materially lowers the parent"
post-hoc degradation check against a real recomputed composite, and reweights with
a 10% minimum-weight floor. See
docs/superpowers/specs/2026-07-11-hierarchical-regime-aware-factor-model-design.md
Section 3.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from research.ic import period_ic
from research.panel import ScorePanel
from research.parent_selection import MAX_SUBS, R2_MAX, SINGLE_CAP, select_subfactors
from research.quintiles import quintile_profile
from research.walkforward.compose import build_parent_panel

DEGRADATION_TOL = 0.10   # reject a candidate if it drops any parent metric >10% relative
MIN_WEIGHT = 0.10        # floor for any selected sub's final weight
MIN_NAMES = 20


def parent_subfactor_weights(
    selected: list[str], production_score: pd.Series, *,
    cap: float = SINGLE_CAP, min_weight: float = MIN_WEIGHT,
) -> dict[str, float]:
    """Weight ∝ positive Production Score, water-filled to ``cap``, then any share
    below ``min_weight`` is floored and the set renormalises. Flooring can only
    ever affect a strict subset (never all) when max_subs * min_weight <= 1 (true
    for the defaults 3 * 0.10 = 0.30) since weights summing to 1 can't all sit
    below 1/max_subs.
    """
    if not selected:
        return {}
    raw = {s: max(float(production_score.get(s, 0.0)), 0.0) for s in selected}
    total = sum(raw.values())
    w = ({s: 1.0 / len(selected) for s in selected} if total <= 1e-12
        else {s: raw[s] / total for s in selected})
    for _ in range(20):
        over = [s for s in w if w[s] > cap + 1e-12]
        if not over:
            break
        excess = sum(w[s] - cap for s in over)
        for s in over:
            w[s] = cap
        under = [s for s in w if w[s] < cap - 1e-12]
        pool = sum(w[s] for s in under)
        if not under or pool <= 1e-12:
            break
        for s in under:
            w[s] += excess * w[s] / pool
    tot = sum(w.values())
    w = {s: v / tot for s, v in w.items()}
    for _ in range(20):
        below = [s for s in w if w[s] < min_weight - 1e-12]
        if not below:
            break
        deficit = sum(min_weight - w[s] for s in below)
        for s in below:
            w[s] = min_weight
        above = [s for s in w if s not in below]
        pool = sum(w[s] for s in above)
        if not above or pool <= 1e-12:
            break
        for s in above:
            w[s] -= deficit * w[s] / pool
    return w


def _apply_degradation_check(
    selected: list[str], metric_map: pd.DataFrame,
    stats_fn: Callable[[dict[str, float]], dict[str, float]], *,
    cap: float, min_weight: float, tol: float,
) -> tuple[list[str], str | None]:
    """Walks ``selected`` in order, truncating at the first candidate whose addition
    drops any of (mean_ic, ic_ir, spread) -- as read from ``stats_fn(weights)`` --
    by more than ``tol`` relative to the metrics without it. ``stats_fn`` maps a
    {sub: weight} dict to {"mean_ic", "ic_ir", "spread"}; injected so this stays
    testable without a real ScorePanel (see ``_composite_ic_stats`` for the real one).
    """
    kept: list[str] = []
    for sub in selected:
        trial = kept + [sub]
        trial_w = parent_subfactor_weights(trial, metric_map["production_score"],
                                           cap=cap, min_weight=min_weight)
        if kept:
            base_w = parent_subfactor_weights(kept, metric_map["production_score"],
                                              cap=cap, min_weight=min_weight)
            base_stats, trial_stats = stats_fn(base_w), stats_fn(trial_w)
            degraded = any(
                pd.notna(base_stats[m]) and pd.notna(trial_stats[m]) and base_stats[m] != 0
                and (trial_stats[m] - base_stats[m]) / abs(base_stats[m]) < -tol
                for m in ("mean_ic", "ic_ir", "spread"))
            if degraded:
                return kept, f"adding `{sub}` degraded a parent metric > {tol:.0%}"
        kept.append(sub)
    return kept, None


def _composite_ic_stats(
    sub_panel: ScorePanel, weights: dict[str, float],
    fwd_by_h: dict[str, dict[str, pd.Series]], *, min_names: int = MIN_NAMES,
) -> dict[str, float]:
    """Real long-run mean IC / IC-IR / Q5-Q1 spread of the composite built from
    ``weights`` over every date in ``sub_panel`` -- used only for the
    materially-lowers-the-parent degradation check, so it stays a plain long-run
    read (no recent/regime blend needed for this local comparison).
    """
    if not weights:
        return {"mean_ic": float("nan"), "ic_ir": float("nan"), "spread": float("nan")}
    trial_panel = build_parent_panel(sub_panel, {"__trial__": weights})
    ics: list[float] = []
    spreads: list[float] = []
    for d in trial_panel.rebal_dates:
        frame = trial_panel.scores.get(d)
        if frame is None or "__trial__" not in frame.columns:
            continue
        score = frame["__trial__"]
        for h in ("3M", "6M"):
            fwd = fwd_by_h.get(h, {}).get(d)
            if fwd is None:
                continue
            ic = period_ic(score, fwd, min_names=min_names)
            if ic is not None:
                ics.append(ic)
        fwd6 = fwd_by_h.get("6M", {}).get(d)
        if fwd6 is not None:
            prof = quintile_profile(score, fwd6, min_names=min_names)
            if prof is not None:
                spreads.append(float(prof[-1] - prof[0]))
    ic_s = pd.Series(ics, dtype=float)
    mean_ic = float(ic_s.mean()) if not ic_s.empty else float("nan")
    std_ic = float(ic_s.std(ddof=1)) if len(ic_s) > 1 else float("nan")
    ic_ir = mean_ic / std_ic if std_ic and std_ic > 1e-9 else float("nan")
    spread = float(pd.Series(spreads).mean() * 2.0) if spreads else float("nan")
    return {"mean_ic": mean_ic, "ic_ir": ic_ir, "spread": spread}


def select_parent_subfactors(
    scored: pd.DataFrame, corr: pd.DataFrame,
    sub_panel: ScorePanel, fwd_by_h: dict[str, dict[str, pd.Series]], *,
    r2_max: float = R2_MAX, max_subs: int = MAX_SUBS,
    degradation_tol: float = DEGRADATION_TOL, min_weight: float = MIN_WEIGHT,
    single_cap: float = SINGLE_CAP,
) -> dict:
    """Select + weight one parent's subfactors from its ``scored`` evidence rows
    (regime_aware_scoring.score_table output, already filtered to one parent).

    Returns {"selected": [...], "weights": {sub: weight}, "stop_reason": str,
    "decisions": list[dict]}.
    """
    eligible = scored[scored["eligible"]].copy()
    if eligible.empty:
        return {"selected": [], "weights": {}, "stop_reason": "no eligible candidate",
               "decisions": []}
    ranked = (eligible.sort_values("production_score", ascending=False)
             .rename(columns={"expected_ic": "mean_ic"})   # reuse select_subfactors' IC gate
             .reset_index(drop=True))
    candidates, decisions, stop_reason = select_subfactors(
        ranked, corr, r2_max=r2_max, max_subs=max_subs)

    metric_map = eligible.set_index("sub_factor")
    kept, degrade_stop = _apply_degradation_check(
        candidates, metric_map,
        lambda w: _composite_ic_stats(sub_panel, w, fwd_by_h),
        cap=single_cap, min_weight=min_weight, tol=degradation_tol)
    if degrade_stop is not None:
        stop_reason = degrade_stop

    weights = parent_subfactor_weights(kept, metric_map["production_score"],
                                       cap=single_cap, min_weight=min_weight)
    return {"selected": kept, "weights": weights, "stop_reason": stop_reason,
           "decisions": decisions}
