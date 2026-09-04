"""Blend a frozen configuration into a composite score, exactly as production does.

The three pure helpers ``_composite_row`` / ``_build_parent_panel`` / ``_ic_ir_weights``
are lifted verbatim (behaviour-preserving) from ``scripts/parent_eval.py`` — that module
is a runnable script with side effects (``main``), not a clean import, so the reusable
core is copied here with attribution rather than imported.

:func:`frozen_composite` then mirrors :func:`factors.composite.build_composite` — the
*live* production composite construction — so the score this validation evaluates is the
same one the fund actually ranks on: parent scores are dispersion-equalised (``zscore``),
missing parents held at the neutral 50, blended by the frozen parent weights, and
re-ranked within GICS sector to a 0–100 composite. It reuses
:func:`factors.composite._normalize_parents` and :func:`factors.utils.sector_percentile`
so it cannot silently drift from production.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from factors.composite import NEUTRAL, _normalize_parents
from factors.utils import sector_percentile
from research.panel import ScorePanel


@dataclass(frozen=True)
class FrozenConfig:
    """A fully-frozen composite specification derived from one training window.

    ``sub_weights`` maps parent → {selected sub-factor: intra-parent weight}; each
    inner dict sums to 1. ``parent_weights`` maps parent → composite weight (the
    IC/IR-blend capped vector); it sums to 1 across the parents that earned weight.
    ``meta`` carries provenance for the report (per-parent selection formulae, flags,
    the parent scorecard used to derive the weights).
    """

    sub_weights: dict[str, dict[str, float]]
    parent_weights: dict[str, float]
    meta: dict


# --------------------------------------------------------------------------- #
# Lifted from scripts/parent_eval.py (pure functions; behaviour preserved).   #
# --------------------------------------------------------------------------- #
def _composite_row(mat: np.ndarray, w: np.ndarray,
                   coverage_aware: bool = True) -> np.ndarray:
    """Per-row weighted mean of ``mat`` with weights ``w``.

    ``coverage_aware=True`` skips NaN cells and renormalises across the *available*
    columns per row; ``coverage_aware=False`` requires every column present (any NaN →
    NaN result). Lifted from scripts/parent_eval.py:_composite_row.
    """
    if not coverage_aware:
        any_missing = np.isnan(mat).any(axis=1)
        with np.errstate(invalid="ignore"):
            wsum = w.sum()
            comp = np.where(wsum > 0, np.nansum(mat * w[None, :], axis=1) / wsum, np.nan)
        return np.where(any_missing, np.nan, comp)
    mask = ~np.isnan(mat)
    wrow = mask * w[None, :]
    wsum = wrow.sum(axis=1)
    arr0 = np.where(mask, mat, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        comp = (arr0 * wrow).sum(axis=1) / wsum
    return np.where(wsum > 0, comp, np.nan)


def build_parent_panel(sub_panel: ScorePanel,
                       weights_by_parent: dict[str, dict[str, float]]) -> ScorePanel:
    """One 'sub' per parent = coverage-aware weighted composite of its selected subs.

    Lifted from scripts/parent_eval.py:_build_parent_panel, generalised to take the
    parent order from ``weights_by_parent`` rather than a module constant.
    """
    parents = [p for p in weights_by_parent if weights_by_parent[p]]
    scores: dict[str, pd.DataFrame] = {}
    for d in sub_panel.rebal_dates:
        frame = sub_panel.scores.get(d)
        if frame is None:
            continue
        cols: dict[str, pd.Series] = {}
        for parent in parents:
            wmap = weights_by_parent[parent]
            present = [s for s in wmap if s in frame.columns]
            if not present:
                cols[parent] = pd.Series(np.nan, index=frame.index)
                continue
            sub_mat = frame[present].to_numpy(dtype=float)
            w = np.array([wmap[s] for s in present], dtype=float)
            cols[parent] = pd.Series(_composite_row(sub_mat, w), index=frame.index)
        # index stays the date's PIT universe (frame.index); reindexing to the
        # all-time union creates ghost rows for non-members, which downstream
        # NEUTRAL-fill turns into holdable names (SBNY 2024 bug)
        scores[d] = pd.DataFrame(cols)
    return ScorePanel(
        rebal_dates=list(scores.keys()), scores=scores,
        parent_keys=parents,
        sub_by_parent={p: [p] for p in parents},
        universe=list(sub_panel.universe),
    )


def ic_ir_weights(parent_score: pd.DataFrame, cap: float = 0.25) -> dict[str, float]:
    """IC+IR-weighted, capped, renormalised parent-weight vector (V4 construction).

    Combined = 0.5·(mean_ic/max_ic⁺) + 0.5·(IR/max_ir⁺), each floored at 0; weights ∝
    combined, water-filled to ``cap``, renormalised. Any parent with combined ≤ 0 gets
    0 weight. Lifted from scripts/parent_eval.py:_ic_ir_weights. ``parent_score`` needs
    columns ``parent``, ``mean_ic_3m6m``, ``information_ratio``.
    """
    # NaN-robust: a parent whose IC/IR is undefined on a short/early training window
    # (e.g. 2015-16, before revisions/short-interest exist) contributes 0, not NaN — else
    # the whole weight vector degenerates to NaN and the composite goes flat. If *every*
    # parent is 0/NaN the ``combined.sum() <= 0`` branch below falls back to equal weight.
    ic = parent_score.set_index("parent")["mean_ic_3m6m"].clip(lower=0.0).fillna(0.0)
    ir = parent_score.set_index("parent")["information_ratio"].clip(lower=0.0).fillna(0.0)
    ic_norm = ic / ic.max() if ic.max() > 0 else ic * 0.0
    ir_norm = ir / ir.max() if ir.max() > 0 else ir * 0.0
    combined = 0.5 * ic_norm + 0.5 * ir_norm
    if combined.sum() <= 0:
        # Degenerate window (no parent has positive IC or IR) → equal weight, flagged
        # by the caller via meta. Avoids a divide-by-zero producing all-NaN weights.
        n = len(combined)
        return {p: 1.0 / n for p in combined.index}
    raw = combined / combined.sum()
    w = raw.to_dict()
    for _ in range(20):
        over = [p for p, x in w.items() if x > cap + 1e-12]
        if not over:
            break
        excess = sum(w[p] - cap for p in over)
        for p in over:
            w[p] = cap
        under = [p for p, x in w.items() if x < cap - 1e-12 and x > 0]
        pool = sum(w[p] for p in under)
        if not under or pool <= 1e-12:
            break
        for p in under:
            w[p] += excess * w[p] / pool
    tot = sum(w.values())
    return {p: w[p] / tot for p in w if w[p] > 0}


# --------------------------------------------------------------------------- #
# Production-faithful composite blend.                                         #
# --------------------------------------------------------------------------- #
def frozen_composite(
    sub_panel: ScorePanel,
    rebals: list[str],
    config: FrozenConfig,
    sectors: pd.Series,
    *,
    norm_mode: str = "zscore",
    target_std: float = 20.0,
    min_obs: int = 5,
) -> dict[str, pd.Series]:
    """Composite score (0–100 sector-relative) per rebalance date for ``config``.

    Faithful to :func:`factors.composite.build_composite`: build each parent from its
    selected subs (coverage-aware weighted mean), dispersion-equalise the parents
    (``zscore``), fill missing parents at the neutral 50, blend by ``parent_weights``,
    renormalise by the weight actually used, then re-rank within sector. Returns
    ``{date: Series[ticker → composite_score]}`` for the requested ``rebals`` only.
    """
    parent_panel = build_parent_panel(sub_panel, config.sub_weights)
    pweights = config.parent_weights
    out: dict[str, pd.Series] = {}
    for d in rebals:
        frame = parent_panel.scores.get(d)
        if frame is None:
            continue
        cols = [p for p in pweights if p in frame.columns]
        parents = frame[cols]
        blend = _normalize_parents(parents, norm_mode, target_std)
        comp_raw = pd.Series(0.0, index=frame.index)
        used = 0.0
        for key in cols:
            comp_raw += pweights[key] * blend[key].fillna(NEUTRAL)
            used += pweights[key]
        if used > 0:
            comp_raw /= used
        secs = sectors.reindex(frame.index).fillna("Unknown")
        out[d] = sector_percentile(comp_raw, secs, higher_is_better=True, min_obs=min_obs)
    return out
