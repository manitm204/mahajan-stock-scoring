"""Shared walk-forward engine: run a list of splits, return pooled OOS artefacts.

Both the main annual walk-forward (Sections 1-7) and the training-window study
(Section 8) do the same thing per split — re-select the whole construction on train-only
data, freeze it, score the unseen test period — and differ only in which splits they feed.
This module holds that loop once.

For each split it records the frozen :class:`~research.walkforward.compose.FrozenConfig`,
the split's out-of-sample composite IC, a top-20 % equal-weight portfolio, the frozen
composite scores, and each parent's frozen-sub-weighted score frame (for the parent-level
analysis). Everything is pooled across splits for the aggregate reads.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from research import HORIZON_MONTHS, compute_forward_returns
from research.panel import ScorePanel

from . import analysis
from . import portfolio as pf
from .compose import build_parent_panel, frozen_composite
from .selection import select_config, slice_panel
from .splits import WalkForwardSplit


@dataclass
class RunResult:
    splits_data: list[dict] = field(default_factory=list)
    pooled_scores: dict[str, pd.Series] = field(default_factory=dict)
    per_split_scores: dict[str, dict[str, pd.Series]] = field(default_factory=dict)
    # {parent: {date: Series}} — each parent's frozen score, pooled across split test dates
    parent_scores: dict[str, dict[str, pd.Series]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def run_splits(panel: ScorePanel, splits: list[WalkForwardSplit],
               matrix: pd.DataFrame, sectors: pd.Series,
               *, verbose: bool = True) -> RunResult:
    """Select-freeze-score every split; return pooled out-of-sample artefacts."""
    res = RunResult()
    for sp in splits:
        train_rebals = sp.train_rebalances(panel.rebal_dates)
        test_rebals = sp.test_rebalances(panel.rebal_dates)
        if not train_rebals or not test_rebals:
            if verbose:
                print(f"  skip {sp.label}: train={len(train_rebals)} test={len(test_rebals)}")
            continue
        if verbose:
            print(f"  {sp.label}: select on {len(train_rebals)} "
                  f"({train_rebals[0]}→{train_rebals[-1]}), test {len(test_rebals)} "
                  f"({test_rebals[0]}→{test_rebals[-1]})")
        cfg = select_config(panel, train_rebals, matrix, boundary=sp.test_start)
        res.notes.append(
            f"Split `{sp.label}` [{sp.policy}]: selection used rebalances through "
            f"**{train_rebals[-1]}** (≥6M before the {sp.test_start} boundary); "
            f"max forward-return start = {cfg.meta.get('max_fwd_end')} — no window "
            f"crossed into the test period.")

        scores = frozen_composite(panel, test_rebals, cfg, sectors)
        res.per_split_scores[sp.label] = scores
        res.pooled_scores.update(scores)

        # Each parent's frozen-sub-weighted score (for parent-level OOS IC / correlations).
        parent_panel = build_parent_panel(slice_panel(panel, test_rebals), cfg.sub_weights)
        for parent in parent_panel.parent_keys:
            bucket = res.parent_scores.setdefault(parent, {})
            for d in parent_panel.rebal_dates:
                col = parent_panel.scores[d].get(parent)
                if col is not None:
                    bucket[d] = col

        test_fwd = compute_forward_returns(matrix, list(scores), HORIZON_MONTHS)
        test_ic = analysis.composite_ic(scores, test_fwd)
        top20 = pf.simulate(scores, matrix, sectors, top_pct=0.20, mode="equal",
                            hold_months=1)
        res.splits_data.append({
            "split": sp, "config": cfg, "test_ic": test_ic,
            "port_metrics": top20.metrics, "scores": scores,
            "n_train": len(train_rebals), "n_test": len(test_rebals),
        })
    return res
