"""Reproduce the live V4 selection on one training window — with no look-ahead.

:func:`select_config` runs the exact Parent-Selection-V4 chain, but only on the training
rebalances of a split and only on forward returns that complete *before the test-start
boundary* (the price matrix is truncated there, so any window reaching into the test
period is dropped by :func:`research.forward_returns.compute_forward_returns`). The chain:

1. slice the candidate score panel to the training rebalances;
2. :func:`research.parent_selection.run_selection` → per parent, the greedily-diversified
   sub-factors and their intra-parent weights (∝ 3M/6M mean IC, capped 50 %);
3. composite each parent from its subs and score the 8 parents (3M/6M IC + IR) with
   :func:`research.subset_selection.subfactor_performance`;
4. :func:`research.walkforward.compose.ic_ir_weights` (cap 25 %) → the parent/composite
   weight vector.

The result is a :class:`~research.walkforward.compose.FrozenConfig` that
:func:`~research.walkforward.compose.frozen_composite` can apply to the unseen test period.
Run on the *full* sample (no boundary) this reproduces the live
``factors.parent_selection_v4.SELECTED_SUBS`` — the property the test-suite asserts.
"""
from __future__ import annotations

import pandas as pd

from research import HORIZON_MONTHS, compute_forward_returns
from research import parent_selection as ps
from research.panel import ScorePanel
from research.subset_selection import inventory, subfactor_performance

from .compose import FrozenConfig, build_parent_panel, ic_ir_weights

STAT_HORIZONS = ("3M", "6M")
PARENT_CAP = 0.25


def slice_panel(panel: ScorePanel, dates: list[str]) -> ScorePanel:
    """A view of ``panel`` restricted to ``dates`` (taxonomy/universe unchanged)."""
    keep = [d for d in dates if d in panel.scores]
    return ScorePanel(
        rebal_dates=keep,
        scores={d: panel.scores[d] for d in keep},
        parent_keys=list(panel.parent_keys),
        sub_by_parent={p: list(s) for p, s in panel.sub_by_parent.items()},
        universe=list(panel.universe),
    )


def _parent_scorecard(sub_panel: ScorePanel,
                      sub_weights: dict[str, dict[str, float]],
                      fwd_by_h: dict[str, dict[str, pd.Series]]) -> pd.DataFrame:
    """Score each selected parent (composite of its subs) on 3M/6M IC + IR."""
    parent_panel = build_parent_panel(sub_panel, sub_weights)
    perf = subfactor_performance(parent_panel, fwd_by_h, stat_horizons=STAT_HORIZONS)
    inv = inventory(parent_panel)
    perf["mean_ic_3m6m"] = perf[["ic_3M", "ic_6M"]].mean(axis=1)
    scored = (perf.drop(columns=["parent"], errors="ignore")
                  .rename(columns={"sub_factor": "parent"})
                  .merge(inv[["sub_factor", "coverage"]].rename(
                      columns={"sub_factor": "parent"}), on="parent", how="left"))
    return scored


def select_config(
    panel: ScorePanel,
    train_rebals: list[str],
    price_matrix: pd.DataFrame,
    boundary: str | None,
) -> FrozenConfig:
    """Derive a :class:`FrozenConfig` from ``train_rebals`` with no look-ahead.

    ``boundary`` (the test-start date) truncates the price matrix so selection forward
    returns never reach into the test period; pass ``None`` for a full-sample run (used
    by the reproduction test). ``price_matrix`` is the adjusted-close matrix.
    """
    px = price_matrix.loc[price_matrix.index <= boundary] if boundary else price_matrix
    train_panel = slice_panel(panel, train_rebals)
    fwd = compute_forward_returns(px, train_panel.rebal_dates, HORIZON_MONTHS)

    parents = [p for p in train_panel.parent_keys if train_panel.sub_by_parent.get(p)]
    results = ps.run_selection(train_panel, fwd, parents)

    sub_weights: dict[str, dict[str, float]] = {}
    selection_meta: dict[str, dict] = {}
    for res in results:
        sub_weights[res.parent] = dict(res.weights)
        selection_meta[res.parent] = {
            "selected": list(res.selected),
            "formula": res.formula,
            "signal_flag": res.signal_flag,
            "stop_reason": res.stop_reason,
        }

    parent_score = _parent_scorecard(train_panel, sub_weights, fwd)
    parent_weights = ic_ir_weights(parent_score, cap=PARENT_CAP)

    meta = {
        "n_train_rebalances": len(train_panel.rebal_dates),
        "train_rebalances": list(train_panel.rebal_dates),
        "boundary": boundary,
        "max_fwd_end": _max_fwd_end(fwd),
        "selection": selection_meta,
        "parent_scorecard": parent_score[
            [c for c in ["parent", "mean_ic_3m6m", "information_ratio",
                         "spread_q5_q1", "hit_rate", "coverage"] if c in parent_score]
        ].to_dict("records"),
    }
    return FrozenConfig(sub_weights=sub_weights, parent_weights=parent_weights, meta=meta)


def _max_fwd_end(fwd_by_h: dict[str, dict[str, pd.Series]]) -> str | None:
    """Latest forward-return *start* date actually used across all horizons — a proxy the
    orchestrator logs to prove no selection window crossed the boundary (the price cap
    guarantees the window *end* ≤ boundary; this reports the last start observed)."""
    starts = [d for h in fwd_by_h.values() for d in h]
    return max(starts) if starts else None
