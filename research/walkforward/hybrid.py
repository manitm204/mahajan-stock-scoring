"""Hybrid blends of two frozen configurations (5-year + 2-year training windows).

Two blend recipes serve the regime study:

* :func:`blend_parent_weights` — build ONE :class:`FrozenConfig` from two: parents present
  in either input contribute the average of their weights (missing = 0), sub-weights are
  averaged across the union of selected subs per parent (missing = 0), and both maps are
  renormalised to sum to 1. The result plugs straight into :func:`frozen_composite`; the
  study evaluates this single averaged construction on the test window.
* :func:`average_composites` — evaluate each of the two frozen configs on the test window
  independently, then blend the resulting per-date composite Series 50/50 (mean of the two
  0-100 sector-relative scores per name per date). Only names present in both are kept per
  date; the result is a composite score dict identical in shape to what a single config
  would produce, so all downstream analysis (IC/quantiles/portfolio) works unchanged.

Both blends are 50/50 by default but accept an arbitrary ``w`` ∈ (0, 1) — the weight on
the first argument (the 5-year config / composite by convention in the study).
"""
from __future__ import annotations

import pandas as pd

from .compose import FrozenConfig


def _blend_dict(a: dict[str, float], b: dict[str, float], w: float) -> dict[str, float]:
    keys = set(a) | set(b)
    blend = {k: w * float(a.get(k, 0.0)) + (1.0 - w) * float(b.get(k, 0.0)) for k in keys}
    total = sum(blend.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in blend.items() if v > 0}


def blend_parent_weights(cfg_a: FrozenConfig, cfg_b: FrozenConfig,
                         w: float = 0.5) -> FrozenConfig:
    """Return a single :class:`FrozenConfig` = ``w·cfg_a + (1-w)·cfg_b`` averaged.

    Parent-weight vector is the (weighted) mean across the union of parents, renormalised.
    Sub-weights per parent are the (weighted) mean across the union of that parent's
    selected subs in either config, renormalised. Parents present in only one config keep
    their surviving subs proportionally reduced through the parent-weight average.
    """
    parents = set(cfg_a.parent_weights) | set(cfg_b.parent_weights)
    sub_weights: dict[str, dict[str, float]] = {}
    for p in parents:
        sw = _blend_dict(cfg_a.sub_weights.get(p, {}), cfg_b.sub_weights.get(p, {}), w)
        if sw:
            sub_weights[p] = sw
    parent_weights = _blend_dict(cfg_a.parent_weights, cfg_b.parent_weights, w)
    meta = {
        "blend": {"w_a": w, "w_b": 1.0 - w,
                  "sources": [cfg_a.meta.get("boundary"), cfg_b.meta.get("boundary")]},
        "n_train_rebalances_a": cfg_a.meta.get("n_train_rebalances"),
        "n_train_rebalances_b": cfg_b.meta.get("n_train_rebalances"),
    }
    return FrozenConfig(sub_weights=sub_weights, parent_weights=parent_weights, meta=meta)


def average_composites(scores_a: dict[str, pd.Series],
                       scores_b: dict[str, pd.Series],
                       w: float = 0.5) -> dict[str, pd.Series]:
    """Blend two composite score dicts per rebalance date: ``w·A + (1-w)·B``.

    Both inputs are ``{date: Series[ticker → 0-100 sector-relative score]}``. For each
    shared date the two series are aligned on the union of tickers; a name scored by only
    one side keeps that side's score (implicitly promoting single-signal coverage rather
    than dropping partial names). Dates present in only one dict are passed through as-is.
    """
    dates = set(scores_a) | set(scores_b)
    out: dict[str, pd.Series] = {}
    for d in sorted(dates):
        sa, sb = scores_a.get(d), scores_b.get(d)
        if sa is None:
            out[d] = sb.copy()
            continue
        if sb is None:
            out[d] = sa.copy()
            continue
        idx = sa.index.union(sb.index)
        a = sa.reindex(idx)
        b = sb.reindex(idx)
        # per-cell blend that falls back to whichever side is populated
        blended = w * a.fillna(b) + (1.0 - w) * b.fillna(a)
        out[d] = blended.dropna()
    return out
