"""Factor Research & Validation Framework (Layer 2 diagnostics).

A read-only research layer that evaluates the predictive power of every parent
factor *and* sub-factor over the same point-in-time monthly rebalance grid the
backtester uses, then judges which sub-factors earn their place in the model.

The framework never alters scoring or trading. It only measures:

* **quintiles** — Q1-Q5 forward-return profiles across 1/3/6/12-month horizons,
  the Q5-Q1 spread, and whether returns rise monotonically with the score;
* **IC** — cross-sectional Spearman rank correlation of score vs forward return,
  with information ratio, hit rate, and stability over time;
* **redundancy** — cross-sectional score correlation among the sub-factors of a
  parent (are two subs measuring the same characteristic?);
* **incremental contribution** — does a sub-factor add unique predictive value
  beyond its siblings, and does including it improve the parent factor's IC?

and finally classifies each sub-factor **Keep / Merge / Remove / Insufficient
Data**. The goal is a smaller, cleaner, more explainable model where every
sub-factor justifies its inclusion — not the largest possible factor count.

Every measurement is strictly point-in-time: a score observed at date ``d_k`` is
only ever paired with returns realized *after* ``d_k``. Sub-factors whose Layer 1
history is a single recent snapshot (revisions / short interest / insider / most
13F fields) cannot be judged and are reported as *Insufficient Data* rather than
mistaken for *Remove*.
"""
from __future__ import annotations

from .panel import ScorePanel, build_score_panel
from .forward_returns import HORIZON_MONTHS, compute_forward_returns, realize_delistings
from .quintiles import aggregate_quintiles, quintile_profile
from .ic import ic_timeseries, summarize_ic
from .redundancy import max_sibling_corr, parent_score_correlation
from .incremental import incremental_metrics
from .classify import Thresholds, classify_subfactors

__all__ = [
    "ScorePanel", "build_score_panel",
    "HORIZON_MONTHS", "compute_forward_returns", "realize_delistings",
    "aggregate_quintiles", "quintile_profile",
    "ic_timeseries", "summarize_ic",
    "max_sibling_corr", "parent_score_correlation",
    "incremental_metrics",
    "Thresholds", "classify_subfactors",
]
