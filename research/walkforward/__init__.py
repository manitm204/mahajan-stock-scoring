"""Walk-forward out-of-sample validation of the composite factor model.

A strict, no-look-ahead validation harness. At each expanding train/test split it
re-runs the *entire* live Parent-Selection-V4 construction (subfactor selection,
intra-parent weights, parent composition, IC/IR-capped parent/composite weights) on
**train-only** data — with forward-return windows capped at the test boundary so no
future information can enter the selection — then freezes that configuration and scores
a completely unseen test period. Every downstream read (composite IC, quintile spreads,
portfolio construction, holding-period sweep, long-short) is therefore genuinely
out-of-sample.

Read-only: nothing here mutates production scoring, ``config.yaml``, or the live DB. The
composite is reproduced through the research expansion library
(:mod:`research.subfactor_expansion`), whose insider window is point-in-time correct, so
the ``factors/utils.py`` look-ahead bug that contaminates the production ``factors/`` path
never touches this study.
"""
from __future__ import annotations

from .splits import WalkForwardSplit, default_splits
from .compose import FrozenConfig, frozen_composite
from .selection import select_config

__all__ = [
    "WalkForwardSplit", "default_splits",
    "FrozenConfig", "frozen_composite",
    "select_config",
]
