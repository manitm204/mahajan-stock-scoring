"""VIX-relative parent-weight overlay study — clean-room research backtest.

Answers one narrow question: does tilting the frozen rolling-5Y parent weights
toward regime-conditional (VIX-comparable) parent performance improve the
existing model out-of-sample?

Design:
* Semiannual (H1/H2) out-of-sample test windows 2017 → present; for each, the
  prior rolling 5 years build the normal baseline model (sub-factor selection,
  intra-parent weights, baseline parent weights), which is then frozen.
* Overlay variants adjust only the parent weights at each monthly test
  rebalance, based on where the rebalance-date VIX sits — never the sub-factor
  selection. Composites are recomputed and holdings re-ranked.
* Everything an overlay consumes (VIX percentiles, regime IC/IR/Q5-Q1 stats,
  regime weights) is computed from training data available strictly before the
  test window; hard assertions enforce this.

This package is research-only: it reads the cached PIT candidate panel, prices,
sectors and VIX; it writes only to ``cache/vix_relative_overlay/`` and
``output/vix_relative_overlay/``. The single deliberate reuse of existing
research code is the per-window baseline construction (see
``baseline_cache.py``) — the object under test. All measurement machinery here
is implemented independently of ``research/``.
"""
