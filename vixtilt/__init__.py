"""VIX-relative parent-weight overlay study — standalone research module.

Answers one narrow question: does a VIX-relative parent-weight overlay improve the
existing rolling 5-year factor model?

This package is deliberately isolated from ``research/``. It imports from the existing
pipeline ONLY the pieces that *define* the baseline model and its data:

* ``research.walkforward.selection.select_config`` — the exact rolling-5y V4 baseline
  (re-implementing it would test a different model than the one in production);
* ``research.walkforward.compose.build_parent_panel`` — the parent construction the
  frozen configs imply;
* the cached PIT candidate panel + ``backtesting.data_loader`` prices/sectors.

Everything study-specific — semiannual windows, VIX percentile machinery, regime
utilities, weight blending, portfolio simulation with costs, IC/quantile stats,
holdings overlap, reporting — is implemented fresh here.

Layout: windows.py (splits) · baseline.py (stage-A per-window frozen baseline cache) ·
overlay.py (VIX overlay logic) · backtest.py (composites, portfolios, stats) ·
report.py (tables, dashboard, markdown). Runner: ``run_vixtilt_study.py``.
Caches: ``cache/vixtilt/`` — outputs: ``output/vixtilt/``.
"""
