"""Layer 3 — point-in-time backtesting for the Mahajan factor composite.

Evaluates whether the composite's top-ranked longs outperform and bottom-ranked
shorts underperform, with strict no-look-ahead discipline: every rebalance scores
the universe using only data filed on/before that date (via
:func:`factors.pipeline.score_universe` with ``reporting_lag=True``).

Public surface:

* :class:`BacktestConfig` / :func:`run_backtest` — configure and run.
* :data:`STRATEGIES` — the four strategy variants (A–D).
"""
from __future__ import annotations

from .engine import BacktestConfig, BacktestResult, run_backtest
from .portfolio_rules import STRATEGIES

__all__ = ["BacktestConfig", "BacktestResult", "run_backtest", "STRATEGIES"]
