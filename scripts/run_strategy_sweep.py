"""Run the 20 pre-registered top-10 holding/exit strategies (user request
2026-09-02) on the SAME production EQEFF composite score and report CAGR,
Sharpe, beta (vs SPY), and max drawdown for each -- see
research/strategies/engine.py + rules.py for the two simulators and the 20
configs, and rules.py's module docstring for the #17/#18 value-cheapness
proxy caveat (no PIT-usable analyst price-target history exists).

Universe/window/scores are identical across all 20 runs; only the
holding-period / exit rule differs, so differences below are attributable to
that alone. 2020-01->2026-06, ~78 monthly reviews, 10bps cost per side.

Outputs to output/strategy_sweep/:
  summary.csv   -- one row per strategy: cagr/sharpe/beta/max_dd + context cols
  equity.csv    -- monthly equity curves, all 20 + SPY + QQQ

Usage: python scripts/run_strategy_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backtesting.data_loader import SPY, QQQ
from research.ablation.engine import alpha_tstat
from research.walkforward.portfolio import performance_metrics
from research.strategies.engine import (
    load_data, simulate_calendar_sleeves, simulate_managed_book,
)
from research.strategies.rules import CALENDAR_STRATS, MANAGED_STRATS

OUT = REPO / "output" / "strategy_sweep"
OUT.mkdir(parents=True, exist_ok=True)


def monthly_returns(nav: pd.Series, rebal_dates: list) -> pd.Series:
    m = nav.reindex(rebal_dates).dropna()
    return m.pct_change().dropna()


def bench_returns(matrix: pd.DataFrame, ticker: str, rebal_dates: list) -> pd.Series:
    m = matrix[ticker].reindex(rebal_dates).dropna()
    return m.pct_change().dropna()


def summarize(name: str, nav: pd.Series, rebal_dates: list, spy: pd.Series,
             qqq: pd.Series) -> dict:
    pr = monthly_returns(nav, rebal_dates)
    m = performance_metrics(pr, hold_months=1, benchmarks={SPY: spy, QQQ: qqq})
    dd = m.get("max_drawdown")
    return {
        "strategy": name,
        "cagr": m["cagr"],
        "sharpe": m["sharpe"],
        "beta": m.get("spy_beta"),
        "max_dd": dd,
        "sortino": m["sortino"],
        "calmar": m["cagr"] / abs(dd) if dd and dd == dd and dd != 0 else float("nan"),
        "alpha_vs_spy": m.get("spy_alpha"),
        "alpha_t_vs_spy": alpha_tstat(pr, spy),
        "ex_spy_cagr": m.get("spy_excess_cagr"),
        "ex_qqq_cagr": m.get("qqq_excess_cagr"),
        "n_months": m["n_periods"],
    }


def main():
    print("loading composite scores + price matrix ...", flush=True)
    data = load_data()
    rebal = data.rebal_dates
    spy = bench_returns(data.matrix, SPY, rebal)
    qqq = bench_returns(data.matrix, QQQ, rebal)

    rows = []
    curves = {"SPY": (1.0 + spy).cumprod(), "QQQ": (1.0 + qqq).cumprod()}

    for name, kw in CALENDAR_STRATS.items():
        print(f"  {name} ...", flush=True)
        nav = simulate_calendar_sleeves(data, **kw)
        rows.append(summarize(name, nav, rebal, spy, qqq))
        curves[name] = nav.reindex(rebal).dropna()

    for name, rule in MANAGED_STRATS.items():
        print(f"  {name} ...", flush=True)
        nav = simulate_managed_book(data, rule)
        rows.append(summarize(name, nav, rebal, spy, qqq))
        curves[name] = nav.reindex(rebal).dropna()

    summary = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT / "equity.csv")

    with pd.option_context("display.width", 200, "display.float_format",
                           lambda v: f"{v:.3f}"):
        print("\n" + summary.to_string(index=False))
    print(f"\nwrote {OUT}/summary.csv, {OUT}/equity.csv")


if __name__ == "__main__":
    main()
