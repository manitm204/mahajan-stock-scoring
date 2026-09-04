"""Follow-up round on the top-10 strategy sweep (user request 2026-09-02):

1. Trailing-stop threshold sensitivity (1%/5%/10%/20%) + stop-loss / score-
   crash combos layered on #13 -- research/strategies/rules.py FOLLOWUP_STRATS.
2. Book-size sweep {1,3,5,10,15,20} for the two best sleeve strategies
   (#05 3M-monthly-sleeve, #06 6M-monthly-sleeve).

Same data, same cost model, same window as scripts/run_strategy_sweep.py.

Outputs to output/strategy_sweep/:
  followup_summary.csv  -- the 6 new managed-book variants
  ksweep_summary.csv    -- 3M/6M sleeve x book size {1,3,5,10,15,20}

Usage: python scripts/run_strategy_followup.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backtesting.data_loader import SPY, QQQ
from research.strategies.engine import load_data, simulate_calendar_sleeves, simulate_managed_book
from research.strategies.rules import FOLLOWUP_STRATS
from scripts.run_strategy_sweep import summarize, bench_returns

OUT = REPO / "output" / "strategy_sweep"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    print("loading composite scores + price matrix ...", flush=True)
    data = load_data()
    rebal = data.rebal_dates
    spy = bench_returns(data.matrix, SPY, rebal)
    qqq = bench_returns(data.matrix, QQQ, rebal)

    rows = []
    for name, rule in FOLLOWUP_STRATS.items():
        print(f"  {name} ...", flush=True)
        nav = simulate_managed_book(data, rule)
        rows.append(summarize(name, nav, rebal, spy, qqq))
    followup = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    followup.to_csv(OUT / "followup_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", lambda v: f"{v:.3f}"):
        print("\n=== trailing-stop sensitivity + combos ===")
        print(followup.to_string(index=False))

    ksweep_rows = []
    for sleeve_name, hold_months, sleeve_count in [
        ("3M_monthly_sleeve", 3, 3), ("6M_monthly_sleeve", 6, 6),
    ]:
        for k in (1, 3, 5, 10, 15, 20):
            name = f"{sleeve_name}_k{k}"
            print(f"  {name} ...", flush=True)
            nav = simulate_calendar_sleeves(data, k=k, hold_months=hold_months, sleeve_count=sleeve_count)
            row = summarize(name, nav, rebal, spy, qqq)
            row["family"] = sleeve_name
            row["k"] = k
            ksweep_rows.append(row)
    ksweep = pd.DataFrame(ksweep_rows)
    ksweep.to_csv(OUT / "ksweep_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", lambda v: f"{v:.3f}"):
        print("\n=== book-size sweep ===")
        print(ksweep.to_string(index=False))


if __name__ == "__main__":
    main()
