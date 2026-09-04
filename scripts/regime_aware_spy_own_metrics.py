"""SPY's OWN Sharpe/vol/CAGR/max-drawdown, computed with the exact same
performance_metrics() methodology and rebalance grid used for the strategy
variants -- not just excess-return-vs-SPY, but SPY's absolute risk profile, so
"our Sharpe is lower" can be judged against "SPY's Sharpe was also X" rather
than assumed.

Writes output/regime_aware/ablation/spy_own_metrics.csv (per window) and
prints the full-period summary.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtesting import data_loader as dl
from backtesting.data_loader import SPY
from data.db import get_db
from research.walkforward.portfolio import performance_metrics
from research.walkforward.splits import semiannual_policy_splits
from run_walkforward import PRICE_END, _load_panel

OUT_DIR = Path("output/regime_aware/ablation")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    panel = _load_panel(rebuild=False)
    with get_db() as db:
        matrix = dl.load_price_matrix(db, panel.universe, "2015-06-30", PRICE_END)

    spy_px = matrix[SPY]
    splits = semiannual_policy_splits("rolling5y", first_test_year=2017)

    rows = []
    for sp in splits:
        test_rebals = sp.test_rebalances(panel.rebal_dates)
        dates_in_matrix = [d for d in test_rebals if d in spy_px.index]
        if len(dates_in_matrix) < 2:
            continue
        px = spy_px.loc[dates_in_matrix]
        returns = px.pct_change().dropna()
        m = performance_metrics(returns, hold_months=1)
        rows.append({"window": sp.label.split(":", 1)[-1], "cagr": m["cagr"],
                     "sharpe": m["sharpe"], "sortino": m["sortino"],
                     "ann_vol": m["ann_vol"], "max_drawdown": m["max_drawdown"]})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "spy_own_metrics.csv", index=False)
    print(df.to_string(index=False))
    print("\nFull-period means:")
    print(df[["cagr", "sharpe", "sortino", "ann_vol", "max_drawdown"]].mean().round(4))


if __name__ == "__main__":
    main()
