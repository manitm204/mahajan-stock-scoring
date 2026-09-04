"""Simple construction ablation, ranked by Sharpe.

Monthly walk-forward (OOS pooled scores, net of 10bps), NO tax / sleeves / 13mo holds.
Sweeps: breadth (% of stocks) x weighting x sector x loser-veto. Uses the locked 3M/6M
composite (production ablation cache). Each config is a cheap re-weight (~1s).
"""
from __future__ import annotations
import pickle
from dataclasses import replace

import numpy as np
import pandas as pd

from hz_experiment import load_env, make_data, PROD_CACHE
from research.ablation.engine import AblationConfig, simulate_config, spy_row
from research.walkforward.portfolio import performance_metrics
from data.db import get_db

TOPS = [0.10, 0.25, 0.50, 0.75, 1.00]
WTS = ["ew", "cap", "cap5", "ewcap", "rank_lin", "inv_vol"]
SECS = ["none", "cap_match"]
EXCL = ["none", "any_p10", "two_p10", "any_p5", "soft_p10"]
OUT = "/home/manit/Desktop/fun_projects/mahajan_hedge_fund/output/ablation/ablation_sharpe_walkforward.csv"


def main():
    panel, matrix_raw, matrix, sectors, vix = load_env()
    run = pickle.load(PROD_CACHE.open("rb"))
    with get_db() as db:
        data = make_data(run, panel, matrix, sectors, vix, db)

    rows = []
    n = 0
    for tp in TOPS:
        for wt in WTS:
            for sec in SECS:
                for ex in EXCL:
                    cfg = AblationConfig(name=f"t{int(tp*100)}_{wt}_{sec}_{ex}",
                                         top_pct=tp, weighting=wt, sector=sec,
                                         exclusion=ex, vix_tilt=False, hold_months=1)
                    rows.append(simulate_config(data, cfg))
                    n += 1
    df = pd.DataFrame(rows)

    # SPY / QQQ reference on the same monthly grid
    form = data.rebal_dates
    spy = spy_row(data)
    qser = (matrix["QQQ"].reindex(form) / matrix["QQQ"].reindex(form).shift(1) - 1.0).dropna()
    qm = performance_metrics(qser, 1)

    keep = ["top_pct", "weighting", "sector", "exclusion", "sharpe", "net_cagr",
            "max_dd", "eff_n", "beta", "alpha_t", "ex_spy", "avg_names", "turnover"]
    df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)
    df[keep].to_csv(OUT, index=False)

    pd.set_option("display.width", 200, "display.max_rows", 60)
    def fmt(d):
        return (f"{d['weighting']:>8} {d['sector']:>9} {d['exclusion']:>8}"
                f"   sh {d['sharpe']:.3f}  cagr {d['net_cagr']*100:5.1f}%  maxDD {d['max_dd']*100:6.1f}%"
                f"  effN {d['eff_n']:5.1f}  beta {d['beta']:.2f}")

    print(f"\n{n} configs. Monthly walk-forward, net 10bps, no tax. Ranked by Sharpe.")
    print(f"SPY  sharpe {spy['sharpe']:.3f}  cagr {spy['net_cagr']*100:.1f}%  beta 1.00")
    print(f"QQQ  sharpe {qm['sharpe']:.3f}  cagr {qm['cagr']*100:.1f}%")

    for tp in [0.10, 0.25, 0.50, 0.75]:
        print(f"\n=== TOP 5 @ top {int(tp*100)}% ===")
        print(f"{'weight':>8} {'sector':>9} {'veto':>8}   sharpe   cagr     maxDD    effN    beta")
        sub = df[df["top_pct"] == tp].head(5)
        for _, d in sub.iterrows():
            print(fmt(d))
    print(f"\nFull table -> {OUT}")


if __name__ == "__main__":
    main()
