"""Monte Carlo curves for the top-20-pool champion `insider_revisions_min_top20`
(user request 2026-09-14): dashboard wants a second Monte Carlo graph showing
this selector against SPY/QQQ, with 10th-90th percentile bounds built from
0.5x-noise perturbation sims (same noise mechanism as perturbation_sweep.py)
rather than a single deterministic line.

Writes output/loop_research/insider_revisions_min_perturbation.json.
Usage: python -m research.loop_research.monte_carlo_insider_revisions_min
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backtesting.data_loader import SPY, QQQ                          # noqa: E402
from research.autoresearch.evaluate import bench_returns              # noqa: E402
from research.loop_research import harness as H                       # noqa: E402
from research.loop_research.exploration_batch_2026_09_11 import (     # noqa: E402
    make_perturbed_selector,
)
from research.loop_research.exploration_batch_100 import (            # noqa: E402
    metric_insider_revisions_min,
)
from research.walkforward.portfolio import performance_metrics        # noqa: E402

OUT_DIR = REPO / "output" / "loop_research"
N_SIMS = 500
NOISE_MULT = 0.5


def _stats(pr, spy, qqq):
    m = performance_metrics(pr, hold_months=1, benchmarks={"SPY": spy, "QQQ": qqq})
    return {"cagr": m["cagr"], "sharpe": m["sharpe"], "sortino": m["sortino"],
            "max_dd": m["max_drawdown"], "beta": m.get("spy_beta"),
            "alpha": m.get("spy_alpha")}


def main():
    bundle = H.get_data()
    data = bundle["data"]
    rebal = list(data.rebal_dates)
    all_dates = rebal[:-1]

    spy = bench_returns(data.matrix, SPY, rebal)
    qqq = bench_returns(data.matrix, QQQ, rebal)

    selector = make_perturbed_selector(metric_insider_revisions_min, False, NOISE_MULT)

    sim_curves = np.zeros((N_SIMS, len(all_dates)))
    sim_stats = {"cagr": [], "sharpe": [], "sortino": [], "max_dd": [], "beta": [], "alpha": []}
    for s in range(N_SIMS):
        print(f"sim {s + 1}/{N_SIMS}", end="\r", flush=True)
        pr, turnover, targets = H.run_one(bundle, selector, seed=s)
        st = _stats(pr, spy, qqq)
        for kk, v in st.items():
            sim_stats[kk].append(v)
        pr_full = pr.reindex(all_dates).fillna(0.0)
        sim_curves[s] = (1.0 + pr_full).cumprod().values
    print()

    median_curve = np.median(sim_curves, axis=0)
    p10 = np.percentile(sim_curves, 10, axis=0)
    p90 = np.percentile(sim_curves, 90, axis=0)
    median_stats = {kk: float(np.median(v)) for kk, v in sim_stats.items()}

    spy_r = spy.reindex(all_dates).fillna(0.0)
    qqq_r = qqq.reindex(all_dates).fillna(0.0)
    spy_curve = (1.0 + spy_r).cumprod().values
    qqq_curve = (1.0 + qqq_r).cumprod().values

    win_rate_vs_spy = float((sim_curves[:, -1] > spy_curve[-1]).mean())
    win_rate_vs_qqq = float((sim_curves[:, -1] > qqq_curve[-1]).mean())

    result = {
        "selector": "insider_revisions_min_top20",
        "noise_mult": NOISE_MULT,
        "n_sims": N_SIMS,
        "dates": [str(d) for d in all_dates],
        "median_curve": median_curve.tolist(),
        "p10_curve": p10.tolist(),
        "p90_curve": p90.tolist(),
        "spy_curve": spy_curve.tolist(),
        "qqq_curve": qqq_curve.tolist(),
        "stats": {
            "portfolio": median_stats,
            "spy": _stats(spy_r, spy_r, qqq_r),
            "qqq": _stats(qqq_r, spy_r, qqq_r),
        },
        "summary": {
            "n_sims": N_SIMS,
            "median_final_return": float(median_curve[-1]),
            "pct_sims_beating_spy": win_rate_vs_spy,
            "pct_sims_beating_qqq": win_rate_vs_qqq,
        },
        "params": {"k": H.N, "hold_months": H.HOLD_MONTHS, "sleeves": H.SLEEVE_COUNT,
                   "refresh_n": H.REFRESH_N, "pool": "top20", "noise_mult": NOISE_MULT},
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "insider_revisions_min_perturbation.json"
    with out_path.open("w") as fh:
        json.dump(result, fh, indent=2)
    print(f"wrote {out_path}")
    print(f"median sharpe={median_stats['sharpe']:.3f}  "
         f"win vs SPY={win_rate_vs_spy:.1%}  win vs QQQ={win_rate_vs_qqq:.1%}")


if __name__ == "__main__":
    main()
