"""Alpha-adjusted block bootstrap + perturbation test on two strategies
(user request 2026-09-13): the champion `insider_revisions_min_top20` and
the near-miss `insider_revisions_min_new_entrant_top20`, both against
`harness.random_top20_selector` -- the same baseline used for the
champion's original significance check (champion.md: p=0.015 at L=4), so
results here are directly comparable to that historical number.

Usage: python -m research.loop_research.bootstrap_two
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from research.loop_research import harness as H                      # noqa: E402
from research.loop_research.exploration_batch_2026_09_11 import (    # noqa: E402
    make_selector, make_perturbed_selector,
)
from research.loop_research.exploration_batch_100 import (           # noqa: E402
    metric_insider_revisions_min,
)
from research.loop_research.exploration_batch_2026_09_12 import (    # noqa: E402
    metric_insider_revisions_min_new_entrant,
)
from research.loop_research.promotion import (                       # noqa: E402
    evaluate_promotion, compute_baseline_paths, OUT_DIR,
)
from research.loop_research.block_bootstrap import bootstrap_table    # noqa: E402

N_SIMS = 500

STRATEGIES = {
    "insider_revisions_min_top20": (metric_insider_revisions_min, False),
    "insider_revisions_min_new_entrant_top20": (metric_insider_revisions_min_new_entrant, False),
}


def main():
    bundle = H.get_data()
    random_baseline = H.random_top20_selector

    print("computing random-baseline paths ONCE (shared by both strategies) ...")
    base_raw, base_metrics = compute_baseline_paths(random_baseline, bundle=bundle, n_sims=N_SIMS)

    for name, (metric_fn, ascending) in STRATEGIES.items():
        print(f"\n{'='*70}\n{name}  vs  random_top20_selector\n{'='*70}")
        candidate_selector = make_selector(name, metric_fn, ascending)

        def make_pert(noise_mult, _mf=metric_fn, _asc=ascending):
            return make_perturbed_selector(_mf, _asc, noise_mult)

        report = evaluate_promotion(
            candidate_selector, baseline_selector=random_baseline,
            make_perturbed_selector=make_pert, n_sims=N_SIMS,
            deterministic_candidate=True, bundle=bundle,
            baseline_precomputed=(base_raw, base_metrics),
        )

        cm, bm = report["candidate_metrics"], report["baseline_metrics"]
        print(f"median sharpe: candidate={cm['median_sharpe']:.4f}  random={bm['median_sharpe']:.4f}  "
             f"(delta {report['sharpe_improve']:+.4f}, win_rate {report['win_rate']:.1%})")
        print(f"beta (monthly): candidate={report['beta_candidate_monthly']:.3f}  "
             f"random={report['beta_baseline_monthly']:.3f}")
        print(f"perturbation win rate (noise_mult=0.5): "
             f"{report['perturbation_win_rate']:.1%}"
             if report["perturbation_win_rate"] is not None else "perturbation: n/a")

        table, _, _ = bootstrap_table(report["diff_by_date"])
        pd.set_option("display.float_format", lambda x: f"{x:.5f}")
        print("\nalpha-adjusted block bootstrap (candidate - random, beta-neutralized):")
        print(table[["block_len_months", "observed_mean_monthly_diff", "ci_lo_95",
                    "ci_hi_95", "excludes_zero", "p_value_two_sided"]].to_string(index=False))


if __name__ == "__main__":
    main()
