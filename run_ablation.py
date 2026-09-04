"""Portfolio-construction ablation runner (see research/ablation/__init__.py).

Usage:
    python run_ablation.py --stage a          # OFAT around the reference
    python run_ablation.py --stage b          # cross Stage-A winners (needs stageA.csv)
    python run_ablation.py --stage c          # confirm Stage-B winner (needs stageB.csv)
    python run_ablation.py --stage all

The walk-forward scoring runs once at startup (~5-10 min); each config afterwards is
seconds. Progress is printed per config and each stage's CSV is written incrementally,
so `tail -f` on the log (or the CSV) shows live progress.
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from run_walkforward import PANEL_START, PRICE_END, _load_panel, get_db
from research.ablation import load_ablation_data
from research.ablation.stages import (full_grid_configs, pick_winner,
                                      report_full_grid, run_stage, stage_a_configs,
                                      stage_b_configs, stage_c_configs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="a",
                    choices=["a", "b", "c", "all", "full"])
    ap.add_argument("--splits", default="rolling5y")
    ap.add_argument("--out", default="output/ablation")
    ap.add_argument("--rebuild-panel", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print("[setup] loading panel + building ablation data bundle ...", flush=True)
    panel = _load_panel(args.rebuild_panel)
    with get_db() as db:
        data = load_ablation_data(panel, db, PANEL_START, PRICE_END,
                                  splits=args.splits)
    print(f"[setup] ready in {(time.time()-t0)/60:.1f} min\n", flush=True)

    if args.stage == "full":
        df = run_stage(data, full_grid_configs(), f"{args.out}/full_grid.csv",
                       "FULL", progress_every=50)
        fin = report_full_grid(df)
        if fin is not None:
            print(f"\n[confirm] running Stage C sensitivities on finalist "
                  f"{fin['config']} ...", flush=True)
            run_stage(data, stage_c_configs(fin), f"{args.out}/full_confirm.csv",
                      "C(full)")
        print("\nPre-registered rule: promote only if search alpha>0 AND holdout "
              "alpha>0 AND both eras' excess vs SPY >=0.", flush=True)
        return

    stages = ["a", "b", "c"] if args.stage == "all" else [args.stage]
    if "a" in stages:
        run_stage(data, stage_a_configs(), f"{args.out}/stageA.csv", "A")
    if "b" in stages:
        stage_a = pd.read_csv(f"{args.out}/stageA.csv")
        run_stage(data, stage_b_configs(stage_a), f"{args.out}/stageB.csv", "B")
    if "c" in stages:
        stage_b = pd.read_csv(f"{args.out}/stageB.csv")
        winner = pick_winner(stage_b)
        print(f"\n[stage C] finalist from Stage B: {winner['config']}", flush=True)
        run_stage(data, stage_c_configs(winner), f"{args.out}/stageC.csv", "C")

    print("\nPre-registered rule: a winner needs positive net alpha vs SPY "
          "full-period AND non-negative excess in BOTH eras (split 2021-12-31).",
          flush=True)


if __name__ == "__main__":
    main()
