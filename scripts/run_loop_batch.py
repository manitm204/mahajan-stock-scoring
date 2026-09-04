"""Automated batch runner for the loop-engineering search (user request:
"run another 50 rounds", 2026-09-02). Rounds 1-20 were hand-proposed one at a
time; at this scale hand-writing every candidate stops being practical, so
this generates candidates by randomly perturbing the CURRENT CHAMPION's
config each round (occasionally restarting from the original round-0
baseline to escape local optima), while reusing the exact same harness,
promotion rule, and logging as scripts/loop_round.py -- same train/val split,
same output layout, same --rule/--outdir switches (see that script's
docstring for what v1/v2/v3 mean).

This is a local search, not a fresh idea generator -- it will re-discover the
same knife-edge-threshold trap seen under the v1 rule if a mutation happens
to land on one. Rule of thumb applied AFTER the batch (not automated):
whatever wins should get its immediate neighbors re-tested before being
trusted -- see the round-18 and round-61 robustness checks for the pattern.

Usage: python scripts/run_loop_batch.py --rounds 50 --candidates-per-round 5 --seed 7
       python scripts/run_loop_batch.py --rounds 30 --rule v3 --outdir output/loop_engineering_v3 --seed 2
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from research.strategies.generic_rule import StrategyConfig
from scripts.loop_round import RULES, DEFAULT_OUTDIR, _append_log, load_or_seed_champion, paths
from research.loop_engineering.harness import run_candidate

FIELD_BOUNDS = {
    "k": [5, 6, 7, 8, 9, 10, 11, 12, 15, 20],
    "trail_pct": (0.04, 0.25),
    "hard_stop_pct": [None, None, None, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25],
    "take_profit_pct": [None, None, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80],
    "cap_months": (3, 24),
    "score_floor": [None, None, None, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0],
    "rank_floor_pct": [None, None, None, 0.50, 0.60, 0.70, 0.75, 0.80],
    "min_hold_months": [0.0, 0.0, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0],
    "max_per_sector": [None, None, 2, 3, 4, 5, 6],
    "min_value_pct": (0.0, 70.0),
    "entry_rank_k": [None, None, None, 15, 20, 25, 30],
}
MUTABLE_FIELDS = list(FIELD_BOUNDS)


def _mutate_field(field: str, current, rng: random.Random):
    spec = FIELD_BOUNDS[field]
    if field == "trail_pct":
        base = current if current is not None else 0.10
        lo, hi = spec
        return None if rng.random() < 0.06 else round(min(hi, max(lo, base + rng.uniform(-0.03, 0.03))), 3)
    if field == "cap_months":
        lo, hi = spec
        base = current if current is not None else 12
        return int(min(hi, max(lo, base + rng.choice([-3, -2, -1, 1, 2, 3]))))
    if field == "min_value_pct":
        lo, hi = spec
        base = current if current is not None else 0.0
        return None if rng.random() < 0.15 else round(min(hi, max(lo, base + rng.uniform(-10, 10))), 1)
    return rng.choice(spec)


def _random_candidate(base_cfg: dict, rng: random.Random) -> tuple[dict, str]:
    n_fields = rng.choices([1, 2, 3], weights=[0.55, 0.35, 0.10])[0]
    fields = rng.sample(MUTABLE_FIELDS, k=n_fields)
    diff = {}
    for f in fields:
        new_val = _mutate_field(f, base_cfg.get(f), rng)
        if new_val != base_cfg.get(f):
            diff[f] = new_val
    idea = "perturb: " + ", ".join(f"{f} {base_cfg.get(f)}->{v}" for f, v in diff.items()) if diff else "perturb: no-op (resampled same value)"
    return diff, idea


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--candidates-per-round", type=int, default=5)
    ap.add_argument("--restart-prob", type=float, default=0.15,
                    help="probability each round starts from the original baseline instead of the current champion")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rule", choices=list(RULES), default="v2")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = ap.parse_args()
    better_than_champion_fn = RULES[args.rule]
    champion_path, baseline_path, log_path = paths(args.outdir)

    rng = random.Random(args.seed)
    champion = load_or_seed_champion(champion_path, baseline_path, log_path)
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else champion
    round_num = int(pd.read_csv(log_path)["round"].max()) + 1
    promotions = 0

    for i in range(args.rounds):
        restart = rng.random() < args.restart_prob
        round_start_config = baseline["config"] if restart else champion["config"]
        tag = " [restart-from-baseline]" if restart else ""
        print(f"=== [{args.rule}] round {round_num}{tag} | champion={champion['name']} "
             f"train={champion['train_sharpe']:.3f} val={champion['val_sharpe']:.3f} ===", flush=True)

        for c in range(args.candidates_per_round):
            diff, idea = _random_candidate(round_start_config, rng)
            cfg = StrategyConfig(**{**round_start_config, **diff})
            name = f"r{round_num}_{c}_" + "_".join(diff) if diff else f"r{round_num}_{c}_noop"
            stats = run_candidate(cfg)
            row = {"round": round_num, "name": name, "idea": idea, "config": cfg.to_dict(), **stats}
            promoted = better_than_champion_fn(stats, champion, baseline)
            _append_log(log_path, row, promoted)
            if promoted:
                champion = row
                champion_path.write_text(json.dumps(champion, indent=2))
                promotions += 1
                print(f"  {name:<45} train={stats['train_sharpe']:.3f} val={stats['val_sharpe']:.3f}  -> PROMOTED")
        round_num += 1

    print(f"\n{args.rounds} rounds done, {promotions} promotion(s).")
    print(f"final champion: {champion['name']} (train {champion['train_sharpe']:.3f}, val {champion['val_sharpe']:.3f})")
    print(json.dumps(champion["config"], indent=2))


if __name__ == "__main__":
    main()
