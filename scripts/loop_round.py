"""Run one round of the loop-engineering search on strategy #13's family
(Karpathy-style propose -> test -> keep-or-discard, user request 2026-09-02).

Ideas are proposed by hand (in practice: by Claude, reading the log each
round and writing the next round's JSON) -- this script just scores them and
keeps the books. Each candidate is evaluated by research/loop_engineering/
harness.py on the train/val split; whether it's promoted depends on --rule:

  v1 -- train Sharpe must improve, val Sharpe must not collapse too far
        relative to the champion OR the original baseline (see harness.py's
        PROMOTE_VAL_RATIO). This is what rounds 1-71 used; it let val drift
        away slowly over many individually-defensible promotions.
  v2 -- (train_sharpe + val_sharpe)/2 must improve, and neither leg may drop
        by more than harness.MAX_LEG_DROP even if the average improves.
  v3 -- (user request 2026-09-03) BOTH train_sharpe AND val_sharpe must
        individually improve over the champion -- strict dominance. Since
        val can only ever go up under this rule, drift is impossible by
        construction. Defaults to its own --outdir so it never touches the
        v1/v2 results.

Usage:
  python scripts/loop_round.py rounds/round01.json
  python scripts/loop_round.py rounds/roundNN.json --rule v3 --outdir output/loop_engineering_v3

Round JSON = a list of candidate ideas:
  [{"name": "24a_trail08_cap12M",
    "idea": "tighter 8% trail, same 12M cap",
    "config": {"trail_pct": 0.08}}, ...]
`config` only needs the fields that DIFFER from the current champion's
config -- missing fields inherit the champion's value, so each round file
stays short and each idea's diff is obvious at a glance.

First-ever call in a given --outdir (no champion.json yet) seeds the
champion with the existing production strategy (#13: trail_pct=0.10,
cap_months=12, k=10) as round 0, before scoring anything in the round file.

Outputs (under --outdir, default output/loop_engineering/):
  champion.json  -- current best config + its full/train/val stats
  log.csv        -- one row per candidate ever tried, across every round
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from research.loop_engineering.harness import (
    better_than_champion, better_than_champion_v2, better_than_champion_v3, run_candidate,
)
from research.strategies.generic_rule import StrategyConfig

RULES = {"v1": better_than_champion, "v2": better_than_champion_v2, "v3": better_than_champion_v3}
DEFAULT_OUTDIR = REPO / "output" / "loop_engineering"

BASELINE_CONFIG = StrategyConfig(k=10, trail_pct=0.10, cap_months=12)
BASELINE_NAME = "13_trailstop10_cap12M"


def paths(outdir: Path) -> tuple[Path, Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / "champion.json", outdir / "baseline.json", outdir / "log.csv"


def _append_log(log_path: Path, row: dict, promoted: bool) -> None:
    flat = {"round": row["round"], "name": row["name"], "idea": row["idea"],
           "promoted": promoted}
    flat.update({k: v for k, v in row.items() if k.startswith(("full_", "train_", "val_"))})
    flat["config"] = json.dumps(row["config"])
    pd.DataFrame([flat]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)


def load_or_seed_champion(champion_path: Path, baseline_path: Path, log_path: Path) -> dict:
    if champion_path.exists():
        return json.loads(champion_path.read_text())
    print(f"no champion.json yet in {champion_path.parent} -- seeding round 0 with the "
         f"existing production strategy ({BASELINE_NAME}) ...", flush=True)
    stats = run_candidate(BASELINE_CONFIG)
    champ = {"round": 0, "name": BASELINE_NAME, "idea": "baseline (current production #13)",
             "config": BASELINE_CONFIG.to_dict(), **stats}
    champion_path.write_text(json.dumps(champ, indent=2))
    baseline_path.write_text(json.dumps(champ, indent=2))
    _append_log(log_path, champ, promoted=True)
    return champ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("round_file", type=Path)
    ap.add_argument("--rule", choices=list(RULES), default="v2")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = ap.parse_args()
    better_than_champion_fn = RULES[args.rule]
    champion_path, baseline_path, log_path = paths(args.outdir)

    ideas = json.loads(args.round_file.read_text())

    champion = load_or_seed_champion(champion_path, baseline_path, log_path)
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    round_num = int(pd.read_csv(log_path)["round"].max()) + 1

    print(f"=== [{args.rule}] round {round_num}: {len(ideas)} candidate(s) | champion = {champion['name']} "
         f"(train sharpe {champion['train_sharpe']:.3f}, val sharpe {champion['val_sharpe']:.3f}) ===")

    # Every idea this round is a diff against the champion AS OF THE START OF
    # THE ROUND, not against whatever won earlier in the same round -- ideas
    # within a round are independent, parallel proposals, not a chain.
    round_start_config = champion["config"]
    for idea in ideas:
        cfg = StrategyConfig(**{**round_start_config, **idea.get("config", {})})
        stats = run_candidate(cfg)
        row = {"round": round_num, "name": idea["name"], "idea": idea.get("idea", ""),
              "config": cfg.to_dict(), **stats}
        promoted = better_than_champion_fn(stats, champion, baseline)
        _append_log(log_path, row, promoted)
        tag = "PROMOTED" if promoted else "rejected"
        print(f"  {idea['name']:<35} train_sharpe={stats['train_sharpe']:.3f} "
             f"val_sharpe={stats['val_sharpe']:.3f}  -> {tag}")
        if promoted:
            champion = row
            champion_path.write_text(json.dumps(champion, indent=2))

    print(f"\nchampion after round {round_num}: {champion['name']} "
         f"(train sharpe {champion['train_sharpe']:.3f}, val sharpe {champion['val_sharpe']:.3f})")
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()
