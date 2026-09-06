"""Champion tracker for the autoresearch loop -- the same propose / test /
keep-or-discard discipline used in scripts/loop_round.py for strategy #13's
exit-rule search, adapted for free-form candidate.py code instead of a fixed
StrategyConfig dict (so there's no per-parameter round file here: each round
is just "edit candidate.py, run this").

Usage, once per round, after editing candidate.py:

    python -m research.autoresearch.promote

What it does:
  1. Runs `python -m research.autoresearch.evaluate` in a fresh subprocess
     (so a candidate can't taint state across rounds via stale imports).
     evaluate.py itself runs the correctness/leakage test gate first and
     refuses to score anything if it fails.
  2. Reads the row evaluate.py just appended to results.tsv.
  3. No champion yet -> seed it with this run (should be the current
     baseline the first time this is ever called).
  4. Champion exists -> promote only if research_score improved (NaN never
     promotes). Rejected or failed runs restore candidate.py to the
     champion's last-known-good content, so a bad idea never lingers as the
     working file for the next round.

Champion state lives in research/autoresearch/champion/ (candidate.py
snapshot + metrics.json) -- never read by candidate.py or evaluate.py, so it
cannot influence scoring; it's bookkeeping for this script alone.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CANDIDATE = HERE / "candidate.py"
RESULTS_TSV = HERE / "results.tsv"
CHAMPION_DIR = HERE / "champion"
CHAMPION_CANDIDATE = CHAMPION_DIR / "candidate.py"
CHAMPION_METRICS = CHAMPION_DIR / "metrics.json"


def _last_result_row() -> dict:
    df = pd.read_csv(RESULTS_TSV, sep="\t")
    return df.iloc[-1].to_dict()


def _promote(row: dict, reason: str) -> None:
    CHAMPION_DIR.mkdir(exist_ok=True)
    shutil.copy(CANDIDATE, CHAMPION_CANDIDATE)
    CHAMPION_METRICS.write_text(json.dumps({k: str(v) for k, v in row.items()}, indent=2))
    print(f"PROMOTED: {reason}")


def _restore_champion(reason: str) -> None:
    if CHAMPION_CANDIDATE.exists():
        shutil.copy(CHAMPION_CANDIDATE, CANDIDATE)
        print(f"REJECTED: {reason} -- candidate.py restored to champion")
    else:
        print(f"REJECTED: {reason} -- no champion snapshot yet, leaving "
             "candidate.py as-is (nothing to restore to)", file=sys.stderr)


def main() -> int:
    proc = subprocess.run([sys.executable, "-m", "research.autoresearch.evaluate"], cwd=REPO)
    if proc.returncode != 0:
        _restore_champion("evaluate.py failed (correctness tests or output validation)")
        return 1

    row = _last_result_row()
    score = float(row["research_score"])

    if not CHAMPION_METRICS.exists():
        _promote(row, "seeding first champion")
        return 0

    champ = json.loads(CHAMPION_METRICS.read_text())
    champ_score = float(champ["research_score"])

    if score != score:  # NaN
        _restore_champion("research_score is NaN")
        return 1
    if score > champ_score:
        _promote(row, f"research_score improved {champ_score:.4f} -> {score:.4f} "
                     f"(dev_ir {champ['dev_ir']}->{row['dev_ir']}, "
                     f"val_ir {champ['val_ir']}->{row['val_ir']})")
        return 0

    _restore_champion(f"research_score {score:.4f} did not beat champion {champ_score:.4f}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
