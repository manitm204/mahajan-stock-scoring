"""Dump parent correlation data for the incremental-weight-study charts.

Saves (to output/crowding/incremental_weights/):
  corr_by_year.json   — the derivation-window mean Pearson C used by each
                        eval year's weight derivation (trailing 5y, exactly as
                        in scripts/incremental_weight_study.py)
  corr_pairs_monthly.json — per-rebalance cross-sectional Pearson correlation
                        for all 28 parent pairs, 2015-2026 (normalized parents)

Usage: python scripts/dump_parent_correlations.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from factors.parent_selection_v4 import SELECTED_SUBS
from research.subfactor_expansion.panel import load_cached_panel
from scripts.crowding_diagnostics import parent_score, normalize
from scripts.incremental_weight_study import PANEL_PKL, OUT, PARENTS, EVAL_YEARS


def main():
    panel = load_cached_panel(PANEL_PKL)
    dates = panel.rebal_dates
    corr_by = {}
    for d in dates:
        frame = panel.scores[d]
        P = pd.DataFrame({p: parent_score(frame, SELECTED_SUBS[p]) for p in PARENTS})
        corr_by[d] = P.apply(normalize).corr(method="pearson")

    yearly = {}
    for Y in EVAL_YEARS:
        deriv = [d for d in dates if f"{Y-5}-01-01" <= d <= f"{Y-1}-12-31"]
        if not deriv:
            continue
        C = (pd.concat([corr_by[d] for d in deriv])
             .groupby(level=0, sort=False).mean().loc[PARENTS, PARENTS])
        yearly[str(Y)] = {"n_dates": len(deriv),
                          "window": f"{deriv[0]}..{deriv[-1]}",
                          "matrix": [[round(float(C.iloc[i, j]), 3)
                                      for j in range(len(PARENTS))]
                                     for i in range(len(PARENTS))]}

    pairs = {}
    for i, a in enumerate(PARENTS):
        for b in PARENTS[i + 1:]:
            pairs[f"{a}|{b}"] = [round(float(corr_by[d].loc[a, b]), 3) for d in dates]

    with open(OUT / "corr_by_year.json", "w") as f:
        json.dump({"parents": PARENTS, "years": yearly}, f, indent=1)
    with open(OUT / "corr_pairs_monthly.json", "w") as f:
        json.dump({"dates": dates, "pairs": pairs}, f)
    print(f"wrote corr_by_year.json ({len(yearly)} years) + "
          f"corr_pairs_monthly.json ({len(dates)} dates, {len(pairs)} pairs)")


if __name__ == "__main__":
    main()
