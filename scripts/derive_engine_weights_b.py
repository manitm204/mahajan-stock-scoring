"""Derive production engine_weights under method B (effective-exposure cap).

Ratified 2026-08-07 ("Lets ship B"), minimal variant: keep the incumbent V4
weight vector's proportions, but enforce the 25% cap on EFFECTIVE exposure
(C·w — a parent's own weight plus what flows in through correlated parents)
instead of nominal weight. Evidence: output/crowding/incremental_weights/
REPORT.md (fully-PIT 2020- and 2021-start backtests + cap/λ sweeps; B-0.25
best book in every test, worst case degenerates to the incumbent).

C = mean per-date Pearson correlation of the zscore-normalized production
parents on the clean post-bugfix battery panel (2023-01→2026-06). Trim loop =
research method B verbatim: scale violators by cap/eff, renormalize, repeat.

Prints the before/after vectors + effective exposures; paste the result into
config.yaml engine_weights + factors/parent_selection_v4.py V4_PARENT_WEIGHTS.
Rerun this script after any SELECTED_SUBS or weight re-derivation (the trim
depends on the current correlation structure).

Usage: python scripts/derive_engine_weights_b.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from factors.parent_selection_v4 import SELECTED_SUBS
from research.analyst_deep_dive.common import load_panel
from scripts.crowding_diagnostics import parent_score, normalize

CAP = 0.25

# The trim's base = the incumbent standalone (method-A) derivation, NOT the
# live V4_PARENT_WEIGHTS — after the 2026-08-07 ship those already carry B's
# trim, and seeding from them would double-trim on rerun.
A_BASE = {
    "momentum": 0.170, "value": 0.106, "quality": 0.044, "growth": 0.171,
    "revisions": 0.141, "institutional": 0.152, "insider": 0.034, "short": 0.182,
}


def main():
    panel = load_panel()
    parents = list(SELECTED_SUBS)
    mats = []
    for d in panel.rebal_dates:
        P = pd.DataFrame({p: parent_score(panel.scores[d], SELECTED_SUBS[p])
                          for p in parents})
        mats.append(P.apply(normalize).corr(method="pearson"))
    C = pd.concat(mats).groupby(level=0, sort=False).mean().loc[parents, parents]

    w = pd.Series(A_BASE)[parents]
    w = w / w.sum()
    eff0 = pd.Series(C.values @ w.values, index=parents)
    for i in range(50):
        eff = C.values @ w.values
        if eff.max() <= CAP + 0.005:
            break
        over = eff > CAP + 1e-12
        w[over] = w[over] * (CAP / eff[over])
        w = w / w.sum()
    eff1 = pd.Series(C.values @ w.values, index=parents)

    out = pd.DataFrame({
        "nominal_before": pd.Series(A_BASE)[parents].round(3),
        "eff_before": eff0.round(3),
        "nominal_after": w.round(3),
        "eff_after": eff1.round(3),
    })
    print(out.to_string())
    print(f"\nconverged in {i} iterations; sum = {w.sum():.4f}")
    print("\nconfig.yaml block:")
    for p in parents:
        print(f"    {p + ':':<14} {w[p]:.4f}")


if __name__ == "__main__":
    main()
