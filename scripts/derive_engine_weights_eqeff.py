"""Derive production engine_weights under method EQEFF (equal effective
exposure — no IC information used at all).

Ratified 2026-09-01. Solves for the nominal weight vector w that EQUALIZES
each parent's effective exposure (C.w -- own weight plus what correlated
parents import) at 1/n. Closed form: w \\propto C^{-1} . 1 (the standard
minimum-variance / "risk parity on correlation" solution), clipped >= 0,
renormalized, then cap-trimmed to 0.25 as a safety net (should not bind --
equalizing at 1/8 = 0.125 is already well under the cap).

Motivation: an extensive 2026-09-01 study (scripts/weight_config_study.py,
scripts/full_pit_backtest_eqeff.py/_2020.py, scripts/weight_config_analysis.py,
scripts/factor_scorecard.py, scripts/full_pit_backtest_10configs.py) tested
EQEFF against the incumbent (A), the previously-shipped method E (cap +
diversifier floor), plain equal nominal weight (EQ), and 10 further
IC-informed hybrids (structural C^{-1}.IC solves at several shrinkage levels,
an IR-target variant, three EQEFF/A linear blends, and two sequential
equal/EQEFF + IC-tilt constructions). None of the 10 hybrids beat pure
EQEFF on ANY of a 10-metric portfolio-agnostic scorecard (IC, IR, hit rate,
Q5-Q1, monotonicity, tent asymmetry, within-sector IC, Fama-MacBeth slope
t-stat, score stability, bottom-decile drag) -- ranking quality degrades
MONOTONICALLY as IC information is mixed back in (BLEND25 -> BLEND50 ->
BLEND75 -> EQEFF is a clean, one-directional dose-response curve). EQEFF's
IC edge also survives a within-sector control (rules out sector-rotation as
the explanation) and clears a rigorous Fama-MacBeth significance test
(t=2.45) that the incumbent A does NOT clear (t=0.45).

Caveat carried forward deliberately: a COSTED, CONCENTRATED (top 10-25%)
PORTFOLIO backtest does NOT show EQEFF beating method E -- E remains better
at that specific concentration (scripts/weight_config_analysis.py's
sensitivity grid: E wins 5/12 constructions, EQEFF only 3/12, and EQEFF is
worst-of-four at top_pct=0.10/0.25; EQEFF only overtakes at broader top_pct
>= 0.40). Shipping EQEFF anyway is a deliberate choice to optimize the
composite SCORE's standalone ranking quality (this is what "scoring" in
run_scoring.py / the Scoring dashboard describes) rather than the specific
top-25%/cap5/cap_match live portfolio construction, which the user judged
easier to find a false edge in than a clean ranking metric. If the live
portfolio construction ever gets revisited, re-check E vs EQEFF at that
construction specifically -- don't assume the scorecard verdict transfers.

C = mean per-date Pearson correlation of the zscore-normalized production
parents, computed on the current candidate panel (2023-01 -> latest
month-end, cache/subfactor_expansion/cand_panel_2023-01-01_<end>_monthly_v2.pkl
-- rebuild via research.subfactor_expansion.panel.build_candidate_panel if
stale). Same panel used to ship method E for continuity.

Prints the solved vector + resulting effective exposures; paste the result
into config.yaml engine_weights + factors/parent_selection_v4.py
V4_PARENT_WEIGHTS. Rerun after any SELECTED_SUBS change (the solve depends
on the current correlation structure).

Usage: python scripts/derive_engine_weights_eqeff.py [--panel PATH]
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from factors.parent_selection_v4 import SELECTED_SUBS
from scripts.crowding_diagnostics import parent_score, normalize

CAP = 0.25  # safety-net cap; should not bind at n=8 (equal target = 0.125)

DEFAULT_PANEL = (REPO / "cache" / "subfactor_expansion"
                 / "cand_panel_2023-01-01_2026-08-28_monthly_v2.pkl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    args = ap.parse_args()

    with args.panel.open("rb") as fh:
        panel = pickle.load(fh)
    print(f"panel: {len(panel.rebal_dates)} dates, "
          f"{panel.rebal_dates[0]} .. {panel.rebal_dates[-1]}")

    parents = list(SELECTED_SUBS)
    mats = []
    for d in panel.rebal_dates:
        P = pd.DataFrame({p: parent_score(panel.scores[d], SELECTED_SUBS[p])
                          for p in parents})
        mats.append(P.apply(normalize).corr(method="pearson"))
    C = pd.concat(mats).groupby(level=0, sort=False).mean().loc[parents, parents]

    ones = np.ones(len(parents))
    raw = np.clip(np.linalg.pinv(C.values) @ ones, 0.0, None)
    w = pd.Series(raw, index=parents)
    w = w / w.sum()

    # Cap-trim safety net (should be a no-op at n=8).
    for i in range(300):
        eff = C.values @ w.values
        if eff.max() <= CAP + 0.005:
            break
        over = eff > CAP + 1e-12
        w[over] = w[over] * (CAP / eff[over])
        w = w / w.sum()
    eff_final = pd.Series(C.values @ w.values, index=parents)

    out = pd.DataFrame({
        "nominal_weight": w.round(4),
        "effective_exposure": eff_final.round(4),
        "target_1_over_n": round(1.0 / len(parents), 4),
    })
    print(out.to_string())
    print(f"\ncap-trim iterations used: {i} (0 = cap never bound, as expected)")
    print(f"sum = {w.sum():.4f}")
    print("\nconfig.yaml block:")
    for p in parents:
        print(f"    {p + ':':<14} {w[p]:.4f}")


if __name__ == "__main__":
    main()
