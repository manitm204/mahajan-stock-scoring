"""Derive production engine_weights under method E (effective-exposure cap +
diversifier floor).

Ratified [pending — see output/crowding/incremental_weights_e_floor/]. Extends
method B (shipped 2026-08-07, scripts/derive_engine_weights_b.py): keep B's
25%-effective-exposure cap-trim exactly as-is, then add a floor — any parent
whose effective exposure (C·w, own weight plus what correlated parents import)
sits below 5% gets its NOMINAL weight lifted to a minimum of 8%, funded
proportionally from parents currently above that floor. Re-run the cap trim
after any floor adjustment (flooring can re-violate the cap) and iterate both
to convergence.

Motivation: B only ever scales a crowded parent DOWN; nothing in B rewards a
parent for being a genuine diversifier. In the current production vector,
value and insider sit at negative effective exposure (they're anti-correlated
with the 5-parent momentum/growth/revisions/institutional/short cluster) but
B leaves them essentially untouched. E explicitly floors them.

Evidence: scripts/incremental_weight_study.py method E clears the 2026-08-04
pre-registered walk-forward IC/IR bar decisively (blend IC 0.0663 vs
incumbent A's 0.0554, paired-diff CI90 [+0.0070, +0.0149], best of A/B/C/D/E).
scripts/full_pit_backtest_e.py (fully-PIT costed portfolio, 2021-2026 OOS)
then confirms it translates to real returns, unlike method D which won on IC
but lost on portfolio Sharpe/CAGR: E CAGR 18.6% / Sharpe 1.078 / Calmar 0.587
/ max_dd -31.6% / alpha_t 1.84 — best alpha-t of A/B/D/E, ties or beats B on
every metric but Sortino (0.945 vs B's 0.988). Not independently pre-
registered ahead of time (same-day follow-up to the B-derivation finding);
read as sufficient given it already spans the same 2021-2026 OOS range every
other method here was judged on.

C = mean per-date Pearson correlation of the zscore-normalized production
parents, computed on the current candidate panel (2023-01 -> latest month-end,
cache/subfactor_expansion/cand_panel_2023-01-01_<end>_monthly_v2.pkl — rebuild
via scripts (research.subfactor_expansion.panel.build_candidate_panel) if the
panel has gone stale; a 2-month staleness check (2026-08-31 vs 2026-06-30)
found the trim unchanged, so this isn't sensitive to small staleness, but
don't let it drift for a year).

Prints the before/after vectors + effective exposures; paste the result into
config.yaml engine_weights + factors/parent_selection_v4.py V4_PARENT_WEIGHTS.
Rerun this script after any SELECTED_SUBS or weight re-derivation (the trim
depends on the current correlation structure).

Usage: python scripts/derive_engine_weights_e.py [--panel PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from factors.parent_selection_v4 import SELECTED_SUBS
from scripts.crowding_diagnostics import parent_score, normalize

CAP = 0.25
FLOOR_EFF = 0.05    # eff exposure below this = "genuine diversifier"
FLOOR_W = 0.08       # minimum nominal weight guaranteed to a diversifier

DEFAULT_PANEL = (REPO / "cache" / "subfactor_expansion"
                 / "cand_panel_2023-01-01_2026-08-28_monthly_v2.pkl")

# The trim's base = the incumbent standalone (method-A) derivation, NOT the
# live V4_PARENT_WEIGHTS — seeding from live weights would double-trim on rerun.
A_BASE = {
    "momentum": 0.170, "value": 0.106, "quality": 0.044, "growth": 0.171,
    "revisions": 0.141, "institutional": 0.152, "insider": 0.034, "short": 0.182,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    args = ap.parse_args()

    import pickle
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

    w = pd.Series(A_BASE)[parents]
    w = w / w.sum()
    eff0 = pd.Series(C.values @ w.values, index=parents)

    # Stage 1: B's cap trim to convergence.
    for _ in range(300):
        eff = C.values @ w.values
        if eff.max() <= CAP + 0.005:
            break
        over = eff > CAP + 1e-12
        w[over] = w[over] * (CAP / eff[over])
        w = w / w.sum()
    w_B = w.copy()
    eff_B = pd.Series(C.values @ w_B.values, index=parents)

    # Stage 2: floor + re-trim to joint convergence.
    for i in range(300):
        eff = C.values @ w.values
        over = eff > CAP + 1e-12
        if over.any():
            w[over] = w[over] * (CAP / eff[over])
            w = w / w.sum()
            continue
        under = w.index[eff < FLOOR_EFF - 1e-12]
        deficit = (FLOOR_W - w[under]).clip(lower=0.0)
        if deficit.sum() < 1e-8:
            break
        donors = w.index.difference(under)
        pool = w[donors].sum()
        if pool <= 1e-9:
            break
        take = deficit.sum() * (w[donors] / pool)
        w[donors] = w[donors] - take
        w[under] = w[under] + deficit
        w = w.clip(lower=0.0)
        w = w / w.sum()
    eff1 = pd.Series(C.values @ w.values, index=parents)

    out = pd.DataFrame({
        "nominal_A_base": pd.Series(A_BASE)[parents].round(3),
        "eff_A_base": eff0.round(3),
        "nominal_B": w_B.round(3),
        "eff_B": eff_B.round(3),
        "nominal_E": w.round(4),
        "eff_E": eff1.round(3),
    })
    print(out.to_string())
    print(f"\nconverged in {i} iterations; sum = {w.sum():.4f}")
    print("\nconfig.yaml block:")
    for p in parents:
        print(f"    {p + ':':<14} {w[p]:.4f}")


if __name__ == "__main__":
    main()
