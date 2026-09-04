"""Method E — B's effective-exposure cap plus a diversifier floor.

Extends output/crowding/PREREGISTRATION_incremental_weights_2026-08-04.md's
method family (A/B/C/D) with one more challenger, motivated by a gap found in
B after it shipped: B only ever scales DOWN a parent whose effective exposure
(C.w) exceeds the 25% cap — it has no mechanism to reward a parent for being
a genuine diversifier (negative/near-zero effective exposure). In the current
production vector, value and insider sit at eff ≈ -0.02 / -0.03 while five
correlated "consensus-bullish" parents (momentum/growth/revisions/
institutional/short) all cluster at ~0.25 — exactly what the cap allows, but
nothing pulls weight the other way.

Method E: run B's cap-trim to convergence, then iteratively floor any parent
whose effective exposure is below FLOOR_EFF at a minimum NOMINAL weight of
FLOOR_W, funding the deficit proportionally from parents currently above
FLOOR_W, then re-run the cap trim (the floor step can re-violate the cap),
repeating until both constraints hold simultaneously.

NOT pre-registered ahead of time like A-D (this is a same-day follow-up to a
finding, not a separate study) — read results as a screening pass. If E clears
the same bar the A-D study used, it should get a proper held-out check before
touching config.yaml, same as any other weight change here.

Usage: python scripts/derive_engine_weights_e_floor.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.incremental_weight_study as base

FLOOR_EFF = 0.05   # eff exposure below this = "genuine diversifier"
FLOOR_W = 0.08     # minimum nominal weight guaranteed to a diversifier


def method_E(ic, ir, C):
    w = pd.Series(base.method_B(ic, ir, C))[base.PARENTS]
    for _ in range(50):
        eff = C.values @ w.values
        over = eff > base.CAP + 1e-12
        if over.any():
            w[over] = w[over] * (base.CAP / eff[over])
            w = w / w.sum()
            continue
        under_mask = eff < FLOOR_EFF - 1e-12
        under = w.index[under_mask]
        deficit = (FLOOR_W - w[under]).clip(lower=0.0)
        if deficit.sum() < 1e-6:
            break
        donors = w.index.difference(under)
        donor_pool = w[donors].sum()
        if donor_pool <= 1e-9:
            break
        take = deficit.sum() * (w[donors] / donor_pool)
        w[donors] = w[donors] - take
        w[under] = w[under] + deficit
        w = w.clip(lower=0.0)
        w = w / w.sum()
    return w.to_dict()


def main():
    base.METHODS["E"] = method_E
    base.OUT = base.REPO / "output" / "crowding" / "incremental_weights_e_floor"
    base.main()


if __name__ == "__main__":
    main()
