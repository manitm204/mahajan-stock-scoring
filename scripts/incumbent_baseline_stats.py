"""Baseline capture for the incremental-weight study: incumbent composite stats.

Computes the production-replica composite (current SELECTED_SUBS + static
engine weights, zscore std-20 normalization, sector-percentile re-rank) on the
42-month battery panel (2023-01→2026-06) and records IC/IR/quintile statistics.
These are the "current approach" numbers the pre-registered challenger methods
will be compared against (alongside the walk-forward baseline the study itself
produces). Run BEFORE any challenger evaluation.

Usage: python scripts/incumbent_baseline_stats.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from factors.parent_selection_v4 import SELECTED_SUBS, V4_PARENT_WEIGHTS
from factors.utils import sector_percentile
from research.analyst_deep_dive.common import (
    load_panel, load_price_matrix, panel_forward_returns, ic_series,
    summarize_ic, sector_map,
)
from scripts.crowding_diagnostics import parent_score, normalize, NEUTRAL

OUT = REPO / "output" / "crowding"


def build_composite_by_date(panel, dates, sectors):
    """Production-replica composite per date: raw blend + sector re-rank."""
    W = pd.Series(V4_PARENT_WEIGHTS)
    W = W / W.sum()
    comp = {}
    for d in dates:
        frame = panel.scores[d]
        P = pd.DataFrame({p: parent_score(frame, SELECTED_SUBS[p])
                          for p in SELECTED_SUBS})
        Pn = P.apply(normalize)
        raw = Pn.mul(W).sum(axis=1)
        sec = sectors.reindex(raw.index)
        comp[d] = sector_percentile(raw, sec, higher_is_better=True)
    return comp


def quintile_spread(comp, fwd, h="3M"):
    """Mean Q5−Q1 forward-return spread of the composite."""
    spreads = []
    for d, s in comp.items():
        if d not in fwd[h]:
            continue
        df = pd.DataFrame({"s": s, "f": fwd[h][d]}).dropna()
        if len(df) < 100:
            continue
        q = pd.qcut(df["s"].rank(method="first"), 5, labels=False)
        m = df.groupby(q)["f"].mean()
        spreads.append(float(m.iloc[-1] - m.iloc[0]))
    return float(np.mean(spreads)), len(spreads)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    dates = panel.rebal_dates
    sectors = sector_map(normalized=False)
    comp = build_composite_by_date(panel, dates, sectors)

    matrix = load_price_matrix(start="2022-06-01", end="2026-07-29")
    fwd = panel_forward_returns(matrix, dates)

    out = {"window": f"{dates[0]}..{dates[-1]}", "n_rebalances": len(dates),
           "weights": dict(V4_PARENT_WEIGHTS)}
    ics = {}
    for h in ("3M", "6M"):
        s = ic_series(comp, fwd[h])
        ics[h] = s
        out[f"ic_{h}"] = summarize_ic(s)
    blend = pd.concat(ics.values(), axis=1).mean(axis=1)
    out["ic_blend_3m6m"] = summarize_ic(blend)
    sp3, n3 = quintile_spread(comp, fwd, "3M")
    sp6, n6 = quintile_spread(comp, fwd, "6M")
    out["q5_q1_spread_3M"] = {"mean": round(sp3, 4), "n": n3}
    out["q5_q1_spread_6M"] = {"mean": round(sp6, 4), "n": n6}

    for k in ("ic_3M", "ic_6M", "ic_blend_3m6m"):
        out[k] = {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                  for kk, vv in out[k].items()}
    with open(OUT / "incumbent_baseline_inwindow.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
