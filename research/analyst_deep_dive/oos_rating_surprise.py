"""Out-of-sample check for the rating-selectivity candidates, 2016-2022.

The battery window is 2023-01→2026-06 and the candidates were designed with
that window in view; analyst_grade_events reaches back to ~2012, so the
2016-2022 monthly panel is genuinely unseen data — the first true OOS test
available to any analyst candidate in this project (price-target history only
starts 2021-04). PIT universe via members_as_of; same scoring and IC
machinery as the battery.

Run:  python -m research.analyst_deep_dive.oos_rating_surprise
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.db import Database
from factors.utils import DataContext, sector_percentile
from research.subfactor_expansion.library_flow import _grade_action_signals
from research.analyst_deep_dive.common import (
    OUT_DIR, DB_PATH, load_price_matrix, panel_forward_returns, sector_map,
    ic_series, summarize_ic, bootstrap_mean_ci,
)

START, END = "2016-01-01", "2022-12-31"


def main():
    db = Database(path=DB_PATH)
    matrix = load_price_matrix(start="2015-01-01", end="2023-07-31")
    idx = pd.DatetimeIndex(pd.to_datetime(matrix.index))
    months = pd.date_range(START, END, freq="ME")
    rebals = []
    for m in months:
        pos = idx.searchsorted(m, side="right") - 1
        if pos >= 0 and (m - idx[pos]).days <= 7:
            rebals.append(matrix.index[pos])
    print(f"{len(rebals)} rebalances {rebals[0]}..{rebals[-1]}")

    sectors = sector_map(normalized=False)
    fwd = panel_forward_returns(matrix, rebals)

    scores = {"rating_surprise": {}, "selective_bull": {}}
    coverage = []
    for i, d in enumerate(rebals, 1):
        members = db.members_as_of(d) or db.universe_tickers()
        ctx = DataContext(db, as_of=d, universe=members, reporting_lag=True)
        _, surprise, bull = _grade_action_signals(ctx, window_days=90)
        sec = sectors.reindex(surprise.index)
        scores["rating_surprise"][d] = sector_percentile(surprise, sec, True)
        scores["selective_bull"][d] = sector_percentile(bull, sec, True)
        coverage.append(float(surprise.notna().mean()))
        if i % 12 == 0 or i == len(rebals):
            print(f"{i}/{len(rebals)} ({d}) cov={coverage[-1]:.2f}", flush=True)

    out = {"n_rebalances": len(rebals), "mean_coverage": float(np.mean(coverage))}
    for name, sc in scores.items():
        for h in ["3M", "6M"]:
            ics = ic_series(sc, fwd[h])
            s = summarize_ic(ics)
            _, lo, hi = bootstrap_mean_ci(ics)
            yearly = ics.groupby(ics.index.map(lambda x: x[:4])).mean()
            out[f"{name}_{h}"] = {**s, "ci90": (round(lo, 4), round(hi, 4)),
                                  "by_year": {k: round(float(v), 4) for k, v in yearly.items()}}
    with open(OUT_DIR / "oos_rating_surprise_2016_2022.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
