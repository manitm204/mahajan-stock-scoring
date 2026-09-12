"""Generalizes insider_edge_paired.py + insider_ic_by_year.py (user request
2026-09-10) to ALL 8 V4 parent factor families, not just insider -- same
paired Monte Carlo design, same decile-1-minus-100 pool, same year-by-year
table (edge sum, that year's IC, trailing-5yr IC), so the insider result can
be compared against the other 7 parents side by side.

Design per parent (identical to insider_edge_paired.py, just parametrized):
  - pool = decile-1 (top 10% by rank) minus the composite==100 tied names
  - each review date, draw a random 10 from the pool, split top-5 / bottom-5
    by THAT PARENT's score, run both legs forward as full staggered-sleeve
    books (paired: identical random draw at every date, both legs, so any
    diff is attributable only to the parent-score split)
  - average the per-sim return diff across sims per calendar date (kills
    selection-draw noise), sum by year -> "is the edge broad-based or a
    regime fluke" read, same method used to debunk the insider finding on
    this same pool and vet the champion afterward
  - IC of the parent's score vs 1M forward returns, restricted to the same
    pool, by year and trailing-5yr

Usage: python -m research.loop_research.parent_edge_paired
Writes: output/loop_research/parent_edge_paired.json (full detail)
        output/loop_research/parent_edge_paired_year_table.csv (long format)
        output/loop_research/parent_edge_paired_summary.csv (per-parent overall)
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backtesting.data_loader import SPY                                    # noqa: E402
from research.autoresearch.evaluate import bench_returns                    # noqa: E402
from research.forward_returns import compute_forward_returns                # noqa: E402
from research.ic import period_ic                                           # noqa: E402
from research.loop_research import harness as H                             # noqa: E402
from research.loop_research.decile_spread import _decile_pool               # noqa: E402
from research.walkforward.portfolio import performance_metrics              # noqa: E402

OUT_DIR = REPO / "output" / "loop_research"
PARENTS = ["momentum", "value", "quality", "growth", "revisions",
          "institutional", "insider", "short"]
DRAW_SIZE = 10
TOP_K = 5
N_SIMS = 500


def _draw_pool(scores: pd.Series) -> list[str]:
    decile1 = set(_decile_pool(scores, 1))
    at_100 = set(scores[scores == 100].index)
    return sorted(decile1 - at_100)


def _split(pool, parent_scores, rng, parent):
    draw = rng.sample(pool, min(DRAW_SIZE, len(pool))) if pool else []
    if not draw:
        return [], []
    p = parent_scores.get(parent, pd.Series(dtype=float)).reindex(draw)
    ranked = p.sort_index().sort_values(ascending=False, kind="stable")
    return list(ranked.index[:TOP_K]), list(ranked.index[TOP_K:])


def _paired_selector(leg: str, parent: str):
    def selector(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
        pool = _draw_pool(scores)
        top, bottom = _split(pool, parent_scores, rng, parent)
        picks = top if leg == "top" else bottom
        return picks if len(picks) >= 2 else (held if held else picks)
    selector.__name__ = f"paired_{parent}_{leg}"
    return selector


def _stats(pr, spy):
    m = performance_metrics(pr, hold_months=1, benchmarks={"SPY": spy})
    return {"cagr": m["cagr"], "sharpe": m["sharpe"], "alpha": m.get("spy_alpha")}


def run_parent(bundle, parent, n_sims=N_SIMS, seed_base=0):
    data = bundle["data"]
    rebal = list(data.rebal_dates)
    all_dates = rebal[:-1]
    spy = bench_returns(data.matrix, SPY, rebal)

    top_sel = _paired_selector("top", parent)
    bot_sel = _paired_selector("bottom", parent)

    ret_diff = np.zeros((n_sims, len(all_dates)))
    diffs = {"cagr": [], "sharpe": [], "alpha": []}
    for s in range(n_sims):
        seed = seed_base + s
        pr_t, _, _ = H.run_one(bundle, top_sel, seed, k=TOP_K, refresh_n=TOP_K)
        pr_b, _, _ = H.run_one(bundle, bot_sel, seed, k=TOP_K, refresh_n=TOP_K)
        st_t, st_b = _stats(pr_t, spy), _stats(pr_b, spy)
        for kk in diffs:
            diffs[kk].append(st_t[kk] - st_b[kk])
        pr_t_r = pr_t.reindex(all_dates).fillna(0.0)
        pr_b_r = pr_b.reindex(all_dates).fillna(0.0)
        ret_diff[s] = (pr_t_r - pr_b_r).values

    period_mean_diff = ret_diff.mean(axis=0)
    years = pd.to_datetime(all_dates).year
    year_sum = pd.Series(period_mean_diff, index=years).groupby(level=0).sum()
    t_stat, p_val = stats.ttest_1samp(period_mean_diff, popmean=0.0)

    return {
        "parent": parent,
        "dates": [str(d) for d in all_dates],
        "period_mean_diff": period_mean_diff.tolist(),
        "year_sum": {int(y): float(v) for y, v in year_sum.items()},
        "t_stat": float(t_stat), "p_value": float(p_val),
        "pct_months_positive": float((period_mean_diff > 0).mean()),
        "diffs_summary": {f"median_{kk}": float(np.median(v)) for kk, v in diffs.items()}
                        | {f"win_rate_{kk}": float(np.mean(np.array(v) > 0)) for kk, v in diffs.items()},
    }


def ic_by_year(bundle, parent):
    data = bundle["data"]
    parent_scores_by_date = bundle["parent_scores"]
    rebal = list(data.rebal_dates)
    fwd_1m = compute_forward_returns(data.matrix, rebal, {"1M": 1})["1M"]

    rows = []
    for d in rebal:
        if d not in fwd_1m or d not in parent_scores_by_date:
            continue
        p_score = parent_scores_by_date[d].get(parent)
        scores_d = data.comp.get(d)
        if p_score is None or scores_d is None:
            continue
        pool = _draw_pool(scores_d)
        if not pool:
            continue
        ic = period_ic(p_score.reindex(pool), fwd_1m[d].reindex(pool))
        rows.append({"date": d, "year": pd.Timestamp(d).year, "ic": ic})
    df = pd.DataFrame(rows)

    out = {}
    for y in sorted(df["year"].unique()):
        ic_y = df[df["year"] == y]["ic"].dropna()
        # strictly PRIOR years only (y-5 .. y-1) -- excluding y itself avoids
        # "trailing" IC silently being concurrent with the year it's compared
        # against.
        trail = df[(df["year"] >= y - 5) & (df["year"] < y)]["ic"].dropna()
        out[int(y)] = {
            "ic_that_year": float(ic_y.mean()) if len(ic_y) else float("nan"),
            "trailing5y_ic": float(trail.mean()) if len(trail) else float("nan"),
        }
    return out


def main():
    bundle = H.get_data()
    all_results = {}
    year_rows = []
    summary_rows = []

    t0 = time.time()
    for i, parent in enumerate(PARENTS):
        print(f"[{i+1}/{len(PARENTS)}] running paired MC for parent={parent} ...", flush=True)
        edge = run_parent(bundle, parent)
        ic = ic_by_year(bundle, parent)
        all_results[parent] = {"edge": edge, "ic": ic}

        for y, edge_sum in edge["year_sum"].items():
            row = {"parent": parent, "year": y, "edge_year_sum": edge_sum}
            row.update(ic.get(y, {}))
            year_rows.append(row)

        summary_rows.append({
            "parent": parent,
            "t_stat": edge["t_stat"], "p_value": edge["p_value"],
            "pct_months_positive": edge["pct_months_positive"],
            **edge["diffs_summary"],
        })
        print(f"  parent={parent}  t={edge['t_stat']:.2f}  p={edge['p_value']:.4f}  "
             f"({time.time()-t0:.0f}s elapsed)", flush=True)

    with (OUT_DIR / "parent_edge_paired.json").open("w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"wrote {OUT_DIR / 'parent_edge_paired.json'}")

    year_table = pd.DataFrame(year_rows)
    year_table.to_csv(OUT_DIR / "parent_edge_paired_year_table.csv", index=False)
    print(f"wrote {OUT_DIR / 'parent_edge_paired_year_table.csv'}")

    summary_table = pd.DataFrame(summary_rows).sort_values("p_value")
    summary_table.to_csv(OUT_DIR / "parent_edge_paired_summary.csv", index=False)
    print(f"wrote {OUT_DIR / 'parent_edge_paired_summary.csv'}")
    print(summary_table.to_string(index=False))


if __name__ == "__main__":
    main()
