"""Tests the user's follow-up idea (2026-09-10): instead of always tie-
breaking the composite==100 pool by INSIDER score (the champion, exp5),
dynamically pick whichever of the 8 V4 parents has the highest TRAILING
5-YEAR IC as of that review date (computed on the ==100 pool, same universe
the champion actually trades), and tie-break by that parent's score instead.
Hypothesis: momentum/institutional/growth showed a (weak, not-significant)
positive correlation between trailing IC and edge in the per-parent
correlation study -- so deliberately routing to "whichever parent's IC is
currently strongest" might do better than always using insider.

No look-ahead: trailing IC at review date d uses only monthly ICs realized
at dates d' < d (i.e. from periods whose forward return had already
occurred by d), over an expanding window capped at 5 years. Before any
history exists, falls back to "insider" (the pre-registered baseline) so
early dates aren't arbitrary.

Usage: python -m research.loop_research.dynamic_best_ic
Writes: output/loop_research/dynamic_best_ic.json
        output/loop_research/dynamic_best_ic.png
"""
from __future__ import annotations

import json
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
from research.loop_research.candidates import top5_by_insider               # noqa: E402

OUT_DIR = REPO / "output" / "loop_research"
PARENTS = ["momentum", "value", "quality", "growth", "revisions",
          "institutional", "insider", "short"]
FALLBACK_PARENT = "insider"
TRAIL_YEARS = 5
N_SIMS = 500
BLUE = "#2a78d6"
MAGENTA = "#e87ba4"
ORANGE = "#eb6834"
GRAY = "#8a8f98"


def _ic_all_parents_by_date(bundle) -> pd.DataFrame:
    """Per-date, per-parent 1M IC on the composite==100 pool (the champion's
    actual selection universe -- NOT the broader decile-1-minus-100 pool
    used in the correlation study)."""
    data = bundle["data"]
    parent_scores_by_date = bundle["parent_scores"]
    rebal = list(data.rebal_dates)
    fwd_1m = compute_forward_returns(data.matrix, rebal, {"1M": 1})["1M"]

    rows = []
    for d in rebal:
        if d not in fwd_1m:
            continue
        scores_d = data.comp.get(d)
        if scores_d is None:
            continue
        pool = list(scores_d[scores_d == 100].index)
        if not pool:
            continue
        pscores = parent_scores_by_date.get(d, {})
        for parent in PARENTS:
            p_score = pscores.get(parent)
            if p_score is None:
                continue
            ic = period_ic(p_score.reindex(pool), fwd_1m[d].reindex(pool), min_names=8)
            rows.append({"date": d, "parent": parent, "ic": ic})
    return pd.DataFrame(rows)


def compute_winner_by_date(ic_df: pd.DataFrame, rebal_dates: list[str]) -> dict[str, str]:
    """No look-ahead: trailing IC at date d uses only ICs from dates strictly
    before d, expanding window capped at TRAIL_YEARS."""
    ic_df = ic_df.copy()
    ic_df["date_ts"] = pd.to_datetime(ic_df["date"])
    winner_by_date = {}
    for d in rebal_dates:
        d_ts = pd.Timestamp(d)
        lo = d_ts - pd.DateOffset(years=TRAIL_YEARS)
        prior = ic_df[(ic_df["date_ts"] < d_ts) & (ic_df["date_ts"] >= lo)]
        if prior.empty:
            winner_by_date[d] = FALLBACK_PARENT
            continue
        trailing_mean = prior.groupby("parent")["ic"].mean().dropna()
        winner_by_date[d] = trailing_mean.idxmax() if not trailing_mean.empty else FALLBACK_PARENT
    return winner_by_date


def make_dynamic_selector(winner_by_date: dict[str, str]):
    def selector(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
        d = ctx["date"]
        parent = winner_by_date.get(d, FALLBACK_PARENT)
        pool = list(scores[scores == 100].index)
        p_score = parent_scores.get(parent, pd.Series(dtype=float)).reindex(pool)
        ranked = p_score.sort_index().sort_values(ascending=False, kind="stable")
        picks = list(ranked.index[:k])
        if len(picks) < k:
            remaining = p_score[~p_score.index.isin(picks)].sort_index().sort_values(ascending=False, kind="stable")
            picks += list(remaining.index[:k - len(picks)])
        return picks if picks else held
    selector.__name__ = "dynamic_best_trailing_ic"
    return selector


def main():
    bundle = H.get_data()
    data = bundle["data"]
    rebal = list(data.rebal_dates)
    all_dates = rebal[:-1]

    print("computing per-date IC for all 8 parents (==100 pool) ...", flush=True)
    ic_df = _ic_all_parents_by_date(bundle)
    winner_by_date = compute_winner_by_date(ic_df, rebal)

    winner_counts = pd.Series(list(winner_by_date.values())).value_counts()
    print("winner (trailing-IC leader) frequency across review dates:")
    print(winner_counts.to_string())

    dyn_sel = make_dynamic_selector(winner_by_date)
    print("\nrunning dynamic-selector path (deterministic) ...", flush=True)
    pr_dyn, _, tg_dyn = H.run_one(bundle, dyn_sel, seed=0)
    pr_dyn_r = pr_dyn.reindex(all_dates).fillna(0.0)

    print("running champion (top5_by_insider) path (deterministic) ...", flush=True)
    pr_champ, _, _ = H.run_one(bundle, top5_by_insider, seed=0)
    pr_champ_r = pr_champ.reindex(all_dates).fillna(0.0)

    print(f"running {N_SIMS} random-baseline sims ...", flush=True)
    t0 = time.time()
    rand_returns = np.zeros((N_SIMS, len(all_dates)))
    for s in range(N_SIMS):
        pr_r, _, _ = H.run_one(bundle, H.random_selector, seed=s)
        rand_returns[s] = pr_r.reindex(all_dates).fillna(0.0).values
        if (s + 1) % 100 == 0:
            print(f"  sim {s+1}/{N_SIMS}  ({time.time()-t0:.0f}s elapsed)", flush=True)
    mean_random = rand_returns.mean(axis=0)

    diff_vs_random = pr_dyn_r.values - mean_random
    diff_vs_champ = pr_dyn_r.values - pr_champ_r.values

    years = pd.to_datetime(all_dates).year
    year_sum_vs_random = pd.Series(diff_vs_random, index=years).groupby(level=0).sum()
    year_sum_vs_champ = pd.Series(diff_vs_champ, index=years).groupby(level=0).sum()

    t_r, p_r = stats.ttest_1samp(diff_vs_random, popmean=0.0)
    t_c, p_c = stats.ttest_1samp(diff_vs_champ, popmean=0.0)

    spy = bench_returns(data.matrix, SPY, rebal).reindex(all_dates).fillna(0.0)

    def cum(x):
        return (1.0 + pd.Series(x, index=all_dates)).cumprod()

    result = {
        "dates": all_dates,
        "winner_counts": winner_counts.to_dict(),
        "winner_by_date": winner_by_date,
        "dyn_curve": cum(pr_dyn_r.values).tolist(),
        "champ_curve": cum(pr_champ_r.values).tolist(),
        "random_mean_curve": cum(mean_random).tolist(),
        "spy_curve": cum(spy.values).tolist(),
        "year_sum_vs_random": {int(y): float(v) for y, v in year_sum_vs_random.items()},
        "year_sum_vs_champ": {int(y): float(v) for y, v in year_sum_vs_champ.items()},
        "t_vs_random": float(t_r), "p_vs_random": float(p_r),
        "t_vs_champ": float(t_c), "p_vs_champ": float(p_c),
        "pct_months_positive_vs_random": float((diff_vs_random > 0).mean()),
        "pct_months_positive_vs_champ": float((diff_vs_champ > 0).mean()),
    }
    with (OUT_DIR / "dynamic_best_ic.json").open("w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nwrote {OUT_DIR / 'dynamic_best_ic.json'}")

    print(f"\ndynamic vs RANDOM baseline: t={t_r:.2f} p={p_r:.4f} "
         f"{result['pct_months_positive_vs_random']:.0%} months positive")
    print(f"  per-year: {result['year_sum_vs_random']}")
    print(f"\ndynamic vs CHAMPION (top5_by_insider): t={t_c:.2f} p={p_c:.4f} "
         f"{result['pct_months_positive_vs_champ']:.0%} months positive")
    print(f"  per-year: {result['year_sum_vs_champ']}")

    _plot(result)


def _plot(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    dates = pd.to_datetime(result["dates"])

    ax = axes[0]
    ax.plot(dates, result["dyn_curve"], color=MAGENTA, linewidth=2, label="dynamic best-trailing-IC")
    ax.plot(dates, result["champ_curve"], color=ORANGE, linewidth=2, label="champion (insider, exp5)")
    ax.plot(dates, result["random_mean_curve"], color=BLUE, linewidth=1.5, label="random baseline (mean)")
    ax.plot(dates, result["spy_curve"], color=GRAY, linestyle="--", linewidth=1.25, label="SPY")
    ax.set_ylabel("growth of $1")
    ax.set_title("Dynamic best-trailing-IC tie-break vs champion vs random")
    ax.legend(fontsize=8.5)

    ax2 = axes[1]
    years = sorted(result["year_sum_vs_champ"].keys())
    vs_champ = [result["year_sum_vs_champ"][y] for y in years]
    colors = [MAGENTA if v > 0 else BLUE for v in vs_champ]
    ax2.bar([str(y) for y in years], vs_champ, color=colors)
    ax2.axhline(0.0, color=GRAY, linestyle="--", linewidth=1.25)
    ax2.set_ylabel("sum of monthly (dynamic - champion) diff")
    ax2.set_title(f"Dynamic vs champion, per year  (p={result['p_vs_champ']:.3f} overall)")

    fig.suptitle("Does routing to the highest-trailing-IC parent beat always using insider?", fontsize=12)
    fig.tight_layout()
    out_png = OUT_DIR / "dynamic_best_ic.png"
    fig.savefig(out_png, dpi=140)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
