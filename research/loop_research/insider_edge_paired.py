"""Paired Monte Carlo test of the insider edge (user request 2026-09-10):
is "pick the best-insider names" actually doing something, or is exp5's win
rate just a lucky near-tie in one pool?

Design: at every review date, take the top-decile pool (same equal-sized
rank decile-1 definition as decile_spread.py) MINUS the composite==100 tied
names -- i.e. the "good but not already elite" pool, typically ~35-45 names.
Draw a random 10 from that pool, rank those 10 by insider parent score, and
split top-5 / bottom-5. Both legs are run forward as a full staggered-sleeve
book (4 sleeves, 4-month hold, full turnover) and BOTH LEGS SHARE THE SAME
RANDOM DRAW at every single review date across the whole path (same rng
seed, same number/order of rng calls in each selector, diverging only in the
deterministic post-draw split) -- so any sim-to-sim difference in performance
is attributable ONLY to which 5 of the 10 names insider score picked, not to
market timing or which stocks entered the random pool. This is a paired
design: for each sim we get one clean estimate of the insider-vs-other diff
with the regime/timing noise cancelled out.

If the insider edge is real, the distribution of per-sim (insider - other)
should be centered clearly above 0, not straddling it.

Usage: python -m research.loop_research.insider_edge_paired
Writes: output/loop_research/insider_edge_paired.json
        output/loop_research/insider_edge_paired.png
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backtesting.data_loader import SPY, QQQ                          # noqa: E402
from research.autoresearch.evaluate import bench_returns               # noqa: E402
from research.loop_research import harness as H                        # noqa: E402
from research.loop_research.decile_spread import _decile_pool          # noqa: E402
from research.walkforward.portfolio import performance_metrics         # noqa: E402

OUT_JSON = REPO / "output" / "loop_research" / "insider_edge_paired.json"
OUT_PNG = REPO / "output" / "loop_research" / "insider_edge_paired.png"

DRAW_SIZE = 10
TOP_K = 5
N_SIMS = 500

# dataviz validated categorical palette
BLUE = "#2a78d6"
MAGENTA = "#e87ba4"
GRAY = "#8a8f98"


def _draw_pool(scores: pd.Series) -> list[str]:
    """decile-1 (top 10% by rank) minus the composite==100 tied names."""
    decile1 = set(_decile_pool(scores, 1))
    at_100 = set(scores[scores == 100].index)
    return sorted(decile1 - at_100)


def _split_by_insider(pool: list[str], parent_scores, rng) -> tuple[list[str], list[str]]:
    """Draw DRAW_SIZE random names from pool, rank by insider score, split
    top-TOP_K / bottom-(rest). Both legs must call rng identically before
    this point so paired sims stay in lockstep."""
    draw = rng.sample(pool, min(DRAW_SIZE, len(pool))) if pool else []
    if not draw:
        return [], []
    ins = parent_scores.get("insider", pd.Series(dtype=float)).reindex(draw)
    ranked = ins.sort_index().sort_values(ascending=False, kind="stable")
    top = list(ranked.index[:TOP_K])
    bottom = list(ranked.index[TOP_K:])
    return top, bottom


def _paired_selector(leg: str):
    def selector(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
        pool = _draw_pool(scores)
        top, bottom = _split_by_insider(pool, parent_scores, rng)
        picks = top if leg == "insider" else bottom
        if len(picks) >= 2:
            return picks
        return held if held else picks
    selector.__name__ = f"paired_{leg}"
    return selector


def _stats(pr, spy):
    m = performance_metrics(pr, hold_months=1, benchmarks={"SPY": spy})
    return {"cagr": m["cagr"], "sharpe": m["sharpe"], "sortino": m["sortino"],
            "max_dd": m["max_drawdown"], "beta": m.get("spy_beta"),
            "alpha": m.get("spy_alpha")}


def run(bundle, n_sims=N_SIMS, seed_base=0):
    data = bundle["data"]
    rebal = list(data.rebal_dates)
    all_dates = rebal[:-1]
    spy = bench_returns(data.matrix, SPY, rebal)

    insider_sel = _paired_selector("insider")
    other_sel = _paired_selector("other")

    curves_ins = np.zeros((n_sims, len(all_dates)))
    curves_oth = np.zeros((n_sims, len(all_dates)))
    ret_diff = np.zeros((n_sims, len(all_dates)))
    stats_ins = {"cagr": [], "sharpe": [], "sortino": [], "max_dd": [], "beta": [], "alpha": []}
    stats_oth = {"cagr": [], "sharpe": [], "sortino": [], "max_dd": [], "beta": [], "alpha": []}
    diffs = {"cagr": [], "sharpe": [], "alpha": []}

    t0 = time.time()
    for s in range(n_sims):
        seed = seed_base + s
        pr_i, to_i, tg_i = H.run_one(bundle, insider_sel, seed, k=TOP_K, refresh_n=TOP_K)
        pr_o, to_o, tg_o = H.run_one(bundle, other_sel, seed, k=TOP_K, refresh_n=TOP_K)

        st_i, st_o = _stats(pr_i, spy), _stats(pr_o, spy)
        for kk in stats_ins:
            stats_ins[kk].append(st_i[kk])
            stats_oth[kk].append(st_o[kk])
        for kk in diffs:
            diffs[kk].append(st_i[kk] - st_o[kk])

        pr_i_r = pr_i.reindex(all_dates).fillna(0.0)
        pr_o_r = pr_o.reindex(all_dates).fillna(0.0)
        curves_ins[s] = (1.0 + pr_i_r).cumprod().values
        curves_oth[s] = (1.0 + pr_o_r).cumprod().values
        ret_diff[s] = (pr_i_r - pr_o_r).values

        if (s + 1) % 50 == 0:
            print(f"  sim {s+1}/{n_sims}  ({time.time()-t0:.0f}s elapsed)", flush=True)

    period_mean_diff = ret_diff.mean(axis=0)   # sim-noise averaged out, per calendar date
    period_std_diff = ret_diff.std(axis=0, ddof=1)

    spy_r = spy.reindex(all_dates).fillna(0.0)

    def _curve_summary(curves):
        return {
            "median_curve": np.median(curves, axis=0).tolist(),
            "p10_curve": np.percentile(curves, 10, axis=0).tolist(),
            "p90_curve": np.percentile(curves, 90, axis=0).tolist(),
        }

    result = {
        "dates": [str(d) for d in all_dates],
        "n_sims": n_sims,
        "draw_size": DRAW_SIZE,
        "top_k": TOP_K,
        "spy_curve": (1.0 + spy_r).cumprod().values.tolist(),
        "insider": _curve_summary(curves_ins) | {
            "stats_median": {kk: float(np.median(v)) for kk, v in stats_ins.items()}},
        "other": _curve_summary(curves_oth) | {
            "stats_median": {kk: float(np.median(v)) for kk, v in stats_oth.items()}},
        "diffs": diffs,
        "period_mean_diff": period_mean_diff.tolist(),
        "period_std_diff": period_std_diff.tolist(),
        "summary": {
            f"median_diff_{kk}": float(np.median(v)) for kk, v in diffs.items()
        } | {
            f"win_rate_{kk}": float(np.mean(np.array(v) > 0)) for kk, v in diffs.items()
        } | {
            f"mean_diff_{kk}": float(np.mean(v)) for kk, v in diffs.items()
        },
    }
    return result


def _plot(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    sharpe_diffs = result["diffs"]["sharpe"]
    ax = axes[0]
    ax.hist(sharpe_diffs, bins=30, color=BLUE, alpha=0.85)
    ax.axvline(0.0, color=GRAY, linestyle="--", linewidth=1.5, label="no edge (0)")
    med = result["summary"]["median_diff_sharpe"]
    ax.axvline(med, color=MAGENTA, linewidth=2, label=f"median = {med:+.3f}")
    ax.set_xlabel("per-sim Sharpe diff (insider-5 minus other-5)")
    ax.set_ylabel("count of sims")
    ax.set_title("Paired Sharpe diff, insider vs other-5")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    dates = result["dates"]
    ax2.plot(dates, np.array(result["insider"]["median_curve"]), color=MAGENTA,
             linewidth=2, label="insider-5 (median)")
    ax2.plot(dates, np.array(result["other"]["median_curve"]), color=BLUE,
             linewidth=2, label="other-5 (median)")
    ax2.plot(dates, np.array(result["spy_curve"]), color=GRAY, linestyle="--",
             linewidth=1.25, label="SPY")
    ax2.set_xticks(ax2.get_xticks()[::max(1, len(dates)//6)])
    ax2.set_ylabel("growth of $1")
    ax2.set_title("Median growth curves (same draws)")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140)
    print(f"wrote {OUT_PNG}")


OUT_PNG_PERIOD = REPO / "output" / "loop_research" / "insider_edge_paired_period.png"


def _period_breakdown(result):
    """The 500-sim t-test only resamples WHICH random names get compared --
    every sim replays the identical 2020-2026 return sequence, so its
    effective independent sample size is nowhere near 500. This averages the
    per-sim return diff ACROSS sims first (killing the selection-draw noise),
    leaving one monthly time series of the insider-vs-other return
    differential -- then treats CALENDAR MONTHS, not sims, as the unit of
    independence. Answers: is the edge broad-based across the ~7-year
    history, or carried by a couple of lucky stretches?"""
    from scipy import stats

    dates = pd.to_datetime(result["dates"])
    period_diff = np.array(result["period_mean_diff"])
    years = dates.year

    by_year = pd.Series(period_diff, index=years).groupby(level=0)
    year_sum = by_year.sum()
    year_mean = by_year.mean()
    year_pct_pos = by_year.apply(lambda s: float((s > 0).mean()))
    n_months = by_year.size()

    t_stat, p_val = stats.ttest_1samp(period_diff, popmean=0.0)
    pct_months_positive = float((period_diff > 0).mean())
    n_years_positive = int((year_sum > 0).sum())
    n_years_total = len(year_sum)

    return {
        "dates": result["dates"],
        "period_mean_diff": period_diff.tolist(),
        "year_sum": year_sum.to_dict(),
        "year_mean": year_mean.to_dict(),
        "year_pct_months_positive": year_pct_pos.to_dict(),
        "n_months_per_year": n_months.to_dict(),
        "t_stat_months": float(t_stat),
        "p_value_months": float(p_val),
        "n_months": len(period_diff),
        "pct_months_positive": pct_months_positive,
        "n_years_positive": n_years_positive,
        "n_years_total": n_years_total,
    }


def _plot_period(period):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    years = sorted(period["year_sum"].keys())
    sums = [period["year_sum"][y] for y in years]
    colors = [MAGENTA if v > 0 else BLUE for v in sums]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))

    ax = axes[0]
    ax.bar([str(y) for y in years], sums, color=colors)
    ax.axhline(0.0, color=GRAY, linestyle="--", linewidth=1.25)
    ax.set_ylabel("sum of monthly (insider - other) return diff")
    ax.set_title(f"Per-year edge  ({period['n_years_positive']}/{period['n_years_total']} years positive)")

    ax2 = axes[1]
    dates = pd.to_datetime(period["dates"])
    cum = np.cumsum(period["period_mean_diff"])
    ax2.plot(dates, cum, color=MAGENTA, linewidth=2)
    ax2.axhline(0.0, color=GRAY, linestyle="--", linewidth=1.25)
    ax2.set_ylabel("cumulative sum of monthly diff")
    ax2.set_title("Cumulative edge over time")
    ax2.text(0.03, 0.95,
             f"one-sample t-test across {period['n_months']} months\n"
             f"t = {period['t_stat_months']:.2f}, p = {period['p_value_months']:.4f}\n"
             f"{period['pct_months_positive']:.0%} of months positive",
             transform=ax2.transAxes, ha="left", va="top", fontsize=8.5,
             bbox=dict(boxstyle="round", fc="white", ec=GRAY, alpha=0.9))

    fig.suptitle("Per-period breakdown: is the edge broad-based across calendar time?", fontsize=12)
    fig.tight_layout()
    OUT_PNG_PERIOD.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG_PERIOD, dpi=140)
    print(f"wrote {OUT_PNG_PERIOD}")


OUT_PNG_CONVERGENCE = REPO / "output" / "loop_research" / "insider_edge_paired_convergence.png"

METRIC_LABEL = {"cagr": "CAGR diff", "sharpe": "Sharpe diff", "alpha": "Alpha diff"}


def _plot_convergence(result):
    """Running mean of the per-sim (insider - other) diff as sims accumulate,
    with a shrinking 95% CI band (mean +/- 1.96*SE, SE = std/sqrt(n)) -- shows
    whether the mean diff is a stable, bounded-away-from-zero estimate or
    just noise that hasn't averaged out yet."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    metrics = ["cagr", "sharpe", "alpha"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, metric in zip(axes, metrics):
        diffs = np.array(result["diffs"][metric])
        n = len(diffs)
        idx = np.arange(1, n + 1)
        running_mean = np.cumsum(diffs) / idx
        running_std = np.array([diffs[:i].std(ddof=1) if i > 1 else 0.0 for i in idx])
        se = running_std / np.sqrt(idx)
        lo = running_mean - 1.96 * se
        hi = running_mean + 1.96 * se

        ax.fill_between(idx, lo, hi, color=BLUE, alpha=0.20, label="95% CI (mean +/- 1.96 SE)")
        ax.plot(idx, running_mean, color=MAGENTA, linewidth=2, label="running mean diff")
        ax.axhline(0.0, color=GRAY, linestyle="--", linewidth=1.25, label="no edge (0)")
        ax.set_xlabel("sims included")
        ax.set_title(METRIC_LABEL[metric])

        t_stat, p_val = stats.ttest_1samp(diffs, popmean=0.0)
        ax.text(0.97, 0.04, f"one-sample t-test vs 0\nmean = {running_mean[-1]:+.4f}\np = {p_val:.4f}",
               transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
               bbox=dict(boxstyle="round", fc="white", ec=GRAY, alpha=0.9))

        if ax is axes[0]:
            ax.set_ylabel("insider-5 minus other-5")
            ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Does the mean insider-edge diff converge away from zero?", fontsize=12)
    fig.tight_layout()
    OUT_PNG_CONVERGENCE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG_CONVERGENCE, dpi=140)
    print(f"wrote {OUT_PNG_CONVERGENCE}")


def main():
    bundle = H.get_data()
    print("running paired insider-edge Monte Carlo ...", flush=True)
    result = run(bundle)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as fh:
        json.dump(result, fh, indent=2)
    print(f"wrote {OUT_JSON}")

    _plot(result)
    _plot_convergence(result)

    period = _period_breakdown(result)
    with (REPO / "output" / "loop_research" / "insider_edge_paired_period.json").open("w") as fh:
        json.dump(period, fh, indent=2)
    _plot_period(period)

    s = result["summary"]
    print("\n=== paired insider-edge summary (n_sims={}) ===".format(result["n_sims"]))
    print(f"  median Sharpe diff = {s['median_diff_sharpe']:+.3f}  "
         f"(win rate {s['win_rate_sharpe']:.1%})")
    print(f"  median CAGR   diff = {s['median_diff_cagr']:+.3f}  "
         f"(win rate {s['win_rate_cagr']:.1%})")
    print(f"  median alpha  diff = {s['median_diff_alpha']:+.3f}  "
         f"(win rate {s['win_rate_alpha']:.1%})")
    print(f"  insider stats (median): {result['insider']['stats_median']}")
    print(f"  other   stats (median): {result['other']['stats_median']}")
    print(f"\n=== per-period breakdown ({period['n_months']} calendar months) ===")
    print(f"  t = {period['t_stat_months']:.2f}, p = {period['p_value_months']:.4f}, "
         f"{period['pct_months_positive']:.0%} of months positive, "
         f"{period['n_years_positive']}/{period['n_years_total']} years net positive")
    print(f"  per-year sums: {period['year_sum']}")


if __name__ == "__main__":
    main()
