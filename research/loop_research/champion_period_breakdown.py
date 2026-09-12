"""Same rigor applied to the actual promoted champion (user request
2026-09-10): exp5 / top5_by_insider (research/loop_research/candidates.py),
which fires within the composite==100 tied pool -- NOT the broader
decile-1-minus-100 pool that insider_edge_paired.py just showed looks like a
2020-2021 regime artifact rather than a real signal.

exp5's original evidence was a 500-sim Monte Carlo win rate (98.4%) against
a random baseline plus a looser "remove best year" robustness check -- not
the per-period-across-calendar-time + IC-by-year rigor just used to debunk
the broader-pool finding. This applies the identical method here:

1. Run the deterministic top5_by_insider selector ONCE (its only real path).
2. Run the random baseline N_SIMS times, average across sims PER CALENDAR
   DATE (kills the baseline's own resampling noise, same trick as before).
3. diff_by_date = insider_path_return(date) - mean_random_return(date).
   Aggregate to per-year sums, t-test across the real calendar months, and
   check hit rate -- is the champion's edge broad-based across 2020-2026,
   or concentrated the same way the broader-pool test was?
4. Pull insider parent-score IC by year, restricted to the composite==100
   pool (mirrors insider_ic_by_year.py's edge_pool column but with the
   champion's actual selection pool).

Usage: python -m research.loop_research.champion_period_breakdown
Writes: output/loop_research/champion_period_breakdown.csv
        output/loop_research/champion_period_breakdown.png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backtesting.data_loader import SPY                                    # noqa: E402
from research.autoresearch.evaluate import bench_returns                    # noqa: E402
from research.forward_returns import compute_forward_returns                # noqa: E402
from research.ic import period_ic                                           # noqa: E402
from research.loop_research import harness as H                             # noqa: E402
from research.loop_research.candidates import top5_by_insider               # noqa: E402
from scipy import stats                                                     # noqa: E402

OUT_DIR = REPO / "output" / "loop_research"
N_SIMS = 500
BLUE = "#2a78d6"
MAGENTA = "#e87ba4"
GRAY = "#8a8f98"


def run():
    bundle = H.get_data()
    data = bundle["data"]
    parent_scores_by_date = bundle["parent_scores"]
    rebal = list(data.rebal_dates)
    all_dates = rebal[:-1]

    print("running champion path (deterministic) ...", flush=True)
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

    mean_random_by_date = rand_returns.mean(axis=0)
    diff_by_date = pr_champ_r.values - mean_random_by_date

    dates = pd.to_datetime(all_dates)
    years = dates.year
    diff_series = pd.Series(diff_by_date, index=years)
    year_sum = diff_series.groupby(level=0).sum()
    t_stat, p_val = stats.ttest_1samp(diff_by_date, popmean=0.0)
    pct_months_positive = float((diff_by_date > 0).mean())

    print("\ncomputing insider IC by year (composite==100 pool) ...", flush=True)
    fwd_1m = compute_forward_returns(data.matrix, rebal, {"1M": 1})["1M"]
    ic_rows = []
    for d in rebal:
        if d not in fwd_1m or d not in parent_scores_by_date:
            continue
        ins = parent_scores_by_date[d].get("insider")
        scores_d = data.comp.get(d)
        if ins is None or scores_d is None:
            continue
        pool = list(scores_d[scores_d == 100].index)
        if not pool:
            continue
        ic = period_ic(ins.reindex(pool), fwd_1m[d].reindex(pool), min_names=8)
        ic_rows.append({"date": d, "year": pd.Timestamp(d).year, "ic_100pool": ic})
    ic_df = pd.DataFrame(ic_rows)

    table_rows = []
    for y in sorted(year_sum.index):
        ic_y = ic_df[ic_df["year"] == y]["ic_100pool"].dropna()
        trail = ic_df[(ic_df["year"] > y - 5) & (ic_df["year"] <= y)]["ic_100pool"].dropna()
        table_rows.append({
            "year": int(y),
            "champion_vs_random_sum": float(year_sum.loc[y]),
            "ic_100pool_that_year": float(ic_y.mean()) if len(ic_y) else float("nan"),
            "trailing5y_ic_100pool": float(trail.mean()) if len(trail) else float("nan"),
        })
    table = pd.DataFrame(table_rows)

    return {
        "table": table,
        "diff_by_date": diff_by_date,
        "dates": all_dates,
        "t_stat": float(t_stat),
        "p_val": float(p_val),
        "pct_months_positive": pct_months_positive,
        "n_months": len(diff_by_date),
        "spy": bench_returns(data.matrix, SPY, rebal).reindex(all_dates).fillna(0.0),
    }


def _plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = res["table"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))

    ax = axes[0]
    colors = [MAGENTA if v > 0 else BLUE for v in table["champion_vs_random_sum"]]
    ax.bar(table["year"].astype(str), table["champion_vs_random_sum"], color=colors)
    ax.axhline(0.0, color=GRAY, linestyle="--", linewidth=1.25)
    n_pos = int((table["champion_vs_random_sum"] > 0).sum())
    ax.set_ylabel("sum of monthly (champion - mean random) diff")
    ax.set_title(f"Champion per-year edge  ({n_pos}/{len(table)} years positive)")

    ax2 = axes[1]
    dates = pd.to_datetime(res["dates"])
    cum = np.cumsum(res["diff_by_date"])
    ax2.plot(dates, cum, color=MAGENTA, linewidth=2)
    ax2.axhline(0.0, color=GRAY, linestyle="--", linewidth=1.25)
    ax2.set_ylabel("cumulative sum of monthly diff")
    ax2.set_title("Cumulative champion edge over time")
    ax2.text(0.03, 0.95,
             f"one-sample t-test across {res['n_months']} months\n"
             f"t = {res['t_stat']:.2f}, p = {res['p_val']:.4f}\n"
             f"{res['pct_months_positive']:.0%} of months positive",
             transform=ax2.transAxes, ha="left", va="top", fontsize=8.5,
             bbox=dict(boxstyle="round", fc="white", ec=GRAY, alpha=0.9))

    fig.suptitle("exp5 (top5_by_insider, ==100 pool): edge vs random, by calendar time", fontsize=12)
    fig.tight_layout()
    out_png = OUT_DIR / "champion_period_breakdown.png"
    fig.savefig(out_png, dpi=140)
    print(f"wrote {out_png}")


def main():
    res = run()
    table = res["table"]
    table.to_csv(OUT_DIR / "champion_period_breakdown.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'champion_period_breakdown.csv'}")
    print(table.to_string(index=False))
    print(f"\noverall: t={res['t_stat']:.2f}  p={res['p_val']:.4f}  "
         f"{res['pct_months_positive']:.0%} of months positive")
    _plot(res)

    diff_df = pd.DataFrame({"date": res["dates"], "diff": res["diff_by_date"]})
    diff_df.to_csv(OUT_DIR / "champion_period_diff_series.csv", index=False)
    print(f"wrote {OUT_DIR / 'champion_period_diff_series.csv'}")


if __name__ == "__main__":
    main()
