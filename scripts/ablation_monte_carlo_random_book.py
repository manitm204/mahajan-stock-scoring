"""Ablation-over-Monte-Carlo grid search: does the book4/hold4/evict3 FAMILY's
configuration (book size N, hold length, refresh fraction) matter at all, or
is everything within noise of a random draw from the composite_score==100
pool? Follow-up to scripts/monte_carlo_random_book.py (which fixed N=4,
HOLD_MONTHS=4, REFRESH_N=3 and only randomized selection) -- this sweeps the
structural knobs themselves, each cell evaluated by its own Monte Carlo
distribution rather than one deterministic backtest.

For every (N, HOLD_MONTHS, REFRESH_FRACTION) combination on the grid (with
SLEEVE_COUNT fixed at 3, same staggered-sleeve structure as the production
autoresearch family): run --sims independent random replays (uniform draw
from that date's score==100 pool at every fill/refill), compute per-sim
metrics (CAGR, Sharpe, beta/alpha/IR vs SPY, IR vs QQQ, max_dd, alpha
t-stat), then aggregate mean/std/p10/p90 per config.

Also reports a variance decomposition (eta-squared: between-config SS /
total SS) for Sharpe and SPY-IR -- the direct answer to "does configuration
matter, or are they mostly the same": a low eta-squared means cell-to-cell
differences are dominated by which random names got picked, not by N/hold/
refresh.

CAVEAT: scanning many configs and reporting the best mean IR is itself a
search -- multiple-comparisons inflation applies. Treat the "top configs"
table as "best in this grid," not a promotion candidate for candidate.py.

Usage: python scripts/ablation_monte_carlo_random_book.py [--sims 100]
Writes: output/ablation_monte_carlo_random_book/results.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backtesting.data_loader import SPY, QQQ                        # noqa: E402
from research.ablation.engine import alpha_tstat                     # noqa: E402
from research.autoresearch.evaluate import (                         # noqa: E402
    compute_portfolio_returns, targets_to_weight_matrix,
)
from research.strategies.engine import load_data                     # noqa: E402
from research.walkforward.portfolio import performance_metrics       # noqa: E402
from scripts.run_strategy_sweep import bench_returns                 # noqa: E402

OUT = REPO / "output" / "ablation_monte_carlo_random_book" / "results.json"

SLEEVE_COUNT = 3
N_GRID = [3, 4, 6, 8, 11, 15]
HOLD_MONTHS_GRID = [1, 2, 4, 6, 12]
REFRESH_FRACTION_GRID = [0.25, 0.5, 1.0]


def _random_book(comp_date_scores: pd.Series, held: list, k: int,
                 refresh_n: int, rng: random.Random) -> list:
    pool = list(comp_date_scores[comp_date_scores == 100].index)
    if not pool:
        return held
    if not held:
        rng.shuffle(pool)
        return pool[:k]
    keep = list(held)
    n_evict = min(refresh_n, len(keep))
    to_evict = set(rng.sample(keep, n_evict))
    keep = [t for t in keep if t not in to_evict]
    need = k - len(keep)
    candidates = [t for t in pool if t not in keep]
    rng.shuffle(candidates)
    fill = candidates[:need]
    if len(fill) < need:
        universe = list(comp_date_scores.dropna().index)
        rng.shuffle(universe)
        extra = [t for t in universe if t not in keep and t not in fill]
        fill += extra[:need - len(fill)]
    return keep + fill


def _random_targets(comp: dict, dates: list, rng: random.Random,
                    n: int, hold_months: int, refresh_n: int) -> pd.DataFrame:
    step = max(hold_months // SLEEVE_COUNT, 1)
    sleeve_holdings = [[] for _ in range(SLEEVE_COUNT)]
    weight_per_name = (1.0 / SLEEVE_COUNT) / n
    rows = []
    for i, d in enumerate(dates):
        for j in range(SLEEVE_COUNT):
            offset = j * step
            if i >= offset and (i - offset) % hold_months == 0:
                scores = comp.get(d)
                if scores is not None:
                    sleeve_holdings[j] = _random_book(
                        scores, sleeve_holdings[j], n, refresh_n, rng)
        agg = {}
        for holdings in sleeve_holdings:
            for t in holdings:
                agg[t] = agg.get(t, 0.0) + weight_per_name
        for t, w in agg.items():
            rows.append({"date": d, "ticker": t, "weight": w})
    return pd.DataFrame(rows, columns=["date", "ticker", "weight"])


def _sim_metrics(pr: pd.Series, spy: pd.Series, qqq: pd.Series) -> dict:
    m = performance_metrics(pr, hold_months=1, benchmarks={"SPY": spy, "QQQ": qqq})
    return {
        "cagr": m["cagr"],
        "sharpe": m["sharpe"],
        "max_dd": m["max_drawdown"],
        "spy_beta": m.get("spy_beta"),
        "spy_alpha": m.get("spy_alpha"),
        "spy_ir": m.get("spy_ir"),
        "qqq_ir": m.get("qqq_ir"),
        "alpha_tstat": alpha_tstat(pr, spy),
    }


def _agg(values: list) -> dict:
    a = np.array([v for v in values if v == v], dtype=float)  # drop NaN
    if a.size == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "p10": float("nan"), "p90": float("nan")}
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90))}


def _eta_squared(per_config_values: list[list[float]]) -> float:
    """Between-config SS / total SS across all (config, sim) pairs -- the
    fraction of total variance attributable to which config a sim belongs
    to, vs. pure within-config Monte Carlo noise."""
    all_vals = np.concatenate([np.array(v, dtype=float) for v in per_config_values])
    all_vals = all_vals[all_vals == all_vals]
    grand_mean = all_vals.mean()
    ss_total = float(((all_vals - grand_mean) ** 2).sum())
    if ss_total == 0:
        return float("nan")
    ss_between = 0.0
    for v in per_config_values:
        a = np.array(v, dtype=float)
        a = a[a == a]
        if a.size == 0:
            continue
        ss_between += a.size * (a.mean() - grand_mean) ** 2
    return ss_between / ss_total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("loading composite scores + price matrix ...", flush=True)
    data = load_data()
    rebal = list(data.rebal_dates)
    spy = bench_returns(data.matrix, SPY, rebal)
    qqq = bench_returns(data.matrix, QQQ, rebal)

    configs = []
    for n, hold, frac in product(N_GRID, HOLD_MONTHS_GRID, REFRESH_FRACTION_GRID):
        refresh_n = int(round(n * frac))
        refresh_n = max(1, min(n, refresh_n))
        configs.append({"n": n, "hold_months": hold, "refresh_fraction": frac,
                        "refresh_n": refresh_n})

    print(f"{len(configs)} configs x {args.sims} sims = "
         f"{len(configs) * args.sims} total simulations", flush=True)

    metric_names = ["cagr", "sharpe", "max_dd", "spy_beta", "spy_alpha",
                    "spy_ir", "qqq_ir", "alpha_tstat"]
    results = []
    per_config_sharpe, per_config_spy_ir = [], []
    t_start = time.time()

    for ci, cfg in enumerate(configs):
        per_sim = {m: [] for m in metric_names}
        for s in range(args.sims):
            rng = random.Random(args.seed * 1_000_003 + ci * 1009 + s)
            targets = _random_targets(data.comp, rebal, rng,
                                      cfg["n"], cfg["hold_months"], cfg["refresh_n"])
            weights = targets_to_weight_matrix(targets, rebal)
            pr, _turnover = compute_portfolio_returns(weights, data.matrix, rebal)
            sm = _sim_metrics(pr, spy, qqq)
            for m in metric_names:
                per_sim[m].append(sm[m])

        agg = {m: _agg(per_sim[m]) for m in metric_names}
        per_config_sharpe.append(per_sim["sharpe"])
        per_config_spy_ir.append(per_sim["spy_ir"])
        results.append({"config": cfg, "metrics": agg})

        if (ci + 1) % 10 == 0 or ci == len(configs) - 1:
            elapsed = time.time() - t_start
            rate = (ci + 1) / elapsed
            eta = (len(configs) - ci - 1) / rate if rate > 0 else float("nan")
            print(f"  config {ci + 1}/{len(configs)}  "
                 f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)", flush=True)

    eta2_sharpe = _eta_squared(per_config_sharpe)
    eta2_spy_ir = _eta_squared(per_config_spy_ir)

    ranked = sorted(results, key=lambda r: r["metrics"]["spy_ir"]["mean"]
                    if r["metrics"]["spy_ir"]["mean"] == r["metrics"]["spy_ir"]["mean"]
                    else -1e9, reverse=True)

    print("\n=== variance decomposition (does configuration matter?) ===")
    print(f"  eta^2(sharpe, config)  = {eta2_sharpe:.3f}  "
         f"({eta2_sharpe * 100:.1f}% of Sharpe variance explained by config choice)")
    print(f"  eta^2(spy_ir, config)  = {eta2_spy_ir:.3f}  "
         f"({eta2_spy_ir * 100:.1f}% of SPY-IR variance explained by config choice)")
    print("\n=== top 5 configs by mean SPY-IR ===")
    for r in ranked[:5]:
        c, m = r["config"], r["metrics"]
        print(f"  N={c['n']:<3} hold={c['hold_months']:<3} refresh={c['refresh_n']}/{c['n']} "
             f"({c['refresh_fraction']:.0%})  "
             f"spy_ir={m['spy_ir']['mean']:.3f}+/-{m['spy_ir']['std']:.3f}  "
             f"sharpe={m['sharpe']['mean']:.3f}  cagr={m['cagr']['mean']:.3f}")
    print("\n=== bottom 5 configs by mean SPY-IR ===")
    for r in ranked[-5:]:
        c, m = r["config"], r["metrics"]
        print(f"  N={c['n']:<3} hold={c['hold_months']:<3} refresh={c['refresh_n']}/{c['n']} "
             f"({c['refresh_fraction']:.0%})  "
             f"spy_ir={m['spy_ir']['mean']:.3f}+/-{m['spy_ir']['std']:.3f}  "
             f"sharpe={m['sharpe']['mean']:.3f}  cagr={m['cagr']['mean']:.3f}")

    out = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"sleeve_count": SLEEVE_COUNT, "n_grid": N_GRID,
                   "hold_months_grid": HOLD_MONTHS_GRID,
                   "refresh_fraction_grid": REFRESH_FRACTION_GRID,
                   "sims_per_config": args.sims, "seed": args.seed,
                   "selection": "uniform random draw from composite_score==100 pool"},
        "variance_decomposition": {"eta_squared_sharpe": eta2_sharpe,
                                   "eta_squared_spy_ir": eta2_spy_ir},
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        json.dump(out, fh, indent=2, default=lambda x: None if x != x else x)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
