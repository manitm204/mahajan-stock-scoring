"""Autonomous research-loop harness for the staggered-sleeves stock-selection
problem (user request 2026-09-09): baseline = 5-stock / 4-month hold / 4
sleeves / random draw from the composite_score==100 pool
(scripts/monte_carlo_random_book.py's featured config, post bench_returns
fix -- see docs/monte_carlo_book_construction.md).

Immutable in this module: data loading, sleeve bookkeeping/staggering, T+1
execution + 10bps costs (both inherited from research.autoresearch.evaluate),
Monte Carlo seeding, and metric definitions. The ONLY thing that varies
between experiments is the `selector` function passed in -- signature:

    selector(scores, parent_scores, held, k, refresh_n, rng) -> list[str]

`scores` is that date's composite score Series, `parent_scores` is
{parent_name: Series} for that date (same date, from the production V4
parent-selection panel), `held` is the sleeve's current holdings, `rng` a
random.Random seeded per-sim (deterministic selectors should ignore it).
"""
from __future__ import annotations

import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backtesting.data_loader import SPY, QQQ                         # noqa: E402
from research.autoresearch.evaluate import (                          # noqa: E402
    bench_returns, compute_portfolio_returns, targets_to_weight_matrix,
)
from research.strategies.engine import load_data                      # noqa: E402
from research.walkforward.portfolio import performance_metrics        # noqa: E402

N = 5
HOLD_MONTHS = 4
SLEEVE_COUNT = 4
REFRESH_N = 5          # full turnover each review (matches the featured config)
STEP = max(HOLD_MONTHS // SLEEVE_COUNT, 1)

CACHE = REPO / "output" / "loop_research" / "data_cache.pkl"


def get_data():
    """Cached load_data() + per-parent score panels (8 V4 parent families).
    load_data()/DB access is the slow part (~tens of seconds); everything in
    this loop re-touches it every iteration, so cache to disk once."""
    if CACHE.exists():
        with CACHE.open("rb") as fh:
            bundle = pickle.load(fh)
    else:
        from factors.parent_selection_v4 import SELECTED_SUBS
        from scripts.crowding_diagnostics import parent_score
        from research.subfactor_expansion.panel import load_cached_panel
        from research.strategies.engine import PANEL_PKL

        data = load_data()
        sub_panel = load_cached_panel(PANEL_PKL)
        parent_scores = {}   # {date: {parent_name: Series}}
        for d in data.rebal_dates:
            if d not in sub_panel.scores:
                continue
            frame = sub_panel.scores[d]
            parent_scores[d] = {p: parent_score(frame, w) for p, w in SELECTED_SUBS.items()
                                if any(c in frame.columns for c in w)}

        bundle = {"data": data, "parent_scores": parent_scores}
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with CACHE.open("wb") as fh:
            pickle.dump(bundle, fh)

    bundle["subfactor_frames"] = get_subfactor_frames()
    return bundle


SUBFACTOR_CACHE = REPO / "output" / "loop_research" / "subfactor_frames_cache.pkl"


def get_subfactor_frames():
    """Cached {date: frame} of raw per-subfactor scores (24 subs across the 8
    V4 parents, SELECTED_SUBS-defined, already sector-percentile-scored and
    correctly signed higher=better) -- for selectors that want to rank by a
    property of the individual subfactors rather than the parent aggregates
    already in get_data()'s `parent_scores`. Separate cache file since it's
    a different (larger) payload than the parent-score bundle."""
    if SUBFACTOR_CACHE.exists():
        with SUBFACTOR_CACHE.open("rb") as fh:
            return pickle.load(fh)
    from research.subfactor_expansion.panel import load_cached_panel
    from research.strategies.engine import PANEL_PKL

    data = load_data()
    sub_panel = load_cached_panel(PANEL_PKL)
    frames = {d: sub_panel.scores[d] for d in data.rebal_dates if d in sub_panel.scores}
    SUBFACTOR_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with SUBFACTOR_CACHE.open("wb") as fh:
        pickle.dump(frames, fh)
    return frames


def random_selector(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
    """Baseline: uniform random draw from the score==100 pool."""
    pool = list(scores[scores == 100].index)
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
        universe = list(scores.dropna().index)
        rng.shuffle(universe)
        extra = [t for t in universe if t not in keep and t not in fill]
        fill += extra[:need - len(fill)]
    return keep + fill


TOP_N_POOL = 20


def random_top20_selector(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
    """New baseline (2026-09-11, user request): instead of the composite==100
    tied pool (~10-15 names, all rank-tied at the very top), take the top 20
    names by composite score rank each date (ties broken by ticker for a
    stable pool), then draw a uniform random k of them. Same refresh/fill
    mechanics as random_selector."""
    ranked = scores.dropna().sort_values(ascending=False, kind="stable")
    pool = list(ranked.index[:TOP_N_POOL])
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
        universe = list(scores.dropna().index)
        rng.shuffle(universe)
        extra = [t for t in universe if t not in keep and t not in fill]
        fill += extra[:need - len(fill)]
    return keep + fill


def _sleeve_targets(bundle, selector, k, hold_months, sleeve_count, refresh_n, rng):
    data = bundle["data"]
    parent_scores_by_date = bundle["parent_scores"]
    dates = list(data.rebal_dates)
    step = max(hold_months // sleeve_count, 1)
    sleeve_holdings = [[] for _ in range(sleeve_count)]
    weight_per_name = (1.0 / sleeve_count) / k
    rows = []
    for i, d in enumerate(dates):
        for j in range(sleeve_count):
            offset = j * step
            if i >= offset and (i - offset) % hold_months == 0:
                scores = data.comp.get(d)
                pscores = parent_scores_by_date.get(d, {})
                ctx = {"date": d, "dates": dates, "i": i, "bundle": bundle}
                if scores is not None:
                    sleeve_holdings[j] = selector(
                        scores, pscores, sleeve_holdings[j], k, refresh_n, rng, ctx=ctx)
        agg = {}
        for holdings in sleeve_holdings:
            for t in holdings:
                agg[t] = agg.get(t, 0.0) + weight_per_name
        for t, w in agg.items():
            rows.append({"date": d, "ticker": t, "weight": w})
    return pd.DataFrame(rows, columns=["date", "ticker", "weight"])


def run_one(bundle, selector, seed, k=N, hold_months=HOLD_MONTHS,
           sleeve_count=SLEEVE_COUNT, refresh_n=REFRESH_N):
    data = bundle["data"]
    rebal = list(data.rebal_dates)
    rng = random.Random(seed)
    targets = _sleeve_targets(bundle, selector, k, hold_months, sleeve_count, refresh_n, rng)
    weights = targets_to_weight_matrix(targets, rebal)
    pr, turnover = compute_portfolio_returns(weights, data.matrix, rebal)
    return pr, turnover, targets


def bench_series(bundle):
    data = bundle["data"]
    rebal = list(data.rebal_dates)
    spy = bench_returns(data.matrix, SPY, rebal)
    qqq = bench_returns(data.matrix, QQQ, rebal)
    return spy, qqq


def sim_metrics(pr, turnover, targets, spy, qqq):
    m = performance_metrics(pr, hold_months=1, turnover=turnover,
                            benchmarks={"SPY": spy, "QQQ": qqq})
    unique_holdings = targets["ticker"].nunique()
    return {
        "cagr": m["cagr"], "sharpe": m["sharpe"], "sortino": m["sortino"],
        "max_dd": m["max_drawdown"], "avg_turnover": m["avg_turnover"],
        "spy_beta": m.get("spy_beta"), "spy_alpha": m.get("spy_alpha"),
        "spy_ir": m.get("spy_ir"), "qqq_ir": m.get("qqq_ir"),
        "unique_holdings": int(unique_holdings),
        "total_return": m["total_return"],
    }


def run_monte_carlo(bundle, selector, n_sims, seed_base=0, k=N, hold_months=HOLD_MONTHS,
                    sleeve_count=SLEEVE_COUNT, refresh_n=REFRESH_N, deterministic=False):
    """Returns list of per-sim metric dicts. If deterministic=True, runs the
    selector once (seed 0) and repeats its metrics n_sims times (so it can
    still be compared seed-for-seed against a random baseline's distribution
    without pretending it has its own variance)."""
    spy, qqq = bench_series(bundle)
    n_run = 1 if deterministic else n_sims
    out = []
    for s in range(n_run):
        pr, turnover, targets = run_one(bundle, selector, seed_base + s, k, hold_months,
                                        sleeve_count, refresh_n)
        out.append(sim_metrics(pr, turnover, targets, spy, qqq))
    if deterministic:
        out = out * n_sims
    return out
