"""Horizon experiment: re-select the composite on 6M/12M (vs the live 3M/6M) and
A/B both through the 13mo/4-sleeve after-tax sim for the locked construction
(top25 / cap5 / cap_match / VIX). OOS walk-forward, no look-ahead. Nothing touches
the production ablation cache; the 6M/12M run is pickled to its own path.
"""
from __future__ import annotations
import pickle, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

ROOT = "/home/manit/Desktop/fun_projects/mahajan_hedge_fund"
sys.path.insert(0, ROOT)
# reuse the already-written tax machinery
sys.path.insert(0, "/tmp/claude-1001/-home-manit-Desktop-fun-projects-mahajan-hedge-fund"
                   "/6f898f8f-ebb6-4fe9-a4bf-7fc00115578f/scratchpad")

import numpy as np
import pandas as pd

from backtesting import data_loader as dl
from data.db import get_db
from research.forward_returns import realize_delistings
from research.walkforward.runner import run_splits
from research.walkforward.splits import resolve_splits
from research.ablation.data import _pit_caps, AblationData
from run_walkforward import _load_panel, PANEL_START, PRICE_END

NEW_CACHE = Path(ROOT) / "cache" / "ablation_run_rolling5y_6m12m.pkl"
PROD_CACHE = Path(ROOT) / "cache" / "ablation_run_rolling5y.pkl"


def patch_horizons():
    """Flip the three horizon knobs to 6M/12M (sub pick, intra-parent wt, parent wt)."""
    import research.parent_selection as ps
    import research.walkforward.selection as sel
    from research.subset_selection import inventory, subfactor_performance
    from research.walkforward.compose import build_parent_panel

    ps.SELECT_HORIZONS = ["ic_6M", "ic_12M"]
    ps.STAT_HORIZONS = ("6M", "12M")
    sel.STAT_HORIZONS = ("6M", "12M")

    def _parent_scorecard_6m12m(sub_panel, sub_weights, fwd_by_h):
        parent_panel = build_parent_panel(sub_panel, sub_weights)
        perf = subfactor_performance(parent_panel, fwd_by_h, stat_horizons=("6M", "12M"))
        inv = inventory(parent_panel)
        perf["mean_ic_3m6m"] = perf[["ic_6M", "ic_12M"]].mean(axis=1)  # name kept for ic_ir_weights
        scored = (perf.drop(columns=["parent"], errors="ignore")
                      .rename(columns={"sub_factor": "parent"})
                      .merge(inv[["sub_factor", "coverage"]].rename(
                          columns={"sub_factor": "parent"}), on="parent", how="left"))
        return scored

    sel._parent_scorecard = _parent_scorecard_6m12m
    print("  [patch] horizons set to 6M/12M for sub pick + parent weights", flush=True)


def load_env():
    panel = _load_panel(False)
    with get_db() as db:
        matrix_raw = dl.load_price_matrix(db, panel.universe, PANEL_START, PRICE_END)
        sectors = dl.global_sectors(db)
        vixdf = db.query_df("SELECT date, close FROM daily_prices WHERE ticker='VIX' ORDER BY date")
    vix = pd.Series(vixdf["close"].values, index=vixdf["date"].astype(str))
    matrix = realize_delistings(matrix_raw)
    return panel, matrix_raw, matrix, sectors, vix


def make_data(run, panel, matrix, sectors, vix, db):
    rebal_dates = [d for d in sorted(run.pooled_scores) if d in matrix.index]
    caps = _pit_caps(db, matrix, rebal_dates)
    ann = matrix.pct_change(fill_method=None).rolling(252, min_periods=60).std() * np.sqrt(252)
    vol = ann.loc[rebal_dates]
    parent_ranks = {p: {d: s.rank(pct=True) * 100.0 for d, s in per_date.items()}
                    for p, per_date in run.parent_scores.items()}
    return AblationData(matrix=matrix, sectors=sectors, run=run, rebal_dates=rebal_dates,
                        caps=caps, vol=vol, vix=vix, parent_ranks=parent_ranks, panel=panel)


def selection_summary(run, tag):
    n = len(run.splits_data)
    sub_counts, parent_w = {}, {}
    for sd in run.splits_data:
        cfg = sd["config"]
        for p, wmap in cfg.sub_weights.items():
            for s, w in wmap.items():
                c = sub_counts.setdefault((p, s), [0, 0.0]); c[0] += 1; c[1] += w
        for p, w in cfg.parent_weights.items():
            parent_w[p] = parent_w.get(p, 0.0) + w
    print(f"\n=== {tag}: mean parent weights across {n} splits ===")
    for p, w in sorted(parent_w.items(), key=lambda kv: -kv[1]):
        print(f"  {p:14} {w/n:.3f}")
    print(f"=== {tag}: subs selected (freq/{n}, mean intra-parent wt) ===")
    for (p, s), (cnt, wsum) in sorted(sub_counts.items()):
        print(f"  {p:14} {s:34} {cnt}/{n}  w~{wsum/cnt:.2f}")


def run_ab():
    import final_stats as fs  # the tax machinery (weights_for, config_nav, bench_nav, stats, CFG)
    panel, matrix_raw, matrix, sectors, vix = load_env()

    # --- 6M/12M run (patched), cached separately ---
    if NEW_CACHE.exists():
        run_new = pickle.load(NEW_CACHE.open("rb"))
        print(f"[new] loaded 6M/12M run from {NEW_CACHE}", flush=True)
    else:
        patch_horizons()
        t0 = time.time()
        print("[new] walk-forward re-score on 6M/12M (~10-20 min)...", flush=True)
        run_new = run_splits(panel, resolve_splits("rolling5y"), matrix_raw, sectors, verbose=True)
        pickle.dump(run_new, NEW_CACHE.open("wb"))
        print(f"[new] done ({time.time()-t0:.0f}s), cached {NEW_CACHE}", flush=True)

    run_prod = pickle.load(PROD_CACHE.open("rb"))
    print(f"[locked] loaded 3M/6M run from {PROD_CACHE}", flush=True)

    selection_summary(run_prod, "LOCKED 3M/6M")
    selection_summary(run_new, "NEW 6M/12M")

    with get_db() as db:
        data_prod = make_data(run_prod, panel, matrix, sectors, vix, db)
        data_new = make_data(run_new, panel, matrix, sectors, vix, db)

    ERA = "2021-12-31"
    def report(data, label):
        wts = fs.weights_for(data, fs.CFG)
        dates = [d for d in data.rebal_dates if d in wts and d in matrix.index]
        nav = fs.config_nav(dates, wts, matrix)
        ds = np.array(dates)
        def window(nav, lo, hi):
            m = (ds >= lo) & (ds <= hi); j = np.where(m)[0]
            return nav[max(j[0]-1, 0): j[-1]+1]
        spans = [("FULL", dates[0], dates[-1]), ("ERA1", dates[0], ERA), ("ERA2", "2022-01-01", dates[-1])]
        print(f"\n### {label}  (after-tax, 13mo x4 sleeves, top25/cap5/cap_match/vix)")
        for nm, lo, hi in spans:
            m = fs.stats(window(nav, lo, hi))
            print(f"  {nm:5} CAGR {m['cagr']*100:5.1f}%  Sharpe {m['sharpe']:.2f}  "
                  f"Sortino {m['sortino']:.2f}  Calmar {m['calmar']:.2f}  maxDD {m['mdd']*100:5.1f}%")
        print(f"  walk-away ${nav[-1]:,.0f}")
        return dates

    report(data_prod, "LOCKED  (3M/6M selection)")
    dates = report(data_new, "NEW     (6M/12M selection)")
    # benchmarks once (same for both)
    from backtesting.data_loader import SPY
    for tkr in (SPY, "QQQ"):
        nav = fs.bench_nav(matrix, dates, tkr)
        m = fs.stats(nav)
        print(f"  {tkr:5} FULL CAGR {m['cagr']*100:5.1f}%  Sharpe {m['sharpe']:.2f}  "
              f"walk-away ${nav[-1]:,.0f}")


if __name__ == "__main__":
    run_ab()
