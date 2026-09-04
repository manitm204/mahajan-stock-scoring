"""Mini ablation: on the 6M/12M composite, sweep the loser-veto exclusion
(drop a stock if ANY factor score < 10), after-tax (13mo x4 sleeves,
top25/cap5/cap_match/VIX). Reference rows: locked 3M/6M with/without the veto,
SPY, QQQ. Nothing touches production; 6M/12M run cached to its own path.
"""
from __future__ import annotations
import pickle, time
from dataclasses import replace
from pathlib import Path

import numpy as np

import hz_experiment as hz          # patch_horizons, load_env, make_data, NEW_CACHE, PROD_CACHE
import final_stats as fs            # config_nav, bench_nav, stats, CFG
from research.walkforward.runner import run_splits
from research.walkforward.splits import resolve_splits
from data.db import get_db
from backtesting.data_loader import SPY

ERA = "2021-12-31"
EXCLUSIONS = ["none", "any_p10", "two_p10", "soft_p10"]


def get_run_new(panel, matrix_raw, sectors):
    if hz.NEW_CACHE.exists():
        print(f"[new] load {hz.NEW_CACHE}", flush=True)
        return pickle.load(hz.NEW_CACHE.open("rb"))
    hz.patch_horizons()
    t0 = time.time()
    print("[new] walk-forward re-score on 6M/12M (~10-20 min)...", flush=True)
    run = run_splits(panel, resolve_splits("rolling5y"), matrix_raw, sectors, verbose=True)
    pickle.dump(run, hz.NEW_CACHE.open("wb"))
    print(f"[new] done ({time.time()-t0:.0f}s)", flush=True)
    return run


def report(data, matrix, cfg, label, rows):
    wts = fs.weights_for(data, cfg)
    dates = [d for d in data.rebal_dates if d in wts and d in matrix.index]
    nav = fs.config_nav(dates, wts, matrix)
    ds = np.array(dates)
    def win(lo, hi):
        j = np.where((ds >= lo) & (ds <= hi))[0]
        return nav[max(j[0]-1, 0): j[-1]+1]
    f = fs.stats(win(dates[0], dates[-1]))
    e1 = fs.stats(win(dates[0], ERA))
    e2 = fs.stats(win("2022-01-01", dates[-1]))
    rows.append((label, f, e1, e2, nav[-1]))
    return dates


def main():
    panel, matrix_raw, matrix, sectors, vix = hz.load_env()
    run_new = get_run_new(panel, matrix_raw, sectors)
    run_prod = pickle.load(hz.PROD_CACHE.open("rb"))
    with get_db() as db:
        data_new = hz.make_data(run_new, panel, matrix, sectors, vix, db)
        data_prod = hz.make_data(run_prod, panel, matrix, sectors, vix, db)

    rows = []
    # 6M/12M composite: sweep the exclusion knob on the locked construction
    for ex in EXCLUSIONS:
        report(data_new, matrix, replace(fs.CFG, exclusion=ex), f"6M12M  {ex}", rows)
    # reference: locked 3M/6M with and without the veto
    dates = report(data_prod, matrix, replace(fs.CFG, exclusion="none"), "3M6M   none (LOCKED)", rows)
    report(data_prod, matrix, replace(fs.CFG, exclusion="any_p10"), "3M6M   any_p10", rows)

    print("\n" + "=" * 92)
    print("AFTER-TAX (13mo x4 sleeves, top25/cap5/cap_match/VIX). 'any_p10' = drop stock if any factor<10")
    print("=" * 92)
    hdr = f"{'config':22}{'FULL cagr/shrp/mdd':>26}{'ERA2 cagr/shrp/mdd':>26}{'walkaway':>13}"
    print(hdr)
    for label, f, e1, e2, wa in rows:
        print(f"{label:22}"
              f"{f['cagr']*100:6.1f}% {f['sharpe']:4.2f} {f['mdd']*100:6.1f}%   "
              f"{e2['cagr']*100:6.1f}% {e2['sharpe']:4.2f} {e2['mdd']*100:6.1f}%   "
              f"${wa:>10,.0f}")
    for tkr in (SPY, "QQQ"):
        nav = fs.bench_nav(matrix, dates, tkr)
        m = fs.stats(nav)
        print(f"{tkr:22}{m['cagr']*100:6.1f}% {m['sharpe']:4.2f} {m['mdd']*100:6.1f}%"
              f"{'':>29}${nav[-1]:>10,.0f}")

    # full 5-metric table for the two headline configs
    print("\n--- full metrics, FULL / ERA1 / ERA2 ---")
    for label, f, e1, e2, wa in rows:
        print(f"\n{label}")
        for nm, m in [("FULL", f), ("ERA1", e1), ("ERA2", e2)]:
            print(f"  {nm}  CAGR {m['cagr']*100:5.1f}%  Sharpe {m['sharpe']:.2f}  "
                  f"Sortino {m['sortino']:.2f}  Calmar {m['calmar']:.2f}  maxDD {m['mdd']*100:5.1f}%")


if __name__ == "__main__":
    main()
