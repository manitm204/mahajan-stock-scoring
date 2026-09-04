"""Account-setup sweeps (user request 2026-07-19, $15k account planning).

1. Hold x sleeves grid through the lot-level after-tax simulator:
   HOLD in {13, 14} months x SLEEVES in {1, 2, 4, 5, 6, 10}.
   (Sim is monthly-granular: 13mo/5 approximates the user's 55wk/5-sleeve idea,
    14mo/10 approximates 60wk/10.)
2. Credit-spread sleeve sizing: risk F in {5%, 10%, 20%} of the sleeve per
   trade; sleeve standalone + 70/30 book blend, after-tax (XSP 60/40).
"""
from __future__ import annotations
import pickle
import numpy as np
import pandas as pd

from hz_experiment import load_env, make_data, PROD_CACHE
import final_stats as fs
import combined_credit as cc
from data.db import get_db
from backtesting.data_loader import SPY


def main():
    panel, mr, matrix, sectors, vix = load_env()
    run = pickle.load(PROD_CACHE.open("rb"))
    with get_db() as db:
        data = make_data(run, panel, matrix, sectors, vix, db)
    wts = fs.weights_for(data, fs.CFG)
    dates = [d for d in data.rebal_dates if d in wts and d in matrix.index]
    spy_gross = matrix[SPY].reindex(dates).astype(float).pct_change().dropna()

    def show(name, r, lo):
        rr = r[r.index.astype(str) >= lo].dropna()
        m = cc.stats(rr, spy_gross[spy_gross.index.astype(str) >= lo])
        print(f"{name:26}{m['cagr']*100:6.1f}%{m['sharpe']:8.2f}{m['sortino']:9.2f}"
              f"{m['mdd']*100:7.1f}%{m['beta']:7.2f}{m['final']:>12,.0f}")

    print("\n================ 1. HOLD x SLEEVES grid (after-tax book) ================")
    navs = {}
    for hold in (13, 14):
        for sleeves in (1, 2, 4, 5, 6, 10):
            fs.HOLD, fs.SLEEVES = hold, sleeves
            nav = pd.Series(fs.config_nav(dates, wts, matrix), index=dates)
            navs[(hold, sleeves)] = nav.pct_change().dropna()
            print(f"  done {hold}mo x {sleeves}", flush=True)
    for wlabel, lo in [("FULL 2017 -> 2026", "2000"), ("2020-start", "2020-01-01")]:
        print(f"\n--- {wlabel} ---")
        print(f"{'':26}{'CAGR':>7}{'Sharpe':>8}{'Sortino':>9}{'maxDD':>8}{'beta':>7}{'final $':>12}")
        for (hold, sleeves), r in navs.items():
            show(f"{hold}mo hold x {sleeves} sleeves", r, lo)

    fs.HOLD, fs.SLEEVES = 13, 4      # restore locked spec for part 2
    r_book = pd.Series(fs.config_nav(dates, wts, matrix), index=dates).pct_change().dropna()
    spy_at = pd.Series(fs.bench_nav(matrix, dates, SPY), index=dates).pct_change().dropna()

    print("\n================ 2. Credit-spread sleeve sizing (F = % of sleeve risked/trade) ================")
    for wlabel, lo in [("FULL 2017 -> 2026", "2000"), ("2020-start", "2020-01-01")]:
        print(f"\n--- {wlabel} (after-tax; sleeve taxed 60/40 XSP) ---")
        print(f"{'':26}{'CAGR':>7}{'Sharpe':>8}{'Sortino':>9}{'maxDD':>8}{'beta':>7}{'final $':>12}")
        show("Book 100% (13mo x4)", r_book, lo)
        show("SPY", spy_at, lo)
        for f in (0.05, 0.10, 0.20):
            cc.F = f
            intr, spr = cc.sleeve_returns(dates, matrix, vix)
            intr = intr.reindex(r_book.index).fillna(0.0)
            spr = spr.reindex(r_book.index).fillna(0.0)
            sleeve = intr * (1 - cc.ST) + spr * (1 - cc.BLEND)
            show(f"[sleeve alone F={f:.0%}]", sleeve, lo)
            show(f"70/30 book + spreads F={f:.0%}", 0.7 * r_book + 0.3 * sleeve, lo)


if __name__ == "__main__":
    main()
