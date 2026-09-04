"""IVR-gate sweep (user request 2026-07-19, follow-up to options_zoo):
  - call credit spread (15d $5 30dte HTE) at IVR gates {none, 20, 30, 40, 50}
  - iron condor (16d $5w 45dte PT50) at the same gates
  - hybrid: condor when IVR >= t, else call spread (always deployed)
Adds % of trading days in a position. Idle cash always earns FFR (SPAXX).
Caches the slow book/env build to cache/options_zoo_env.pkl for fast iteration.
"""
from __future__ import annotations
import pickle
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

import options_zoo as oz
import final_stats as fs
import combined_credit as cc

ROOT = Path("/home/manit/Desktop/fun_projects/mahajan_hedge_fund")
CACHE = ROOT / "cache" / "options_zoo_env.pkl"
GAP, F, SLIP = oz.GAP, oz.F, oz.SLIP
FFR, ST, BLEND = oz.FFR, oz.ST, oz.BLEND

CS = (oz.b_callspread, 30, None)      # incumbent call spread: 30 dte, hold to expiry
IC = (oz.b_condor, 45, 0.5)           # iron condor: 45 dte, 50% profit target


def get_env():
    if CACHE.exists():
        return pickle.load(CACHE.open("rb"))
    from hz_experiment import load_env, make_data, PROD_CACHE
    from data.db import get_db
    from backtesting.data_loader import SPY
    panel, mr, matrix, sectors, vix = load_env()
    run = pickle.load(PROD_CACHE.open("rb"))
    with get_db() as db:
        data = make_data(run, panel, matrix, sectors, vix, db)
    wts = fs.weights_for(data, fs.CFG)
    mdates = [d for d in data.rebal_dates if d in wts and d in matrix.index]
    spy_s = matrix[SPY].dropna()
    env = dict(
        mdates=mdates,
        r_book=pd.Series(fs.config_nav(mdates, wts, matrix), index=mdates).pct_change().dropna(),
        spy_at=pd.Series(fs.bench_nav(matrix, mdates, SPY), index=mdates).pct_change().dropna(),
        spy_gross=matrix[SPY].reindex(mdates).astype(float).pct_change().dropna(),
        tdays=[str(d) for d in spy_s.index],
        spy=spy_s.values.astype(float),
        vixd=vix.reindex(spy_s.index).astype(float).ffill().bfill().values,
    )
    pickle.dump(env, CACHE.open("wb"))
    print(f"  [env cached -> {CACHE}]")
    return env


def run_flex(chooser, tdays, td_dt, spy, vixd, ivr, i0):
    """Like oz.run_strategy but the structure is chosen per entry from IVR.
    chooser(ivr_now) -> (build, dte, pt, tag) or None (stay in cash)."""
    n = len(tdays)
    trades, daily = [], defaultdict(float)
    last_entry = -999
    i = i0
    while i < n - 2:
        pick = chooser(ivr[i]) if i - last_entry >= GAP else None
        if pick is None:
            i += 1
            continue
        build, dte, pt, tag = pick
        S0, atm = spy[i], vixd[i] / 100.0
        r = FFR[int(tdays[i][:4])]
        T0 = dte / 365.0
        legs, basis, _ = build(S0, atm, T0, r)
        nl = sum(abs(q) for q, _, _ in legs)
        V0 = oz.val(legs, S0, atm, T0, r)
        credit = -V0 - nl * SLIP
        if credit <= 0.02:
            i += 5
            continue
        if basis is None:
            grid = np.arange(0.30 * S0, 1.60 * S0, S0 * 0.004)
            basis = max(-(credit + min(oz.payoff(legs, s) for s in grid)), 0.05)
        exp_dt = td_dt[i] + pd.Timedelta(days=dte)
        prevV, pnl, j = V0, None, i + 1
        while j < n:
            Trem = max((exp_dt - td_dt[j]).days, 0) / 365.0
            V = (oz.val(legs, spy[j], vixd[j] / 100.0, Trem, r) if Trem > 0
                 else oz.payoff(legs, spy[j]))
            daily[tdays[j]] += F * (V - prevV) / basis
            profit = credit + V
            if Trem <= 0:
                pnl = profit
                break
            if pt is not None and profit >= pt * credit:
                daily[tdays[j]] -= F * nl * SLIP / basis
                pnl = profit - nl * SLIP
                break
            prevV = V
            j += 1
        if pnl is None:
            break
        trades.append(dict(roi=pnl / basis, credit=credit, basis=basis, tag=tag,
                           tdi=j - i, days=(td_dt[j] - td_dt[i]).days, win=pnl > 0))
        last_entry = i
        i = max(j + 1, last_entry + GAP)
    return trades, daily


def main():
    env = get_env()
    mdates, r_book, spy_at, spy_gross = (env["mdates"], env["r_book"],
                                         env["spy_at"], env["spy_gross"])
    tdays, spy, vixd = env["tdays"], env["spy"], env["vixd"]
    td_dt = pd.to_datetime(tdays)
    vser = pd.Series(vixd)
    lo, hi = vser.rolling(252).min(), vser.rolling(252).max()
    ivr = ((vser - lo) / (hi - lo).replace(0, np.nan) * 100).fillna(50).values
    i0 = max(252, int(np.searchsorted(np.array(tdays), "2017-01-01")))
    ndays = len(tdays) - 1 - i0
    yrs = (td_dt[-1] - td_dt[i0]).days / 365.25

    def gated(struct, t):
        b, dte, pt = struct
        tag = "ic" if struct is IC else "cs"
        return lambda v: (b, dte, pt, tag) if (t is None or v >= t) else None

    def hybrid(t):
        return lambda v: (IC[0], IC[1], IC[2], "ic") if v >= t else (CS[0], CS[1], CS[2], "cs")

    configs = ([(f"CallSpr {'ungated' if t is None else f'IVR>={t}'}", gated(CS, t))
                for t in (None, 20, 30, 40, 50)]
               + [(f"IronCondor {'ungated' if t is None else f'IVR>={t}'}", gated(IC, t))
                  for t in (None, 20, 30, 40, 50)]
               + [(f"Hybrid: IC if IVR>={t} else CS", hybrid(t))
                  for t in (20, 30, 40, 50)])

    print("\n=========== 1. Trade stats + time deployed (2017 -> mid-2026) ===========")
    print(f"{'strategy':30}{'n':>4}{'tr/yr':>6}{'%in':>6}{'mix ic/cs':>10}{'win%':>6}"
          f"{'avg':>7}{'worst':>8}{'hold_d':>7}")
    sleeves = {}
    for label, chooser in configs:
        trades, daily = run_flex(chooser, tdays, td_dt, spy, vixd, ivr, i0)
        t = pd.DataFrame(trades)
        nic = int((t["tag"] == "ic").sum()); ncs = int((t["tag"] == "cs").sum())
        print(f"{label:30}{len(t):>4}{len(t)/yrs:>6.1f}{t['tdi'].sum()/ndays*100:>5.0f}%"
              f"{f'{nic}/{ncs}':>10}{t['win'].mean()*100:>5.0f}%"
              f"{t['roi'].mean()*100:>6.1f}%{t['roi'].min()*100:>7.1f}%{t['days'].median():>7.0f}")
        sleeves[label] = oz.monthly_sleeve(daily, mdates)

    print("\n=========== 2. After-tax sleeve + 70/30 blend (XSP 60/40; F=10%/trade; idle cash at FFR) ===========")
    for wlabel, cut in [("FULL 2017 -> 2026", "2000"), ("2020-start", "2020-01-01")]:
        sg = spy_gross[spy_gross.index.astype(str) >= cut]
        def m_of(r):
            return cc.stats(r[r.index.astype(str) >= cut].dropna(), sg)
        print(f"\n--- {wlabel} ---")
        print(f"{'':30}{'slvCAGR':>8}{'slvShp':>7}{'slvBeta':>8} |"
              f"{'CAGR':>7}{'Sharpe':>8}{'maxDD':>8}{'beta':>7}{'final $':>12}")
        mb, ms = m_of(r_book), m_of(spy_at)
        print(f"{'Book 100% (13mo x4)':30}{'':8}{'':7}{'':8} |{mb['cagr']*100:6.1f}%"
              f"{mb['sharpe']:8.2f}{mb['mdd']*100:7.1f}%{mb['beta']:7.2f}{mb['final']:>12,.0f}")
        print(f"{'SPY':30}{'':8}{'':7}{'':8} |{ms['cagr']*100:6.1f}%"
              f"{ms['sharpe']:8.2f}{ms['mdd']*100:7.1f}%{ms['beta']:7.2f}{ms['final']:>12,.0f}")
        for label, (intr, opt) in sleeves.items():
            intr2 = intr.reindex(r_book.index).fillna(0.0)
            opt2 = opt.reindex(r_book.index).fillna(0.0)
            sleeve = intr2 * (1 - ST) + opt2 * (1 - BLEND)
            a, b = m_of(sleeve), m_of(0.7 * r_book + 0.3 * sleeve)
            print(f"{label:30}{a['cagr']*100:7.1f}%{a['sharpe']:7.2f}{a['beta']:8.2f} |"
                  f"{b['cagr']*100:6.1f}%{b['sharpe']:8.2f}{b['mdd']*100:7.1f}%"
                  f"{b['beta']:7.2f}{b['final']:>12,.0f}")


if __name__ == "__main__":
    main()
