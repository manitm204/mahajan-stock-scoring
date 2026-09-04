"""Options-strategy zoo (user request 2026-07-19, $20k account planning).

Tests index (XSP-style) versions of premium-selling structures on the daily SPY
path, BS-priced with VIX (call skew -0.9 vol-pt / 1% OTM as validated in
combined_credit; put skew +0.5 vol-pt / 1% OTM). One position at a time,
entries ~monthly (>=23 trading days apart), 50% profit target where specified,
$0.03/leg slippage. "High IVR" proxied by VIX 252-day IV rank.

Outputs:
  1. trade-level stats per strategy (n, win%, avg/worst ROI on basis, hold days)
  2. sleeve monthly stream (risk F=10% of sleeve per trade, idle cash at FFR,
     XSP 60/40 tax) standalone + 70/30 blend with the after-tax book,
     vs the incumbent 0.15d call credit spread.

Single-stock variants (short puts on beaten-down high-IVR names) are NOT
testable here: no historical per-stock IV/options data.
"""
from __future__ import annotations
import math, pickle
from collections import defaultdict
import numpy as np
import pandas as pd

from hz_experiment import load_env, make_data, PROD_CACHE
import final_stats as fs
import combined_credit as cc
from data.db import get_db
from backtesting.data_loader import SPY

FFR = cc.FFR
ST, BLEND = cc.ST, cc.BLEND          # 32% ST; 60/40 blend 21.8% (XSP)
SLIP = 0.03
F = 0.10                              # fraction of sleeve risked (basis) per trade
# calibrated to real SPY 30-DTE quotes 2026-07-21 (see
# output/ablation/options_calibration/): ATM IV ~0.80*VIX, call skew 0.68,
# put skew 1.07 (real index put skew is ~2x steeper than the old 0.5 assumed).
ATM_VIX = 0.80
CSKEW, PSKEW, IVFLOOR = 0.68, 1.07, 0.06
GAP = 23                              # min trading days between entries


def _n(x): return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def leg_iv(kind, S, K, atm):
    if kind == "c":
        return max(atm - CSKEW * max(K / S - 1.0, 0.0), IVFLOOR)
    return max(atm + PSKEW * max(1.0 - K / S, 0.0), IVFLOOR)


def bs(kind, S, K, sig, T, r):
    if T <= 0:
        return max(S - K, 0.0) if kind == "c" else max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    if kind == "c":
        return S * _n(d1) - K * math.exp(-r * T) * _n(d2)
    return K * math.exp(-r * T) * _n(-d2) - S * _n(-d1)


def strike_at(kind, S, atm, T, r, tgt):
    """Strike whose (skew-adjusted) |delta| is closest to tgt."""
    best, berr = S, 9.9
    for otm in np.arange(0.0, 0.45, 0.002):
        K = S * (1 + otm) if kind == "c" else S * (1 - otm)
        sig = leg_iv(kind, S, K, atm)
        d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
        dlt = _n(d1) if kind == "c" else _n(-d1)
        if abs(dlt - tgt) < berr:
            berr, best = abs(dlt - tgt), K
    return best


def val(legs, S, atm, T, r):
    """Position value to holder (net-short structures are negative)."""
    return sum(q * bs(k, S, K, leg_iv(k, S, K, atm), T, r) for q, k, K in legs)


def payoff(legs, S):
    return sum(q * (max(S - K, 0.0) if k == "c" else max(K - S, 0.0))
               for q, k, K in legs)


def put_margin(S, K, prem):
    """Reg-T naked put per-unit: prem + max(15% S - OTM amt, 10% K)."""
    return prem + max(0.15 * S - max(S - K, 0.0), 0.10 * K)


# ---- strategy builders: (S, atm, T, r) -> (legs, basis, note) -----------------
def b_callspread(S, a, T, r):
    K = strike_at("c", S, a, T, r, 0.15)
    return [(-1, "c", K), (1, "c", K + 5)], None, ""

def b_shortput(S, a, T, r):
    K = strike_at("p", S, a, T, r, 0.19)
    prem = bs("p", S, K, leg_iv("p", S, K, a), T, r)
    return [(-1, "p", K)], put_margin(S, K, prem), ""

def b_putspread(S, a, T, r):
    K = strike_at("p", S, a, T, r, 0.30)
    return [(-1, "p", K), (1, "p", K - 5)], None, ""

def b_ratio(S, a, T, r):
    Kl = strike_at("p", S, a, T, r, 0.30)
    Ks = strike_at("p", S, a, T, r, 0.15)
    prem = bs("p", S, Ks, leg_iv("p", S, Ks, a), T, r)
    return [(1, "p", Kl), (-2, "p", Ks)], put_margin(S, Ks, prem), ""

def b_jade(S, a, T, r):
    Kp = strike_at("p", S, a, T, r, 0.20)
    Kc = strike_at("c", S, a, T, r, 0.28)
    prem = bs("p", S, Kp, leg_iv("p", S, Kp, a), T, r)
    return ([(-1, "p", Kp), (-1, "c", Kc), (1, "c", Kc + 5)],
            put_margin(S, Kp, prem) + 5.0, "")

def b_bwb(S, a, T, r):
    K = strike_at("p", S, a, T, r, 0.20)
    return [(1, "p", K + 5), (-2, "p", K), (1, "p", K - 15)], None, ""

def b_condor(S, a, T, r):
    Kp = strike_at("p", S, a, T, r, 0.16)
    Kc = strike_at("c", S, a, T, r, 0.16)
    return [(-1, "p", Kp), (1, "p", Kp - 5), (-1, "c", Kc), (1, "c", Kc + 5)], None, ""

def b_strangle(S, a, T, r):
    Kp = strike_at("p", S, a, T, r, 0.18)
    Kc = strike_at("c", S, a, T, r, 0.18)
    pp = bs("p", S, Kp, leg_iv("p", S, Kp, a), T, r)
    pc = bs("c", S, Kc, leg_iv("c", S, Kc, a), T, r)
    return [(-1, "p", Kp), (-1, "c", Kc)], put_margin(S, Kp, pp) + pc, ""


STRATS = [
    # label, builder, dte, profit-target, time-exit dte, ivr gate
    ("CallSpr 15d $5 30dte HTE (incumbent)", b_callspread, 30, None, None, None),
    ("ShortPut 19d 45dte PT50",              b_shortput,   45, 0.5,  None, None),
    ("ShortPut 19d PT50, IVR>=30",           b_shortput,   45, 0.5,  None, 30),
    ("PutSpread 30d $5 45dte PT50",          b_putspread,  45, 0.5,  None, None),
    ("Ratio 1x2 put 30/15d 45dte PT50/21d",  b_ratio,      45, 0.5,  21,   None),
    ("JadeLizard 20p/28c+$5 50dte PT50",     b_jade,       50, 0.5,  None, None),
    ("BWB put +5/-2x20d/-15 40dte PT50",     b_bwb,        40, 0.5,  None, None),
    ("IronCondor 16d $5w 45dte PT50",        b_condor,     45, 0.5,  None, None),
    ("IronCondor 16d PT50, IVR>=30",         b_condor,     45, 0.5,  None, 30),
    ("Strangle 18d 45dte PT50",              b_strangle,   45, 0.5,  None, None),
    ("Strangle 18d PT50, IVR>=30",           b_strangle,   45, 0.5,  None, 50),
]


def run_strategy(build, dte, pt, texit, gate, tdays, td_dt, spy, vixd, ivr, i0):
    n = len(tdays)
    trades, daily = [], defaultdict(float)
    last_entry = -999
    i = i0
    while i < n - 2:
        if (i - last_entry < GAP) or (gate is not None and ivr[i] < gate):
            i += 1
            continue
        S0, atm = spy[i], vixd[i] / 100.0 * ATM_VIX     # calibrated ATM IV
        r = FFR[int(tdays[i][:4])]
        T0 = dte / 365.0
        legs, basis, _ = build(S0, atm, T0, r)
        nl = sum(abs(q) for q, _, _ in legs)
        V0 = val(legs, S0, atm, T0, r)
        credit = -V0 - nl * SLIP
        if credit <= 0.02:
            i += 5
            continue
        if basis is None:  # defined risk: numeric max loss at expiry
            grid = np.arange(0.30 * S0, 1.60 * S0, S0 * 0.004)
            basis = max(-(credit + min(payoff(legs, s) for s in grid)), 0.05)
        exp_dt = td_dt[i] + pd.Timedelta(days=dte)
        prevV, pnl, j = V0, None, i + 1
        while j < n:
            Trem = max((exp_dt - td_dt[j]).days, 0) / 365.0
            V = val(legs, spy[j], vixd[j] / 100.0, Trem, r) if Trem > 0 else payoff(legs, spy[j])
            daily[tdays[j]] += F * (V - prevV) / basis
            profit = credit + V
            if Trem <= 0:
                pnl = profit
                break
            if pt is not None and profit >= pt * credit:
                daily[tdays[j]] -= F * nl * SLIP / basis
                pnl = profit - nl * SLIP
                break
            if texit is not None and Trem <= texit / 365.0:
                daily[tdays[j]] -= F * nl * SLIP / basis
                pnl = profit - nl * SLIP
                break
            prevV = V
            j += 1
        if pnl is None:
            break  # ran off the data with an open trade; drop it
        trades.append(dict(roi=pnl / basis, credit=credit, basis=basis,
                           days=(td_dt[j] - td_dt[i]).days, win=pnl > 0))
        last_entry = i
        i = max(j + 1, last_entry + GAP)
    return trades, daily


def monthly_sleeve(daily, mdates):
    """Bucket F-scaled daily option P&L into the book's monthly periods + FFR interest."""
    keys = sorted(daily)
    out_i, out_o = {}, {}
    dt = pd.to_datetime([str(d) for d in mdates])
    for k in range(len(mdates) - 1):
        lo, hi = str(mdates[k]), str(mdates[k + 1])
        out_o[mdates[k + 1]] = sum(daily[d] for d in keys if lo < d <= hi)
        out_i[mdates[k + 1]] = FFR[int(hi[:4])] * max((dt[k + 1] - dt[k]).days, 1) / 365.0
    return pd.Series(out_i), pd.Series(out_o)


def main():
    panel, mr, matrix, sectors, vix = load_env()
    run = pickle.load(PROD_CACHE.open("rb"))
    with get_db() as db:
        data = make_data(run, panel, matrix, sectors, vix, db)
    wts = fs.weights_for(data, fs.CFG)
    mdates = [d for d in data.rebal_dates if d in wts and d in matrix.index]

    spy_s = matrix[SPY].dropna()
    tdays = [str(d) for d in spy_s.index]
    td_dt = pd.to_datetime(tdays)
    spy = spy_s.values.astype(float)
    vixd = vix.reindex(spy_s.index).astype(float).ffill().bfill().values
    vser = pd.Series(vixd)
    lo, hi = vser.rolling(252).min(), vser.rolling(252).max()
    ivr = ((vser - lo) / (hi - lo).replace(0, np.nan) * 100).fillna(50).values
    i0 = max(252, int(np.searchsorted(np.array(tdays), "2017-01-01")))

    r_book = pd.Series(fs.config_nav(mdates, wts, matrix), index=mdates).pct_change().dropna()
    spy_at = pd.Series(fs.bench_nav(matrix, mdates, SPY), index=mdates).pct_change().dropna()
    spy_gross = matrix[SPY].reindex(mdates).astype(float).pct_change().dropna()

    print("\n=========== 1. Trade-level stats (ROI on basis = max loss or Reg-T margin) ===========")
    print(f"{'strategy':38}{'n':>4}{'tr/yr':>6}{'win%':>6}{'avg':>7}{'worst':>8}"
          f"{'credit':>8}{'basis$':>8}{'hold_d':>7}")
    sleeves = {}
    for label, build, dte, pt, texit, gate in STRATS:
        trades, daily = run_strategy(build, dte, pt, texit, gate,
                                     tdays, td_dt, spy, vixd, ivr, i0)
        if not trades:
            print(f"{label:38}   0   -- no trades (credit filter/gate)")
            continue
        t = pd.DataFrame(trades)
        yrs = (td_dt[-1] - td_dt[i0]).days / 365.25
        print(f"{label:38}{len(t):>4}{len(t)/yrs:>6.1f}{t['win'].mean()*100:>5.0f}%"
              f"{t['roi'].mean()*100:>6.1f}%{t['roi'].min()*100:>7.1f}%"
              f"{t['credit'].mean()*100:>8,.0f}{t['basis'].mean()*100:>8,.0f}"
              f"{t['days'].median():>7.0f}")
        sleeves[label] = monthly_sleeve(daily, mdates)

    print("\n=========== 2. Sleeve standalone + 70/30 blend, after-tax (XSP 60/40; F=10%/trade) ===========")
    for wlabel, cut in [("FULL 2017 -> 2026", "2000"), ("2020-start", "2020-01-01")]:
        print(f"\n--- {wlabel} ---")
        print(f"{'':44}{'CAGR':>7}{'Sharpe':>8}{'Sortino':>9}{'maxDD':>8}{'beta':>7}{'final $':>12}")
        def show(name, r):
            rr = r[r.index.astype(str) >= cut].dropna()
            m = cc.stats(rr, spy_gross[spy_gross.index.astype(str) >= cut])
            print(f"{name:44}{m['cagr']*100:6.1f}%{m['sharpe']:8.2f}{m['sortino']:9.2f}"
                  f"{m['mdd']*100:7.1f}%{m['beta']:7.2f}{m['final']:>12,.0f}")
        show("Book 100% (13mo x4)", r_book)
        show("SPY", spy_at)
        for label, (intr, opt) in sleeves.items():
            intr2 = intr.reindex(r_book.index).fillna(0.0)
            opt2 = opt.reindex(r_book.index).fillna(0.0)
            sleeve = intr2 * (1 - ST) + opt2 * (1 - BLEND)
            show(f"  [{label}] alone", sleeve)
            show(f"  70/30 + {label}", 0.7 * r_book + 0.3 * sleeve)


if __name__ == "__main__":
    main()
