"""Textbook credit-spread test with DAILY path + trade management.
Enter monthly (non-overlapping), 30 DTE, short strike at target delta, long strike
$W wide. Reprice daily (BS, sigma = daily VIX). Compare:
  - hold to expiry
  - close at 50% of max profit (buy back when spread value <= 0.5 x entry credit)
  - + a 2x-credit stop-loss variant
Call credit spreads (bearish/hedge) AND put credit spreads (bullish/income) for contrast.
Per-leg slippage $0.03 on entry and on any early close. Returns are on capital-at-risk
(max loss = width - credit), one trade at a time, compounded.
"""
from __future__ import annotations
import math, pickle
import numpy as np
import pandas as pd

from hz_experiment import load_env
from backtesting.data_loader import SPY

FFR = {2017: 0.0100, 2018: 0.0183, 2019: 0.0216, 2020: 0.0038, 2021: 0.0008,
       2022: 0.0168, 2023: 0.0503, 2024: 0.0510, 2025: 0.0433, 2026: 0.0390}
SLIP = 0.03          # $ per option leg, per transaction
DTE = 30
STEP = 23            # trading days between (non-overlapping) entries (>30 cal days)


def _n(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _d1(S, K, sig, T, r):
    return (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))


def call_px(S, K, sig, T, r):
    if T <= 0:
        return max(S - K, 0.0)
    d1 = _d1(S, K, sig, T, r)
    return S * _n(d1) - K * math.exp(-r * T) * _n(d1 - sig * math.sqrt(T))


def put_px(S, K, sig, T, r):
    if T <= 0:
        return max(K - S, 0.0)
    d1 = _d1(S, K, sig, T, r)
    return K * math.exp(-r * T) * _n(-(d1 - sig * math.sqrt(T))) - S * _n(-d1)


def find_strike(S, sig, T, r, target, kind):
    best, berr = S, 9.9
    for otm in np.arange(0.0, 0.20, 0.0015):
        K = S * (1 + otm) if kind == "call" else S * (1 - otm)
        d1 = _d1(S, K, sig, T, r)
        dlt = _n(d1) if kind == "call" else _n(-d1)   # magnitude
        if abs(dlt - target) < berr:
            berr, best = abs(dlt - target), K
    return best


def spread_val(S, Ks, Kl, sig, T, r, kind):
    if kind == "call":
        return call_px(S, Ks, sig, T, r) - call_px(S, Kl, sig, T, r)
    return put_px(S, Ks, sig, T, r) - put_px(S, Kl, sig, T, r)


def run(kind, target, W, manage, spy, vixd, tdays, td_dt, stop=None, redeploy=False):
    n = len(tdays)
    rois, wins = [], 0
    i = STEP
    while i < n - 1:
        S0 = spy[i]; sig0 = vixd[i] / 100.0; r = FFR[int(tdays[i][:4])]
        T0 = DTE / 365.0
        Ks = find_strike(S0, sig0, T0, r, target, kind)
        Kl = Ks + W if kind == "call" else Ks - W
        width = abs(Kl - Ks)
        credit = spread_val(S0, Ks, Kl, sig0, T0, r, kind) - 2 * SLIP
        if credit <= 0.02:
            i += STEP; continue
        maxloss = width - credit
        exp_dt = td_dt[i] + pd.Timedelta(days=DTE)
        pnl, close_j = None, None
        j = i + 1
        while j < n and td_dt[j] <= exp_dt:
            Trem = max((exp_dt - td_dt[j]).days, 0) / 365.0
            S = spy[j]; sig = vixd[j] / 100.0
            val = spread_val(S, Ks, Kl, sig, Trem, r, kind)
            if manage and val <= 0.5 * credit:                     # 50% profit target
                pnl = credit - val - 2 * SLIP; close_j = j; break
            if stop and val >= stop * credit:                      # stop-loss
                pnl = credit - val - 2 * SLIP; close_j = j; break
            if Trem <= 0:
                intr = max(S - Ks, 0.0) if kind == "call" else max(Ks - S, 0.0)
                pnl = credit - min(intr, width); close_j = j; break
            j += 1
        if pnl is None:   # settle at expiry price
            close_j = min(j, n - 1); S = spy[close_j]
            intr = max(S - Ks, 0.0) if kind == "call" else max(Ks - S, 0.0)
            pnl = credit - min(intr, width)
        rois.append(pnl / maxloss); wins += pnl > 0
        # next entry: right after this trade closes (redeploy) or fixed monthly cadence
        i = close_j + 1 if redeploy else i + STEP
    rois = np.array(rois)
    F = 0.10                                       # risk 10% of equity per trade (realistic)
    eq = np.cumprod(1 + F * rois)
    yrs = (td_dt[-1] - td_dt[STEP]).days / 365.25
    cagr = eq[-1] ** (1 / yrs) - 1 if eq[-1] > 0 else -1.0
    mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
    sharpe = rois.mean() / rois.std(ddof=1) * math.sqrt(len(rois) / yrs)
    return dict(n=len(rois), tpy=len(rois) / yrs, win=wins / len(rois), avg=rois.mean(),
                worst=rois.min(), cagr=cagr, mdd=mdd, sharpe=sharpe, final=eq[-1])


def main():
    panel, mr, matrix, sec, vix = load_env()
    spy_s = matrix[SPY].dropna()
    spy_s = spy_s[spy_s.index.astype(str) >= "2017-01-01"]
    tdays = list(spy_s.index)
    td_dt = pd.to_datetime(tdays)
    spy = spy_s.values.astype(float)
    vixd = vix.reindex(tdays).astype(float).ffill().values

    print(f"\nDaily-path credit spreads: 30 DTE, SPY, slippage ${SLIP}/leg. "
          f"Returns on capital-at-risk.\n")
    hdr = (f"{'strategy':44}{'n':>4}{'tr/yr':>6}{'win%':>6}{'avg/tr':>8}"
           f"{'CAGR*':>8}{'maxDD*':>8}{'Sharpe':>8}")
    print(hdr)
    print("(* CAGR/maxDD risk 10% of capital per trade; hold cadence vs continuous redeploy)")
    # label, kind, tgt, W, manage, stop, redeploy
    configs = [
        ("CALL 0.20d $5, hold-to-expiry (redeploy)", "call", 0.20, 5.0, False, None, True),
        ("CALL 0.20d $5, close 50% (fixed monthly)", "call", 0.20, 5.0, True, None, False),
        ("CALL 0.20d $5, close 50% (REDEPLOY)", "call", 0.20, 5.0, True, None, True),
        ("CALL 0.15d $5, hold-to-expiry (redeploy)", "call", 0.15, 5.0, False, None, True),
        ("CALL 0.15d $5, close 50% (fixed monthly)", "call", 0.15, 5.0, True, None, False),
        ("CALL 0.15d $5, close 50% (REDEPLOY)", "call", 0.15, 5.0, True, None, True),
        ("PUT  0.20d $5, close 50% (fixed monthly)", "put", 0.20, 5.0, True, None, False),
        ("PUT  0.20d $5, close 50% (REDEPLOY)", "put", 0.20, 5.0, True, None, True),
    ]
    for label, kind, tgt, W, manage, stop, redep in configs:
        m = run(kind, tgt, W, manage, spy, vixd, tdays, td_dt, stop, redeploy=redep)
        print(f"{label:44}{m['n']:>4}{m['tpy']:>6.1f}{m['win']*100:>5.0f}%{m['avg']*100:>7.1f}%"
              f"{m['cagr']*100:>7.1f}%{m['mdd']*100:>7.0f}%{m['sharpe']:>8.2f}")


if __name__ == "__main__":
    main()
