"""Protective-PUT hedge (BUYING options) vs the covered-call (SELLING) overlay.

You BUY monthly OTM puts on `cov` of the book. Max loss per month = premium paid.
Premium priced Black-Scholes with sigma = VIX; OTM puts marked up 20% for the
well-known put skew (VIX understates OTM put cost) + slippage. Put pays off in a
crash: payoff = max(-(spy_ret + otm), 0). Overlay unit = payoff - premium.
Tax: 60/40 (Section 1256), applied to net monthly overlay P&L.
Book after-tax return from the mark-to-liquidation sleeve sim (final_stats).
"""
from __future__ import annotations
import math, pickle
import numpy as np
import pandas as pd

from hz_experiment import load_env, make_data, PROD_CACHE
import final_stats as fs
from research.walkforward.portfolio import _sortino
from data.db import get_db
from backtesting.data_loader import SPY

FFR = {2017: 0.0100, 2018: 0.0183, 2019: 0.0216, 2020: 0.0038, 2021: 0.0008,
       2022: 0.0168, 2023: 0.0503, 2024: 0.0510, 2025: 0.0433, 2026: 0.0390}
BLEND1256 = 0.6 * fs.LT + 0.4 * fs.ST
PUT_MARKUP = 0.20      # OTM put skew + slippage (you PAY more than VIX-flat)
CALL_HAIRCUT = 0.03    # for the sold-call comparison line
START = fs.START


def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(S, K, sig, T, r):
    if T <= 0 or sig <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)


def bs_put(S, K, sig, T, r):
    if T <= 0 or sig <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


def overlay_units(dates, matrix, vix):
    """Return dict of per-period unit returns (coverage=1) for put/call variants."""
    px = matrix[SPY].reindex(dates).astype(float)
    dt = pd.to_datetime([str(d) for d in dates])
    vser = vix.reindex([str(d) for d in dates]).astype(float).ffill()
    puts = {otm: {} for otm in (0.05, 0.10)}
    call2 = {}
    for i in range(len(dates) - 1):
        d0, d1 = dates[i], dates[i + 1]
        T = max((dt[i + 1] - dt[i]).days, 1) / 365.0
        sig = float(vser.iloc[i]) / 100.0
        r = FFR[int(str(d0)[:4])]
        spy_ret = float(px.iloc[i + 1] / px.iloc[i] - 1.0)
        for otm in puts:
            prem = bs_put(1.0, 1.0 - otm, sig, T, r) * (1 + PUT_MARKUP)
            payoff = max(-(spy_ret + otm), 0.0)          # pays when drop > otm
            puts[otm][d1] = payoff - prem
        cprem = bs_call(1.0, 1.02, sig, T, r) * (1 - CALL_HAIRCUT)
        call2[d1] = cprem - max(spy_ret - 0.02, 0.0)
    return {otm: pd.Series(s) for otm, s in puts.items()}, pd.Series(call2)


def stats(r, spy_r):
    r = r.dropna()
    eq = (1 + r).cumprod()
    cagr = eq.iloc[-1] ** (12 / len(r)) - 1
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(12)
    mdd = float((eq / eq.cummax() - 1).min())
    df = pd.concat([r.rename("p"), spy_r.rename("b")], axis=1).dropna()
    v = df["b"].var(ddof=1)
    beta = float(df.cov().loc["p", "b"] / v) if v > 0 else float("nan")
    return dict(cagr=cagr, sharpe=sharpe, sortino=_sortino(r, 12.0),
                calmar=cagr / abs(mdd) if mdd else float("nan"), mdd=mdd, beta=beta,
                final=START * float(eq.iloc[-1]))


def main():
    panel, matrix_raw, matrix, sectors, vix = load_env()
    run = pickle.load(PROD_CACHE.open("rb"))
    with get_db() as db:
        data = make_data(run, panel, matrix, sectors, vix, db)
    wts = fs.weights_for(data, fs.CFG)
    dates = [d for d in data.rebal_dates if d in wts and d in matrix.index]

    at_nav = pd.Series(fs.config_nav(dates, wts, matrix), index=dates)
    r_book = (at_nav / at_nav.shift(1) - 1.0).dropna()
    spy_at = pd.Series(fs.bench_nav(matrix, dates, SPY), index=dates).pct_change().dropna()
    qqq_at = pd.Series(fs.bench_nav(matrix, dates, "QQQ"), index=dates).pct_change().dropna()
    spy_gross = (matrix[SPY].reindex(dates).astype(float).pct_change()).dropna()

    puts, call2 = overlay_units(dates, matrix, vix)

    def add(unit, cov):
        o = (cov * unit).reindex(r_book.index).fillna(0.0) * (1 - BLEND1256)
        return r_book + o

    rows = [
        ("SPY", spy_at),
        ("QQQ", qqq_at),
        ("Book 1x (no overlay)", r_book),
        ("+ SELL 30% call (2% OTM)", add(call2, 0.30)),
        ("+ BUY puts 5% OTM, 30%", add(puts[0.05], 0.30)),
        ("+ BUY puts 5% OTM, 100%", add(puts[0.05], 1.00)),
        ("+ BUY puts 10% OTM, 100%", add(puts[0.10], 1.00)),
    ]
    for wlabel, lo in [("FULL 2017-2026", "2000"), ("2020-start", "2020-01-01")]:
        print(f"\n=== {wlabel}  (after-tax; monthly roll, 60/40 tax) ===")
        print(f"{'':28}{'CAGR':>7}{'Sharpe':>8}{'Sortino':>9}{'Calmar':>8}{'maxDD':>8}{'beta':>7}{'final $':>12}")
        sg = spy_gross[spy_gross.index.astype(str) >= lo]
        for name, r in rows:
            rr = r[r.index.astype(str) >= lo]
            m = stats(rr, sg)
            print(f"{name:28}{m['cagr']*100:6.1f}%{m['sharpe']:8.2f}{m['sortino']:9.2f}"
                  f"{m['calmar']:8.2f}{m['mdd']*100:7.1f}%{m['beta']:7.2f}{m['final']:>12,.0f}")

    # annual premium bleed of the bought-put lines (what you pay in a calm year)
    print("\nApprox premium *paid* per year (drag before any payoff):")
    for otm in (0.05, 0.10):
        prem_only = -puts[otm].copy()
        # reconstruct gross premium: unit = payoff - prem  ->  prem = payoff - unit
        # simpler: report mean monthly cost in calm months (payoff=0)
        calm = puts[otm][puts[otm] < 0]
        print(f"  BUY puts {int(otm*100)}% OTM, 100% cov: ~{-calm.mean()*12*100:.1f}%/yr in months it expires worthless")


if __name__ == "__main__":
    main()
