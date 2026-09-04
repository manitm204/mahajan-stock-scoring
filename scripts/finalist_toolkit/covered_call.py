"""30% XSP monthly covered-call overwrite on the locked book.

Overlay mechanic, per month t (period d_{t-1} -> d_t):
  - sell a 1-month SPX/XSP call struck `otm` above spot, on `cov` of book notional.
  - premium priced Black-Scholes with sigma = VIX at d_{t-1} (real IV proxy),
    T = actual days/365, r = FFR; haircut 3% for bid-ask/slippage.
  - at expiry the short call pays out max(spy_ret - otm, 0) of notional.
  - overlay unit return = premium - payoff ; contribution = cov * unit.
Tax: XSP is Section 1256 -> 60/40 blended = 0.6*LT + 0.4*ST, realized monthly.
Book after-tax return comes from the mark-to-liquidation sleeve sim (final_stats).
Leverage uses the scale-invariant after-tax formula; on an Lx book the 30%
overwrite covers 30% of the *gross* equity, so effective coverage = 0.30*L.
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

MARGIN_SPREAD = 0.010
FFR = {2017: 0.0100, 2018: 0.0183, 2019: 0.0216, 2020: 0.0038, 2021: 0.0008,
       2022: 0.0168, 2023: 0.0503, 2024: 0.0510, 2025: 0.0433, 2026: 0.0390}
BLEND1256 = 0.6 * fs.LT + 0.4 * fs.ST      # = 0.218
COV = 0.30                                  # 30% notional overwrite
PREM_HAIRCUT = 0.03                         # bid-ask / slippage on collected premium
START = fs.START


def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(S, K, sigma, T, r):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)


def overlay_unit_returns(dates, matrix, vix, otm):
    """Pre-tax overlay unit return per period (indexed at dates[1:]), coverage=1."""
    px = matrix[SPY].reindex(dates).astype(float)
    dt = pd.to_datetime([str(d) for d in dates])
    vser = vix.reindex([str(d) for d in dates]).astype(float).ffill()
    out = {}
    prem_log = []
    for i in range(len(dates) - 1):
        d0, d1 = dates[i], dates[i + 1]
        T = max((dt[i + 1] - dt[i]).days, 1) / 365.0
        sigma = float(vser.iloc[i]) / 100.0
        r = FFR[int(str(d0)[:4])]
        prem = bs_call(1.0, 1.0 + otm, sigma, T, r) * (1 - PREM_HAIRCUT)
        spy_ret = float(px.iloc[i + 1] / px.iloc[i] - 1.0)
        payoff = max(spy_ret - otm, 0.0)
        out[d1] = prem - payoff
        prem_log.append(prem)
    return pd.Series(out), float(np.mean(prem_log))


def lever_ret(r, L):
    ffr = pd.Series([FFR[int(str(d)[:4])] for d in r.index], index=r.index)
    return L * r - (L - 1.0) * (ffr + MARGIN_SPREAD) / 12.0


def stats(r, spy_r):
    r = r.dropna()
    eq = (1 + r).cumprod()
    cagr = eq.iloc[-1] ** (12 / len(r)) - 1
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(12)
    mdd = float((eq / eq.cummax() - 1).min())
    df = pd.concat([r.rename("p"), spy_r.rename("b")], axis=1).dropna()
    var = df["b"].var(ddof=1)
    beta = float(df.cov().loc["p", "b"] / var) if var > 0 else float("nan")
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

    # after-tax book return stream (mark-to-liquidation, sleeve-combined)
    at_nav = pd.Series(fs.config_nav(dates, wts, matrix), index=dates)
    r_book = (at_nav / at_nav.shift(1) - 1.0).dropna()
    spy_gross = (matrix[SPY].reindex(dates).astype(float).pct_change()).dropna()  # for beta
    spy_at = pd.Series(fs.bench_nav(matrix, dates, SPY), index=dates).pct_change().dropna()
    qqq_at = pd.Series(fs.bench_nav(matrix, dates, "QQQ"), index=dates).pct_change().dropna()

    # ---- strike sensitivity on the 1x book, full period (after-tax) ----
    print("\n== Strike sensitivity: 30% overwrite on 1x book, FULL period (after-tax) ==")
    print(f"{'strike':>10}{'prem/yr':>9}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>8}{'beta':>7}")
    base = stats(r_book, spy_gross)
    print(f"{'no overlay':>10}{'-':>9}{base['cagr']*100:6.1f}%{base['sharpe']:8.2f}"
          f"{base['mdd']*100:7.1f}%{base['beta']:7.2f}")
    overlays = {}
    for otm in (0.00, 0.02, 0.03, 0.05):
        unit, mprem = overlay_unit_returns(dates, matrix, vix, otm)
        overlays[otm] = unit
        o_at = (COV * unit).reindex(r_book.index).fillna(0.0) * (1 - BLEND1256)
        r_c = r_book + o_at
        m = stats(r_c, spy_gross)
        tag = "ATM" if otm == 0 else f"{otm*100:.0f}% OTM"
        print(f"{tag:>10}{COV*mprem*12*100:8.1f}%{m['cagr']*100:6.1f}%{m['sharpe']:8.2f}"
              f"{m['mdd']*100:7.1f}%{m['beta']:7.2f}")

    OTM = 0.02
    unit = overlays[OTM]

    def combined(L, overwrite):
        r = lever_ret(r_book, L)
        if overwrite:
            cov_eff = COV * L
            o_at = (cov_eff * unit).reindex(r.index).fillna(0.0) * (1 - BLEND1256)
            r = r + o_at
        return r

    rows = [
        ("SPY", spy_at),
        ("QQQ", qqq_at),
        ("Book 1x", combined(1.0, False)),
        ("Book 1x + 30% XSP", combined(1.0, True)),
        ("Book 1.25x", combined(1.25, False)),
        ("Book 1.25x + 30% XSP", combined(1.25, True)),
    ]
    windows = [("FULL 2017-2026", "2000"), ("2020-start", "2020-01-01")]
    for wlabel, lo in windows:
        print(f"\n=== {wlabel}  (after-tax; XSP = 2% OTM monthly, 60/40 tax) ===")
        print(f"{'':22}{'CAGR':>7}{'Sharpe':>8}{'Sortino':>9}{'Calmar':>8}{'maxDD':>8}{'beta':>7}{'final $':>12}")
        sg = spy_gross[spy_gross.index.astype(str) >= lo]
        for name, r in rows:
            rr = r[r.index.astype(str) >= lo]
            m = stats(rr, sg)
            print(f"{name:22}{m['cagr']*100:6.1f}%{m['sharpe']:8.2f}{m['sortino']:9.2f}"
                  f"{m['calmar']:8.2f}{m['mdd']*100:7.1f}%{m['beta']:7.2f}{m['final']:>12,.0f}")


if __name__ == "__main__":
    main()
