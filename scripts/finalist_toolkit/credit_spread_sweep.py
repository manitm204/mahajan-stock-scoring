"""Sweep short-strike and width for a 30% call credit-spread overlay on the book.
Rows grouped by short-strike OTM; 'covered call' row = no long leg (infinite width).
After-tax (60/40), full period 2017 -> mid-2026. Premiums BS-priced with sigma=VIX,
2% per-leg slippage.
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
BLEND = 0.6 * fs.LT + 0.4 * fs.ST
SLIP = 0.02
COV = 0.30
START = fs.START


def _n(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(S, K, sig, T, r):
    if T <= 0 or sig <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    return S * _n(d1) - K * math.exp(-r * T) * _n(d1 - sig * math.sqrt(T))


def bs_delta(S, K, sig, T, r):
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    return _n(d1)


def precompute(dates, matrix, vix):
    px = matrix[SPY].reindex(dates).astype(float)
    dt = pd.to_datetime([str(d) for d in dates])
    vser = vix.reindex([str(d) for d in dates]).astype(float).ffill()
    rows = []
    for i in range(len(dates) - 1):
        T = max((dt[i + 1] - dt[i]).days, 1) / 365.0
        rows.append((dates[i + 1], T, float(vser.iloc[i]) / 100.0,
                     FFR[int(str(dates[i])[:4])], float(px.iloc[i + 1] / px.iloc[i] - 1.0)))
    return rows


def unit_series(rows, sell_otm, buy_otm):
    """buy_otm=None -> plain covered call (no long leg)."""
    out, deltas = {}, []
    for d1, T, sig, r, ret in rows:
        credit = bs_call(1.0, 1.0 + sell_otm, sig, T, r) * (1 - SLIP)
        pay = max(ret - sell_otm, 0.0)
        if buy_otm is not None:
            credit -= bs_call(1.0, 1.0 + buy_otm, sig, T, r) * (1 + SLIP)
            pay -= max(ret - buy_otm, 0.0)
        out[d1] = credit - pay
        deltas.append(bs_delta(1.0, 1.0 + sell_otm, sig, T, r))
    return pd.Series(out), float(np.mean(deltas))


def stats(r, spy_r):
    r = r.dropna(); eq = (1 + r).cumprod()
    cagr = eq.iloc[-1] ** (12 / len(r)) - 1
    sh = r.mean() / r.std(ddof=1) * np.sqrt(12)
    mdd = float((eq / eq.cummax() - 1).min())
    df = pd.concat([r.rename("p"), spy_r.rename("b")], axis=1).dropna()
    v = df["b"].var(ddof=1); beta = float(df.cov().loc["p", "b"] / v) if v > 0 else np.nan
    return cagr, sh, mdd, beta, START * float(eq.iloc[-1])


def main():
    panel, matrix_raw, matrix, sectors, vix = load_env()
    run = pickle.load(PROD_CACHE.open("rb"))
    with get_db() as db:
        data = make_data(run, panel, matrix, sectors, vix, db)
    wts = fs.weights_for(data, fs.CFG)
    dates = [d for d in data.rebal_dates if d in wts and d in matrix.index]
    at_nav = pd.Series(fs.config_nav(dates, wts, matrix), index=dates)
    r_book = (at_nav / at_nav.shift(1) - 1.0).dropna()
    spy_gross = (matrix[SPY].reindex(dates).astype(float).pct_change()).dropna()
    rows = precompute(dates, matrix, vix)

    def line(label, unit, credit_yr):
        o = (COV * unit).reindex(r_book.index).fillna(0.0) * (1 - BLEND)
        cagr, sh, mdd, beta, fin = stats(r_book + o, spy_gross)
        print(f"{label:34}{credit_yr*100:7.1f}%{cagr*100:7.1f}%{sh:8.2f}"
              f"{mdd*100:8.1f}%{beta:7.2f}{fin:>12,.0f}")

    cb, sb, mb, bb, fb = stats(r_book, spy_gross)
    print(f"\n30% call credit-spread sweep, after-tax FULL 2017-2026")
    print(f"{'':34}{'net':>8}{'':7}{'':8}{'':8}{'':7}")
    print(f"{'variant (short delta)':34}{'cr/yr':>7}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>8}{'beta':>7}{'final $':>12}")
    print(f"{'no overlay':34}{'-':>8}{cb*100:7.1f}%{sb:8.2f}{mb*100:8.1f}%{bb:7.2f}{fb:>12,.0f}")
    for sell in (0.01, 0.02, 0.03):
        ccu, dlt = unit_series(rows, sell, None)
        cc_credit = float(pd.Series([bs_call(1, 1 + sell, s, T, r) * (1 - SLIP)
                                     for _, T, s, r, _ in rows]).mean()) * 12 * COV
        print(f"  -- short {sell*100:.0f}% OTM (delta {dlt:.2f}) --")
        line(f"    covered call (no long leg)", ccu, cc_credit)
        for width in (0.02, 0.03, 0.05, 0.08):
            buy = sell + width
            u, _ = unit_series(rows, sell, buy)
            credit = float(pd.Series(
                [bs_call(1, 1 + sell, s, T, r) * (1 - SLIP)
                 - bs_call(1, 1 + buy, s, T, r) * (1 + SLIP) for _, T, s, r, _ in rows]
            ).mean()) * 12 * COV
            line(f"    spread {sell*100:.0f}/{buy*100:.0f} (width {width*100:.0f}%)", u, credit)


if __name__ == "__main__":
    main()
