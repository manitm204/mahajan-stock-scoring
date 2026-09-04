"""Two capital-efficient call-selling variants vs the plain covered call:
  (A) CALL CREDIT SPREAD overlay  - sell 2% OTM, buy 5% OTM (defined risk, low margin).
      Tested as a 30% overlay on the book, same as the covered call, so it's comparable.
  (B) POOR MAN'S COVERED CALL      - buy a 1y ~20%-ITM LEAPS as a stock substitute, sell
      monthly 2% OTM calls against it. Standalone, self-financing sim (NOT additive to the
      factor book - it's levered SPY beta). Reported next to the book so the trade is visible.
Premiums BS-priced with sigma = VIX; 60/40 (Sec 1256) tax; 2% slippage per option leg.
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
BLEND = 0.6 * fs.LT + 0.4 * fs.ST      # 60/40 = 0.218
SLIP = 0.02                             # per-leg slippage
START = fs.START


def _n(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(S, K, sig, T, r):
    if T <= 0 or sig <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    return S * _n(d1) - K * math.exp(-r * T) * _n(d1 - sig * math.sqrt(T))


def bs_delta(S, K, sig, T, r):
    if T <= 0 or sig <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    return _n(d1)


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


def overlay_units(dates, matrix, vix):
    """Per-period unit returns (coverage=1) for covered call & call credit spread."""
    px = matrix[SPY].reindex(dates).astype(float)
    dt = pd.to_datetime([str(d) for d in dates])
    vser = vix.reindex([str(d) for d in dates]).astype(float).ffill()
    cc, cs = {}, {}
    for i in range(len(dates) - 1):
        d0, d1 = dates[i], dates[i + 1]
        T = max((dt[i + 1] - dt[i]).days, 1) / 365.0
        sig = float(vser.iloc[i]) / 100.0
        r = FFR[int(str(d0)[:4])]
        ret = float(px.iloc[i + 1] / px.iloc[i] - 1.0)
        sell2 = bs_call(1.0, 1.02, sig, T, r) * (1 - SLIP)
        buy5 = bs_call(1.0, 1.05, sig, T, r) * (1 + SLIP)
        cc[d1] = sell2 - max(ret - 0.02, 0.0)                       # covered call
        cs[d1] = (sell2 - buy5) - (max(ret - 0.02, 0.0) - max(ret - 0.05, 0.0))  # credit spread
    return pd.Series(cc), pd.Series(cs)


def pmcc_returns(dates, matrix, vix):
    """Standalone poor-man's covered call: 1y 20%-ITM LEAPS + monthly 2% OTM short call.
    All-NAV-into-LEAPS at each annual roll; sell one short call per LEAPS unit."""
    px = matrix[SPY].reindex(dates).astype(float).values
    dt = pd.to_datetime([str(d) for d in dates])
    vser = vix.reindex([str(d) for d in dates]).astype(float).ffill().values / 100.0
    nav = START
    navs = {dates[0]: START}
    # open first LEAPS
    S = px[0]; sig = vser[0]; r = FFR[int(str(dates[0])[:4])]
    K_L = 0.80 * S; T_L = 1.0
    lp = bs_call(S, K_L, sig, T_L, r)
    units = nav / lp; cash = 0.0
    lev_log, prem_log = [], []
    for i in range(len(dates) - 1):
        S0, S1 = px[i], px[i + 1]
        sig0, sig1 = vser[i], vser[i + 1]
        r = FFR[int(str(dates[i])[:4])]
        h = max((dt[i + 1] - dt[i]).days, 1) / 365.0
        # sell 2% OTM monthly call, one per LEAPS unit
        Ks = S0 * 1.02
        prem = bs_call(S0, Ks, sig0, h, r) * (1 - SLIP)
        cash += units * prem
        prem_log.append(units * prem / nav)
        # realize the month
        payoff = max(S1 - Ks, 0.0)
        cash -= units * payoff
        T_L -= h
        lp1 = bs_call(S1, K_L, sig1, max(T_L, 1 / 365), r)
        lev_log.append(units * bs_delta(S0, K_L, sig0, T_L + h, r) * S0 / nav)
        nav = units * lp1 + cash
        navs[dates[i + 1]] = nav
        # annual roll of the LEAPS
        if T_L < 0.25:
            K_L = 0.80 * S1; T_L = 1.0
            lp = bs_call(S1, K_L, sig1, T_L, r)
            units = nav / lp; cash = 0.0
    ser = pd.Series(navs)
    r = (ser / ser.shift(1) - 1.0).dropna()
    return r, float(np.mean(lev_log)), float(np.mean(prem_log) * 12)


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

    cc, cs = overlay_units(dates, matrix, vix)

    def add(unit, cov):
        o = (cov * unit).reindex(r_book.index).fillna(0.0) * (1 - BLEND)
        return r_book + o

    pmcc_r, pmcc_lev, pmcc_prem = pmcc_returns(dates, matrix, vix)
    pmcc_at = pmcc_r * (1 - BLEND)  # illustrative 60/40 on the standalone P&L

    print(f"\n[PMCC diagnostics] avg effective leverage ~{pmcc_lev:.1f}x, "
          f"premium collected ~{pmcc_prem*100:.1f}%/yr of NAV")

    rows = [
        ("SPY", spy_at),
        ("QQQ", qqq_at),
        ("Book 1x (no overlay)", r_book),
        ("+ SELL 30% call (2% OTM)", add(cc, 0.30)),
        ("+ 30% call CREDIT SPREAD 2/5", add(cs, 0.30)),
        ("PMCC standalone (levered)", pmcc_at),
    ]
    for wlabel, lo in [("FULL 2017-2026", "2000"), ("2020-start", "2020-01-01")]:
        print(f"\n=== {wlabel}  (after-tax; monthly roll, 60/40 tax) ===")
        print(f"{'':30}{'CAGR':>7}{'Sharpe':>8}{'Sortino':>9}{'Calmar':>8}{'maxDD':>8}{'beta':>7}{'final $':>12}")
        sg = spy_gross[spy_gross.index.astype(str) >= lo]
        for name, r in rows:
            rr = r[r.index.astype(str) >= lo]
            m = stats(rr, sg)
            print(f"{name:30}{m['cagr']*100:6.1f}%{m['sharpe']:8.2f}{m['sortino']:9.2f}"
                  f"{m['calmar']:8.2f}{m['mdd']*100:7.1f}%{m['beta']:7.2f}{m['final']:>12,.0f}")


if __name__ == "__main__":
    main()
