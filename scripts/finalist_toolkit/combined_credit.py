"""70/30 = 70% stock book + 30% cash sleeve running SPY 0.15-delta $5 call credit
spreads (30 DTE, hold to expiry, risk 10% of the sleeve per trade; idle cash earns FFR).
Hold-to-expiry settles at the next month-end, so this needs only the monthly series.
Blend 70/30 monthly. Compare after-tax to 100% book, SPY, QQQ. Sleeve P&L taxed ST
(SPY) and also 60/40 (XSP) for contrast; idle-cash interest always ST.
"""
from __future__ import annotations
import math, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hz_experiment import load_env, make_data, PROD_CACHE
import final_stats as fs
from research.walkforward.portfolio import _sortino
from data.db import get_db
from backtesting.data_loader import SPY

FFR = {2017: 0.0100, 2018: 0.0183, 2019: 0.0216, 2020: 0.0038, 2021: 0.0008,
       2022: 0.0168, 2023: 0.0503, 2024: 0.0510, 2025: 0.0433, 2026: 0.0390}
ST, BLEND = 0.32, 0.218          # short-term (SPY) vs 60/40 (XSP)
SLIP = 0.03
WIDTH = 5.0
TGT = 0.15
F = 0.10                          # fraction of sleeve risked per trade
# --- calibrated to real SPY 30-DTE quotes 2026-07-21 (see
#     output/ablation/options_calibration/). Real ATM IV is ~0.80*VIX (VIX
#     overstates ATM via its skew-loading); real call skew ~0.68 (was 0.9). ---
ATM_VIX = 0.80                    # real 30-DTE ATM IV / VIX
SKEW = 0.68                       # call skew: IV drops ~0.68 vol-pt per 1% OTM
IVFLOOR = 0.06
MARGIN_SPREAD = 0.010
START = fs.START


def _n(x): return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
def _d1(S, K, s, T, r): return (math.log(S / K) + (r + 0.5 * s * s) * T) / (s * math.sqrt(T))
def call_px(S, K, s, T, r):
    if T <= 0: return max(S - K, 0.0)
    d1 = _d1(S, K, s, T, r); return S * _n(d1) - K * math.exp(-r * T) * _n(d1 - s * math.sqrt(T))
def iv_call(S, K, atm):
    """Call-side skew: OTM calls trade below ATM/VIX."""
    return max(atm - SKEW * max(K / S - 1.0, 0.0), IVFLOOR)
def find_short(S, s, T, r, tgt):
    best, be = S, 9.9
    for otm in np.arange(0.0, 0.20, 0.0015):
        K = S * (1 + otm)
        if abs(_n(_d1(S, K, s, T, r)) - tgt) < be:
            be, best = abs(_n(_d1(S, K, s, T, r)) - tgt), K
    return best


def sleeve_returns(dates, matrix, vix):
    """Monthly sleeve pre-tax return = FFR interest + F*roi from one call credit spread."""
    px = matrix[SPY].reindex(dates).astype(float)
    dt = pd.to_datetime([str(d) for d in dates])
    vser = vix.reindex([str(d) for d in dates]).astype(float).ffill()
    out_int, out_spread = {}, {}
    for i in range(len(dates) - 1):
        T = max((dt[i + 1] - dt[i]).days, 1) / 365.0
        S0, S1 = float(px.iloc[i]), float(px.iloc[i + 1])
        sig = float(vser.iloc[i]) / 100.0 * ATM_VIX      # calibrated ATM IV
        r = FFR[int(str(dates[i])[:4])]
        Ks = find_short(S0, sig, T, r, TGT); Kl = Ks + WIDTH
        credit = (call_px(S0, Ks, iv_call(S0, Ks, sig), T, r)
                  - call_px(S0, Kl, iv_call(S0, Kl, sig), T, r)) - 2 * SLIP
        if credit <= 0.02:
            out_int[dates[i + 1]] = r * T; out_spread[dates[i + 1]] = 0.0; continue
        maxloss = WIDTH - credit
        intr = min(max(S1 - Ks, 0.0), WIDTH)
        roi = (credit - intr) / maxloss
        out_int[dates[i + 1]] = r * T           # idle cash interest for the month
        out_spread[dates[i + 1]] = F * roi       # spread P&L on the sleeve
    return pd.Series(out_int), pd.Series(out_spread)


def stats(r, spy_r):
    r = r.dropna(); eq = (1 + r).cumprod()
    cagr = eq.iloc[-1] ** (12 / len(r)) - 1
    sh = r.mean() / r.std(ddof=1) * np.sqrt(12)
    mdd = float((eq / eq.cummax() - 1).min())
    df = pd.concat([r.rename("p"), spy_r.rename("b")], axis=1).dropna()
    v = df["b"].var(ddof=1); beta = float(df.cov().loc["p", "b"] / v) if v > 0 else np.nan
    return dict(cagr=cagr, sharpe=sh, sortino=_sortino(r, 12.0),
               calmar=cagr / abs(mdd) if mdd else np.nan, mdd=mdd, beta=beta,
               final=START * float(eq.iloc[-1]))


def main():
    panel, mr, matrix, sectors, vix = load_env()
    run = pickle.load(PROD_CACHE.open("rb"))
    with get_db() as db:
        data = make_data(run, panel, matrix, sectors, vix, db)
    wts = fs.weights_for(data, fs.CFG)
    dates = [d for d in data.rebal_dates if d in wts and d in matrix.index]

    r_book = (pd.Series(fs.config_nav(dates, wts, matrix), index=dates)
              .pct_change().dropna())
    spy_at = pd.Series(fs.bench_nav(matrix, dates, SPY), index=dates).pct_change().dropna()
    qqq_at = pd.Series(fs.bench_nav(matrix, dates, "QQQ"), index=dates).pct_change().dropna()
    spy_gross = (matrix[SPY].reindex(dates).astype(float).pct_change()).dropna()

    def lever(r, L):
        ffr = pd.Series([FFR[int(str(d)[:4])] for d in r.index], index=r.index)
        return L * r - (L - 1.0) * (ffr + MARGIN_SPREAD) / 12.0

    book_125 = lever(r_book, 1.25)

    intr, spr = sleeve_returns(dates, matrix, vix)
    intr = intr.reindex(r_book.index).fillna(0.0); spr = spr.reindex(r_book.index).fillna(0.0)
    sleeve_st = intr * (1 - ST) + spr * (1 - ST)
    sleeve_xsp = intr * (1 - ST) + spr * (1 - BLEND)

    rows = [
        ("Book 100% (1x)", r_book),
        ("Book 100% (1.25x)", book_125),
        ("70/30  book1x + SPY spreads", 0.7 * r_book + 0.3 * sleeve_st),
        ("70/30  book1.25x + SPY spreads", 0.7 * book_125 + 0.3 * sleeve_st),
        ("70/30  book1.25x + XSP spreads", 0.7 * book_125 + 0.3 * sleeve_xsp),
        ("[sleeve alone, ST]", sleeve_st),
        ("SPY", spy_at),
        ("QQQ", qqq_at),
    ]
    for wlabel, lo in [("FULL 2017-2026", "2000"), ("2020-start", "2020-01-01")]:
        print(f"\n=== {wlabel}  (after-tax) ===")
        print(f"{'':30}{'CAGR':>7}{'Sharpe':>8}{'Sortino':>9}{'Calmar':>8}{'maxDD':>8}{'beta':>7}{'final $':>12}")
        sg = spy_gross[spy_gross.index.astype(str) >= lo]
        for name, r in rows:
            rr = r[r.index.astype(str) >= lo]
            m = stats(rr, sg)
            print(f"{name:30}{m['cagr']*100:6.1f}%{m['sharpe']:8.2f}{m['sortino']:9.2f}"
                  f"{m['calmar']:8.2f}{m['mdd']*100:7.1f}%{m['beta']:7.2f}{m['final']:>12,.0f}")

    # ---- equity-curve chart, after-tax growth of $100k, full period ----
    plot = {
        "Book 1x  (14.1%)": (dict(rows)["Book 100% (1x)"], "#2c6fbb", 2.0),
        "Book 1.25x  (16.4%)": (dict(rows)["Book 100% (1.25x)"], "#9467bd", 1.6),
        "70/30 book-1x + spreads  (10.8%)": (dict(rows)["70/30  book1x + SPY spreads"], "#17becf", 2.0),
        "70/30 book-1.25x + spreads  (12.5%)": (dict(rows)["70/30  book1.25x + SPY spreads"], "#d62728", 2.4),
        "SPY  (12.4%)": (spy_at, "#777777", 1.4),
        "QQQ  (19.4%)": (qqq_at, "#2ca02c", 1.4),
    }
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for label, (r, col, lw) in plot.items():
        eq = START * (1 + r).cumprod()
        eq = pd.concat([pd.Series([START], index=[r.index[0]]), eq])
        ax.plot(pd.to_datetime([str(d) for d in eq.index]), eq.values,
                label=f"{label}  →  ${eq.iloc[-1]:,.0f}", color=col, lw=lw)
    ax.set_yscale("log")
    ax.set_title("After-tax growth of $100k, 2017 → mid-2026\n"
                 "70/30 book + 0.15Δ $5 SPY call credit-spread sleeve (skew-corrected) vs book / SPY / QQQ")
    ax.set_ylabel("value ($, log scale)")
    ax.legend(fontsize=8.5, loc="upper left"); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = "/home/manit/Desktop/fun_projects/mahajan_hedge_fund/output/ablation/credit_spread_7030.png"
    fig.savefig(out, dpi=130)
    print(f"\nchart -> {out}")


if __name__ == "__main__":
    main()
