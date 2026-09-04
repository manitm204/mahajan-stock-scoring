"""Leverage / Kelly sweep for the LOCKED config (top25/cap5/cap_match/VIX, 13mo x4
sleeves). Margin-financed at FFR+1%. Pre-tax growth-optimal (full-Kelly) leverage is
where CAGR peaks; we mark full-Kelly, half-Kelly and the 1.25x back-pocket level.
Chart: CAGR vs leverage + maxDD vs leverage, SPY/QQQ as reference. Pre-tax.
"""
from __future__ import annotations
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hz_experiment import load_env, make_data, PROD_CACHE
import final_stats as fs
from data.db import get_db
from backtesting.data_loader import SPY

MARGIN_SPREAD = 0.010
FFR = {2017: 0.0100, 2018: 0.0183, 2019: 0.0216, 2020: 0.0038, 2021: 0.0008,
       2022: 0.0168, 2023: 0.0503, 2024: 0.0510, 2025: 0.0433, 2026: 0.0390}
OUT_PNG = "/home/manit/Desktop/fun_projects/mahajan_hedge_fund/output/ablation/leverage_kelly.png"
OUT_CSV = "/home/manit/Desktop/fun_projects/mahajan_hedge_fund/output/ablation/leverage_kelly.csv"


def config_pretax_nav(dates, wts, matrix):
    off = fs.HOLD // fs.SLEEVES
    per = fs.START / fs.SLEEVES
    pt = np.zeros(len(dates) - 1)
    for k in range(fs.SLEEVES):
        p, _ = fs.sleeve_nav(dates, wts, matrix, k * off, per)
        pt += p
    return pd.Series(np.concatenate([[fs.START], pt]), index=dates)


def lever(r: pd.Series, L: float) -> pd.Series:
    ffr = pd.Series([FFR[int(str(d)[:4])] for d in r.index], index=r.index)
    return L * r - (L - 1.0) * (ffr + MARGIN_SPREAD) / 12.0


def stats(r: pd.Series):
    eq = (1 + r).cumprod()
    yrs = len(r) / 12.0
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(12)
    mdd = float((eq / eq.cummax() - 1).min())
    return cagr, sharpe, mdd


def main():
    panel, matrix_raw, matrix, sectors, vix = load_env()
    run = pickle.load(PROD_CACHE.open("rb"))
    with get_db() as db:
        data = make_data(run, panel, matrix, sectors, vix, db)

    wts = fs.weights_for(data, fs.CFG)                       # locked config, VIX on
    dates = [d for d in data.rebal_dates if d in wts and d in matrix.index]
    nav = config_pretax_nav(dates, wts, matrix)
    r = (nav / nav.shift(1) - 1.0).dropna()                  # unlevered monthly pre-tax

    # analytical full-Kelly: L* = (mean(r) - mean_borrow_monthly) / var(r)
    ffr_m = pd.Series([FFR[int(str(d)[:4])] for d in r.index], index=r.index)
    rf_m = float((ffr_m + MARGIN_SPREAD).mean() / 12.0)
    kelly_analytic = (r.mean() - rf_m) / r.var(ddof=1)

    grid = np.round(np.arange(1.0, 4.01, 0.05), 2)
    rows = []
    for L in grid:
        c, s, dd = stats(lever(r, L))
        rows.append((L, c, s, dd))
    df = pd.DataFrame(rows, columns=["leverage", "cagr", "sharpe", "max_dd"])
    df.to_csv(OUT_CSV, index=False)

    kelly_emp = float(df.loc[df["cagr"].idxmax(), "leverage"])   # CAGR-peak = full Kelly
    half_kelly = round(kelly_emp / 2, 2)

    # SPY / QQQ unlevered pre-tax reference CAGR over the same window
    def bench_cagr(tkr):
        p = matrix[tkr].reindex(dates).astype(float)
        return (p.iloc[-1] / p.iloc[0]) ** (12 / (len(p) - 1)) - 1
    spy_c, qqq_c = bench_cagr(SPY), bench_cagr("QQQ")

    def at(L):
        return df.iloc[(df["leverage"] - L).abs().idxmin()]

    print(f"\nLevered LOCKED config (top25/cap5/cap_match/VIX, 13mo x4), margin @ FFR+1%. PRE-TAX.")
    print(f"Full-Kelly: analytic L*={kelly_analytic:.2f}, empirical CAGR-peak L={kelly_emp:.2f}")
    print(f"SPY {spy_c*100:.1f}%  QQQ {qqq_c*100:.1f}% (unlevered)\n")
    print(f"{'lev':>5}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>9}   note")
    marks = {1.0: "unlevered", 1.25: "back-pocket", 1.5: "", 2.0: "",
             half_kelly: "HALF-Kelly", kelly_emp: "FULL-Kelly (peak)"}
    for L in sorted(set([1.0, 1.25, 1.5, 2.0, 2.5, 3.0, half_kelly, kelly_emp])):
        d = at(L)
        print(f"{d['leverage']:5.2f}{d['cagr']*100:7.1f}%{d['sharpe']:8.2f}"
              f"{d['max_dd']*100:8.1f}%   {marks.get(round(L,2),'')}")

    # ---- chart ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    ax1.plot(df["leverage"], df["cagr"] * 100, color="#2c6fbb", lw=2, label="portfolio CAGR")
    ax1.axhline(spy_c * 100, color="#999", ls=":", lw=1.2, label=f"SPY {spy_c*100:.1f}%")
    ax1.axhline(qqq_c * 100, color="#c77", ls=":", lw=1.2, label=f"QQQ {qqq_c*100:.1f}%")
    for L, col, lab in [(kelly_emp, "#d62728", f"full Kelly {kelly_emp:.2f}x"),
                        (half_kelly, "#ff7f0e", f"half Kelly {half_kelly:.2f}x"),
                        (1.25, "#2ca02c", "1.25x back-pocket")]:
        for ax in (ax1, ax2):
            ax.axvline(L, color=col, ls="--", lw=1.3)
        ax1.text(L, ax1.get_ylim()[1], lab, rotation=90, va="top", ha="right",
                 fontsize=8, color=col)
    pk = df["cagr"].max() * 100
    ax1.plot(kelly_emp, pk, "o", color="#d62728")
    ax1.set_ylabel("CAGR (%)")
    ax1.set_title("Levered locked config — pre-tax CAGR & drawdown vs leverage "
                  "(margin @ FFR+1%)", fontsize=11)
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(alpha=0.3)

    ax2.plot(df["leverage"], df["max_dd"] * 100, color="#8c564b", lw=2)
    ax2.set_ylabel("max drawdown (%)")
    ax2.set_xlabel("leverage")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    print(f"\nchart -> {OUT_PNG}\ncsv   -> {OUT_CSV}")


if __name__ == "__main__":
    main()
