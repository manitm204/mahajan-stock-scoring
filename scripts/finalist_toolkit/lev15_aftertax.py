"""AFTER-TAX 1x vs 1.5x-levered locked config vs SPY vs QQQ. Growth of $100k.
Levered after-tax return = 1.5*(after-tax book return) - 0.5*monthly borrow
(FFR+1%); exact in the HIFO model since tax is scale-invariant per $ of book
(interest conservatively NOT deducted). SPY/QQQ mark-to-liquidation after-tax.
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
from research.walkforward.portfolio import _sortino
from data.db import get_db
from backtesting.data_loader import SPY

MARGIN_SPREAD = 0.010
FFR = {2017: 0.0100, 2018: 0.0183, 2019: 0.0216, 2020: 0.0038, 2021: 0.0008,
       2022: 0.0168, 2023: 0.0503, 2024: 0.0510, 2025: 0.0433, 2026: 0.0390}
START = 100_000.0
ERA = "2021-12-31"
OUT_PNG = "/home/manit/Desktop/fun_projects/mahajan_hedge_fund/output/ablation/leverage_1p5x_aftertax_curve.png"


def lever(r, L):
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


def era_cagr(r, lo, hi):
    idx = r.index.astype(str)
    rr = r[(idx >= lo) & (idx <= hi)].dropna()
    return (1 + rr).prod() ** (12 / len(rr)) - 1 if len(rr) else float("nan")


def main():
    panel, matrix_raw, matrix, sectors, vix = load_env()
    run = pickle.load(PROD_CACHE.open("rb"))
    with get_db() as db:
        data = make_data(run, panel, matrix, sectors, vix, db)

    wts = fs.weights_for(data, fs.CFG)
    dates = [d for d in data.rebal_dates if d in wts and d in matrix.index]
    nav_at = fs.config_nav(dates, wts, matrix)              # after-tax, mark-to-liquidation
    nav_at = pd.Series(nav_at, index=dates)
    r1 = (nav_at / nav_at.shift(1) - 1.0).dropna()          # after-tax book return
    r15 = lever(r1, 1.5)

    spy_at = pd.Series(fs.bench_nav(matrix, dates, SPY), index=dates)
    qqq_at = pd.Series(fs.bench_nav(matrix, dates, "QQQ"), index=dates)
    spy_r = (spy_at / spy_at.shift(1) - 1.0).dropna()
    qqq_r = (qqq_at / qqq_at.shift(1) - 1.0).dropna()
    spy_gross = (matrix[SPY].reindex(dates).astype(float).pct_change()).dropna()

    series = {"Portfolio 1x": r1, "Portfolio 1.5x": r15, "SPY": spy_r, "QQQ": qqq_r}
    print("\nAFTER-TAX, $100k, locked config 1x vs 1.5x vs SPY/QQQ (margin @ FFR+1%)\n")
    print(f"{'':16}{'CAGR':>7}{'Sharpe':>8}{'Sortino':>9}{'Calmar':>8}{'maxDD':>8}{'beta':>7}{'final $':>12}"
          f"{'ERA1':>8}{'ERA2':>8}")
    for name, r in series.items():
        m = stats(r, spy_gross)
        e1, e2 = era_cagr(r, dates[0], ERA), era_cagr(r, "2022-01-01", dates[-1])
        print(f"{name:16}{m['cagr']*100:6.1f}%{m['sharpe']:8.2f}{m['sortino']:9.2f}"
              f"{m['calmar']:8.2f}{m['mdd']*100:7.1f}%{m['beta']:7.2f}{m['final']:>12,.0f}"
              f"{e1*100:7.1f}%{e2*100:7.1f}%")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"Portfolio 1x": "#2c6fbb", "Portfolio 1.5x": "#d62728",
              "SPY": "#777777", "QQQ": "#2ca02c"}
    for name, r in series.items():
        eq = START * (1 + r).cumprod()
        eq = pd.concat([pd.Series([START], index=[dates[0]]), eq])
        ax.plot(pd.to_datetime(eq.index), eq.values, label=f"{name}  (${eq.iloc[-1]:,.0f})",
                color=colors[name], lw=2 if "Portfolio" in name else 1.5)
    ax.set_yscale("log")
    ax.set_title("AFTER-TAX growth of $100k — locked config 1x vs 1.5x levered vs SPY / QQQ")
    ax.set_ylabel("value ($, log)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    print(f"\nchart -> {OUT_PNG}")


if __name__ == "__main__":
    main()
