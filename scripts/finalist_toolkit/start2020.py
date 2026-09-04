"""After-tax results if we START in 2020 (drop the thin-training 2017-19 test years).
Fresh $100k invested at first rebal >= 2020-01. Compares full-history vs 2020-start
for Portfolio 1x / 1.5x / SPY / QQQ. Mark-to-liquidation after-tax; 1.5x levered
after-tax = 1.5*(after-tax book ret) - 0.5*borrow (FFR+1%).
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
OUT_PNG = "/home/manit/Desktop/fun_projects/mahajan_hedge_fund/output/ablation/start2020_curve.png"


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


def build(data, matrix, wts, dsub):
    nav = pd.Series(fs.config_nav(dsub, wts, matrix), index=dsub)
    r1 = (nav / nav.shift(1) - 1.0).dropna()
    spy = pd.Series(fs.bench_nav(matrix, dsub, SPY), index=dsub)
    qqq = pd.Series(fs.bench_nav(matrix, dsub, "QQQ"), index=dsub)
    return {
        "Portfolio 1x": r1,
        "Portfolio 1.25x": lever(r1, 1.25),
        "Portfolio 1.5x": lever(r1, 1.5),
        "SPY": (spy / spy.shift(1) - 1.0).dropna(),
        "QQQ": (qqq / qqq.shift(1) - 1.0).dropna(),
    }


def main():
    panel, matrix_raw, matrix, sectors, vix = load_env()
    run = pickle.load(PROD_CACHE.open("rb"))
    with get_db() as db:
        data = make_data(run, panel, matrix, sectors, vix, db)
    wts = fs.weights_for(data, fs.CFG)
    all_dates = [d for d in data.rebal_dates if d in wts and d in matrix.index]

    windows = {
        "FULL 2017-2026": all_dates,
        "2020-start": [d for d in all_dates if str(d) >= "2020-01-01"],
    }
    for wlabel, dsub in windows.items():
        ser = build(data, matrix, wts, dsub)
        spy_gross = (matrix[SPY].reindex(dsub).astype(float).pct_change()).dropna()
        yrs = (len(dsub) - 1) / 12.0
        print(f"\n=== {wlabel}  ({dsub[0]} -> {dsub[-1]}, {yrs:.1f}y)  AFTER-TAX ===")
        print(f"{'':16}{'CAGR':>7}{'Sharpe':>8}{'Sortino':>9}{'Calmar':>8}{'maxDD':>8}{'beta':>7}{'final $':>12}")
        for name, r in ser.items():
            m = stats(r, spy_gross)
            print(f"{name:16}{m['cagr']*100:6.1f}%{m['sharpe']:8.2f}{m['sortino']:9.2f}"
                  f"{m['calmar']:8.2f}{m['mdd']*100:7.1f}%{m['beta']:7.2f}{m['final']:>12,.0f}")
        if wlabel == "2020-start":
            fig, ax = plt.subplots(figsize=(10, 6))
            col = {"Portfolio 1x": "#2c6fbb", "Portfolio 1.25x": "#9467bd",
                   "Portfolio 1.5x": "#d62728", "SPY": "#777", "QQQ": "#2ca02c"}
            for name, r in ser.items():
                eq = START * (1 + r).cumprod()
                eq = pd.concat([pd.Series([START], index=[dsub[0]]), eq])
                ax.plot(pd.to_datetime(eq.index), eq.values,
                        label=f"{name}  (${eq.iloc[-1]:,.0f})", color=col[name],
                        lw=2 if "Portfolio" in name else 1.5)
            ax.set_yscale("log")
            ax.set_title("AFTER-TAX growth of $100k invested 2020 — locked config 1x / 1.5x vs SPY / QQQ")
            ax.set_ylabel("value ($, log)"); ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
            fig.tight_layout(); fig.savefig(OUT_PNG, dpi=130)
            print(f"chart -> {OUT_PNG}")


if __name__ == "__main__":
    main()
