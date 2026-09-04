"""R^2 of the locked portfolio vs SPY and vs QQQ (pre-tax monthly returns).
R^2 = corr(portfolio, benchmark)^2. Also reports correlation and beta, full period + 2020+.
"""
from __future__ import annotations
import pickle
import numpy as np
import pandas as pd

from hz_experiment import load_env, make_data, PROD_CACHE
import final_stats as fs
from data.db import get_db
from backtesting.data_loader import SPY


def config_pretax_nav(dates, wts, matrix):
    off = fs.HOLD // fs.SLEEVES
    per = fs.START / fs.SLEEVES
    pt = np.zeros(len(dates) - 1)
    for k in range(fs.SLEEVES):
        p, _ = fs.sleeve_nav(dates, wts, matrix, k * off, per)
        pt += p
    return pd.Series(np.concatenate([[fs.START], pt]), index=dates)


def rr(port, bench):
    df = pd.concat([port.rename("p"), bench.rename("b")], axis=1).dropna()
    corr = df["p"].corr(df["b"])
    beta = df.cov().loc["p", "b"] / df["b"].var(ddof=1)
    return corr, corr ** 2, beta


def main():
    panel, matrix_raw, matrix, sectors, vix = load_env()
    run = pickle.load(PROD_CACHE.open("rb"))
    with get_db() as db:
        data = make_data(run, panel, matrix, sectors, vix, db)
    wts = fs.weights_for(data, fs.CFG)
    dates = [d for d in data.rebal_dates if d in wts and d in matrix.index]
    nav = config_pretax_nav(dates, wts, matrix)
    r = (nav / nav.shift(1) - 1.0).dropna()
    spy = (matrix[SPY].reindex(dates).astype(float).pct_change()).dropna()
    qqq = (matrix["QQQ"].reindex(dates).astype(float).pct_change()).dropna()

    for label, lo in [("FULL 2017-2026", "2000"), ("2020-start", "2020-01-01")]:
        idx = r.index.astype(str)
        rr_ = r[idx >= lo]
        print(f"\n=== {label} (pre-tax monthly) ===")
        for name, b in [("SPY", spy), ("QQQ", qqq)]:
            corr, r2, beta = rr(rr_, b)
            print(f"  vs {name}:  corr {corr:.3f}   R^2 {r2:.3f}   beta {beta:.2f}")


if __name__ == "__main__":
    main()
