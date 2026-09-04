"""Run the user's 4 hand-picked constructions through the post-tax sim
(13mo x4 sleeves, HIFO, ST32/LT15), vs SPY/QQQ. Stats: CAGR, Sharpe, Sortino,
Calmar, maxDD, beta (after-tax monthly ret vs SPY), effN (mean 1/sum w^2).
VIX tilt OFF (matches the sweep these were chosen from).
"""
from __future__ import annotations
import pickle

import numpy as np
import pandas as pd

from hz_experiment import load_env, make_data, PROD_CACHE
import final_stats as fs
from research.ablation.engine import AblationConfig
from data.db import get_db
from backtesting.data_loader import SPY

ERA = "2021-12-31"
PICKS = [
    ("top10 cap/cap_match/none",      AblationConfig(name="t10", top_pct=0.10, weighting="cap",  sector="cap_match", exclusion="none",    vix_tilt=False)),
    ("top25 cap/cap_match/none",      AblationConfig(name="t25", top_pct=0.25, weighting="cap",  sector="cap_match", exclusion="none",    vix_tilt=False)),
    ("top50 cap5/cap_match/two_p10",  AblationConfig(name="t50", top_pct=0.50, weighting="cap5", sector="cap_match", exclusion="two_p10", vix_tilt=False)),
    ("top75 cap5/none/soft_p10",      AblationConfig(name="t75", top_pct=0.75, weighting="cap5", sector="none",      exclusion="soft_p10",vix_tilt=False)),
]


def beta_vs_spy(nav, dates, matrix):
    r = pd.Series(np.diff(nav) / nav[:-1], index=pd.Index(dates[1:]))
    spy = matrix[SPY].reindex(dates)
    sr = (spy / spy.shift(1) - 1.0)
    sr.index = dates
    sr = sr.reindex(r.index)
    df = pd.concat([r.rename("p"), sr.rename("b")], axis=1).dropna()
    var = df["b"].var(ddof=1)
    return float(df.cov().loc["p", "b"] / var) if var > 0 else float("nan")


def eff_n(data, cfg):
    wts = fs.weights_for(data, cfg)
    es = [1.0 / float((w ** 2).sum()) for w in wts.values() if len(w)]
    return float(np.mean(es)), wts


def eras(nav, dates):
    ds = np.array(dates)
    def win(lo, hi):
        j = np.where((ds >= lo) & (ds <= hi))[0]
        return nav[max(j[0]-1, 0): j[-1]+1]
    return fs.stats(win(dates[0], ERA)), fs.stats(win("2022-01-01", dates[-1]))


def main():
    panel, matrix_raw, matrix, sectors, vix = load_env()
    run = pickle.load(PROD_CACHE.open("rb"))
    with get_db() as db:
        data = make_data(run, panel, matrix, sectors, vix, db)

    rows = []
    for label, cfg in PICKS:
        en, wts = eff_n(data, cfg)
        dates = [d for d in data.rebal_dates if d in wts and d in matrix.index]
        nav = fs.config_nav(dates, wts, matrix)
        m = fs.stats(nav)
        b = beta_vs_spy(nav, dates, matrix)
        e1, e2 = eras(nav, dates)
        rows.append((label, m, b, en, nav[-1], e1, e2))
        last_dates = dates

    # benchmarks (post-tax, mark-to-liquidation)
    bench = {}
    for tkr in (SPY, "QQQ"):
        nav = fs.bench_nav(matrix, last_dates, tkr)
        bench[tkr] = (fs.stats(nav), nav[-1])
    # SPY cap-universe effN proxy + QQQ beta
    caps = data.caps.reindex(last_dates)
    spy_en = float(np.mean([1.0/((c[c>0]/c[c>0].sum())**2).sum()
                            for _, c in caps.iterrows() if c.dropna().sum() > 0]))
    qb = beta_vs_spy(fs.bench_nav(matrix, last_dates, "QQQ"), last_dates, matrix)

    print("\n" + "="*104)
    print("POST-TAX (13mo x4 sleeves, ST32/LT15, $100k, VIX OFF). Beta vs SPY on after-tax monthly returns.")
    print("="*104)
    h = f"{'config':30}{'CAGR':>7}{'Sharpe':>8}{'Sortino':>9}{'Calmar':>8}{'maxDD':>8}{'beta':>7}{'effN':>7}{'walkaway':>13}"
    print(h)
    for label, m, b, en, wa, e1, e2 in rows:
        print(f"{label:30}{m['cagr']*100:6.1f}%{m['sharpe']:8.2f}{m['sortino']:9.2f}"
              f"{m['calmar']:8.2f}{m['mdd']*100:7.1f}%{b:7.2f}{en:7.1f}{wa:>13,.0f}")
    (sm, swa) = bench[SPY]; (qm, qwa) = bench["QQQ"]
    print(f"{'SPY':30}{sm['cagr']*100:6.1f}%{sm['sharpe']:8.2f}{sm['sortino']:9.2f}"
          f"{sm['calmar']:8.2f}{sm['mdd']*100:7.1f}%{1.00:7.2f}{spy_en:7.1f}{swa:>13,.0f}")
    print(f"{'QQQ':30}{qm['cagr']*100:6.1f}%{qm['sharpe']:8.2f}{qm['sortino']:9.2f}"
          f"{qm['calmar']:8.2f}{qm['mdd']*100:7.1f}%{qb:7.2f}{'—':>7}{qwa:>13,.0f}")

    print("\n--- by era (CAGR / Sharpe) ---")
    print(f"{'config':30}{'ERA1 cagr/shrp':>20}{'ERA2 cagr/shrp':>20}")
    for label, m, b, en, wa, e1, e2 in rows:
        print(f"{label:30}{e1['cagr']*100:11.1f}% {e1['sharpe']:6.2f}{e2['cagr']*100:11.1f}% {e2['sharpe']:6.2f}")


if __name__ == "__main__":
    main()
