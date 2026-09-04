"""Diagnostic: for each parent, do its bottom-tail names underperform the whole
universe, and is it stable across 2017-2021 vs 2022-2026?

Bottom tail = per-date percentile rank of the parent score below the threshold
(rank<10 = bottom 10%, rank<20 = bottom 20%), matching the loser-screen veto.
Equal-weight monthly baskets held to next rebalance; two eras. Parent scores are
the rolling-5y OOS frozen parent scores (production-faithful)."""
import pandas as pd
import numpy as np

from run_walkforward import _load_panel, get_db, PANEL_START, PRICE_END
from research.walkforward.runner import run_splits
from research.walkforward.splits import resolve_splits
from research.walkforward.portfolio import performance_metrics
from backtesting import data_loader as dl

panel = _load_panel(False)
with get_db() as db:
    matrix = dl.load_price_matrix(db, panel.universe, PANEL_START, PRICE_END)
    sectors = dl.global_sectors(db)

run = run_splits(panel, resolve_splits("rolling5y"), matrix, sectors, verbose=False)
parent_scores = run.parent_scores            # {parent: {date: Series}}
comp = run.pooled_scores                      # {date: Series}  (universe reference)

dates = [d for d in sorted(comp) if d in matrix.index]
parents = sorted(parent_scores)
SPLIT = "2021-12-31"
THRESH = [10, 20]

def basket_series(pred):
    out = {}
    for i in range(len(dates) - 1):
        d, nxt = dates[i], dates[i + 1]
        names = pred(d)
        if names is None or len(names) == 0:
            continue
        r = (matrix.loc[nxt].reindex(names) / matrix.loc[d].reindex(names) - 1.0).dropna()
        if not r.empty:
            out[nxt] = float(r.mean())
    return pd.Series(out, dtype=float).sort_index()

def era_metrics(s):
    if s.empty:
        nan = (float("nan"), float("nan"))
        return nan, nan
    idx = s.index.astype(str)
    e1, e2 = s[idx <= SPLIT], s[idx > SPLIT]
    m1, m2 = performance_metrics(e1, 1), performance_metrics(e2, 1)
    return (m1["cagr"], m1["sharpe"]), (m2["cagr"], m2["sharpe"])

def avg_n(pred, era):
    ns = [len(pred(d)) for d in dates[:-1] if (era == 1) == (d <= SPLIT)]
    return np.mean(ns) if ns else 0

def pred_bottom(p, t):
    def f(d):
        col = parent_scores[p].get(d)
        if col is None:
            return pd.Index([])
        rk = col.rank(pct=True) * 100.0        # low score -> low rank
        return rk.index[rk < t]
    return f

uni_pred = lambda d: comp[d].dropna().index
(u1c, u1s), (u2c, u2s) = era_metrics(basket_series(uni_pred))

for era, (uc, us), label in ((1, (u1c, u1s), "ERA 1  (2017-01 -> 2021-12)"),
                             (2, (u2c, u2s), "ERA 2  (2022-01 -> 2026-06)")):
    print(f"\n================  {label}  ================")
    print(f"  UNIVERSE (all names):   CAGR {uc*100:+6.2f}%   Sharpe {us:5.2f}")
    print(f"  {'parent':14s} | {'<10 n':>5} {'CAGR':>7} {'Shp':>5} | {'<20 n':>5} {'CAGR':>7} {'Shp':>5}")
    print("  " + "-" * 60)
    for p in parents:
        row = []
        for t in THRESH:
            pred = pred_bottom(p, t)
            (e1, e2) = era_metrics(basket_series(pred))
            c, sh = e1 if era == 1 else e2
            row.append((avg_n(pred, era), c, sh))
        (n1, c1, s1), (n2, c2, s2) = row
        print(f"  {p:14s} | {n1:5.0f} {c1*100:+6.2f}% {s1:5.2f} | "
              f"{n2:5.0f} {c2*100:+6.2f}% {s2:5.2f}")
print("\n(baskets that trail the UNIVERSE row in BOTH eras = parents whose bottom tail is a robust loser signal)")
