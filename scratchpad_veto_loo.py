"""Leave-one-out veto test on the clean rolling-5y composite.

Books (equal-weight survivors, monthly, PIT):
  UNIVERSE            - all scored names, no veto
  VETO_ALL            - drop any name in the bottom 10% (per-date rank) of ANY parent
  VETO_ALL_ex_<P>     - same, but parent P's veto disabled (its bottom-10% not dropped)
Reference: SPY on the same grid. Full-period + two-era CAGR (split 2021-12-31)."""
import pandas as pd
import numpy as np

from run_walkforward import _load_panel, get_db, PANEL_START, PRICE_END
from research.walkforward.runner import run_splits
from research.walkforward.splits import resolve_splits
from research.walkforward.portfolio import performance_metrics
from backtesting import data_loader as dl
from backtesting.data_loader import SPY

panel = _load_panel(False)
with get_db() as db:
    matrix = dl.load_price_matrix(db, panel.universe, PANEL_START, PRICE_END)
    sectors = dl.global_sectors(db)

run = run_splits(panel, resolve_splits("rolling5y"), matrix, sectors, verbose=False)
parent_scores = run.parent_scores
comp = run.pooled_scores
dates = [d for d in sorted(comp) if d in matrix.index]
parents = sorted(parent_scores)
SPLIT = "2021-12-31"
CUT = 10  # bottom-10% percentile veto

# per-parent bottom-10% name sets per date
bottom = {p: {d: set(rk.index[rk < CUT])
              for d, s in parent_scores[p].items()
              for rk in [s.rank(pct=True) * 100.0]}
          for p in parents}

def drop_set(d, exclude):
    out = set()
    for p in parents:
        if p == exclude:
            continue
        out |= bottom[p].get(d, set())
    return out

def book_pred(kind):
    def f(d):
        uni = comp[d].dropna().index
        if kind == "UNIVERSE":
            return uni
        exclude = None if kind == "ALL" else kind   # kind == parent name
        return uni.difference(drop_set(d, exclude))
    return f

def basket_series(pred):
    out = {}
    for i in range(len(dates) - 1):
        d, nxt = dates[i], dates[i + 1]
        names = pred(d)
        if len(names) == 0:
            continue
        r = (matrix.loc[nxt].reindex(names) / matrix.loc[d].reindex(names) - 1.0).dropna()
        if not r.empty:
            out[nxt] = float(r.mean())
    return pd.Series(out, dtype=float).sort_index()

def avg_n(pred):
    return np.mean([len(pred(d)) for d in dates[:-1]])

def row(pred):
    s = basket_series(pred)
    m = performance_metrics(s, 1)
    idx = s.index.astype(str)
    e1 = performance_metrics(s[idx <= SPLIT], 1)["cagr"]
    e2 = performance_metrics(s[idx > SPLIT], 1)["cagr"]
    return (avg_n(pred), m["cagr"], m["sharpe"], m["sortino"], m["max_drawdown"], e1, e2)

# SPY on the same grid
spy = (matrix[SPY].reindex(dates) / matrix[SPY].reindex(dates).shift(1) - 1.0).dropna()
sm = performance_metrics(spy, 1)
si = spy.index.astype(str)
spy_e1 = performance_metrics(spy[si <= SPLIT], 1)["cagr"]
spy_e2 = performance_metrics(spy[si > SPLIT], 1)["cagr"]

books = [("UNIVERSE (no veto)", book_pred("UNIVERSE")),
         ("VETO_ALL (<10 any)", book_pred("ALL"))]
books += [(f"  ..ex {p}", book_pred(p)) for p in parents]

print(f"\n{'book':22s} | {'n':>4} {'CAGR':>7} {'Shp':>5} {'Srt':>5} {'maxDD':>7} | {'e1 CAGR':>8} {'e2 CAGR':>8}")
print("-" * 84)
for name, pred in books:
    n, c, sh, so, dd, e1, e2 = row(pred)
    print(f"{name:22s} | {n:4.0f} {c*100:+6.2f}% {sh:5.2f} {so:5.2f} {dd*100:+6.1f}% | "
          f"{e1*100:+7.2f}% {e2*100:+7.2f}%")
print("-" * 84)
print(f"{'SPY':22s} | {'':>4} {sm['cagr']*100:+6.2f}% {sm['sharpe']:5.2f} {sm['sortino']:5.2f} "
      f"{sm['max_drawdown']*100:+6.1f}% | {spy_e1*100:+7.2f}% {spy_e2*100:+7.2f}%")
print("\nread: an '..ex P' row ABOVE VETO_ALL = P's veto was a drag (its bottom names were fine);")
print("      BELOW VETO_ALL = P's veto was pulling its weight.")
