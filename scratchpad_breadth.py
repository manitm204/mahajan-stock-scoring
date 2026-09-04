"""Do stocks with BROAD multi-factor support beat stocks driven by ONE strong parent,
holding overall strength constant?

Per stock/date (>=5 parents present): composite = mean parent score; breadth = LOW
cross-parent std (broad) vs HIGH std (one factor dominates); n_strong = # parents >=60.
Double-sort composite tercile x dispersion tercile (dispersion ranked WITHIN each
composite tercile per date) isolates breadth from strength. Equal-weight monthly cells."""
import pandas as pd
import numpy as np
from collections import defaultdict

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
ps = run.parent_scores
comp = run.pooled_scores
parents = sorted(ps)
dates = [d for d in sorted(comp) if d in matrix.index]
SPLIT = "2021-12-31"
MIN_P, STRONG = 5, 60

def tercile(x, labels):
    return pd.qcut(x.rank(method="first"), 3, labels=labels)

grid = defaultdict(dict)          # (comp_t, disp_t) -> {nxt: ret}
nstrong = defaultdict(dict)       # bucket -> {nxt: ret}
nstrong_comp = defaultdict(list)  # bucket -> [avg composite]

for i in range(len(dates) - 1):
    d, nxt = dates[i], dates[i + 1]
    P = pd.DataFrame({p: ps[p].get(d) for p in parents if ps[p].get(d) is not None})
    if P.empty:
        continue
    P = P[P.notna().sum(axis=1) >= MIN_P]
    if len(P) < 30:
        continue
    c = P.mean(axis=1)
    sd = P.std(axis=1)
    ns = (P >= STRONG).sum(axis=1)
    fwd = matrix.loc[nxt].reindex(P.index) / matrix.loc[d].reindex(P.index) - 1.0

    ct = tercile(c, ["Lo", "Mid", "Hi"])
    for cl in ["Lo", "Mid", "Hi"]:
        idx = ct.index[ct == cl]
        if len(idx) < 9:
            continue
        dt = tercile(sd[idx], ["Broad", "Mid", "Conc"])   # low std -> Broad
        for dlab in ["Broad", "Mid", "Conc"]:
            r = fwd.reindex(dt.index[dt == dlab]).dropna()
            if len(r):
                grid[(cl, dlab)][nxt] = float(r.mean())

    for k in range(0, 8):
        lab = "5+" if k >= 5 else str(k)
        sel = ns.index[(ns == k) if k < 5 else (ns >= 5)]
        r = fwd.reindex(sel).dropna()
        if len(r):
            nstrong[lab][nxt] = float(r.mean())
            nstrong_comp[lab].append(float(c.reindex(sel).mean()))

def stats(series):
    m = performance_metrics(pd.Series(series, dtype=float).sort_index(), 1)
    return m["cagr"], m["sharpe"]

print("\n=== n_strong: # parents >=60 (RAW single sort, confounded with strength) ===")
print(f"  {'#strong':8s} {'avg comp':>8} {'CAGR':>8} {'Sharpe':>7} {'obs':>5}")
for lab in ["0", "1", "2", "3", "4", "5+"]:
    if lab in nstrong:
        c, sh = stats(nstrong[lab])
        ac = np.mean(nstrong_comp[lab])
        print(f"  {lab:8s} {ac:8.1f} {c*100:+7.2f}% {sh:7.2f} {len(nstrong[lab]):5d}")

print("\n=== DOUBLE SORT: composite tercile x dispersion (breadth controlled for strength) ===")
print("   Broad = low cross-parent std (multi-factor) ... Conc = high std (one strong parent)")
for metric, fn in (("CAGR", lambda s: stats(s)[0] * 100), ("Sharpe", lambda s: stats(s)[1])):
    print(f"\n  -- {metric} --")
    print(f"  {'composite':10s} | {'Broad':>8} {'Mid':>8} {'Conc':>8} | {'Broad-Conc':>10}")
    for cl in ["Hi", "Mid", "Lo"]:
        vals = {dl_: fn(grid[(cl, dl_)]) for dl_ in ["Broad", "Mid", "Conc"]}
        diff = vals["Broad"] - vals["Conc"]
        unit = "%" if metric == "CAGR" else ""
        print(f"  {cl:10s} | {vals['Broad']:7.2f}{unit} {vals['Mid']:7.2f}{unit} "
              f"{vals['Conc']:7.2f}{unit} | {diff:+9.2f}{unit}")

print("\n=== Hi-composite Broad vs Conc, per era (regime robustness) ===")
for cl in ["Hi", "Lo"]:
    for dl_ in ["Broad", "Conc"]:
        s = pd.Series(grid[(cl, dl_)], dtype=float).sort_index()
        idx = s.index.astype(str)
        e1 = performance_metrics(s[idx <= SPLIT], 1)
        e2 = performance_metrics(s[idx > SPLIT], 1)
        print(f"  {cl:3s} {dl_:6s}: era1 CAGR {e1['cagr']*100:+6.2f}% Shp {e1['sharpe']:4.2f} | "
              f"era2 CAGR {e2['cagr']*100:+6.2f}% Shp {e2['sharpe']:4.2f}")
