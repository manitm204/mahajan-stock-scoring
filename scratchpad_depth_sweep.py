import pandas as pd
from run_walkforward import _load_panel, get_db, PANEL_START, PRICE_END
from research.walkforward.runner import run_splits
from research.walkforward.splits import resolve_splits
from research.walkforward import portfolio as pf
from research.walkforward.portfolio import performance_metrics
from backtesting import data_loader as dl
from backtesting.data_loader import SPY

panel = _load_panel(False)  # cached, insider-clean
with get_db() as db:
    matrix = dl.load_price_matrix(db, panel.universe, PANEL_START, PRICE_END)
    sectors = dl.global_sectors(db)

run = run_splits(panel, resolve_splits("rolling5y"), matrix, sectors)
scores = run.pooled_scores

pcts = [round(0.1*i,1) for i in range(1,11)]  # 0.1 .. 1.0
rows = []
spy_row = None
for pct in pcts:
    sim = pf.simulate(scores, matrix, sectors, top_pct=pct, mode="sector_neutral")
    m = sim.metrics
    rows.append((f"top{int(pct*100)}%", m["cagr"], m["sharpe"], m["sortino"],
                 m["max_drawdown"], m["ann_vol"], m["n_periods"]))
    if spy_row is None:
        sm = performance_metrics(sim.spy_period_returns, hold_months=1)
        spy_row = ("SPY", sm["cagr"], sm["sharpe"], sm["sortino"],
                   sm["max_drawdown"], sm["ann_vol"], sm["n_periods"])

print(f"\n{'book':>8} | {'CAGR':>7} | {'Sharpe':>6} | {'Sortino':>7} | {'maxDD':>7} | {'vol':>6}")
print("-"*56)
for r in rows:
    print(f"{r[0]:>8} | {r[1]*100:+6.2f}% | {r[2]:6.2f} | {r[3]:7.2f} | {r[4]*100:+6.1f}% | {r[5]*100:5.1f}%")
print("-"*56)
print(f"{spy_row[0]:>8} | {spy_row[1]*100:+6.2f}% | {spy_row[2]:6.2f} | {spy_row[3]:7.2f} | {spy_row[4]*100:+6.1f}% | {spy_row[5]*100:5.1f}%")
print(f"\n(sector_neutral construction, rolling-5y composite, n_periods={rows[0][6]})")
