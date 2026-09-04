"""Quantify the delisting-realization fix: re-run the rolling-5y reference books
through the (now fixed) simulate() and compare against the recorded pre-fix numbers."""
import sys

sys.path.insert(0, "/home/manit/Desktop/fun_projects/mahajan_hedge_fund")

from run_walkforward import _load_panel, get_db, PANEL_START, PRICE_END
from research.walkforward.runner import run_splits
from research.walkforward.splits import resolve_splits
from research.walkforward.portfolio import simulate
from backtesting import data_loader as dl

panel = _load_panel(False)
with get_db() as db:
    matrix = dl.load_price_matrix(db, panel.universe, PANEL_START, PRICE_END)
    sectors = dl.global_sectors(db)
run = run_splits(panel, resolve_splits("rolling5y"), matrix, sectors, verbose=False)

PRE_FIX = {  # recorded 2026-07-16, before delisting realization
    "top10_sector_neutral": {"cagr": 0.1243, "spy_excess_cagr": -0.0142},
    "top100_equal": {"cagr": 0.1006},
}

print(f"{'book':24s} {'CAGR':>8} {'Shp':>6} {'Srt':>6} {'maxDD':>8} "
      f"{'exSPY':>8} {'alpha':>8} {'IR':>6}")
for top_pct, mode in [(0.10, "sector_neutral"), (0.10, "equal"),
                      (0.25, "equal"), (1.00, "equal")]:
    res = simulate(run.pooled_scores, matrix, sectors, top_pct=top_pct, mode=mode)
    m = res.metrics
    print(f"{res.label:24s} {m['cagr']*100:+7.2f}% {m['sharpe']:6.2f} {m['sortino']:6.2f} "
          f"{m['max_drawdown']*100:+7.1f}% {m['spy_excess_cagr']*100:+7.2f}% "
          f"{m['spy_alpha']*100:+7.2f}% {m['spy_ir']:6.2f}")

print("\npre-fix reference: top10_sector_neutral CAGR +12.43% exSPY -1.42% | "
      "top100_equal CAGR +10.06%")
