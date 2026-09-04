"""VIX-relative parent-weight overlay — clean-room walk-forward study runner.

Does tilting the frozen rolling-5Y parent weights toward regime-conditional
(comparable-VIX) parent performance improve the existing model out-of-sample?
Semiannual OOS windows 2017→present, baseline rebuilt+frozen per window,
overlays move parent weights only (training-data VIX statistics, hard PIT
assertions), composites re-ranked, identical portfolio rules per variant.

Research-only: writes to ``cache/vix_relative_overlay/`` and
``output/vix_relative_overlay/``; touches nothing in production.

Usage:
    python run_vix_relative_overlay.py                 # uses cached baselines
    python run_vix_relative_overlay.py --rebuild-cache # refreeze all windows
    python run_vix_relative_overlay.py --cost-bps 10 --out output/vix_relative_overlay
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

from vix_relative_overlay import baseline_cache as bc
from vix_relative_overlay import data_io, report
from vix_relative_overlay.metrics import forward_returns, quintile_spread, spearman_ic
from vix_relative_overlay.overlay import VARIANTS, variant_scores, window_decisions
from vix_relative_overlay.portfolio import (DEFAULT_COST_BPS, holdings_overlap,
                                            simulate)
from vix_relative_overlay.windows import build_windows

OUT_DIR = Path("output/vix_relative_overlay")
PRICE_END = "2026-12-31"     # loader takes whatever actually exists
PORTFOLIO_CONFIGS = [(0.10, "equal"), (0.20, "equal"), (0.30, "equal"),
                     (0.20, "sector_neutral")]


def main() -> int:
    ap = argparse.ArgumentParser(description="VIX-relative overlay study")
    ap.add_argument("--rebuild-cache", action="store_true",
                    help="refreeze every window baseline (slow, one-time)")
    ap.add_argument("--refresh-prices", action="store_true")
    ap.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    ap.add_argument("--first-test-year", type=int, default=2017)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    t0 = time.time()

    print("Loading inputs…")
    panel = data_io.load_panel()
    matrix = data_io.load_price_matrix(panel.universe, panel.rebal_dates[0],
                                       PRICE_END, refresh=args.refresh_prices)
    vix = data_io.load_vix()
    sectors = data_io.load_sectors()
    print(f"  panel: {len(panel.rebal_dates)} rebalances "
          f"({panel.rebal_dates[0]}→{panel.rebal_dates[-1]}), "
          f"{len(panel.universe)} PIT-union names")
    print(f"  prices: {matrix.shape[1]} tickers × {matrix.shape[0]} days "
          f"({matrix.index[0]}→{matrix.index[-1]}); "
          f"VIX {vix.index[0].date()}→{vix.index[-1].date()}")

    windows = build_windows(last_end=panel.rebal_dates[-1],
                            first_test_year=args.first_test_year)
    print(f"\n=== Freezing rolling-5Y baselines ({len(windows)} windows) ===")
    frozen = {}
    for win in windows:
        frozen[win.label] = bc.load_or_build(win, panel, matrix, vix,
                                             rebuild=args.rebuild_cache)
    fidelity = bc.verify_composite_fidelity(next(iter(frozen.values())),
                                            panel, sectors)
    print(f"  composite fidelity vs research implementation: "
          f"max abs diff {fidelity:.2e}")

    print("\n=== Overlay decisions + composite re-scoring ===")
    window_of: dict[str, str] = {}
    for lbl, fw in frozen.items():
        for d in fw.test_rebals:
            window_of[d] = lbl
    decisions: dict[str, list] = {}
    scores: dict[str, dict[str, pd.Series]] = {}
    for variant in VARIANTS:
        decs, sc = [], {}
        for fw in frozen.values():
            ds = window_decisions(fw, variant)
            decs.extend(ds)
            sc.update(variant_scores(fw, ds, sectors))
        assert set(sc) <= set(window_of), \
            f"{variant.name}: scored a date outside the OOS test windows"
        decisions[variant.name] = decs
        scores[variant.name] = sc
        active = [d for d in decs if d.strength > 0]
        print(f"  {variant.name:<14s} {len(sc)} rebalances scored, "
              f"tilt active on {len(active)}")

    all_dates = sorted(window_of)
    print("\n=== Composite IC / Q5-Q1 (pooled OOS) ===")
    fwd_all = forward_returns(matrix, all_dates)
    ics: dict[str, pd.DataFrame] = {}
    for vname, sc in scores.items():
        rows = []
        for h, fwd_h in fwd_all.items():
            for d, f in fwd_h.items():
                s = sc.get(d)
                if s is None:
                    continue
                rows.append({"date": d, "window": window_of[d], "horizon": h,
                             "ic": spearman_ic(s, f),
                             "spread": quintile_spread(s, f)})
        ics[vname] = pd.DataFrame(rows)

    print("=== Portfolio simulations ===")
    sims = {}
    for vname, sc in scores.items():
        for top_pct, mode in PORTFOLIO_CONFIGS:
            sims[(vname, top_pct, mode)] = simulate(
                sc, window_of, matrix, sectors,
                top_pct=top_pct, mode=mode, cost_bps=args.cost_bps)

    overlaps = {}
    for vname, sc in scores.items():
        if vname == "baseline":
            continue
        for top_pct, mode in PORTFOLIO_CONFIGS:
            if mode == "equal":
                overlaps[(vname, top_pct)] = holdings_overlap(
                    scores["baseline"], sc, window_of, top_pct)

    print("=== Writing reports ===")
    out_dir = Path(args.out)
    dash = report.build_dashboard(frozen, decisions, sims)
    report.write_reports(out_dir, frozen=frozen, decisions=decisions, sims=sims,
                         ics=ics, overlaps=overlaps, dash=dash,
                         cost_bps=args.cost_bps, fidelity_diff=fidelity)

    perf_full = report.performance_table(sims, "full")
    show = perf_full[(perf_full["top_pct"] == 0.20)
                     & (perf_full["mode"] == "equal")]
    print("\n=== Full period — top 20% equal, net ===")
    pd.set_option("display.float_format", "{:.3f}".format)
    print(show[["variant", "cagr", "sharpe", "max_drawdown", "spy_excess_cagr",
                "spy_ir", "avg_turnover"]].to_string(index=False))
    print(f"\nDone in {time.time() - t0:.0f}s → {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
