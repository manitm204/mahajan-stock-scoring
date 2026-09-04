"""Runner — VIX-relative parent-weight overlay study (research-only, see vixtilt/).

Question: does a VIX-relative parent-weight overlay improve the existing rolling-5y
factor model? Baseline per window is the exact production walk-forward construction
(select_config); overlays adjust parent weights only, from training-only VIX statistics.

Usage:
    python run_vixtilt_study.py                  # uses cache/vixtilt/ window caches
    python run_vixtilt_study.py --rebuild-cache  # rebuild the per-window baselines
    python run_vixtilt_study.py --cost-bps 10 --out output/vixtilt
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

from backtesting import data_loader as dl
from data.db import get_db
from research.panel import ScorePanel
from research.subfactor_expansion.panel import cache_key, load_cached_panel

from vixtilt import report as rp
from vixtilt import study as st

PANEL_START, PANEL_END = "2015-06-30", "2026-06-30"
PRICE_END = "2026-07-06"
PANEL_CACHE = Path("cache/subfactor_expansion")


def _load_panel() -> ScorePanel:
    path = PANEL_CACHE / cache_key(PANEL_START, PANEL_END, "monthly")
    cand = load_cached_panel(path)
    if cand is None:
        raise SystemExit(f"candidate panel cache missing: {path}\n"
                         "build it first (e.g. python run_walkforward.py)")
    parent_keys = [p for p, subs in cand.candidates_by_parent.items() if subs]
    return ScorePanel(rebal_dates=list(cand.rebal_dates), scores=cand.scores,
                      parent_keys=parent_keys,
                      sub_by_parent={p: list(cand.candidates_by_parent[p])
                                     for p in parent_keys},
                      universe=list(cand.universe))


def _load_vix(db) -> pd.Series:
    df = db.query_df("SELECT date, adj_close FROM daily_prices WHERE ticker='VIX' "
                     "ORDER BY date")
    if df.empty:
        raise SystemExit("no VIX history in daily_prices")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["adj_close"].astype(float)


def main() -> int:
    ap = argparse.ArgumentParser(description="VIX-relative overlay walk-forward study")
    ap.add_argument("--out", default="output/vixtilt")
    ap.add_argument("--cache-dir", default="cache/vixtilt")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--first-test-year", type=int, default=2017)
    ap.add_argument("--last-end", default="2026-06-30")
    ap.add_argument("--cost-bps", type=float, default=10.0,
                    help="one-side transaction cost in bps of traded notional")
    ap.add_argument("--variant-set",
                    choices=["default", "lit", "smooth", "sens", "sensw"],
                    default="default",
                    help="'lit' = pre-registered fixed literature rule variants; "
                         "'smooth' = linear-ramp rule vs step rule vs baseline; "
                         "'sens' = band-knot sensitivity grid (robustness only)")
    args = ap.parse_args()

    t0 = time.time()
    print(f"Loading candidate panel ({PANEL_START} → {PANEL_END})…")
    panel = _load_panel()
    with get_db() as db:
        matrix = dl.load_price_matrix(db, panel.universe, PANEL_START, PRICE_END)
        sectors = dl.global_sectors(db)
        vix = _load_vix(db)
    print(f"Panel: {len(panel.rebal_dates)} rebalances, {len(panel.universe)} names | "
          f"prices {matrix.shape[0]}d × {matrix.shape[1]} | "
          f"VIX {vix.index[0].date()} → {vix.index[-1].date()}")

    from vixtilt.overlay import (DEFAULT_VARIANTS, LIT_VARIANTS, SENS_VARIANTS,
                                 SENSW_VARIANTS, SMOOTH_VARIANTS)
    variants = {"default": DEFAULT_VARIANTS, "lit": LIT_VARIANTS,
                "smooth": SMOOTH_VARIANTS, "sens": SENS_VARIANTS,
                "sensw": SENSW_VARIANTS}[args.variant_set]
    res = st.run_study(panel, matrix, sectors, vix, variants=variants,
                       first_test_year=args.first_test_year, last_end=args.last_end,
                       cache_dir=Path(args.cache_dir),
                       rebuild_cache=args.rebuild_cache,
                       cost_per_side=args.cost_bps / 1e4)

    out_dir = Path(args.out)
    print(f"Writing outputs to {out_dir}/ …")
    rp.write_all(res, out_dir, args.cost_bps / 1e4)

    full = pd.read_csv(out_dir / "full_net.csv")
    show = full[full["top_pct"] == 0.20][
        ["variant", "cagr", "sharpe", "spy_excess_cagr", "spy_ir", "ic_6m",
         "avg_turnover"]]
    pd.set_option("display.float_format", "{:.3f}".format)
    print("\n=== Full period (net, top 20%) ===")
    print(show.to_string(index=False))
    print(f"\nDone in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
