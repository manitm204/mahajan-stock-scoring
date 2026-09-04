"""Runner — loser-screen walk-forward study (research-only, see loserscreen/).

Question: does excluding the composite's bottom-ranked names from a broad
equal-weight book beat the same book unscreened? Pre-registered bar in
loserscreen/__init__.py. Reuses the vixtilt window caches (fast after Stage A).

Usage:
    python run_loserscreen_study.py                  # uses cache/vixtilt/
    python run_loserscreen_study.py --out output/loserscreen --cost-bps 10
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

from loserscreen import report as rp
from loserscreen import study as st

PANEL_START, PANEL_END = "2015-06-30", "2026-06-30"
PRICE_END = "2026-07-06"
PANEL_CACHE = Path("cache/subfactor_expansion")


def _load_panel() -> ScorePanel:
    path = PANEL_CACHE / cache_key(PANEL_START, PANEL_END, "monthly")
    cand = load_cached_panel(path)
    if cand is None:
        raise SystemExit(f"candidate panel cache missing: {path}")
    parent_keys = [p for p, subs in cand.candidates_by_parent.items() if subs]
    return ScorePanel(rebal_dates=list(cand.rebal_dates), scores=cand.scores,
                      parent_keys=parent_keys,
                      sub_by_parent={p: list(cand.candidates_by_parent[p])
                                     for p in parent_keys},
                      universe=list(cand.universe))


def main() -> int:
    ap = argparse.ArgumentParser(description="Loser-screen walk-forward study")
    ap.add_argument("--out", default="output/loserscreen")
    ap.add_argument("--cache-dir", default="cache/vixtilt")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--first-test-year", type=int, default=2017)
    ap.add_argument("--last-end", default="2026-06-30")
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--books", choices=["v1", "v2", "mix", "tilt", "wmix", "veto", "veto2",
                             "cands"],
                    default="v1",
                    help="v1 = original EW study; v2 = cap-weighted confirmation "
                         "+ depth curve; mix = 50/50 EW-cap blend confirmation; "
                         "tilt = EXPLORATORY mix depth 40/50 + quintile rank-tilt; "
                         "wmix = EXPLORATORY weighting-blend grid on screen30; "
                         "veto = v4 parent-veto within screen25 + random null; "
                         "veto2 = EXPLORATORY veto decomposition (leave-one-"
                         "out + subsets + size-matched nulls); "
                         "cands = v5 new-candidate vetoes (pre-registered) "
                         "(all modes except v1 need PIT market caps)")
    ap.add_argument("--null-draws", type=int, default=200,
                    help="random-characteristic null draws (veto mode only)")
    args = ap.parse_args()

    t0 = time.time()
    print(f"Loading candidate panel ({PANEL_START} → {PANEL_END})…")
    panel = _load_panel()
    mcaps = None
    with get_db() as db:
        matrix = dl.load_price_matrix(db, panel.universe, PANEL_START, PRICE_END)
        sectors = dl.global_sectors(db)
        if args.books != "v1":
            from loserscreen.mcap import market_caps
            print("Building PIT market caps…")
            mcaps = market_caps(db, list(panel.rebal_dates), panel.universe, matrix)
    print(f"Panel: {len(panel.rebal_dates)} rebalances, {len(panel.universe)} names")

    books = {"v1": st.BOOKS, "v2": st.BOOKS_V2, "mix": st.BOOKS_MIX,
             "tilt": st.BOOKS_TILT, "wmix": st.BOOKS_WMIX,
             "veto": st.BOOKS_VETO, "veto2": st.BOOKS_VETO2,
             "cands": st.BOOKS_CANDS}[args.books]
    extra = None
    if args.books == "cands":
        from loserscreen.candidates import candidate_scores
        print("Building candidate veto scores (fundamentals + prices)…")
        with get_db() as db:
            extra = candidate_scores(db, matrix, list(panel.rebal_dates),
                                     panel.universe)
    res = st.run_study(panel, matrix, sectors, books=books, mcaps=mcaps,
                       extra_cols=extra,
                       first_test_year=args.first_test_year, last_end=args.last_end,
                       cache_dir=Path(args.cache_dir),
                       rebuild_cache=args.rebuild_cache,
                       cost_per_side=args.cost_bps / 1e4)

    out_dir = Path(args.out)
    print(f"Writing outputs to {out_dir}/ …")
    if args.books == "cands":
        nulls = {}
        for i, c in enumerate(st._CANDS):
            name = f"vp6_{c}20"
            print(f"Size-matched null for {name} ({args.null_draws} draws)…")
            nulls[name] = st.random_null(
                res, name, "vpos6_20", matrix, mcaps,
                n_draws=args.null_draws, seed=20260715 + i,
                cost_per_side=args.cost_bps / 1e4)
        verdicts = rp.write_cands(res, out_dir, args.cost_bps / 1e4, nulls)
        print("\nPER-CANDIDATE VERDICTS: "
              + ", ".join(f"{c}={'PASS' if ok else 'FAIL'}"
                          for c, ok in verdicts.items()))
    elif args.books == "veto2":
        nulls = {}
        for i, bname in enumerate(("vall20", "vpos4_20", "vpos6_20")):
            print(f"Size-matched null for {bname} ({args.null_draws} draws)…")
            nulls[bname] = st.random_null(
                res, bname, "mix_screen25", matrix, mcaps,
                n_draws=args.null_draws, seed=20260714 + i,
                cost_per_side=args.cost_bps / 1e4)
        rp.write_veto2(res, out_dir, args.cost_bps / 1e4, nulls)
    elif args.books == "veto":
        print(f"Random-characteristic null ({args.null_draws} draws)…")
        null = st.random_null(res, "vqs20", "mix_screen25", matrix, mcaps,
                              n_draws=args.null_draws,
                              cost_per_side=args.cost_bps / 1e4)
        passed = rp.write_veto(res, out_dir, args.cost_bps / 1e4, null)
        print(f"\nCONFIRMATORY VERDICT (vqs20 vs mix_screen25): "
              f"{'PASS' if passed else 'FAIL'} — see {out_dir}/REPORT.md")
    elif args.books == "v2":
        rp.write_v2(res, out_dir, args.cost_bps / 1e4)
    elif args.books == "mix":
        rp.write_mix(res, out_dir, args.cost_bps / 1e4)
    elif args.books == "tilt":
        rp.write_explore(res, out_dir, args.cost_bps / 1e4)
    elif args.books == "wmix":
        rp.write_explore(res, out_dir, args.cost_bps / 1e4, ref="s30_ew")
    else:
        rp.write_all(res, out_dir, args.cost_bps / 1e4)

    full = pd.read_csv(out_dir / "full_net.csv")
    pd.set_option("display.float_format", "{:.3f}".format)
    print("\n=== Full period (net) ===")
    print(full[["book", "avg_n_names", "cagr", "sharpe", "sortino", "max_drawdown",
                "spy_excess_cagr", "spy_ir", "avg_turnover"]].to_string(index=False))
    if args.books in ("tilt", "wmix", "veto", "veto2", "cands"):
        act = pd.read_csv(out_dir / "active_stats.csv")
        print("\n=== Active vs reference ===")
        print(act.to_string(index=False))
    else:
        wins_file, pair = {
            "v1": ("window_wins.csv", "screen20 vs broad"),
            "v2": ("window_wins_cap.csv", "cap_screen20 vs cap_broad"),
            "mix": ("window_wins_mix.csv", "mix_screen20 vs mix_broad"),
        }[args.books]
        wins = pd.read_csv(out_dir / wins_file)
        print(f"\n{pair} window wins: {int(wins['win'].sum())}/{len(wins)}")
    print(f"Done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
