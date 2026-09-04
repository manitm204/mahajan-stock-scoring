"""Parent-Factor Selection Study — rank & diversify each parent from its sub-factors.

A simple, transparent alternative to the combo-enumeration study
(``run_parent_construction.py``). For every parent bucket it scores each sub-factor on
five reads (mean IC, IC-IR, Q5-Q1 spread, hit rate, coverage), blends within-bucket
percentile ranks into one Sub-factor Score, then builds the parent greedily — top sub
first, then only diversifying (cross-sectional R² < 0.60) positive-IC subs, up to three
— and weights them ∝ mean IC, capped at 50%. The intended model shape stays:

    Raw Sub-factors → Parent Factor Scores → Final Composite

**Read-only. Nothing is applied to production scoring or weights.** Outputs (per-parent
ranking / R² / decision CSVs, a selections CSV, REPORT.md) land under
``output/parent_selection/``.

Usage:
    python run_parent_selection.py
    python run_parent_selection.py --parents momentum,value,growth
    python run_parent_selection.py --no-cache        # force a fresh re-score
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtesting import data_loader as dl
from data.config import load_config
from data.db import get_db
from research import HORIZON_MONTHS, compute_forward_returns
from research import parent_selection as ps
from research.panel import (ScorePanel, build_score_panel, cache_key,
                            load_cached_panel, save_cached_panel)

DEFAULT_START = "2022-01-01"
DEFAULT_END = "2026-06-01"
DEFAULT_FREQ = "monthly"
OUT_DIR = Path("output/parent_selection")
# The cached full panel already scores every parent + sub; reuse it if present.
PANEL_DIR = Path("output/factor_subset_selection")

# --- expanded-library source (research subfactor library, incl. the new insider
# buy-focused subs + the PIT-window fix) -------------------------------------
EXP_START = "2023-01-01"          # matches run_subfactor_expansion.py defaults so the
EXP_END = "2026-06-30"            # same on-disk candidate-panel cache is reused
EXP_CACHE_DIR = Path("cache/subfactor_expansion")
EXP_OUT_DIR = Path("output/parent_selection_expansion")


def _load_panel(args, cfg):
    """Reuse the cached full ScorePanel (all parents + subs) or build + cache one."""
    ck = PANEL_DIR / cache_key(args.start, args.end, args.freq)
    panel = None if args.no_cache else load_cached_panel(ck)
    if panel is not None:
        print(f"Loaded cached panel ({len(panel.rebal_dates)} rebalances, "
              f"{len(panel.all_subs)} subs).")
        return panel
    with get_db() as db:
        trading = dl.trading_calendar(db)
        universe = db.universe_tickers()
        matrix = dl.load_price_matrix(db, universe, args.start, args.end)
        rebal = [d for d in dl.generate_rebalance_dates(
            trading, args.start, args.end, args.freq) if d in matrix.index]
        if len(rebal) < 6:
            raise RuntimeError("Need at least six rebalances for a meaningful study.")
        print(f"Scoring {len(rebal)} rebalances ({rebal[0]} → {rebal[-1]})...")
        panel = build_score_panel(db, rebal, cfg)
        save_cached_panel(panel, ck)
    return panel


def _adapt_candidate_panel(cand) -> ScorePanel:
    """View a :class:`CandidatePanel` (research subfactor library) as a
    :class:`ScorePanel` so the parent-selection engine can consume it unchanged.

    Both carry per-rebalance 0-100 GICS-sector-relative score frames on the same
    universe; only the taxonomy field names differ. ``sub_by_parent`` becomes the
    candidate-by-parent map (``higher_is_better`` is already baked into each score),
    so ``parent_of`` / ``all_subs`` / ``signal_frame`` all work as-is."""
    parent_keys = [p for p, subs in cand.candidates_by_parent.items() if subs]
    return ScorePanel(
        rebal_dates=list(cand.rebal_dates),
        scores=cand.scores,
        parent_keys=parent_keys,
        sub_by_parent={p: list(cand.candidates_by_parent[p]) for p in parent_keys},
        universe=list(cand.universe),
    )


def _load_expansion_panel(args, cfg) -> ScorePanel:
    """Load (or build) the expanded candidate panel and adapt it to a ScorePanel.

    Reuses the exact on-disk cache that ``run_subfactor_expansion.py --build`` writes
    (``cache/subfactor_expansion/cand_panel_<start>_<end>_monthly_v1.pkl``) so the two
    entry points stay in sync — build once, then rank here."""
    from research.subfactor_expansion.panel import (
        build_candidate_panel, cache_key as exp_cache_key,
        load_cached_panel as exp_load, save_cached_panel as exp_save)

    ckey = exp_cache_key(args.start, args.end, "monthly")
    cache_path = EXP_CACHE_DIR / ckey
    cand = None if args.no_cache else exp_load(cache_path)
    if cand is None:
        with get_db() as db:
            rebals = _expansion_rebalances(db, args.start, args.end)
            if len(rebals) < 6:
                raise RuntimeError("Need at least six rebalances for a meaningful study.")
            print(f"Scoring {len(rebals)} expanded-library rebalances "
                  f"({rebals[0]} → {rebals[-1]})...")
            cand = build_candidate_panel(db, rebals, cfg, verbose=True)
        exp_save(cand, cache_path)
    else:
        print(f"Loaded cached expanded panel ({len(cand.rebal_dates)} rebalances, "
              f"{len(cand.all_candidates)} candidates).")
    return _adapt_candidate_panel(cand)


def _expansion_rebalances(db, start: str, end: str) -> list[str]:
    """Month-end trading dates in [start, end] — mirrors run_subfactor_expansion.py."""
    df = db.query_df(
        "SELECT DISTINCT date FROM daily_prices WHERE date >= ? AND date <= ? "
        "ORDER BY date", (start, end))
    if df.empty:
        return []
    dates = pd.to_datetime(df["date"])
    return [str(grp.max().date()) for _, grp in dates.groupby(dates.dt.to_period("M"))]


def _print_parent(res: ps.ParentSelection) -> None:
    tag = f"{len(res.selected)} sub(s)"
    print(f"\n{res.parent} ({tag}): {res.formula}")
    print(f"    selected: {', '.join(res.selected)}"
          + (f"  [{res.signal_flag}]" if res.signal_flag else ""))
    print(f"    stop: {res.stop_reason}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Parent-factor rank-and-diversify selection")
    ap.add_argument("--source", default="production", choices=["production", "expansion"],
                    help="which sub-factor library to rank: `production` = the live "
                         "factors/ engine (3 insider subs); `expansion` = the research "
                         "subfactor library incl. the new insider buy-focused subs + the "
                         "PIT-window fix (the panel run_subfactor_expansion.py --build writes)")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--freq", default=DEFAULT_FREQ, choices=["weekly", "monthly", "quarterly"])
    ap.add_argument("--parents", default="", help="comma-separated subset (default: all)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-cache", action="store_true", help="ignore cached panel and re-score")
    args = ap.parse_args()

    # Source-specific defaults so the expansion window/out land on the right cache + dir.
    is_exp = args.source == "expansion"
    if args.start is None:
        args.start = EXP_START if is_exp else DEFAULT_START
    if args.end is None:
        args.end = EXP_END if is_exp else DEFAULT_END
    if args.out is None:
        args.out = str(EXP_OUT_DIR if is_exp else OUT_DIR)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    panel = _load_expansion_panel(args, cfg) if is_exp else _load_panel(args, cfg)

    with get_db() as db:
        matrix = dl.load_price_matrix(db, db.universe_tickers(), args.start, args.end)
    fwd = compute_forward_returns(matrix, panel.rebal_dates, HORIZON_MONTHS)

    parents = ([p.strip() for p in args.parents.split(",") if p.strip()]
               or [p for p in panel.parent_keys if panel.sub_by_parent.get(p)])
    results = ps.run_selection(panel, fwd, parents)

    sel_rows: list[dict] = []
    for res in results:
        res.ranking.to_csv(out_dir / f"ranking_{res.parent}.csv", index=False)
        res.r2.to_csv(out_dir / f"r2_{res.parent}.csv")
        res.decisions.to_csv(out_dir / f"decisions_{res.parent}.csv", index=False)
        sel_rows.append({
            "parent": res.parent, "n_subs": len(res.selected),
            "selected": ", ".join(res.selected), "formula": res.formula,
            "weights": ", ".join(f"{s}={res.weights[s]:.3f}" for s in res.selected),
            "signal_flag": res.signal_flag, "stop_reason": res.stop_reason})
        _print_parent(res)

    pd.DataFrame(sel_rows).to_csv(out_dir / "selections.csv", index=False)
    meta = {"start": args.start, "end": args.end, "freq": args.freq,
            "n_rebalances": len(panel.rebal_dates), "n_universe": len(panel.universe)}
    ps.write_report(out_dir, results, meta)

    print("\n" + "=" * 88)
    print(" SELECTED PARENT CONSTRUCTIONS")
    print("=" * 88)
    print(pd.DataFrame(sel_rows)[["parent", "n_subs", "formula", "signal_flag"]]
          .to_string(index=False))
    print(f"\nSaved rankings, R² matrices, decisions, selections.csv and REPORT.md to {out_dir}/")


if __name__ == "__main__":
    main()
