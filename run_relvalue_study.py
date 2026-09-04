"""Relative-value study driver.

Usage:
  python run_relvalue_study.py discover              # candidate inventory
  python run_relvalue_study.py backtest --family F1  # run a family's declared grid
  python run_relvalue_study.py backtest --configs f1_base f3_base
  python run_relvalue_study.py leaderboard
  python run_relvalue_study.py holdout --configs ... # ONLY after freeze (guarded)

Protocol: output/relvalue/PREREGISTRATION.md. Eval windows 2017-H1..2025-H1;
2025-07→2026-07 is the untouched holdout (the runner refuses to touch it
without --i-know-the-leaderboard-is-frozen).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from pairtrading.study import load_composites, load_sectors, members_as_of
from relvalue import discovery as dsc
from relvalue.engine import RVRule, simulate_rv_window
from relvalue.gates import HotStreakGate, RVGate
from relvalue.portfolio import (bench_returns, blend, committed_returns,
                                invested_returns, perf_stats, trade_stats,
                                yearly_table)
from relvalue.prices import build_tr_panel
from vixtilt.windows import semiannual_windows

OUT = Path("output/relvalue")
EVAL_LAST_END = "2025-06-30"
HOLDOUT_LAST_END = "2026-06-30"
ERA_SPLIT = "2022-01-01"
FORMATION_MONTHS = 12

# --------------------------------------------------------------------------- #
# Declared configuration grids (multiple-testing ledger: every entry counts)
# --------------------------------------------------------------------------- #
BASE = RVRule()          # 2.0 entry, 0.5 exit, ≤2.75 cap, ≥40d runway, dollar

CONFIGS: dict[str, dict] = {
    # ---- F1: cointegration stock pairs, residual signal --------------------
    "f1_base":   {"family": "coint", "signal": "residual", "rule": BASE, "n": 20},
    "f1_e15":    {"family": "coint", "signal": "residual", "rule": replace(BASE, entry_z=1.5), "n": 20},
    "f1_e25":    {"family": "coint", "signal": "residual", "rule": replace(BASE, entry_z=2.5, max_entry_z=3.5), "n": 20},
    "f1_x0":     {"family": "coint", "signal": "residual", "rule": replace(BASE, exit_z=0.0), "n": 20},
    "f1_hedge":  {"family": "coint", "signal": "residual", "rule": replace(BASE, mode="hedge"), "n": 20},
    "f1_tilt":   {"family": "coint", "signal": "residual", "rule": BASE, "n": 20, "gate": "tilt"},
    # ---- F2: stock vs sector ETF ------------------------------------------
    "f2_base":   {"family": "setf", "signal": "residual", "rule": replace(BASE, mode="hedge"), "n": 30},
    "f2_dollar": {"family": "setf", "signal": "residual", "rule": BASE, "n": 30},
    "f2_e25":    {"family": "setf", "signal": "residual", "rule": replace(BASE, entry_z=2.5, max_entry_z=3.5, mode="hedge"), "n": 30},
    "f2_long":   {"family": "setf", "signal": "residual", "rule": replace(BASE, mode="long_only"), "n": 30},
    "f2_tilt":   {"family": "setf", "signal": "residual", "rule": replace(BASE, mode="hedge"), "n": 30, "gate": "tilt"},
    # ---- F3: ETF vs ETF ----------------------------------------------------
    "f3_base":   {"family": "etf", "signal": "residual", "rule": replace(BASE, min_days_left=20), "n": 10},
    "f3_e15":    {"family": "etf", "signal": "residual", "rule": replace(BASE, entry_z=1.5, min_days_left=20), "n": 10},
    "f3_hedge":  {"family": "etf", "signal": "residual", "rule": replace(BASE, mode="hedge", min_days_left=20), "n": 10},
    "f3_stop":   {"family": "etf", "signal": "residual", "rule": replace(BASE, stop_z=4.0, min_days_left=20), "n": 10},
    # ---- F4: dual share classes -------------------------------------------
    "f4_base":   {"family": "dual", "signal": "logspread", "rule": replace(BASE, entry_z=1.5, exit_z=0.25, max_entry_z=4.0, min_days_left=10), "n": 3},
    "f4_e2":     {"family": "dual", "signal": "logspread", "rule": replace(BASE, entry_z=2.0, exit_z=0.25, max_entry_z=4.0, min_days_left=10), "n": 3},
    # ---- F5: ratio-MA 20/20/20 (user baseline spec) on SSD pairs ----------
    "f5_ratio":  {"family": "ssd", "signal": "ratio", "rule": replace(BASE, max_entry_z=4.0), "n": 20},
    "f5_ratio_e25": {"family": "ssd", "signal": "ratio", "rule": replace(BASE, entry_z=2.5, max_entry_z=4.5), "n": 20},
    "f5_ratio_4040": {"family": "ssd", "signal": "ratio", "rule": replace(BASE, max_entry_z=4.0), "n": 20,
                      "signal_kwargs": {"ma": 40, "ma2": 40, "vol": 40}},
    "f5_dist":   {"family": "ssd", "signal": "distance", "rule": BASE, "n": 20},   # incumbent-equivalent control on clean data
    "f5_dist_tilt": {"family": "ssd", "signal": "distance", "rule": BASE, "n": 20, "gate": "tilt"},
    "f5_ratio_tilt": {"family": "ssd", "signal": "ratio", "rule": replace(BASE, max_entry_z=4.0), "n": 20, "gate": "tilt"},
    # ---- stage 2 (declared 2026-07-18 after first-pass leaderboard) --------
    # survivors + validated regime gate, staged entry, overlay ablations,
    # parameter neighbors for the plateau bar
    "f5dt_hot":    {"family": "ssd", "signal": "distance", "rule": BASE, "n": 20, "gate": "tilt", "hot_ref": "f5_dist_tilt"},
    "f5dt_stage":  {"family": "ssd", "signal": "distance", "rule": replace(BASE, staged=(2.0, 3.0)), "n": 20, "gate": "tilt"},
    "f5dt_hot_stage": {"family": "ssd", "signal": "distance", "rule": replace(BASE, staged=(2.0, 3.0)), "n": 20, "gate": "tilt", "hot_ref": "f5_dist_tilt"},
    "f5d_mindiff": {"family": "ssd", "signal": "distance", "rule": BASE, "n": 20, "gate": "mindiff"},
    "f5d_veto":    {"family": "ssd", "signal": "distance", "rule": BASE, "n": 20, "gate": "veto"},
    "f1t_hot":     {"family": "coint", "signal": "residual", "rule": BASE, "n": 20, "gate": "tilt", "hot_ref": "f1_tilt"},
    "f5dt_e175":   {"family": "ssd", "signal": "distance", "rule": replace(BASE, entry_z=1.75), "n": 20, "gate": "tilt"},
    "f5dt_e225":   {"family": "ssd", "signal": "distance", "rule": replace(BASE, entry_z=2.25), "n": 20, "gate": "tilt"},
    "f5dt_x025":   {"family": "ssd", "signal": "distance", "rule": replace(BASE, exit_z=0.25), "n": 20, "gate": "tilt"},
    "f5dt_x075":   {"family": "ssd", "signal": "distance", "rule": replace(BASE, exit_z=0.75), "n": 20, "gate": "tilt"},
    "f5dt_n15":    {"family": "ssd", "signal": "distance", "rule": BASE, "n": 15, "gate": "tilt"},
    "f5dt_n25":    {"family": "ssd", "signal": "distance", "rule": BASE, "n": 25, "gate": "tilt"},
    "f5dt_persist2": {"family": "ssd", "signal": "distance", "rule": replace(BASE, persist=2), "n": 20, "gate": "tilt"},
    "f5dt_confirm":  {"family": "ssd", "signal": "distance", "rule": replace(BASE, confirm=True), "n": 20, "gate": "tilt"},
}

CONFIRM = replace(BASE, confirm=True)

# stage 3 (user-requested 2026-07-18, AFTER holdout consumption — eval-window
# and era-replication evidence only; ledger grows accordingly)
CONFIGS |= {
    # rank slices of the SSD ranking (skip = drop the tightest-ranked pairs)
    "s3_top10":   {"family": "ssd", "signal": "distance", "rule": CONFIRM, "n": 10, "gate": "tilt"},
    "s3_r6_20":   {"family": "ssd", "signal": "distance", "rule": CONFIRM, "n": 20, "skip": 5, "gate": "tilt"},
    "s3_r11_20":  {"family": "ssd", "signal": "distance", "rule": CONFIRM, "n": 20, "skip": 10, "gate": "tilt"},
    "s3_r11_30":  {"family": "ssd", "signal": "distance", "rule": CONFIRM, "n": 30, "skip": 10, "gate": "tilt"},
    # same-day dislocation trigger (long leg fell ≥ x% today OR short leg rose ≥ x%)
    "s3_move1":   {"family": "ssd", "signal": "distance", "rule": replace(CONFIRM, move_pct=0.01), "n": 20, "gate": "tilt"},
    "s3_move2":   {"family": "ssd", "signal": "distance", "rule": replace(CONFIRM, move_pct=0.02), "n": 20, "gate": "tilt"},
    "s3_move2_nc": {"family": "ssd", "signal": "distance", "rule": replace(BASE, move_pct=0.02), "n": 20, "gate": "tilt"},
    # score-distance / score-level gates on the confirm base
    "s3_mindiff5":  {"family": "ssd", "signal": "distance", "rule": CONFIRM, "n": 20, "gate": "mindiff", "gate_kwargs": {"min_diff": 5.0}},
    "s3_mindiff20": {"family": "ssd", "signal": "distance", "rule": CONFIRM, "n": 20, "gate": "mindiff", "gate_kwargs": {"min_diff": 20.0}},
    "s3_lv7050":  {"family": "ssd", "signal": "distance", "rule": CONFIRM, "n": 20, "gate": "levels", "gate_kwargs": {"long_min": 70.0, "short_max": 50.0}},
    "s3_lv6040":  {"family": "ssd", "signal": "distance", "rule": CONFIRM, "n": 20, "gate": "levels", "gate_kwargs": {"long_min": 60.0, "short_max": 40.0}},
}

# stage 4 (user-requested 2026-07-18: stops, partial profit, one-sided legs,
# vol-leg). Same evidence caveat as stage 3 (eval window only).
CONFIGS |= {
    "s4_stop35":   {"family": "ssd", "signal": "distance", "rule": replace(CONFIRM, stop_z=3.5), "n": 20, "gate": "tilt"},
    "s4_stop45":   {"family": "ssd", "signal": "distance", "rule": replace(CONFIRM, stop_z=4.5), "n": 20, "gate": "tilt"},
    "s4_half1":    {"family": "ssd", "signal": "distance", "rule": replace(CONFIRM, half_exit_z=1.0), "n": 20, "gate": "tilt"},
    "s4_half75":   {"family": "ssd", "signal": "distance", "rule": replace(CONFIRM, half_exit_z=0.75), "n": 20, "gate": "tilt"},
    "s4_long":     {"family": "ssd", "signal": "distance", "rule": replace(CONFIRM, mode="long_only"), "n": 20, "gate": "tilt"},
    "s4_short":    {"family": "ssd", "signal": "distance", "rule": replace(CONFIRM, mode="short_only"), "n": 20, "gate": "tilt"},
    "s4_legs7030": {"family": "ssd", "signal": "distance", "rule": replace(CONFIRM, mode="score_legs", leg_long_min=70.0, leg_short_max=30.0), "n": 20, "gate": "none"},
    "s4_legs6040": {"family": "ssd", "signal": "distance", "rule": replace(CONFIRM, mode="score_legs", leg_long_min=60.0, leg_short_max=40.0), "n": 20, "gate": "none"},
    "s4_volleg":   {"family": "ssd", "signal": "distance", "rule": replace(CONFIRM, mode="vol_leg"), "n": 20, "gate": "tilt"},
}

# stage 5: combinations of the stage-3/4 winners (declared before running)
CONFIGS |= {
    "s5_half_legs":  {"family": "ssd", "signal": "distance", "rule": replace(CONFIRM, half_exit_z=1.0, mode="score_legs", leg_long_min=70.0, leg_short_max=30.0), "n": 20, "gate": "none"},
    "s5_half_slice": {"family": "ssd", "signal": "distance", "rule": replace(CONFIRM, half_exit_z=1.0), "n": 30, "skip": 10, "gate": "tilt"},
    "s5_all":        {"family": "ssd", "signal": "distance", "rule": replace(CONFIRM, half_exit_z=1.0, mode="score_legs", leg_long_min=70.0, leg_short_max=30.0), "n": 30, "skip": 10, "gate": "none"},
    # missing matrix cell (declared 2026-07-18): legs on the slice, no half-exit
    "s6_legs_slice": {"family": "ssd", "signal": "distance", "rule": replace(CONFIRM, mode="score_legs", leg_long_min=70.0, leg_short_max=30.0), "n": 30, "skip": 10, "gate": "none"},
}

# GOOG/GOOGL exclusion ablation on the top-5 (declared 2026-07-18; the pair
# is dropped at selection time, book not backfilled — clean counterfactual)
_XG = {"exclude_pairs": [["GOOG", "GOOGL"]]}
CONFIGS |= {
    "x_half1":    {**CONFIGS["s4_half1"], **_XG},
    "x_legs7030": {**CONFIGS["s4_legs7030"], **_XG},
    "x_r11_20":   {**CONFIGS["s3_r11_20"], **_XG},
    "x_confirm":  {**CONFIGS["f5dt_confirm"], **_XG},
    "x_legs6040": {**CONFIGS["s4_legs6040"], **_XG},
}

DISCOVERY = {
    "coint": lambda tr, mem, sec, fs, fe, n: dsc.coint_pairs(tr, mem, sec, fs, fe, max_pairs=n),
    "setf": lambda tr, mem, sec, fs, fe, n: dsc.stock_etf_pairs(tr, mem, sec, fs, fe, max_pairs=n),
    "etf": lambda tr, mem, sec, fs, fe, n: dsc.etf_etf_pairs(tr, fs, fe, max_pairs=n),
    "dual": lambda tr, mem, sec, fs, fe, n: dsc.share_class_pairs(tr, fs, fe),
    "ssd": lambda tr, mem, sec, fs, fe, n: dsc.ssd_pairs_specs(None, tr, mem, sec, fs, fe, n=n),
}


def formation_dates(w) -> tuple[str, str, str]:
    f_end = (pd.Timestamp(w.test_start) - pd.Timedelta(days=1)).date().isoformat()
    f_start = (pd.Timestamp(w.test_start)
               - pd.DateOffset(months=FORMATION_MONTHS)).date().isoformat()
    return f_start, f_end, w.test_start


def run_config(name: str, cfg: dict, tr: pd.DataFrame, sectors: pd.Series,
               composites: dict, windows, spec_cache: dict,
               cost: float = 0.0010, borrow: float = 0.0025,
               hot_gate_daily: pd.Series | None = None) -> dict:
    all_payoff, all_open, all_trades = [], [], []
    pairs_per_window = {}
    for w in windows:
        f_start, f_end, _ = formation_dates(w)
        key = (cfg["family"], cfg.get("n"), w.label)
        if key not in spec_cache:
            members = members_as_of(w.test_start)
            spec_cache[key] = DISCOVERY[cfg["family"]](
                tr, members, sectors, f_start, f_end, cfg.get("n", 20))
        specs = spec_cache[key][cfg.get("skip", 0):]
        for ex in cfg.get("exclude_pairs", []):
            specs = [s for s in specs if {s.a, s.b} != set(ex)]
        pairs_per_window[w.label] = len(specs)
        if not specs:
            continue
        fpx = tr.loc[(tr.index >= f_start) & (tr.index <= f_end)].dropna(how="all")
        anchor = fpx.index[-1]
        gate = None
        if cfg.get("gate"):
            if hot_gate_daily is not None:
                gate = HotStreakGate(composites, cfg["gate"], hot_gate_daily)
            else:
                gate = RVGate(composites, cfg["gate"], **cfg.get("gate_kwargs", {}))
        elif hot_gate_daily is not None:
            gate = HotStreakGate(composites, "none", hot_gate_daily)
        payoff, open_u, trades = simulate_rv_window(
            tr, specs, cfg["signal"], anchor, w.test_start, w.test_end,
            rule=cfg["rule"], gate=gate, cost=cost, borrow=borrow,
            signal_kwargs=cfg.get("signal_kwargs"))
        if payoff.empty:
            continue
        all_payoff.append(payoff)
        all_open.append(open_u)
        all_trades.extend(trades)
    if not all_payoff:
        return {"name": name, "error": "no trades"}
    n_slots = max(cfg.get("n", 20) - cfg.get("skip", 0), 1)
    daily_c = pd.concat([committed_returns(p, n_slots) for p in all_payoff]).sort_index()
    daily_i = pd.concat([invested_returns(p, o, floor=min(8, n_slots))
                         for p, o in zip(all_payoff, all_open)]).sort_index()
    return {"name": name, "daily_committed": daily_c, "daily_invested": daily_i,
            "trades": all_trades, "pairs_per_window": pairs_per_window}


def summarize(res: dict, tr: pd.DataFrame) -> dict:
    bench = bench_returns(tr)
    di = res["daily_invested"]
    dc = res["daily_committed"]
    era_a = di[di.index < ERA_SPLIT]
    era_b = di[di.index >= ERA_SPLIT]
    row = {"config": res["name"],
           **{f"inv_{k}": v for k, v in perf_stats(di, bench).items()},
           "com_sharpe": perf_stats(dc).get("sharpe", np.nan),
           "com_cagr": perf_stats(dc).get("cagr", np.nan),
           "eraA_cagr": perf_stats(era_a).get("cagr", np.nan),
           "eraA_sharpe": perf_stats(era_a).get("sharpe", np.nan),
           "eraB_cagr": perf_stats(era_b).get("cagr", np.nan),
           "eraB_sharpe": perf_stats(era_b).get("sharpe", np.nan),
           **trade_stats(res["trades"])}
    return row


def save_result(res: dict, outdir: Path) -> None:
    d = outdir / res["name"]
    d.mkdir(parents=True, exist_ok=True)
    res["daily_committed"].to_csv(d / "daily_committed.csv", header=["ret"])
    res["daily_invested"].to_csv(d / "daily_invested.csv", header=["ret"])
    rows = [{"a": t.spec.a, "b": t.spec.b, "family": t.spec.family,
             "sector": t.spec.sector, "long": t.long, "short": t.short,
             "open": t.open_date, "close": t.close_date, "payoff": t.payoff,
             "days": t.days, "reason": t.reason, "entry_z": t.entry_z,
             "units": t.units, "long_score": t.long_score,
             "short_score": t.short_score} for t in res["trades"]]
    pd.DataFrame(rows).to_csv(d / "trades.csv", index=False)
    json.dump(res["pairs_per_window"], open(d / "pairs_per_window.json", "w"), indent=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["discover", "backtest", "leaderboard", "holdout"])
    ap.add_argument("--configs", nargs="*")
    ap.add_argument("--family")
    ap.add_argument("--cost", type=float, default=0.0010)
    ap.add_argument("--borrow", type=float, default=0.0025)
    ap.add_argument("--tag", default="")
    ap.add_argument("--i-know-the-leaderboard-is-frozen", action="store_true")
    args = ap.parse_args()

    tr = build_tr_panel()
    sectors = load_sectors()
    last_end = EVAL_LAST_END
    if args.phase == "holdout":
        if not args.i_know_the_leaderboard_is_frozen:
            raise SystemExit("holdout is locked: freeze the leaderboard in "
                             "RESEARCH_LOG.md first, then pass the flag")
        last_end = HOLDOUT_LAST_END
    windows = semiannual_windows(last_end=last_end)
    if args.phase == "holdout":
        windows = [w for w in windows if w.test_start > EVAL_LAST_END]
    composites = load_composites(semiannual_windows(), sectors)

    if args.phase == "discover":
        discover(tr, sectors, windows)
        return
    if args.phase == "leaderboard":
        lb = pd.read_csv(OUT / "leaderboard.csv")
        cols = ["config", "inv_cagr", "inv_sharpe", "inv_sortino", "inv_max_dd",
                "inv_beta_SPY", "inv_alpha_SPY", "eraA_sharpe", "eraB_sharpe",
                "n_trades", "hit_rate", "pct_converged"]
        print(lb[[c for c in cols if c in lb.columns]]
              .sort_values("inv_sharpe", ascending=False).to_string(index=False))
        return

    names = args.configs or [k for k in CONFIGS
                             if not args.family
                             or k.startswith(args.family.lower())]
    spec_cache: dict = {}
    rows = []
    outdir = OUT / (("holdout_" if args.phase == "holdout" else "") + (args.tag or "eval"))
    for name in names:
        cfg = CONFIGS[name]
        hot_daily = None
        if cfg.get("hot_ref"):
            ref = OUT / "eval" / cfg["hot_ref"] / "daily_committed.csv"
            hot_daily = pd.read_csv(ref, index_col=0)["ret"]
        res = run_config(name, cfg, tr, sectors, composites, windows,
                         spec_cache, cost=args.cost, borrow=args.borrow,
                         hot_gate_daily=hot_daily)
        if "error" in res:
            print(f"{name}: {res['error']}")
            continue
        save_result(res, outdir)
        row = summarize(res, tr)
        rows.append(row)
        print(f"{name}: inv Sharpe {row.get('inv_sharpe', float('nan')):.2f} "
              f"CAGR {row.get('inv_cagr', float('nan')):.2%} "
              f"trades {row.get('n_trades')} hit {row.get('hit_rate', float('nan')):.0%}")
    if rows:
        suffix = "_holdout" if args.phase == "holdout" else (
            f"_{args.tag}" if args.tag else "")
        lb_path = OUT / f"leaderboard{suffix}.csv"
        new = pd.DataFrame(rows)
        if lb_path.exists():
            old = pd.read_csv(lb_path)
            new = pd.concat([old[~old.config.isin(new.config)], new])
        new.to_csv(lb_path, index=False)


def discover(tr, sectors, windows) -> None:
    """Descriptive candidate inventory: PIT pair lists per window + a
    full-eval-period diagnostic table for the union of selected pairs."""
    OUT.mkdir(parents=True, exist_ok=True)
    per_window = []
    union: dict[tuple, dsc.PairSpec] = {}
    for w in windows:
        f_start, f_end, _ = formation_dates(w)
        members = members_as_of(w.test_start)
        fams = {
            "coint": dsc.coint_pairs(tr, members, sectors, f_start, f_end),
            "setf": dsc.stock_etf_pairs(tr, members, sectors, f_start, f_end),
            "etf": dsc.etf_etf_pairs(tr, f_start, f_end),
            "dual": dsc.share_class_pairs(tr, f_start, f_end),
            "ssd": dsc.ssd_pairs_specs(None, tr, members, sectors, f_start, f_end),
        }
        for fam, specs in fams.items():
            for s in specs:
                union.setdefault((s.a, s.b, fam), s)
                per_window.append({"window": w.label, "family": fam, "a": s.a,
                                   "b": s.b, "sector": s.sector, **s.diag})
        print(f"{w.label}: " + ", ".join(f"{f}={len(s)}" for f, s in fams.items()))
    pd.DataFrame(per_window).to_csv(OUT / "discovery_per_window.csv", index=False)
    inv = dsc.inventory_diagnostics(tr, list(union.values()),
                                    windows[0].test_start, windows[-1].test_end)
    inv.to_csv(OUT / "candidate_inventory.csv", index=False)
    print(f"inventory: {len(inv)} unique pairs -> {OUT/'candidate_inventory.csv'}")


if __name__ == "__main__":
    main()
