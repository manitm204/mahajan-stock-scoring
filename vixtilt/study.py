"""Orchestrates the full study: windows × variants → scores, portfolios, statistics.

Stage A (cached): per-window frozen baseline (see :mod:`vixtilt.baseline`).
Stage B (fast):   per (window, variant, test rebalance) overlay decision → adjusted
parent weights → recomputed composite → global monthly portfolio simulation per
variant/top-percentile, OOS IC/Q5-Q1, holdings overlap vs baseline, PIT check log.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import backtest as bt
from .baseline import (CACHE_DIR, WindowBaseline, assert_window_pit, forward_returns,
                       load_or_build)
from .overlay import DEFAULT_VARIANTS, VariantSpec, WindowOverlay
from .windows import Window, block_of, semiannual_windows

OOS_HORIZONS = {"1M": 1, "3M": 3, "6M": 6}
TOP_PCTS = (0.10, 0.20, 0.30)
COST_PER_SIDE = 0.0010          # 10 bps per unit traded notional, identical everywhere


@dataclass
class StudyResult:
    variants: list[VariantSpec]
    windows: list[Window]
    scores: dict[str, dict[str, pd.Series]] = field(default_factory=dict)
    #   variant -> {test date -> composite Series}
    decisions: pd.DataFrame = field(default_factory=pd.DataFrame)
    weights_long: pd.DataFrame = field(default_factory=pd.DataFrame)
    baseline_weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    portfolios: dict[tuple[str, float], pd.DataFrame] = field(default_factory=dict)
    #   (variant, top_pct) -> simulate() frame indexed by formation date
    ic: pd.DataFrame = field(default_factory=pd.DataFrame)          # date,horizon,ic,variant
    q5q1: dict[str, pd.Series] = field(default_factory=dict)        # variant -> per-date 1M
    overlap: dict[tuple[str, float], pd.DataFrame] = field(default_factory=dict)
    train_vix_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    pit_checks: pd.DataFrame = field(default_factory=pd.DataFrame)
    window_of_date: dict[str, str] = field(default_factory=dict)


def run_study(panel, matrix: pd.DataFrame, sectors: pd.Series, vix: pd.Series, *,
              variants: list[VariantSpec] = DEFAULT_VARIANTS,
              first_test_year: int = 2017, last_end: str = "2026-06-30",
              cache_dir: Path = CACHE_DIR, rebuild_cache: bool = False,
              top_pcts=TOP_PCTS, cost_per_side: float = COST_PER_SIDE,
              verbose: bool = True) -> StudyResult:
    wins = semiannual_windows(first_test_year, last_end)
    res = StudyResult(variants=list(variants), windows=wins,
                      scores={v.name: {} for v in variants})

    dec_rows: list[dict] = []
    w_rows: list[dict] = []
    base_w_rows: list[dict] = []
    vix_rows: list[dict] = []
    pit_rows: list[dict] = []

    if verbose:
        print(f"Stage A/B over {len(wins)} windows × {len(variants)} variants")
    for win in wins:
        wb = load_or_build(panel, matrix, win, cache_dir=cache_dir,
                           rebuild=rebuild_cache, verbose=verbose)
        for chk in assert_window_pit(wb):
            pit_rows.append({"window": win.label, "check": chk, "status": "PASS"})
        ov = WindowOverlay(wb, vix)
        tv = ov.train_vix
        vix_rows.append({
            "window": win.label, "train_start": win.train_start,
            "train_end": win.train_end, "n_train_vix_days": len(tv),
            "train_vix_median": float(pd.Series(tv).median()) if len(tv) else float("nan"),
            "train_vix_p30": float(pd.Series(tv).quantile(0.30)) if len(tv) else float("nan"),
            "train_vix_p70": float(pd.Series(tv).quantile(0.70)) if len(tv) else float("nan"),
        })
        for p, x in wb.parent_weights.items():
            base_w_rows.append({"window": win.label, "parent": p, "weight": x})
        for d in wb.test_rebals:
            res.window_of_date[d] = win.label
            frame = wb.parent_test.get(d)
            if frame is None:
                continue
            for v in variants:
                dec = ov.decide(v, d)
                res.scores[v.name][d] = bt.composite_from_parents(
                    frame, dec.weights, sectors)
                dec_rows.append({
                    "window": win.label, "variant": v.name, "date": d,
                    "vix_date": dec.vix_date, "vix": dec.vix, "vix_pct": dec.vix_pct,
                    "bucket": dec.bucket, "strength": dec.strength,
                    "n_comparable": dec.n_comparable, "shrink_lambda": dec.shrink_lambda,
                    "fallback": dec.fallback, "l1_shift": dec.l1_shift,
                })
                for p, x in dec.weights.items():
                    w_rows.append({"window": win.label, "variant": v.name,
                                   "date": d, "parent": p, "weight": x})

    res.decisions = pd.DataFrame(dec_rows)
    res.weights_long = pd.DataFrame(w_rows)
    res.baseline_weights = pd.DataFrame(base_w_rows)
    res.train_vix_summary = pd.DataFrame(vix_rows)
    res.pit_checks = pd.DataFrame(pit_rows)

    # ---- OOS forward returns on the test grid (full matrix — realised, not training)
    all_test = sorted({d for v in variants for d in res.scores[v.name]})
    fwd_by_h: dict[str, dict[str, pd.Series]] = {}
    for h, m in OOS_HORIZONS.items():
        fr = forward_returns(matrix, all_test, m)
        fwd_by_h[h] = {d: s for d, (_, s) in fr.items()}

    if verbose:
        print("Simulating portfolios / computing OOS stats…")
    ic_frames = []
    for v in variants:
        sc = res.scores[v.name]
        icf = bt.per_date_ic(sc, fwd_by_h)
        icf["variant"] = v.name
        ic_frames.append(icf)
        res.q5q1[v.name] = bt.per_date_q5q1(sc, fwd_by_h["1M"])
        for tp in top_pcts:
            res.portfolios[(v.name, tp)] = bt.simulate(sc, matrix, tp, cost_per_side)
            if v.name != "baseline":
                res.overlap[(v.name, tp)] = bt.overlap_stats(
                    sc, res.scores["baseline"], tp)
    res.ic = pd.concat(ic_frames, ignore_index=True) if ic_frames else pd.DataFrame()
    return res


# --------------------------------------------------------------------------- #
# Aggregation helpers used by report.py
# --------------------------------------------------------------------------- #
def slice_dates(index, labels_by_date: dict[str, str], want: set[str]) -> list:
    return [d for d in index if labels_by_date.get(d) in want]


def grouping_metrics(res: StudyResult, group_windows: dict[str, set[str]],
                     top_pcts=TOP_PCTS, leg: str = "net") -> pd.DataFrame:
    """Portfolio metrics per (grouping, variant, top_pct) on pooled monthly returns."""
    rows = []
    for gname, wset in group_windows.items():
        for v in res.variants:
            for tp in top_pcts:
                pf = res.portfolios.get((v.name, tp))
                if pf is None or pf.empty:
                    continue
                keep = slice_dates(pf.index, res.window_of_date, wset)
                sub = pf.loc[keep]
                if sub.empty:
                    continue
                m = bt.perf_metrics(sub[leg], sub["spy"], sub["turnover"])
                ic6 = res.ic[(res.ic["variant"] == v.name)
                             & (res.ic["horizon"] == "6M")
                             & (res.ic["date"].isin(wset_dates(res, wset)))]["ic"]
                ic1 = res.ic[(res.ic["variant"] == v.name)
                             & (res.ic["horizon"] == "1M")
                             & (res.ic["date"].isin(wset_dates(res, wset)))]["ic"]
                q = res.q5q1.get(v.name, pd.Series(dtype=float))
                q = q[q.index.isin(wset_dates(res, wset))]
                m.update({"grouping": gname, "variant": v.name, "top_pct": tp,
                          "ic_1m": float(ic1.mean()) if len(ic1) else float("nan"),
                          "ic_6m": float(ic6.mean()) if len(ic6) else float("nan"),
                          "q5q1_1m_ann": float(q.mean()) * 12 if len(q) else float("nan")})
                ol = res.overlap.get((v.name, tp))
                if ol is not None and not ol.empty:
                    keep_ol = [d for d in ol.index if res.window_of_date.get(d) in wset]
                    ols = ol.loc[keep_ol]
                    if not ols.empty:
                        m["overlap_vs_base"] = float(ols["overlap"].mean())
                        m["names_entered_avg"] = float(ols["entered"].mean())
                rows.append(m)
    return pd.DataFrame(rows)


def wset_dates(res: StudyResult, wset: set[str]) -> set[str]:
    return {d for d, lbl in res.window_of_date.items() if lbl in wset}


def group_defs(res: StudyResult) -> dict[str, dict[str, set[str]]]:
    all_w = {w.label for w in res.windows}
    per_window = {w.label: {w.label} for w in res.windows}
    blocks: dict[str, set[str]] = {}
    for w in res.windows:
        blocks.setdefault(block_of(w.label), set()).add(w.label)
    return {"window": per_window, "block": blocks, "full": {"full": all_w}}
