"""Walk-forward out-of-sample validation of the composite factor model (2015→2026).

Re-runs the entire Parent-Selection-V4 construction on train-only data for each annual
expanding split — re-selecting sub-factors, intra-parent weights, parent composition and
parent/composite weights with no look-ahead (selection forward returns capped 6 months
before the test boundary) — freezes it, scores the unseen test year, and writes nine
deliverables to ``output/walkforward/`` covering:

    1  walk-forward validation (IC/IR/hit/spread/monotonicity by year + pooled)
    2  quantile analysis (Q1-Q5, spreads, monotonicity, per-year consistency)
    3  parent & sub-factor analysis (OOS IC, correlations, selection frequency)
    4  weight stability (parent/composite weight drift)
    5  portfolio construction (top 10/20/30 % × equal / sector-neutral, full metrics)
    6  benchmark comparison vs SPY & QQQ (excess, alpha, beta, IR, rel-DD)
    7  holding-period analysis (1/3/6/12-month)
    8  training-window study (expanding vs rolling 5y/3y/2y)
       + RECOMMENDATION.md (synthesis + verdict)

Universe membership is point-in-time (survivorship-free) at every rebalance. No LLM, no
production mutation. Usage:
    python run_walkforward.py                 # full 8-section run
    python run_walkforward.py --rebuild-panel # re-score the candidate panel first
    python run_walkforward.py --no-training-study
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from backtesting import data_loader as dl
from data.config import load_config
from data.db import get_db
from research import HORIZON_MONTHS, compute_forward_returns
from research.panel import ScorePanel
from research.subfactor_expansion.panel import (build_candidate_panel,
                                                cache_key as exp_cache_key,
                                                load_cached_panel as exp_load,
                                                save_cached_panel as exp_save)
from research.walkforward import analysis, report, training_study
from research.walkforward.runner import run_splits
from research.walkforward.splits import PANEL_END, PANEL_START, resolve_splits

CACHE_DIR = Path("cache/subfactor_expansion")
OUT_DIR = Path("output/walkforward")
PRICE_END = "2026-07-06"


def _adapt(cand) -> ScorePanel:
    parent_keys = [p for p, subs in cand.candidates_by_parent.items() if subs]
    return ScorePanel(
        rebal_dates=list(cand.rebal_dates), scores=cand.scores,
        parent_keys=parent_keys,
        sub_by_parent={p: list(cand.candidates_by_parent[p]) for p in parent_keys},
        universe=list(cand.universe),
    )


def _month_ends(db, start: str, end: str) -> list[str]:
    df = db.query_df("SELECT DISTINCT date FROM daily_prices WHERE date >= ? AND date <= ? "
                     "ORDER BY date", (start, end))
    if df.empty:
        return []
    dates = pd.to_datetime(df["date"])
    return [str(g.max().date()) for _, g in dates.groupby(dates.dt.to_period("M"))]


def _load_panel(rebuild: bool) -> ScorePanel:
    ckey = CACHE_DIR / exp_cache_key(PANEL_START, PANEL_END, "monthly")
    cand = None if rebuild else exp_load(ckey)
    if cand is None:
        cfg = load_config()
        with get_db() as db:
            rebals = _month_ends(db, PANEL_START, PANEL_END)
            print(f"Building PIT candidate panel over {len(rebals)} rebalances "
                  f"({rebals[0]} → {rebals[-1]})…")
            cand = build_candidate_panel(db, rebals, cfg, verbose=True, keep_raw=False)
        exp_save(cand, ckey)
    else:
        print(f"Loaded candidate panel: {len(cand.rebal_dates)} rebalances "
              f"({cand.rebal_dates[0]}→{cand.rebal_dates[-1]}), "
              f"{len(cand.all_candidates)} candidates, {len(cand.universe)} PIT-union names.")
    return _adapt(cand)


# --------------------------------------------------------------------------- #
# Section aggregations
# --------------------------------------------------------------------------- #
def _by_year_table(splits_data: list[dict], per_split_q: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for sd in splits_data:
        label = sd["split"].label
        tic = sd["test_ic"].set_index("horizon")

        def g(h, col):
            return float(tic.loc[h, col]) if h in tic.index and col in tic.columns \
                else np.nan
        q6 = per_split_q.get(label, {}).get("6M", {})
        rows.append({
            "year": label, "n_test": sd["n_test"],
            "ic_3M": g("3M", "mean_ic"), "ir_3M": g("3M", "information_ratio"),
            "hit_3M": g("3M", "hit_rate"),
            "ic_6M": g("6M", "mean_ic"), "ir_6M": g("6M", "information_ratio"),
            "hit_6M": g("6M", "hit_rate"),
            "spread_6M_ann": q6.get("spread", {}).get("annualized") if q6 else np.nan,
            "mono_6M": q6.get("monotonic_rate") if q6 else np.nan,
        })
    return pd.DataFrame(rows)


def _subfactor_frequency(splits_data: list[dict]) -> pd.DataFrame:
    n_years = len(splits_data)
    counts: dict[tuple[str, str], list] = {}
    for sd in splits_data:
        for parent, wmap in sd["config"].sub_weights.items():
            for sub, w in wmap.items():
                c = counts.setdefault((parent, sub), [0, 0.0])
                c[0] += 1
                c[1] += w
    rows = []
    for (parent, sub), (cnt, wsum) in counts.items():
        freq = cnt / n_years if n_years else np.nan
        stability = ("persistent" if freq >= 0.75 else
                     "unstable" if freq < 0.40 else "intermittent")
        rows.append({"sub_factor": sub, "parent": parent, "n_selected": cnt,
                     "n_years": n_years, "freq": freq, "avg_weight": wsum / cnt,
                     "stability": stability})
    return pd.DataFrame(rows)


def _parent_weight_table(splits_data: list[dict]) -> pd.DataFrame:
    parents = sorted({p for sd in splits_data for p in sd["config"].parent_weights})
    rows = []
    for p in parents:
        row = {"parent": p}
        for sd in splits_data:
            row[sd["split"].label] = sd["config"].parent_weights.get(p, 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def _weight_drift(table: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    year_cols = [c for c in table.columns if c != "parent"]
    rows = []
    for _, r in table.iterrows():
        vals = np.array([r[c] for c in year_cols], dtype=float)
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
        rows.append({"parent": r["parent"], "mean": mean, "std": std,
                     "min": float(vals.min()), "max": float(vals.max()),
                     "cv": std / mean if mean > 0 else np.nan})
    # Average YoY ½·L1 turnover of the full parent-weight vector.
    mat = table[year_cols].to_numpy(dtype=float)
    l1 = [0.5 * np.abs(mat[:, i] - mat[:, i - 1]).sum() for i in range(1, mat.shape[1])]
    return pd.DataFrame(rows), float(np.mean(l1)) if l1 else float("nan")


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #
def _build_verdict(pooled_ic, pooled_q, by_year, sweep, lo_m, ls_m,
                   window_rec, sub_freq) -> dict:
    def _ic(h):
        r = pooled_ic[pooled_ic["horizon"] == h]
        return None if r.empty else r.iloc[0]
    core = [x for x in (_ic("3M"), _ic("6M"))
            if x is not None and int(x.get("n_periods", 0)) > 0]
    mean_core = float(np.mean([x["mean_ic"] for x in core])) if core else float("nan")
    hit_core = float(np.mean([x["hit_rate"] for x in core])) if core else float("nan")
    sp6 = pooled_q.get("6M", {}).get("spread", {}).get("annualized")

    yr_core = by_year[["ic_3M", "ic_6M"]].mean(axis=1).dropna()
    pos_frac = float((yr_core > 0).mean()) if not yr_core.empty else float("nan")
    n_years = int(len(yr_core))

    if not core or np.isnan(mean_core):
        head = "**Inconclusive** — insufficient out-of-sample periods to judge."
    elif mean_core > 0.02 and hit_core > 0.55 and (sp6 or 0) > 0 and pos_frac >= 0.6:
        head = (f"**Yes — the composite shows genuine out-of-sample predictive power.** "
                f"Pooled 3M/6M OOS IC averages {mean_core:+.3f} (hit {hit_core:.0%}) with a "
                f"positive top-minus-bottom spread, and the edge is positive in "
                f"{pos_frac:.0%} of the {n_years} out-of-sample years — modest but real and "
                f"broadly consistent across regimes.")
    elif mean_core > 0 and hit_core >= 0.5:
        head = (f"**Weak / borderline.** Pooled 3M/6M OOS IC is {mean_core:+.3f} (hit "
                f"{hit_core:.0%}), positive in {pos_frac:.0%} of {n_years} years — a small, "
                f"regime-sensitive edge; size positions conservatively.")
    else:
        head = (f"**Not demonstrated.** Pooled 3M/6M OOS IC is {mean_core:+.3f} (hit "
                f"{hit_core:.0%}); the composite does not reliably rank future returns "
                f"out-of-sample across {n_years} years.")

    best = sweep.sort_values("sharpe", ascending=False).iloc[0]
    portfolio = (f"Concentration and sector treatment: top{int(best['top_pct']*100)}% "
                 f"{best['mode']} is the risk-adjusted sweet spot here.")
    spread_sharpe = {h: pooled_q.get(h, {}).get("spread", {}).get("sharpe")
                     for h in ("1M", "3M", "6M", "12M")}
    valid = {h: v for h, v in spread_sharpe.items() if v is not None and not np.isnan(v)}
    if not valid:
        holding = "Holding-period edge inconclusive on this sample."
    else:
        bh_h = max(valid, key=valid.get)
        if valid[bh_h] > 0:
            holding = (f"By the well-sampled Q5−Q1 spread Sharpe, the ranking edge is "
                       f"strongest at **{bh_h}** ({valid[bh_h]:.2f}).")
        else:
            holding = (f"No horizon shows a positive Q5−Q1 spread Sharpe (best is "
                       f"**{bh_h}** at {valid[bh_h]:.2f}); the ranking signal does not "
                       f"pay at any holding period on this sample, so the 'best' hold is "
                       f"least-bad, not genuinely edge-bearing.")
    if all(m.get("sharpe") == m.get("sharpe") for m in (lo_m, ls_m)):
        better = "long-short" if ls_m["sharpe"] > lo_m["sharpe"] else "long-only"
        longshort = (f"On a Sharpe basis **{better}** wins out-of-sample (long-only "
                     f"{lo_m['sharpe']:.2f} vs long-short {ls_m['sharpe']:.2f}); the short "
                     f"leg " + ("adds risk-adjusted value." if better == "long-short"
                                else "does not pay for its drawdown/borrow here."))
    else:
        longshort = "Long-short comparison unavailable (insufficient periods)."

    persistent = (sub_freq[sub_freq["stability"] == "persistent"]["sub_factor"].tolist()
                  if not sub_freq.empty else [])
    production = (
        f"Score on the point-in-time composite with the **{window_rec.get('best_policy','expanding')}** "
        f"training window, re-selecting annually. Build the book as **top"
        f"{int(best['top_pct']*100)}% {best['mode']}**, monthly rebalance, "
        f"{max(valid, key=valid.get) if valid else '1M'}-horizon signal. Lean on the "
        f"persistent sub-factors ({', '.join(persistent[:6]) or 'the stable core'}); treat "
        f"intermittently-selected subs as optional. Expected edge ≈ pooled OOS IC "
        f"{mean_core:+.3f}; expected excess vs SPY ≈ {_pct(best.get('spy_excess_cagr'))}, "
        f"vs QQQ ≈ {_pct(best.get('qqq_excess_cagr'))} at ~{best['avg_turnover']:.2f} "
        f"turnover/rebalance.")

    caveats = [
        f"Out-of-sample span ≈ {n_years} annual expanding splits (2017-2026); confidence "
        "bands remain wide and specific regimes (2018 Q4, 2020 COVID, 2022 drawdown) can "
        "dominate individual years.",
        "Revisions data begins 2018-12 and short interest 2017-12; composites for early test "
        "years run without those parents (neutral-filled), so their edge reflects fewer "
        "signals than the live 8-parent model.",
        "Multi-month horizons overlap, understating volatility — 3M/6M/12M IRs/Sharpes are "
        "indicative, not independent-sample t-stats.",
        "Non-overlapping 6M/12M holding portfolios have few periods per year; read their "
        "CAGR/Sharpe alongside the better-sampled Q5−Q1 spreads.",
        "The LLM overlay is intentionally excluded; this validates the quant composite only. "
        "The production factors/ path retains a known insider look-ahead bug (utils.py:378) "
        "that this research path avoids — noted for completeness, not exercised here.",
    ]
    return {"headline": head, "portfolio": portfolio, "holding": holding,
            "longshort": longshort, "production": production, "caveats": caveats}


def _pct(v):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.1%}"


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Walk-forward OOS validation (2015→2026)")
    ap.add_argument("--rebuild-panel", action="store_true")
    ap.add_argument("--no-training-study", action="store_true")
    ap.add_argument("--splits", default="expanding",
                    help="training-window scheme for the main scorecard "
                         "(expanding | rolling5y | rolling3y | rolling2y)")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = _load_panel(args.rebuild_panel)
    with get_db() as db:
        matrix = dl.load_price_matrix(db, panel.universe, PANEL_START, PRICE_END)
        sectors = dl.global_sectors(db)
    print(f"Price matrix: {matrix.shape[1]} tickers × {matrix.shape[0]} days; "
          f"sectors for {sectors.notna().sum()} names.")

    # ---- main annual walk-forward (Sections 1-7) ----
    print(f"\n=== Annual walk-forward (splits={args.splits}) ===")
    run = run_splits(panel, resolve_splits(args.splits), matrix, sectors)
    if not run.splits_data:
        raise SystemExit("no usable splits — check data span / panel coverage")

    pooled_fwd = compute_forward_returns(matrix, list(run.pooled_scores), HORIZON_MONTHS)
    pooled_ic = analysis.composite_ic(run.pooled_scores, pooled_fwd)
    pooled_q = analysis.quantile_analysis(run.pooled_scores, pooled_fwd)
    per_split_q = {lbl: analysis.quantile_analysis(
        sc, compute_forward_returns(matrix, list(sc), HORIZON_MONTHS))
        for lbl, sc in run.per_split_scores.items()}
    by_year = _by_year_table(run.splits_data, per_split_q)

    parent_ic = analysis.parent_oos_ic(run.parent_scores, pooled_fwd)
    parent_corr = analysis.parent_correlation(run.parent_scores)
    sub_freq = _subfactor_frequency(run.splits_data)
    pw_table = _parent_weight_table(run.splits_data)
    drift, avg_l1 = _weight_drift(pw_table)

    sweep = analysis.portfolio_sweep(run.pooled_scores, matrix, sectors)
    hold = analysis.holding_period_sweep(run.pooled_scores, matrix, sectors)
    lo, ls = analysis.long_vs_longshort(run.pooled_scores, matrix, sectors)

    # ---- Section 8 — training-window study ----
    if args.no_training_study:
        study = pd.DataFrame()
        window_rec = {"best_policy": "expanding",
                      "text": "Training-window study skipped (--no-training-study)."}
    else:
        print("\n=== Training-window study (Section 8) ===")
        study = training_study.run_training_study(panel, matrix, sectors)
        window_rec = training_study.recommend_window(study)

    verdict = _build_verdict(pooled_ic, pooled_q, by_year, sweep, lo.metrics, ls.metrics,
                             window_rec, sub_freq)

    # ---- write deliverables ----
    report.write_walkforward_report(out_dir, run.splits_data, pooled_ic, by_year, run.notes)
    report.write_quantile_report(out_dir, pooled_q, per_split_q)
    report.write_parent_report(out_dir, parent_ic, parent_corr, sub_freq, pw_table)
    report.write_weight_stability_report(out_dir, pw_table, drift, avg_l1)
    report.write_portfolio_report(out_dir, sweep)
    report.write_benchmark_report(out_dir, sweep)
    report.write_holding_report(out_dir, hold)
    if not study.empty:
        report.write_training_study_report(out_dir, study, window_rec)
    report.write_recommendation(
        out_dir, pooled_ic=pooled_ic, pooled_q=pooled_q, sweep=sweep, hold=hold,
        long_only=lo.metrics, long_short=ls.metrics, sub_freq=sub_freq,
        parent_ic=parent_ic, study=study, window_rec=window_rec, verdict=verdict)

    print(f"\n{verdict['headline']}\n")
    print(f"Wrote reports + CSVs to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
