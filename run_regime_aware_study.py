"""Hierarchical, regime-aware factor model study runner.

Runs all 6 named variants (A baseline, B stable-no-VIX, C proposed model,
D dynamic-monthly ablation, C+2%floor, C+3%floor) in the same strict PIT
semiannual walk-forward (rolling-5y, 2017-2026) every other regime study in
this repo uses, and writes the full comparison + evidence tables.

Outputs (output/regime_aware/):
    monthly_subfactor_cache.csv    -- full per-(sub_factor, month) IC/spread cache
    subfactor_evidence_latest.csv  -- long-run/recent/regime evidence + scores, latest cutoff
    selection_decisions.csv        -- selected/rejected subs + reasons, latest cutoff
    parent_formulas.csv            -- intra-parent weights, latest cutoff
    parent_evidence.csv            -- parent-level evidence/utility, latest cutoff
    vix_probabilities_latest.csv   -- current smooth VIX regime probabilities + shrunk
                                       per-regime IC/n_eff summary
    walkforward_comparison.csv     -- one row per (window, variant), full metric suite
    state_history.csv              -- full monthly (cutoff, variant, parent) realized
                                       weight + active-subfactor log
    churn_chart.png                 -- variant C's monthly realized parent-weight path
    REGIME_AWARE_REPORT.md

Usage:
    python run_regime_aware_study.py
    python run_regime_aware_study.py --rebuild-panel
    python run_regime_aware_study.py --first-test-year 2017 --last-end 2026-06-30
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from backtesting import data_loader as dl
from data.db import get_db
from research import compute_forward_returns
from research.subset_selection import correlation_matrix
from research.walkforward.regime_aware_evidence import as_of, build_monthly_cache
from research.walkforward.regime_aware_parents import parent_utility_table
from research.walkforward.regime_aware_scoring import score_table
from research.walkforward.regime_aware_selection import select_parent_subfactors
from research.walkforward.regime_aware_walkforward import run_walkforward_comparison
from research.walkforward.regime_probability import (
    REGIME_ORDER, SHRINKAGE_K, regime_probabilities, shrink_regime_ic,
)
from research.walkforward.compose import build_parent_panel
from research.walkforward.splits import PANEL_START
from research.walkforward.vix_regime_study import _vix_spot, load_vix_series

from run_walkforward import PRICE_END, _load_panel

OUT_DIR = Path("output/regime_aware")
REPORT_METRIC_COLS = ["ic_6m", "ic_ir_6m", "q5q1_ann", "hit_rate_6m", "cagr", "sharpe",
                      "sortino", "max_drawdown", "spy_excess_cagr", "spy_ir", "avg_turnover",
                      "mean_abs_weight_chg", "subfactor_churn_per_year"]

# Single-ingredient ablation battery (2026-07-13): each variant isolates exactly
# ONE evidence source for expected_ic (the 30% "adaptive" parent-weight component
# and sub-factor selection within each parent), all sharing the same hysteresis +
# weight-cap discipline (spec Section 4) and no diversification floor, so
# differences are attributable to the evidence choice alone, not to confounded
# recency+VIX blending (as "C" does) or to floor effects. "C" is kept as the
# original spec-proposed reference blend, not part of the ablation itself.
STUDY_VARIANTS = ["5Y", "B", "12M", "VIXOnly", "VIX+12M", "VIX+LongRun", "C"]
STUDY_INTRO = (
    "Rolling-5y semiannual walk-forward, same 19-window grid as every other regime "
    "study in this repo. All 7 variants run the same continuous monthly evidence + "
    "hysteresis + capped-weight pipeline (spec Section 4) with identical discipline and "
    "no diversification floor -- the only thing that varies is which single evidence "
    "source (or 50/50 pair) drives expected_ic: 5Y = trailing-5-year IC only; "
    "B = full-history (\"long-run\") IC only; 12M = trailing-12-month IC only; "
    "VIXOnly = pure VIX-regime-shrunk IC, no long-run or recency at all; "
    "VIX+12M = 50% VIX-regime + 50% trailing-12-month, no long-run anchor; "
    "VIX+LongRun = 50% VIX-regime + 50% full-history, no short-window recency; "
    "C = the original spec-proposed reference blend (50% long-run + 25% recent-24m + "
    "25% VIX-regime), kept for comparison, not part of the ablation itself.")


def _write_latest_evidence(panel, matrix, vix, sub_cache, out_dir: Path) -> pd.DataFrame:
    """Deliverables 1-5: evidence table, selection decisions, formulas, parent
    evidence/utility, and current VIX probabilities, all as of the latest cutoff.

    Returns the VIX-probabilities/shrunk-regime table (also written to
    ``vix_probabilities_latest.csv``) so ``main()`` can additionally render it as
    a table inside ``REGIME_AWARE_REPORT.md`` (spec Section 4 deliverable 2 asks
    for "Table in report", not just a CSV artifact).
    """
    cutoff = panel.rebal_dates[-1]
    evidence = as_of(sub_cache, cutoff, vix)
    scored = score_table(evidence)
    scored.to_csv(out_dir / "subfactor_evidence_latest.csv", index=False)

    fwd_by_h = compute_forward_returns(matrix, panel.rebal_dates, {"3M": 3, "6M": 6})
    corr = correlation_matrix(panel)
    decision_rows: list[dict] = []
    formula_rows: list[dict] = []
    parent_sub_weights: dict[str, dict[str, float]] = {}
    for parent in panel.parent_keys:
        p_scored = scored[scored.parent == parent]
        if p_scored.empty:
            continue
        result = select_parent_subfactors(p_scored, corr, panel, fwd_by_h)
        for dec in result["decisions"]:
            decision_rows.append({"parent": parent, **dec})
        for sub, w in result["weights"].items():
            formula_rows.append({"parent": parent, "sub_factor": sub, "weight": w})
        parent_sub_weights[parent] = result["weights"]
    pd.DataFrame(decision_rows).to_csv(out_dir / "selection_decisions.csv", index=False)
    pd.DataFrame(formula_rows).to_csv(out_dir / "parent_formulas.csv", index=False)

    parent_panel = build_parent_panel(panel, parent_sub_weights)
    parent_cache = build_monthly_cache(parent_panel, matrix, vix)
    parent_evidence = as_of(parent_cache, cutoff, vix)
    parent_utility_table(parent_evidence).to_csv(out_dir / "parent_evidence.csv", index=False)

    vix_now = _vix_spot(vix, cutoff)
    probs = regime_probabilities(vix_now)
    prob_df = pd.DataFrame([{"regime": r, "probability": probs[r]} for r in REGIME_ORDER])
    vix_probs = prob_df.merge(_shrunk_regime_summary(scored), on="regime", how="left")
    vix_probs.to_csv(out_dir / "vix_probabilities_latest.csv", index=False)
    return vix_probs


def _shrunk_regime_summary(evidence: pd.DataFrame) -> pd.DataFrame:
    """Deliverable 2 (spec Section 4): per-regime shrunk-IC summary. For each of the
    3 VIX regimes, shrinks every subfactor's regime IC toward its long-run mean via
    shrink_regime_ic() (spec Section 2), then averages the shrunk IC and n_eff across
    all subfactor rows (NaNs dropped, not zero-filled, so one thin subfactor can't
    drag down the regime average). Returns one row per regime:
    {regime, avg_shrunk_ic, avg_n_eff} -- probability is a function of the current
    VIX spot, not of the evidence table, so the caller merges that in separately
    (see _write_latest_evidence)."""
    rows: list[dict] = []
    for regime in REGIME_ORDER:
        key = regime.split(" ")[0].lower()
        if evidence.empty:
            rows.append({"regime": regime, "avg_shrunk_ic": float("nan"),
                        "avg_n_eff": float("nan")})
            continue
        shrunk = evidence.apply(
            lambda row, key=key: shrink_regime_ic(
                row[f"regime_ic_{key}"], row["long_run_mean_ic"],
                row[f"regime_n_eff_{key}"], k=SHRINKAGE_K),
            axis=1)
        rows.append({
            "regime": regime,
            "avg_shrunk_ic": float(shrunk.mean(skipna=True)),
            "avg_n_eff": float(evidence[f"regime_n_eff_{key}"].mean(skipna=True)),
        })
    return pd.DataFrame(rows)


def _fmt(v, fmt: str = ".3f") -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{v:{fmt}}"


def _recommendation(
    comparison: pd.DataFrame, *, baseline: str = "A", compare: str = "C",
    turnover_benchmark: str = "D",
) -> list[str]:
    """Deliverable 8: a data-driven verdict derived from the actual walk-forward
    numbers -- never a placeholder, since the comparison table is always available
    by the time this runs.

    ``baseline``/``compare`` name the two variants the verdict is built from
    (default "A"/"C" -- current production vs. the proposed regime-aware model).
    ``turnover_benchmark``, if present in ``comparison``, adds a churn comparison
    against it (default "D", the deliberately-undisciplined dynamic-monthly
    ablation) -- silently omitted if that variant wasn't run, so a study that
    drops it (e.g. one comparing evidence-weighting modes with equal discipline
    instead) still gets a coherent verdict, just without the churn gate."""
    if comparison.empty:
        return ["No walk-forward windows were evaluated -- check the study inputs."]
    summary = comparison.groupby("variant")[REPORT_METRIC_COLS].mean()
    if baseline not in summary.index or compare not in summary.index:
        return [f"Variant {baseline} or {compare} did not produce results -- "
                "check the study inputs."]
    a, c = summary.loc[baseline], summary.loc[compare]
    has_turnover_benchmark = turnover_benchmark in summary.index
    bench_turnover = (float(summary.loc[turnover_benchmark, "avg_turnover"])
                      if has_turnover_benchmark else float("nan"))
    c_turnover = float(c["avg_turnover"])
    sharpe_delta = float(c["sharpe"] - a["sharpe"])
    ic_delta = float(c["ic_6m"] - a["ic_6m"])
    dd_delta = float(c["max_drawdown"] - a["max_drawdown"])   # less negative = improved
    lines = [
        f"- Sharpe: {compare} {_fmt(c['sharpe'], '.2f')} vs {baseline} {_fmt(a['sharpe'], '.2f')} "
        f"(delta {sharpe_delta:+.2f})",
        f"- IC (6M): {compare} {_fmt(c['ic_6m'])} vs {baseline} {_fmt(a['ic_6m'])} "
        f"(delta {ic_delta:+.3f})",
        f"- Max drawdown: {compare} {_fmt(c['max_drawdown'], '.1%')} vs {baseline} "
        f"{_fmt(a['max_drawdown'], '.1%')} (delta {dd_delta:+.1%})",
    ]
    if has_turnover_benchmark:
        lines.append(
            f"- Turnover: {compare} {c_turnover:.2f} vs {turnover_benchmark} "
            f"(dynamic-monthly benchmark) {bench_turnover:.2f} -- {compare} should churn "
            f"materially less than {turnover_benchmark} if the hysteresis/shrinkage "
            "machinery is earning its complexity.")
    turnover_ok = (not has_turnover_benchmark) or c_turnover < bench_turnover
    if sharpe_delta > 0 and dd_delta >= 0 and turnover_ok:
        lines.append(f"\n**Verdict:** {compare} improves risk-adjusted return and drawdown "
                     f"over {baseline}" +
                     (f" while churning less than the {turnover_benchmark} overfitting "
                      "benchmark" if has_turnover_benchmark else "") +
                     " -- supports moving to a production pilot.")
    else:
        lines.append(f"\n**Verdict:** {compare} does not clearly beat {baseline}" +
                     (f" and/or does not clearly beat {turnover_benchmark}'s churn"
                      if has_turnover_benchmark else "") +
                     " -- the added complexity is not yet earning its keep; treat "
                     "as research, not a production candidate.")
    return lines


def write_report(
    comparison: pd.DataFrame, out_dir: Path, vix_probs: pd.DataFrame | None = None, *,
    baseline: str = "A", compare: str = "C", turnover_benchmark: str = "D",
    intro: str | None = None,
) -> None:
    """Deliverable 7: walk-forward comparison, with both the full-period summary
    and the actual per-(window, variant) table in the report itself (not just a
    CSV pointer). ``vix_probs``, if given, renders Deliverable 2 (current VIX
    regime probabilities + shrunk-IC summary) as a table too. ``baseline``/
    ``compare``/``turnover_benchmark`` are forwarded to ``_recommendation()``.
    ``intro``, if given, replaces the default paragraph describing the variant
    lineup (useful when a study run doesn't use the full default A/B/C/D(+floor)
    set)."""
    lines = [
        "# Hierarchical, Regime-Aware Factor Model — Walk-Forward Comparison",
        "",
        intro or (
            "Rolling-5y semiannual walk-forward, same 19-window grid as every other "
            "regime study in this repo. Variant A is the existing production "
            "construction, called unmodified; B/C/D run the new continuous monthly "
            "evidence + hysteresis + capped-weight pipeline (spec Section 4)."),
        "",
    ]
    if vix_probs is not None and not vix_probs.empty:
        vix_cols = ["regime", "probability", "avg_shrunk_ic", "avg_n_eff"]
        lines += [
            "## VIX Regime Probabilities",
            "",
            "| " + " | ".join(vix_cols) + " |",
            "|" + "---|" * len(vix_cols),
        ]
        for _, row in vix_probs.iterrows():
            vals = [str(row["regime"]), _fmt(row["probability"], ".1%")]
            vals += [_fmt(row[c]) for c in ("avg_shrunk_ic", "avg_n_eff")]
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")
    lines += [
        "## Full-Period Summary",
        "",
        "| Variant | n_windows | " + " | ".join(REPORT_METRIC_COLS) + " |",
        "|---|---|" + "---|" * len(REPORT_METRIC_COLS),
    ]
    if not comparison.empty:
        for variant, g in comparison.groupby("variant"):
            vals = [_fmt(g[c].mean()) for c in REPORT_METRIC_COLS]
            lines.append(f"| {variant} | {len(g)} | " + " | ".join(vals) + " |")
    lines += [
        "",
        "## Per-Window Detail",
        "",
        "See `walkforward_comparison.csv` for the full per-(window, variant) table "
        "(reproduced below).",
        "",
    ]
    detail_cols = ["window", "variant", *REPORT_METRIC_COLS]
    lines += [
        "| " + " | ".join(detail_cols) + " |",
        "|" + "---|" * len(detail_cols),
    ]
    if not comparison.empty:
        detail = comparison.sort_values(["window", "variant"])
        for _, row in detail.iterrows():
            vals = [str(row["window"]), str(row["variant"])]
            vals += [_fmt(row[c]) for c in REPORT_METRIC_COLS]
            lines.append("| " + " | ".join(vals) + " |")
    lines += [
        "",
        "## Recommendation",
        "",
    ]
    lines += _recommendation(comparison, baseline=baseline, compare=compare,
                             turnover_benchmark=turnover_benchmark)
    (out_dir / "REGIME_AWARE_REPORT.md").write_text("\n".join(lines))


def write_churn_chart(state_log: pd.DataFrame, out_dir: Path) -> None:
    """Deliverable 6 (spec Section 4): variant C's monthly realized parent-weight
    path -- one stacked area per parent, over the full monthly cutoff grid (not just
    the semiannual test boundaries), so the churn/stability of the proposed model's
    weights is visible directly rather than only via the summary churn metrics."""
    if state_log.empty:
        return
    sub = state_log[state_log["variant"] == "C"]
    mat = sub.pivot(index="cutoff", columns="parent", values="weight").sort_index()
    dates = pd.to_datetime(mat.index)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(dates, mat.T.to_numpy(dtype=float), labels=mat.columns, alpha=0.85)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Realized parent weight")
    ax.set_title("Variant C — Monthly Realized Parent-Weight Path (Churn)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "churn_chart.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-panel", action="store_true")
    parser.add_argument("--first-test-year", type=int, default=2017)
    parser.add_argument("--last-end", type=str, default=None)
    parser.add_argument("--out", type=str, default=str(OUT_DIR))
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Hierarchical Regime-Aware Factor Model Study ===\n")
    print("[1/4] Loading panel + price matrix + VIX + sectors...")
    panel = _load_panel(args.rebuild_panel)
    with get_db() as db:
        matrix = dl.load_price_matrix(db, panel.universe, PANEL_START, PRICE_END)
        vix = load_vix_series(db)
        sectors = dl.global_sectors(db)
    print(f"      {len(panel.rebal_dates)} rebalances, {len(panel.universe)} names, "
          f"sectors for {sectors.notna().sum()} names.")

    print("\n[2/4] Building the monthly PIT evidence cache...")
    sub_cache = build_monthly_cache(panel, matrix, vix)
    sub_cache.to_csv(out_dir / "monthly_subfactor_cache.csv", index=False)

    print("\n[3/4] Writing latest-cutoff evidence, selection, and parent tables...")
    vix_probs = _write_latest_evidence(panel, matrix, vix, sub_cache, out_dir)

    print("\n[4/4] Running the walk-forward comparison "
          f"({', '.join(STUDY_VARIANTS)})...")
    comparison, state_log = run_walkforward_comparison(
        panel, matrix, vix, sectors,
        first_test_year=args.first_test_year, last_end=args.last_end,
        variant_names=STUDY_VARIANTS, include_baseline_a=False)
    comparison.to_csv(out_dir / "walkforward_comparison.csv", index=False)
    state_log.to_csv(out_dir / "state_history.csv", index=False)

    write_report(comparison, out_dir, vix_probs, baseline="B", compare="VIXOnly",
                 turnover_benchmark="12M", intro=STUDY_INTRO)
    write_churn_chart(state_log, out_dir)
    print(f"\nDone. Key files:\n  {out_dir / 'REGIME_AWARE_REPORT.md'}\n"
          f"  {out_dir / 'walkforward_comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
