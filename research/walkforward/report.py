"""Render the walk-forward deliverables (markdown + backing CSVs).

Pure formatting: every number is computed by :mod:`research.walkforward.selection` /
``analysis`` / ``portfolio`` / ``training_study`` and handed in. Writers dump the
underlying frames to CSV alongside each report so the tables are auditable.

Deliverables (→ ``output/walkforward/``):
    WALKFORWARD_REPORT.md        Section 1 — OOS IC/IR/hit/spread/monotonicity by year + pooled
    quantile_analysis.md         Section 2 — Q1-Q5 tables, spreads, monotonicity, per-year
    parent_subfactor_analysis.md Section 3 — parent OOS IC, correlations, sub selection frequency
    weight_stability.md          Section 4 — parent/composite weight stability + drift
    portfolio_analysis.md        Section 5 — top 10/20/30 % × EW/sector-neutral, full metrics
    benchmark_comparison.md      Section 6 — vs SPY & QQQ: excess, alpha, beta, IR, rel-DD
    holding_period_analysis.md   Section 7 — 1/3/6/12-month holds
    training_window_study.md     Section 8 — expanding vs rolling 5y/3y/2y
    RECOMMENDATION.md            final synthesis + verdict
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PCT = "{:+.2%}"
NUM = "{:+.4f}"
F2 = "{:.2f}"
P0 = "{:.0%}"


def _fmt(v, spec: str = NUM) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    try:
        return spec.format(v)
    except (ValueError, TypeError):
        return str(v)


def _table(df: pd.DataFrame, cols: list[str], fmts: dict[str, str] | None = None,
           label: str | None = None, index_fmt=str) -> str:
    fmts = fmts or {}
    header = ([label] if label else []) + cols
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    for idx, row in df.iterrows():
        cells = ([index_fmt(idx)] if label else [])
        cells += [_fmt(row.get(c), fmts.get(c, NUM)) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 1. Walk-forward performance
# --------------------------------------------------------------------------- #
def write_walkforward_report(out_dir: Path, splits_data: list[dict],
                             pooled_ic: pd.DataFrame, by_year: pd.DataFrame,
                             notes: list[str]) -> None:
    L = ["# Section 1 — Walk-Forward Validation\n",
         "Each **test year** re-derives the *entire* Parent-Selection-V4 construction "
         "(sub-factors, intra-parent weights, parent composition, IC/IR-capped parent "
         "weights) on **train-only** history, with selection forward returns capped 6 "
         "months before the test boundary so no future information can leak. The frozen "
         "configuration is then applied to the unseen year. Universe membership is "
         "point-in-time (survivorship-free) at every rebalance.\n"]

    L.append("## Out-of-sample results by year\n")
    L.append("3M/6M composite IC, information ratio, hit rate, and 6M Q5−Q1 spread for each "
             "expanding-window test year.\n")
    L.append(_table(by_year,
                    ["n_test", "ic_3M", "ir_3M", "hit_3M", "ic_6M", "ir_6M", "hit_6M",
                     "spread_6M_ann", "mono_6M"],
                    {"n_test": "{:.0f}", "ic_3M": NUM, "ir_3M": F2, "hit_3M": F2,
                     "ic_6M": NUM, "ir_6M": F2, "hit_6M": F2, "spread_6M_ann": PCT,
                     "mono_6M": P0}, label="year",
                    index_fmt=lambda i: str(by_year.loc[i, "year"])))
    by_year.to_csv(out_dir / "walkforward_by_year.csv", index=False)

    L.append("\n## Aggregate out-of-sample composite IC (all test years pooled)\n")
    L.append("The headline read on Q1 — *do higher composite scores lead to higher future "
             "returns?* — pooled across every test rebalance in every split.\n")
    L.append(_table(pooled_ic, ["n_periods", "mean_ic", "median_ic", "information_ratio",
                                "t_stat", "hit_rate"],
                    {"mean_ic": NUM, "median_ic": NUM, "information_ratio": NUM,
                     "t_stat": F2, "hit_rate": F2, "n_periods": "{}"},
                    label="horizon",
                    index_fmt=lambda i: pooled_ic.loc[i, "horizon"]))
    pooled_ic.to_csv(out_dir / "walkforward_pooled_ic.csv", index=False)

    L.append("\n## Per-split selection & frozen configuration\n")
    for note in notes:
        L.append(f"> {note}\n")
    for sd in splits_data:
        sp, cfg, m = sd["split"], sd["config"], sd["config"].meta
        L.append(f"\n### Test year `{sp.label}` — train {sp.train_start}→{sp.train_end} "
                 f"({m['n_train_rebalances']} rebalances), test {sp.test_start}→{sp.test_end}\n")
        rows = [{"parent": p, "formula": s["formula"], "flag": s["signal_flag"] or "—"}
                for p, s in m["selection"].items()]
        L.append(_table(pd.DataFrame(rows), ["parent", "formula", "flag"],
                        {"formula": "{}", "flag": "{}"}))
        pw = pd.DataFrame([{"parent": p, "weight": w}
                           for p, w in cfg.parent_weights.items()])
        L.append("\n_Parent / composite weights (IC-IR blend, cap 25 %):_ "
                 + ", ".join(f"{r.parent} {r.weight:.2f}" for r in pw.itertuples()) + "\n")
        pm = sd["port_metrics"]
        L.append(f"_OOS top-20 % EW:_ CAGR {_fmt(pm.get('cagr'), PCT)}, Sharpe "
                 f"{_fmt(pm.get('sharpe'), F2)}, max-DD {_fmt(pm.get('max_drawdown'), PCT)}, "
                 f"vs SPY excess {_fmt(pm.get('spy_excess_cagr'), PCT)}.\n")
    (out_dir / "WALKFORWARD_REPORT.md").write_text("\n".join(L))


# --------------------------------------------------------------------------- #
# 2. Quantile analysis
# --------------------------------------------------------------------------- #
def write_quantile_report(out_dir: Path, pooled_q: dict[str, dict],
                          per_split_q: dict[str, dict[str, dict]]) -> None:
    L = ["# Section 2 — Quantile Analysis\n",
         "Names are sorted by the frozen composite into five equal buckets each rebalance "
         "(Q1 = bottom 20 %, Q5 = top 20 %); each bucket's forward return is measured at "
         "1/3/6/12-month horizons, pooled across all out-of-sample rebalances. Multi-month "
         "horizons overlap, so volatilities are understated and Sharpes are indicative.\n"]
    for h in ["1M", "3M", "6M", "12M"]:
        blk = pooled_q.get(h)
        if not blk or blk["n_periods"] == 0:
            L.append(f"\n## {h} — insufficient data\n")
            continue
        L.append(f"\n## {h} horizon  (n={blk['n_periods']} periods)\n")
        L.append(_table(blk["table"], ["avg", "median", "annualized", "volatility",
                                       "sharpe", "hit_rate"],
                        {"avg": PCT, "median": PCT, "annualized": PCT, "volatility": PCT,
                         "sharpe": F2, "hit_rate": F2}, label="bucket", index_fmt=str))
        sp = blk["spread"]
        L.append(f"\n**Q5−Q1 spread:** avg {_fmt(sp.get('avg'), PCT)}, annualized "
                 f"{_fmt(sp.get('annualized'), PCT)}, Sharpe {_fmt(sp.get('sharpe'), F2)}, "
                 f"hit {_fmt(sp.get('hit_rate'), F2)}.")
        L.append(f"**Monotonicity:** strictly increasing Q1<…<Q5 in "
                 f"{_fmt(blk['monotonic_rate'], P0)} of periods; averaged-profile rank "
                 f"corr = {_fmt(blk['profile_spearman'], F2)}.\n")
        blk["table"].to_csv(out_dir / f"quantiles_{h}.csv")

    L.append("\n## Per-year consistency — Q5−Q1 spread (annualized)\n")
    L.append("Is the ranking edge present every year, or driven by a few? One row per test "
             "year; positive across most years = consistent, sign-flipping = regime-driven.\n")
    rows = []
    for label, qd in per_split_q.items():
        row = {"year": label}
        for h in ["1M", "3M", "6M", "12M"]:
            blk = qd.get(h)
            row[h] = blk["spread"].get("annualized") if blk and blk["n_periods"] else np.nan
        rows.append(row)
    dfp = pd.DataFrame(rows)
    L.append(_table(dfp, ["1M", "3M", "6M", "12M"],
                    {h: PCT for h in ["1M", "3M", "6M", "12M"]}, label="year",
                    index_fmt=lambda i: str(dfp.loc[i, "year"])))
    dfp.to_csv(out_dir / "quantile_spread_by_year.csv", index=False)
    (out_dir / "quantile_analysis.md").write_text("\n".join(L))


# --------------------------------------------------------------------------- #
# 3. Parent & sub-factor analysis
# --------------------------------------------------------------------------- #
def write_parent_report(out_dir: Path, parent_ic: pd.DataFrame,
                        parent_corr: pd.DataFrame, sub_freq: pd.DataFrame,
                        parent_weight_table: pd.DataFrame) -> None:
    L = ["# Section 3 — Parent & Sub-Factor Analysis\n",
         "Out-of-sample behaviour of the eight parent factors and the sub-factors the "
         "walk-forward re-selects each year. Parent scores here are each split's "
         "frozen-sub-weighted composite applied to its own test period, pooled.\n"]

    L.append("## Parent out-of-sample IC (pooled)\n")
    piv = (parent_ic.pivot_table(index="parent", columns="horizon", values="mean_ic")
           if not parent_ic.empty else pd.DataFrame())
    if not piv.empty:
        piv = piv.reindex(columns=[c for c in ["1M", "3M", "6M", "12M"] if c in piv.columns])
        piv["mean_3m6m"] = piv[[c for c in ["3M", "6M"] if c in piv.columns]].mean(axis=1)
        piv = piv.sort_values("mean_3m6m", ascending=False)
        L.append(_table(piv, list(piv.columns), {c: NUM for c in piv.columns},
                        label="parent", index_fmt=str))
        L.append("\n_Parents ranked by pooled 3M/6M OOS IC — the persistent return "
                 "predictors sit at the top; near-zero / negative parents add diversification "
                 "or noise, not directional edge._\n")
    parent_ic.to_csv(out_dir / "parent_oos_ic.csv", index=False)

    L.append("\n## Parent correlation (avg cross-sectional Spearman, OOS)\n")
    L.append("How much the parent scores overlap out of sample — high pairwise correlation "
             "means redundant bets; low/negative means genuine diversification.\n")
    if not parent_corr.empty:
        pc = parent_corr.round(2)
        L.append(_table(pc, list(pc.columns), {c: F2 for c in pc.columns},
                        label="parent", index_fmt=str))
        parent_corr.to_csv(out_dir / "parent_correlation.csv")

    L.append("\n## Sub-factor selection frequency across test years\n")
    L.append("How often each sub-factor is chosen when the model re-selects annually without "
             "seeing the future. **Persistent** signals (selected most years) are trustworthy; "
             "**unstable** ones (selected sporadically) are sample-driven and fragile.\n")
    if not sub_freq.empty:
        sf = sub_freq.sort_values(["parent", "freq"], ascending=[True, False])
        L.append(_table(sf, ["parent", "n_selected", "n_years", "freq", "avg_weight",
                             "stability"],
                        {"parent": "{}", "n_selected": "{:.0f}", "n_years": "{:.0f}",
                         "freq": P0, "avg_weight": F2, "stability": "{}"},
                        label="sub_factor",
                        index_fmt=lambda i: str(sf.loc[i, "sub_factor"])))
        sub_freq.to_csv(out_dir / "subfactor_selection_frequency.csv", index=False)
        persistent = sf[sf["stability"] == "persistent"]["sub_factor"].tolist()
        unstable = sf[sf["stability"] == "unstable"]["sub_factor"].tolist()
        L.append(f"\n- **Persistent (selected ≥75 % of years):** {', '.join(persistent) or '—'}")
        L.append(f"- **Unstable (selected <40 % of years):** {', '.join(unstable) or '—'}\n")
    (out_dir / "parent_subfactor_analysis.md").write_text("\n".join(L))


# --------------------------------------------------------------------------- #
# 4. Weight stability
# --------------------------------------------------------------------------- #
def write_weight_stability_report(out_dir: Path, parent_weight_table: pd.DataFrame,
                                  drift: pd.DataFrame, avg_l1_drift: float) -> None:
    L = ["# Section 4 — Weight Stability\n",
         "How the IC/IR-blended **parent (composite) weights** move as the model re-selects "
         "each year. A production model wants weights that drift gradually — large year-to-"
         "year swings mean the construction is chasing sample noise.\n"]

    year_cols = [c for c in parent_weight_table.columns if c != "parent"]
    L.append("## Parent weight by test year\n")
    L.append(_table(parent_weight_table, year_cols, {c: F2 for c in year_cols},
                    label="parent",
                    index_fmt=lambda i: str(parent_weight_table.loc[i, "parent"])))
    parent_weight_table.to_csv(out_dir / "parent_weight_by_year.csv", index=False)

    L.append("\n## Weight stability per parent\n")
    L.append("Mean / std / min / max of each parent's composite weight across the annual "
             "re-selections. Low std relative to mean = stable allocation.\n")
    L.append(_table(drift, ["mean", "std", "min", "max", "cv"],
                    {"mean": F2, "std": F2, "min": F2, "max": F2, "cv": F2},
                    label="parent", index_fmt=lambda i: str(drift.loc[i, "parent"])))
    drift.to_csv(out_dir / "weight_drift.csv", index=False)

    L.append(f"\n**Average year-over-year weight turnover (½·L1 of the parent-weight "
             f"vector):** {_fmt(avg_l1_drift, F2)} — the fraction of the composite that "
             f"re-allocates between consecutive annual re-selections. Lower is more robust.\n")
    (out_dir / "weight_stability.md").write_text("\n".join(L))


# --------------------------------------------------------------------------- #
# 5. Portfolio construction
# --------------------------------------------------------------------------- #
_PORT_COLS = ["cagr", "total_return", "sharpe", "sortino", "ann_vol", "max_drawdown",
              "avg_turnover", "hit_rate"]
_PORT_FMT = {"cagr": PCT, "total_return": PCT, "sharpe": F2, "sortino": F2, "ann_vol": PCT,
             "max_drawdown": PCT, "avg_turnover": F2, "hit_rate": F2}


def _with_construction(sweep: pd.DataFrame) -> pd.DataFrame:
    disp = sweep.copy()
    disp["construction"] = disp.apply(
        lambda r: f"top{int(r['top_pct']*100)}% {r['mode']}", axis=1)
    return disp


def write_portfolio_report(out_dir: Path, sweep: pd.DataFrame) -> None:
    L = ["# Section 5 — Portfolio Construction Analysis\n",
         "Monthly-rebalanced long portfolios on the pooled out-of-sample composite. "
         "**Equal-weight** is 1/N over the selected names; **sector-neutral** weights each "
         "GICS sector to its share of the scored universe (names equal-weighted within "
         "sector), removing sector bets. Full risk/return metric set per construction.\n"]
    disp = _with_construction(sweep)
    L.append(_table(disp.set_index("construction"), _PORT_COLS, _PORT_FMT,
                    label="construction", index_fmt=str))
    sweep.to_csv(out_dir / "portfolio_sweep.csv", index=False)
    (out_dir / "portfolio_analysis.md").write_text("\n".join(L))


# --------------------------------------------------------------------------- #
# 6. Benchmark comparison
# --------------------------------------------------------------------------- #
def write_benchmark_report(out_dir: Path, sweep: pd.DataFrame) -> None:
    L = ["# Section 6 — Benchmark Comparison (SPY & QQQ)\n",
         "Every construction versus the two benchmarks on the identical rebalance grid. "
         "**Excess** = portfolio − benchmark CAGR; **alpha** = annualised CAPM intercept; "
         "**beta** = sensitivity to the benchmark; **IR** = excess-return information ratio "
         "(mean active ÷ tracking error); **rel-DD** = worst drawdown of the "
         "portfolio/benchmark equity ratio.\n"]
    disp = _with_construction(sweep)
    for bench in ("spy", "qqq"):
        cols = [f"{bench}_cagr", f"{bench}_excess_cagr", f"{bench}_alpha", f"{bench}_beta",
                f"{bench}_ir", f"{bench}_rel_max_drawdown"]
        if not all(c in disp.columns for c in cols):
            continue
        L.append(f"\n## vs {bench.upper()}\n")
        L.append(_table(disp.set_index("construction"), cols,
                        {f"{bench}_cagr": PCT, f"{bench}_excess_cagr": PCT,
                         f"{bench}_alpha": PCT, f"{bench}_beta": F2, f"{bench}_ir": F2,
                         f"{bench}_rel_max_drawdown": PCT},
                        label="construction", index_fmt=str))
    (out_dir / "benchmark_comparison.md").write_text("\n".join(L))


# --------------------------------------------------------------------------- #
# 7. Holding period
# --------------------------------------------------------------------------- #
def write_holding_report(out_dir: Path, hold: pd.DataFrame) -> None:
    L = ["# Section 7 — Holding-Period Analysis\n",
         "Top-20 % equal-weight long portfolio held for non-overlapping 1/3/6/12-month "
         "periods on the pooled out-of-sample composite. Identifies the horizon where the "
         "factor edge is strongest per unit of turnover.\n",
         "> ⚠ **Read `n_periods` first.** Longer non-overlapping holds have proportionally "
         "fewer periods, so their CAGR/Sharpe are lower-confidence. The better-sampled read "
         "on where signal decays is the per-horizon **Q5−Q1 spread** in "
         "`quantile_analysis.md`, which the recommendation cross-checks.\n"]
    disp = hold.copy()
    disp["hold"] = disp["hold_months"].map(lambda m: f"{m}M")
    L.append(_table(disp.set_index("hold"),
                    ["n_periods", "cagr", "sharpe", "sortino", "max_drawdown", "ann_vol",
                     "avg_turnover", "hit_rate"],
                    {"n_periods": "{:.0f}", "cagr": PCT, "sharpe": F2, "sortino": F2,
                     "max_drawdown": PCT, "ann_vol": PCT, "avg_turnover": F2,
                     "hit_rate": F2}, label="hold", index_fmt=str))
    hold.to_csv(out_dir / "holding_period.csv", index=False)
    (out_dir / "holding_period_analysis.md").write_text("\n".join(L))


# --------------------------------------------------------------------------- #
# 8. Training-window study
# --------------------------------------------------------------------------- #
def write_training_study_report(out_dir: Path, study: pd.DataFrame, rec: dict) -> None:
    L = ["# Section 8 — Training-Window Study\n",
         "Every policy re-selects the **entire** construction (sub-factors, sub weights, "
         "parent weights, composite) over the same out-of-sample test years, differing only "
         "in how far back training reaches: **expanding** (all history) vs **rolling 5y / 3y "
         "/ 2y**. Compares pooled OOS edge to decide the production training window.\n"]
    L.append(_table(study.set_index("policy"),
                    ["n_test_years", "n_test_periods", "mean_ic_3m6m", "ic_hit_rate",
                     "spread_6m_ann", "top20_cagr", "top20_sharpe", "top20_sortino",
                     "top20_max_dd", "spy_excess_cagr", "qqq_excess_cagr", "avg_turnover",
                     "config_churn"],
                    {"n_test_years": "{:.0f}", "n_test_periods": "{:.0f}",
                     "mean_ic_3m6m": NUM, "ic_hit_rate": F2, "spread_6m_ann": PCT,
                     "top20_cagr": PCT, "top20_sharpe": F2, "top20_sortino": F2,
                     "top20_max_dd": PCT, "spy_excess_cagr": PCT, "qqq_excess_cagr": PCT,
                     "avg_turnover": F2, "config_churn": P0},
                    label="policy", index_fmt=str))
    study.to_csv(out_dir / "training_window_study.csv", index=False)
    L.append("\n## Verdict\n")
    L.append(rec.get("text", "") + "\n")
    (out_dir / "training_window_study.md").write_text("\n".join(L))


# --------------------------------------------------------------------------- #
# 9. Final recommendation
# --------------------------------------------------------------------------- #
def write_recommendation(out_dir: Path, *, pooled_ic: pd.DataFrame, pooled_q: dict,
                         sweep: pd.DataFrame, hold: pd.DataFrame, long_only: dict,
                         long_short: dict, sub_freq: pd.DataFrame, parent_ic: pd.DataFrame,
                         study: pd.DataFrame, window_rec: dict, verdict: dict) -> None:
    L = ["# Final Recommendation\n",
         "## 1. Does the model have genuine out-of-sample predictive power?\n",
         verdict["headline"] + "\n", "**Evidence (pooled OOS):**\n"]
    for h in ["1M", "3M", "6M", "12M"]:
        row = pooled_ic[pooled_ic["horizon"] == h]
        if row.empty or int(row.iloc[0].get("n_periods", 0)) == 0:
            continue
        r = row.iloc[0]
        q = pooled_q.get(h, {})
        spread = q.get("spread", {}).get("annualized") if q else None
        L.append(f"- **{h}:** IC {_fmt(r['mean_ic'])} (hit {_fmt(r['hit_rate'], P0)}, IR "
                 f"{_fmt(r['information_ratio'], F2)}), Q5−Q1 annualized {_fmt(spread, PCT)}, "
                 f"monotonic {_fmt(q.get('monotonic_rate'), P0)}.")

    L.append("\n## 2. Which parents & sub-factors are genuinely persistent?\n")
    if not parent_ic.empty:
        piv = parent_ic.pivot_table(index="parent", columns="horizon", values="mean_ic")
        core = piv[[c for c in ["3M", "6M"] if c in piv.columns]].mean(axis=1).sort_values(
            ascending=False)
        top = ", ".join(f"{p} ({v:+.3f})" for p, v in core.head(4).items())
        L.append(f"Strongest pooled 3M/6M OOS parent IC: {top}.")
    if not sub_freq.empty:
        persistent = sub_freq[sub_freq["stability"] == "persistent"]["sub_factor"].tolist()
        L.append(f"Sub-factors selected in ≥75 % of the annual re-selections (persistent): "
                 f"{', '.join(persistent) or '—'}.\n")

    L.append("## 3. Best portfolio construction\n")
    best = sweep.sort_values("sharpe", ascending=False).iloc[0]
    L.append(f"Highest OOS Sharpe: **top{int(best['top_pct']*100)}% {best['mode']}** — CAGR "
             f"{_fmt(best['cagr'], PCT)}, Sharpe {_fmt(best['sharpe'], F2)}, Sortino "
             f"{_fmt(best['sortino'], F2)}, max-DD {_fmt(best['max_drawdown'], PCT)}, turnover "
             f"{_fmt(best['avg_turnover'], F2)}. " + verdict.get("portfolio", "") + "\n")

    L.append("## 4. Best holding period\n")
    hold_ok = hold[hold["n_periods"] >= 6]
    bh = (hold_ok if not hold_ok.empty else hold).sort_values(
        "sharpe", ascending=False).iloc[0]
    L.append(f"Best well-sampled hold: **{int(bh['hold_months'])}-month** (n={int(bh['n_periods'])}"
             f", Sharpe {_fmt(bh['sharpe'], F2)}, CAGR {_fmt(bh['cagr'], PCT)}). "
             + verdict.get("holding", "") + "\n")

    L.append("## 5. Best training-window length\n")
    L.append(window_rec.get("text", "") + "\n")

    L.append("## 6. Expected alpha vs SPY & QQQ\n")
    b = best
    L.append(f"At the recommended construction (top{int(b['top_pct']*100)}% {b['mode']}): "
             f"vs **SPY** excess CAGR {_fmt(b.get('spy_excess_cagr'), PCT)}, alpha "
             f"{_fmt(b.get('spy_alpha'), PCT)}, beta {_fmt(b.get('spy_beta'), F2)}, IR "
             f"{_fmt(b.get('spy_ir'), F2)}; vs **QQQ** excess CAGR "
             f"{_fmt(b.get('qqq_excess_cagr'), PCT)}, alpha {_fmt(b.get('qqq_alpha'), PCT)}, "
             f"beta {_fmt(b.get('qqq_beta'), F2)}, IR {_fmt(b.get('qqq_ir'), F2)}.\n")

    L.append("## 7. Long-only vs long-short\n")
    ls_tbl = pd.DataFrame([{"strategy": "long-only top20%", **long_only},
                           {"strategy": "long-short 20/20", **long_short}])
    L.append(_table(ls_tbl.set_index("strategy"),
                    ["cagr", "sharpe", "sortino", "max_drawdown", "ann_vol", "hit_rate"],
                    {"cagr": PCT, "sharpe": F2, "sortino": F2, "max_drawdown": PCT,
                     "ann_vol": PCT, "hit_rate": F2}, label="strategy", index_fmt=str))
    L.append("\n" + verdict.get("longshort", "") + "\n")

    L.append("## 8. Recommended production configuration\n")
    L.append(verdict.get("production", "") + "\n")

    L.append("## Caveats\n")
    for c in verdict.get("caveats", []):
        L.append(f"- {c}")
    (out_dir / "RECOMMENDATION.md").write_text("\n".join(L))
