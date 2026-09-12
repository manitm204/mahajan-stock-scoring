"""Promotion rule for loop-engineering experiments (user-specified, 2026-09-11
chat -- see session_log.md for the full back-and-forth that pinned down each
gate's exact definition). One function, `evaluate_promotion`, takes a
candidate selector and returns PROMOTE / PROMISING / REJECT plus every gate's
pass/fail and the metrics behind it. Reuses harness.py (data/sim mechanics),
performance_metrics (research.walkforward.portfolio) for the leave-one-year-out
recomputation, and perturbation.py for the noise-robustness gate.

Alpha-adjusted calendar-time excess return (added 2026-09-12, user request):
the calendar-time excess-return series (avg_monthly_excess, year_sum,
n_years_positive, majority_years, best_year_removed_negative, and the
diff_by_date series consumed by block_bootstrap.py) is now computed on the
CAPM-residual difference -- (candidate_return - beta_candidate*SPY_return) -
(baseline_return - beta_baseline*SPY_return), each leg's beta estimated once
via full-sample OLS against SPY -- rather than the raw return difference.
Raw excess return doesn't control for beta: a candidate that simply carries
more market exposure than the baseline shows positive "excess return" almost
by construction in a mostly-bullish window, which is leverage, not selection
skill. The raw (pre-adjustment) series is still computed and returned
alongside the alpha-adjusted one (as raw_avg_monthly_excess / raw_diff_by_date
/ beta_candidate_monthly / beta_baseline_monthly) for transparency, but every
gate and the block-bootstrap significance check now use the alpha-adjusted
version.

Decision logic
--------------
REJECT if any of:
  - median Sharpe improvement < 0.025
  - sim win rate < 50%
  - average monthly excess return (sim-averaged, real calendar months,
    ALPHA-ADJUSTED -- see module docstring above) < 0
  - removing the single BEST year, the remaining cumulative excess return
    goes negative

Else PROMOTE if all of:
  - median Sharpe improvement >= 0.05
  - sim win rate >= 60%
  - average monthly excess return (alpha-adjusted) > 0
  - candidate beats baseline in a majority of calendar years (alpha-adjusted)
  - leave-one-year-out (strict): for EVERY year removed, recompute Sharpe and
    alpha on the remaining years for both candidate and baseline; both
    candidate_sharpe - baseline_sharpe > 0 AND candidate_alpha - baseline_alpha
    > 0 must hold, for every single year removed
  - max drawdown does not worsen by more than 4 percentage points
  - full-period alpha vs SPY improves vs baseline (no beta cap)
  - win rate stays >= 55% after mild Gaussian noise on the ranking score
    (requires a `make_perturbed_selector(noise_mult) -> selector` factory)

Else PROMISING.

Usage as a library:
    from research.loop_research.promotion import evaluate_promotion
    from research.loop_research.candidates import top5_by_insider
    from research.loop_research.perturbation import make_perturbed_single_parent_selector

    report = evaluate_promotion(
        top5_by_insider,
        make_perturbed_selector=lambda nm: make_perturbed_single_parent_selector("insider", nm),
    )

CLI: python -m research.loop_research.promotion <candidate_module.function>
     [--parent NAME] [--baseline module.function] [--n-sims N]
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from research.loop_research import harness as H                       # noqa: E402
from research.walkforward.portfolio import performance_metrics        # noqa: E402

OUT_DIR = REPO / "output" / "loop_research"

REJECT_SHARPE_FLOOR = 0.025
REJECT_WIN_RATE_FLOOR = 0.50
PROMOTE_SHARPE_MIN = 0.05
PROMOTE_WIN_RATE_MIN = 0.60
PROMOTE_MAX_DD_TOLERANCE_PP = 0.04
PROMOTE_PERTURBATION_WIN_RATE_MIN = 0.55
DEFAULT_NOISE_MULT = 0.5


def _median(vals):
    vals = [v for v in vals if v == v]  # drop NaN
    return float(np.median(vals)) if vals else float("nan")


def _collect_paths(bundle, selector, n_sims, seed_base, deterministic, dates, spy, qqq):
    """Runs `selector` n_sims times (or once + replicate, if deterministic).
    Returns (raw_returns: ndarray[n_run, T] with n_run==1 for deterministic,
    per_sim_metrics: list[dict] length n_sims, paired 1:1 with the baseline's
    seeds for win-rate/median gates)."""
    n_run = 1 if deterministic else n_sims
    raw = np.zeros((n_run, len(dates)))
    metrics = []
    for s in range(n_run):
        pr, turnover, targets = H.run_one(bundle, selector, seed_base + s)
        pr_r = pr.reindex(dates).fillna(0.0)
        raw[s] = pr_r.values
        metrics.append(H.sim_metrics(pr, turnover, targets, spy, qqq))
    paired_metrics = metrics * n_sims if deterministic else metrics
    return raw, paired_metrics


def _masked_median_sharpe_alpha(raw, spy, mask, dates):
    """Median Sharpe/alpha across raw's rows, restricted to `mask` (a boolean
    array over `dates`/columns) -- used for the strict leave-one-year-out
    gate. `raw` may have 1 row (deterministic candidate) or n_sims rows."""
    idx = pd.DatetimeIndex(dates)[mask]
    spy_masked = pd.Series(np.asarray(spy)[mask], index=idx)
    sharpes, alphas = [], []
    for row in raw:
        r = pd.Series(row[mask], index=idx)
        m = performance_metrics(r, hold_months=1, benchmarks={"SPY": spy_masked})
        sharpes.append(m.get("sharpe", float("nan")))
        alphas.append(m.get("spy_alpha", float("nan")))
    return _median(sharpes), _median(alphas)


def evaluate_promotion(candidate_selector, *, baseline_selector=None,
                       make_perturbed_selector=None, n_sims=500, seed_base=0,
                       deterministic_candidate=True, noise_mult=DEFAULT_NOISE_MULT):
    """Runs the full promotion protocol for one candidate selector against a
    baseline (default: harness.random_selector, the production featured
    config). Returns a dict: decision, reject_gates, promote_gates,
    candidate_metrics, baseline_metrics, per_year table, leave-one-year-out
    detail, reasons.
    """
    baseline_selector = baseline_selector or H.random_selector
    bundle = H.get_data()
    spy_full, qqq_full = H.bench_series(bundle)
    dates = list(bundle["data"].rebal_dates)[:-1]
    spy = spy_full.reindex(dates).fillna(0.0)
    qqq = qqq_full.reindex(dates).fillna(0.0)

    cand_raw, cand_metrics = _collect_paths(
        bundle, candidate_selector, n_sims, seed_base, deterministic_candidate, dates, spy, qqq)
    base_raw, base_metrics = _collect_paths(
        bundle, baseline_selector, n_sims, seed_base, False, dates, spy, qqq)

    cand_sharpes = [m["sharpe"] for m in cand_metrics]
    base_sharpes = [m["sharpe"] for m in base_metrics]
    cand_alphas = [m.get("spy_alpha", float("nan")) for m in cand_metrics]
    base_alphas = [m.get("spy_alpha", float("nan")) for m in base_metrics]
    cand_dds = [m["max_dd"] for m in cand_metrics]
    base_dds = [m["max_dd"] for m in base_metrics]
    cand_betas = [m.get("spy_beta", float("nan")) for m in cand_metrics]
    base_betas = [m.get("spy_beta", float("nan")) for m in base_metrics]
    cand_cagrs = [m["cagr"] for m in cand_metrics]
    base_cagrs = [m["cagr"] for m in base_metrics]

    median_cand_sharpe = _median(cand_sharpes)
    median_base_sharpe = _median(base_sharpes)
    sharpe_improve = median_cand_sharpe - median_base_sharpe
    win_rate = float(np.mean([c > b for c, b in zip(cand_sharpes, base_sharpes)]))

    median_cand_alpha = _median(cand_alphas)
    median_base_alpha = _median(base_alphas)
    alpha_improves = median_cand_alpha > median_base_alpha

    median_cand_dd = _median(cand_dds)
    median_base_dd = _median(base_dds)
    dd_deterioration_pp = median_base_dd - median_cand_dd  # positive = worse

    median_cand_beta = _median(cand_betas)
    median_base_beta = _median(base_betas)
    median_cand_cagr = _median(cand_cagrs)
    median_base_cagr = _median(base_cagrs)

    # Calendar-time excess return: mean across raw rows (kills sim-resampling
    # noise), one number per real calendar month -- NOT the 500 pseudo-
    # replicated sims (see champion_period_breakdown.py precedent).
    mean_cand_by_date = cand_raw.mean(axis=0)
    mean_base_by_date = base_raw.mean(axis=0)
    raw_diff_by_date = mean_cand_by_date - mean_base_by_date
    raw_avg_monthly_excess = float(raw_diff_by_date.mean())

    # Alpha-adjust before testing significance (2026-09-12): raw excess
    # return doesn't control for beta, so a candidate carrying more market
    # exposure than the baseline shows positive "excess return" almost by
    # construction in a mostly-bullish window. Net out each leg's own
    # full-sample beta against SPY first.
    spy_arr = np.asarray(spy)
    var_spy = float(np.var(spy_arr))
    beta_cand_monthly = float(np.cov(mean_cand_by_date, spy_arr)[0, 1] / var_spy) if var_spy else float("nan")
    beta_base_monthly = float(np.cov(mean_base_by_date, spy_arr)[0, 1] / var_spy) if var_spy else float("nan")
    alpha_cand_by_date = mean_cand_by_date - beta_cand_monthly * spy_arr
    alpha_base_by_date = mean_base_by_date - beta_base_monthly * spy_arr
    diff_by_date = alpha_cand_by_date - alpha_base_by_date
    avg_monthly_excess = float(diff_by_date.mean())

    years = pd.DatetimeIndex(dates).year.values
    diff_series = pd.Series(diff_by_date, index=years)
    year_sum = diff_series.groupby(level=0).sum()
    n_years = len(year_sum)
    n_years_positive = int((year_sum > 0).sum())
    majority_years = n_years_positive > n_years / 2

    total_edge = float(year_sum.sum())
    best_year = int(year_sum.idxmax())
    remainder_after_best_year = float(total_edge - year_sum.loc[best_year])
    best_year_removed_negative = remainder_after_best_year < 0

    # Strict leave-one-year-out: recompute Sharpe + alpha on the remaining
    # years, for both legs, for every year removed.
    loo = {}
    for y in year_sum.index:
        mask = years != y
        c_sh, c_al = _masked_median_sharpe_alpha(cand_raw, spy, mask, dates)
        b_sh, b_al = _masked_median_sharpe_alpha(base_raw, spy, mask, dates)
        loo[int(y)] = {
            "sharpe_diff": c_sh - b_sh,
            "alpha_diff": c_al - b_al,
            "pass": bool((c_sh - b_sh) > 0 and (c_al - b_al) > 0),
        }
    leave_one_year_out_strict = all(v["pass"] for v in loo.values())

    # Perturbation gate.
    perturbation_win_rate = None
    if make_perturbed_selector is not None:
        perturbed_selector = make_perturbed_selector(noise_mult)
        pert_raw, pert_metrics = _collect_paths(
            bundle, perturbed_selector, n_sims, seed_base, False, dates, spy, qqq)
        pert_sharpes = [m["sharpe"] for m in pert_metrics]
        perturbation_win_rate = float(np.mean(
            [p > b for p, b in zip(pert_sharpes, base_sharpes)]))

    # --- Reject gates ---
    reject_gates = {
        "sharpe_improve_ge_0.025": sharpe_improve >= REJECT_SHARPE_FLOOR,
        "win_rate_ge_50pct": win_rate >= REJECT_WIN_RATE_FLOOR,
        "avg_monthly_excess_ge_0": avg_monthly_excess >= 0,
        "best_year_removed_not_negative": not best_year_removed_negative,
    }
    rejected = not all(reject_gates.values())

    # --- Promote gates (computed regardless, for a full transparent report) ---
    promote_gates = {
        "sharpe_improve_ge_0.05": sharpe_improve >= PROMOTE_SHARPE_MIN,
        "win_rate_ge_60pct": win_rate >= PROMOTE_WIN_RATE_MIN,
        "avg_monthly_excess_gt_0": avg_monthly_excess > 0,
        "majority_of_years": majority_years,
        "leave_one_year_out_strict": leave_one_year_out_strict,
        "max_dd_within_4pp": dd_deterioration_pp <= PROMOTE_MAX_DD_TOLERANCE_PP,
        "alpha_improves": alpha_improves,
        "perturbation_win_rate_ge_55pct": (
            perturbation_win_rate is not None
            and perturbation_win_rate >= PROMOTE_PERTURBATION_WIN_RATE_MIN
        ),
    }

    if rejected:
        decision = "REJECT"
        reasons = [g for g, ok in reject_gates.items() if not ok]
    elif all(promote_gates.values()):
        decision = "PROMOTE"
        reasons = ["all reject and promote gates passed"]
    else:
        decision = "PROMISING"
        reasons = [g for g, ok in promote_gates.items() if not ok]

    if make_perturbed_selector is None:
        reasons.append("no make_perturbed_selector supplied -- perturbation gate "
                       "treated as failed / not evaluated")

    return {
        "decision": decision,
        "reasons": reasons,
        "reject_gates": reject_gates,
        "promote_gates": promote_gates,
        "candidate_metrics": {
            "median_sharpe": median_cand_sharpe, "median_alpha": median_cand_alpha,
            "median_max_dd": median_cand_dd, "median_beta": median_cand_beta,
            "median_cagr": median_cand_cagr,
        },
        "baseline_metrics": {
            "median_sharpe": median_base_sharpe, "median_alpha": median_base_alpha,
            "median_max_dd": median_base_dd, "median_beta": median_base_beta,
            "median_cagr": median_base_cagr,
        },
        "sharpe_improve": sharpe_improve,
        "win_rate": win_rate,
        "avg_monthly_excess": avg_monthly_excess,
        "raw_avg_monthly_excess": raw_avg_monthly_excess,
        "beta_candidate_monthly": beta_cand_monthly,
        "beta_baseline_monthly": beta_base_monthly,
        "raw_diff_by_date": raw_diff_by_date.tolist(),
        "dd_deterioration_pp": dd_deterioration_pp,
        "perturbation_win_rate": perturbation_win_rate,
        "n_years": n_years,
        "n_years_positive": n_years_positive,
        "best_year": best_year,
        "remainder_after_best_year": remainder_after_best_year,
        "year_sum": {int(y): float(v) for y, v in year_sum.items()},
        "leave_one_year_out": loo,
        "diff_by_date": diff_by_date.tolist(),
        "dates": list(dates),
    }


def build_summary(report, experiment: str, hypothesis: str) -> dict:
    """Compact per-experiment record (user-specified format, 2026-09-11):
    the full `report` from evaluate_promotion has every raw number and gate
    detail for debugging; this is the condensed version meant to accumulate
    one file per experiment as a durable, at-a-glance log across the whole
    search (mirrors session_log.md's per-experiment write-ups, machine
    readable). Sign convention: every *_delta is candidate-minus-baseline
    (positive = candidate higher/more of that metric); for max_drawdown_delta
    that means positive = candidate's drawdown is SHALLOWER (better), since
    max_dd values are themselves negative."""
    cm, bm = report["candidate_metrics"], report["baseline_metrics"]
    return {
        "experiment": experiment,
        "hypothesis": hypothesis,
        "primary": {
            "median_sharpe_delta": round(report["sharpe_improve"], 4),
        },
        "robustness": {
            "seed_win_rate": round(report["win_rate"], 4),
            "calendar_excess_positive": report["avg_monthly_excess"] > 0,
            "calendar_excess_alpha_adjusted": round(report["avg_monthly_excess"], 6),
            "calendar_excess_raw": round(report["raw_avg_monthly_excess"], 6),
            "beta_candidate_monthly": round(report["beta_candidate_monthly"], 4),
            "beta_baseline_monthly": round(report["beta_baseline_monthly"], 4),
            "positive_years": f"{report['n_years_positive']}/{report['n_years']}",
            "leave_one_year_out_stable": report["promote_gates"]["leave_one_year_out_strict"],
            "best_year_removed_still_positive": not (report["remainder_after_best_year"] < 0),
        },
        "sanity": {
            "max_drawdown_delta": round(cm["median_max_dd"] - bm["median_max_dd"], 4),
            "beta_delta": round(cm["median_beta"] - bm["median_beta"], 4)
            if cm["median_beta"] == cm["median_beta"] and bm["median_beta"] == bm["median_beta"] else None,
            "alpha_delta": round(cm["median_alpha"] - bm["median_alpha"], 4),
            "perturbation_survived": (
                report["perturbation_win_rate"] is not None
                and report["perturbation_win_rate"] >= PROMOTE_PERTURBATION_WIN_RATE_MIN
            ),
        },
        "raw_metrics": {
            "candidate": cm, "baseline": bm,
            "perturbation_win_rate": report["perturbation_win_rate"],
        },
        "decision": report["decision"].lower(),
    }


def print_report(report):
    print(f"\nDECISION: {report['decision']}")
    print(f"reasons: {report['reasons']}")
    print(f"\ncandidate median sharpe={report['candidate_metrics']['median_sharpe']:.4f}  "
         f"alpha={report['candidate_metrics']['median_alpha']:.4f}  "
         f"max_dd={report['candidate_metrics']['median_max_dd']:.4f}")
    print(f"baseline  median sharpe={report['baseline_metrics']['median_sharpe']:.4f}  "
         f"alpha={report['baseline_metrics']['median_alpha']:.4f}  "
         f"max_dd={report['baseline_metrics']['median_max_dd']:.4f}")
    print(f"\nsharpe_improve={report['sharpe_improve']:.4f}  win_rate={report['win_rate']:.1%}  "
         f"avg_monthly_excess (alpha-adj)={report['avg_monthly_excess']:.5f}  "
         f"(raw={report['raw_avg_monthly_excess']:.5f})  "
         f"dd_deterioration_pp={report['dd_deterioration_pp']:.4f}")
    print(f"beta (monthly, cand/base)={report['beta_candidate_monthly']:.3f}/{report['beta_baseline_monthly']:.3f}")
    if report["perturbation_win_rate"] is not None:
        print(f"perturbation_win_rate={report['perturbation_win_rate']:.1%}")
    print(f"\nyears positive: {report['n_years_positive']}/{report['n_years']}  "
         f"best_year={report['best_year']}  "
         f"remainder_after_best_year_removed={report['remainder_after_best_year']:.4f}")
    print("\nreject gates:")
    for g, ok in report["reject_gates"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {g}")
    print("promote gates:")
    for g, ok in report["promote_gates"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {g}")
    print("\nleave-one-year-out detail:")
    for y, v in sorted(report["leave_one_year_out"].items()):
        print(f"  {y}: sharpe_diff={v['sharpe_diff']:+.4f}  alpha_diff={v['alpha_diff']:+.4f}  "
             f"{'PASS' if v['pass'] else 'FAIL'}")


def _load(path):
    mod_name, fn_name = path.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, fn_name)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate a loop-engineering candidate selector.")
    ap.add_argument("candidate", help="dotted path, e.g. research.loop_research.candidates.top5_by_insider")
    ap.add_argument("--baseline", default="research.loop_research.harness.random_selector")
    ap.add_argument("--parent", default=None,
                    help="if the candidate is a single-parent tie-break, name of the parent "
                         "(e.g. insider) -- used to build the Gaussian-noise perturbation test")
    ap.add_argument("--n-sims", type=int, default=500)
    ap.add_argument("--noise-mult", type=float, default=DEFAULT_NOISE_MULT)
    ap.add_argument("--not-deterministic", action="store_true",
                    help="candidate uses its own per-sim rng (like a weighted-random selector)")
    ap.add_argument("--experiment", default=None,
                    help="experiment name for the compact summary JSON (defaults to the "
                         "candidate function's name)")
    ap.add_argument("--hypothesis", default="",
                    help="one-line hypothesis, stored in the summary JSON")
    args = ap.parse_args()

    candidate_selector = _load(args.candidate)
    baseline_selector = _load(args.baseline)

    make_perturbed_selector = None
    if args.parent:
        from research.loop_research.perturbation import make_perturbed_single_parent_selector
        make_perturbed_selector = lambda nm: make_perturbed_single_parent_selector(args.parent, nm)  # noqa: E731

    report = evaluate_promotion(
        candidate_selector, baseline_selector=baseline_selector,
        make_perturbed_selector=make_perturbed_selector, n_sims=args.n_sims,
        deterministic_candidate=not args.not_deterministic, noise_mult=args.noise_mult,
    )
    print_report(report)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cand_name = args.candidate.rsplit(".", 1)[-1]
    out_path = OUT_DIR / f"promotion_{cand_name}.json"
    with out_path.open("w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(f"\nwrote {out_path}")

    experiment_name = args.experiment or cand_name
    summary = build_summary(report, experiment_name, args.hypothesis)
    exp_dir = OUT_DIR / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    summary_path = exp_dir / f"{experiment_name}.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"wrote {summary_path}")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
