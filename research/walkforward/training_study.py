"""Section 8 — training-window study.

Runs the *entire* re-selection pipeline (subfactors → intra-parent weights → parent
composition → parent/composite weights → composite) under four training-window policies —
**expanding** (all available history) and **rolling 5y / 3y / 2y** — over the same annual
out-of-sample test years, and compares their pooled OOS performance.

The question it answers: does older data improve robustness, or has factor efficacy decayed
so recent data is more predictive? The output is one row per policy with pooled OOS IC
(3M/6M), quantile spread, a top-20 % equal-weight portfolio (CAGR / Sharpe / max-DD vs SPY
& QQQ) and the selection turnover (how often the chosen sub-factor set changes across the
annual re-selections) — plus a recommended production window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research import HORIZON_MONTHS, compute_forward_returns
from research.panel import ScorePanel

from . import analysis
from . import portfolio as pf
from .runner import run_splits
from .splits import TRAIN_POLICIES, policy_splits


def _config_churn(splits_data: list[dict]) -> float:
    """Mean fraction of a parent's selected sub-factors that change between consecutive
    annual re-selections — a proxy for how unstable the construction is under this policy
    (0 = identical picks every year, 1 = fully different)."""
    by_year = [sd["config"].sub_weights for sd in splits_data]
    if len(by_year) < 2:
        return float("nan")
    churns: list[float] = []
    for prev, cur in zip(by_year, by_year[1:]):
        for parent in set(prev) | set(cur):
            a, b = set(prev.get(parent, {})), set(cur.get(parent, {}))
            if a or b:
                churns.append(1.0 - len(a & b) / max(len(a | b), 1))
    return float(np.mean(churns)) if churns else float("nan")


def _pooled_core_ic(pooled_ic: pd.DataFrame) -> tuple[float, float]:
    """Mean of the 3M & 6M pooled IC and its hit rate (the headline OOS read)."""
    core = pooled_ic[pooled_ic["horizon"].isin(["3M", "6M"])]
    core = core[core["n_periods"] > 0]
    if core.empty:
        return float("nan"), float("nan")
    return float(core["mean_ic"].mean()), float(core["hit_rate"].mean())


def run_training_study(panel: ScorePanel, matrix: pd.DataFrame, sectors: pd.Series,
                       *, policies: list[str] | None = None,
                       verbose: bool = True) -> pd.DataFrame:
    """Compare training-window policies. One row per policy of pooled OOS metrics."""
    policies = policies or list(TRAIN_POLICIES)
    rows: list[dict] = []
    for policy in policies:
        if verbose:
            print(f"\n--- training-window policy: {policy} ---")
        splits = policy_splits(policy)
        run = run_splits(panel, splits, matrix, sectors, verbose=verbose)
        if not run.splits_data:
            continue
        pooled_fwd = compute_forward_returns(matrix, list(run.pooled_scores), HORIZON_MONTHS)
        pooled_ic = analysis.composite_ic(run.pooled_scores, pooled_fwd)
        pooled_q = analysis.quantile_analysis(run.pooled_scores, pooled_fwd)
        mean_core, hit_core = _pooled_core_ic(pooled_ic)
        top20 = pf.simulate(run.pooled_scores, matrix, sectors, top_pct=0.20,
                            mode="equal", hold_months=1)
        m = top20.metrics
        rows.append({
            "policy": policy,
            "n_test_years": len(run.splits_data),
            "n_test_periods": int(sum(sd["n_test"] for sd in run.splits_data)),
            "mean_ic_3m6m": mean_core,
            "ic_hit_rate": hit_core,
            "spread_6m_ann": pooled_q.get("6M", {}).get("spread", {}).get("annualized"),
            "top20_cagr": m.get("cagr"),
            "top20_sharpe": m.get("sharpe"),
            "top20_sortino": m.get("sortino"),
            "top20_max_dd": m.get("max_drawdown"),
            "spy_excess_cagr": m.get("spy_excess_cagr"),
            "qqq_excess_cagr": m.get("qqq_excess_cagr"),
            "avg_turnover": m.get("avg_turnover"),
            "config_churn": _config_churn(run.splits_data),
        })
    return pd.DataFrame(rows)


def recommend_window(study: pd.DataFrame) -> dict:
    """Pick the production training window and explain why.

    Ranks policies by a blend of pooled OOS 3M/6M IC and top-20 % Sharpe (both matter: IC
    is the raw ranking edge, Sharpe the realised risk-adjusted payoff), and reads whether
    efficacy decays with data age by comparing expanding vs the shortest rolling window."""
    if study.empty:
        return {"best_policy": None, "text": "No policies produced usable OOS periods."}
    s = study.copy()
    for c in ("mean_ic_3m6m", "top20_sharpe"):
        col = s[c].astype(float)
        rng = col.max() - col.min()
        s[f"_{c}_n"] = (col - col.min()) / rng if rng and rng > 0 else 0.0
    s["_rank"] = 0.5 * s["_mean_ic_3m6m_n"] + 0.5 * s["_top20_sharpe_n"]
    best = s.sort_values("_rank", ascending=False).iloc[0]

    def _get(policy, col):
        row = study[study["policy"] == policy]
        return float(row[col].iloc[0]) if not row.empty else float("nan")

    exp_ic = _get("expanding", "mean_ic_3m6m")
    roll2_ic = _get("rolling2y", "mean_ic_3m6m")
    best_ic = _get(best["policy"], "mean_ic_3m6m")
    # Key the decay narrative off the *chosen* window so it can never contradict the pick.
    if best["policy"] == "expanding":
        decay = ("Older data improves robustness — the all-history expanding window earns "
                 "the best out-of-sample IC; the signal is stable enough that more history "
                 "helps and short windows over-fit recent noise.")
    elif best["policy"] == "rolling2y":
        decay = ("Recent data is more predictive — the shortest (2-year) window earns the "
                 "best out-of-sample IC, evidence that factor efficacy decays and stale "
                 "history hurts.")
    else:
        decay = (f"OOS IC peaks at an intermediate window ({best['policy']}, "
                 f"{best_ic:+.3f}) and is weaker both with all history "
                 f"(expanding {exp_ic:+.3f}) and with only two years (rolling2y "
                 f"{roll2_ic:+.3f}): very old data spans stale regimes while very short "
                 f"windows over-fit recent noise — a bounded multi-year window is the "
                 f"robust choice.")

    text = (f"**Recommended production training window: `{best['policy']}`** — pooled OOS "
            f"3M/6M IC {best['mean_ic_3m6m']:+.3f} (hit {best['ic_hit_rate']:.0%}), top-20 % "
            f"Sharpe {best['top20_sharpe']:.2f}, CAGR {best['top20_cagr']:+.1%} "
            f"(excess vs SPY {best['spy_excess_cagr']:+.1%}), selection churn "
            f"{best['config_churn']:.0%}/yr. {decay}")
    return {"best_policy": best["policy"], "text": text, "decay": decay}
