"""LLM pilot — tiers, pre-registered 3-check bar, nulls, verdict.

Pre-registration (docs/llm_pilot_design.md, fixed before any scoring):
  llm_score = mean(earnings_quality_score, balance_sheet_score,
                   100 - accounting_risk); worst tier = bottom 10% per cohort.
  Check 1  worst-tier EW forward return < rest-of-book EW return in BOTH
           cohorts.
  Check 2  pooled OLS  fwd_ret ~ const + composite_pct + worst_dummy,
           worst_dummy t <= -2.0.
  Check 3  pooled drag (rest minus tier) > q90 of 200 size-matched random
           tiers, seed 20260716.
  Exclusion guard: if unparseable/missing scores exceed 10% of a cohort the
  model fails on execution grounds.

Usage: python scripts/llm_pilot_analysis.py
Outputs: output/llm_pilot/{results.csv,verdict.csv,pilot.png}
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("output/llm_pilot")
SCORES = OUT / "scores"
TIER_PCT = 0.10
N_DRAWS, SEED = 200, 20260716
MAX_EXCLUDED = 0.10


def load_scores() -> pd.DataFrame:
    rows = []
    for p in sorted(SCORES.glob("*.json")):
        cohort, ticker = p.stem.split("_", 1)
        d = json.load(open(p))
        row = {"cohort": cohort, "ticker": ticker}
        if "error" in d:
            row["llm_score"] = np.nan
        else:
            try:
                eq = float(d["earnings_quality_score"])
                bs = float(d["balance_sheet_score"])
                ar = float(d["accounting_risk"])
                row["llm_score"] = (eq + bs + (100 - ar)) / 3
                row.update(eq=eq, bs=bs, acct_risk=ar,
                           n_red_flags=len(d.get("red_flags") or []),
                           confidence=d.get("confidence"))
            except (KeyError, TypeError, ValueError):
                row["llm_score"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    cohorts = pd.read_csv(OUT / "cohorts.csv", dtype={"cohort": str})
    df = cohorts.merge(load_scores(), on=["cohort", "ticker"], how="left")

    # exclusion guard
    for c, grp in df.groupby("cohort"):
        exc = grp.llm_score.isna().mean()
        print(f"cohort {c}: {len(grp)} names, {exc:.1%} without a usable score")
        if exc > MAX_EXCLUDED:
            print(f"FAIL (execution): exclusions {exc:.1%} > {MAX_EXCLUDED:.0%}")
            return 1

    df = df.dropna(subset=["llm_score", "fwd_12m_ret"]).copy()
    df["worst"] = False
    for c, grp in df.groupby("cohort"):
        k = max(1, int(round(len(grp) * TIER_PCT)))
        idx = grp.sort_values(["llm_score", "ticker"]).head(k).index
        df.loc[idx, "worst"] = True

    # check 1 — direction, both cohorts
    per = df.groupby(["cohort", "worst"]).fwd_12m_ret.agg(["mean", "count"])
    c1 = all(per.loc[(c, True), "mean"] < per.loc[(c, False), "mean"]
             for c in df.cohort.unique())

    # check 2 — pooled OLS with composite control
    X = np.column_stack([np.ones(len(df)), df.composite_pct.to_numpy(),
                         df.worst.astype(float).to_numpy()])
    y = df.fwd_12m_ret.to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(df) - X.shape[1]
    se = np.sqrt((resid @ resid) / dof * np.linalg.inv(X.T @ X).diagonal())
    t_worst = float(beta[2] / se[2])
    c2 = t_worst <= -2.0

    # check 3 — size-matched null on the pooled drag
    def pooled_drag(mask: pd.Series) -> float:
        drags = []
        for c, grp in df.groupby("cohort"):
            m = mask[grp.index]
            drags.append(grp.fwd_12m_ret[~m].mean() - grp.fwd_12m_ret[m].mean())
        return float(np.mean(drags))

    actual = pooled_drag(df.worst)
    rng = np.random.default_rng(SEED)
    nulls = []
    for _ in range(N_DRAWS):
        m = pd.Series(False, index=df.index)
        for c, grp in df.groupby("cohort"):
            k = int(grp.worst.sum())
            m[rng.choice(grp.index, size=k, replace=False)] = True
        nulls.append(pooled_drag(m))
    nulls = np.array(nulls)
    q90 = float(np.quantile(nulls, 0.90))
    c3 = actual > q90

    checks = pd.DataFrame([
        {"check": "1_direction_both_cohorts", "value": "", "passed": c1},
        {"check": "2_pooled_t_le_-2", "value": f"t={t_worst:.2f}", "passed": c2},
        {"check": "3_null_q90", "value": f"drag={actual:.3f} q90={q90:.3f} "
         f"pct={float((nulls < actual).mean()):.0%}", "passed": c3},
    ])
    verdict = ("PASS" if c1 and c2 and c3 else
               "DIRECTIONAL_ONLY" if c1 else "FAIL")

    df.to_csv(OUT / "results.csv", index=False)
    checks.assign(verdict=verdict).to_csv(OUT / "verdict.csv", index=False)

    print()
    for c in sorted(df.cohort.unique()):
        w, r = per.loc[(c, True)], per.loc[(c, False)]
        print(f"cohort {c}: worst tier n={w['count']:.0f} "
              f"ret={w['mean']:+.1%} | rest n={r['count']:.0f} "
              f"ret={r['mean']:+.1%}")
    print(checks.to_string(index=False))
    print(f"\nVERDICT: {verdict}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (c, grp) in zip(axes[:2], df.groupby("cohort")):
        ax.scatter(grp.llm_score[~grp.worst], grp.fwd_12m_ret[~grp.worst],
                   s=12, alpha=0.5, label="rest")
        ax.scatter(grp.llm_score[grp.worst], grp.fwd_12m_ret[grp.worst],
                   s=25, color="crimson", label="worst tier")
        ax.axhline(grp.fwd_12m_ret.mean(), color="gray", lw=0.8, ls="--")
        ax.set_title(f"cohort {c}")
        ax.set_xlabel("llm_score")
        ax.set_ylabel("fwd 12M return")
        ax.legend(fontsize=8)
    axes[2].hist(nulls, bins=30, alpha=0.7, label="null drags")
    axes[2].axvline(actual, color="crimson", label=f"actual {actual:.3f}")
    axes[2].axvline(q90, color="gray", ls="--", label=f"q90 {q90:.3f}")
    axes[2].set_title("size-matched null (200 draws)")
    axes[2].legend(fontsize=8)
    fig.suptitle(f"LLM pilot — {verdict}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "pilot.png", dpi=120)
    print(f"wrote {OUT}/results.csv, verdict.csv, pilot.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
