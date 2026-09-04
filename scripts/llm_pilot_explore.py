"""LLM pilot — EXPLORATORY battery over the 239 scored filings.

*** NOT PRE-REGISTERED. ~25 cuts on ~244 non-independent observations
(54 tickers appear in both cohorts): expect 1-2 nominal p<0.05 by chance.
Anything interesting here is a candidate for a forward test, nothing more. ***

All pooled statistics use within-cohort demeaned forward returns (the two
cohorts have very different base years). Tests:
  A  llm_score quintiles (pooled + per cohort), Q5-Q1 spread
  B  component quintiles: earnings quality, balance sheet, accounting risk
  C  red-flag count buckets (0 / 1-2 / 3+)
  D  continuous OLS: fwd ~ cohort-demeaned, composite_pct control, one LLM
     field at a time (t-stat of the field)
  E  confidence moderation: field effect among high- vs low-confidence names
  F  redundancy: rank correlations of LLM fields vs composite percentile

Usage: python scripts/llm_pilot_explore.py
Output: output/llm_pilot/exploration.csv + printed tables
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

OUT = Path("output/llm_pilot")
FIELDS = ["llm_score", "eq", "bs", "acct_risk", "n_red_flags", "confidence"]


def ols_t(y: np.ndarray, X: np.ndarray) -> tuple[float, float]:
    """(beta, t) of the LAST column."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    se = np.sqrt((resid @ resid) / dof * np.linalg.inv(X.T @ X).diagonal())
    return float(beta[-1]), float(beta[-1] / se[-1])


def main() -> int:
    df = pd.read_csv(OUT / "results.csv", dtype={"cohort": str})
    df = df.dropna(subset=["llm_score", "fwd_12m_ret"]).copy()
    df["ret_dm"] = df.fwd_12m_ret - df.groupby("cohort").fwd_12m_ret.transform("mean")
    print(f"n={len(df)} obs ({df.ticker.nunique()} tickers, "
          f"{df[df.cohort == '2024'].shape[0]}/{df[df.cohort == '2025'].shape[0]} per cohort)\n")
    rows = []

    # A/B — quintiles per field (pooled, cohort-demeaned)
    print("=== A/B: quintile mean demeaned fwd 12M return (pooled) ===")
    qtab = {}
    for f in ["llm_score", "eq", "bs", "acct_risk"]:
        q = df.groupby("cohort")[f].transform(
            lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
        means = df.groupby(q).ret_dm.mean()
        qtab[f] = means
        spread = means.get(4, np.nan) - means.get(0, np.nan)
        rows.append({"test": f"quintile_spread_{f}", "value": spread,
                     "detail": "Q5-Q1 demeaned"})
    print(pd.DataFrame(qtab).round(3).to_string())
    print()

    # C — red-flag buckets
    print("=== C: red-flag count buckets ===")
    b = pd.cut(df.n_red_flags, [-1, 0, 2, 99], labels=["0", "1-2", "3+"])
    tab = df.groupby(b).agg(n=("ret_dm", "count"), ret_dm=("ret_dm", "mean"))
    print(tab.round(3).to_string())
    rows.append({"test": "redflags_3plus_minus_0",
                 "value": tab.loc["3+", "ret_dm"] - tab.loc["0", "ret_dm"],
                 "detail": f"n={tab.n.to_dict()}"})
    print()

    # D — continuous OLS with composite control
    print("=== D: OLS ret_dm ~ composite_pct + z(field), t of field ===")
    for f in FIELDS:
        z = df.groupby("cohort")[f].transform(
            lambda s: (s - s.mean()) / (s.std() or 1))
        X = np.column_stack([np.ones(len(df)), df.composite_pct.to_numpy(),
                             z.to_numpy()])
        beta, t = ols_t(df.ret_dm.to_numpy(), X)
        rows.append({"test": f"ols_t_{f}", "value": t,
                     "detail": f"beta={beta:.4f} per 1sd"})
        print(f"  {f:<14} beta/sd={beta:+.4f}  t={t:+.2f}")
    print()

    # E — confidence moderation of llm_score
    print("=== E: llm_score OLS-t within confidence halves ===")
    med = df.groupby("cohort").confidence.transform("median")
    for name, sub in (("high_conf", df[df.confidence >= med]),
                      ("low_conf", df[df.confidence < med])):
        z = sub.groupby("cohort").llm_score.transform(
            lambda s: (s - s.mean()) / (s.std() or 1))
        X = np.column_stack([np.ones(len(sub)),
                             sub.composite_pct.to_numpy(), z.to_numpy()])
        beta, t = ols_t(sub.ret_dm.to_numpy(), X)
        rows.append({"test": f"ols_t_llm_score_{name}", "value": t,
                     "detail": f"n={len(sub)} beta={beta:.4f}"})
        print(f"  {name:<10} n={len(sub):<4} beta/sd={beta:+.4f}  t={t:+.2f}")
    print()

    # F — redundancy vs the quant composite
    print("=== F: Spearman rank corr with composite_pct (pooled) ===")
    for f in FIELDS:
        c = df[[f, "composite_pct"]].corr(method="spearman").iloc[0, 1]
        rows.append({"test": f"spearman_vs_composite_{f}", "value": c,
                     "detail": ""})
        print(f"  {f:<14} rho={c:+.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "exploration.csv", index=False)
    print(f"\nwrote {OUT}/exploration.csv")
    print("REMINDER: exploratory, ~25 cuts, expect 1-2 spurious |t|>2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
