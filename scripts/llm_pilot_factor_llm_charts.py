"""LLM pilot — composite-quintile × llm_score interaction charts (EXPLORATORY).

Panels 1-5: one per QUANT COMPOSITE quintile (Q5 = top 20% by composite
percentile at formation, per cohort). Within each bucket, names are split at
the median overall llm_score (per cohort), and the two halves are held as EW
portfolios through the stitched 24 forward months (cohort 2024's forward
year, then cohort 2025's). Sharpe in the legend.

Panel 6: summary bars — mean cohort-demeaned 12M forward return of the
high-llm and low-llm half in every composite quintile, so high-llm names can
be compared ACROSS buckets (e.g. top-composite high-llm vs mid-composite
high-llm).

Cell sizes are ~12 names per cohort — this is a visualization of thin cells,
not a test.

Usage: python scripts/llm_pilot_factor_llm_charts.py
Output: output/llm_pilot/factor_llm_charts.png (+ printed table)
"""
from __future__ import annotations

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

from scripts.llm_pilot_charts import monthly_prices, sharpe

OUT = Path("output/llm_pilot")
WINDOWS = {"2024": ("2024-06-30", "2025-06-30"),
           "2025": ("2025-06-30", "2026-06-30")}


def ew_series(names_by_cohort: dict[str, list[str]],
              mpx: pd.DataFrame) -> pd.Series:
    parts = []
    for c, (start, end) in WINDOWS.items():
        names = [t for t in names_by_cohort.get(c, []) if t in mpx.columns]
        rets = mpx.loc[start:end, names].pct_change().iloc[1:]
        parts.append(rets.mean(axis=1))
    return pd.concat(parts)


def main() -> int:
    df = pd.read_csv(OUT / "results.csv", dtype={"cohort": str})
    df = df.dropna(subset=["llm_score", "fwd_12m_ret"]).copy()
    df["ret_dm"] = df.fwd_12m_ret - df.groupby("cohort").fwd_12m_ret.transform("mean")
    df["comp_q"] = df.groupby("cohort").composite_pct.transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
    df["llm_half"] = df.groupby(["cohort", "comp_q"]).llm_score.transform(
        lambda s: (s >= s.median()).astype(int))   # 1 = high-llm half
    mpx = monthly_prices(sorted(df.ticker.unique()))

    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    rows = []
    for q in range(5):
        ax = axes.flat[q]
        bucket = df[df.comp_q == q]
        for half, label, color in [(1, "high llm_score", "seagreen"),
                                   (0, "low llm_score", "crimson")]:
            grp = bucket[bucket.llm_half == half]
            names = {c: list(g.ticker) for c, g in grp.groupby("cohort")}
            s = ew_series(names, mpx)
            sr = sharpe(s)
            ax.plot((1 + s).cumprod(), color=color, lw=1.8,
                    label=f"{label} (n={len(grp)})  SR {sr:.2f}")
            rows.append({"comp_quintile": f"Q{q+1}", "half": label,
                         "n": len(grp), "sharpe": sr,
                         "mean_demeaned_fwd": grp.ret_dm.mean()})
        title = {0: "composite Q1 (bottom 20%)", 4: "composite Q5 (top 20%)"}
        ax.set_title(title.get(q, f"composite Q{q+1}"))
        ax.axhline(1.0, color="gray", lw=0.6)
        ax.legend(fontsize=8)

    res = pd.DataFrame(rows)
    ax = axes.flat[5]
    width = 0.35
    for j, (label, color) in enumerate([("high llm_score", "seagreen"),
                                        ("low llm_score", "crimson")]):
        sub = res[res.half == label]
        ax.bar(np.arange(5) + (j - 0.5) * width, sub.mean_demeaned_fwd,
               width, color=color, label=label)
    ax.set_xticks(range(5), [f"comp\nQ{i+1}" for i in range(5)])
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_title("mean demeaned fwd 12M return by cell")
    ax.set_ylabel("mean demeaned fwd return")
    ax.legend(fontsize=9)

    fig.suptitle("Quant composite quintile × LLM score half — EXPLORATORY, "
                 "~24 names per cell", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "factor_llm_charts.png", dpi=120)

    print(res.round(3).to_string(index=False))
    print(f"wrote {OUT}/factor_llm_charts.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
