"""LLM pilot — (composite quintile × overlay label) cells, one common scale.

Metric everywhere: mean 6-MONTH forward return of the cell (pooled across the
two cohorts; quintiles formed per cohort). EXPLORATORY — cells are 5-40
names; n is printed on every bar.

Panel 1  final_research_status labels across quintiles (grouped bars)
Panel 2  quant_signal_review labels across quintiles (grouped bars)
         -- panels 1-2 share identical y-limits --
Panel 3  leaderboard: every (label x quintile) cell with n >= 3, both fields
         on the same axis, sorted best to worst.

Usage: python scripts/llm_pilot_quintile_labels.py
Output: output/llm_pilot/quintile_label_charts.png
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

from scripts.llm_pilot_charts import monthly_prices
from scripts.llm_pilot_overlay_6m import fwd6m, load

OUT = Path("output/llm_pilot")
STATUS = ["PASS", "WATCHLIST", "REVIEW", "AVOID_RED_FLAG"]
REVIEW = ["CONFIRMS", "MIXED", "WEAKENS"]
COLORS = {"PASS": "seagreen", "WATCHLIST": "tab:blue", "REVIEW": "tab:orange",
          "AVOID_RED_FLAG": "crimson", "CONFIRMS": "seagreen",
          "MIXED": "tab:blue", "WEAKENS": "tab:orange"}
MIN_N = 3


def cell_table(df: pd.DataFrame, field: str, labels: list[str]) -> pd.DataFrame:
    rows = []
    for q in range(1, 6):
        for lab in labels:
            grp = df[(df.comp_q == q) & (df[field] == lab)]
            grp = grp.dropna(subset=["fwd_6m_ret"])
            if len(grp):
                rows.append({"field": field, "label": lab, "comp_q": q,
                             "n": len(grp),
                             "ret": float(grp.fwd_6m_ret.mean())})
    return pd.DataFrame(rows)


def main() -> int:
    df = load()
    mpx = monthly_prices(sorted(df.ticker.unique()))
    df["fwd_6m_ret"] = [fwd6m(mpx, t, c) for t, c in zip(df.ticker, df.cohort)]
    df["comp_q"] = df.groupby("cohort").composite_pct.transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1)

    t_status = cell_table(df, "final_research_status", STATUS)
    t_review = cell_table(df, "quant_signal_review", REVIEW)
    both = pd.concat([t_status, t_review])
    ylim = (both[both.n >= MIN_N].ret.min() - 0.02,
            both[both.n >= MIN_N].ret.max() + 0.02)

    fig = plt.figure(figsize=(18, 13))

    for pi, (t, labels, title) in enumerate((
            (t_status, ["PASS", "WATCHLIST"], "final_research_status"),
            (t_review, REVIEW, "quant_signal_review"))):
        ax = fig.add_subplot(2, 2, pi + 1)
        width = 0.8 / len(labels)
        for j, lab in enumerate(labels):
            sub = t[(t.label == lab) & (t.n >= MIN_N)].set_index("comp_q")
            xs = [q + (j - (len(labels) - 1) / 2) * width
                  for q in sub.index]
            ax.bar(xs, sub.ret, width * 0.95, color=COLORS[lab], label=lab)
            for x, (n, v) in zip(xs, zip(sub.n, sub.ret)):
                ax.annotate(f"{int(n)}", (x, v), ha="center", fontsize=8,
                            va="bottom" if v >= 0 else "top")
        ax.set_xticks(range(1, 6),
                      [f"Q{q}" for q in range(1, 6)])
        ax.set_ylim(*ylim)
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_title(f"mean 6M fwd return: {title} x composite quintile "
                     f"(bar labels = n)")
        ax.set_ylabel("mean 6M fwd return")
        ax.legend(fontsize=9)

    ax = fig.add_subplot(2, 1, 2)
    lb = both[both.n >= MIN_N].copy()
    lb["cell"] = lb.apply(
        lambda r: f"Q{int(r.comp_q)} {r.label} (n={int(r.n)})", axis=1)
    lb = lb.sort_values("ret")
    ax.barh(lb.cell, lb.ret,
            color=[COLORS[l] for l in lb.label],
            edgecolor=["black" if f == "quant_signal_review" else "none"
                       for f in lb.field], linewidth=1.2)
    ax.axvline(0, color="gray", lw=0.8)
    ax.axvline(df.fwd_6m_ret.mean(), color="purple", lw=1.2, ls="--",
               label=f"whole book {df.fwd_6m_ret.mean():+.1%}")
    ax.set_title("leaderboard — every (quintile x label) cell with n >= "
                 f"{MIN_N}, same scale (black edge = quant_signal_review)")
    ax.set_xlabel("mean 6M fwd return")
    ax.legend(fontsize=9)

    fig.suptitle("Composite quintile x overlay label — 6M forward returns, "
                 "common scale (EXPLORATORY, cells are small)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "quintile_label_charts.png", dpi=120)

    print(both.sort_values("ret", ascending=False).round(3).to_string(index=False))
    dropped = both[both.n < MIN_N]
    if len(dropped):
        print(f"\ncells dropped from charts (n < {MIN_N}):")
        print(dropped.round(3).to_string(index=False))
    print(f"\nwrote {OUT}/quintile_label_charts.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
