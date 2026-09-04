"""LLM pilot phase 2 — overlay category analysis (pre-registered read).

Pre-registration (docs/llm_pilot_design.md, phase 2): primary = {REVIEW,
AVOID_RED_FLAG} vs {PASS, WATCHLIST} on cohort-demeaned 12M forward return,
two-sided; secondary = {WEAKENS, CONTRADICTS} vs {CONFIRMS}; any side with
<15 pooled names -> descriptive only, no verdict.

Usage: python scripts/llm_pilot_overlay_analysis.py
Outputs: output/llm_pilot/overlay_results.csv, overlay_charts.png
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

from scripts.llm_pilot_charts import monthly_prices, sharpe
from scripts.llm_pilot_factor_llm_charts import ew_series

OUT = Path("output/llm_pilot")
STATUS_ORDER = ["PASS", "WATCHLIST", "REVIEW", "AVOID_RED_FLAG"]
REVIEW_ORDER = ["CONFIRMS", "MIXED", "WEAKENS", "CONTRADICTS"]
FIELDS = ["final_research_status", "quant_signal_review", "thesis_alignment",
          "qualitative_risk_level", "business_quality", "confidence"]


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    return float((a.mean() - b.mean()) / np.sqrt(va + vb))


def main() -> int:
    rows = []
    for p in sorted((OUT / "overlay").glob("*.json")):
        cohort, ticker = p.stem.split("_", 1)
        d = json.load(open(p))
        if "error" in d:
            continue
        rows.append({"cohort": cohort, "ticker": ticker,
                     **{f: d.get(f) for f in FIELDS}})
    ov = pd.DataFrame(rows)
    df = pd.read_csv(OUT / "cohorts.csv", dtype={"cohort": str}).merge(
        ov, on=["cohort", "ticker"], how="inner")
    df = df.dropna(subset=["fwd_12m_ret", "final_research_status"])
    df["ret_dm"] = df.fwd_12m_ret - df.groupby("cohort").fwd_12m_ret.transform("mean")
    print(f"n={len(df)} overlays merged\n")

    print("=== label distributions ===")
    for f in ("final_research_status", "quant_signal_review",
              "thesis_alignment"):
        print(f"{f}: {df[f].value_counts().to_dict()}")
    print()

    print("=== mean demeaned fwd 12M return by category ===")
    tabs = {}
    for f, order in (("final_research_status", STATUS_ORDER),
                     ("quant_signal_review", REVIEW_ORDER)):
        t = df.groupby(f).agg(n=("ret_dm", "count"), ret=("ret_dm", "mean"),
                              raw=("fwd_12m_ret", "mean")).reindex(order)
        tabs[f] = t
        print(t.round(3).to_string(), "\n")

    # pre-registered comparisons (guard: both sides >= 15)
    verdicts = []
    for name, good, bad in (
            ("primary_status", ("PASS", "WATCHLIST"),
             ("REVIEW", "AVOID_RED_FLAG")),
            ("secondary_review", ("CONFIRMS",), ("WEAKENS", "CONTRADICTS"))):
        f = "final_research_status" if "status" in name else "quant_signal_review"
        a = df[df[f].isin(bad)].ret_dm.to_numpy()
        b = df[df[f].isin(good)].ret_dm.to_numpy()
        if len(a) < 15 or len(b) < 15:
            verdicts.append(f"{name}: DESCRIPTIVE ONLY "
                            f"(n flagged={len(a)}, clean={len(b)})")
            continue
        t = welch_t(a, b)
        verdicts.append(f"{name}: flagged-minus-clean = "
                        f"{a.mean() - b.mean():+.3f} (t={t:+.2f}, "
                        f"n={len(a)}/{len(b)})")
    print("=== pre-registered comparisons (two-sided) ===")
    for v in verdicts:
        print(" ", v)

    df.to_csv(OUT / "overlay_results.csv", index=False)

    mpx = monthly_prices(sorted(df.ticker.unique()))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for ax, (f, order) in zip(axes[0], (("final_research_status", STATUS_ORDER),
                                        ("quant_signal_review", REVIEW_ORDER))):
        t = tabs[f].dropna(subset=["ret"])
        ax.bar(t.index, t.ret,
               color=["seagreen" if v >= 0 else "crimson" for v in t.ret])
        for x, (n, v) in enumerate(zip(t.n, t.ret)):
            ax.annotate(f"n={int(n)}", (x, v), ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=9)
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_title(f"mean demeaned fwd 12M return by {f}")

    ax = axes[1][0]
    for status, color in (("PASS", "seagreen"), ("WATCHLIST", "tab:blue"),
                          ("REVIEW", "tab:orange"),
                          ("AVOID_RED_FLAG", "crimson")):
        grp = df[df.final_research_status == status]
        if len(grp) < 5:
            continue
        s = ew_series({c: list(g.ticker) for c, g in grp.groupby("cohort")}, mpx)
        ax.plot((1 + s).cumprod(), color=color, lw=1.8,
                label=f"{status} (n={len(grp)})  SR {sharpe(s):.2f}")
    ax.axhline(1.0, color="gray", lw=0.6)
    ax.set_title("growth of $1 by research status (24M stitched, EW)")
    ax.legend(fontsize=8)

    ax = axes[1][1]
    q = df.groupby("cohort").composite_pct.transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=False))
    flagged = df.final_research_status.isin(("REVIEW", "AVOID_RED_FLAG"))
    tab = df.assign(q=q, flagged=flagged).groupby(["q", "flagged"]).ret_dm.mean().unstack()
    tab.plot.bar(ax=ax, color=["seagreen", "crimson"], rot=0)
    ax.set_xticklabels(["low composite\ntercile", "mid", "high"])
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_title("clean vs flagged within composite terciles")
    ax.legend(["clean (PASS/WATCH)", "flagged (REVIEW/AVOID)"], fontsize=8)

    fig.suptitle("Production-overlay replication — pre-registered phase 2 "
                 "(gpt-oss-120b, PIT quant block + filing analysis)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "overlay_charts.png", dpi=120)
    print(f"\nwrote {OUT}/overlay_results.csv, overlay_charts.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
