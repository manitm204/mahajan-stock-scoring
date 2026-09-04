"""LLM pilot phase 2 — full overlay-field chart dashboard (EXPLORATORY).

Every categorical judgment the overlay emitted, against forward returns:
Rows 1-2  mean cohort-demeaned 12M forward return per category (n annotated)
          for thesis_alignment, qualitative_risk_level, business_quality,
          management_tone, accounting_risk, filing_risk.
Row 3     growth of $1 (24M stitched, EW, Sharpe in legend) by thesis
          alignment and by qualitative risk level; red-flag-count buckets.

Cells with n < 5 are dropped from growth panels. This is a picture of the
same 237 mined observations — no test, no verdict.

Usage: python scripts/llm_pilot_overlay_charts2.py
Output: output/llm_pilot/overlay_charts2.png
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
import pandas as pd

from scripts.llm_pilot_charts import monthly_prices, sharpe
from scripts.llm_pilot_factor_llm_charts import ew_series

OUT = Path("output/llm_pilot")
ORDERS = {
    "thesis_alignment": ["BULLISH", "NEUTRAL", "BEARISH"],
    "qualitative_risk_level": ["LOW", "MEDIUM", "HIGH"],
    "business_quality": ["HIGH", "MEDIUM", "LOW"],
    "management_tone": ["CONFIDENT", "REALISTIC", "CAUTIOUS", "DEFENSIVE",
                        "PROMOTIONAL", "UNKNOWN"],
    "accounting_risk": ["LOW", "MEDIUM", "HIGH", "UNKNOWN"],
    "filing_risk": ["LOW", "MEDIUM", "HIGH", "UNKNOWN"],
}


def load() -> pd.DataFrame:
    rows = []
    for p in sorted((OUT / "overlay").glob("*.json")):
        cohort, ticker = p.stem.split("_", 1)
        d = json.load(open(p))
        if "error" in d:
            continue
        rows.append({"cohort": cohort, "ticker": ticker,
                     "n_red_flags": len(d.get("red_flags") or []),
                     **{f: d.get(f) for f in ORDERS}})
    ov = pd.DataFrame(rows)
    df = pd.read_csv(OUT / "cohorts.csv", dtype={"cohort": str}).merge(
        ov, on=["cohort", "ticker"], how="inner").dropna(subset=["fwd_12m_ret"])
    df["ret_dm"] = df.fwd_12m_ret - df.groupby("cohort").fwd_12m_ret.transform("mean")
    return df


def bar_panel(ax, df: pd.DataFrame, field: str) -> None:
    t = (df.groupby(field).agg(n=("ret_dm", "count"), ret=("ret_dm", "mean"))
         .reindex([c for c in ORDERS[field]
                   if c in set(df[field].dropna())]).dropna())
    ax.bar(t.index, t.ret,
           color=["seagreen" if v >= 0 else "crimson" for v in t.ret])
    for x, (n, v) in enumerate(zip(t.n, t.ret)):
        ax.annotate(f"n={int(n)}", (x, v), ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=9)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_title(field)
    ax.tick_params(axis="x", labelsize=8)


def growth_panel(ax, df: pd.DataFrame, field: str, mpx: pd.DataFrame,
                 colors: dict[str, str]) -> None:
    for cat, color in colors.items():
        grp = df[df[field] == cat]
        if len(grp) < 5:
            continue
        s = ew_series({c: list(g.ticker) for c, g in grp.groupby("cohort")}, mpx)
        ax.plot((1 + s).cumprod(), color=color, lw=1.8,
                label=f"{cat} (n={len(grp)})  SR {sharpe(s):.2f}")
    ax.axhline(1.0, color="gray", lw=0.6)
    ax.set_title(f"growth of $1 by {field}")
    ax.legend(fontsize=8)


def main() -> int:
    df = load()
    print(f"n={len(df)}")
    mpx = monthly_prices(sorted(df.ticker.unique()))

    fig, axes = plt.subplots(3, 3, figsize=(19, 14))
    for ax, field in zip(axes.flat[:6], ORDERS):
        bar_panel(ax, df, field)
    axes.flat[0].set_ylabel("mean demeaned fwd 12M ret")
    axes.flat[3].set_ylabel("mean demeaned fwd 12M ret")

    growth_panel(axes.flat[6], df, "thesis_alignment", mpx,
                 {"BULLISH": "seagreen", "NEUTRAL": "tab:blue",
                  "BEARISH": "crimson"})
    growth_panel(axes.flat[7], df, "qualitative_risk_level", mpx,
                 {"LOW": "seagreen", "MEDIUM": "tab:blue", "HIGH": "crimson"})

    ax = axes.flat[8]
    b = pd.cut(df.n_red_flags, [-1, 1, 3, 99], labels=["0-1", "2-3", "4+"])
    t = df.groupby(b).agg(n=("ret_dm", "count"), ret=("ret_dm", "mean")).dropna()
    ax.bar(t.index.astype(str), t.ret,
           color=["seagreen" if v >= 0 else "crimson" for v in t.ret])
    for x, (n, v) in enumerate(zip(t.n, t.ret)):
        ax.annotate(f"n={int(n)}", (x, v), ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=9)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_title("overlay red-flag count")

    fig.suptitle("Production-overlay fields vs forward return — EXPLORATORY "
                 "(237 names, gpt-oss-120b)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "overlay_charts2.png", dpi=120)

    for field in ORDERS:
        t = df.groupby(field).ret_dm.agg(["count", "mean"]).round(3)
        print(f"\n{field}:\n{t.to_string()}")
    print(f"\nwrote {OUT}/overlay_charts2.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
