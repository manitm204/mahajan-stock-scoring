"""LLM pilot — overlay fields vs 6-MONTH forward growth, every field (EXPLORATORY).

Panels 1-8   one per overlay field: EW growth of $1 over the first 6 forward
             months, one curve per category (average of the two cohort paths,
             n pooled in the legend). Every category is drawn, however small
             — read n before believing a line.
Panel 9      mean 6M forward return by quant-composite quintile (Q1 low →
             Q5 high, formed per cohort), n annotated.
Panels 10-11 label composition within each composite quintile:
             quant_signal_review and final_research_status shares.

Usage: python scripts/llm_pilot_overlay_6m.py
Output: output/llm_pilot/overlay_6m_charts.png
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

OUT = Path("output/llm_pilot")
WINDOWS_6M = {"2024": ("2024-06-30", "2024-12-31"),
              "2025": ("2025-06-30", "2025-12-31")}
FIELDS = {
    "final_research_status": ["PASS", "WATCHLIST", "REVIEW", "AVOID_RED_FLAG"],
    "quant_signal_review": ["CONFIRMS", "MIXED", "WEAKENS", "CONTRADICTS"],
    "thesis_alignment": ["BULLISH", "NEUTRAL", "BEARISH"],
    "qualitative_risk_level": ["LOW", "MEDIUM", "HIGH"],
    "business_quality": ["HIGH", "MEDIUM", "LOW"],
    "management_tone": ["CONFIDENT", "REALISTIC", "CAUTIOUS", "DEFENSIVE",
                        "PROMOTIONAL", "UNKNOWN"],
    "accounting_risk_cat": ["LOW", "MEDIUM", "HIGH", "UNKNOWN"],
    "filing_risk": ["LOW", "MEDIUM", "HIGH", "UNKNOWN"],
}
PALETTE = ["seagreen", "tab:blue", "tab:orange", "crimson", "purple", "gray"]


def load() -> pd.DataFrame:
    rows = []
    for p in sorted((OUT / "overlay").glob("*.json")):
        cohort, ticker = p.stem.split("_", 1)
        d = json.load(open(p))
        if "error" in d:
            continue
        rows.append({"cohort": cohort, "ticker": ticker,
                     "final_research_status": d.get("final_research_status"),
                     "quant_signal_review": d.get("quant_signal_review"),
                     "thesis_alignment": d.get("thesis_alignment"),
                     "qualitative_risk_level": d.get("qualitative_risk_level"),
                     "business_quality": d.get("business_quality"),
                     "management_tone": d.get("management_tone"),
                     "accounting_risk_cat": d.get("accounting_risk"),
                     "filing_risk": d.get("filing_risk")})
    ov = pd.DataFrame(rows)
    return pd.read_csv(OUT / "cohorts.csv", dtype={"cohort": str}).merge(
        ov, on=["cohort", "ticker"], how="inner")


def growth_path(names_by_cohort: dict[str, list[str]],
                mpx: pd.DataFrame) -> pd.Series | None:
    """Average of the per-cohort EW cumulative-growth paths, months 0..6."""
    paths = []
    for c, (start, end) in WINDOWS_6M.items():
        names = [t for t in names_by_cohort.get(c, []) if t in mpx.columns]
        if not names:
            continue
        rets = mpx.loc[start:end, names].pct_change().iloc[1:].mean(axis=1)
        g = (1 + rets).cumprod()
        g = pd.concat([pd.Series([1.0]), pd.Series(g.values)],
                      ignore_index=True)          # month 0..6
        paths.append(g)
    if not paths:
        return None
    n = min(len(p) for p in paths)
    return pd.concat([p.iloc[:n] for p in paths], axis=1).mean(axis=1)


def fwd6m(mpx: pd.DataFrame, ticker: str, cohort: str) -> float:
    start, end = WINDOWS_6M[cohort]
    if ticker not in mpx.columns:
        return float("nan")
    px = mpx.loc[start:end, ticker].dropna()
    return float(px.iloc[-1] / px.iloc[0] - 1) if len(px) >= 2 else float("nan")


def main() -> int:
    df = load()
    mpx = monthly_prices(sorted(df.ticker.unique()))
    df["fwd_6m_ret"] = [fwd6m(mpx, t, c) for t, c in zip(df.ticker, df.cohort)]
    df["comp_q"] = df.groupby("cohort").composite_pct.transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1)
    print(f"n={len(df)} names")

    fig, axes = plt.subplots(4, 3, figsize=(19, 19))

    for ax, (field, order) in zip(axes.flat[:8], FIELDS.items()):
        cats = [c for c in order if c in set(df[field].dropna())]
        for cat, color in zip(cats, PALETTE):
            grp = df[df[field] == cat]
            path = growth_path({c: list(g.ticker)
                                for c, g in grp.groupby("cohort")}, mpx)
            if path is None:
                continue
            ax.plot(path.index, path.values, color=color, lw=1.8,
                    label=f"{cat} (n={len(grp)})")
        ax.axhline(1.0, color="gray", lw=0.6)
        ax.set_title(field)
        ax.set_xlabel("months since formation")
        ax.legend(fontsize=8)

    ax = axes.flat[8]
    t = df.dropna(subset=["fwd_6m_ret"]).groupby("comp_q").agg(
        n=("fwd_6m_ret", "count"), ret=("fwd_6m_ret", "mean"))
    ax.bar([f"Q{int(q)}" for q in t.index], t.ret,
           color=["seagreen" if v >= 0 else "crimson" for v in t.ret])
    for x, (n, v) in enumerate(zip(t.n, t.ret)):
        ax.annotate(f"n={int(n)}", (x, v), ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=9)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_title("mean 6M fwd return by composite quintile (Q5 = best factor score)")

    for ax, field in zip(axes.flat[9:11],
                         ("quant_signal_review", "final_research_status")):
        shares = (df.groupby("comp_q")[field].value_counts(normalize=True)
                  .unstack().reindex(columns=[c for c in FIELDS[field]
                                              if c in df[field].dropna().unique()])
                  .fillna(0))
        counts = df.groupby("comp_q")[field].count()
        bottom = np.zeros(len(shares))
        for cat, color in zip(shares.columns, PALETTE):
            ax.bar([f"Q{int(q)}\nn={counts[q]}" for q in shares.index],
                   shares[cat], bottom=bottom, color=color, label=cat)
            bottom += shares[cat].values
        ax.set_title(f"{field} mix by composite quintile")
        ax.set_ylabel("share of names")
        ax.legend(fontsize=8)

    axes.flat[11].axis("off")
    fig.suptitle("Overlay fields vs 6-MONTH forward growth — EXPLORATORY "
                 "(237 LLM-covered names; curves = average of the two "
                 "cohort paths)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "overlay_6m_charts.png", dpi=120)

    print(t.round(3).to_string())
    print(f"wrote {OUT}/overlay_6m_charts.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
