"""LLM pilot — score-vs-performance chart dashboard (EXPLORATORY).

Row 1  scatter: each LLM score vs 12M forward return (colored by cohort,
       per-cohort OLS fit).
Row 2  quintile bars: mean cohort-demeaned forward return per score quintile.
Row 3  Sharpe by quintile (24-month stitched series: cohort 2024's forward
       year then cohort 2025's, EW within quintile, renormalized monthly)
       + cumulative growth of accounting-risk Q5 vs Q1 vs the whole book.

Usage: python scripts/llm_pilot_charts.py
Output: output/llm_pilot/score_charts.png (+ printed Sharpe table)
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

from backtesting import data_loader as dl
from data.db import get_db

OUT = Path("output/llm_pilot")
FIELDS = [("eq", "earnings quality"), ("bs", "balance sheet"),
          ("acct_risk", "accounting risk"), ("llm_score", "overall llm_score")]


def monthly_prices(tickers: list[str]) -> pd.DataFrame:
    with get_db() as db:
        px = dl.load_price_matrix(db, tickers, "2024-06-01", "2026-07-06")
    px.index = pd.to_datetime(px.index)
    return px.resample("ME").last()


def quintile_series(df: pd.DataFrame, field: str,
                    mpx: pd.DataFrame) -> dict[int, pd.Series]:
    """24-month EW monthly-return series per quintile (cohorts stitched)."""
    out: dict[int, list[pd.Series]] = {q: [] for q in range(5)}
    for cohort, (start, end) in {"2024": ("2024-06-30", "2025-06-30"),
                                 "2025": ("2025-06-30", "2026-06-30")}.items():
        grp = df[df.cohort == cohort]
        q = pd.qcut(grp[field].rank(method="first"), 5, labels=False)
        for qi in range(5):
            names = [t for t in grp.ticker[q == qi] if t in mpx.columns]
            sub = mpx.loc[start:end, names]
            rets = sub.pct_change().iloc[1:]          # monthly, EW of available
            out[qi].append(rets.mean(axis=1))
    return {qi: pd.concat(parts) for qi, parts in out.items()}


def sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std(ddof=1) * np.sqrt(12))


def main() -> int:
    df = pd.read_csv(OUT / "results.csv", dtype={"cohort": str})
    df = df.dropna(subset=["llm_score", "fwd_12m_ret"]).copy()
    df["ret_dm"] = df.fwd_12m_ret - df.groupby("cohort").fwd_12m_ret.transform("mean")
    mpx = monthly_prices(sorted(df.ticker.unique()))

    fig = plt.figure(figsize=(19, 13))
    colors = {"2024": "tab:blue", "2025": "tab:orange"}

    for i, (f, label) in enumerate(FIELDS):
        ax = fig.add_subplot(3, 4, i + 1)
        for c, grp in df.groupby("cohort"):
            ax.scatter(grp[f], grp.fwd_12m_ret, s=14, alpha=0.55,
                       color=colors[c], label=f"cohort {c}")
            b = np.polyfit(grp[f], grp.fwd_12m_ret, 1)
            xs = np.linspace(grp[f].min(), grp[f].max(), 20)
            ax.plot(xs, np.polyval(b, xs), color=colors[c], lw=1.5)
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_title(f"{label} vs fwd 12M return")
        ax.set_xlabel(f)
        if i == 0:
            ax.set_ylabel("fwd 12M return")
            ax.legend(fontsize=8)

    for i, (f, label) in enumerate(FIELDS):
        ax = fig.add_subplot(3, 4, 4 + i + 1)
        q = df.groupby("cohort")[f].transform(
            lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
        means = df.groupby(q).ret_dm.mean()
        ax.bar([f"Q{int(k) + 1}" for k in means.index], means.values,
               color=["crimson" if v < 0 else "seagreen" for v in means.values])
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_title(f"{label}: quintile mean (demeaned)")
        if i == 0:
            ax.set_ylabel("mean demeaned fwd 12M ret")

    # Sharpe by quintile (stitched 24M series)
    ax = fig.add_subplot(3, 2, 5)
    width = 0.35
    sharpe_rows = {}
    for j, (f, label) in enumerate([("llm_score", "overall llm_score"),
                                    ("acct_risk", "accounting risk")]):
        qs = quintile_series(df, f, mpx)
        sh = [sharpe(qs[qi]) for qi in range(5)]
        sharpe_rows[label] = sh
        ax.bar(np.arange(5) + (j - 0.5) * width, sh, width, label=label)
    ax.set_xticks(range(5), [f"Q{i+1}" for i in range(5)])
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_title("Sharpe by score quintile (24M stitched, EW)")
    ax.legend(fontsize=9)

    # cumulative growth: acct_risk Q5 vs Q1 vs book
    ax = fig.add_subplot(3, 2, 6)
    qs = quintile_series(df, "acct_risk", mpx)
    book = pd.concat([
        mpx.loc["2024-06-30":"2025-06-30",
                [t for t in df[df.cohort == "2024"].ticker
                 if t in mpx.columns]].pct_change().iloc[1:].mean(axis=1),
        mpx.loc["2025-06-30":"2026-06-30",
                [t for t in df[df.cohort == "2025"].ticker
                 if t in mpx.columns]].pct_change().iloc[1:].mean(axis=1)])
    for s, label, color in [(qs[4], "acct risk Q5 (scariest)", "crimson"),
                            (qs[0], "acct risk Q1 (cleanest)", "seagreen"),
                            (book, "whole book (EW)", "gray")]:
        ax.plot((1 + s).cumprod(), label=f"{label}  SR {sharpe(s):.2f}",
                color=color, lw=1.8)
    ax.set_title("growth of $1 over the two stitched forward years")
    ax.legend(fontsize=9)

    fig.suptitle("LLM filing scores vs forward performance — EXPLORATORY "
                 "(239 names, 2 cohorts, scores by gpt-oss-120b)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "score_charts.png", dpi=120)

    print(pd.DataFrame(sharpe_rows, index=[f"Q{i+1}" for i in range(5)])
          .round(2).to_string())
    print(f"book EW Sharpe: {sharpe(book):.2f}")
    print(f"wrote {OUT}/score_charts.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
