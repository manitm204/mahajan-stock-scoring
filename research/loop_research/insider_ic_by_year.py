"""Pulls the insider parent factor's IC/IR by year (user request 2026-09-10):
is the 2020-2021 concentration of the paired insider-edge test
(insider_edge_paired.py) explained by the insider PARENT SCORE itself having
its highest IC/IR in those years?

Two IC series per date, both against 1-month forward returns
(research.forward_returns / research.ic -- same machinery used everywhere
else factor IC is measured in this repo):
  full_universe -- insider score vs fwd return across every name with a
                   score that date (the standard, standalone-factor read)
  edge_pool     -- insider score vs fwd return restricted to the same
                   decile-1-minus-100 pool the paired edge test draws from
                   (the "was insider actually differentiating names WITHIN
                   the exact pool we tested" read)

Writes output/loop_research/insider_ic_by_year.csv (per-date) and
output/loop_research/insider_ic_by_year_summary.csv (per-year mean IC / IR /
hit rate for both series, plus the per-year insider-edge sum from
insider_edge_paired_period.json for side-by-side comparison) and a chart.

Usage: python -m research.loop_research.insider_ic_by_year
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from research.forward_returns import compute_forward_returns              # noqa: E402
from research.ic import period_ic                                          # noqa: E402
from research.loop_research import harness as H                            # noqa: E402
from research.loop_research.insider_edge_paired import _draw_pool          # noqa: E402

OUT_DIR = REPO / "output" / "loop_research"
BLUE = "#2a78d6"
MAGENTA = "#e87ba4"
GRAY = "#8a8f98"


def pull(bundle):
    data = bundle["data"]
    parent_scores = bundle["parent_scores"]
    rebal = list(data.rebal_dates)
    fwd_by_h = compute_forward_returns(data.matrix, rebal, {"1M": 1})
    fwd_1m = fwd_by_h["1M"]

    rows = []
    for d in rebal:
        if d not in fwd_1m or d not in parent_scores:
            continue
        ins = parent_scores[d].get("insider")
        if ins is None:
            continue
        fwd = fwd_1m[d]

        ic_full = period_ic(ins, fwd)

        scores_d = data.comp.get(d)
        pool = _draw_pool(scores_d) if scores_d is not None else []
        ic_pool = period_ic(ins.reindex(pool), fwd.reindex(pool)) if pool else None

        rows.append({
            "date": d, "year": pd.Timestamp(d).year,
            "ic_full_universe": ic_full, "ic_edge_pool": ic_pool,
            "n_pool": len(pool),
        })

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for year, g in df.groupby("year"):
        row = {"year": year, "n_periods": len(g)}
        for col in ["ic_full_universe", "ic_edge_pool"]:
            s = g[col].dropna()
            row[f"{col}_mean"] = float(s.mean()) if len(s) else float("nan")
            row[f"{col}_std"] = float(s.std(ddof=1)) if len(s) > 1 else float("nan")
            row[f"{col}_ir"] = (row[f"{col}_mean"] / row[f"{col}_std"]
                                if row[f"{col}_std"] and not np.isnan(row[f"{col}_std"]) and row[f"{col}_std"] > 1e-9
                                else float("nan"))
            row[f"{col}_hit_rate"] = float((s > 0).mean()) if len(s) else float("nan")
        out.append(row)
    return pd.DataFrame(out).sort_values("year")


def _attach_edge(summary: pd.DataFrame) -> pd.DataFrame:
    period_path = OUT_DIR / "insider_edge_paired_period.json"
    if not period_path.exists():
        return summary
    with period_path.open() as fh:
        period = json.load(fh)
    year_sum = {int(k): v for k, v in period["year_sum"].items()}
    summary = summary.copy()
    summary["edge_year_sum"] = summary["year"].map(year_sum)
    return summary


def _plot(summary: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(9, 5))
    years = summary["year"].astype(str)
    x = np.arange(len(years))
    width = 0.35

    ax1.bar(x - width / 2, summary["ic_full_universe_mean"], width,
           color=BLUE, label="insider mean IC, full universe")
    ax1.bar(x + width / 2, summary["ic_edge_pool_mean"], width,
           color=MAGENTA, label="insider mean IC, edge-test pool")
    ax1.axhline(0.0, color=GRAY, linestyle="--", linewidth=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(years)
    ax1.set_ylabel("mean 1M IC (Spearman)")
    ax1.set_title("Insider parent-score IC by year vs the paired-edge finding")
    ax1.legend(loc="upper left", fontsize=8.5)

    if "edge_year_sum" in summary.columns:
        ax2 = ax1.twinx()
        ax2.plot(x, summary["edge_year_sum"], color="black", marker="o",
                linewidth=1.5, label="edge test: per-year (insider-other) sum")
        ax2.set_ylabel("per-year paired-edge sum")
        ax2.legend(loc="upper right", fontsize=8.5)

    fig.tight_layout()
    out_png = OUT_DIR / "insider_ic_by_year.png"
    fig.savefig(out_png, dpi=140)
    print(f"wrote {out_png}")


def main():
    bundle = H.get_data()
    df = pull(bundle)
    df.to_csv(OUT_DIR / "insider_ic_by_year.csv", index=False)
    print(f"wrote {OUT_DIR / 'insider_ic_by_year.csv'} ({len(df)} rows)")

    summary = summarize(df)
    summary = _attach_edge(summary)
    summary.to_csv(OUT_DIR / "insider_ic_by_year_summary.csv", index=False)
    print(f"wrote {OUT_DIR / 'insider_ic_by_year_summary.csv'}")
    print(summary.to_string(index=False))

    _plot(summary)


if __name__ == "__main__":
    main()
