"""Did the LLM PASS verdicts from the 2026-07-27 report batch outperform?

The 2026-07-27 run generated 107 research memos (output/reports/2026-07-27/),
all quant Signal=LONG, each tagged with a Research status:
PASS / WATCHLIST / REVIEW / AVOID (RED FLAG).

This checks realized return from 2026-07-27 (report date) through the latest
available price date for PASS names vs. the rest of the same batch. Small
n (11 PASS) -- directional look, not a significance test.

Usage: python scripts/llm_report_batch_0727_analysis.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from backtesting import data_loader as dl
from data.db import get_db

REPORT_DIR = Path("output/reports/2026-07-27")
START = "2026-07-27"

STATUS_RE = re.compile(r"\*\*Research status:\s*([A-Z ()]+?)\s*-")
COMPOSITE_RE = re.compile(r"\*\*Composite score:\*\*\s*([\d.]+)")


def load_batch() -> pd.DataFrame:
    rows = []
    for p in sorted(REPORT_DIR.glob("*.md")):
        text = p.read_text()
        status_m = STATUS_RE.search(text)
        comp_m = COMPOSITE_RE.search(text)
        rows.append({
            "ticker": p.stem,
            "status": status_m.group(1).strip() if status_m else None,
            "composite": float(comp_m.group(1)) if comp_m else None,
        })
    return pd.DataFrame(rows)


def main() -> None:
    batch = load_batch()
    print(f"Loaded {len(batch)} reports from {REPORT_DIR}")
    print(batch["status"].value_counts())

    tickers = batch["ticker"].tolist()
    with get_db() as db:
        px = dl.load_price_matrix(db, tickers, START, "2026-12-31")
    px.index = pd.to_datetime(px.index)

    last_date = px.index.max()
    start_px = px.loc[px.index >= START].iloc[0]
    end_px = px.iloc[-1]
    fwd_ret = (end_px / start_px - 1.0).rename("fwd_return")

    batch = batch.merge(fwd_ret, left_on="ticker", right_index=True, how="left")
    batch["is_pass"] = batch["status"] == "PASS"

    print(f"\nReturn window: {px.loc[px.index >= START].index[0].date()} "
          f"to {last_date.date()} ({(last_date - pd.Timestamp(START)).days} days)")

    n_missing = batch["fwd_return"].isna().sum()
    if n_missing:
        print(f"Missing price data for {n_missing} tickers: "
              f"{batch.loc[batch.fwd_return.isna(), 'ticker'].tolist()}")
    batch = batch.dropna(subset=["fwd_return"])

    print("\n=== By research status ===")
    summary = batch.groupby("status")["fwd_return"].agg(["mean", "median", "std", "count"])
    print(summary.sort_values("mean", ascending=False).to_string(float_format=lambda x: f"{x:.4f}"))

    pass_ret = batch.loc[batch.is_pass, "fwd_return"]
    rest_ret = batch.loc[~batch.is_pass, "fwd_return"]
    print("\n=== PASS vs. rest of batch ===")
    print(f"PASS   (n={len(pass_ret)}): mean={pass_ret.mean():.4f}  median={pass_ret.median():.4f}")
    print(f"REST   (n={len(rest_ret)}): mean={rest_ret.mean():.4f}  median={rest_ret.median():.4f}")
    print(f"Spread (PASS - REST): {pass_ret.mean() - rest_ret.mean():.4f}")

    from scipy import stats
    t, p = stats.ttest_ind(pass_ret, rest_ret, equal_var=False)
    print(f"Welch t-test: t={t:.3f}  p={p:.3f}  (n too small to trust; directional only)")

    out = Path("output/llm_pilot/report_batch_0727.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    batch.sort_values("fwd_return", ascending=False).to_csv(out, index=False)
    print(f"\nSaved detail to {out}")


if __name__ == "__main__":
    main()
