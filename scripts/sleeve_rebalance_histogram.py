"""Per-sleeve weekly- or monthly-rebalance vs drift, one observation per 4-month hold.

For every sim (same random books as monte_carlo_random_book.py / seed s) and
every sleeve review, takes that sleeve's names over its full hold (T+1 exec
at review -> T+1 exec at next review) and computes the sleeve's return two ways:

  drift  : equal-weight at entry, then untouched for the hold
  rebal  : --freq weekly: reset to equal weight on the first session of every
           week inside the hold; --freq monthly: reset on each monthly exec
           date inside the hold (T+1 after each month-end). COST_BPS is paid
           on each reset's turnover

and records diff = rebal - drift. Entry/exit trades are the same for both
and are not charged. Holds that run past the end of data are dropped.

Usage: python scripts/sleeve_rebalance_histogram.py [--freq weekly|monthly] [--sims 500]
Writes: output/monte_carlo_random_book/sleeve_rebalance_hist_{freq}_n5_hold4_sleeves4_sims500.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from research.autoresearch.evaluate import COST_BPS, EXEC_LAG_DAYS, _snap_forward  # noqa: E402
from research.strategies.engine import load_data                                   # noqa: E402
from scripts.monte_carlo_rebalance_frequency import _sleeve_schedule               # noqa: E402

OUT_DIR = REPO / "output" / "monte_carlo_random_book"


def _sleeve_returns(px: np.ndarray, cols: np.ndarray, start: int, end: int,
                    resets: list) -> tuple[float, float]:
    """(drift return, rebalanced return) of an equal-weight basket reset to
    equal weight at each position in `resets`."""
    def rel(a, b):
        r = px[b, cols] / px[a, cols]
        return np.where(np.isfinite(r), r, 1.0)

    drift = float(rel(start, end).mean() - 1.0)
    k = len(cols)
    val = np.full(k, 1.0 / k)
    last = start
    for p in resets:
        val = val * rel(last, p)
        nav = val.sum()
        traded = np.abs(val - nav / k).sum()
        val = np.full(k, (nav - traded * COST_BPS / 1e4) / k)
        last = p
    val = val * rel(last, end)
    return drift, float(val.sum() - 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", choices=("weekly", "monthly"), default="weekly")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--hold", type=int, default=4)
    ap.add_argument("--refresh", type=int, default=5)
    ap.add_argument("--sleeves", type=int, default=4)
    ap.add_argument("--sims", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("loading composite scores + price matrix ...", flush=True)
    data = load_data()
    rebal = list(data.rebal_dates)
    index = data.matrix.index
    pool = sorted({t for s in data.comp.values() for t in s.dropna().index}
                  & set(data.matrix.columns))
    col = {t: i for i, t in enumerate(pool)}
    px = data.matrix[pool].to_numpy(dtype=float)
    exec_pos = [min(index.get_loc(_snap_forward(index, d)) + EXEC_LAG_DAYS, len(index) - 1)
                for d in rebal]
    weeks = pd.DatetimeIndex(pd.to_datetime(index)).to_period("W")

    def resets_for(m: int) -> list:
        a, b = exec_pos[m], exec_pos[m + args.hold]
        if args.freq == "monthly":
            return [exec_pos[m + i] for i in range(1, args.hold)]
        return [p for p in range(a + 1, b) if weeks[p] != weeks[p - 1]]

    rows = []
    for s in range(args.sims):
        sched = _sleeve_schedule(data.comp, rebal, random.Random(args.seed + s),
                                 args.n, args.hold, args.refresh, args.sleeves)
        for m, (holdings, reviewed) in enumerate(sched):
            if m + args.hold >= len(rebal):
                continue                      # hold runs past end of data
            start, end = exec_pos[m], exec_pos[m + args.hold]
            for j in reviewed:
                cols = np.array([col[t] for t in holdings[j]], dtype=int)
                drift, rb = _sleeve_returns(px, cols, start, end, resets_for(m))
                rows.append((s, rebal[m], j, drift, rb))

    df = pd.DataFrame(rows, columns=["sim", "start", "sleeve", "drift", "rebal"])
    df["diff"] = df["rebal"] - df["drift"]
    by_start = df.groupby("start")["diff"].mean()

    def t_stat(x: pd.Series) -> float:
        return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))

    nonoverlap = by_start.iloc[::args.hold]    # starts 4 months apart: holds don't overlap
    summary = {
        "n_sleeve_periods": int(len(df)),
        "n_start_dates": int(len(by_start)),
        "mean_diff": float(df["diff"].mean()),
        "median_diff": float(df["diff"].median()),
        "pct_positive": float((df["diff"] > 0).mean()),
        "p10_diff": float(df["diff"].quantile(0.10)),
        "p90_diff": float(df["diff"].quantile(0.90)),
        "mean_drift_return": float(df["drift"].mean()),
        "mean_rebal_return": float(df["rebal"].mean()),
        "by_start_mean": float(by_start.mean()),
        "by_start_pct_positive": float((by_start > 0).mean()),
        "nonoverlap_n": int(len(nonoverlap)),
        "nonoverlap_t_stat": t_stat(nonoverlap),
    }
    print(json.dumps(summary, indent=2))

    out = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"freq": args.freq, "book_size": args.n, "hold_months": args.hold, "sleeve_count": args.sleeves,
                   "refresh_n": args.refresh, "n_sims": args.sims, "seed": args.seed,
                   "cost_bps": COST_BPS},
        "diffs": df["diff"].round(6).tolist(),
        "by_start": {str(k): float(v) for k, v in by_start.items()},
        "summary": summary,
    }
    name = (f"sleeve_rebalance_hist_{args.freq}_n{args.n}_hold{args.hold}_sleeves{args.sleeves}"
            f"_sims{args.sims}.json")
    out_path = OUT_DIR / name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(out, fh)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
