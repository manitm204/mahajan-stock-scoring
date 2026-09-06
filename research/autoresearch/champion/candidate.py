"""AUTORESEARCH CANDIDATE -- this is the ONLY file the research agent may
modify. Everything else in research/autoresearch/ (evaluate.py, program.md,
results.tsv) and the correctness tests in tests/test_autoresearch_evaluate.py
are fixed: data loading, dev/val/holdout date boundaries, T+1 execution,
transaction costs, return/metric/score calculation, and leakage tests all
live outside this file and cannot be changed from here.

Contract
--------
    generate_targets(context) -> pd.DataFrame

`context` exposes exactly three read-only attributes -- nothing else, ever:
    context.rebal_dates : tuple[str]  monthly PIT signal dates (2020-01 -> 2026-06)
    context.comp         : {date -> pd.Series(ticker -> 0-100 composite score)}
                            already point-in-time correct, read-only
    context.k            : int, default book size (10)

No prices. No returns. No performance results of any kind, for any period
(dev, val, or the locked 2025+ holdout) -- so there is nothing here to
overfit a holdout to even in principle.

Must return a long-format DataFrame with columns:
    date    (str, must be one of context.rebal_dates)
    ticker  (str)
    weight  (float, target PORTFOLIO weight as of that decision date)

------------------------------------------------------------------------
Current champion (session 3, round 67 of 2026-09-06, 146 candidates tried
this session on top of 49 from sessions 1-2, research_score = min(dev_ir,
val_ir) vs SPY, strict promotion rule):

  - top 10 names by composite score
  - THREE overlapping sleeves, each held 4 months (round 67; was 6 months
    in sessions 1-2 -- session 3 retested the full HOLD_MONTHS sweep under
    momentum-eviction, since the original sweep in session 1 predated it,
    and 4 turned out to beat 6 once combined with REFRESH_N=2)
  - at each reform, replace the 2 held names with the LARGEST score DECLINE
    since the sleeve's own last reform (4 months ago) -- not just the
    absolute worst-ranked names this month (round 36, session 2). A name
    whose score has risen since last review is never evicted purely for
    being the lowest-ranked of the 10; eviction targets genuine
    deterioration.
  - equal weighting within each sleeve (1/3 of NAV / 10 names once ramped)
  - costs and T+1 execution are handled entirely by evaluate.py

research_score history: baseline 0.253 -> HOLD_MONTHS=6 (session 1, round 3)
0.302 -> worst-rank partial rotation (round 9) 0.362 -> momentum-eviction
(session 2, round 36) 0.585 -> HOLD_MONTHS=4 + REFRESH_N=2 under
momentum-eviction (session 3, round 67) 1.060. 146 other session-3 ideas
(finer HOLD/REFRESH_N grids, decline-metric variants, no-recycle memory,
adaptive/buffered refresh counts, multi-horizon sleeves, percentile/entry
floors, tenure locks, calendar-skip rules, etc.) were all tried and
rejected -- see session_log.md for the full round-by-round table across
all three sessions.
"""
from __future__ import annotations

import pandas as pd

from research.strategies.engine import _top_k



def _fill_to_k(preferred, exclude, full_scores, need):
    out = [t for t in preferred if t not in exclude][:need]
    if len(out) < need:
        seen = set(exclude) | set(out)
        remaining = full_scores[~full_scores.index.isin(seen)]
        out += _top_k(remaining, need - len(out))
    return out

SLEEVE_COUNT = 3
HOLD_MONTHS = 4
STEP = max(HOLD_MONTHS // SLEEVE_COUNT, 1)
REFRESH_N = 2


def _momentum_evict(scores, held, prev_scores, k, refresh_n):
    s = scores.dropna()
    if s.empty:
        return []
    if not held:
        return _top_k(s, k)
    held_scored = [t for t in held if t in s.index]
    if prev_scores is not None:
        decline = {t: prev_scores.get(t, s[t]) - s[t] for t in held_scored}
        worst_first = sorted(held_scored, key=lambda t: -decline[t])
    else:
        worst_first = list(reversed(_top_k(s[s.index.isin(held_scored)], len(held_scored))))
    to_drop = set(worst_first[:refresh_n])
    keep = [t for t in held_scored if t not in to_drop]
    need = k - len(keep)
    return keep + _fill_to_k(_top_k(s, len(s)), keep, s, need)


def generate_targets(context):
    dates = list(context.rebal_dates)
    comp = context.comp
    k = context.k
    sleeve_holdings = [[] for _ in range(SLEEVE_COUNT)]
    weight_per_name = (1.0 / SLEEVE_COUNT) / k
    rows = []
    for i, d in enumerate(dates):
        for j in range(SLEEVE_COUNT):
            offset = j * STEP
            if i >= offset and (i - offset) % HOLD_MONTHS == 0:
                scores = comp.get(d)
                prev_scores = comp.get(dates[i - HOLD_MONTHS]) if i >= HOLD_MONTHS else None
                if scores is not None:
                    sleeve_holdings[j] = _momentum_evict(scores, sleeve_holdings[j], prev_scores, k, REFRESH_N)
        agg = {}
        for holdings in sleeve_holdings:
            for t in holdings:
                agg[t] = agg.get(t, 0.0) + weight_per_name
        for t, w in agg.items():
            rows.append({"date": d, "ticker": t, "weight": w})
    return pd.DataFrame(rows, columns=["date", "ticker", "weight"])
