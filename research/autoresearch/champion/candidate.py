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
Current champion (2026-09-05/06 autoresearch loop, 49 candidates tried
across two sessions, research_score = min(dev_ir, val_ir) vs SPY, strict
promotion rule):

  - top 10 names by composite score
  - THREE overlapping sleeves, each held 6 months (round 3)
  - at each reform, replace the 3 held names with the LARGEST score DECLINE
    since the sleeve's own last reform (6 months ago) -- not just the
    absolute worst-ranked names this month (round 36). A name whose score
    has risen since last review is never evicted purely for being the
    lowest-ranked of the 10; eviction targets genuine deterioration.
  - equal weighting within each sleeve (1/3 of NAV / 10 names once ramped)
  - costs and T+1 execution are handled entirely by evaluate.py

research_score history: baseline 0.253 -> HOLD_MONTHS=6 (round 3) 0.302 ->
worst-rank partial rotation (round 9) 0.362 -> momentum-eviction (round 36,
this file) 0.585. A no-immediate-recycle cooldown (round 30) reached 0.455
on its own but was superseded by momentum-eviction, and does NOT stack with
it -- momentum-eviction + no-recycle combined (tested explicitly) scores
0.566, slightly worse than momentum-eviction alone. See session_log.md for
the full round-by-round table across both sessions.
"""
from __future__ import annotations

import pandas as pd

from research.strategies.engine import _top_k

SLEEVE_COUNT = 3
HOLD_MONTHS = 6
# one new sleeve starts every `STEP` signal dates; with 3 sleeves held 6
# months each, STEP=2 means a sleeve reforms every other month
STEP = max(HOLD_MONTHS // SLEEVE_COUNT, 1)

REFRESH_N = 3


def _momentum_eviction(scores: pd.Series, held: list[str], prev_scores: pd.Series | None,
                       k: int, refresh_n: int) -> list[str]:
    """Evict the `refresh_n` held names with the biggest score DECLINE since
    the sleeve's own last reform (HOLD_MONTHS ago), not simply the
    worst-ranked names this month. A name that is still the "worst of the
    10" but has actually been improving is left alone; eviction targets
    real deterioration in the underlying signal. Falls back to plain
    worst-rank rotation on a sleeve's first-ever formation or if no prior
    score is available. Always backfills to exactly k names."""
    s = scores.dropna()
    if s.empty:
        return []
    if not held:
        return _top_k(s, k)
    held_scored = [t for t in held if t in s.index]
    if prev_scores is not None:
        decline = {t: prev_scores.get(t, s[t]) - s[t] for t in held_scored}
        ranked_worst_decline_first = sorted(held_scored, key=lambda t: -decline[t])
    else:
        ranked_worst_decline_first = list(reversed(_top_k(s[s.index.isin(held_scored)], len(held_scored))))
    to_drop = set(ranked_worst_decline_first[:refresh_n])
    keep = [t for t in held_scored if t not in to_drop]
    need = k - len(keep)
    fresh_pool = s[~s.index.isin(keep)]
    return keep + _top_k(fresh_pool, need)


def generate_targets(context) -> pd.DataFrame:
    dates = list(context.rebal_dates)
    comp = context.comp
    k = context.k

    # sleeve_holdings[j] = list of tickers currently held by sleeve j
    sleeve_holdings: list[list[str]] = [[] for _ in range(SLEEVE_COUNT)]
    weight_per_name = (1.0 / SLEEVE_COUNT) / k

    rows = []
    for i, d in enumerate(dates):
        for j in range(SLEEVE_COUNT):
            offset = j * STEP
            if i >= offset and (i - offset) % HOLD_MONTHS == 0:
                scores = comp.get(d)
                prev_scores = comp.get(dates[i - HOLD_MONTHS]) if i >= HOLD_MONTHS else None
                if scores is not None:
                    sleeve_holdings[j] = _momentum_eviction(scores, sleeve_holdings[j], prev_scores, k, REFRESH_N)

        agg: dict[str, float] = {}
        for holdings in sleeve_holdings:
            for t in holdings:
                agg[t] = agg.get(t, 0.0) + weight_per_name
        for t, w in agg.items():
            rows.append({"date": d, "ticker": t, "weight": w})

    return pd.DataFrame(rows, columns=["date", "ticker", "weight"])
