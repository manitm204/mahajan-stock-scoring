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
    context.k            : int, SUGGESTED default book size (10) -- evaluate.py
                            does not enforce targets against this value. Book
                            size is a candidate.py design choice: declare your
                            own module-level BOOK_SIZE constant and use that
                            instead of context.k if you want a different book
                            size (the fixed correctness tests read BOOK_SIZE
                            back off this module, so they stay valid for
                            whatever size you pick).

No prices. No returns. No performance results of any kind, for any period
(dev, val, or the locked 2025+ holdout) -- so there is nothing here to
overfit a holdout to even in principle.

Must return a long-format DataFrame with columns:
    date    (str, must be one of context.rebal_dates)
    ticker  (str)
    weight  (float, target PORTFOLIO weight as of that decision date)

------------------------------------------------------------------------
Current champion (session 3 follow-up, 2026-09-06): a user-requested
REFRESH_N sweep (1-10) at the session-3 champion's HOLD_MONTHS=4 found
REFRESH_N=1 beats REFRESH_N=2 -- evicting only the SINGLE worst-declining
name per sleeve per reform (instead of 2) improves both dev_ir and val_ir.
research_score history: ... -> HOLD=4/REFRESH_N=2 (session 3, round 67)
1.060 -> HOLD=4/REFRESH_N=1 (this file) 1.198. This is the slowest possible
non-zero rotation: each sleeve trades exactly one name every 4 months.
See session_log.md for the full history across all sessions.
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

BOOK_SIZE = 10  # candidate.py now owns book size; context.k is only a
                # suggested default (see evaluate.py's Context docstring) --
                # nothing in evaluate.py enforces targets against context.k,
                # so this is free to differ from it.
SLEEVE_COUNT = 3
HOLD_MONTHS = 4
STEP = max(HOLD_MONTHS // SLEEVE_COUNT, 1)
REFRESH_N = 1


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
    k = BOOK_SIZE
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
