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
Current champion (session 4, round 198 of 2026-09-06): after BOOK_SIZE
became a candidate.py-owned constant, a 50-round book-size sweep found
BOOK_SIZE=11 beats the prior BOOK_SIZE=10 champion -- one extra name at
the same HOLD_MONTHS=4/REFRESH_N=1 recipe improved both dev_ir and val_ir.
research_score history: ... -> HOLD=4/REFRESH_N=1 at BOOK_SIZE=10 (session
3 follow-up) 1.198 -> BOOK_SIZE=11 (this file) 1.283.

Book size beyond ~11 was tested extensively (up to 30) and consistently
HURT research_score, monotonically -- more diversified books diluted this
composite's signal rather than reducing risk, at every HOLD_MONTHS/
REFRESH_N combination tried. See session_log.md for the full round-by-round
table across all four sessions, including an open robustness caveat: like
the BOOK_SIZE=10 champion before it, this is a fairly narrow peak (its
REFRESH_N and BOOK_SIZE neighbors both fall off quickly), not yet resolved
in favor of one of the flatter, more robust alternatives identified in the
neighbor-robustness analysis.
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

BOOK_SIZE = 11
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
