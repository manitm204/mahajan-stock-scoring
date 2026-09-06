"""AUTORESEARCH CANDIDATE -- session 3 round 198: book size 11 at HOLD=4/REFRESH_N=1 (champion's recipe, scaled up)"""
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
