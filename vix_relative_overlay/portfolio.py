"""Portfolio simulation over the pooled test rebalances — one continuous
strategy per (variant, top %, weighting mode), identical rules for every
variant: monthly rebalance, hold to the next rebalance, one-way transaction
costs charged on turnover, equal or sector-proportional weighting.

Because consecutive 6-month windows are contiguous, the pooled monthly grid is
continuous; the book carries across window boundaries (as a live strategy
would) and each realised period is attributed to the window of its *form*
date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_COST_BPS = 10.0     # one-way, charged on 0.5·L1 turnover


def select_top(score: pd.Series, top_pct: float) -> list[str]:
    s = score.dropna()
    if s.empty:
        return []
    k = max(1, int(round(len(s) * top_pct)))
    return s.sort_values(ascending=False).index[:k].tolist()


def _equal_weights(names: list[str]) -> pd.Series:
    if not names:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(names), index=names)


def _sector_neutral_weights(names: list[str], universe: list[str],
                            sectors: pd.Series) -> pd.Series:
    """Each sector weighted to its share of the scored universe, names
    equal-weighted within sector; empty sectors redistributed proportionally."""
    if not names:
        return pd.Series(dtype=float)
    uni_share = sectors.reindex(universe).fillna("Unknown").value_counts(normalize=True)
    sel_sec = sectors.reindex(names).fillna("Unknown")
    fillable = uni_share.reindex(sel_sec.unique()).fillna(0.0)
    if fillable.sum() <= 0:
        return _equal_weights(names)
    fillable = fillable / fillable.sum()
    w = pd.Series(0.0, index=names)
    for s, share in fillable.items():
        members = sel_sec.index[sel_sec == s].tolist()
        if members:
            w.loc[members] = share / len(members)
    total = w.sum()
    return w / total if total > 0 else _equal_weights(names)


def _period_return(weights: pd.Series, px_now: pd.Series,
                   px_next: pd.Series) -> float:
    if weights.empty:
        return np.nan
    ret = (px_next.reindex(weights.index) / px_now.reindex(weights.index)) - 1.0
    ok = ret.notna()
    if not ok.any():
        return np.nan
    w = weights[ok]
    w = w / w.sum()      # renormalise over names that actually priced
    return float((w * ret[ok]).sum())


def _turnover(prev: pd.Series, cur: pd.Series) -> float:
    idx = prev.index.union(cur.index)
    return float(0.5 * (cur.reindex(idx).fillna(0.0)
                        - prev.reindex(idx).fillna(0.0)).abs().sum())


def simulate(scores: dict[str, pd.Series], window_of: dict[str, str],
             matrix: pd.DataFrame, sectors: pd.Series, *,
             top_pct: float, mode: str = "equal",
             cost_bps: float = DEFAULT_COST_BPS, benchmark: str = "SPY",
             ) -> pd.DataFrame:
    """One continuous simulation over every scored rebalance.

    Returns one row per realised period: form/realize dates, window label,
    gross and net (post-cost) return, turnover, benchmark return, book size.
    """
    dates = [d for d in sorted(scores) if d in matrix.index]
    rows: list[dict] = []
    prev = pd.Series(dtype=float)
    for i, d in enumerate(dates[:-1]):
        nxt = dates[i + 1]
        px_now, px_next = matrix.loc[d], matrix.loc[nxt]
        score = scores[d]
        universe = score.dropna().index.tolist()
        longs = select_top(score, top_pct)
        if mode == "equal":
            w = _equal_weights(longs)
        elif mode == "sector_neutral":
            w = _sector_neutral_weights(longs, universe, sectors)
        else:
            raise ValueError(f"unknown weighting mode: {mode!r}")
        gross = _period_return(w, px_now, px_next)
        turn = _turnover(prev, w)
        net = gross - turn * cost_bps / 1e4 if gross == gross else np.nan
        prev = w
        spy = (float(px_next[benchmark] / px_now[benchmark] - 1.0)
               if benchmark in px_now.index and benchmark in px_next.index
               and px_now[benchmark] > 0 else np.nan)
        rows.append({"form_date": d, "realize_date": nxt,
                     "window": window_of.get(d, "?"), "gross": gross, "net": net,
                     "turnover": turn, "spy": spy, "n_names": len(longs)})
    return pd.DataFrame(rows)


def holdings_overlap(base_scores: dict[str, pd.Series],
                     var_scores: dict[str, pd.Series], window_of: dict[str, str],
                     top_pct: float) -> pd.DataFrame:
    """Per rebalance: how much the VIX overlay changed the actual book.

    ``overlap`` = |base ∩ variant| / |base|; ``n_enter``/``n_exit`` are names
    the overlay pulled in / pushed out of the top bucket.
    """
    rows: list[dict] = []
    for d in sorted(base_scores):
        if d not in var_scores:
            continue
        b = set(select_top(base_scores[d], top_pct))
        v = set(select_top(var_scores[d], top_pct))
        if not b:
            continue
        rows.append({"date": d, "window": window_of.get(d, "?"),
                     "n_base": len(b), "overlap": len(b & v) / len(b),
                     "n_enter": len(v - b), "n_exit": len(b - v)})
    return pd.DataFrame(rows)
