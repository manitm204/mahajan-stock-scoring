"""Re-run of the dynamic-best-trailing-IC idea (user request 2026-09-11) on
the NEW top-20-by-composite-rank pool (harness.random_top20_selector),
instead of the old composite==100 tied pool the original dynamic_best_ic.py
used. Same method, no look-ahead: at each review date, pick whichever of the
8 V4 parents had the highest TRAILING 5-year IC on the top-20 pool (using
only ICs realized strictly before that date), then rank the top-20 pool by
that parent's score and take the top k. Falls back to "insider" before any
trailing history exists.

The original version of this idea (on the ==100 pool) was rejected: it
underperformed both the random baseline and the insider-tie-break champion,
because "highest trailing IC" was usually won by quality/institutional
(weak standalone tie-breakers) rather than insider (the one parent with
real marginal tie-break power) -- see session_log.md "Tested and rejected:
dynamic ... tie-break". This module checks whether that conclusion holds on
the new, less pre-screened top-20 pool.

Usage: python -m research.loop_research.promotion \
    research.loop_research.dynamic_best_ic_top20.dynamic_best_ic_top20_selector \
    --baseline research.loop_research.harness.random_top20_selector
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from research.forward_returns import compute_forward_returns   # noqa: E402
from research.ic import period_ic                              # noqa: E402
from research.loop_research import harness as H                # noqa: E402

PARENTS = ["momentum", "value", "quality", "growth", "revisions",
          "institutional", "insider", "short"]
FALLBACK_PARENT = "insider"
TRAIL_YEARS = 5
TOP_N_POOL = 20


def _ic_all_parents_by_date(bundle) -> pd.DataFrame:
    """Per-date, per-parent 1M IC on the top-20-by-composite-rank pool."""
    data = bundle["data"]
    parent_scores_by_date = bundle["parent_scores"]
    rebal = list(data.rebal_dates)
    fwd_1m = compute_forward_returns(data.matrix, rebal, {"1M": 1})["1M"]

    rows = []
    for d in rebal:
        if d not in fwd_1m:
            continue
        scores_d = data.comp.get(d)
        if scores_d is None:
            continue
        ranked = scores_d.dropna().sort_values(ascending=False, kind="stable")
        pool = list(ranked.index[:TOP_N_POOL])
        if not pool:
            continue
        pscores = parent_scores_by_date.get(d, {})
        for parent in PARENTS:
            p_score = pscores.get(parent)
            if p_score is None:
                continue
            ic = period_ic(p_score.reindex(pool), fwd_1m[d].reindex(pool), min_names=8)
            rows.append({"date": d, "parent": parent, "ic": ic})
    return pd.DataFrame(rows)


def compute_winner_by_date(ic_df: pd.DataFrame, rebal_dates: list[str]) -> dict[str, str]:
    """No look-ahead: trailing IC at date d uses only ICs from dates strictly
    before d, expanding window capped at TRAIL_YEARS."""
    ic_df = ic_df.copy()
    ic_df["date_ts"] = pd.to_datetime(ic_df["date"])
    winner_by_date = {}
    for d in rebal_dates:
        d_ts = pd.Timestamp(d)
        lo = d_ts - pd.DateOffset(years=TRAIL_YEARS)
        prior = ic_df[(ic_df["date_ts"] < d_ts) & (ic_df["date_ts"] >= lo)]
        if prior.empty:
            winner_by_date[d] = FALLBACK_PARENT
            continue
        trailing_mean = prior.groupby("parent")["ic"].mean().dropna()
        winner_by_date[d] = trailing_mean.idxmax() if not trailing_mean.empty else FALLBACK_PARENT
    return winner_by_date


def make_dynamic_selector(winner_by_date: dict[str, str]):
    def selector(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
        d = ctx["date"]
        parent = winner_by_date.get(d, FALLBACK_PARENT)
        ranked_scores = scores.dropna().sort_values(ascending=False, kind="stable")
        pool = list(ranked_scores.index[:TOP_N_POOL])
        p_score = parent_scores.get(parent, pd.Series(dtype=float)).reindex(pool)
        ranked = p_score.sort_index().sort_values(ascending=False, kind="stable")
        picks = list(ranked.index[:k])
        if len(picks) < k:
            remaining = p_score[~p_score.index.isin(picks)].sort_index().sort_values(
                ascending=False, kind="stable")
            picks += list(remaining.index[:k - len(picks)])
        return picks if picks else held
    selector.__name__ = "dynamic_best_trailing_ic_top20"
    return selector


def make_perturbed_dynamic_selector(winner_by_date: dict[str, str], noise_mult: float):
    """Perturbation-test variant (option 1, 2026-09-11): the winning-parent
    DECISION for each date is left untouched (winner_by_date is precomputed,
    not perturbed) -- only the final ranking step adds Gaussian noise
    (scaled to that date's top-20-pool cross-sectional std of the winning
    parent's score) before picking the top k, same mechanic as
    perturbation.make_perturbed_single_parent_selector. Tests whether the
    top-5-within-the-chosen-parent is a robust separation or a lucky
    near-tie, holding the rotation logic itself fixed."""
    def selector(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
        d = ctx["date"]
        parent = winner_by_date.get(d, FALLBACK_PARENT)
        ranked_scores = scores.dropna().sort_values(ascending=False, kind="stable")
        pool = list(ranked_scores.index[:TOP_N_POOL])
        p_score = parent_scores.get(parent, pd.Series(dtype=float)).reindex(scores.dropna().index)
        pool_vals = p_score.reindex(pool).dropna()
        base_std = float(pool_vals.std()) if len(pool_vals) >= 2 and pool_vals.std() == pool_vals.std() else 1.0
        base_std = base_std or 1.0

        def _noisy_rank(names):
            noise = pd.Series({t: rng.gauss(0, noise_mult * base_std) for t in names})
            vals = p_score.reindex(names).fillna(0) + noise
            return vals.sort_index().sort_values(ascending=False, kind="stable")

        ranked = _noisy_rank(pool)
        picks = list(ranked.index[:k])
        if len(picks) < k:
            remaining_pool = [t for t in scores.dropna().index if t not in picks]
            remaining_ranked = _noisy_rank(remaining_pool)
            picks += list(remaining_ranked.index[:k - len(picks)])
        return picks if picks else held
    selector.__name__ = f"dynamic_perturbed_{noise_mult}"
    return selector


def winner_frequency():
    bundle = H.get_data()
    rebal = list(bundle["data"].rebal_dates)
    ic_df = _ic_all_parents_by_date(bundle)
    winner_by_date = compute_winner_by_date(ic_df, rebal)
    return pd.Series(list(winner_by_date.values())).value_counts()


_bundle = H.get_data()
_ic_df = _ic_all_parents_by_date(_bundle)
_winner_by_date = compute_winner_by_date(_ic_df, list(_bundle["data"].rebal_dates))
dynamic_best_ic_top20_selector = make_dynamic_selector(_winner_by_date)


if __name__ == "__main__":
    print("winner (trailing-5y-IC leader) frequency across review dates, top-20 pool:")
    print(winner_frequency().to_string())
