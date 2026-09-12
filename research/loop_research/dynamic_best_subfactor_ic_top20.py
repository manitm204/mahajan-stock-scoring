"""Finer-grained follow-up to dynamic_best_ic_top20.py (user request
2026-09-11): instead of choosing among the 8 aggregate V4 parents, go
through all 24 individual SUBFACTORS currently used to build the composite
(factors.parent_selection_v4.SELECTED_SUBS) and route to whichever single
subfactor has the highest TRAILING 5-year IC on the top-20-by-composite-rank
pool, then rank/pick that pool by the winning subfactor's own raw score.
Same no-look-ahead convention as dynamic_best_ic_top20: trailing IC at
review date d uses only IC observations realized strictly before d,
expanding window capped at 5 years, falls back to a fixed subfactor before
any trailing history exists.

The parent-level version of this idea was rejected: "highest trailing IC"
was usually won by quality/institutional (weak standalone tie-breakers),
not insider (the one parent with real marginal tie-break power) -- see
session_log.md "Tested and rejected: dynamic ... tie-break". This checks
whether going one level more granular -- individual subfactors, which can
carry a sharper (or noisier) signal than their parent aggregate -- avoids
that same failure mode.

Usage: python -m research.loop_research.promotion \
    research.loop_research.dynamic_best_subfactor_ic_top20.dynamic_best_subfactor_ic_top20_selector \
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
from research.loop_research.candidates import _all_sub_cols, TOP_N_POOL  # noqa: E402

FALLBACK_SUBFACTOR = "ins_no_selling_flag"
TRAIL_YEARS = 5


def _ic_all_subfactors_by_date(bundle) -> pd.DataFrame:
    """Per-date, per-subfactor 1M IC on the top-20-by-composite-rank pool."""
    data = bundle["data"]
    sub_frames = bundle["subfactor_frames"]
    rebal = list(data.rebal_dates)
    fwd_1m = compute_forward_returns(data.matrix, rebal, {"1M": 1})["1M"]
    cols = _all_sub_cols()

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
        frame = sub_frames.get(d)
        if frame is None:
            continue
        for sub in cols:
            if sub not in frame.columns:
                continue
            ic = period_ic(frame[sub].reindex(pool), fwd_1m[d].reindex(pool), min_names=8)
            rows.append({"date": d, "subfactor": sub, "ic": ic})
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
            winner_by_date[d] = FALLBACK_SUBFACTOR
            continue
        trailing_mean = prior.groupby("subfactor")["ic"].mean().dropna()
        winner_by_date[d] = trailing_mean.idxmax() if not trailing_mean.empty else FALLBACK_SUBFACTOR
    return winner_by_date


def make_dynamic_selector(winner_by_date: dict[str, str]):
    def selector(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
        d = ctx["date"]
        sub = winner_by_date.get(d, FALLBACK_SUBFACTOR)
        ranked_scores = scores.dropna().sort_values(ascending=False, kind="stable")
        pool = list(ranked_scores.index[:TOP_N_POOL])

        bundle = ctx.get("bundle", {})
        frame = bundle.get("subfactor_frames", {}).get(d)
        if frame is None or sub not in frame.columns:
            rng.shuffle(pool)
            return pool[:k]
        sub_score = frame[sub].reindex(pool)

        ranked = sub_score.sort_index().sort_values(ascending=False, kind="stable")
        picks = list(ranked.dropna().index[:k])
        if len(picks) < k:
            remaining = [t for t in pool if t not in picks]
            picks += remaining[:k - len(picks)]
        return picks if picks else held
    selector.__name__ = "dynamic_best_trailing_subfactor_ic_top20"
    return selector


def make_perturbed_dynamic_selector(winner_by_date: dict[str, str], noise_mult: float):
    """Perturbation test (option 1, same design as
    dynamic_best_ic_top20.make_perturbed_dynamic_selector): the winning-
    subfactor DECISION for each date is left untouched (winner_by_date is
    precomputed, not perturbed) -- only the final ranking step adds Gaussian
    noise (scaled to that date's top-20-pool cross-sectional std of the
    winning subfactor's score) before picking the top k. Tests whether the
    top-5-within-the-chosen-subfactor is a robust separation or a lucky
    near-tie, holding the rotation logic itself fixed."""
    def selector(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
        d = ctx["date"]
        sub = winner_by_date.get(d, FALLBACK_SUBFACTOR)
        ranked_scores = scores.dropna().sort_values(ascending=False, kind="stable")
        pool = list(ranked_scores.index[:TOP_N_POOL])

        bundle = ctx.get("bundle", {})
        frame = bundle.get("subfactor_frames", {}).get(d)
        if frame is None or sub not in frame.columns:
            rng.shuffle(pool)
            return pool[:k]
        sub_score = frame[sub]

        pool_vals = sub_score.reindex(pool).dropna()
        base_std = float(pool_vals.std()) if len(pool_vals) >= 2 and pool_vals.std() == pool_vals.std() else 1.0
        base_std = base_std or 1.0

        def _noisy_rank(names):
            noise = pd.Series({t: rng.gauss(0, noise_mult * base_std) for t in names})
            vals = sub_score.reindex(names).fillna(0) + noise
            return vals.sort_index().sort_values(ascending=False, kind="stable")

        ranked = _noisy_rank(pool)
        picks = list(ranked.index[:k])
        if len(picks) < k:
            remaining_pool = [t for t in scores.dropna().index if t not in picks]
            remaining_ranked = _noisy_rank(remaining_pool)
            picks += list(remaining_ranked.index[:k - len(picks)])
        return picks if picks else held
    selector.__name__ = f"dynamic_subfactor_perturbed_{noise_mult}"
    return selector


def winner_frequency():
    bundle = H.get_data()
    rebal = list(bundle["data"].rebal_dates)
    ic_df = _ic_all_subfactors_by_date(bundle)
    winner_by_date = compute_winner_by_date(ic_df, rebal)
    return pd.Series(list(winner_by_date.values())).value_counts()


_bundle = H.get_data()
_ic_df = _ic_all_subfactors_by_date(_bundle)
_winner_by_date = compute_winner_by_date(_ic_df, list(_bundle["data"].rebal_dates))
dynamic_best_subfactor_ic_top20_selector = make_dynamic_selector(_winner_by_date)


if __name__ == "__main__":
    print("winner (trailing-5y-IC leader subfactor) frequency across review dates, top-20 pool:")
    print(winner_frequency().to_string())
