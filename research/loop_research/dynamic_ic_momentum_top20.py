"""Follow-up to dynamic_best_ic_top20.py (user request 2026-09-11): instead
of routing to whichever parent has the highest trailing-5y IC LEVEL (that
idea was rejected -- quality wins that contest most often despite being a
weak standalone tie-breaker), route to whichever parent's IC has JUMPED the
most year-over-year: mean IC over the trailing 1 year minus mean IC over the
1 year before that, picked at every review date, so the selector isn't
locked onto one parent for long stretches. Hypothesis: a parent whose
predictive power is currently IMPROVING might be a better real-time signal
than one whose average level is currently highest (which can be a stale,
slow-moving average dominated by a parent's typical/historical edge).

No look-ahead: at review date d, "current 1y" = ICs realized in [d-1y, d),
"prior 1y" = ICs realized in [d-2y, d-1y) -- both strictly before d. Needs 2
full years of trailing history; falls back to "insider" before that.

Usage: python -m research.loop_research.promotion \
    research.loop_research.dynamic_ic_momentum_top20.dynamic_ic_momentum_top20_selector \
    --baseline research.loop_research.harness.random_top20_selector
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from research.forward_returns import compute_forward_returns                 # noqa: E402
from research.loop_research import harness as H                              # noqa: E402
from research.loop_research.dynamic_best_ic_top20 import (                   # noqa: E402
    _ic_all_parents_by_date, make_dynamic_selector, FALLBACK_PARENT, PARENTS, TOP_N_POOL,
)

WINDOW_YEARS = 1


def compute_winner_by_date_momentum(ic_df: pd.DataFrame, rebal_dates: list[str]) -> dict[str, str]:
    """Winner = parent with the largest (current 1y mean IC) - (prior 1y mean
    IC), both strictly before the review date. Needs both windows to have
    data for a parent to be eligible; falls back to FALLBACK_PARENT if no
    parent has both windows populated yet."""
    ic_df = ic_df.copy()
    ic_df["date_ts"] = pd.to_datetime(ic_df["date"])
    winner_by_date = {}
    for d in rebal_dates:
        d_ts = pd.Timestamp(d)
        cur_lo = d_ts - pd.DateOffset(years=WINDOW_YEARS)
        prior_lo = d_ts - pd.DateOffset(years=2 * WINDOW_YEARS)

        current = ic_df[(ic_df["date_ts"] < d_ts) & (ic_df["date_ts"] >= cur_lo)]
        prior = ic_df[(ic_df["date_ts"] < cur_lo) & (ic_df["date_ts"] >= prior_lo)]

        cur_mean = current.groupby("parent")["ic"].mean().dropna()
        prior_mean = prior.groupby("parent")["ic"].mean().dropna()
        common = cur_mean.index.intersection(prior_mean.index)
        if common.empty:
            winner_by_date[d] = FALLBACK_PARENT
            continue
        delta = (cur_mean.loc[common] - prior_mean.loc[common])
        winner_by_date[d] = delta.idxmax()
    return winner_by_date


def make_perturbed_winner_decision_selector(ic_df: pd.DataFrame, noise_mult: float):
    """Perturbation option 2 (2026-09-11): instead of perturbing the final
    ranking step (option 1, in dynamic_best_ic_top20.py), perturb the
    WINNER-PARENT DECISION itself -- add Gaussian noise (scaled to the
    global cross-sectional std of all per-date-per-parent ICs) to every
    individual monthly IC estimate before computing the trailing/YoY-delta
    winner, independently per sim. Tests whether the specific parent chosen
    each month matters, or a noisily-similar rotation would do about as
    well. Final ranking step (within the noisily-chosen parent) is left
    exact/noise-free, isolating this from option 1's question."""
    ic_std = float(ic_df["ic"].std())
    cache_attr = f"_perturbed_winner_cache_{id(ic_df)}_{noise_mult}"

    def selector(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
        # Cache on the rng OBJECT itself (not a module-level dict keyed by
        # id(rng)) -- id() can be recycled once a prior sim's rng is
        # garbage-collected, which silently leaked sim N's cached result
        # into sim N+1 in an earlier version of this function. An attribute
        # on the object's own lifetime has no such collision risk.
        winner_by_date = getattr(rng, cache_attr, None)
        if winner_by_date is None:
            noisy = ic_df.copy()
            noisy["ic"] = noisy["ic"] + [rng.gauss(0, noise_mult * ic_std) for _ in range(len(noisy))]
            winner_by_date = compute_winner_by_date_momentum(noisy, ctx["dates"])
            setattr(rng, cache_attr, winner_by_date)

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
    selector.__name__ = f"dynamic_winner_decision_perturbed_{noise_mult}"
    return selector


def _bootstrap_ic_all_parents_by_date(bundle, rng) -> pd.DataFrame:
    """Perturbation option 3 (2026-09-11, user-specified, more principled
    than option 2's independent-per-parent Gaussian noise): at each date,
    resample the ~20 pool stocks WITH REPLACEMENT once, and reuse the exact
    same resampled index for all 8 parents' IC calculation that date. This
    preserves the real cross-parent correlation structure (all 8 parents'
    IC estimates are computed from the SAME small, noisy 20-stock sample
    each month, so their estimation errors are correlated, not independent)
    and ties the noise size directly to the actual small-sample problem
    (resampling ~20 names) rather than an assumed noise scale.

    Uses `rng` (the harness's per-sim random.Random instance) for the
    resample draw, so each Monte Carlo sim gets an independent bootstrap
    realization of "what if we'd seen a different sample of this pool.\""""
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
        n = len(pool)
        if n < 2:
            continue
        fwd_pool = fwd_1m[d].reindex(pool).to_numpy()
        boot_idx = [rng.randrange(n) for _ in range(n)]   # SAME draw reused for all 8 parents
        boot_fwd = fwd_pool[boot_idx]

        pscores = parent_scores_by_date.get(d, {})
        for parent in PARENTS:
            p_score = pscores.get(parent)
            if p_score is None:
                continue
            p_vals = p_score.reindex(pool).to_numpy()
            boot_p = p_vals[boot_idx]
            valid = ~(np.isnan(boot_p) | np.isnan(boot_fwd))
            if valid.sum() < 8 or len(set(boot_p[valid])) < 2:
                continue
            ic = pd.Series(boot_p[valid]).corr(pd.Series(boot_fwd[valid]), method="spearman")
            if ic == ic:  # not NaN
                rows.append({"date": d, "parent": parent, "ic": float(ic)})
    return pd.DataFrame(rows)


def make_cross_sectional_bootstrap_selector(bundle):
    """Full decision-level bootstrap selector: on first call per sim (cached
    by the sim's rng identity), draws one full bootstrap realization of the
    IC table (see _bootstrap_ic_all_parents_by_date), recomputes the
    YoY-delta winner-by-date from it, then ranks/picks using the REAL
    (non-bootstrapped) parent scores -- isolating "does bootstrap
    resampling uncertainty in the winner-parent DECISION change the
    outcome," holding the final within-parent ranking exact, same framing
    as perturbation option 2."""
    cache_attr = f"_bootstrap_winner_cache_{id(bundle)}"

    def selector(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
        # See make_perturbed_winner_decision_selector's comment: cache on the
        # rng object itself, not a module-level dict keyed by id(rng), which
        # is vulnerable to id() recycling across sims.
        winner_by_date = getattr(rng, cache_attr, None)
        if winner_by_date is None:
            boot_ic_df = _bootstrap_ic_all_parents_by_date(bundle, rng)
            winner_by_date = compute_winner_by_date_momentum(boot_ic_df, ctx["dates"])
            setattr(rng, cache_attr, winner_by_date)

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
    selector.__name__ = "dynamic_winner_cross_sectional_bootstrap"
    return selector


def winner_frequency():
    bundle = H.get_data()
    rebal = list(bundle["data"].rebal_dates)
    ic_df = _ic_all_parents_by_date(bundle)
    winner_by_date = compute_winner_by_date_momentum(ic_df, rebal)
    return pd.Series(list(winner_by_date.values())).value_counts()


_bundle = H.get_data()
_ic_df = _ic_all_parents_by_date(_bundle)
_winner_by_date = compute_winner_by_date_momentum(_ic_df, list(_bundle["data"].rebal_dates))
dynamic_ic_momentum_top20_selector = make_dynamic_selector(_winner_by_date)


if __name__ == "__main__":
    print("winner (largest YoY IC jump) frequency across review dates, top-20 pool:")
    print(winner_frequency().to_string())
