"""10 new, deliberately varied hypotheses (user request 2026-09-11: "come up
with a new strategy ... unrelated to anything we tested ... repeat in a loop
for 10 different ideas"), tested on the top-20-by-composite-rank pool
against `harness.random_top20_selector`, through the same codified
`promotion.evaluate_promotion` 8-gate rule used for every prior candidate.
Deliberately spans mechanisms never tried before on this pool: price-based
risk (vol, beta, risk-adjusted momentum), score dynamics (momentum,
stability), pool-membership dynamics (new entrant, tenure), a pure
behavioral/naive control (nominal price level), trend-following (200dma
gap), and a categorical breadth count (parents above cross-sectional
median) -- as opposed to every prior candidate, which ranked by a single
score-based quantity (a parent, a subfactor, or an aggregate of parents).

Each idea is expressed as one `metric_fn(scores, parent_scores, pool, ctx)
-> pd.Series | None` returning a per-ticker ranking metric (None = fall back
to random, e.g. insufficient trailing history) plus an `ascending` flag.
`make_selector`/`make_perturbed_selector` turn any metric_fn into a full
selector / noise-perturbed selector for free, so every idea automatically
gets the same perturbation-robustness test as the hand-built ones.

Usage: python -m research.loop_research.exploration_batch_2026_09_11
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backtesting.data_loader import SPY                       # noqa: E402
from research.loop_research.candidates import TOP_N_POOL      # noqa: E402

VOL_LOOKBACK_DAYS = 63       # ~3 trading months
BETA_LOOKBACK_DAYS = 126     # ~6 trading months
MOM_LOOKBACK_DAYS = 126
SCORE_MOM_LOOKBACK_MONTHS = 3
SCORE_STABILITY_LOOKBACK_MONTHS = 6
TENURE_MAX_LOOKBACK_MONTHS = 24
DMA_LOOKBACK_DAYS = 200
DMA_MIN_HISTORY_DAYS = 150


def _rank_pick(metric: pd.Series, pool: list[str], k: int, ascending: bool,
               rng=None, noise_std: float | None = None) -> list[str]:
    s = metric.reindex(pool)
    if noise_std is not None:
        noise = pd.Series({t: rng.gauss(0, noise_std) for t in pool})
        s = s.fillna(0.0) + noise
    s = s.dropna()
    s = s.sort_index().sort_values(ascending=ascending, kind="stable")
    picks = list(s.index[:k])
    if len(picks) < k:
        remaining = [t for t in pool if t not in picks]
        picks += remaining[:k - len(picks)]
    return picks


def make_selector(name: str, metric_fn, ascending: bool):
    def selector(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
        ranked = scores.dropna().sort_values(ascending=False, kind="stable")
        pool = list(ranked.index[:TOP_N_POOL])
        if not pool:
            return held
        metric = metric_fn(scores, parent_scores, pool, ctx)
        if metric is None:
            rng.shuffle(pool)
            return pool[:k]
        return _rank_pick(metric, pool, k, ascending)
    selector.__name__ = name
    return selector


def make_perturbed_selector(metric_fn, ascending: bool, noise_mult: float):
    def selector(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
        ranked = scores.dropna().sort_values(ascending=False, kind="stable")
        pool = list(ranked.index[:TOP_N_POOL])
        if not pool:
            return held
        metric = metric_fn(scores, parent_scores, pool, ctx)
        if metric is None:
            rng.shuffle(pool)
            return pool[:k]
        vals = metric.reindex(pool).dropna()
        base_std = float(vals.std()) if len(vals) >= 2 and vals.std() == vals.std() else 1.0
        base_std = base_std or 1.0
        return _rank_pick(metric, pool, k, ascending, rng=rng, noise_std=noise_mult * base_std)
    return selector


# ---------------------------------------------------------------------------
# 1. Low realized volatility (defensive / low-vol anomaly)
def metric_low_vol(scores, parent_scores, pool, ctx):
    matrix = ctx["bundle"]["data"].matrix
    d = ctx["date"]
    if d not in matrix.index:
        return None
    idx = matrix.index.get_loc(d)
    start = max(idx - VOL_LOOKBACK_DAYS, 0)
    cols = [t for t in pool if t in matrix.columns]
    window = matrix.iloc[start:idx + 1][cols]
    rets = window.pct_change().dropna(how="all")
    return rets.std()


# 2. Low beta to SPY (defensive tilt, different axis than volatility)
def metric_low_beta(scores, parent_scores, pool, ctx):
    matrix = ctx["bundle"]["data"].matrix
    d = ctx["date"]
    if d not in matrix.index or SPY not in matrix.columns:
        return None
    idx = matrix.index.get_loc(d)
    start = max(idx - BETA_LOOKBACK_DAYS, 0)
    cols = [t for t in pool if t in matrix.columns]
    window = matrix.iloc[start:idx + 1][cols + [SPY]]
    rets = window.pct_change().dropna(how="all")
    mkt = rets[SPY]
    var_mkt = mkt.var()
    if not var_mkt or var_mkt != var_mkt:
        return None
    betas = {t: rets[t].cov(mkt) / var_mkt for t in cols}
    return pd.Series(betas)


# 3. Risk-adjusted momentum (return/vol, reward smooth compounding over raw momentum)
def metric_riskadj_mom(scores, parent_scores, pool, ctx):
    matrix = ctx["bundle"]["data"].matrix
    d = ctx["date"]
    if d not in matrix.index:
        return None
    idx = matrix.index.get_loc(d)
    start = max(idx - MOM_LOOKBACK_DAYS, 0)
    cols = [t for t in pool if t in matrix.columns]
    window = matrix.iloc[start:idx + 1][cols]
    total_ret = window.iloc[-1] / window.iloc[0] - 1.0
    rets = window.pct_change().dropna(how="all")
    vol = rets.std().replace(0, float("nan"))
    return total_ret / vol


# 4. Composite score momentum (re-rating: score rising, not just high)
def metric_score_momentum(scores, parent_scores, pool, ctx):
    data = ctx["bundle"]["data"]
    i, dates = ctx["i"], ctx["dates"]
    j = i - SCORE_MOM_LOOKBACK_MONTHS
    if j < 0:
        return None
    now = data.comp.get(ctx["date"])
    prior = data.comp.get(dates[j])
    if now is None or prior is None:
        return None
    return now.reindex(pool) - prior.reindex(pool)


# 5. Composite score stability (lowest score volatility -- steady strength)
def metric_score_stability(scores, parent_scores, pool, ctx):
    data = ctx["bundle"]["data"]
    i, dates = ctx["i"], ctx["dates"]
    j = max(i - SCORE_STABILITY_LOOKBACK_MONTHS + 1, 0)
    if j == i:
        return None
    frames = [data.comp[dd].reindex(pool) for dd in dates[j:i + 1] if dd in data.comp]
    if len(frames) < 3:
        return None
    return pd.concat(frames, axis=1).std(axis=1)


# 6. New entrant to the pool (freshly re-rated names)
def metric_new_entrant(scores, parent_scores, pool, ctx):
    data = ctx["bundle"]["data"]
    i, dates = ctx["i"], ctx["dates"]
    if i == 0:
        return None
    prior_scores = data.comp.get(dates[i - 1])
    if prior_scores is None:
        return None
    prior_ranked = prior_scores.dropna().sort_values(ascending=False, kind="stable")
    prior_pool = set(prior_ranked.index[:TOP_N_POOL])
    is_new = pd.Series({t: (0.0 if t in prior_pool else 1.0) for t in pool})
    score_rank = scores.reindex(pool).rank(pct=True)
    return is_new * 10.0 + score_rank


# 7. Longest tenure in the pool (persistence, opposite of #6)
def metric_tenure(scores, parent_scores, pool, ctx):
    data = ctx["bundle"]["data"]
    i, dates = ctx["i"], ctx["dates"]
    tenure = {}
    for t in pool:
        cnt = 0
        for back in range(0, TENURE_MAX_LOOKBACK_MONTHS + 1):
            j = i - back
            if j < 0:
                break
            s = data.comp.get(dates[j])
            if s is None:
                break
            ranked = s.dropna().sort_values(ascending=False, kind="stable")
            if t in set(ranked.index[:TOP_N_POOL]):
                cnt += 1
            else:
                break
        tenure[t] = cnt
    return pd.Series(tenure)


# 8. Lowest nominal share price (naive behavioral control, expected to fail)
def metric_low_price(scores, parent_scores, pool, ctx):
    matrix = ctx["bundle"]["data"].matrix
    d = ctx["date"]
    if d not in matrix.index:
        return None
    cols = [t for t in pool if t in matrix.columns]
    return matrix.loc[d, cols]


# 9. Largest positive gap above the 200-day moving average (trend-following)
def metric_200dma_gap(scores, parent_scores, pool, ctx):
    matrix = ctx["bundle"]["data"].matrix
    d = ctx["date"]
    if d not in matrix.index:
        return None
    idx = matrix.index.get_loc(d)
    start = max(idx - DMA_LOOKBACK_DAYS + 1, 0)
    if idx - start < DMA_MIN_HISTORY_DAYS:
        return None
    cols = [t for t in pool if t in matrix.columns]
    window = matrix.iloc[start:idx + 1][cols]
    ma = window.mean()
    return window.iloc[-1] / ma - 1.0


# 10. Parent breadth (count of the 8 parents above the day's cross-sectional
# median, a categorical "strong on how many dimensions" measure -- distinct
# from continuous avg/min-parent-score ideas tried earlier)
def metric_parent_breadth(scores, parent_scores, pool, ctx):
    if not parent_scores:
        return None
    df = pd.DataFrame({p: s for p, s in parent_scores.items()})
    med = df.median()
    above = (df >= med).sum(axis=1)
    return above.reindex(pool)


EXPERIMENTS = [
    ("low_realized_vol_top20",
     "defensive: lowest trailing 3-month realized volatility within the top-20 pool",
     metric_low_vol, True),
    ("low_beta_top20",
     "defensive: lowest trailing 6-month beta to SPY within the top-20 pool",
     metric_low_beta, True),
    ("risk_adjusted_momentum_top20",
     "reward smooth compounding: highest trailing 6-month return/volatility ratio",
     metric_riskadj_mom, False),
    ("score_momentum_top20",
     "re-rating: composite score risen the most over the trailing 3 months",
     metric_score_momentum, False),
    ("score_stability_top20",
     "steady strength: lowest composite-score volatility over the trailing 6 months",
     metric_score_stability, True),
    ("new_entrant_top20",
     "freshness: names newly promoted into the top-20 pool this month",
     metric_new_entrant, False),
    ("longest_tenure_top20",
     "persistence: names that have been continuously in the top-20 pool the longest",
     metric_tenure, False),
    ("low_nominal_price_top20",
     "naive behavioral control: lowest nominal share price (expected null result)",
     metric_low_price, True),
    ("above_200dma_gap_top20",
     "trend-following: largest positive gap above the 200-day moving average",
     metric_200dma_gap, False),
    ("parent_breadth_top20",
     "breadth: most of the 8 parent scores above that day's cross-sectional median",
     metric_parent_breadth, False),
]
