"""Regime utilities → regime-conditional parent weights (training data only).

For a set of training rebalances whose VIX resembles the current one, each
parent gets a utility from its performance on exactly those dates:

    utility = 0.50 · rank(mean IC) + 0.25 · rank(IC-IR) + 0.25 · rank(Q5-Q1)

with ranks scaled to [0, 1] across parents (worst = 0). Thin samples are
shrunk toward the full-training-window utility by n / (n + 12); below the
minimum sample the caller falls back to the baseline weights entirely.
Positive utilities become weights, capped at 25 % per parent and renormalised.
"""
from __future__ import annotations

import pandas as pd

MIN_REGIME_OBS = 4     # below this the tilt is skipped (baseline weights)
SHRINK_K = 12          # λ = n / (n + SHRINK_K)
PARENT_CAP = 0.25


def _stats_by_parent(train_stats: pd.DataFrame, dates: set[str]) -> pd.DataFrame:
    sub = train_stats[train_stats["date"].isin(dates)]
    g = sub.groupby("parent")
    out = pd.DataFrame({
        "mean_ic": g["ic"].mean(),
        "ic_std": g["ic"].std(ddof=1),
        "mean_spread": g["spread"].mean(),
        "n_obs": g["ic"].count(),
    })
    out["ir"] = out["mean_ic"] / out["ic_std"].where(out["ic_std"] > 1e-9)
    return out


def _rank01(s: pd.Series) -> pd.Series:
    """Percentile ranks scaled to [0, 1]; NaN ranks worst."""
    r = s.rank(method="average", na_option="top")
    n = len(r)
    return (r - 1.0) / (n - 1.0) if n > 1 else r * 0.0


def _utility(stats: pd.DataFrame) -> pd.Series:
    return (0.50 * _rank01(stats["mean_ic"])
            + 0.25 * _rank01(stats["ir"])
            + 0.25 * _rank01(stats["mean_spread"]))


def _cap_and_normalize(raw: dict[str, float], cap: float = PARENT_CAP) -> dict[str, float]:
    """Water-fill weights to the per-parent cap, then renormalise (mirrors the
    V4 capping convention)."""
    w = dict(raw)
    for _ in range(20):
        over = [p for p, x in w.items() if x > cap + 1e-12]
        if not over:
            break
        excess = sum(w[p] - cap for p in over)
        for p in over:
            w[p] = cap
        under = [p for p, x in w.items() if 0 < x < cap - 1e-12]
        pool = sum(w[p] for p in under)
        if not under or pool <= 1e-12:
            break
        for p in under:
            w[p] += excess * w[p] / pool
    tot = sum(w.values())
    return {p: x / tot for p, x in w.items() if x > 0}


def regime_weights(train_stats: pd.DataFrame, sample_dates: list[str],
                   all_train_dates: list[str],
                   ) -> tuple[dict[str, float] | None, dict]:
    """Regime-conditional parent weights, or ``None`` meaning "do not tilt".

    ``sample_dates`` are the training rebalances in the comparable-VIX regime;
    ``all_train_dates`` the full training window (shrinkage target).
    """
    n = len(sample_dates)
    meta = {"n_sample": n, "shrink_lambda": 0.0}
    if n < MIN_REGIME_OBS:
        meta["fallback"] = "thin_sample"
        return None, meta
    u_regime = _utility(_stats_by_parent(train_stats, set(sample_dates)))
    u_full = _utility(_stats_by_parent(train_stats, set(all_train_dates)))
    lam = n / (n + SHRINK_K)
    u = (lam * u_regime).add((1.0 - lam) * u_full, fill_value=0.0)
    meta["shrink_lambda"] = lam
    pos = u[u > 0]
    if pos.empty:
        meta["fallback"] = "no_positive_utility"
        return None, meta
    raw = (pos / pos.sum()).to_dict()
    return _cap_and_normalize(raw), meta
