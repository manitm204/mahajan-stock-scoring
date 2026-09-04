"""Quintile forward-return profiles and monotonicity.

For one rebalance period a signal's names are sorted by score into five equal
buckets (Q1 = lowest scores, Q5 = highest) and the mean forward return of each
bucket is recorded. Averaging those per-bucket means across every period gives the
signal's quintile profile for a horizon. Three reads come out of it:

* **Q5-Q1 spread** — the tradable tail separation, in return units. A working
  long signal earns positive spread (top scores out-return bottom scores).
* **monotonicity** — the Spearman rank correlation between bucket order (1..5)
  and bucket mean return. +1 means returns climb cleanly with the score; values
  near 0 or negative mean the signal does not order names the way it claims to.
* **increasing steps** — how many of the four Q->Q+1 steps actually rise, a
  blunt, easy-to-read companion to the rank correlation.

Buckets are cut on the *rank* of the score (ties broken first-come) so the heavy
mass of neutral-50 scores can't collapse the cut into fewer than five bins.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_QUINTILES = 5
DEFAULT_MIN_NAMES = 25   # >= 5 names per bucket


def quintile_profile(
    scores: pd.Series,
    fwd: pd.Series,
    n_quintiles: int = N_QUINTILES,
    min_names: int = DEFAULT_MIN_NAMES,
) -> np.ndarray | None:
    """Mean forward return of each score quintile for one period.

    Returns an array of length ``n_quintiles`` (index 0 = Q1 = lowest scores) or
    ``None`` when the cross-section is too small or too degenerate to bucket.
    """
    df = pd.DataFrame({"s": scores, "f": fwd}).dropna()
    if len(df) < min_names or df["s"].nunique() < n_quintiles:
        return None
    # Rank-based cut: robust to the percentile saturation (many ties at 50/0/100).
    ranks = df["s"].rank(method="first")
    try:
        buckets = pd.qcut(ranks, n_quintiles, labels=False)
    except ValueError:
        return None
    means = df["f"].groupby(buckets).mean()
    means = means.reindex(range(n_quintiles))
    if means.isna().any():
        return None
    return means.to_numpy(dtype=float)


def _monotonicity(profile: np.ndarray) -> float:
    """Spearman rank corr between quintile order and quintile mean return."""
    order = np.arange(len(profile), dtype=float)
    if np.std(profile) < 1e-12:
        return 0.0
    return float(pd.Series(order).corr(pd.Series(profile), method="spearman"))


def _increasing_steps(profile: np.ndarray) -> int:
    return int(np.sum(np.diff(profile) > 0))


def aggregate_quintiles(
    panel,
    fwd_returns: dict[str, pd.Series],
    signals: list[str],
    *,
    min_names: int = DEFAULT_MIN_NAMES,
    n_quintiles: int = N_QUINTILES,
) -> pd.DataFrame:
    """Per-signal quintile profile averaged over every period of one horizon.

    ``fwd_returns`` is the ``{start_date: fwd Series}`` map for a single horizon.
    Returns one row per signal with q1..qN mean returns, Q5-Q1 spread, the
    monotonicity correlation, increasing-step count and the period count.
    """
    accum: dict[str, list[np.ndarray]] = {s: [] for s in signals}
    for d, fwd in fwd_returns.items():
        if d not in panel.scores:
            continue
        frame = panel.scores[d]
        for sig in signals:
            if sig not in frame.columns:
                continue
            prof = quintile_profile(
                frame[sig], fwd, n_quintiles=n_quintiles, min_names=min_names
            )
            if prof is not None:
                accum[sig].append(prof)

    rows: list[dict] = []
    for sig in signals:
        profs = accum[sig]
        if not profs:
            rows.append({"signal": sig, "n_periods": 0})
            continue
        mean_profile = np.mean(np.vstack(profs), axis=0)
        row: dict[str, float | int | str] = {"signal": sig, "n_periods": len(profs)}
        for i in range(n_quintiles):
            row[f"q{i + 1}_ret"] = float(mean_profile[i])
        row["spread_q5_q1"] = float(mean_profile[-1] - mean_profile[0])
        row["monotonicity"] = _monotonicity(mean_profile)
        row["increasing_steps"] = _increasing_steps(mean_profile)
        rows.append(row)
    return pd.DataFrame(rows)
