"""Information Coefficient: how well a score ranks future returns.

For each period a signal's cross-sectional scores are rank-correlated (Spearman)
with the forward returns realized over that period — that period's IC. Collapsing
the per-period series gives the headline reads:

* **mean IC** — average rank correlation; the cleanest single "does a higher
  score mean a higher return" number.
* **IR** (information ratio) — mean IC / std IC; reward per unit of noise. With
  overlapping multi-month windows the std is understated, so IR is comparative,
  not an independent-sample t-stat (see :mod:`research.forward_returns`).
* **hit rate** — fraction of periods with positive IC; a high mean IC driven by
  one lucky month is exposed by a mediocre hit rate.
* **stability** — fraction of trailing rolling windows whose mean IC keeps the
  full-sample sign. A signal can have a fine average yet flip sign every other
  quarter; stability says whether the edge actually persisted.

Pairs with :mod:`research.quintiles`: IC measures the whole ranking, quintile
spread measures only the tails we trade. They should agree for a real signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_MIN_NAMES = 20
ROLLING_WINDOW = 6   # periods, for the stability read


def period_ic(
    scores: pd.Series, fwd: pd.Series, min_names: int = DEFAULT_MIN_NAMES
) -> float | None:
    """Cross-sectional Spearman IC of ``scores`` vs ``fwd`` for one period."""
    df = pd.DataFrame({"s": scores, "f": fwd}).dropna()
    if len(df) < min_names or df["s"].nunique() < 2:
        return None
    ic = df["s"].corr(df["f"], method="spearman")
    return None if pd.isna(ic) else float(ic)


def ic_timeseries(
    panel,
    fwd_returns: dict[str, pd.Series],
    signals: list[str],
    *,
    horizon: str,
    min_names: int = DEFAULT_MIN_NAMES,
) -> pd.DataFrame:
    """Long frame of per-period ICs: columns ``date, signal, horizon, ic``."""
    rows: list[dict] = []
    for d, fwd in fwd_returns.items():
        if d not in panel.scores:
            continue
        frame = panel.scores[d]
        for sig in signals:
            if sig not in frame.columns:
                continue
            ic = period_ic(frame[sig], fwd, min_names=min_names)
            if ic is None:
                continue
            rows.append({"date": d, "signal": sig, "horizon": horizon, "ic": ic})
    return pd.DataFrame(rows)


def _stability(ics: pd.Series, window: int = ROLLING_WINDOW) -> float:
    """Share of rolling-``window`` mean ICs that keep the full-sample sign."""
    finite = ics.dropna()
    if len(finite) < window:
        return float("nan")
    overall_sign = np.sign(finite.mean())
    if overall_sign == 0:
        return 0.0
    roll = finite.rolling(window).mean().dropna()
    if roll.empty:
        return float("nan")
    return float((np.sign(roll) == overall_sign).mean())


def summarize_ic(ic_ts: pd.DataFrame, signals: list[str] | None = None) -> pd.DataFrame:
    """Collapse a per-period IC frame to one summary row per signal.

    ``ic_ts`` is sorted by date inside so the rolling stability read is correct.
    Signals with no usable periods still get a row (n_periods 0) when listed in
    ``signals`` so the report shows the full taxonomy.
    """
    rows: list[dict] = []
    present = list(ic_ts["signal"].unique()) if not ic_ts.empty else []
    keys = signals if signals is not None else present
    for sig in keys:
        sub = ic_ts[ic_ts["signal"] == sig].sort_values("date") if not ic_ts.empty else ic_ts
        ics = sub["ic"].astype(float) if not sub.empty else pd.Series(dtype=float)
        if ics.empty:
            rows.append({"signal": sig, "n_periods": 0, "mean_ic": float("nan"),
                         "std_ic": float("nan"), "information_ratio": float("nan"),
                         "hit_rate": float("nan"), "median_ic": float("nan"),
                         "stability": float("nan")})
            continue
        std = float(ics.std(ddof=1)) if len(ics) >= 2 else float("nan")
        mean = float(ics.mean())
        ir = (mean / std) if std and not np.isnan(std) and std > 1e-9 else float("nan")
        rows.append({
            "signal": sig,
            "n_periods": int(len(ics)),
            "mean_ic": mean,
            "std_ic": std,
            "information_ratio": ir,
            "hit_rate": float((ics > 0).mean()),
            "median_ic": float(ics.median()),
            "stability": _stability(ics),
        })
    return pd.DataFrame(rows)
