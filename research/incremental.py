"""Does a sub-factor add predictive value its siblings don't already provide?

Two complementary, strictly point-in-time tests, both run per period and then
averaged:

* **incremental IC** — regress the sub-factor's score cross-sectionally on its
  siblings' scores and keep the residual (the part of the sub no sibling can
  explain), then IC that residual against the forward return. If a redundant sub
  is just a noisy copy of a sibling, its residual carries little signal and the
  incremental IC collapses toward zero even when its *standalone* IC looked fine.

* **delta parent IC** — the parent factor is the equal-weight mean of its subs, so
  we can rebuild it with and without the sub and difference the two parent ICs.
  A positive delta means including the sub measurably sharpened the parent's
  ranking; a delta around zero (or negative) means the parent is just as good — or
  better — without it. This is the most decision-relevant "incremental improvement
  to factor performance" the classification leans on.

``standalone_ic`` (the sub's own IC over the same periods) is reported alongside so
the *share of edge that survives* orthogonalization is visible: a Keep retains much
of its standalone IC incrementally; a Merge loses most of it to a sibling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ic import period_ic, DEFAULT_MIN_NAMES


def _residual(y: pd.Series, x: pd.DataFrame) -> pd.Series | None:
    """Cross-sectional OLS residual of ``y`` on ``x`` (intercept included)."""
    df = pd.concat([y.rename("y"), x], axis=1).dropna()
    if df.shape[0] < x.shape[1] + 2:
        return None
    yv = df["y"].to_numpy(dtype=float)
    xv = df.drop(columns=["y"]).to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(xv)), xv])
    beta, *_ = np.linalg.lstsq(design, yv, rcond=None)
    resid = yv - design @ beta
    return pd.Series(resid, index=df.index)


def incremental_metrics(
    panel,
    fwd_returns: dict[str, pd.Series],
    *,
    min_names: int = DEFAULT_MIN_NAMES,
) -> pd.DataFrame:
    """Per sub-factor incremental contribution, averaged over a horizon's periods.

    ``fwd_returns`` is the ``{start_date: fwd Series}`` map for one horizon
    (the 1-month horizon is the natural choice — most periods, least overlap).
    Columns: ``sub_factor, parent, n_periods, standalone_ic, incremental_ic,
    delta_parent_ic``.
    """
    acc: dict[str, dict[str, list[float]]] = {}
    for parent, subs in panel.sub_by_parent.items():
        for sub in subs:
            acc[sub] = {"standalone": [], "incremental": [], "delta": []}

    for d, fwd in fwd_returns.items():
        if d not in panel.scores:
            continue
        frame = panel.scores[d]
        for parent, subs in panel.sub_by_parent.items():
            present = [s for s in subs if s in frame.columns]
            if not present:
                continue
            parent_all = frame[present].mean(axis=1)
            ic_all = period_ic(parent_all, fwd, min_names=min_names)
            for sub in present:
                siblings = [s for s in present if s != sub]
                # standalone
                ic_self = period_ic(frame[sub], fwd, min_names=min_names)
                if ic_self is not None:
                    acc[sub]["standalone"].append(ic_self)
                # incremental: IC of residual vs siblings
                if siblings:
                    resid = _residual(frame[sub], frame[siblings])
                    if resid is not None:
                        ic_resid = period_ic(resid, fwd, min_names=min_names)
                        if ic_resid is not None:
                            acc[sub]["incremental"].append(ic_resid)
                    # delta parent IC: with vs without this sub
                    parent_drop = frame[siblings].mean(axis=1)
                    ic_drop = period_ic(parent_drop, fwd, min_names=min_names)
                    if ic_all is not None and ic_drop is not None:
                        acc[sub]["delta"].append(ic_all - ic_drop)
                else:
                    # Single-sub parent: residual undefined, dropping kills parent.
                    if ic_self is not None:
                        acc[sub]["incremental"].append(ic_self)

    rows: list[dict] = []
    for parent, subs in panel.sub_by_parent.items():
        for sub in subs:
            a = acc[sub]
            n = max(len(a["standalone"]), len(a["incremental"]), len(a["delta"]))
            rows.append({
                "sub_factor": sub,
                "parent": parent,
                "n_periods": int(n),
                "standalone_ic": _mean_or_nan(a["standalone"]),
                "incremental_ic": _mean_or_nan(a["incremental"]),
                "delta_parent_ic": _mean_or_nan(a["delta"]),
            })
    return pd.DataFrame(rows)


def _mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")
