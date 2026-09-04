"""Sub-factor redundancy via cross-sectional score correlation.

Two sub-factors are redundant when they rank the universe the same way — i.e.
they measure the same underlying characteristic. We test that directly on the
*scores*: within each parent, for every rebalance date, compute the Spearman rank
correlation among that parent's sub-factor score columns, then average those
per-date matrices. Correlating per date and then averaging (rather than pooling
all dates into one stack) keeps a common market-wide drift from manufacturing
correlation that isn't there cross-sectionally.

The headline number classification consumes is, per sub-factor, the largest
absolute correlation to any sibling: a value near 1 means a near-duplicate exists
and the pair is a Merge candidate (only one of them can be carrying unique
information). This is distinct from IC-time-series correlation (do two subs win in
the same months) — here we ask whether they *order names* alike, which is the
question that decides redundancy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def parent_score_correlation(
    panel, parent: str, *, min_names: int = 20
) -> pd.DataFrame | None:
    """Average per-date Spearman correlation matrix among ``parent``'s subs."""
    subs = panel.sub_by_parent.get(parent, [])
    if len(subs) < 2:
        return None
    mats: list[pd.DataFrame] = []
    for d in panel.rebal_dates:
        frame = panel.signal_frame(d, subs)
        valid = frame.dropna(how="all", axis=1)
        # Need enough names and at least two non-degenerate columns to correlate.
        usable = [c for c in valid.columns if valid[c].nunique() >= 2]
        if len(usable) < 2 or valid[usable].dropna().shape[0] < min_names:
            continue
        corr = valid[usable].corr(method="spearman")
        mats.append(corr.reindex(index=subs, columns=subs))
    if not mats:
        return None
    stacked = np.stack([m.to_numpy(dtype=float) for m in mats])
    avg = np.nanmean(stacked, axis=0)
    return pd.DataFrame(avg, index=subs, columns=subs)


def max_sibling_corr(panel, *, min_names: int = 20) -> pd.DataFrame:
    """Per sub-factor: its largest |correlation| to a sibling and that sibling.

    Columns: ``sub_factor, parent, max_abs_sibling_corr, closest_sibling,
    n_siblings``. Sub-factors whose parent has no computable sibling correlation
    (single sub, or thin history) get NaN — they cannot be redundant.
    """
    rows: list[dict] = []
    for parent, subs in panel.sub_by_parent.items():
        corr = parent_score_correlation(panel, parent, min_names=min_names)
        for sub in subs:
            if corr is None or sub not in corr.columns:
                rows.append({"sub_factor": sub, "parent": parent,
                             "max_abs_sibling_corr": float("nan"),
                             "closest_sibling": None, "n_siblings": len(subs) - 1})
                continue
            others = corr[sub].drop(labels=[sub], errors="ignore").abs()
            others = others.dropna()
            if others.empty:
                rows.append({"sub_factor": sub, "parent": parent,
                             "max_abs_sibling_corr": float("nan"),
                             "closest_sibling": None, "n_siblings": len(subs) - 1})
                continue
            partner = others.idxmax()
            rows.append({"sub_factor": sub, "parent": parent,
                         "max_abs_sibling_corr": float(others.max()),
                         "closest_sibling": partner, "n_siblings": len(subs) - 1})
    return pd.DataFrame(rows)
