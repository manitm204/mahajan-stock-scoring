"""Per sub-factor verdict: Keep / Merge / Remove / Insufficient Data.

The decision is a transparent cascade over the assembled metrics — no black box,
every verdict carries the numbers that produced it so the model stays explainable
(the whole point of the exercise is fewer, better-justified sub-factors, not a
higher count).

Order matters; the first matching rule wins:

1. **Insufficient Data** — too few point-in-time periods (or near-zero coverage)
   to judge. This is where the thin-history signals (revisions, short interest,
   insider, most 13F fields) land: their Layer 1 depth is a single recent
   snapshot, so they are *untested*, not proven worthless. Never Removed for lack
   of data.
2. **Remove** — a real history shows no edge: non-positive trailing information
   ratio *and* the parent factor is no better for including it (delta parent IC
   <= 0), or it actively drags the parent.
3. **Merge** — predictive but redundant: it ranks names almost identically to a
   sibling (high score correlation) and, once that sibling is accounted for, adds
   little unique predictive information and little to the parent. Fold it into the
   sibling.
4. **Keep** — earns its place: it adds unique predictive value (incremental IC)
   and/or sharpens the parent, and isn't a near-duplicate.

All cut-offs live in :class:`Thresholds` so the policy can be tuned without
touching the logic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

KEEP, MERGE, REMOVE, INSUFFICIENT = "Keep", "Merge", "Remove", "Insufficient Data"


@dataclass(frozen=True)
class Thresholds:
    """Tunable decision cut-offs (defaults chosen to be deliberately lenient on
    *Keep* and strict on *Remove* — we only drop a sub on real, adverse evidence)."""

    min_periods: int = 12            # below -> Insufficient Data
    min_coverage: float = 0.05       # avg non-neutral fraction below -> Insufficient
    ir_weak: float = 0.0             # IR <= this is "no standalone edge"
    delta_hurt: float = 0.003        # delta parent IC below -this -> actively harmful
    corr_high: float = 0.70          # |sibling corr| >= this -> redundant pair
    incremental_min: float = 0.005   # incremental IC >= this -> unique edge present
    delta_keep: float = 0.003        # delta parent IC >= this -> meaningfully additive


def _f(row: pd.Series, key: str, default: float = float("nan")) -> float:
    val = row.get(key, default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _verdict_for_row(row: pd.Series, t: Thresholds) -> tuple[str, str, float]:
    """Return (verdict, reason, keep_score) for one sub-factor's metrics."""
    n = _f(row, "n_periods", 0.0)
    coverage = _f(row, "coverage", float("nan"))
    ir = _f(row, "information_ratio")
    mean_ic = _f(row, "mean_ic")
    incr = _f(row, "incremental_ic")
    delta = _f(row, "delta_parent_ic")
    corr = _f(row, "max_abs_sibling_corr")
    mono = _f(row, "monotonicity")
    sibling = row.get("closest_sibling", None)

    # 1) Data gate -----------------------------------------------------------
    if not np.isnan(n) and n < t.min_periods:
        return (INSUFFICIENT,
                f"only {int(n)} usable PIT periods (< {t.min_periods}); thin Layer 1 "
                f"history — untested, not judged.", float("nan"))
    if not np.isnan(coverage) and coverage < t.min_coverage:
        return (INSUFFICIENT,
                f"near-degenerate coverage {coverage:.2f} (< {t.min_coverage}); almost "
                f"all names score neutral — untested.", float("nan"))

    # Decision inputs, treating unknown incremental/delta as 0 (conservative).
    incr_eff = 0.0 if np.isnan(incr) else incr
    delta_eff = 0.0 if np.isnan(delta) else delta
    ir_eff = 0.0 if np.isnan(ir) else ir

    keep_score = _keep_score(ir_eff, incr_eff, delta_eff, corr, mono)

    # 2) Remove --------------------------------------------------------------
    if delta_eff < -t.delta_hurt:
        return (REMOVE,
                f"including it lowers parent IC (delta {delta_eff:+.4f}); actively "
                f"drags the factor.", keep_score)
    if ir_eff <= t.ir_weak and delta_eff <= 0:
        return (REMOVE,
                f"non-positive trailing IR ({ir_eff:+.2f}, mean IC {mean_ic:+.4f}) and no "
                f"lift to the parent (delta {delta_eff:+.4f}); no demonstrated edge.",
                keep_score)

    # 3) Merge ---------------------------------------------------------------
    if (not np.isnan(corr) and corr >= t.corr_high
            and incr_eff < t.incremental_min and delta_eff < t.delta_keep):
        return (MERGE,
                f"ranks names like '{sibling}' (corr {corr:.2f}) and adds little unique "
                f"signal (incremental IC {incr_eff:+.4f}, delta {delta_eff:+.4f}); fold in.",
                keep_score)

    # 4) Keep ----------------------------------------------------------------
    reasons = []
    if incr_eff >= t.incremental_min:
        reasons.append(f"unique incremental IC {incr_eff:+.4f}")
    if delta_eff >= t.delta_keep:
        reasons.append(f"lifts parent IC (delta {delta_eff:+.4f})")
    if ir_eff > t.ir_weak:
        reasons.append(f"positive IR {ir_eff:+.2f}")
    if not np.isnan(mono) and mono >= 0.5:
        reasons.append(f"monotonic quintiles ({mono:+.2f})")
    if not reasons:
        reasons.append("not redundant and not adverse")
    return (KEEP, "; ".join(reasons) + ".", keep_score)


def _keep_score(ir: float, incr: float, delta: float, corr: float, mono: float) -> float:
    """Single comparable 'worth keeping' number for ranking the table.

    Rewards standalone edge, unique incremental edge, parent lift and monotonicity;
    penalizes redundancy. Purely for ordering/intuition — the verdict comes from the
    cascade above, not from this score.
    """
    redund_pen = 0.0 if np.isnan(corr) else max(0.0, corr - 0.5)
    mono_term = 0.0 if np.isnan(mono) else mono
    return float(
        2.0 * ir + 60.0 * incr + 60.0 * delta + 0.5 * mono_term - 1.0 * redund_pen
    )


def redundancy_watch(table: pd.DataFrame, corr_high: float = 0.70) -> pd.DataFrame:
    """Pairs of *Keep* siblings that rank names alike (|corr| >= ``corr_high``).

    These survived the per-sub cascade because each still adds a sliver of unique
    incremental signal, so neither is *Merge* on its own — but as a pair they are
    near-duplicates the operator may still choose to fold by hand. The weaker
    (lower ``keep_score``) is suggested as the one to fold into the stronger.
    Returns columns ``fold, into, corr`` (one row per pair).
    """
    if table.empty or "verdict" not in table.columns:
        return pd.DataFrame(columns=["fold", "into", "corr"])
    keep = table[table["verdict"] == KEEP].set_index("sub_factor")
    ks = keep["keep_score"].to_dict()
    seen: set[tuple[str, str]] = set()
    rows: list[dict] = []
    for sub, row in keep.iterrows():
        sib = row.get("closest_sibling")
        corr = _f(row, "max_abs_sibling_corr")
        if sib not in ks or np.isnan(corr) or corr < corr_high:
            continue
        pair = tuple(sorted([sub, str(sib)]))
        if pair in seen:
            continue
        seen.add(pair)
        weaker, stronger = (sub, sib) if ks[sub] <= ks[sib] else (sib, sub)
        rows.append({"fold": weaker, "into": stronger, "corr": corr})
    return pd.DataFrame(rows).sort_values("corr", ascending=False).reset_index(drop=True) \
        if rows else pd.DataFrame(columns=["fold", "into", "corr"])


def classify_subfactors(merged: pd.DataFrame, thresholds: Thresholds | None = None) -> pd.DataFrame:
    """Append ``verdict``, ``reason`` and ``keep_score`` columns to ``merged``.

    ``merged`` is one row per sub-factor carrying (where available): ``n_periods``,
    ``coverage``, ``mean_ic``, ``information_ratio``, ``monotonicity``,
    ``max_abs_sibling_corr``, ``closest_sibling``, ``incremental_ic``,
    ``delta_parent_ic``. Missing columns are treated as unknown and handled
    conservatively.
    """
    t = thresholds or Thresholds()
    out = merged.copy()
    verdicts, reasons, scores = [], [], []
    for _, row in out.iterrows():
        v, r, s = _verdict_for_row(row, t)
        verdicts.append(v)
        reasons.append(r)
        scores.append(s)
    out["verdict"] = verdicts
    out["reason"] = reasons
    out["keep_score"] = scores
    return out
