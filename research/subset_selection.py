"""Fresh sub-factor *subset selection* engine — pick the smallest robust set.

Successor to the per-sub Keep/Merge/Remove classifier (:mod:`research.classify`): instead
of judging each sub in isolation it searches for the best *combination* via
**forward-selection + backward-pruning** over an out-of-sample composite objective, so
redundant / regime-fragile subs are rejected by their marginal contribution. Kept honest
by (1) an **equal-weight selection composite** — no fitted weights, so no return info
leaks into *which* subs are chosen (weighting is compared afterwards on the frozen set);
(2) a **multi-horizon information-ratio objective** with a mean-IC guard — a sub that
only helps in one horizon, adds noise, or is redundant can't raise the IR (confirmed
on a chronological hold-out and year-by-year by the orchestrator). PIT throughout."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .baseline import (
    BaselineConfig, WeightSet, equal_weights, static_composite, walk_forward_composite,
)
from .ic import period_ic
from .panel import ScorePanel
from .quintiles import _monotonicity, quintile_profile

NEUTRAL = 50.0
ACTIVE_BAND = 5.0          # |score-50| > band => the name is "active" (not neutral)
DEFAULT_MIN_NAMES = 20
CORR_HIGH = 0.70


# --- 1. Inventory — coverage, missing rate, distribution, date availability 
def _pooled(panel: ScorePanel, sub: str) -> pd.DataFrame:
    """Long frame (date, ticker, score) of one sub across every rebalance date."""
    parts = []
    for d in panel.rebal_dates:
        s = panel.scores[d].get(sub)
        if s is not None:
            parts.append(pd.DataFrame({"date": d, "score": s.to_numpy()}, index=s.index))
    return pd.concat(parts) if parts else pd.DataFrame(columns=["date", "score"])


def inventory(panel: ScorePanel) -> pd.DataFrame:
    """One row per sub: source factor, coverage, missing rate, distro, date span."""
    n_uni = len(panel.universe)
    rows: list[dict] = []
    for sub in panel.all_subs:
        pooled = _pooled(panel, sub)
        vals = pooled["score"].astype(float)
        present = vals.dropna()
        n_cells = len(panel.rebal_dates) * n_uni
        active = float((present.sub(NEUTRAL).abs() > ACTIVE_BAND).mean()) if len(present) else np.nan
        # First/last rebalance where the sub actually separates names (>=2 uniques).
        active_dates = [d for d in panel.rebal_dates
                        if sub in panel.scores[d].columns
                        and panel.scores[d][sub].dropna().nunique() >= 2]
        q = present.quantile([.10, .25, .50, .75, .90]) if len(present) else pd.Series(dtype=float)
        rows.append({
            "sub_factor": sub,
            "parent": panel.parent_of(sub),
            "coverage": active,
            "missing_rate": float(1.0 - len(present) / n_cells) if n_cells else np.nan,
            "n_active_dates": len(active_dates),
            "first_active": active_dates[0] if active_dates else None,
            "last_active": active_dates[-1] if active_dates else None,
            "mean": float(present.mean()) if len(present) else np.nan,
            "median": float(present.median()) if len(present) else np.nan,
            "std": float(present.std()) if len(present) else np.nan,
            "min": float(present.min()) if len(present) else np.nan,
            "max": float(present.max()) if len(present) else np.nan,
            "p10": float(q.get(.10, np.nan)), "p25": float(q.get(.25, np.nan)),
            "p50": float(q.get(.50, np.nan)), "p75": float(q.get(.75, np.nan)),
            "p90": float(q.get(.90, np.nan)),
        })
    return pd.DataFrame(rows)


# --- 2. Baseline validation per sub-factor -----------------------
def _ic_series(frame_scores: dict[str, pd.DataFrame], fwd: dict[str, pd.Series],
               sub: str, method: str, min_names: int) -> list[float]:
    out: list[float] = []
    for d, f in fwd.items():
        frame = frame_scores.get(d)
        if frame is None or sub not in frame.columns:
            continue
        df = pd.DataFrame({"s": frame[sub], "f": f}).dropna()
        if len(df) < min_names or df["s"].nunique() < 2:
            continue
        ic = df["s"].corr(df["f"], method=method)
        if pd.notna(ic):
            out.append(float(ic))
    return out


def _stability(ics: list[float], window: int = 6) -> float:
    s = pd.Series(ics).dropna()
    if len(s) < window:
        return np.nan
    sign = np.sign(s.mean())
    if sign == 0:
        return 0.0
    roll = s.rolling(window).mean().dropna()
    return float((np.sign(roll) == sign).mean()) if len(roll) else np.nan


def _top_quintile_turnover(panel: ScorePanel, sub: str, frac: float = 0.2) -> float:
    """Avg fraction of the top-quintile membership that churns rebalance-to-rebalance."""
    dates = [d for d in panel.rebal_dates if sub in panel.scores[d].columns]
    prev: set | None = None
    churn: list[float] = []
    for d in dates:
        s = panel.scores[d][sub].dropna()
        if s.nunique() < 5:
            prev = None
            continue
        k = max(1, int(len(s) * frac))
        top = set(s.nlargest(k).index)
        if prev is not None and prev:
            churn.append(1.0 - len(top & prev) / len(prev))
        prev = top
    return float(np.mean(churn)) if churn else np.nan


def subfactor_performance(panel: ScorePanel, fwd_by_h: dict[str, dict[str, pd.Series]],
                          *, min_names: int = DEFAULT_MIN_NAMES,
                          stat_horizons: tuple[str, ...] = ("1M",)) -> pd.DataFrame:
    """Full per-sub scorecard: quintiles, IC (both), t-stat, hit, mono, stability, turnover, per-horizon IC.

    ``stat_horizons`` selects which forward-return horizon(s) the *distributional*
    stats are computed on: the pooled IC series (→ ic_spearman/ic_pearson/ic_std/
    information_ratio/ic_tstat/hit_rate/stability) and the averaged quintile profile
    (→ q1..q5/spread_q5_q1/monotonicity). Default ``("1M",)`` is byte-identical to the
    legacy single-horizon behaviour (preserves every existing caller). Pass multiple
    horizons (e.g. ``("3M","6M")``) to pool their IC series and average their quintile
    profiles, making IR / spread / hit / monotonicity horizon-consistent with a 3M/6M
    pick metric. The per-horizon ``ic_{h}`` columns below always cover every horizon in
    ``fwd_by_h`` regardless of ``stat_horizons``."""
    stat_fwds = [fwd_by_h[h] for h in stat_horizons]
    rows: list[dict] = []
    for sub in panel.all_subs:
        # Pool the IC series across the chosen stat horizon(s).
        sp: list[float] = []
        pe: list[float] = []
        for fwd_h in stat_fwds:
            sp += _ic_series(panel.scores, fwd_h, sub, "spearman", min_names)
            pe += _ic_series(panel.scores, fwd_h, sub, "pearson", min_names)
        # Quintile profile averaged across every (date × stat-horizon) period.
        profs = []
        for fwd_h in stat_fwds:
            for d, f in fwd_h.items():
                frame = panel.scores.get(d)
                if frame is None or sub not in frame.columns:
                    continue
                pr = quintile_profile(frame[sub], f, min_names=max(25, min_names))
                if pr is not None:
                    profs.append(pr)
        prof = np.mean(np.vstack(profs), axis=0) if profs else None
        sp_arr = np.array(sp, dtype=float)
        mean_ic = float(sp_arr.mean()) if len(sp_arr) else np.nan
        std_ic = float(sp_arr.std(ddof=1)) if len(sp_arr) >= 2 else np.nan
        tstat = (mean_ic / (std_ic / np.sqrt(len(sp_arr)))
                 if std_ic and std_ic > 1e-9 and len(sp_arr) else np.nan)
        row = {
            "sub_factor": sub, "parent": panel.parent_of(sub),
            "n_periods": len(sp),
            "ic_spearman": mean_ic,
            "ic_pearson": float(np.mean(pe)) if pe else np.nan,
            "ic_std": std_ic,
            "information_ratio": (mean_ic / std_ic if std_ic and std_ic > 1e-9 else np.nan),
            "ic_tstat": float(tstat) if tstat == tstat else np.nan,
            "hit_rate": float((sp_arr > 0).mean()) if len(sp_arr) else np.nan,
            "stability": _stability(sp),
            "turnover": _top_quintile_turnover(panel, sub),
        }
        row.update({f"q{i + 1}": (float(prof[i]) if prof is not None else np.nan) for i in range(5)})
        row["spread_q5_q1"] = float(prof[-1] - prof[0]) if prof is not None else np.nan
        row["monotonicity"] = _monotonicity(prof) if prof is not None else np.nan
        for h, fwd_h in fwd_by_h.items():
            ics = _ic_series(panel.scores, fwd_h, sub, "spearman", min_names)
            row[f"ic_{h}"] = float(np.mean(ics)) if ics else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# --- 3. Redundancy — global cross-sub correlation matrix ---------
def correlation_matrix(panel: ScorePanel, subs: list[str] | None = None,
                       *, min_names: int = DEFAULT_MIN_NAMES) -> pd.DataFrame:
    """Avg per-date Spearman corr among all subs (per-date then averaged, not pooled)."""
    subs = subs or panel.all_subs
    mats: list[pd.DataFrame] = []
    for d in panel.rebal_dates:
        frame = panel.signal_frame(d, subs)
        usable = [c for c in frame.columns if frame[c].nunique() >= 2]
        if len(usable) < 2 or frame[usable].dropna().shape[0] < min_names:
            continue
        mats.append(frame[usable].corr(method="spearman").reindex(index=subs, columns=subs))
    if not mats:
        return pd.DataFrame(index=subs, columns=subs, dtype=float)
    avg = np.nanmean(np.stack([m.to_numpy(dtype=float) for m in mats]), axis=0)
    return pd.DataFrame(avg, index=subs, columns=subs)


def redundant_pairs(corr: pd.DataFrame, threshold: float = CORR_HIGH) -> pd.DataFrame:
    """Every sub-factor pair with |avg correlation| >= ``threshold`` (upper tri)."""
    subs = list(corr.columns)
    rows: list[dict] = []
    for i, a in enumerate(subs):
        for b in subs[i + 1:]:
            c = corr.loc[a, b]
            if pd.notna(c) and abs(c) >= threshold:
                rows.append({"a": a, "b": b, "corr": float(c)})
    if not rows:
        return pd.DataFrame(columns=["a", "b", "corr"])
    return pd.DataFrame(rows).sort_values("corr", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def max_abs_corr(corr: pd.DataFrame) -> pd.DataFrame:
    """Per sub-factor: its largest |corr| to any other and that partner."""
    rows: list[dict] = []
    for sub in corr.columns:
        others = corr[sub].drop(labels=[sub], errors="ignore").abs().dropna()
        if others.empty:
            rows.append({"sub_factor": sub, "max_abs_corr": np.nan, "closest": None})
        else:
            rows.append({"sub_factor": sub, "max_abs_corr": float(others.max()),
                         "closest": others.idxmax()})
    return pd.DataFrame(rows)


# --- 4/5. Selection score (transparent, normalized) — ranks individual subs 
def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def selection_score(perf: pd.DataFrame, inv: pd.DataFrame, mac: pd.DataFrame,
                    threshold: float = CORR_HIGH) -> pd.DataFrame:
    """Transparent per-sub ranking (0.35*nIC + 0.30*nSpread + 0.15*hit + 0.10*mono01 + 0.10*stability - redundancy - missing). Ranking only — the set comes from the search."""
    df = perf.merge(inv[["sub_factor", "missing_rate", "coverage"]], on="sub_factor", how="left")
    df = df.merge(mac, on="sub_factor", how="left")
    n_ic = _minmax(df["ic_spearman"].fillna(df["ic_spearman"].min()))
    n_spread = _minmax(df["spread_q5_q1"].fillna(df["spread_q5_q1"].min()))
    hit = df["hit_rate"].fillna(0.0)
    mono01 = ((df["monotonicity"].fillna(0.0) + 1.0) / 2.0).clip(0, 1)
    stab = df["stability"].fillna(0.0)
    corr = df["max_abs_corr"].fillna(0.0)
    redund = ((corr - threshold) / (1.0 - threshold)).clip(lower=0.0) * 0.5
    miss = df["missing_rate"].fillna(0.0) * 0.5
    df["normalized_ic"] = n_ic
    df["normalized_spread"] = n_spread
    df["redundancy_penalty"] = redund
    df["missing_penalty"] = miss
    df["selection_score"] = (0.35 * n_ic + 0.30 * n_spread + 0.15 * hit
                             + 0.10 * mono01 + 0.10 * stab - redund - miss)
    return df.sort_values("selection_score", ascending=False).reset_index(drop=True)


# --- 6. Out-of-sample composite objective + forward/backward search 
def time_folds(dates: list[str], k: int) -> list[list[str]]:
    """Split ``dates`` (chronological) into ``k`` contiguous folds."""
    dates = sorted(dates)
    if k <= 1 or len(dates) < k:
        return [dates]
    return [list(a) for a in np.array_split(np.array(dates, dtype=object), k)]


def ew_composite(panel: ScorePanel, subs: list[str],
                 dates: list[str]) -> dict[str, pd.Series]:
    """Equal-weight composite = mean of ``subs`` scores per date (missing→neutral)."""
    out: dict[str, pd.Series] = {}
    for d in dates:
        frame = panel.scores[d]
        present = [s for s in subs if s in frame.columns]
        if not present:
            continue
        acc = pd.Series(0.0, index=panel.universe)
        for s in present:
            acc += frame[s].reindex(panel.universe).fillna(NEUTRAL)
        out[d] = acc / len(present)
    return out


def _composite_period_ics(comp: dict[str, pd.Series], fwd: dict[str, pd.Series],
                          dates: list[str], min_names: int) -> list[float]:
    ics = []
    for d in dates:
        if d in comp and d in fwd:
            ic = period_ic(comp[d], fwd[d], min_names=min_names)
            if ic is not None:
                ics.append(ic)
    return ics


DEFAULT_HORIZONS = ("1M", "3M", "6M", "12M")


def _horizon_ir(comp, fwd_h, dates, min_names) -> tuple[float | None, float]:
    """(IR = mean(period ICs)/std, mean IC) of a composite over ``dates`` for one horizon."""
    ics = np.array(_composite_period_ics(comp, fwd_h, dates, min_names), dtype=float)
    if len(ics) < 2:
        return None, (float(ics.mean()) if len(ics) else np.nan)
    sd = ics.std(ddof=1)
    return (float(ics.mean() / sd) if sd > 1e-9 else None), float(ics.mean())


def evaluate_subset(panel: ScorePanel, subs: list[str],
                    fwd_by_h: dict[str, dict[str, pd.Series]], folds: list[list[str]], *,
                    horizons: tuple[str, ...] = DEFAULT_HORIZONS,
                    min_names: int = DEFAULT_MIN_NAMES) -> dict:
    """OOS read of an EW composite. ``oos_ic`` = multi-horizon composite IR (rewards breadth); ``oos_meanic`` = multi-horizon mean IC (noise guard)."""
    empty = {"oos_ic": float("-inf"), "oos_meanic": float("-inf"), "oos_ic_1M": np.nan,
             "fold_stability": 0.0, "spread": np.nan, "hit_rate": np.nan,
             "monotonicity": np.nan, "n_subs": len(subs)}
    if not subs:
        return empty
    dates = [d for f in folds for d in f]
    comp = ew_composite(panel, subs, dates)
    irs, mics, ic1m = [], [], np.nan
    for h in horizons:
        ir, mic = _horizon_ir(comp, fwd_by_h.get(h, {}), dates, min_names)
        ic1m = mic if h == "1M" else ic1m
        if mic == mic:
            mics.append(mic)
        if ir is not None:
            irs.append(ir)
    if not irs:
        return empty
    fwd_1m = fwd_by_h.get("1M", {})
    fm1m = [float(np.mean(ic)) for f in folds if (ic := _composite_period_ics(comp, fwd_1m, f, min_names))]
    sign = np.sign(ic1m) or 1
    dd = [d for d in comp if d in fwd_1m]
    profs = [p for d in dd if (p := quintile_profile(comp[d], fwd_1m[d], min_names=max(25, min_names))) is not None]
    all_ics = [i for d in dd if (i := period_ic(comp[d], fwd_1m[d], min_names=min_names)) is not None]
    prof = np.mean(np.vstack(profs), axis=0) if profs else None
    return {"oos_ic": float(np.mean(irs)),
            "oos_meanic": float(np.mean(mics)) if mics else np.nan,
            "oos_ic_1M": float(ic1m) if ic1m == ic1m else np.nan,
            "fold_stability": float(np.mean([np.sign(m) == sign for m in fm1m])) if fm1m else np.nan,
            "spread": float(prof[-1] - prof[0]) if prof is not None else np.nan,
            "hit_rate": float(np.mean(np.array(all_ics) > 0)) if all_ics else np.nan,
            "monotonicity": _monotonicity(prof) if prof is not None else np.nan,
            "n_subs": len(subs)}


def forward_select(panel: ScorePanel, candidates: list[str],
                   fwd_by_h: dict[str, dict[str, pd.Series]], folds: list[list[str]], *,
                   min_gain: float = 0.01, min_candidate_ic: float = 0.01,
                   max_subs: int | None = None,
                   horizons: tuple[str, ...] = DEFAULT_HORIZONS,
                   min_names: int = DEFAULT_MIN_NAMES) -> tuple[list[str], pd.DataFrame]:
    """Greedy add by multi-horizon OOS IR gain (>= ``min_gain``), from candidates first
    screened to a genuine positive standalone mean IC (>= ``min_candidate_ic``) so
    noise/sign-wrong subs — which can lift IR via spurious variance reduction — are out."""
    selected: list[str] = []
    remaining = [s for s in candidates
                 if evaluate_subset(panel, [s], fwd_by_h, folds, horizons=horizons,
                                    min_names=min_names)["oos_meanic"] >= min_candidate_ic]
    steps: list[dict] = []
    cur = evaluate_subset(panel, selected, fwd_by_h, folds, horizons=horizons,
                          min_names=min_names)
    cur_ic = cur["oos_ic"] if np.isfinite(cur["oos_ic"]) else 0.0
    max_subs = max_subs or len(candidates)
    while remaining and len(selected) < max_subs:
        best_sub, best_eval, best_ic = None, None, cur_ic
        for sub in remaining:
            ev = evaluate_subset(panel, selected + [sub], fwd_by_h, folds,
                                 horizons=horizons, min_names=min_names)
            if np.isfinite(ev["oos_ic"]) and ev["oos_ic"] > best_ic:
                best_sub, best_eval, best_ic = sub, ev, ev["oos_ic"]
        if best_sub is None or (best_ic - cur_ic) < min_gain:
            break
        selected.append(best_sub)
        remaining.remove(best_sub)
        steps.append({"step": len(selected), "action": "add", "sub_factor": best_sub,
                      "parent": panel.parent_of(best_sub), "oos_ic": best_ic,
                      "oos_ic_1M": best_eval["oos_ic_1M"], "delta_ic": best_ic - cur_ic,
                      "spread": best_eval["spread"], "hit_rate": best_eval["hit_rate"],
                      "monotonicity": best_eval["monotonicity"],
                      "fold_stability": best_eval["fold_stability"],
                      "n_selected": len(selected)})
        cur_ic = best_ic
    return selected, pd.DataFrame(steps)


def backward_prune(panel: ScorePanel, selected: list[str],
                   fwd_by_h: dict[str, dict[str, pd.Series]], folds: list[list[str]], *,
                   min_gain: float = 0.01, horizons: tuple[str, ...] = DEFAULT_HORIZONS,
                   min_names: int = DEFAULT_MIN_NAMES) -> tuple[list[str], pd.DataFrame]:
    """Drop any sub whose removal costs <= ``min_gain`` OOS IR (prefer the smaller model)."""
    kept = list(selected)
    steps: list[dict] = []
    cur = evaluate_subset(panel, kept, fwd_by_h, folds, horizons=horizons,
                          min_names=min_names)["oos_ic"]
    changed = True
    while changed and len(kept) > 1:
        changed = False
        best_drop, best_ic = None, None
        for sub in kept:
            ev = evaluate_subset(panel, [s for s in kept if s != sub], fwd_by_h, folds,
                                 horizons=horizons, min_names=min_names)
            # Removal is acceptable if it does not hurt beyond min_gain.
            if ev["oos_ic"] >= cur - min_gain and (best_ic is None or ev["oos_ic"] > best_ic):
                best_drop, best_ic = sub, ev["oos_ic"]
        if best_drop is not None:
            kept.remove(best_drop)
            steps.append({"action": "drop", "sub_factor": best_drop, "oos_ic": best_ic,
                          "delta_ic": best_ic - cur, "n_selected": len(kept)})
            cur = best_ic
            changed = True
    return kept, pd.DataFrame(steps)


# --- 6b. Weighting-method comparison on the frozen selected set --
def _restrict(panel: ScorePanel, selected: list[str]) -> ScorePanel:
    """A panel view whose taxonomy is only the selected subs (parents pruned)."""
    sub_by_parent: dict[str, list[str]] = {}
    for sub in selected:
        sub_by_parent.setdefault(panel.parent_of(sub), []).append(sub)
    parents = [p for p in panel.parent_keys if p in sub_by_parent]
    return replace(panel, parent_keys=parents, sub_by_parent=sub_by_parent)


def _flat_ew_weightset(sub_by_parent: dict[str, list[str]]) -> WeightSet:
    """Parent weights ∝ #selected subs → composite is a flat mean of all subs."""
    total = sum(len(v) for v in sub_by_parent.values()) or 1
    parent = {p: len(v) / total for p, v in sub_by_parent.items()}
    sub = {p: equal_weights(v) for p, v in sub_by_parent.items()}
    return WeightSet(parent=parent, sub=sub)


def _ew_parent_ew_sub(sub_by_parent: dict[str, list[str]]) -> WeightSet:
    parent = equal_weights(list(sub_by_parent))
    sub = {p: equal_weights(v) for p, v in sub_by_parent.items()}
    return WeightSet(parent=parent, sub=sub)


def weighting_composites(panel: ScorePanel, selected: list[str],
                         fwd_1m: dict[str, pd.Series], *,
                         shrink: float = 0.5) -> dict[str, dict[str, pd.Series]]:
    """Composite series for the frozen set under 4 neutral weightings: flat EW, IC-weighted (WF), shrunk-IC, EW-parent/EW-sub."""
    rp = _restrict(panel, selected)
    out: dict[str, dict[str, pd.Series]] = {}
    out["equal_flat"] = static_composite(rp, _flat_ew_weightset(rp.sub_by_parent))
    out["equal_parent_sub"] = static_composite(rp, _ew_parent_ew_sub(rp.sub_by_parent))
    ic_cfg = BaselineConfig(shrink=0.0)
    out["ic_weighted"], _ = walk_forward_composite(rp, fwd_1m, ic_cfg)
    shr_cfg = BaselineConfig(shrink=shrink)
    out["shrunk_ic"], _ = walk_forward_composite(rp, fwd_1m, shr_cfg)
    return out


# --- 7. Validation windows (full / recent / year / regime) + momentum check 
def window_ic(comp: dict[str, pd.Series], fwd_1m: dict[str, pd.Series],
              dates: list[str], *, min_names: int = DEFAULT_MIN_NAMES) -> dict:
    """Pooled IC / spread / hit / n over an arbitrary date subset."""
    ics = _composite_period_ics(comp, fwd_1m, dates, min_names)
    profs = []
    for d in dates:
        if d in comp and d in fwd_1m:
            pr = quintile_profile(comp[d], fwd_1m[d], min_names=max(25, min_names))
            if pr is not None:
                profs.append(pr)
    prof = np.mean(np.vstack(profs), axis=0) if profs else None
    arr = np.array(ics, dtype=float)
    return {"ic": float(arr.mean()) if len(arr) else np.nan,
            "ic_ir": float(arr.mean() / arr.std(ddof=1)) if len(arr) >= 2 and arr.std(ddof=1) > 1e-9 else np.nan,
            "hit_rate": float((arr > 0).mean()) if len(arr) else np.nan,
            "spread": float(prof[-1] - prof[0]) if prof is not None else np.nan,
            "n_periods": len(arr)}


def named_windows(dates: list[str], regimes: dict[str, str] | None,
                  recent_n: int = 18) -> dict[str, list[str]]:
    """full / recent / per-year / per-regime date subsets for a validation table."""
    dates = sorted(dates)
    windows: dict[str, list[str]] = {"full": list(dates),
                                     "recent": dates[-recent_n:]}
    years = sorted({d[:4] for d in dates})
    for y in years:
        windows[f"year_{y}"] = [d for d in dates if d.startswith(y)]
    if regimes:
        for label in sorted(set(regimes.values())):
            sub = [d for d in dates if regimes.get(d) == label]
            if sub:
                windows[f"regime_{label}"] = sub
    return windows


def momentum_dependence(panel: ScorePanel, selected: list[str],
                        comp: dict[str, pd.Series], fwd_1m: dict[str, pd.Series],
                        *, min_names: int = DEFAULT_MIN_NAMES) -> dict:
    """Momentum leanness: share, composite↔momentum corr, IC lost when momentum dropped."""
    mom = [s for s in selected if panel.parent_of(s) == "momentum"]
    non_mom = [s for s in selected if panel.parent_of(s) != "momentum"]
    dates = list(comp)
    full_ic = np.mean(_composite_period_ics(comp, fwd_1m, dates, min_names) or [np.nan])
    non_ic = np.nan
    if non_mom:
        non_comp = ew_composite(panel, non_mom, dates)
        non_ic = np.mean(_composite_period_ics(non_comp, fwd_1m, dates, min_names) or [np.nan])
    # Cross-sectional correlation of full composite to a momentum-only composite.
    corrs = []
    if mom:
        mom_comp = ew_composite(panel, mom, dates)
        for d in dates:
            if d in comp and d in mom_comp:
                df = pd.DataFrame({"a": comp[d], "b": mom_comp[d]}).dropna()
                if len(df) >= min_names and df["a"].nunique() > 1 and df["b"].nunique() > 1:
                    corrs.append(df["a"].corr(df["b"], method="spearman"))
    return {"n_selected": len(selected), "n_momentum": len(mom),
            "momentum_share": len(mom) / len(selected) if selected else np.nan,
            "corr_composite_to_momentum": float(np.mean(corrs)) if corrs else np.nan,
            "ic_full": float(full_ic), "ic_without_momentum": float(non_ic),
            "ic_lost_dropping_momentum": float(full_ic - non_ic) if non_ic == non_ic else np.nan}
