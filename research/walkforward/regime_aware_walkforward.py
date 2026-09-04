"""The continuous monthly walk-forward: ties regime_probability, regime_aware_
evidence, regime_aware_scoring, regime_aware_selection, and regime_aware_parents
together into variants A/B/C/D(/+floor) and scores each on the same 19-window
semiannual rolling-5Y grid every other regime study in this repo uses. See
docs/superpowers/specs/2026-07-11-hierarchical-regime-aware-factor-model-design.md
Section 4.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

import numpy as np

from research import compute_forward_returns
from research.panel import ScorePanel
from research.parent_selection import SIGN_INV_IC
from research.subset_selection import DEFAULT_MIN_NAMES
from research.walkforward import analysis
from research.walkforward import portfolio as pf
from research.walkforward.compose import FrozenConfig, build_parent_panel, frozen_composite
from research.walkforward.regime_aware_evidence import as_of, build_monthly_cache
from research.walkforward.regime_aware_parents import (
    HysteresisState, adaptive_weight, apply_change_caps, apply_quarterly_membership,
    blend_parent_weights, parent_utility_table, update_streaks,
)
from research.walkforward.regime_aware_scoring import score_table
from research.walkforward.regime_aware_selection import (
    parent_subfactor_weights, select_parent_subfactors,
)
from research.walkforward.regime_probability import (
    REGIME_HIGH, REGIME_LOW, REGIME_ORDER, SHRINKAGE_K, regime_probabilities,
    shrink_regime_ic,
)
from research.walkforward.selection import select_config, slice_panel
from research.walkforward.splits import (
    DATA_START, SELECTION_HORIZON_MONTHS, WalkForwardSplit, semiannual_policy_splits,
)
from research.walkforward.vix_regime_study import _vix_spot


@dataclass(frozen=True)
class VariantSpec:
    name: str
    # "full" | "long_run" | "recent" | "regime_only" | "regime_recent" | "regime_longrun"
    expected_ic_mode: str
    hysteresis: bool
    weight_caps: bool
    diversification_floor: float | None = None
    # Window length (months) backing "recent"/"regime_recent" -- as_of()'s
    # ``recent_months`` param controls how far back its "recent_24m_ic" column
    # looks; the column NAME stays fixed regardless of the value passed (it's
    # hardcoded in regime_aware_evidence._BASE_COLUMNS), so a variant using a 5Y
    # or 12M window still reads that same column -- it just holds a different
    # window's average because as_of() was called with a different recent_months.
    recent_months: int = 24
    # Fraction of a parent's final weight from the fixed trailing-5Y "base"
    # construction (spec Section 5's "70%"); the remainder comes from the
    # variant-specific adaptive tilt. Default matches the spec; a lower value
    # lets the adaptive/evidence-driven component matter more.
    base_weight_frac: float = 0.70
    # If True, "regime_only"/"regime_recent"/"regime_longrun" classify VIX Low/
    # Medium/High by PIT expanding-window PERCENTILE (bottom 20% / middle 60% /
    # top 20% of VIX levels seen so far) instead of the fixed absolute levels
    # (~15/25) baked into regime_probability.py's defaults. No effect on "full"
    # (variant C keeps the unmodified spec formula) or "long_run"/"recent".
    vix_percentile: bool = False


VARIANTS: dict[str, VariantSpec] = {
    "B": VariantSpec("B", "long_run", hysteresis=True, weight_caps=True,
                     base_weight_frac=0.50),
    "C": VariantSpec("C", "full", hysteresis=True, weight_caps=True,
                     base_weight_frac=0.50),
    "D": VariantSpec("D", "recent", hysteresis=False, weight_caps=False),
    "C+2%floor": VariantSpec("C+2%floor", "full", hysteresis=True, weight_caps=True,
                             diversification_floor=0.02),
    "C+3%floor": VariantSpec("C+3%floor", "full", hysteresis=True, weight_caps=True,
                             diversification_floor=0.03),
    # "24M" mirrors D's expected-IC mode (recent-24-month evidence only, no
    # long-run anchor or VIX tilt) but -- unlike D -- keeps hysteresis and weight
    # caps ON, so it's directly comparable to B/C on equal discipline: the only
    # thing varying across B/C/24M is which evidence blend drives expected_ic.
    "24M": VariantSpec("24M", "recent", hysteresis=True, weight_caps=True),
    "B+2%floor": VariantSpec("B+2%floor", "long_run", hysteresis=True, weight_caps=True,
                             diversification_floor=0.02),
    "24M+2%floor": VariantSpec("24M+2%floor", "recent", hysteresis=True, weight_caps=True,
                               diversification_floor=0.02),
    # Single-ingredient ablation battery (2026-07-13): each isolates exactly one
    # evidence source for expected_ic, all sharing the same hysteresis/cap
    # discipline and no floor, so differences are attributable to the evidence
    # choice alone. "5Y"/"12M" reuse "recent" mode with a non-default window
    # (see VariantSpec.recent_months); "VIXOnly" is a 100% pure VIX-regime-shrunk
    # signal (no long-run or recency at all); "VIX+12M"/"VIX+LongRun" pair VIX
    # with one other ingredient at 50/50, deliberately excluding the third.
    # base_weight_frac=0.50 (2026-07-13 rerun): drops the fixed-base share from
    # the spec default 70% to 50%, giving the variant-specific adaptive tilt
    # more influence on the final parent weight -- requested to make the
    # differences between evidence recipes more visible than they were at 70/30.
    "5Y": VariantSpec("5Y", "recent", hysteresis=True, weight_caps=True, recent_months=60,
                      base_weight_frac=0.50),
    "12M": VariantSpec("12M", "recent", hysteresis=True, weight_caps=True, recent_months=12,
                       base_weight_frac=0.50),
    # vix_percentile=True (2026-07-13 rerun): Low/Medium/High now means bottom
    # 20% / middle 60% / top 20% of the PIT VIX distribution seen so far,
    # instead of the fixed ~15/25 absolute levels. "C" deliberately keeps the
    # fixed thresholds -- it stays the unmodified spec-formula reference.
    "VIXOnly": VariantSpec("VIXOnly", "regime_only", hysteresis=True, weight_caps=True,
                           base_weight_frac=0.50, vix_percentile=True),
    "VIX+12M": VariantSpec("VIX+12M", "regime_recent", hysteresis=True, weight_caps=True,
                           recent_months=12, base_weight_frac=0.50, vix_percentile=True),
    "VIX+LongRun": VariantSpec("VIX+LongRun", "regime_longrun", hysteresis=True,
                               weight_caps=True, base_weight_frac=0.50, vix_percentile=True),
}


def _is_quarter_boundary(cutoff: str) -> bool:
    return pd.Timestamp(cutoff).month in (1, 4, 7, 10)


def _percentile_vix_thresholds(
    vix: pd.Series, cutoff: str, *, low_pct: float = 0.20, high_pct: float = 0.80,
    panel_inception: str = DATA_START,
) -> tuple[float, float]:
    """PIT-safe 20th/80th-percentile VIX levels from the full daily VIX history
    between ``panel_inception`` and ``cutoff`` (expanding window, no look-ahead)
    -- a distribution-relative alternative to the fixed absolute Low/High
    cutoffs (~15/25): bottom 20% of VIX days seen so far = "Low", top 20% =
    "High", middle 60% = "Medium". Falls back to the fixed defaults if there's
    no VIX history yet (shouldn't happen once the panel has started)."""
    idx = pd.to_datetime(vix.index)
    hist = vix[(idx >= pd.Timestamp(panel_inception)) & (idx <= pd.Timestamp(cutoff))]
    if hist.empty:
        return REGIME_LOW, REGIME_HIGH
    return float(hist.quantile(low_pct)), float(hist.quantile(high_pct))


def _percentile_regime_columns(
    raw_cache: pd.DataFrame, cutoff: str, low: float, high: float, *,
    panel_inception: str = DATA_START,
) -> pd.DataFrame:
    """Rebuilds regime_ic_<key>/regime_n_eff_<key> (one row per sub_factor) using
    percentile-based (low, high) VIX thresholds instead of the fixed thresholds
    baked into as_of()'s output -- same probability-weighted aggregation as
    regime_probability.effective_regime_stats(), computed directly from the raw
    per-date cache (sub_cache/parent_cache) so the threshold choice stays fully
    inside this module (no changes to regime_probability.py or
    regime_aware_evidence.py)."""
    cols = ["sub_factor"] + [f"regime_{f}_{k}" for k in ("low", "medium", "high")
                             for f in ("ic", "n_eff")]
    hist = raw_cache[(raw_cache["date"] <= cutoff) & (raw_cache["date"] >= panel_inception)]
    if hist.empty:
        return pd.DataFrame(columns=cols)
    hist = hist.copy()
    hist["mean_ic"] = hist[["ic_3M", "ic_6M"]].mean(axis=1)
    rows = []
    for sub, g in hist.groupby("sub_factor"):
        ic_by_date = g.set_index("date")["mean_ic"].dropna()
        vix_by_date = g.set_index("date")["vix_level"]
        weighted_ic = {r: 0.0 for r in REGIME_ORDER}
        n_eff = {r: 0.0 for r in REGIME_ORDER}
        for d, ic in ic_by_date.items():
            probs = regime_probabilities(float(vix_by_date.loc[d]), low=low, high=high)
            for r in REGIME_ORDER:
                p = probs[r]
                if not pd.notna(p):
                    continue
                n_eff[r] += p
                weighted_ic[r] += p * float(ic)
        row = {"sub_factor": sub}
        for r in REGIME_ORDER:
            key = r.split(" ")[0].lower()
            row[f"regime_ic_{key}"] = (weighted_ic[r] / n_eff[r]
                                       if n_eff[r] > 1e-9 else float("nan"))
            row[f"regime_n_eff_{key}"] = n_eff[r]
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def _regime_only_ic(
    row: pd.Series, vix_now: float, *, k: float = SHRINKAGE_K,
    low: float | None = None, high: float | None = None,
) -> float:
    """Probability-weighted, shrinkage-adjusted VIX-regime IC with NO long-run or
    recent blending -- mirrors the regime_component computed inside
    research.walkforward.regime_probability.expected_ic() (the 25%-weighted third
    term of the production "full" blend), just isolated on its own so a variant
    can use it as 100% of expected_ic instead of only ever seeing it diluted by
    the other two terms. Reconstructed from evidence columns as_of() already
    exposes (regime_ic_<key>/regime_n_eff_<key>/long_run_mean_ic) rather than by
    touching expected_ic() itself, so this doesn't change any production formula.
    ``low``/``high``, if given, override the fixed Low/High VIX thresholds used to
    weight the CURRENT vix_now into regime probabilities (paired with
    percentile-rebuilt regime_ic_*/regime_n_eff_* columns for consistency --
    see _percentile_regime_columns)."""
    long_run = row["long_run_mean_ic"]
    if not pd.notna(long_run):
        return float("nan")
    kwargs = {}
    if low is not None:
        kwargs["low"] = low
    if high is not None:
        kwargs["high"] = high
    probs = regime_probabilities(vix_now, **kwargs)
    component, total_p = 0.0, 0.0
    for r in REGIME_ORDER:
        p = probs.get(r, float("nan"))
        if not pd.notna(p):
            continue
        key = r.split(" ")[0].lower()
        shrunk = shrink_regime_ic(row[f"regime_ic_{key}"], long_run,
                                  row[f"regime_n_eff_{key}"], k=k)
        component += p * shrunk
        total_p += p
    return long_run if total_p <= 1e-9 else component / total_p


def _apply_variant_ic(
    evidence: pd.DataFrame, mode: str, *, vix_now: float | None = None,
    regime_low: float | None = None, regime_high: float | None = None,
) -> pd.DataFrame:
    """Overrides the "expected_ic" column per variant (spec Section 4): "full" is
    the Section 1 blend computed by as_of() (left unchanged); "long_run" (variant B)
    and "recent" (variants 24M/12M/5Y, window set via VariantSpec.recent_months)
    read a single evidence column directly, with "recent" falling back to long-run
    when fewer months of history exist yet than the window needs. "regime_only",
    "regime_recent", and "regime_longrun" isolate/recombine the VIX-regime
    component via _regime_only_ic (require ``vix_now``). ``regime_low``/
    ``regime_high``, if given, use percentile-based VIX thresholds instead of the
    fixed defaults -- the caller is expected to have already replaced
    ``evidence``'s regime_ic_*/regime_n_eff_* columns with percentile-rebuilt
    ones (see _percentile_regime_columns) so both halves stay consistent."""
    df = evidence.copy()
    recent = df["recent_24m_ic"].where(df["recent_24m_ic"].notna(), df["long_run_mean_ic"])
    if mode == "long_run":
        df["expected_ic"] = df["long_run_mean_ic"]
    elif mode == "recent":
        df["expected_ic"] = recent
    elif mode in ("regime_only", "regime_recent", "regime_longrun"):
        regime = df.apply(
            lambda row: _regime_only_ic(row, vix_now, low=regime_low, high=regime_high), axis=1)
        if mode == "regime_only":
            df["expected_ic"] = regime
        elif mode == "regime_recent":
            df["expected_ic"] = 0.5 * regime + 0.5 * recent
        else:
            df["expected_ic"] = 0.5 * regime + 0.5 * df["long_run_mean_ic"]
    return df


def _apply_floor(weights: dict[str, float], floor: float) -> dict[str, float]:
    """Floors every parent with a positive-but-below-floor weight up to ``floor``,
    funding the raise by shrinking the above-floor parents proportionally (so the
    floored parents land at exactly ``floor``, not ``floor`` diluted by a global
    renormalisation). The "separately tested variant" diversification floor from
    spec Section 3/5 -- not part of the primary C construction."""
    below = {p: v for p, v in weights.items() if 0 < v < floor}
    if not below:
        return weights
    w = dict(weights)
    for p in below:
        w[p] = floor
    above = [p for p in w if p not in below]
    above_total = sum(weights[p] for p in above)
    remaining = sum(weights.values()) - floor * len(below)
    if above_total > 0:
        for p in above:
            w[p] = weights[p] / above_total * remaining
    return w


def _possible_sign_inversion(p_scored: pd.DataFrame) -> pd.Series:
    """Bounded proxy for the spec's ``POSSIBLE_SIGN_INVERSION`` immediate-exit flag
    (spec Section 3, reusing ``research.parent_selection.horizon_flag``'s intent).

    ``horizon_flag`` needs four per-horizon IC columns (``ic_1M``/``ic_3M``/``ic_6M``/
    ``ic_12M``) that ``regime_aware_evidence.as_of()``'s schema doesn't carry -- only
    ``long_run_mean_ic``/``recent_24m_ic``/``expected_ic``. This mirrors the same
    "negative at (nearly) every horizon AND multi-horizon IC <= SIGN_INV_IC" test over
    the two horizons this schema actually tracks (long-run, recent) instead of four.
    """
    return (
        (p_scored["long_run_mean_ic"] < 0) & (p_scored["recent_24m_ic"] < 0)
        & (p_scored["expected_ic"] <= SIGN_INV_IC)
    )


def _mean_abs_weight_change(state_log: pd.DataFrame, variant: str, end_cutoff: str) -> float:
    """Mean absolute month-over-month change in realized parent weight, pooled across
    all parents, over the full monthly history from panel inception through
    ``end_cutoff`` -- a cumulative-to-date churn figure (not scoped to just the most
    recent window), since the walk-forward is one continuous simulation and later
    windows should reflect the accumulated stability of the whole path taken to reach
    them. Returns NaN if this variant has no logged monthly state (e.g. variant A,
    which never runs through run_monthly_variant) or fewer than 2 distinct months."""
    sub = state_log[(state_log["variant"] == variant) & (state_log["cutoff"] <= end_cutoff)]
    if sub.empty:
        return float("nan")
    pivot = sub.pivot(index="cutoff", columns="parent", values="weight").sort_index()
    if len(pivot) < 2:
        return float("nan")
    diffs = pivot.diff().iloc[1:].abs()
    return float(diffs.to_numpy().mean()) if diffs.size else float("nan")


def _subfactor_churn_per_year(state_log: pd.DataFrame, variant: str, end_cutoff: str) -> float:
    """Total subfactor entry+exit events across all parents (symmetric difference of
    consecutive months' active-subfactor sets, summed over parents and month
    transitions), annualised over the elapsed span from the first logged month
    through ``end_cutoff``. Same cumulative-to-date framing and NaN cases as
    _mean_abs_weight_change."""
    sub = state_log[(state_log["variant"] == variant) & (state_log["cutoff"] <= end_cutoff)]
    if sub.empty:
        return float("nan")
    events = 0
    for _parent, g in sub.groupby("parent"):
        g = g.sort_values("cutoff")
        sets = [set(s.split(",")) if s else set() for s in g["active_subs"]]
        for prev, cur in zip(sets, sets[1:]):
            events += len(prev ^ cur)
    dates = pd.to_datetime(sub["cutoff"].unique())
    span_years = (dates.max() - dates.min()).days / 365.25
    if span_years <= 0:
        return float("nan")
    return events / span_years


def _incremental_correlation_matrix(
    panel: ScorePanel, dates: list[str], cache: dict[str, pd.DataFrame | None], *,
    min_names: int = DEFAULT_MIN_NAMES,
) -> pd.DataFrame:
    """Same average-per-date Spearman correlation as
    ``research.subset_selection.correlation_matrix``, but a date's cross-sectional
    corr matrix never changes once that date's subfactor scores are realized, so
    each date is computed at most once and reused forever. Calling this every month
    over a monotonically-growing ``dates`` window (as ``run_monthly_variant`` does)
    then costs O(1) new work per month instead of recomputing the whole expanding
    history from scratch -- ``cache`` (keyed by date, ``None`` sentinel for an
    unusable date) is expected to persist across calls for the same walk."""
    subs = panel.all_subs
    for d in dates:
        if d in cache:
            continue
        frame = panel.signal_frame(d, subs)
        usable = [c for c in frame.columns if frame[c].nunique() >= 2]
        if len(usable) < 2 or frame[usable].dropna().shape[0] < min_names:
            cache[d] = None
            continue
        cache[d] = frame[usable].corr(method="spearman").reindex(index=subs, columns=subs)
    mats = [cache[d] for d in dates if cache.get(d) is not None]
    if not mats:
        return pd.DataFrame(index=subs, columns=subs, dtype=float)
    avg = np.nanmean(np.stack([m.to_numpy(dtype=float) for m in mats]), axis=0)
    return pd.DataFrame(avg, index=subs, columns=subs)


def _incremental_forward_returns(
    px: pd.DataFrame, dates: list[str], horizons: dict[str, int],
    cache: dict[str, dict[str, pd.Series]],
) -> dict[str, dict[str, pd.Series]]:
    """Same output as ``research.compute_forward_returns``, but a (horizon, date)
    forward return is fixed forever once its window-end falls inside the available
    price history, so it is computed at most once. Only the trailing
    ``max(horizons) + 1`` dates can still be *unresolved* (window not yet realized
    in ``px``) -- older dates either already resolved or, if a data gap ever
    prevents resolution, would stay unresolved regardless of retrying, so bounding
    the retry window keeps this O(1) new work per month instead of O(n)."""
    max_h = max(horizons.values())
    trailing = dates[-(max_h + 1):]
    pending = [d for d in trailing if any(d not in cache.get(h, {}) for h in horizons)]
    if pending:
        fresh = compute_forward_returns(px, pending, horizons)
        for h in horizons:
            cache.setdefault(h, {}).update(fresh.get(h, {}))
    return {h: {d: cache[h][d] for d in dates if d in cache.get(h, {})} for h in horizons}


@dataclass
class _ParentCacheState:
    """Persists one variant's parent-level ``build_monthly_cache`` output across
    the monthly loop, keyed by the sub_weights that produced it (see
    ``_incremental_parent_cache``)."""
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    dates: set[str] = field(default_factory=set)
    signature: tuple | None = None


def _sub_weights_signature(sub_weights: dict[str, dict[str, float]]) -> tuple:
    return tuple(sorted(
        (parent, tuple(sorted(w.items()))) for parent, w in sub_weights.items() if w))


def _incremental_parent_cache(
    parent_panel: ScorePanel, px: pd.DataFrame, vix: pd.Series,
    sub_weights: dict[str, dict[str, float]], state: _ParentCacheState,
) -> pd.DataFrame:
    """``build_monthly_cache`` recomputes IC/spread for every (parent, date) pair
    from scratch on every call. A composite parent's score at date ``d`` depends
    only on that date's subfactor scores and the CURRENT ``sub_weights``
    (``build_parent_panel`` has no cross-date dependency) -- so as long as
    ``sub_weights`` is unchanged since the last call, every previously-cached row
    is still valid and only genuinely new dates (normally just the newest month)
    need computing. ``sub_weights`` only changes at quarterly hysteresis
    boundaries for variants with hysteresis on -- when it does, the whole
    historical composite is invalidated and rebuilt once. Variant D has no freeze,
    so its sub_weights can change every month, and it pays the same full-rebuild
    cost every month as before -- an intentional consequence of D's "no freeze"
    design (the dynamic-monthly overfitting benchmark), not a regression."""
    sig = _sub_weights_signature(sub_weights)
    if sig != state.signature:
        state.df = build_monthly_cache(parent_panel, px, vix)
        state.dates = set(parent_panel.rebal_dates)
        state.signature = sig
        return state.df
    new_dates = [d for d in parent_panel.rebal_dates if d not in state.dates]
    if new_dates:
        addition = build_monthly_cache(slice_panel(parent_panel, new_dates), px, vix)
        state.df = pd.concat([state.df, addition], ignore_index=True)
        state.dates.update(new_dates)
    return state.df


@dataclass
class _MonthlyState:
    """Carried across the monthly loop for ONE variant."""
    hysteresis: dict[str, HysteresisState] = field(default_factory=dict)
    sub_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    prev_month_parent_w: dict[str, float] = field(default_factory=dict)
    quarter_start_parent_w: dict[str, float] = field(default_factory=dict)
    parent_cache: _ParentCacheState = field(default_factory=_ParentCacheState)


def run_monthly_variant(
    panel: ScorePanel, matrix: pd.DataFrame, vix: pd.Series,
    sub_cache: pd.DataFrame, variant: VariantSpec, *,
    test_boundaries: list[str], state_log: list[dict] | None = None,
    verbose: bool = False,
    corr_cache: dict[str, pd.DataFrame | None] | None = None,
    fwd_cache: dict[str, dict[str, pd.Series]] | None = None,
    select_config_cache: dict[str, FrozenConfig] | None = None,
) -> dict[str, FrozenConfig]:
    """Steps monthly from the panel's first rebalance through the last test
    boundary, carrying hysteresis/weight state, and snapshots a FrozenConfig at
    each date in ``test_boundaries``. Returns {boundary_date: FrozenConfig}.

    ``state_log``, if given, is an accumulator list that gets one row per
    (cutoff, parent) appended for **every** month walked (not just the semiannual
    test-boundary months captured in the returned snapshot dict) -- the full
    monthly (realized parent weight, active-subfactor-membership) history that
    ``state_history.csv`` and the churn metrics (``_mean_abs_weight_change``,
    ``_subfactor_churn_per_year``) are derived from.

    ``verbose``, if set, prints one line per month walked (index/total + cutoff
    date) -- purely an observability aid for the slow real-panel case; off by
    default so it doesn't clutter the small synthetic-panel unit tests.

    ``corr_cache``/``fwd_cache``/``select_config_cache``, if given, back the
    incremental correlation-matrix/forward-returns helpers and the per-cutoff
    ``select_config`` (5Y base-weight) call -- all three depend only on
    ``panel``/``matrix``/``cutoff``, never on the variant, so
    ``run_walkforward_comparison`` passes the SAME three dicts into every
    variant's call so this work is paid for once per cutoff, not once per
    (cutoff, variant). ``select_config`` (via ``subfactor_performance``) is by
    far the most expensive of the three on real data -- profiling showed it at
    ~88% of total per-month cost, dwarfing the other three hotspots this
    incremental-caching scheme originally targeted. Each dict defaults to a
    fresh one per call when not given (e.g. the single-variant unit
    tests/integration test).
    """
    corr_cache = {} if corr_cache is None else corr_cache
    fwd_cache = {} if fwd_cache is None else fwd_cache
    select_config_cache = {} if select_config_cache is None else select_config_cache
    state = _MonthlyState()
    for p in panel.parent_keys:
        state.hysteresis[p] = HysteresisState()
        state.sub_weights[p] = {}
    snapshots: dict[str, FrozenConfig] = {}

    walked = [d for d in panel.rebal_dates if d <= test_boundaries[-1]]
    for i, cutoff in enumerate(walked, start=1):
        if verbose:
            print(f"  [{variant.name}] month {i}/{len(walked)}: {cutoff}", flush=True)
        px = matrix.loc[matrix.index <= cutoff]
        # Full expanding history -- feeds subfactor evidence/selection (spec Section 1
        # deliberately anchors long_run evidence at panel inception, not a rolling
        # window). The 70% BASE weight below is a separate, narrower window -- see
        # base_train_rebals.
        train_rebals = [d for d in panel.rebal_dates if d <= cutoff]
        # Trailing 5Y ending at cutoff, with the same forward-return-horizon safety
        # cap WalkForwardSplit.train_rebalances() applies -- "on a trailing 5-year
        # training window ending at cutoff. This is exactly variant A's construction"
        # (spec Section 3, "Base weight (the '70%')"). Without the horizon cap, a
        # training rebal too close to cutoff would have its 6M forward return
        # silently truncated by the boundary-clipped price matrix instead of dropped,
        # corrupting the IC estimate used for ic_ir_weights.
        base_train_start = max(
            (pd.Timestamp(cutoff) - pd.DateOffset(years=5)).date().isoformat(), DATA_START)
        base_cap = (pd.Timestamp(cutoff)
                   - pd.DateOffset(months=SELECTION_HORIZON_MONTHS)).date().isoformat()
        base_train_rebals = [d for d in train_rebals if base_train_start <= d <= base_cap]

        vix_now = _vix_spot(vix, cutoff)
        regime_low = regime_high = None
        if variant.vix_percentile:
            regime_low, regime_high = _percentile_vix_thresholds(vix, cutoff)
        sub_evidence = as_of(sub_cache, cutoff, vix, recent_months=variant.recent_months)
        if variant.vix_percentile:
            pct_cols = _percentile_regime_columns(sub_cache, cutoff, regime_low, regime_high)
            sub_evidence = sub_evidence.drop(
                columns=[c for c in pct_cols.columns if c != "sub_factor"]
            ).merge(pct_cols, on="sub_factor", how="left")
        evidence = _apply_variant_ic(sub_evidence, variant.expected_ic_mode, vix_now=vix_now,
                                     regime_low=regime_low, regime_high=regime_high)
        scored = score_table(evidence)
        fwd_by_h = _incremental_forward_returns(px, train_rebals, {"3M": 3, "6M": 6}, fwd_cache)
        sub_panel = slice_panel(panel, train_rebals)
        corr = _incremental_correlation_matrix(sub_panel, train_rebals, corr_cache)

        active_members_by_parent: dict[str, list[str]] = {}
        for parent in panel.parent_keys:
            p_scored = scored[scored.parent == parent]
            if p_scored.empty:
                state.sub_weights[parent] = {}
                continue
            result = select_parent_subfactors(p_scored, corr, sub_panel, fwd_by_h)
            would_select = result["selected"]
            eligible = set(p_scored[p_scored.eligible]["sub_factor"])
            immediate_exit = set(p_scored[
                _possible_sign_inversion(p_scored) | (p_scored.coverage < 0.50)
            ]["sub_factor"])

            hstate = state.hysteresis[parent]
            if variant.hysteresis:
                update_streaks(hstate, would_select, eligible, immediate_exit)
                if _is_quarter_boundary(cutoff):
                    apply_quarterly_membership(hstate, would_select)
                active_members = list(hstate.members)
            else:
                active_members = list(would_select)     # D: no freeze, one-shot every month
            active_members_by_parent[parent] = active_members

            metric_map = p_scored.set_index("sub_factor")
            state.sub_weights[parent] = parent_subfactor_weights(
                active_members, metric_map["production_score"])

        # --- Parent-level evidence + weighting (reuses build_monthly_cache/as_of on
        # the constructed parent-composite panel -- see Task 5's reuse note). ---
        parent_panel = build_parent_panel(sub_panel, state.sub_weights)
        if parent_panel.parent_keys:
            parent_cache = _incremental_parent_cache(
                parent_panel, px, vix, state.sub_weights, state.parent_cache)
            parent_raw_evidence = as_of(parent_cache, cutoff, vix,
                                        recent_months=variant.recent_months)
            if variant.vix_percentile:
                pct_cols = _percentile_regime_columns(parent_cache, cutoff, regime_low,
                                                      regime_high)
                parent_raw_evidence = parent_raw_evidence.drop(
                    columns=[c for c in pct_cols.columns if c != "sub_factor"]
                ).merge(pct_cols, on="sub_factor", how="left")
            parent_evidence = _apply_variant_ic(
                parent_raw_evidence, variant.expected_ic_mode, vix_now=vix_now,
                regime_low=regime_low, regime_high=regime_high)
            utility = parent_utility_table(parent_evidence).set_index("sub_factor")
        else:
            # No parent has a hysteresis-frozen subfactor yet (early warm-up months,
            # before ENTRY_MONTHS + a quarter boundary can seat one) -- no parent-level
            # evidence exists to build an adaptive component from. adaptive_weight/
            # blend_parent_weights below both treat an empty utility/expected_ic as "no
            # adaptive signal", so the blend falls back to 100% base weight.
            utility = pd.DataFrame({"utility_score": pd.Series(dtype=float),
                                    "expected_ic": pd.Series(dtype=float)})

        if cutoff not in select_config_cache:
            select_config_cache[cutoff] = select_config(panel, base_train_rebals, matrix,
                                                         boundary=cutoff)
        base_cfg = select_config_cache[cutoff]
        adaptive_w = adaptive_weight(utility["utility_score"], utility["expected_ic"])
        target_w = blend_parent_weights(base_cfg.parent_weights, adaptive_w,
                                        dict(utility["expected_ic"]),
                                        base_weight_frac=variant.base_weight_frac)
        if variant.diversification_floor:
            # _apply_floor lands floored parents at exactly `floor` in target_w, but
            # apply_change_caps below can still clip that raise back down in the same
            # month (the cap is a uniform risk control regardless of where target_w
            # came from) -- the floor then ramps in over subsequent months instead of
            # landing exactly on `floor` immediately.
            target_w = _apply_floor(target_w, variant.diversification_floor)

        if variant.weight_caps:
            # No real prior state yet on the first month (or before the first quarter
            # boundary) -- `or target_w` uses the target itself as the cap's reference
            # so the band is a no-op instead of clamping every parent toward the
            # pm=qs=0.0 default, which would cap every weight at min(monthly_cap,
            # quarterly_cap) and leave the redistribution loop unable to reach sum=1.
            realized_w = apply_change_caps(
                target_w, state.prev_month_parent_w or target_w,
                state.quarter_start_parent_w or target_w)
        else:
            realized_w = target_w

        if state_log is not None:
            for parent in panel.parent_keys:
                state_log.append({
                    "cutoff": cutoff, "variant": variant.name, "parent": parent,
                    "weight": realized_w.get(parent, 0.0),
                    "active_subs": ",".join(sorted(active_members_by_parent.get(parent, []))),
                })

        state.prev_month_parent_w = realized_w
        if _is_quarter_boundary(cutoff):
            state.quarter_start_parent_w = realized_w

        if cutoff in test_boundaries:
            snapshots[cutoff] = FrozenConfig(
                sub_weights={p: dict(w) for p, w in state.sub_weights.items()},
                parent_weights=dict(realized_w),
                meta={"cutoff": cutoff, "variant": variant.name})
    return snapshots


METRIC_COLS = ["cagr", "sharpe", "sortino", "max_drawdown", "spy_excess_cagr",
              "spy_ir", "spy_beta", "spy_alpha", "avg_turnover"]


def _score_variant(name: str, sp: WalkForwardSplit, cfg: FrozenConfig, panel: ScorePanel,
                   matrix: pd.DataFrame, sectors: pd.Series,
                   fwd: dict[str, dict[str, pd.Series]],
                   state_log: pd.DataFrame, boundary: str | None) -> dict:
    test_rebals = sp.test_rebalances(panel.rebal_dates)
    scores = frozen_composite(panel, test_rebals, cfg, sectors)
    ic_df = analysis.composite_ic(scores, fwd)

    def _g(h: str, col: str) -> float:
        r = ic_df[ic_df["horizon"] == h]
        return float(r.iloc[0][col]) if not r.empty and col in r.columns else float("nan")

    qres = analysis.quantile_analysis(scores, fwd)
    q6 = qres.get("6M", {})
    spr = q6.get("spread") or {}
    book = pf.simulate(scores, matrix, sectors, top_pct=0.20, mode="equal", hold_months=1)
    row = {
        "window": sp.label.split(":", 1)[-1], "variant": name,
        "ic_3m": _g("3M", "mean_ic"), "ic_6m": _g("6M", "mean_ic"),
        "ic_ir_6m": _g("6M", "information_ratio"), "hit_rate_6m": _g("6M", "hit_rate"),
        "q5q1_ann": float(spr.get("annualized", float("nan"))),
        "monotonic_rate_6m": float(q6.get("monotonic_rate", float("nan"))),
    }
    row.update({k: book.metrics.get(k, float("nan")) for k in METRIC_COLS})
    row["mean_abs_weight_chg"] = (_mean_abs_weight_change(state_log, name, boundary)
                                  if boundary is not None else float("nan"))
    row["subfactor_churn_per_year"] = (_subfactor_churn_per_year(state_log, name, boundary)
                                       if boundary is not None else float("nan"))
    return row


def run_walkforward_comparison(
    panel: ScorePanel, matrix: pd.DataFrame, vix: pd.Series, sectors: pd.Series, *,
    first_test_year: int = 2017, last_end: str | None = None, verbose: bool = True,
    variant_names: list[str] | None = None, include_baseline_a: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs the named variants (spec Section 4; defaults to all of ``VARIANTS`` plus
    baseline "A") over the same 19-window semiannual rolling-5Y grid every other
    regime study in this repo uses, and returns one row per (window, variant) with
    the full metric suite, plus the full monthly state log (one row per (cutoff,
    parent, variant) covering every month walked, for ``state_history.csv`` and the
    churn metrics).

    ``variant_names``, if given, restricts the run to that subset of ``VARIANTS``
    keys (default: run all of them). ``include_baseline_a``, if False, skips
    scoring the current-production baseline "A" entirely -- useful for a study
    that only wants to compare among the new pipeline's variants.

    Returns ``(comparison_df, state_log_df)``.
    """
    names_to_run = list(VARIANTS) if variant_names is None else list(variant_names)
    kwargs: dict = {"first_test_year": first_test_year}
    if last_end is not None:
        kwargs["last_end"] = last_end
    splits = semiannual_policy_splits("rolling5y", **kwargs)
    if not splits:
        return pd.DataFrame(), pd.DataFrame()
    # A split's ``test_start`` is a first-of-month date (e.g. "2019-01-01"), but
    # run_monthly_variant's loop only ever visits literal ``panel.rebal_dates``
    # (month-end), so snapshotting must key off the last actual rebalance on or
    # before ``test_start`` -- not ``test_start`` itself, which would never match.
    boundary_by_split = {
        sp.label: max((d for d in panel.rebal_dates if d <= sp.test_start), default=None)
        for sp in splits
    }
    test_boundaries = sorted({b for b in boundary_by_split.values() if b is not None})

    sub_cache = build_monthly_cache(panel, matrix, vix)

    # corr_cache/fwd_cache/select_config_cache are shared across every variant: at
    # a given cutoff, train_rebals/px/base_train_rebals depend only on the panel
    # and cutoff, not on the variant, so every variant would otherwise recompute
    # the identical correlation matrix, forward returns, AND (most expensively --
    # profiling showed ~88% of total per-month cost) 5Y base-weight select_config
    # call for that cutoff. Sharing the three caches here means that expanding-
    # history work is paid for once per cutoff instead of once per
    # (cutoff, variant) -- an Nx reduction (N = number of variants run) on top of
    # each variant's own month-over-month incremental reuse.
    corr_cache: dict[str, pd.DataFrame | None] = {}
    fwd_cache: dict[str, dict[str, pd.Series]] = {}
    select_config_cache: dict[str, FrozenConfig] = {}

    state_log: list[dict] = []
    monthly_snapshots: dict[str, dict[str, FrozenConfig]] = {}
    for name in names_to_run:
        vspec = VARIANTS[name]
        if verbose:
            print(f"  running variant {name} (monthly loop)...")
        monthly_snapshots[name] = run_monthly_variant(
            panel, matrix, vix, sub_cache, vspec, test_boundaries=test_boundaries,
            state_log=state_log, corr_cache=corr_cache, fwd_cache=fwd_cache,
            select_config_cache=select_config_cache)
    state_log_df = pd.DataFrame(state_log)

    rows: list[dict] = []
    for sp in splits:
        test_rebals = sp.test_rebalances(panel.rebal_dates)
        if not test_rebals:
            continue
        fwd = compute_forward_returns(matrix, test_rebals, {"3M": 3, "6M": 6})
        boundary = boundary_by_split.get(sp.label)

        if include_baseline_a:
            cfg_a = select_config(panel, sp.train_rebalances(panel.rebal_dates), matrix,
                                  boundary=sp.test_start)
            rows.append(_score_variant("A", sp, cfg_a, panel, matrix, sectors, fwd,
                                       state_log_df, boundary))

        for name in names_to_run:
            cfg = monthly_snapshots[name].get(boundary) if boundary is not None else None
            if cfg is None:
                continue
            rows.append(_score_variant(name, sp, cfg, panel, matrix, sectors, fwd,
                                       state_log_df, boundary))
    return pd.DataFrame(rows), state_log_df
