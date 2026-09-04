"""Candidate subfactors for the flow-based parents: revisions, institutional,
insider, short. Separated from :mod:`library` so both files stay under 500 lines.

Same rules as ``library.py``: raw Series only, direction flags, no percentile
step here — the panel handles that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.grades import PT_MAX_RATIO, PT_MIN_RATIO, PT_TARGET_SQL
from factors.utils import DataContext, col

# Shared WHERE fragment for every price-target-event query below: read the
# split-adjusted target (same basis as price_when_posted / adj_close — see
# data/grades.py PT_* constants for the 2026-08-02 split-artifact fix) and
# drop the residual junk rows outside the plausibility band.
_PT_EVENT_FILTER = (
    f"price_when_posted > 0 AND {PT_TARGET_SQL} > 0 "
    f"AND {PT_TARGET_SQL} / price_when_posted BETWEEN {PT_MIN_RATIO} AND {PT_MAX_RATIO}"
)


def _pit_avail(ctx: DataContext, column: str, modifier: str) -> str:
    """SQL predicate admitting ``column`` rows only once publicly available.

    Period-keyed tables (13F quarter ends, FINRA settlement dates, monthly
    grade rows) become public *after* their key date; in PIT mode (real cutoff
    + reporting_lag) shift availability by ``modifier`` (e.g. ``'+14 day'``).
    Live mode keeps the raw ``column <= ?`` bound.
    """
    if ctx.reporting_lag and ctx.cutoff <= "9000-01-01":
        return f"date({column}, '{modifier}') <= ?"
    return f"{column} <= ?"


def _safe_cutoff(ctx: DataContext) -> str:
    """Return a real trading date to base window arithmetic on.

    ``ctx.cutoff`` is a far-future sentinel ("9999-12-31") when no --date is
    passed — pandas overflows if you try to build a Timedelta from it. Use
    ``ref_date`` (the latest observed daily-price date) as the anchor instead.
    """
    if ctx.cutoff > "9000-01-01":
        return ctx.ref_date
    return ctx.cutoff

# Late import to avoid the circular from library.py.
def _Candidate(*args, **kwargs):  # pragma: no cover — trivial passthrough
    from .library import Candidate
    return Candidate(*args, **kwargs)


# =============================================================================
# Revisions (analyst)
# =============================================================================


def build_revisions(ctx: DataContext) -> list:
    rev = ctx.revisions()
    est = ctx.estimate_features()

    rating_30d = col(rev, "rating_change_30d")
    rating_90d = col(rev, "rating_change_90d")
    pt_momentum = col(rev, "pt_momentum")
    pt_upside = col(rev, "pt_target_upside_30d")
    pt_upgrade_ratio = col(rev, "pt_upgrade_ratio_30d")

    # Analyst-count change 90d: derived from the total_analysts snapshots stored
    # in analyst_revision_features. This lives outside the incumbent factor.
    analyst_count_change = _analyst_count_change(ctx, days=90)

    # Grade diffusion + its 90d change from analyst_grades (monthly).
    grade_diff_latest, grade_diff_change = _grade_diffusion(ctx, days=90)

    # Numeric forward-EPS revision (already coverage-gated in production).
    fwd_eps_rev = col(est, "forward_eps_revision_90d")

    # --- 2026-08-01 candidate battery: is there anything left on the table in
    # data we already ingest from price-target-news (analyst_price_target_events),
    # which currently only feeds unweighted 30d upside/upgrade-ratio aggregates?
    # Coverage LEVEL (vs. the existing 90d *change* candidate above), a firm
    # track-record-weighted consensus target, and target dispersion/disagreement.
    coverage_level = col(rev, "total_analysts")
    firm_skill_weighted_upside = _firm_skill_weighted_upside(ctx, window_days=30)
    pt_dispersion = _target_dispersion(ctx, window_days=30)
    bull_decile_upside, bear_decile_upside = _decile_upside(ctx, window_days=30)

    # --- 2026-08-01 revision-surprise battery: does the CHANGE in a firm's
    # target, net of the stock's own move since their last quote, carry more
    # information than the target LEVEL (pt_target_upside_30d) does? Per-event
    # ARS = ln(target_i/target_{i-1}) - ln(price_i/price_{i-1}) for consecutive
    # same-firm/same-ticker quotes — see _revision_surprise_events for the full
    # spec and PIT/quality controls.
    ev_30 = _revision_surprise_events(ctx, window_days=30)
    ev_60 = _revision_surprise_events(ctx, window_days=60)
    _equal_w = lambda e: pd.Series(1.0, index=e.index)  # noqa: E731
    _skill_w = lambda e: e["skill"]  # noqa: E731
    _decay_w = lambda e: np.power(2.0, -e["age_days"] / _REVISION_DECAY_HALF_LIFE)  # noqa: E731
    _skill_decay_w = lambda e: e["skill"] * _decay_w(e)  # noqa: E731

    raw_revision_30d = _wins_ctx(ctx, _agg_weighted(ctx, ev_30, "raw_revision", _equal_w))
    surprise_equal_30d = _wins_ctx(ctx, _agg_weighted(ctx, ev_30, "ars", _equal_w))
    surprise_skill_30d = _wins_ctx(ctx, _agg_weighted(ctx, ev_30, "ars", _skill_w))
    surprise_decay_60d = _wins_ctx(ctx, _agg_weighted(ctx, ev_60, "ars", _decay_w))
    surprise_skill_decay_60d = _wins_ctx(ctx, _agg_weighted(ctx, ev_60, "ars", _skill_decay_w))
    ev_60_down = ev_60[ev_60["ars"] < 0] if not ev_60.empty else ev_60
    surprise_down_skill_decay_60d = _wins_ctx(
        ctx, _agg_weighted(ctx, ev_60_down, "ars", _skill_decay_w))
    surprise_breadth_60d = _wins_ctx(
        ctx, _revision_breadth_confirmed(ctx, ev_60, _REVISION_DECAY_HALF_LIFE))

    # --- 2026-08-03 confirmation + J-shape magnitude candidates, from the
    # analyst deep-dive (output/analyst_deep_dive/FINDINGS.md). Thresholds and
    # windows fixed a priori — see the _RECO_*/_HIGH_UPSIDE_*/_UPSIDE_TROUGH
    # constants. `raise_and_bullish` was the deep-dive's one incremental winner
    # (+0.0066 parent IC in-window); the two magnitude candidates exploit the
    # J-shape (returns rise with implied upside only above ~+10-20%, so the
    # linear mean is poisoned by the 0-10% "faint praise" band).
    agreement_90d, raise_and_bullish_90d, high_upside_breadth_90d = \
        _reco_confirmation_signals(ctx)
    upside_above_trough_30d = _upside_above_trough(ctx, window_days=30)

    # --- 2026-08-03 rating-selectivity battery (analyst_grade_events, new
    # /stable/grades ingest — individual actions with firm identity, ~2012+).
    # See _grade_action_signals and PREREGISTRATION_2026-08-03 addendum.
    action_net_90d, rating_surprise_90d, selective_bull_90d = \
        _grade_action_signals(ctx, window_days=90)
    # Interaction: consensus bullishness confirmed by directional agreement —
    # product of within-universe percentile ranks of the two components.
    upgrade_x_agreement_90d = (pt_upgrade_ratio.rank(pct=True)
                               * agreement_90d.rank(pct=True))

    return [
        _Candidate("rev_rating_change_30d", rating_30d, True, "revisions", "rating"),
        _Candidate("rev_rating_change_90d", rating_90d, True, "revisions", "rating"),
        _Candidate("rev_pt_momentum", pt_momentum, True, "revisions", "price_target"),
        _Candidate("rev_pt_target_upside_30d", pt_upside, True, "revisions", "price_target"),
        _Candidate("rev_pt_upgrade_ratio_30d", pt_upgrade_ratio, True, "revisions", "price_target"),
        _Candidate("rev_analyst_count_change_90d", analyst_count_change, True, "revisions", "coverage"),
        _Candidate("rev_grade_diffusion", grade_diff_latest, True, "revisions", "diffusion"),
        _Candidate("rev_grade_diffusion_change_90d", grade_diff_change, True, "revisions", "diffusion"),
        _Candidate("rev_forward_eps_revision_90d", fwd_eps_rev, True, "revisions", "numeric"),
        # Hypothesis: neglected names (thin coverage) are rewarded (higher_is_better=False).
        _Candidate("rev_coverage_level", coverage_level, False, "revisions", "coverage"),
        _Candidate("rev_firm_skill_weighted_upside_30d", firm_skill_weighted_upside, True, "revisions", "skill"),
        # Hypothesis: analyst disagreement is a negative/uncertainty signal (higher_is_better=False).
        _Candidate("rev_pt_target_dispersion_30d", pt_dispersion, False, "revisions", "price_target"),
        _Candidate("rev_pt_bull_decile_upside_30d", bull_decile_upside, True, "revisions", "price_target"),
        _Candidate("rev_pt_bear_decile_upside_30d", bear_decile_upside, True, "revisions", "price_target"),
        _Candidate("rev_target_revision_raw_30d", raw_revision_30d, True, "revisions", "revision_surprise"),
        _Candidate("rev_revision_surprise_equal_30d", surprise_equal_30d, True, "revisions", "revision_surprise"),
        _Candidate("rev_revision_surprise_skill_30d", surprise_skill_30d, True, "revisions", "revision_surprise"),
        _Candidate("rev_revision_surprise_decay_60d", surprise_decay_60d, True, "revisions", "revision_surprise"),
        _Candidate("rev_revision_surprise_skill_decay_60d", surprise_skill_decay_60d, True, "revisions", "revision_surprise"),
        _Candidate("rev_revision_surprise_down_skill_decay_60d", surprise_down_skill_decay_60d, True, "revisions", "revision_surprise"),
        _Candidate("rev_revision_surprise_breadth_60d", surprise_breadth_60d, True, "revisions", "revision_surprise"),
        _Candidate("rev_agreement_90d", agreement_90d, True, "revisions", "confirmation"),
        _Candidate("rev_raise_and_bullish_90d", raise_and_bullish_90d, True, "revisions", "confirmation"),
        _Candidate("rev_upgrade_x_agreement_90d", upgrade_x_agreement_90d, True, "revisions", "confirmation"),
        _Candidate("rev_high_upside_breadth_90d", high_upside_breadth_90d, True, "revisions", "magnitude"),
        _Candidate("rev_upside_above_trough_30d", upside_above_trough_30d, True, "revisions", "magnitude"),
        _Candidate("rev_grade_action_net_90d", action_net_90d, True, "revisions", "selectivity"),
        _Candidate("rev_rating_surprise_90d", rating_surprise_90d, True, "revisions", "selectivity"),
        _Candidate("rev_selective_bull_90d", selective_bull_90d, True, "revisions", "selectivity"),
    ]


def _analyst_count_change(ctx: DataContext, days: int) -> pd.Series:
    """Difference in ``total_analysts`` between the latest snapshot and the one
    ~``days`` earlier per ticker. Missing when we don't have both endpoints.
    """
    end = _safe_cutoff(ctx)
    start = (pd.Timestamp(end) - pd.Timedelta(days=days * 2)).date().isoformat()
    df = ctx.db.query_df(
        "SELECT ticker, snapshot_date, total_analysts FROM analyst_revision_features "
        "WHERE snapshot_date <= ? AND snapshot_date >= ?",
        (end, start),
    )
    # Early-era snapshots can carry NULL total_analysts; a None−None subtraction
    # below would raise and take the whole revisions family down for that date.
    df["total_analysts"] = pd.to_numeric(df["total_analysts"], errors="coerce")
    df = df.dropna(subset=["total_analysts"])
    if df.empty:
        return pd.Series(dtype=float, index=ctx.universe)
    target = (pd.Timestamp(end) - pd.Timedelta(days=days)).date().isoformat()

    def _diff(g: pd.DataFrame) -> float:
        g = g.sort_values("snapshot_date")
        latest = g.iloc[-1]
        prior = g[g["snapshot_date"] <= target]
        if prior.empty:
            return np.nan
        return float(latest["total_analysts"] - prior.iloc[-1]["total_analysts"])

    return df.groupby("ticker").apply(_diff).reindex(ctx.universe)


def _grade_diffusion(ctx: DataContext, days: int) -> tuple[pd.Series, pd.Series]:
    """Latest diffusion index and its 90d change from ``analyst_grades``.

    Diffusion = (strong_buy + buy - sell - strong_sell) / total. Latest month
    snapshot per ticker; the 90d change looks up the snapshot closest to
    ``as_of - 90d`` and takes the difference.
    """
    end = _safe_cutoff(ctx)
    start = (pd.Timestamp(end) - pd.Timedelta(days=days * 2)).date().isoformat()
    # Month rows are keyed at month START and accrete in place until the month
    # ends, so in PIT mode only completed months are admitted (+1 month).
    df = ctx.db.query_df(
        "SELECT ticker, date, strong_buy, buy, hold, sell, strong_sell, total "
        f"FROM analyst_grades WHERE {_pit_avail(ctx, 'date', '+1 month')} AND date >= ?",
        (end, start),
    )
    empty = pd.Series(dtype=float, index=ctx.universe)
    if df.empty:
        return empty, empty.copy()
    total = df["total"].replace(0, np.nan)
    df["diffusion"] = (df["strong_buy"] + df["buy"] - df["sell"] - df["strong_sell"]) / total
    df = df.sort_values(["ticker", "date"])

    latest = df.drop_duplicates("ticker", keep="last").set_index("ticker")["diffusion"]

    target = (pd.Timestamp(end) - pd.Timedelta(days=days)).date().isoformat()
    prior = (df[df["date"] <= target]
             .drop_duplicates("ticker", keep="last")
             .set_index("ticker")["diffusion"])
    change = (latest - prior).reindex(ctx.universe)
    return latest.reindex(ctx.universe), change


# Firm-level price-target skill weighting + disagreement — both self-contained
# in ``analyst_price_target_events`` (already ingested from price-target-news),
# no new data pull required.

_NEUTRAL_SKILL = 0.5      # fallback for a firm with zero resolved checkpoints at all
_SKILL_HALF_LIFE_DAYS = 730  # 2y recency half-life on each resolved checkpoint
_SKILL_SHRINK_K = 25         # shrinkage strength (effective-checkpoints) toward the cross-firm mean
_SKILL_DIRECTION_WEIGHT = 0.30  # q = (1-w)*magnitude_accuracy + w*direction_hit
_SKILL_CHECKPOINT_MONTHS = (3, 6, 9, 12)  # a target is a ~12mo forecast; check progress at each
_SKILL_CHECKPOINT_TOLERANCE_DAYS = 15     # nearest-trading-day match tolerance per checkpoint
_VOL_LOOKBACK_DAYS = 252
_VOL_FLOOR = 0.02             # ann. vol floor so the error denominator can't collapse


def _annualized_vol(ctx: DataContext, lookback_days: int = _VOL_LOOKBACK_DAYS) -> pd.Series:
    """Trailing annualized daily-return vol per ticker, as of the as-of cutoff.

    Used only to scale firm-skill forecast errors by how much a name normally
    moves (see :func:`_firm_skill_scores`) — reused as one "current" vol level
    across all of a firm's historical events rather than a vol series matched
    to each event's own date (that would need a rolling-window join per event;
    more precision than a research weighting input needs). Still strictly
    PIT: the underlying price history never extends past the as-of cutoff.
    """
    from .library import _price_history_days  # late import — avoids the library.py/library_flow.py cycle
    px = _price_history_days(ctx, lookback_days)
    if px.empty:
        return pd.Series(dtype=float)
    return px.pct_change(fill_method=None).std() * np.sqrt(252)


def _firm_skill_scores(ctx: DataContext) -> dict[str, float]:
    """PIT-safe per-firm price-target skill, keyed by ``analyst_company``.

    Checks each target against the stock's *actual* price at fixed checkpoints
    (3, 6, 9, 12 months after it was issued) rather than against whatever
    price happened to be current the next time the firm re-quoted that ticker.
    A target is roughly a 12-month forecast, so a name is only expected to
    have covered a proportional share of the implied move at each earlier
    checkpoint — checking after 1 month should only show ~1/12 of the way
    there, not the whole move. One event can contribute up to 4 checkpoint
    observations (only the ones that have already happened as of the as-of
    cutoff are used, so this stays strictly PIT). Firm-level rather than
    per-analyst: ``analyst_name`` is null/blank on ~35% of rows while
    ``analyst_company`` is always populated.

    For each (event, checkpoint) pair::

        u        = ln(target / price_when_posted)          # ~12mo expected log return
        frac     = checkpoint_months / 12                  # how far along we expect to be
        r_hat    = frac * u                                 # expected progress by the checkpoint
        r        = ln(actual_price_at_checkpoint / price_when_posted)  # realized log return
        e        = |r - r_hat| / (ticker_ann_vol * sqrt(frac) + vol_floor)
        acc      = exp(-e)                                  # in (0, 1], 1 = spot on
        dir      = 1 if sign(r) == sign(u) else 0
        q        = 0.70*acc + 0.30*dir

    Each checkpoint observation is recency-weighted (2y half-life on the
    checkpoint date itself, not the original quote date) and a firm's score
    is its weighted mean ``q`` shrunk toward the cross-firm weighted mean
    using effective-sample-size shrinkage — a firm with a handful of
    checkpoints lands near the population average rather than at a perfect
    (or hard-neutral) score, and confidence rises smoothly with checkpoint
    count instead of snapping in at a hard cutoff.
    """
    end = _safe_cutoff(ctx)
    cutoff_ts = pd.Timestamp(end)
    df = ctx.db.query_df(
        f"SELECT ticker, analyst_company, published_date, "
        f"{PT_TARGET_SQL} AS price_target, price_when_posted "
        "FROM analyst_price_target_events "
        f"WHERE published_date <= ? AND {_PT_EVENT_FILTER}",
        (end,),
    )
    if df.empty:
        return {}
    df["published_date"] = pd.to_datetime(df["published_date"], utc=True).dt.tz_localize(None)

    # Long frame: one row per (event, checkpoint) whose checkpoint date has
    # already occurred as of the as-of cutoff.
    rows = []
    for months in _SKILL_CHECKPOINT_MONTHS:
        cp_date = df["published_date"] + pd.DateOffset(months=months)
        due = cp_date <= cutoff_ts
        if not due.any():
            continue
        sub = df.loc[due, ["ticker", "analyst_company", "price_target", "price_when_posted"]].copy()
        sub["checkpoint_date"] = cp_date[due]
        sub["horizon_months"] = months
        rows.append(sub)
    if not rows:
        return {}
    long = pd.concat(rows, ignore_index=True)

    # Actual price at each checkpoint: one bulk price pull spanning every
    # checkpoint date needed, then a per-ticker nearest-date match — much
    # cheaper than a query per event.
    px = ctx.db.query_df(
        "SELECT ticker, date, adj_close FROM daily_prices WHERE date >= ? AND date <= ?",
        (long["checkpoint_date"].min().date().isoformat(), end),
    )
    if px.empty:
        return {}
    px["date"] = pd.to_datetime(px["date"])
    long = long.sort_values("checkpoint_date")
    px = px.sort_values("date")
    matched = pd.merge_asof(
        long, px, left_on="checkpoint_date", right_on="date", by="ticker",
        direction="nearest", tolerance=pd.Timedelta(days=_SKILL_CHECKPOINT_TOLERANCE_DAYS),
    )
    matched = matched.dropna(subset=["adj_close"]).copy()
    if matched.empty:
        return {}

    u = np.log(matched["price_target"] / matched["price_when_posted"])
    frac = matched["horizon_months"] / 12.0
    r_hat = frac * u
    r = np.log(matched["adj_close"] / matched["price_when_posted"])

    vol = matched["ticker"].map(_annualized_vol(ctx)).fillna(_VOL_FLOOR).clip(lower=_VOL_FLOOR)
    denom = vol * np.sqrt(frac) + _VOL_FLOOR
    accuracy = np.exp(-(r - r_hat).abs() / denom)
    direction = (np.sign(r) == np.sign(u)).astype(float)
    matched["q"] = _SKILL_DIRECTION_WEIGHT * direction + (1 - _SKILL_DIRECTION_WEIGHT) * accuracy

    age_days = (cutoff_ts - matched["checkpoint_date"]).dt.days.clip(lower=0)
    matched["w"] = np.power(2.0, -age_days / _SKILL_HALF_LIFE_DAYS)
    df = matched

    global_mean = float((df["w"] * df["q"]).sum() / df["w"].sum())

    def _firm_score(g: pd.DataFrame) -> float:
        w_sum = g["w"].sum()
        if w_sum <= 0:
            return global_mean
        q_bar = float((g["w"] * g["q"]).sum() / w_sum)
        n_eff = float(w_sum ** 2 / (g["w"] ** 2).sum())
        shrink = n_eff / (n_eff + _SKILL_SHRINK_K)
        return shrink * q_bar + (1 - shrink) * global_mean

    return df.groupby("analyst_company")[["w", "q"]].apply(_firm_score).to_dict()


def _firm_skill_weighted_upside(ctx: DataContext, window_days: int) -> pd.Series:
    """Trailing-window consensus target upside, weighted by each event's
    issuing firm's :func:`_firm_skill_scores` (neutral for unscored firms).

    A track-record-aware alternative to the incumbent ``pt_target_upside_30d``,
    which treats every analyst/firm identically.
    """
    skill = _firm_skill_scores(ctx)
    end = _safe_cutoff(ctx)
    start = (pd.Timestamp(end) - pd.Timedelta(days=window_days)).date().isoformat()
    df = ctx.db.query_df(
        f"SELECT ticker, analyst_company, {PT_TARGET_SQL} AS price_target, price_when_posted "
        "FROM analyst_price_target_events "
        "WHERE published_date > ? AND published_date <= ? "
        f"AND {_PT_EVENT_FILTER}",
        (start, end),
    )
    empty = pd.Series(dtype=float, index=ctx.universe)
    if df.empty:
        return empty
    df["upside"] = df["price_target"] / df["price_when_posted"] - 1.0
    df["weight"] = df["analyst_company"].map(skill).fillna(_NEUTRAL_SKILL)

    def _weighted(g: pd.DataFrame) -> float:
        w = g["weight"].sum()
        return float((g["upside"] * g["weight"]).sum() / w) if w > 0 else np.nan

    return df.groupby("ticker").apply(_weighted).reindex(ctx.universe)


def _target_dispersion(ctx: DataContext, window_days: int) -> pd.Series:
    """Cross-analyst disagreement: stdev of trailing-window price targets,
    normalized by the mean price-when-posted. NaN with fewer than 2 events.
    """
    end = _safe_cutoff(ctx)
    start = (pd.Timestamp(end) - pd.Timedelta(days=window_days)).date().isoformat()
    df = ctx.db.query_df(
        f"SELECT ticker, {PT_TARGET_SQL} AS price_target, price_when_posted "
        "FROM analyst_price_target_events "
        "WHERE published_date > ? AND published_date <= ? "
        f"AND {_PT_EVENT_FILTER}",
        (start, end),
    )
    empty = pd.Series(dtype=float, index=ctx.universe)
    if df.empty:
        return empty

    def _disp(g: pd.DataFrame) -> float:
        if len(g) < 2:
            return np.nan
        return float(g["price_target"].std() / g["price_when_posted"].mean())

    return df.groupby("ticker").apply(_disp).reindex(ctx.universe)


def _decile_upside(ctx: DataContext, window_days: int) -> tuple[pd.Series, pd.Series]:
    """Only the most-bullish and only the most-bearish 10% of a ticker's
    trailing-window analyst targets, each averaged separately (vs. the
    incumbent ``pt_target_upside_30d``, which averages all of them together).

    Coverage caveat: median analysts-per-ticker in a 30d window is only ~2-3,
    so "10%" rounds up to a single analyst for most names — this mostly reads
    as "the single most/least bullish call" rather than a real decile except
    for the handful of heavily-covered names.
    """
    end = _safe_cutoff(ctx)
    start = (pd.Timestamp(end) - pd.Timedelta(days=window_days)).date().isoformat()
    df = ctx.db.query_df(
        f"SELECT ticker, {PT_TARGET_SQL} AS price_target, price_when_posted "
        "FROM analyst_price_target_events "
        "WHERE published_date > ? AND published_date <= ? "
        f"AND {_PT_EVENT_FILTER}",
        (start, end),
    )
    empty = pd.Series(dtype=float, index=ctx.universe)
    if df.empty:
        return empty, empty.copy()
    df["upside"] = df["price_target"] / df["price_when_posted"] - 1.0

    def _bull(g: pd.DataFrame) -> float:
        n = max(1, int(np.ceil(len(g) * 0.10)))
        return float(g["upside"].nlargest(n).mean())

    def _bear(g: pd.DataFrame) -> float:
        n = max(1, int(np.ceil(len(g) * 0.10)))
        return float(g["upside"].nsmallest(n).mean())

    grp = df.groupby("ticker")
    return grp.apply(_bull).reindex(ctx.universe), grp.apply(_bear).reindex(ctx.universe)


def _wins_ctx(ctx: DataContext, series: pd.Series) -> pd.Series:
    """Sector-relative winsorize at [5, 95]% — see ``library._wins``."""
    from .library import _wins  # late import — avoids the library.py/library_flow.py cycle
    return _wins(series, ctx.sectors())


# --- 2026-08-03 confirmation/magnitude candidates (analyst deep-dive follow-up).
# Parameters fixed a priori from output/analyst_deep_dive/ BEFORE this battery:
# the deep-dive's economic-bucket table (buckets chosen before outcomes were
# seen) showed 12M sector-adjusted returns turn positive above ~+10% implied
# upside and are strongest above +20% (the "J-shape"); the 90d/30d-half-life
# window family matches the existing candidates. Do not tune these here.
_RECO_WINDOW_DAYS = 90
_RECO_HALF_LIFE_DAYS = 30.0
_HIGH_UPSIDE_THRESHOLD = 0.20   # J-shape rising region starts here
_UPSIDE_TROUGH = 0.10           # J-shape trough: upside below this carries no reward


def _reco_confirmation_signals(ctx: DataContext) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Latest-call-per-firm confirmation signals over the trailing 90d.

    Returns three per-ticker Series (each one vote per covering firm — a
    single loud firm cannot dominate through repeated events):

    * ``agreement``          — |recency-weighted mean sign of implied upside|:
                               do the covering firms agree on direction?
    * ``raise_and_bullish``  — share of firms whose latest call BOTH raises
                               their own prior target (≤365d old) AND sits
                               above spot: a revealed change of mind, not a
                               standing opinion.
    * ``high_upside_breadth``— share of firms whose latest call implies more
                               than ``_HIGH_UPSIDE_THRESHOLD`` upside: breadth
                               of conviction in the J-shape's rising region,
                               robust to any single outlier target.
    """
    end = _safe_cutoff(ctx)
    df = ctx.db.query_df(
        f"SELECT ticker, analyst_company, published_date, "
        f"{PT_TARGET_SQL} AS price_target, price_when_posted "
        "FROM analyst_price_target_events "
        f"WHERE published_date <= ? AND {_PT_EVENT_FILTER}",
        (end,),
    )
    empty = pd.Series(dtype=float, index=ctx.universe)
    if df.empty:
        return empty, empty.copy(), empty.copy()
    df["published_date"] = pd.to_datetime(df["published_date"], utc=True).dt.tz_localize(None)
    df = (df.sort_values(["ticker", "analyst_company", "published_date"])
            .drop_duplicates(["ticker", "analyst_company", "published_date"], keep="last"))
    grp = df.groupby(["ticker", "analyst_company"])
    df["prev_target"] = grp["price_target"].shift(1)
    df["prev_date"] = grp["published_date"].shift(1)
    prior_age = (df["published_date"] - df["prev_date"]).dt.days
    df["is_raise"] = np.where(
        df["prev_target"].notna() & (prior_age <= 365),
        (df["price_target"] > df["prev_target"]).astype(float), np.nan)
    df["upside"] = df["price_target"] / df["price_when_posted"] - 1.0

    cutoff_ts = pd.Timestamp(end)
    win = df[df["published_date"] > cutoff_ts - pd.Timedelta(days=_RECO_WINDOW_DAYS)]
    if win.empty:
        return empty, empty.copy(), empty.copy()
    latest = (win.sort_values("published_date")
                 .drop_duplicates(["ticker", "analyst_company"], keep="last")).copy()
    latest["age"] = (cutoff_ts - latest["published_date"]).dt.days.clip(lower=0)
    latest["w"] = np.power(2.0, -latest["age"] / _RECO_HALF_LIFE_DAYS)

    g = latest.groupby("ticker")
    agreement = g.apply(
        lambda x: float(abs((np.sign(x["upside"]) * x["w"]).sum()) / x["w"].sum()),
        include_groups=False)
    high_breadth = g.apply(
        lambda x: float((x["upside"] > _HIGH_UPSIDE_THRESHOLD).mean()),
        include_groups=False)
    both = latest[latest["is_raise"].notna()]
    if both.empty:
        raise_bull = pd.Series(dtype=float)
    else:
        raise_bull = both.groupby("ticker").apply(
            lambda x: float(((x["is_raise"] > 0) & (x["upside"] > 0)).mean()),
            include_groups=False)
    return (agreement.reindex(ctx.universe),
            raise_bull.reindex(ctx.universe),
            high_breadth.reindex(ctx.universe))


# --- 2026-08-03 rating-selectivity candidates (analyst_grade_events, FMP
# /stable/grades — individual rating actions WITH firm identity, history to
# ~2012). Parameters mirror the existing candidate conventions and are fixed
# a priori: 90d window, 2y-half-life firm baselines, shrinkage k=25.
_GRADE_NUMERIC = {
    # +2 strong conviction        +1 bullish                    0 neutral
    "Strong Buy": 2, "Conviction Buy": 2, "Top Pick": 2,
    "Buy": 1, "Overweight": 1, "Outperform": 1, "Positive": 1,
    "Market Outperform": 1, "Sector Outperform": 1, "Long Term Buy": 1,
    "Accumulate": 1,
    "Neutral": 0, "Equal Weight": 0, "Hold": 0, "Market Perform": 0,
    "Sector Perform": 0, "In Line": 0, "Perform": 0, "Sector Weight": 0,
    "Peer Perform": 0, "Mixed": 0,
    "Underweight": -1, "Underperform": -1, "Reduce": -1, "Negative": -1,
    "Sell": -2, "Strong Sell": -2,
}
_GRADE_BASELINE_HALF_LIFE = 730   # days
_GRADE_SHRINK_K = 25


def _grade_action_signals(ctx: DataContext, window_days: int = 90
                          ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Three signals from individual rating actions (one vote per firm).

    * ``action_net``      — (#upgrades − #downgrades) / #actions over the
                            trailing window: the true event-level NetRevision
                            the aggregate-count tables could only proxy.
    * ``rating_surprise`` — mean over covering firms' latest actions of
                            (numeric grade − that firm's PIT baseline mean
                            grade): a Buy from a firm that rates everything
                            Buy counts ~0; a Buy from a stingy firm counts a
                            lot. Baselines are expanding (strictly < cutoff),
                            recency-weighted, shrunk toward the global mean.
    * ``selective_bull``  — selectivity-weighted bullish share: each firm's
                            latest-stance bullish flag weighted by
                            (1 − firm's PIT bullish share).
    """
    end = _safe_cutoff(ctx)
    cutoff_ts = pd.Timestamp(end)
    df = ctx.db.query_df(
        "SELECT ticker, date, grading_company, new_grade, action "
        "FROM analyst_grade_events WHERE date <= ?",
        (end,),
    )
    empty = pd.Series(dtype=float, index=ctx.universe)
    if df.empty:
        return empty, empty.copy(), empty.copy()
    df["numeric"] = df["new_grade"].map(_GRADE_NUMERIC)
    df = df.dropna(subset=["numeric", "grading_company"])
    dates = pd.to_datetime(df["date"])

    # PIT firm baselines from actions strictly before the window start (so the
    # window's own actions never inform the baseline they're judged against).
    win_start = cutoff_ts - pd.Timedelta(days=window_days)
    hist = df[dates < win_start]
    if hist.empty:
        return empty, empty.copy(), empty.copy()
    age = (cutoff_ts - dates[dates < win_start]).dt.days.values
    w = np.power(2.0, -age / _GRADE_BASELINE_HALF_LIFE)
    hw = pd.DataFrame({"firm": hist["grading_company"].values,
                       "num": hist["numeric"].values,
                       "bull": (hist["numeric"] > 0).astype(float).values,
                       "w": w})
    g = hw.groupby("firm")
    wsum = g["w"].sum()
    n_eff = wsum ** 2 / g.apply(lambda x: (x["w"] ** 2).sum(), include_groups=False)
    shrink = n_eff / (n_eff + _GRADE_SHRINK_K)
    global_num = float((hw["num"] * hw["w"]).sum() / hw["w"].sum())
    global_bull = float((hw["bull"] * hw["w"]).sum() / hw["w"].sum())
    firm_num = shrink * (g.apply(lambda x: (x["num"] * x["w"]).sum(), include_groups=False)
                         / wsum) + (1 - shrink) * global_num
    firm_bull = shrink * (g.apply(lambda x: (x["bull"] * x["w"]).sum(), include_groups=False)
                          / wsum) + (1 - shrink) * global_bull

    win = df[(dates > win_start) & (dates <= cutoff_ts)].copy()
    if win.empty:
        return empty, empty.copy(), empty.copy()

    # 1. action_net over ALL window actions
    acts = win.groupby("ticker")["action"].agg(
        lambda a: (float((a == "upgrade").sum()) - float((a == "downgrade").sum())) / len(a))
    action_net = acts

    # 2/3. latest action per (ticker, firm)
    latest = (win.sort_values("date")
                 .drop_duplicates(["ticker", "grading_company"], keep="last")).copy()
    latest["baseline"] = latest["grading_company"].map(firm_num).fillna(global_num)
    latest["surprise"] = latest["numeric"] - latest["baseline"]
    latest["selectivity"] = 1.0 - latest["grading_company"].map(firm_bull).fillna(global_bull)
    latest["bullish"] = (latest["numeric"] > 0).astype(float)
    gl = latest.groupby("ticker")
    rating_surprise = gl["surprise"].mean()
    selective_bull = gl.apply(
        lambda x: float((x["bullish"] * x["selectivity"]).sum() / x["selectivity"].sum())
        if x["selectivity"].sum() > 0 else np.nan, include_groups=False)

    return (action_net.reindex(ctx.universe),
            rating_surprise.reindex(ctx.universe),
            selective_bull.reindex(ctx.universe))


def _upside_above_trough(ctx: DataContext, window_days: int = 30) -> pd.Series:
    """Mean of ``max(upside − trough, 0)`` over trailing-window target events.

    The linear mean (``pt_target_upside_30d``) fails because the 0–10% "faint
    praise" band predicts *negative* relative returns and dominates the
    average; this keeps only each target's distance into the J-shape's rising
    region, so lukewarm targets contribute zero instead of poisoning the mean.
    """
    end = _safe_cutoff(ctx)
    start = (pd.Timestamp(end) - pd.Timedelta(days=window_days)).date().isoformat()
    df = ctx.db.query_df(
        f"SELECT ticker, {PT_TARGET_SQL} AS price_target, price_when_posted "
        "FROM analyst_price_target_events "
        "WHERE published_date > ? AND published_date <= ? "
        f"AND {_PT_EVENT_FILTER}",
        (start, end),
    )
    if df.empty:
        return pd.Series(dtype=float, index=ctx.universe)
    upside = df["price_target"] / df["price_when_posted"] - 1.0
    df["above"] = (upside - _UPSIDE_TROUGH).clip(lower=0.0)
    return df.groupby("ticker")["above"].mean().reindex(ctx.universe)


_REVISION_DECAY_HALF_LIFE = 30  # days; used by every *_decay_60d / breadth candidate below


def _revision_surprise_events(ctx: DataContext, window_days: int,
                              max_prior_age_days: int = 365) -> pd.DataFrame:
    """Per-event active revision surprise (ARS) inside the trailing window.

    ``ARS_i = ln(T_i/T_{i-1}) - ln(P_i/P_{i-1})`` for consecutive same-firm,
    same-ticker price-target events — the part of a target change *not*
    explained by the stock's own move between the two quotes (beta fixed at
    1; a fitted PIT-safe beta is a possible follow-up, not implemented here).
    Also returns the raw (unadjusted) target revision ``ln(T_i/T_{i-1})``.

    Quality controls: reiterated (unchanged) targets are dropped — zero
    information, and would let a plain price move masquerade as a "revision";
    same-firm/same-day duplicate events collapse to the last one; a pair only
    counts if the prior target is <= ``max_prior_age_days`` old (otherwise the
    baseline is too stale to be a meaningful reference point). ``skill`` is
    each event's issuing firm's :func:`_firm_skill_scores`.
    """
    end = _safe_cutoff(ctx)
    df = ctx.db.query_df(
        f"SELECT ticker, analyst_company, published_date, "
        f"{PT_TARGET_SQL} AS price_target, price_when_posted "
        "FROM analyst_price_target_events "
        f"WHERE published_date <= ? AND {_PT_EVENT_FILTER}",
        (end,),
    )
    if df.empty:
        return pd.DataFrame()
    df["published_date"] = pd.to_datetime(df["published_date"], utc=True).dt.tz_localize(None)
    df = (df.sort_values(["ticker", "analyst_company", "published_date"])
            .drop_duplicates(["ticker", "analyst_company", "published_date"], keep="last"))
    grp = df.groupby(["ticker", "analyst_company"])
    df["prev_target"] = grp["price_target"].shift(1)
    df["prev_pw"] = grp["price_when_posted"].shift(1)
    df["prev_date"] = grp["published_date"].shift(1)
    df = df.dropna(subset=["prev_target", "prev_pw", "prev_date"]).copy()
    if df.empty:
        return pd.DataFrame()

    prior_age = (df["published_date"] - df["prev_date"]).dt.days
    df = df[(prior_age > 0) & (prior_age <= max_prior_age_days)]
    df = df[df["price_target"] != df["prev_target"]]
    if df.empty:
        return pd.DataFrame()

    cutoff_ts = pd.Timestamp(end)
    window_start = cutoff_ts - pd.Timedelta(days=window_days)
    df = df[(df["published_date"] > window_start) & (df["published_date"] <= cutoff_ts)]
    if df.empty:
        return pd.DataFrame()

    df["raw_revision"] = np.log(df["price_target"] / df["prev_target"])
    price_move = np.log(df["price_when_posted"] / df["prev_pw"])
    df["ars"] = df["raw_revision"] - price_move
    df["age_days"] = (cutoff_ts - df["published_date"]).dt.days.clip(lower=0)
    skill = _firm_skill_scores(ctx)
    df["skill"] = df["analyst_company"].map(skill).fillna(_NEUTRAL_SKILL)
    return df[["ticker", "analyst_company", "raw_revision", "ars", "age_days", "skill"]]


def _agg_weighted(ctx: DataContext, events: pd.DataFrame, value_col: str, weight_fn) -> pd.Series:
    """Ticker-level weighted mean of ``events[value_col]`` using ``weight_fn(events)``."""
    empty = pd.Series(dtype=float, index=ctx.universe)
    if events.empty:
        return empty
    w = weight_fn(events)
    tmp = events.assign(_w=w, _wv=w * events[value_col])

    def _agg(g: pd.DataFrame) -> float:
        w_sum = g["_w"].sum()
        return float(g["_wv"].sum() / w_sum) if w_sum > 0 else np.nan

    return tmp.groupby("ticker").apply(_agg).reindex(ctx.universe)


def _revision_breadth_confirmed(ctx: DataContext, events: pd.DataFrame, half_life: int) -> pd.Series:
    """Skill x decay-weighted mean ARS, scaled by cross-firm agreement and breadth.

    ``F = combined_ARS * agreement * log(1 + n_firms)``, where ``agreement``
    is the weighted share of events pointing the same direction as the
    combined score (0 = evenly split, 1 = unanimous) — a consensus surprise
    from several independent, skilled firms should count for more than the
    same average magnitude produced by one or two.
    """
    empty = pd.Series(dtype=float, index=ctx.universe)
    if events.empty:
        return empty
    w = events["skill"] * np.power(2.0, -events["age_days"] / half_life)
    tmp = events.assign(_w=w, _wv=w * events["ars"], _wsign=w * np.sign(events["ars"]))

    def _agg(g: pd.DataFrame) -> float:
        w_sum = g["_w"].sum()
        if w_sum <= 0:
            return np.nan
        combined = g["_wv"].sum() / w_sum
        agreement = abs(g["_wsign"].sum()) / w_sum
        n_firms = g["analyst_company"].nunique()
        return float(combined * agreement * np.log1p(n_firms))

    return tmp.groupby("ticker").apply(_agg).reindex(ctx.universe)


# =============================================================================
# Institutional (13F flows)
# =============================================================================


def build_institutional(ctx: DataContext) -> list:
    inst = ctx.institutional()

    fund_count = col(inst, "fund_count")
    net_share_change = col(inst, "net_share_change")
    new_positions = col(inst, "new_positions")
    position_increases = col(inst, "position_increases")

    # >=2 funds opened — preserves NaN so absent names stay neutral instead of
    # collapsing to a hard zero.
    multi_fund_open = new_positions.where(new_positions.isna(),
                                          (new_positions >= 2).astype(float))

    # Whole-market ownership summary — richer signal set unused by production.
    own = _ownership_latest(ctx)
    ownership_pct_change = col(own, "ownership_percent_change")
    investors_holding_change = col(own, "investors_holding_change")
    put_call_ratio = col(own, "put_call_ratio")
    ownership_pct_level = col(own, "ownership_percent")

    # Net-flow dollars proxy: share change × price. Sign kept: buys > 0.
    prices = ctx.prices()
    net_flow_dollars = col(own, "shares_change") * prices

    return [
        _Candidate("inst_fund_count", fund_count, True, "institutional", "breadth"),
        _Candidate("inst_net_share_change", net_share_change, True, "institutional", "flow"),
        _Candidate("inst_new_positions", new_positions, True, "institutional", "breadth"),
        _Candidate("inst_multi_fund_open", multi_fund_open, True, "institutional", "confirmation"),
        _Candidate("inst_high_conviction", position_increases, True, "institutional", "conviction"),
        _Candidate("inst_ownership_pct_change", ownership_pct_change, True, "institutional", "flow"),
        _Candidate("inst_investors_holding_change", investors_holding_change, True, "institutional", "breadth"),
        _Candidate("inst_net_flow_dollars", net_flow_dollars, True, "institutional", "flow"),
        _Candidate("inst_put_call_ratio_inv", put_call_ratio, False, "institutional", "sentiment"),
        _Candidate("inst_concentration_pct", ownership_pct_level, True, "institutional", "sponsorship"),
    ]


def _ownership_latest(ctx: DataContext) -> pd.DataFrame:
    """Latest institutional_ownership_summary row per ticker available on/before as-of.

    The summary is keyed by quarter-end report_date, but the underlying 13Fs
    are not public until up to 45 days later — in PIT mode admit a quarter only
    once that deadline has passed (mirrors ``DataContext._institutional_pit``).
    """
    avail = _pit_avail(ctx, "report_date", f"+{ctx.lag_quarterly_days} day")
    sql = (
        "WITH latest AS (SELECT ticker, MAX(report_date) d "
        f"FROM institutional_ownership_summary WHERE {avail} GROUP BY ticker) "
        "SELECT o.* FROM institutional_ownership_summary o "
        "JOIN latest l ON o.ticker=l.ticker AND o.report_date=l.d"
    )
    df = ctx.db.query_df(sql, (ctx.cutoff,))
    if df.empty:
        return pd.DataFrame(index=ctx.universe)
    df = df.drop_duplicates("ticker", keep="last").set_index("ticker")
    return df.reindex(ctx.universe)


# =============================================================================
# Insider (Form-4)
# =============================================================================

_CEO_CFO_TERMS = ("CEO", "CHIEF EXECUTIVE", "CFO", "CHIEF FINANCIAL")
_OFFICER_TERMS = _CEO_CFO_TERMS + ("PRESIDENT", "COO", "CHIEF OPERATING")


def build_insider(ctx: DataContext) -> list:
    long_win = _insider_window_rows(ctx, days=180)
    short_win = _insider_window_rows(ctx, days=90)

    long_agg = _insider_agg(long_win, ctx)
    short_agg = _insider_agg(short_win, ctx)

    return [
        _Candidate("ins_net_dollar_flow", long_agg["net_flow"], True, "insider", "flow"),
        _Candidate("ins_buy_sell_ratio", long_agg["buy_sell_ratio"], True, "insider", "balance"),
        _Candidate("ins_high_conviction_buy", long_agg["high_conviction"], True, "insider", "conviction"),
        _Candidate("ins_purchase_frequency_180d", long_agg["purchase_freq"], True, "insider", "activity"),
        _Candidate("ins_cluster_buyers_180d", long_agg["cluster_buyers"], True, "insider", "conviction"),
        _Candidate("ins_ceo_cfo_buy_dollars", long_agg["ceo_cfo_buy"], True, "insider", "conviction"),
        _Candidate("ins_sell_pressure_inv", long_agg["sell_to_mcap"], False, "insider", "pressure"),
        _Candidate("ins_net_flow_90d", short_agg["net_flow"], True, "insider", "flow"),
        _Candidate("ins_officer_buy_ratio", long_agg["officer_buy_ratio"], True, "insider", "conviction"),
        _Candidate("ins_no_selling_flag", long_agg["no_selling"], True, "insider", "balance"),
        # Buy-only variants (2026-07-05) — the insider forensic audit showed the
        # net signal is sell-dominated; these isolate the *buy* side (gross buy
        # dollars, buy transaction count, and the large-buy ≥$1M dollars/count)
        # so the selector can consider raw purchasing pressure on its own.
        _Candidate("ins_buy_dollar_volume", long_agg["buy_dollar_vol"], True, "insider", "buy"),
        _Candidate("ins_buy_count", long_agg["buy_count"], True, "insider", "buy"),
        _Candidate("ins_large_buy_dollars", long_agg["large_buy_dollars"], True, "insider", "buy"),
        _Candidate("ins_large_buy_count", long_agg["large_buy_count"], True, "insider", "buy"),
    ]


def _insider_window_rows(ctx: DataContext, days: int) -> pd.DataFrame:
    # PIT fix (2026-07-05): the old guard `ctx.as_of > "9000-01-01"` was a
    # lexicographic test with inverted branches — every real historical as_of
    # (e.g. "2024-06-03" < "9000-01-01") fell to `else today`, so each backtest
    # rebalance read the *future* last-180d window (the look-ahead bug documented
    # in research/insider_forensics.py / factors/utils.py:378). `min(as_of, today)`
    # already caps the far-future sentinel and preserves real historical dates.
    today = pd.Timestamp.today().date().isoformat()
    upper = min(ctx.as_of, today)
    start = (pd.Timestamp(upper) - pd.Timedelta(days=days)).date().isoformat()
    start = max(start, "1990-01-01")
    df = ctx.db.query_df(
        "SELECT ticker, insider_name, insider_title, transaction_code, shares, "
        "price, value, transaction_date FROM insider_transactions "
        "WHERE transaction_date <= ? AND transaction_date >= ?",
        (upper, start),
    )
    return df[df["ticker"].isin(ctx.universe)] if not df.empty else df


def _insider_agg(df: pd.DataFrame, ctx: DataContext) -> dict[str, pd.Series]:
    """Aggregate one insider window into every insider candidate at once."""
    empty = pd.Series(dtype=float, index=ctx.universe)
    keys = ("net_flow", "buy_sell_ratio", "high_conviction", "purchase_freq",
            "cluster_buyers", "ceo_cfo_buy", "sell_to_mcap", "officer_buy_ratio",
            "no_selling", "buy_dollar_vol", "buy_count", "large_buy_dollars",
            "large_buy_count")
    if df.empty:
        return {k: empty.copy() for k in keys}

    d = df.copy()
    d["code"] = d["transaction_code"].fillna("").str.upper().str.strip()
    d["value"] = pd.to_numeric(d["value"], errors="coerce").abs().fillna(0.0)
    d["title"] = d["insider_title"].fillna("").str.upper()
    is_buy = d["code"] == "P"
    is_sell = d["code"] == "S"

    # Signed dollar flow (production convention: sales weighted 0.5).
    d["signed"] = 0.0
    d.loc[is_buy, "signed"] = d.loc[is_buy, "value"]
    d.loc[is_sell, "signed"] = -0.5 * d.loc[is_sell, "value"]
    net_flow = d.groupby("ticker")["signed"].sum()

    buy_usd = d[is_buy].groupby("ticker")["value"].sum()
    sell_usd = d[is_sell].groupby("ticker")["value"].sum()

    traded = d[is_buy | is_sell]
    active_ps = pd.Index(traded["ticker"].unique())
    buy_usd_a = buy_usd.reindex(active_ps, fill_value=0.0)
    sell_usd_a = sell_usd.reindex(active_ps, fill_value=0.0)
    gross = buy_usd_a + sell_usd_a
    buy_sell_ratio = (buy_usd_a / gross.where(gross > 0)).reindex(active_ps)

    # Conviction flag (single combined 0/1).
    buys = d[is_buy]
    ceo_cfo = set(buys[buys["title"].str.contains("|".join(_CEO_CFO_TERMS), regex=True)]["ticker"])
    large = set(buys[buys["value"] >= 1_000_000]["ticker"])
    cluster = set(buys.groupby("ticker")["insider_name"].nunique()
                  .loc[lambda s: s >= 3].index)
    conv = ceo_cfo | large | cluster
    active_any = pd.Index(d["ticker"].unique())
    high_conviction = pd.Series(
        [1.0 if t in conv else 0.0 for t in active_any], index=active_any, dtype=float)

    # New: purchase frequency (unique buy days).
    purchase_freq = (buys.assign(date=buys["transaction_date"].str[:10])
                     .groupby("ticker")["date"].nunique())

    # Cluster buyers: distinct insider names among P transactions.
    cluster_buyers = buys.groupby("ticker")["insider_name"].nunique()

    # CEO/CFO buy dollars only.
    ceo_cfo_buys = buys[buys["title"].str.contains("|".join(_CEO_CFO_TERMS), regex=True)]
    ceo_cfo_buy = ceo_cfo_buys.groupby("ticker")["value"].sum()

    # Sell dollars scaled by market cap — pressure signal (raw is monotone
    # increasing bad; the direction flag inverts it downstream).
    mcap = ctx.market_cap()
    sell_to_mcap = (sell_usd / mcap.replace(0, np.nan)).reindex(ctx.universe)

    # Officer buy $ / total buy $. NaN when no buys.
    officer_buys = buys[buys["title"].str.contains("|".join(_OFFICER_TERMS), regex=True)]
    officer_dollars = officer_buys.groupby("ticker")["value"].sum()
    officer_buy_ratio = (officer_dollars / buy_usd).reindex(buy_usd.index)

    # "No selling" flag — 1 when any activity but zero sell $, else 0.
    no_selling = pd.Series(0.0, index=active_any, dtype=float)
    sold = set(sell_usd.index)
    no_selling.loc[[t for t in active_any if t not in sold]] = 1.0

    # Buy-only variants — gross buy pressure isolated from the sell side. Defined
    # over active_any (any insider activity), 0 when the name had no buys, so a
    # name with activity-but-no-buys scores the low end and names with no insider
    # data at all stay missing → neutral 50. Large = single trade ≥ $1M.
    buy_dollar_vol = buy_usd.reindex(active_any, fill_value=0.0)
    buy_count = buys.groupby("ticker").size().reindex(active_any, fill_value=0).astype(float)
    large_buys = buys[buys["value"] >= 1_000_000]
    large_buy_dollars = large_buys.groupby("ticker")["value"].sum().reindex(active_any, fill_value=0.0)
    large_buy_count = large_buys.groupby("ticker").size().reindex(active_any, fill_value=0).astype(float)

    def _idx(s: pd.Series) -> pd.Series:
        return s.reindex(ctx.universe)

    return {
        "net_flow": _idx(net_flow),
        "buy_sell_ratio": _idx(buy_sell_ratio),
        "high_conviction": _idx(high_conviction),
        "purchase_freq": _idx(purchase_freq),
        "cluster_buyers": _idx(cluster_buyers),
        "ceo_cfo_buy": _idx(ceo_cfo_buy),
        "sell_to_mcap": _idx(sell_to_mcap),
        "officer_buy_ratio": _idx(officer_buy_ratio),
        "no_selling": _idx(no_selling),
        "buy_dollar_vol": _idx(buy_dollar_vol),
        "buy_count": _idx(buy_count),
        "large_buy_dollars": _idx(large_buy_dollars),
        "large_buy_count": _idx(large_buy_count),
    }


# =============================================================================
# Short interest
# =============================================================================


def build_short(ctx: DataContext) -> list:
    si = ctx.short_interest()
    pct_float = col(si, "short_percent_of_float")
    dtc = col(si, "days_to_cover")
    change = col(si, "short_interest_change")

    hist_pct, trend_90d, change_accel, covering = _short_history_features(ctx, days=252)
    hist_pct_pct = _own_history_percentile(hist_pct, pct_float)

    dollars_to_mcap = _short_dollars_to_mcap(ctx, si)

    # Squeeze setup: 1 when the raw squeeze flag is set OR pct_float in top decile.
    squeeze_flag = _squeeze_setup(ctx, pct_float)

    return [
        _Candidate("si_short_pct_float", pct_float, False, "short", "level"),
        _Candidate("si_days_to_cover", dtc, False, "short", "level"),
        _Candidate("si_short_interest_change", change, False, "short", "flow"),
        _Candidate("si_short_pct_float_pctile_252d", hist_pct_pct, False, "short", "context"),
        _Candidate("si_short_trend_90d", trend_90d, False, "short", "flow"),
        _Candidate("si_short_change_accel", change_accel, False, "short", "flow"),
        _Candidate("si_short_covering_signal", covering, True, "short", "reversal"),
        _Candidate("si_squeeze_setup_flag", squeeze_flag, True, "short", "setup"),
        _Candidate("si_short_dollars_to_mcap", dollars_to_mcap, False, "short", "level"),
    ]


def _short_history_features(ctx: DataContext, days: int) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    end = _safe_cutoff(ctx)
    start = (pd.Timestamp(end) - pd.Timedelta(days=days)).date().isoformat()
    # ``date`` is the FINRA settlement date; dissemination is ~2 weeks later.
    df = ctx.db.query_df(
        "SELECT ticker, date, short_percent_of_float, short_percent_float_change "
        f"FROM short_interest WHERE "
        f"{_pit_avail(ctx, 'date', f'+{ctx.lag_short_interest_days} day')} "
        "AND date >= ?", (end, start)
    )
    empty = pd.Series(dtype=float, index=ctx.universe)
    if df.empty:
        return empty, empty.copy(), empty.copy(), empty.copy()
    df = df.sort_values(["ticker", "date"])
    ninety_start = (pd.Timestamp(end) - pd.Timedelta(days=90)).date().isoformat()

    def _slope(g: pd.DataFrame) -> float:
        g90 = g[g["date"] >= ninety_start].dropna(subset=["short_percent_of_float"])
        if len(g90) < 3:
            return float("nan")
        y = g90["short_percent_of_float"].astype(float).values
        x = np.arange(len(y), dtype=float)
        return float(np.polyfit(x, y, 1)[0])

    def _accel(g: pd.DataFrame) -> float:
        changes = g["short_percent_float_change"].dropna().values
        if len(changes) < 2:
            return float("nan")
        return float(changes[-1] - changes[-2])

    def _cover(g: pd.DataFrame) -> float:
        latest = g["short_percent_float_change"].dropna()
        return float(-latest.iloc[-1]) if not latest.empty else float("nan")

    def _own_pct(g: pd.DataFrame) -> float:
        vals = g["short_percent_of_float"].dropna().values
        if len(vals) < 4:
            return float("nan")
        latest = float(vals[-1])
        # Percentile of the latest observation within the trailing distribution.
        return float((vals <= latest).mean())

    trend_90d = df.groupby("ticker").apply(_slope)
    change_accel = df.groupby("ticker").apply(_accel)
    covering = df.groupby("ticker").apply(_cover)
    own_pct = df.groupby("ticker").apply(_own_pct)

    return (own_pct.reindex(ctx.universe), trend_90d.reindex(ctx.universe),
            change_accel.reindex(ctx.universe), covering.reindex(ctx.universe))


def _own_history_percentile(own_pct: pd.Series, latest: pd.Series) -> pd.Series:
    """The own-history percentile of the latest short_percent_of_float — a real
    contextualization instead of the raw level. When there are too few history
    points, falls back to the raw pct_float so the candidate still ranks.
    """
    return own_pct.fillna(latest)


def _short_dollars_to_mcap(ctx: DataContext, si: pd.DataFrame) -> pd.Series:
    if si.empty or "shares_short" not in si.columns:
        return pd.Series(dtype=float, index=ctx.universe)
    shares_short = col(si, "shares_short")
    px = ctx.prices()
    mcap = ctx.market_cap().replace(0, np.nan)
    return (shares_short * px / mcap).reindex(ctx.universe)


def _squeeze_setup(ctx: DataContext, pct_float: pd.Series) -> pd.Series:
    """1 when short pressure is heavy (top decile of pct_float OR squeeze_risk_flag)."""
    threshold = pct_float.dropna().quantile(0.90) if pct_float.notna().any() else np.nan
    heavy_pct = pct_float >= (threshold if pd.notna(threshold) else 1e9)
    si = ctx.short_interest()
    flag = col(si, "short_squeeze_risk_flag").fillna(0) > 0
    setup = (heavy_pct | flag).astype(float)
    setup = setup.where(pct_float.notna())
    return setup.reindex(ctx.universe)


__all__ = ["build_revisions", "build_institutional", "build_insider", "build_short"]
