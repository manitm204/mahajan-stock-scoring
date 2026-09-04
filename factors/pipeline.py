"""Reusable scoring entry point for point-in-time (backtest) consumption.

``run_scoring.py`` wires the Layer 2 pipeline together for the CLI (it prints a
summary and persists to the DB). Backtesting needs the *same* computation but as
a plain function it can call once per rebalance date and get a ranked frame back
— no printing, no DB writes. :func:`score_universe` is that function.

It reuses every existing piece (``ALL_FACTORS``, :func:`score_factor`,
:func:`build_composite`, :func:`resolve_weights`); the only behavioural addition
is ``reporting_lag=True`` by default, which makes the underlying
:class:`DataContext` admit fundamentals/13F only once they would realistically
have been filed. That is the difference between a defensible backtest and one
that silently uses tomorrow's financials.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from data.config import Config, load_config
from data.db import Database

from . import ALL_FACTORS, ALL_FACTORS_V4, SELECTED_SUBS, score_factor
from .composite import build_composite
from .regime_weights import resolve_weights
from .utils import DataContext

# Named factor set → wrapper-factor registry override. When active, the
# ``ALL_FACTORS_V4`` registry replaces production ``ALL_FACTORS`` for the run so
# each parent emits only its selected subs (research library) instead of the
# hand-coded production subs.
_FACTOR_REGISTRY_BY_SET: dict[str, list] = {
    "parent_selection_v4": ALL_FACTORS_V4,
}
# Intra-parent weight map baked into the registry. Consulted when a sub-factor
# set does not enumerate weights explicitly, so the parent-selection weights
# flow through to :func:`score_factor` without duplication in config.
_WEIGHT_MAP_BY_SET: dict[str, dict[str, dict[str, float]]] = {
    "parent_selection_v4": SELECTED_SUBS,
}


@dataclass
class ScoredUniverse:
    """Point-in-time scoring result for one as-of date.

    ``frame`` is indexed by ticker, sorted best-composite first, and carries the
    parent factor scores, ``composite_raw``/``composite_score``, ``sector_rank``,
    ``long_short_flag`` plus two global helpers used by basket selection:

    * ``rank``             — 1 = strongest long candidate in the whole universe.
    * ``rank_from_bottom`` — 1 = strongest short candidate.

    Ranks break ties on ``composite_score`` (a sector-relative percentile, so it
    saturates at 100/0) with the continuous ``composite_raw`` so the top/bottom
    baskets are deterministic instead of arbitrary among percentile ties.
    """

    as_of: str
    frame: pd.DataFrame
    weights: dict[str, float]
    regime: str
    applied: str
    vix: float | None
    pit_notes: list[str] = field(default_factory=list)


def score_universe(
    as_of_date: str | None = None,
    *,
    db: Database | None = None,
    cfg: Config | None = None,
    reporting_lag: bool = True,
    use_regime: bool = False,
    min_obs: int | None = None,
    weights_override: dict[str, float] | None = None,
    sub_factor_set: str | None = None,
) -> ScoredUniverse:
    """Score the full universe as of ``as_of_date`` (default: latest data).

    Cross-sectional ranking needs the whole universe, so this always scores every
    name. ``reporting_lag`` defaults to True because the primary caller is the
    backtester; pass False to reproduce live ``run_scoring`` semantics.

    ``weights_override`` replaces the regime/static weight vector for this call —
    the parent factor scores are independent of the weights, so a caller (e.g. the
    walk-forward IC weighter) can blend them differently without re-scoring. The
    override is used verbatim; :func:`build_composite` renormalizes it.

    ``sub_factor_set`` selects a named filtered set of sub-factors from
    ``cfg.factor_sets.<name>`` (e.g. ``"v2"``). ``None`` keeps the full V1
    behaviour (every sub-factor a parent emits feeds the parent score).
    Unknown / missing parents in the set fall back to the V1 default for that
    parent; missing or null entries mean "use all sub-factors of this parent."
    """
    cfg = cfg or load_config()
    if min_obs is None:
        min_obs = int(cfg.get("factors", "min_sector_obs", default=5))
    insider_days = int(cfg.get("factors", "insider_window_days", default=90))
    allowlist_map = _resolve_factor_set(cfg, sub_factor_set)
    # Named factor sets can swap the factor registry (V4 uses the research-
    # library builders). Falls back to production ``ALL_FACTORS`` otherwise.
    factor_registry = _FACTOR_REGISTRY_BY_SET.get(sub_factor_set or "", ALL_FACTORS)
    weight_map = _WEIGHT_MAP_BY_SET.get(sub_factor_set or "", {})

    ctx = DataContext(
        db=db,
        as_of=as_of_date,
        insider_window_days=insider_days,
        reporting_lag=reporting_lag,
    )
    try:
        results = [score_factor(f, ctx, min_obs=min_obs,
                                sub_factor_allowlist=(
                                    weight_map.get(f.key)
                                    or allowlist_map.get(f.key)))
                   for f in factor_registry]
        if weights_override is not None:
            weights = {k: float(v) for k, v in weights_override.items()}
            regime, applied, vix = "ic", "ic_walkforward", None
        else:
            decision = resolve_weights(cfg, ctx, use_regime=use_regime)
            weights, regime, applied, vix = (
                decision.weights, decision.regime, decision.applied, decision.vix)
        composite = build_composite(results, weights, ctx.sectors(), cfg, min_obs=min_obs)
        frame = _assemble(results, composite)
        return ScoredUniverse(
            as_of=ctx.as_of,
            frame=frame,
            weights=weights,
            regime=regime,
            applied=applied,
            vix=vix,
            pit_notes=list(ctx.pit_notes),
        )
    finally:
        ctx.close()


def resolve_factor_registry(set_name: str | None):
    """Return ``(registry, weight_map)`` for a named factor set.

    ``registry`` is the ordered list of ``Factor`` instances every consumer
    should score (defaults to production ``ALL_FACTORS``); ``weight_map`` is the
    ``{parent_key: {sub_name: weight}}`` intra-parent weight override baked into
    the set (defaults to empty). Both entry points that score factors
    (:func:`score_universe` and ``run_scoring.py``) resolve their registry
    through this helper so a new set stays in one place.
    """
    return (_FACTOR_REGISTRY_BY_SET.get(set_name or "", ALL_FACTORS),
            _WEIGHT_MAP_BY_SET.get(set_name or "", {}))


_FULL_SET_ALIASES = {"full", "v1", "none", "all"}


def active_sub_factor_set(cfg, cli_value: str | None) -> str | None:
    """The sub-factor set name to use: an explicit CLI value, else the
    production default in ``config.factors.sub_factor_set`` (null = full V1).

    A CLI value of ``full`` / ``v1`` / ``none`` / ``all`` is an explicit override
    back to the full V1 set (every sub-factor), which is how a run opts out of the
    production set when ``config.factors.sub_factor_set`` is non-null. Centralizes
    the precedence so scoring, the weight engine and the backtester all default to
    the same production set unless a run overrides it.
    """
    if cli_value is not None:
        return None if cli_value.strip().lower() in _FULL_SET_ALIASES else cli_value
    name = cfg.get("factors", "sub_factor_set", default=None)
    return str(name) if name else None


def _resolve_factor_set(cfg, set_name: str | None
                        ) -> "dict[str, list[str] | dict[str, float] | None]":
    """Resolve a named factor set to a {parent_key: allowlist | weight_map | None} map.

    ``None`` / unknown set name → empty dict (V1: nothing filtered). Within a
    known set a parent whose value is null or absent means "keep all of that
    parent's sub-factors" — the practical way to wire revisions / short /
    insider into V2 even though we don't enumerate them.

    A parent whose value is a **list** is an allowlist (equal-weight combine); a
    **mapping** ``{sub_name: weight}`` is an allowlist *and* an intra-parent
    weight override. :func:`factors.base.score_factor` consumes both shapes
    natively (see ``sub_factor_allowlist``).
    """
    if not set_name:
        return {}
    raw = cfg.get("factor_sets", set_name, default=None)
    if not raw:
        return {}
    out: "dict[str, list[str] | dict[str, float] | None]" = {}
    for k, v in raw.items():
        if v is None:
            out[k] = None
        elif isinstance(v, dict):
            out[k] = {str(sub): float(w) for sub, w in v.items()}
        else:
            out[k] = [str(x) for x in v]
    return out


def _assemble(results, composite: pd.DataFrame) -> pd.DataFrame:
    """Build the wide ranked frame with global long/short ranks."""
    parent_cols = [f"{r.key}_score" for r in results]
    ordered = (
        ["sector"]
        + parent_cols
        + ["composite_raw", "composite_score", "sector_rank", "long_short_flag"]
    )
    frame = composite.reindex(columns=[c for c in ordered if c in composite.columns]).copy()
    frame.index.name = "ticker"

    # Deterministic global ordering: best composite first, ties broken by the
    # continuous blend so percentile saturation (100/0) never makes the top or
    # bottom basket arbitrary.
    frame = frame.sort_values(
        ["composite_score", "composite_raw"], ascending=[False, False]
    )
    frame["rank"] = range(1, len(frame) + 1)
    frame["rank_from_bottom"] = range(len(frame), 0, -1)
    return frame
