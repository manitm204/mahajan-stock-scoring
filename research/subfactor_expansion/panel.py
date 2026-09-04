"""Candidate score panel — builds a per-rebalance frame of candidate scores.

Mirrors :mod:`research.panel` for the production factor engine, but scores the
research :class:`Candidate` objects instead of the production ``factors/``
modules. Each rebalance date gets one wide DataFrame (ticker × candidate) where
every column is a 0-100 GICS-sector-relative percentile — the same rule
production uses (:func:`factors.utils.sector_percentile`), so ICs and quintile
spreads are directly comparable.

Strictly point-in-time: each date's frame comes from a
``DataContext(as_of=date, reporting_lag=True)``.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data.config import Config
from data.db import Database
from factors.utils import DataContext, sector_percentile

from .library import Candidate, CANDIDATE_BUILDERS, iter_parents

CACHE_VERSION = 2   # v2: point-in-time universe per rebalance (members_as_of)


@dataclass
class CandidatePanel:
    """Per-rebalance scored candidates + taxonomy.

    ``raw`` is optional: kept for the distribution/sign-sanity checks, dropped
    if you only need the scored frames (saves memory over long backtests).
    """

    rebal_dates: list[str]
    scores: dict[str, pd.DataFrame]        # date -> (ticker x candidate) 0-100
    raws: dict[str, pd.DataFrame]          # date -> (ticker x candidate) raw
    candidates_by_parent: dict[str, list[str]]
    universe: list[str]
    parent_by_candidate: dict[str, str]
    higher_by_candidate: dict[str, bool]

    @property
    def all_candidates(self) -> list[str]:
        out: list[str] = []
        for subs in self.candidates_by_parent.values():
            out.extend(subs)
        return out

    def parent_of(self, name: str) -> str:
        return self.parent_by_candidate.get(name, "unknown")

    def signal_frame(self, date: str, columns: list[str]) -> pd.DataFrame:
        frame = self.scores[date]
        keep = [c for c in columns if c in frame.columns]
        return frame[keep]


def build_candidate_panel(
    db: Database,
    rebal_dates: list[str],
    cfg: Config,
    *,
    min_obs: int | None = None,
    reporting_lag: bool = True,
    insider_days: int = 180,
    verbose: bool = True,
    keep_raw: bool = True,
) -> CandidatePanel:
    """Score every candidate at each rebalance date into one panel."""
    if min_obs is None:
        min_obs = int(cfg.get("factors", "min_sector_obs", default=5))

    scores: dict[str, pd.DataFrame] = {}
    raws: dict[str, pd.DataFrame] = {}
    candidates_by_parent: dict[str, list[str]] = {}
    parent_by_candidate: dict[str, str] = {}
    higher_by_candidate: dict[str, bool] = {}
    seen: set[str] = set()                     # union of PIT members across all rebalances

    for parent in iter_parents():
        candidates_by_parent.setdefault(parent, [])

    for d in rebal_dates:
        # Point-in-time universe: only the names that were index members on ``d``
        # (includes since-departed names, excludes not-yet-added ones) — so the panel
        # is survivorship-bias free. Falls back to today's universe before feed coverage.
        members = db.members_as_of(d) or db.universe_tickers()
        seen.update(members)
        ctx = DataContext(
            db=db, as_of=d, universe=members, insider_window_days=insider_days,
            reporting_lag=reporting_lag,
        )
        try:
            frame_scores, frame_raws, meta = _score_one(ctx, min_obs=min_obs)
            scores[d] = frame_scores            # already indexed to ctx.universe (PIT members)
            if keep_raw:
                raws[d] = frame_raws
            for name, (parent, higher) in meta.items():
                if name not in parent_by_candidate:
                    parent_by_candidate[name] = parent
                    higher_by_candidate[name] = higher
                    candidates_by_parent[parent].append(name)
            if verbose:
                print(f"  scored {d}: {frame_scores.shape[1]} candidates × "
                      f"{frame_scores.shape[0]} tickers")
        finally:
            ctx.close()

    return CandidatePanel(
        rebal_dates=list(scores.keys()),
        scores=scores,
        raws=raws,
        candidates_by_parent=candidates_by_parent,
        universe=sorted(seen),
        parent_by_candidate=parent_by_candidate,
        higher_by_candidate=higher_by_candidate,
    )


def _score_one(ctx: DataContext, min_obs: int
               ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[str, bool]]]:
    """Score one rebalance date across every candidate.

    Returns (scored_frame, raw_frame, {name: (parent, higher_is_better)}).
    """
    sectors = ctx.sectors()
    score_cols: dict[str, pd.Series] = {}
    raw_cols: dict[str, pd.Series] = {}
    meta: dict[str, tuple[str, bool]] = {}
    for parent, builder in CANDIDATE_BUILDERS.items():
        for cand in _safe_build(builder, ctx):
            if cand.name in meta:
                continue
            raw = cand.raw.reindex(ctx.universe)
            raw_cols[cand.name] = raw
            score_cols[cand.name] = sector_percentile(
                raw, sectors, higher_is_better=cand.higher_is_better, min_obs=min_obs
            )
            meta[cand.name] = (parent, cand.higher_is_better)
    scores = pd.DataFrame(score_cols).reindex(ctx.universe)
    raws = pd.DataFrame(raw_cols).reindex(ctx.universe)
    return scores, raws, meta


def _safe_build(builder, ctx: DataContext) -> list[Candidate]:
    """Wrap a builder so a single-parent failure doesn't stop the whole panel."""
    from data.utils import get_logger
    try:
        return builder(ctx)
    except Exception as exc:  # noqa: BLE001 — factor failures are individual
        get_logger("subfactor_expansion.panel").warning(
            "candidate builder failed at %s: %s", ctx.as_of, exc)
        return []


# ---------------------------------------------------------------------------
# On-disk cache — same shape as ``research.panel``.
# ---------------------------------------------------------------------------
def cache_key(start: str, end: str, freq: str) -> str:
    return f"cand_panel_{start}_{end}_{freq}_v{CACHE_VERSION}.pkl"


def load_cached_panel(path: Path) -> CandidatePanel | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as fh:
            obj = pickle.load(fh)
        return obj if isinstance(obj, CandidatePanel) else None
    except Exception:
        return None


def save_cached_panel(panel: CandidatePanel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(panel, fh)
