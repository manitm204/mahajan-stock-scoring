"""Point-in-time score panel — the shared input for every research metric.

Scoring the universe once per rebalance is the expensive step, so we do it a
single time and hand every downstream analysis (quintiles, IC, redundancy,
incremental contribution) the same captured panel. For each rebalance date the
panel holds one wide frame (ticker x signal) where *signals* are the eight parent
factor scores **and** every sub-factor score, all on the same 0-100 sector-relative
scale. Treating parents and subs as interchangeable "signals" lets the identical
quintile/IC machinery run at both levels with no special-casing.

Strictly point-in-time: each date's frame comes from a
``DataContext(as_of=date, reporting_lag=True)`` so a score at ``d_k`` only sees
Layer 1 rows observable on/before ``d_k`` (fundamentals admitted with a filing
lag). Nothing here looks at returns.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from data.config import Config
from data.db import Database
from factors import ALL_FACTORS
from factors.base import score_factor
from factors.utils import DataContext

CACHE_VERSION = 3   # v3: point-in-time universe per rebalance (members_as_of)


@dataclass
class ScorePanel:
    """Captured per-rebalance score frames plus the parent/sub-factor taxonomy."""

    rebal_dates: list[str]
    scores: dict[str, pd.DataFrame]          # date -> (ticker x signal) 0-100
    parent_keys: list[str]                   # e.g. ["momentum", "value", ...]
    sub_by_parent: dict[str, list[str]]      # parent -> [sub names]
    universe: list[str]

    @property
    def all_subs(self) -> list[str]:
        out: list[str] = []
        for subs in self.sub_by_parent.values():
            out.extend(subs)
        return out

    def parent_of(self, sub: str) -> str:
        for parent, subs in self.sub_by_parent.items():
            if sub in subs:
                return parent
        return "unknown"

    def signal_frame(self, date: str, columns: list[str]) -> pd.DataFrame:
        """The ticker x ``columns`` slice for ``date`` (missing cols dropped)."""
        frame = self.scores[date]
        keep = [c for c in columns if c in frame.columns]
        return frame[keep]


def build_score_panel(
    db: Database,
    rebal_dates: list[str],
    cfg: Config,
    *,
    min_obs: int | None = None,
    reporting_lag: bool = True,
    verbose: bool = True,
) -> ScorePanel:
    """Score every parent + sub-factor at each rebalance date into one panel."""
    if min_obs is None:
        min_obs = int(cfg.get("factors", "min_sector_obs", default=5))
    insider_days = int(cfg.get("factors", "insider_window_days", default=90))

    scores: dict[str, pd.DataFrame] = {}
    sub_by_parent: dict[str, list[str]] = {}
    parent_keys: list[str] = []
    seen: set[str] = set()                     # union of PIT members across all rebalances

    for d in rebal_dates:
        # Point-in-time universe: score only names that were index members on ``d``.
        members = db.members_as_of(d) or db.universe_tickers()
        seen.update(members)
        ctx = DataContext(
            db=db, as_of=d, universe=members,
            insider_window_days=insider_days, reporting_lag=reporting_lag
        )
        try:
            cols: dict[str, pd.Series] = {}
            for factor in ALL_FACTORS:
                res = score_factor(factor, ctx, min_obs=min_obs)
                # Parent score is itself a signal (factor-level evaluation).
                cols[res.key] = res.parent
                if res.key not in parent_keys:
                    parent_keys.append(res.key)
                # The parent's sub-factors are signals too (sub-level evaluation).
                subs = list(res.sub_scores.columns)
                sub_by_parent.setdefault(res.key, [])
                for sub in subs:
                    cols[sub] = res.sub_scores[sub]
                    if sub not in sub_by_parent[res.key]:
                        sub_by_parent[res.key].append(sub)
            frame = pd.DataFrame(cols).reindex(sorted(members))
            scores[d] = frame
            if verbose:
                print(f"  scored {d}: {len(parent_keys)} parents, "
                      f"{sum(len(s) for s in sub_by_parent.values())} subs x "
                      f"{frame.shape[0]} tickers")
        finally:
            ctx.close()

    return ScorePanel(
        rebal_dates=list(scores.keys()),
        scores=scores,
        parent_keys=parent_keys,
        sub_by_parent=sub_by_parent,
        universe=sorted(seen),
    )


# ---------------------------------------------------------------------------
# Optional on-disk cache so report iteration doesn't re-score the universe.
# ---------------------------------------------------------------------------
def cache_key(start: str, end: str, freq: str) -> str:
    return f"panel_{start}_{end}_{freq}_v{CACHE_VERSION}.pkl"


def load_cached_panel(path: Path) -> ScorePanel | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as fh:
            obj = pickle.load(fh)
        return obj if isinstance(obj, ScorePanel) else None
    except Exception:
        return None


def save_cached_panel(panel: ScorePanel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(panel, fh)
