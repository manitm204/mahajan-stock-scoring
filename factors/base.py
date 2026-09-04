"""Factor framework: the contract every factor implements and the engine that
turns raw sub-factor values into explainable 0–100 scores.

A factor is a small class that knows nothing about ranking or persistence. It
only produces, per sub-factor, a raw value per ticker plus a direction
(higher-is-better). The engine applies the universal rules uniformly:

* each sub-factor -> GICS-sector-relative percentile (0–100, missing = 50);
* parent factor score = equal-weight average of its sub-factor scores.

This keeps the scoring methodology in one place: adding or removing a factor is
a matter of writing/deleting a small class and editing the registry — no
changes to ranking, aggregation, or storage. The Layer 2 output schema and the
persistence helpers also live here so all DB writes share one definition.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd

from data.db import Database
from .utils import DataContext, sector_percentile


@dataclass
class SubFactor:
    """One raw sub-factor signal awaiting ranking.

    ``raw`` is a per-ticker Series of the underlying metric (e.g. 12-1 month
    return). ``higher_is_better`` tells the engine which tail deserves the high
    percentile; inverted metrics (EV/EBITDA, debt/equity, short interest) set it
    False instead of pre-negating, which keeps the raw values readable.
    ``weight`` is the sub-factor's share of its parent's composite score; None or
    1.0 both give the legacy equal-weight behaviour when every sub in the parent
    shares the default. Weights are renormalised across selected subs so the
    parent stays on a 0-100 scale.
    """

    name: str
    raw: pd.Series
    higher_is_better: bool = True
    weight: float = 1.0


class Factor:
    """Base class for a parent factor composed of equal-weighted sub-factors.

    Subclasses set ``name`` (and ``weight_key`` if it differs) and implement
    :meth:`compute`, returning the list of :class:`SubFactor` signals. Everything
    else — ranking, aggregation, missing-data neutrality — is handled by
    :func:`score_factor`.
    """

    name: str = "factor"
    weight_key: str = ""

    def compute(self, ctx: DataContext) -> list[SubFactor]:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def key(self) -> str:
        return self.weight_key or self.name


@dataclass
class FactorResult:
    """Scored output of one factor for the whole universe."""

    name: str
    key: str
    parent: pd.Series                     # 0–100 per ticker
    sub_scores: pd.DataFrame              # ticker x sub-factor (0–100)
    raw: pd.DataFrame = field(default_factory=pd.DataFrame)

    def neutral_fraction(self) -> float:
        """Share of names whose parent score is exactly neutral (50).

        A high value means the factor barely differentiates the universe (almost
        all inputs missing) — surfaced by crowding's degenerate-factor check.
        """
        if self.parent.empty:
            return 1.0
        return float((self.parent.round(6) == 50.0).mean())


def score_factor(factor: Factor, ctx: DataContext, min_obs: int = 5,
                 sub_factor_allowlist: "list[str] | dict[str, float] | None" = None
                 ) -> FactorResult:
    """Rank a factor's sub-factors sector-relative and combine them.

    ``sub_factor_allowlist`` selects which of the parent's emitted subs feed the
    parent score and, optionally, how heavily each contributes:

    * ``None`` — use every sub-factor the parent emits (historical V1 behaviour).
    * ``list[str]`` — keep only these names; each contributes its ``SubFactor.weight``
      (defaults to 1.0 → equal weight).
    * ``dict[str, float]`` — keep only the keys, each with the supplied weight (this
      overrides ``SubFactor.weight`` and is how the parent-selection framework
      applies its intra-parent weightings). Missing sub → dropped from the parent.

    Weights are renormalised across the kept subs, so the parent score stays on
    the same 0-100 scale regardless of how many subs contribute. Missing sub
    scores are neutral 50 (from :func:`sector_percentile`) — no NaN handling
    needed.

    An allowlist that matches zero of the parent's available sub-factors is a
    configuration error and raises so a typo cannot silently collapse the parent
    to neutral.
    """
    sectors = ctx.sectors()
    subs = factor.compute(ctx)
    weight_override: dict[str, float] | None = None
    if sub_factor_allowlist is not None:
        if isinstance(sub_factor_allowlist, dict):
            weight_override = {str(k): float(v) for k, v in sub_factor_allowlist.items()}
            keep = set(weight_override)
        else:
            keep = {str(s) for s in sub_factor_allowlist}
        kept = [sf for sf in subs if sf.name in keep]
        if not kept:
            raise ValueError(
                f"sub_factor_allowlist for parent {factor.key!r} matched none "
                f"of the available sub-factors. allowlist={sorted(keep)}, "
                f"available={[sf.name for sf in subs]}.")
        subs = kept
    sub_scores: dict[str, pd.Series] = {}
    raw_cols: dict[str, pd.Series] = {}
    weights: dict[str, float] = {}
    for sf in subs:
        raw = sf.raw.reindex(ctx.universe)
        raw_cols[sf.name] = raw
        sub_scores[sf.name] = sector_percentile(
            raw, sectors, higher_is_better=sf.higher_is_better, min_obs=min_obs)
        w = weight_override[sf.name] if weight_override else float(sf.weight)
        weights[sf.name] = max(w, 0.0)
    sub_df = pd.DataFrame(sub_scores).reindex(ctx.universe)
    w_series = pd.Series(weights, dtype=float).reindex(sub_df.columns).fillna(0.0)
    total_w = float(w_series.sum())
    if total_w > 0:
        parent = (sub_df * w_series).sum(axis=1) / total_w
    else:
        # All-zero weight vector → fall back to equal-weight (defensive; the
        # config validator should have caught this earlier).
        parent = sub_df.mean(axis=1)
    parent = parent.round(4)
    return FactorResult(
        name=factor.name,
        key=factor.key,
        parent=parent,
        sub_scores=sub_df,
        raw=pd.DataFrame(raw_cols).reindex(ctx.universe),
    )


# ---------------------------------------------------------------------------
# Layer 2 output schema + persistence (separate tables; never mutates Layer 1)
# ---------------------------------------------------------------------------
LAYER2_SCHEMA = """
CREATE TABLE IF NOT EXISTS composite_scores (
    as_of_date      TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    sector          TEXT,
    regime          TEXT,
    composite_score REAL,
    sector_rank     INTEGER,
    long_short_flag TEXT,
    computed_at     TEXT,
    PRIMARY KEY (as_of_date, ticker)
);
CREATE INDEX IF NOT EXISTS idx_comp_date ON composite_scores(as_of_date);
CREATE INDEX IF NOT EXISTS idx_comp_flag ON composite_scores(long_short_flag);

CREATE TABLE IF NOT EXISTS parent_factor_scores (
    as_of_date TEXT NOT NULL,
    ticker     TEXT NOT NULL,
    factor     TEXT NOT NULL,
    score      REAL,
    PRIMARY KEY (as_of_date, ticker, factor)
);
CREATE INDEX IF NOT EXISTS idx_parent_date ON parent_factor_scores(as_of_date);

CREATE TABLE IF NOT EXISTS sub_factor_scores (
    as_of_date TEXT NOT NULL,
    ticker     TEXT NOT NULL,
    factor     TEXT NOT NULL,
    sub_factor TEXT NOT NULL,
    score      REAL,
    raw_value  REAL,   -- raw metric retained for explainability of any score
    PRIMARY KEY (as_of_date, ticker, factor, sub_factor)
);
CREATE INDEX IF NOT EXISTS idx_sub_date ON sub_factor_scores(as_of_date);
"""


def ensure_layer2_schema(db: Database) -> None:
    db._conn.executescript(LAYER2_SCHEMA)
    db._conn.commit()


def persist_results(
    db: Database,
    as_of: str,
    results: Sequence[FactorResult],
    composite: pd.DataFrame,
    regime: str,
    computed_at: str,
) -> None:
    """Write composite, parent, and sub-factor scores for ``as_of``.

    Idempotent: keyed by (as_of_date, ticker[, factor[, sub_factor]]) so re-runs
    refresh in place rather than duplicating — the same UPSERT discipline Layer 1
    uses.
    """
    ensure_layer2_schema(db)

    comp_rows = [
        {
            "as_of_date": as_of,
            "ticker": tkr,
            "sector": row["sector"],
            "regime": regime,
            "composite_score": _num(row["composite_score"]),
            "sector_rank": int(row["sector_rank"]) if pd.notna(row["sector_rank"]) else None,
            "long_short_flag": row["long_short_flag"],
            "computed_at": computed_at,
        }
        for tkr, row in composite.iterrows()
    ]
    db.upsert("composite_scores", comp_rows, conflict=["as_of_date", "ticker"])

    parent_rows: list[dict] = []
    sub_rows: list[dict] = []
    for res in results:
        for tkr in res.parent.index:
            parent_rows.append({
                "as_of_date": as_of, "ticker": tkr, "factor": res.key,
                "score": _num(res.parent.get(tkr)),
            })
        for sub_name in res.sub_scores.columns:
            score_col = res.sub_scores[sub_name]
            raw_col = res.raw[sub_name] if sub_name in res.raw.columns else None
            for tkr in score_col.index:
                sub_rows.append({
                    "as_of_date": as_of, "ticker": tkr, "factor": res.key,
                    "sub_factor": sub_name, "score": _num(score_col.get(tkr)),
                    "raw_value": _num(raw_col.get(tkr)) if raw_col is not None else None,
                })
    db.upsert("parent_factor_scores", parent_rows,
              conflict=["as_of_date", "ticker", "factor"])

    # A factor's sub-factor set can change (renamed/removed sub-factors). Since
    # sub_factor_scores is keyed by sub_factor name, a plain UPSERT would leave
    # the old names as orphan rows. Clear this date's rows for the factors being
    # written first so the table reflects only the current definitions.
    factor_keys = [res.key for res in results]
    if factor_keys:
        ph = ",".join("?" * len(factor_keys))
        db._conn.execute(
            f"DELETE FROM sub_factor_scores WHERE as_of_date = ? AND factor IN ({ph})",
            (as_of, *factor_keys))
        db._conn.commit()
    db.upsert("sub_factor_scores", sub_rows,
              conflict=["as_of_date", "ticker", "factor", "sub_factor"])


def _num(value: object) -> float | None:
    """Coerce to a plain float, mapping NaN/None to SQL NULL."""
    if value is None:
        return None
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f
