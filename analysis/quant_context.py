"""Assemble the Layer 2 quant picture for the overlay prompt.

The overlay analyzer must understand WHY the factor model likes or dislikes
a stock before it can confirm or contradict the signal. This module bundles
everything Layer 2 already knows about a ticker into one artifact:

  * composite score + sector rank + long/short flag (composite_scores)
  * the eight parent factor scores (parent_factor_scores)
  * the strongest / weakest sub-factor drivers (sub_factor_scores)
  * universe percentile of the composite on the scoring date
  * recent price performance (price_features)
  * key valuation / fundamental ratios (fundamental_features)

Read-only SQL, same contract as :mod:`analysis.data_access`. Every section
is optional-tolerant: a missing table or row drops the section from the
formatted block instead of raising.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from data.db import Database
from data.utils import get_logger

from .data_access import CompositeArtifact, latest_composite

log = get_logger("analysis.quant_context")

# Parents at/above the high bar are called out as strong drivers, at/below
# the low bar as weak drivers, mirroring how the PM reads the scorecard.
STRONG_DRIVER_MIN = 70.0
WEAK_DRIVER_MAX = 40.0
N_TOP_SUBS = 5
N_BOTTOM_SUBS = 3

VALUATION_COLS = [
    "fcf_yield", "ev_to_ebitda", "price_to_sales",
    "roe", "roic", "debt_to_equity",
    "revenue_growth_yoy", "eps_growth_yoy",
]

RETURN_COLS = [
    "return_20d", "return_60d", "return_252d",
    "volatility_20d", "distance_from_52w_high",
]


@dataclass
class QuantContext:
    composite: CompositeArtifact
    company_name: str | None = None
    industry: str | None = None
    regime: str | None = None
    universe_percentile: float | None = None   # 0-100, higher = better
    universe_size: int | None = None
    parent_scores: dict[str, float] = field(default_factory=dict)
    sub_drivers: list[dict[str, Any]] = field(default_factory=list)
    returns: dict[str, float] = field(default_factory=dict)
    valuation: dict[str, float] = field(default_factory=dict)

    @property
    def ticker(self) -> str:
        return self.composite.ticker

    @property
    def as_of_date(self) -> str:
        return self.composite.as_of_date


def quant_context(db: Database, ticker: str,
                  as_of: str | None = None) -> QuantContext | None:
    """Build the full quant picture for ``ticker``.

    ``as_of`` pins a scoring date; default is the ticker's latest composite.
    Returns ``None`` only when there is no composite row at all - everything
    else degrades to empty sections.
    """
    composite = _composite_at(db, ticker, as_of) if as_of else latest_composite(db, ticker)
    if composite is None:
        return None
    date = composite.as_of_date
    qctx = QuantContext(composite=composite)

    row = _safe_query_one(
        db, "SELECT company_name, gics_sub_industry FROM universe WHERE ticker = ?",
        (ticker,))
    if row:
        qctx.company_name = row["company_name"]
        qctx.industry = row["gics_sub_industry"]

    row = _safe_query_one(
        db, "SELECT regime FROM composite_scores WHERE ticker = ? AND as_of_date = ?",
        (ticker, date))
    if row:
        qctx.regime = row["regime"]

    row = _safe_query_one(
        db,
        "SELECT COUNT(*) AS n, "
        "       SUM(CASE WHEN composite_score <= ? THEN 1 ELSE 0 END) AS below "
        "FROM composite_scores WHERE as_of_date = ?",
        (composite.composite_score, date))
    if row and row["n"]:
        qctx.universe_size = int(row["n"])
        qctx.universe_percentile = 100.0 * float(row["below"] or 0) / float(row["n"])

    rows = _safe_query(
        db, "SELECT factor, score FROM parent_factor_scores "
            "WHERE ticker = ? AND as_of_date = ? ORDER BY score DESC",
        (ticker, date))
    qctx.parent_scores = {r["factor"]: float(r["score"])
                          for r in rows if r["score"] is not None}

    rows = _safe_query(
        db, "SELECT factor, sub_factor, score, raw_value FROM sub_factor_scores "
            "WHERE ticker = ? AND as_of_date = ? AND score IS NOT NULL "
            "ORDER BY score DESC",
        (ticker, date))
    subs = [dict(r) for r in rows]
    if subs:
        picked = subs[:N_TOP_SUBS] + subs[max(N_TOP_SUBS, len(subs) - N_BOTTOM_SUBS):]
        qctx.sub_drivers = picked

    row = _safe_query_one(
        db, f"SELECT {', '.join(RETURN_COLS)} FROM price_features "
            f"WHERE ticker = ? AND date <= ? ORDER BY date DESC LIMIT 1",
        (ticker, date))
    if row:
        qctx.returns = {c: float(row[c]) for c in RETURN_COLS
                        if row[c] is not None}

    row = _safe_query_one(
        db, f"SELECT fiscal_date, {', '.join(VALUATION_COLS)} "
            f"FROM fundamental_features "
            f"WHERE ticker = ? AND period_type = 'quarterly' "
            f"ORDER BY fiscal_date DESC LIMIT 1",
        (ticker,))
    if row:
        qctx.valuation = {c: float(row[c]) for c in VALUATION_COLS
                          if row[c] is not None}

    return qctx


def _composite_at(db: Database, ticker: str, as_of: str) -> CompositeArtifact | None:
    row = _safe_query_one(
        db, "SELECT ticker, sector, composite_score, sector_rank, "
            "       long_short_flag, as_of_date "
            "FROM composite_scores WHERE ticker = ? AND as_of_date = ?",
        (ticker, as_of))
    if not row:
        return None
    return CompositeArtifact(
        ticker=row["ticker"], sector=row["sector"] or "Unknown",
        composite_score=float(row["composite_score"] or 0),
        sector_rank=row["sector_rank"],
        long_short_flag=row["long_short_flag"],
        as_of_date=row["as_of_date"],
    )


def _safe_query(db: Database, sql: str, params: tuple) -> list[Any]:
    try:
        return db.query(sql, params)
    except Exception as e:  # missing table on older DBs -> empty section
        log.debug("quant_context query skipped: %s", e)
        return []


def _safe_query_one(db: Database, sql: str, params: tuple) -> Any | None:
    try:
        return db.query_one(sql, params)
    except Exception as e:
        log.debug("quant_context query skipped: %s", e)
        return None


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------
def format_quant_context(qctx: QuantContext) -> str:
    """Render the quant picture as the compact text block fed to the LLM."""
    c = qctx.composite
    lines: list[str] = []
    name = f"{qctx.company_name} ({c.ticker})" if qctx.company_name else c.ticker
    lines.append(f"Company: {name}")
    sector_line = f"Sector: {c.sector}"
    if qctx.industry:
        sector_line += f"  Industry: {qctx.industry}"
    lines.append(sector_line)
    head = f"As of: {c.as_of_date}"
    if qctx.regime:
        head += f"  Market regime: {qctx.regime}"
    lines.append(head)

    score_line = f"Composite score: {c.composite_score:.1f}/100"
    if qctx.universe_percentile is not None:
        score_line += (f"  (universe percentile {qctx.universe_percentile:.0f}"
                       f" of {qctx.universe_size} names)")
    if c.sector_rank is not None:
        score_line += f"  Sector rank: {c.sector_rank}"
    score_line += f"  Signal: {c.long_short_flag or 'n/a'}"
    lines.append(score_line)

    if qctx.parent_scores:
        lines.append("")
        lines.append("Parent factor scores (0-100, sector-relative percentile; higher = better):")
        for factor, score in sorted(qctx.parent_scores.items(),
                                    key=lambda kv: kv[1], reverse=True):
            tag = ""
            if score >= STRONG_DRIVER_MIN:
                tag = "  <- strong driver"
            elif score <= WEAK_DRIVER_MAX:
                tag = "  <- weak driver"
            lines.append(f"  {factor:14s} {score:5.1f}{tag}")

    if qctx.sub_drivers:
        lines.append("")
        lines.append("Notable sub-factor drivers (score 0-100; raw = metric before ranking):")
        for s in qctx.sub_drivers:
            raw = s.get("raw_value")
            raw_txt = f"  raw={raw:.4g}" if isinstance(raw, (int, float)) else ""
            lines.append(f"  {s['factor']}/{s['sub_factor']:32s} "
                         f"score={float(s['score']):5.1f}{raw_txt}")

    if qctx.returns:
        parts = []
        for col, label in (("return_20d", "1M"), ("return_60d", "3M"),
                           ("return_252d", "12M")):
            v = qctx.returns.get(col)
            if v is not None:
                parts.append(f"{label} {v:+.1%}")
        extra = []
        vol = qctx.returns.get("volatility_20d")
        if vol is not None:
            extra.append(f"20d vol {vol:.0%} ann.")
        dist = qctx.returns.get("distance_from_52w_high")
        if dist is not None:
            extra.append(f"{abs(dist):.0%} below 52w high")
        if parts or extra:
            lines.append("")
            line = "Recent price performance: " + "  ".join(parts)
            if extra:
                line += f"  ({'; '.join(extra)})"
            lines.append(line)

    if qctx.valuation:
        lines.append("")
        lines.append("Key valuation / fundamentals (latest reported quarter):")
        pct_cols = {"fcf_yield", "roe", "roic",
                    "revenue_growth_yoy", "eps_growth_yoy"}
        for col in VALUATION_COLS:
            v = qctx.valuation.get(col)
            if v is None:
                continue
            txt = f"{v:.1%}" if col in pct_cols else f"{v:.2f}"
            lines.append(f"  {col:20s} {txt}")

    return "\n".join(lines)
