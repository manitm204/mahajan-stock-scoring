"""Persist Layer 3 research overlays into the warehouse.

One row per (as_of_date, ticker) holding the categorical overlay produced
by :mod:`analysis.overlay_analyzer`. Purely informational: nothing in
portfolio construction reads this table for ranking or sizing - the Layer 2
composite is the sole ranking engine. The dashboard surfaces the research
status so a human can act on it at the approvals step.

``composite_score`` is snapshotted for audit only (what the quant said when
the overlay was written); it is never re-served as a ranking.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Iterable

from data.db import Database
from data.utils import get_logger

log = get_logger("analysis.overlay_store")

SCHEMA = """
CREATE TABLE IF NOT EXISTS research_overlays (
    as_of_date                    TEXT NOT NULL,
    ticker                        TEXT NOT NULL,
    sector                        TEXT,
    sector_rank                   INTEGER,
    long_short_flag               TEXT,
    composite_score               REAL,
    quant_signal_review           TEXT,
    thesis_alignment              TEXT,
    qualitative_risk_level        TEXT,
    business_quality              TEXT,
    management_tone               TEXT,
    competitive_position          TEXT,
    accounting_risk               TEXT,
    filing_risk                   TEXT,
    insider_signal_interpretation TEXT,
    research_status               TEXT NOT NULL,
    red_flags                     TEXT,
    open_questions                TEXT,
    confirming_evidence           TEXT,
    contradicting_evidence        TEXT,
    overlay_json                  TEXT,
    confidence                    INTEGER,
    model                         TEXT,
    computed_at                   TEXT,
    PRIMARY KEY (as_of_date, ticker)
);
CREATE INDEX IF NOT EXISTS idx_overlay_date   ON research_overlays(as_of_date);
CREATE INDEX IF NOT EXISTS idx_overlay_status ON research_overlays(research_status);
"""


def ensure_schema(db: Database) -> None:
    db._conn.executescript(SCHEMA)
    db._conn.commit()


def persist_overlays(
    db: Database,
    analyses: Iterable["TickerAnalysis"],     # noqa: F821 (forward ref)
    *,
    model: str | None = None,
) -> int:
    """Write one row per analyzed ticker. Returns the number persisted.

    Each :class:`TickerAnalysis` must carry a populated ``composite`` and
    ``overlay``; tickers missing either are skipped. ``as_of_date`` comes
    from the ticker's own composite row so a re-run over a stale snapshot
    doesn't backfill today's date over old data.
    """
    ensure_schema(db)
    now = _now()
    rows: list[dict] = []
    for a in analyses:
        if a.composite is None or a.overlay is None:
            continue
        o = a.overlay
        rows.append({
            "as_of_date":       a.composite.as_of_date,
            "ticker":           a.ticker.upper(),
            "sector":           a.composite.sector,
            "sector_rank":      a.composite.sector_rank,
            "long_short_flag":  a.composite.long_short_flag,
            "composite_score":  a.composite.composite_score,
            "quant_signal_review":           o.get("quant_signal_review"),
            "thesis_alignment":              o.get("thesis_alignment"),
            "qualitative_risk_level":        o.get("qualitative_risk_level"),
            "business_quality":              o.get("business_quality"),
            "management_tone":               o.get("management_tone"),
            "competitive_position":          o.get("competitive_position"),
            "accounting_risk":               o.get("accounting_risk"),
            "filing_risk":                   o.get("filing_risk"),
            "insider_signal_interpretation": o.get("insider_signal_interpretation"),
            "research_status":  o.get("final_research_status") or "REVIEW",
            "red_flags":              _json_list(o.get("red_flags")),
            "open_questions":         _json_list(o.get("open_questions")),
            "confirming_evidence":    _json_list(o.get("key_confirming_evidence")),
            "contradicting_evidence": _json_list(o.get("key_contradicting_evidence")),
            "overlay_json":     json.dumps(o, separators=(",", ":"), default=str),
            "confidence":       o.get("confidence"),
            "model":            model,
            "computed_at":      now,
        })

    if not rows:
        log.info("persist_overlays: no rows to write")
        return 0

    db.upsert("research_overlays", rows, conflict=["as_of_date", "ticker"])
    log.info("persist_overlays: wrote %d row(s)", len(rows))
    return len(rows)


def _json_list(v) -> str:
    return json.dumps(v if isinstance(v, list) else [], default=str)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
