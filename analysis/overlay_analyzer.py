"""Research-overlay analyzer.

Synthesises the four upstream analyzer outputs plus the full Layer 2 quant
picture into a categorical due-diligence overlay. This is the only analyzer
that takes prior analyzer outputs as input, and the only one that sees the
factor breakdown.

The overlay never scores or recommends - it answers whether the qualitative
evidence confirms, weakens, or contradicts the quant signal, and assigns a
research status (PASS / WATCHLIST / REVIEW / AVOID_RED_FLAG) for the human
at the approvals step.
"""
from __future__ import annotations

import json
from typing import Any

from data.db import Database
from data.utils import get_logger

from .base import AnalyzerContext, run_cached
from .prompts import OVERLAY_SYSTEM, OVERLAY_USER_TEMPLATE
from .quant_context import QuantContext, format_quant_context, quant_context

log = get_logger("analysis.overlay_analyzer")

DEFAULT_MAX_TOKENS = 3072

RESEARCH_STATUSES = ("PASS", "WATCHLIST", "REVIEW", "AVOID_RED_FLAG")

# Allowed values per categorical field; anything else is coerced to the
# fallback (last element listed here) with a logged warning.
_ENUM_FIELDS: dict[str, tuple[str, ...]] = {
    "quant_signal_review":  ("CONFIRMS", "MIXED", "WEAKENS", "CONTRADICTS"),
    "thesis_alignment":     ("BULLISH", "NEUTRAL", "BEARISH"),
    "qualitative_risk_level": ("LOW", "MEDIUM", "HIGH"),
    "business_quality":     ("HIGH", "MEDIUM", "LOW"),
    "management_tone":      ("CONFIDENT", "REALISTIC", "CAUTIOUS",
                             "DEFENSIVE", "PROMOTIONAL", "UNKNOWN"),
    "competitive_position": ("IMPROVING", "STABLE", "DETERIORATING", "UNKNOWN"),
    "accounting_risk":      ("LOW", "MEDIUM", "HIGH", "UNKNOWN"),
    "filing_risk":          ("LOW", "MEDIUM", "HIGH", "UNKNOWN"),
    "insider_signal_interpretation": (
        "STRONGLY_POSITIVE", "MODESTLY_POSITIVE", "NEUTRAL",
        "MODESTLY_NEGATIVE", "STRONGLY_NEGATIVE", "UNKNOWN"),
}

_LIST_FIELDS = (
    "red_flags", "open_questions",
    "key_confirming_evidence", "key_contradicting_evidence",
)


def validate_overlay(data: dict[str, Any], *, ticker: str = "?") -> dict[str, Any]:
    """Coerce a raw model payload into a well-typed overlay dict.

    * Categorical fields outside their enum collapse to UNKNOWN (or None
      when the field has no UNKNOWN member) with a warning.
    * ``final_research_status`` outside the four allowed values is forced
      to REVIEW - an unparseable status must fail toward human attention,
      never toward a silent PASS.
    * List fields are coerced to lists of strings.
    Never raises on sloppy model output.
    """
    out = dict(data)

    for field, allowed in _ENUM_FIELDS.items():
        value = str(out.get(field) or "").strip().upper().replace(" ", "_")
        if value in allowed:
            out[field] = value
        else:
            fallback = "UNKNOWN" if "UNKNOWN" in allowed else None
            if out.get(field) is not None:
                log.warning("overlay/%s: %s=%r not in enum; coerced to %s",
                            ticker, field, data.get(field), fallback)
            out[field] = fallback

    status = str(out.get("final_research_status") or "").strip().upper().replace(" ", "_")
    if status not in RESEARCH_STATUSES:
        log.warning("overlay/%s: final_research_status=%r invalid; forcing REVIEW",
                    ticker, data.get("final_research_status"))
        status = "REVIEW"
    out["final_research_status"] = status

    for field in _LIST_FIELDS:
        v = out.get(field)
        if not isinstance(v, list):
            v = [v] if isinstance(v, str) and v else []
        out[field] = [str(x) for x in v if x]

    wct = out.get("what_could_change_thesis")
    if not isinstance(wct, dict):
        wct = {}
    for key in ("bull_breakers", "bear_breakers", "catalysts"):
        v = wct.get(key)
        if not isinstance(v, list):
            v = [v] if isinstance(v, str) and v else []
        wct[key] = [str(x) for x in v if x]
    out["what_could_change_thesis"] = wct

    conf = out.get("confidence")
    out["confidence"] = int(conf) if isinstance(conf, (int, float)) else None

    return out


def _compact_json(d: dict[str, Any] | None) -> str:
    """Serialise an analyzer output for inclusion in a prompt; ``null`` if missing."""
    if not d:
        return "null"
    return json.dumps(d, separators=(",", ":"), default=str)


def synthesize(
    ctx: AnalyzerContext,
    qctx: QuantContext,
    *,
    earnings_analysis: dict[str, Any] | None = None,
    filing_analysis:   dict[str, Any] | None = None,
    risk_analysis:     dict[str, Any] | None = None,
    insider_analysis:  dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    quant_block = format_quant_context(qctx)
    user = OVERLAY_USER_TEMPLATE.format(
        quant_context_block=quant_block,
        earnings_json=_compact_json(earnings_analysis),
        filing_json=_compact_json(filing_analysis),
        risk_json=_compact_json(risk_analysis),
        insider_json=_compact_json(insider_analysis),
    )
    payload = {
        "quant_block": quant_block,
        "earnings_h": _compact_json(earnings_analysis),
        "filing_h":   _compact_json(filing_analysis),
        "risk_h":     _compact_json(risk_analysis),
        "insider_h":  _compact_json(insider_analysis),
    }
    data = run_cached(
        ctx,
        analyzer="overlay",
        ticker=qctx.ticker,
        artifact_id=f"overlay:{qctx.as_of_date}",
        artifact_payload=payload,
        system_prompt=OVERLAY_SYSTEM,
        user_prompt=user,
        max_tokens=DEFAULT_MAX_TOKENS,
    )
    if data is None:
        return None
    return validate_overlay(data, ticker=qctx.ticker)


def analyze(ctx: AnalyzerContext, db: Database, ticker: str,
            qctx: QuantContext | None = None,
            **upstream: dict[str, Any] | None) -> dict[str, Any] | None:
    if qctx is None:
        qctx = quant_context(db, ticker)
    if qctx is None:
        return None
    return synthesize(ctx, qctx, **upstream)
