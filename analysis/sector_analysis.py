"""Cross-company sector comparison.

Given the Layer 2 + Layer 3 outputs for every name in a GICS sector, asks
Claude to pick the strongest competitive position, best balance sheet,
best management, most attractive long, and the highest-risk short
candidate. Used by the report generator to add sector context to
individual memos and to surface sector-level conviction.
"""
from __future__ import annotations

import json
from typing import Any

from .base import AnalyzerContext, run_cached
from .data_access import CompositeArtifact
from .prompts import SECTOR_SYSTEM, SECTOR_USER_TEMPLATE

DEFAULT_MAX_TOKENS = 2048
# Sectors can have 60+ names; truncate to avoid blowing budget on a single call.
MAX_COMPANIES = 30


def _format_companies_table(scores: list[CompositeArtifact]) -> str:
    lines = ["ticker | composite | sector_rank | flag"]
    lines.append("-" * len(lines[0]))
    for s in scores:
        lines.append(
            f"{s.ticker:6s} | {s.composite_score:>9.2f} | "
            f"{(str(s.sector_rank) if s.sector_rank is not None else 'n/a'):>11s} | "
            f"{(s.long_short_flag or 'n/a')}"
        )
    return "\n".join(lines)


def _trim_overlay(overlay: dict[str, Any] | None) -> dict[str, Any] | None:
    """Cherry-pick the few fields the sector call actually needs.

    The full overlay JSON would balloon prompt tokens; the sector call only
    needs the headline categoricals + the evidence bullets.
    """
    if not overlay:
        return None
    keep = (
        "quant_signal_review", "final_research_status",
        "thesis_alignment", "business_quality",
        "qualitative_risk_level",
        "key_confirming_evidence", "key_contradicting_evidence",
        "red_flags",
    )
    return {k: overlay.get(k) for k in keep if k in overlay}


def analyze_sector(
    ctx: AnalyzerContext,
    sector: str,
    scores: list[CompositeArtifact],
    overlays: dict[str, dict[str, Any] | None],
    *,
    max_companies: int = MAX_COMPANIES,
) -> dict[str, Any] | None:
    if not scores:
        return None
    # Order by composite_score desc so the most relevant names always fit
    # under the truncation cap.
    ordered = sorted(scores, key=lambda s: s.composite_score, reverse=True)[:max_companies]
    summaries = {s.ticker: _trim_overlay(overlays.get(s.ticker)) for s in ordered}
    summaries_blob = json.dumps(summaries, separators=(",", ":"), default=str, indent=2)
    table = _format_companies_table(ordered)
    user = SECTOR_USER_TEMPLATE.format(
        sector=sector,
        n=len(ordered),
        companies_table=table,
        thesis_summaries=summaries_blob,
    )
    # Hash on the truncated inputs so a re-run with the same data is cached.
    payload = {"table": table, "summaries": summaries_blob}
    return run_cached(
        ctx,
        analyzer="sector",
        ticker=sector.replace(" ", "_").upper(),
        artifact_id=f"sector:{sector}:{len(ordered)}",
        artifact_payload=payload,
        system_prompt=SECTOR_SYSTEM,
        user_prompt=user,
        max_tokens=DEFAULT_MAX_TOKENS,
    )
