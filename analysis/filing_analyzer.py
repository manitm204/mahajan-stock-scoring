"""Forensic accounting analyzer.

Reads the latest 10-K's Item 7 MD&A section and asks Claude to flag
earnings-quality concerns, liquidity/balance-sheet stress, and capital
allocation problems from *management's own narrative*.

We deliberately do NOT feed the computed fundamental metrics table here:
Layer 2 already scores those ratios (see factors/quality.py), so scoring
them again in the LLM double-counts the quality factor. MD&A prose is the
orthogonal source - management's discussion, liquidity commentary, and
critical-accounting-estimate disclosures that the numeric factors never see.
"""
from __future__ import annotations

from typing import Any

from data.db import Database

from .base import AnalyzerContext, run_cached, truncate_preserving_edges
from .data_access import MDAArtifact, mdna_artifact
from .prompts import FILING_SYSTEM, FILING_USER_TEMPLATE

# ~45k chars ~ 11k tokens of MD&A; the results-of-operations discussion sits
# in the middle and the liquidity/outlook language at the end, so edge-
# preserving truncation keeps the forensically interesting spans.
DEFAULT_MAX_MDNA_CHARS = 45_000
DEFAULT_MAX_TOKENS = 2048


def analyze_mdna(
    ctx: AnalyzerContext,
    artifact: MDAArtifact,
    *,
    max_chars: int = DEFAULT_MAX_MDNA_CHARS,
) -> dict[str, Any] | None:
    mdna = truncate_preserving_edges(artifact.text, max_chars)
    user = FILING_USER_TEMPLATE.format(
        ticker=artifact.ticker,
        sector=artifact.sector,
        form_type=artifact.form_type,
        filing_date=artifact.filing_date,
        mdna_text=mdna,
    )
    return run_cached(
        ctx,
        analyzer="filing",
        ticker=artifact.ticker,
        artifact_id=artifact.artifact_id,
        artifact_payload=mdna,   # hash on truncated text -> stable
        system_prompt=FILING_SYSTEM,
        user_prompt=user,
        max_tokens=DEFAULT_MAX_TOKENS,
    )


def analyze(ctx: AnalyzerContext, db: Database, ticker: str,
            **kwargs: Any) -> dict[str, Any] | None:
    """Convenience: fetch the latest 10-K MD&A and analyze it.

    Returns ``None`` if no 10-K with an extractable Item 7 is stored.
    """
    art = mdna_artifact(db, ticker)
    if art is None:
        return None
    return analyze_mdna(ctx, art, **kwargs)
