"""Earnings call analyzer.

Reads the latest transcript from the warehouse and asks Claude to deliver
a senior-analyst read of management's posture, guidance credibility,
competitive position, and the verbatim quotes that matter.
"""
from __future__ import annotations

from typing import Any

from data.db import Database

from .base import AnalyzerContext, run_cached, truncate_preserving_edges
from .data_access import TranscriptArtifact, latest_transcript
from .prompts import EARNINGS_SYSTEM, EARNINGS_USER_TEMPLATE

# 60k chars ~ 15k tokens of transcript; leaves comfortable headroom in a
# 200k-context model for the system + response. Transcripts that hit this
# threshold are truncated preserving beginning, middle, and end.
DEFAULT_MAX_TRANSCRIPT_CHARS = 60_000
DEFAULT_MAX_TOKENS = 2048


def analyze_transcript(
    ctx: AnalyzerContext,
    artifact: TranscriptArtifact,
    *,
    max_chars: int = DEFAULT_MAX_TRANSCRIPT_CHARS,
) -> dict[str, Any] | None:
    transcript = truncate_preserving_edges(artifact.text, max_chars)
    user = EARNINGS_USER_TEMPLATE.format(
        ticker=artifact.ticker,
        fiscal_year=artifact.fiscal_year,
        fiscal_quarter=artifact.fiscal_quarter,
        call_date=artifact.call_date or "unknown",
        transcript=transcript,
    )
    return run_cached(
        ctx,
        analyzer="earnings",
        ticker=artifact.ticker,
        artifact_id=artifact.artifact_id,
        artifact_payload=transcript,   # hash on truncated text -> stable
        system_prompt=EARNINGS_SYSTEM,
        user_prompt=user,
        max_tokens=DEFAULT_MAX_TOKENS,
    )


def analyze(ctx: AnalyzerContext, db: Database, ticker: str,
            **kwargs: Any) -> dict[str, Any] | None:
    """Convenience: fetch the latest transcript and analyze it.

    Returns ``None`` if no transcript is stored for the ticker.
    """
    art = latest_transcript(db, ticker)
    if art is None:
        return None
    return analyze_transcript(ctx, art, **kwargs)
