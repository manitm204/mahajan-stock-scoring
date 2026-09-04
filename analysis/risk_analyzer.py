"""10-K Risk Factors analyzer.

Compares the current 10-K's Item 1A section to the prior year's, asks
Claude to surface what was added, removed, or meaningfully reworded.
Returns ``None`` (per spec) when no parseable risk section is available -
either no 10-K is on file or HTML extraction failed.
"""
from __future__ import annotations

from typing import Any

from data.db import Database

from .base import AnalyzerContext, run_cached
from .data_access import RiskArtifact, risk_artifact
from .prompts import RISK_SYSTEM, RISK_USER_TEMPLATE, RISK_USER_TEMPLATE_SOLO

# Risk-factor sections can hit 100k+ chars. Truncate the tail rather than
# the head - the boilerplate is at the end. Both sides of the diff get the
# same budget so the model can compare like-for-like.
DEFAULT_MAX_SECTION_CHARS = 40_000
DEFAULT_MAX_TOKENS = 2048


def analyze_risk(
    ctx: AnalyzerContext,
    artifact: RiskArtifact,
    *,
    max_section_chars: int = DEFAULT_MAX_SECTION_CHARS,
) -> dict[str, Any] | None:
    current = artifact.current_text[:max_section_chars]
    if artifact.prior_text:
        prior = artifact.prior_text[:max_section_chars]
        user = RISK_USER_TEMPLATE.format(
            ticker=artifact.ticker,
            current_date=artifact.current_filing_date,
            prior_date=artifact.prior_filing_date,
            current_text=current,
            prior_text=prior,
        )
        payload = (current, prior)
    else:
        user = RISK_USER_TEMPLATE_SOLO.format(
            ticker=artifact.ticker,
            current_date=artifact.current_filing_date,
            current_text=current,
        )
        payload = (current, None)
    return run_cached(
        ctx,
        analyzer="risk",
        ticker=artifact.ticker,
        artifact_id=artifact.artifact_id,
        artifact_payload=payload,
        system_prompt=RISK_SYSTEM,
        user_prompt=user,
        max_tokens=DEFAULT_MAX_TOKENS,
    )


def analyze(ctx: AnalyzerContext, db: Database, ticker: str,
            **kwargs: Any) -> dict[str, Any] | None:
    """Returns ``None`` if no 10-K Risk Factors can be extracted."""
    art = risk_artifact(db, ticker)
    if art is None:
        return None
    return analyze_risk(ctx, art, **kwargs)
