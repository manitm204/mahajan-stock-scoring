"""Shared scaffolding for the per-artifact analyzers.

Every analyzer follows the same flow:

  1. Build / fetch the input artifact from :mod:`analysis.data_access`.
  2. Hash the artifact and consult :class:`analysis.cache.AnalysisCache`.
  3. On miss, render a user prompt, call :class:`ClaudeClient.analyze`, parse
     JSON, store in cache.
  4. Return the structured dict.

This module collects the pieces that would otherwise be copy-pasted across
``earnings_analyzer``, ``filing_analyzer``, etc.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from data.utils import get_logger

from . import PROMPT_VERSIONS
from .api_client import ClaudeClient
from .cache import AnalysisCache, CacheKey, hash_artifact
from .cost_tracker import BudgetExceededError

log = get_logger("analysis.base")


@dataclass
class AnalyzerContext:
    """Shared services threaded into every analyzer."""
    client: ClaudeClient
    cache: AnalysisCache | None = None

    @property
    def model(self) -> str:
        return self.client.model


def run_cached(
    ctx: AnalyzerContext,
    *,
    analyzer: str,
    ticker: str,
    artifact_id: str,
    artifact_payload: Any,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> dict[str, Any] | None:
    """Look up cache; on miss, call Claude and store the result.

    Returns ``None`` if the model output cannot be parsed as JSON or if the
    call hits the cost-tracker budget cap (we log and skip rather than
    crash the whole run).
    """
    prompt_version = PROMPT_VERSIONS.get(analyzer, "v1")
    artifact_hash = hash_artifact(artifact_payload)
    key = CacheKey(
        analyzer=analyzer,
        ticker=ticker,
        artifact_id=artifact_id,
        artifact_hash=artifact_hash,
        prompt_version=prompt_version,
        model=ctx.model,
    )
    if ctx.cache is not None:
        cached = ctx.cache.get(key)
        if cached is not None:
            log.debug("cache HIT  %s/%s/%s", analyzer, ticker, artifact_id)
            return cached
    try:
        resp = ctx.client.analyze(
            system=system_prompt,
            user=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except BudgetExceededError as e:
        log.warning("Skipping %s/%s/%s: %s", analyzer, ticker, artifact_id, e)
        return None
    try:
        data = resp.json()
    except ValueError as e:
        log.warning("Bad JSON from %s/%s/%s: %s", analyzer, ticker, artifact_id, e)
        return None
    if not isinstance(data, dict):
        log.warning("Expected dict from %s/%s/%s, got %s",
                    analyzer, ticker, artifact_id, type(data).__name__)
        return None
    if ctx.cache is not None:
        ctx.cache.put(key, data)
    return data


def truncate_preserving_edges(text: str, max_chars: int,
                              head_frac: float = 0.30,
                              mid_frac:  float = 0.20,
                              tail_frac: float = 0.50) -> str:
    """Shrink ``text`` to ~``max_chars`` while keeping beginning, middle, end.

    Designed for long transcripts where the most colourful Q&A is in the
    middle but management's set-up is at the start and forward guide is at
    the end. Drops two contiguous spans rather than re-chunking everything
    so the resulting text still reads naturally.
    """
    if len(text) <= max_chars:
        return text
    assert abs(head_frac + mid_frac + tail_frac - 1.0) < 1e-6, \
        "edge fractions must sum to 1.0"
    head_n = int(max_chars * head_frac)
    mid_n  = int(max_chars * mid_frac)
    tail_n = max_chars - head_n - mid_n
    head = text[:head_n]
    tail = text[-tail_n:] if tail_n > 0 else ""
    mid_center = len(text) // 2
    mid = text[mid_center - mid_n // 2 : mid_center + mid_n - mid_n // 2]
    sep = "\n\n... [omitted] ...\n\n"
    return head + sep + mid + sep + tail


def maybe_run(
    fn: Callable[..., dict[str, Any] | None],
    *args: Any,
    label: str,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Call ``fn`` and swallow exceptions into a logged warning.

    Used by orchestrators that fan out across many analyzers - a single
    bad transcript should not abort the whole run.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as e:  # pragma: no cover - defensive at the boundary
        log.exception("Analyzer %s failed: %s", label, e)
        return None
