"""Reusable Anthropic SDK wrapper for every Layer 3 analyzer.

Responsibilities:

* Load ``ANTHROPIC_API_KEY`` from ``.env`` (via :mod:`data.config`).
* Enable prompt caching on every system prompt - the system block is the
  static analyst persona, identical across calls, so a cache hit pays the
  $0.30/MTok read price instead of $3.00/MTok input.
* Auto-retry on HTTP 429 and 5xx with exponential backoff + jitter.
* Robust JSON extraction (raw JSON, fenced ```json blocks, or first
  brace-balanced object in prose).
* Cheap pre-flight token estimator so :class:`CostTracker` can veto calls.
* Per-request logging of model, tokens, cost.

The client is deliberately stateless w.r.t. analyzers - one ``ClaudeClient``
is shared by every analyzer in a run, threaded through to apply the same
cost tracker and cache.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any

import anthropic

from data.config import load_config
from data.utils import get_logger

from .cost_tracker import CostTracker, Usage, estimate_cost

log = get_logger("analysis.api_client")

# Anthropic publishes a chars-per-token average around 3.5 for English prose;
# we round to 4 so estimates skew slightly conservative (better to overshoot
# the budget check than undershoot).
CHARS_PER_TOKEN = 4

DEFAULT_MAX_TOKENS = 2048
DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_BASE = 1.5
DEFAULT_BACKOFF_CAP = 30.0


@dataclass
class ClaudeResponse:
    """The slice of an Anthropic response Layer 3 actually consumes."""
    text: str
    usage: Usage
    raw: Any  # underlying SDK Message, kept for debugging only

    def json(self) -> dict[str, Any] | list[Any]:
        """Best-effort JSON extraction. Raises ``ValueError`` if nothing parses."""
        return extract_json(self.text)


class ClaudeClient:
    """Thin wrapper over :class:`anthropic.Anthropic` with retries + caching."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        api_key: str | None = None,
        cost_tracker: CostTracker | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        cfg = load_config()
        key = api_key or os.environ.get("ANTHROPIC_API_KEY") or cfg.env("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY missing. Add it to .env before running Layer 3."
            )
        self.model = model
        self.max_retries = max_retries
        self.cost = cost_tracker or CostTracker()
        # The SDK already handles its own connection pooling; we just hold one.
        self._sdk = anthropic.Anthropic(api_key=key)

    # ---------------------------------------------------------------- utils
    def estimate_tokens(self, text: str) -> int:
        """Rough token count for budgeting; conservative by ~10%."""
        return max(1, len(text) // CHARS_PER_TOKEN)

    def estimate_call_cost(
        self,
        system: str,
        user: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> float:
        """Pre-flight $ estimate that assumes worst case output usage."""
        in_tok = self.estimate_tokens(system) + self.estimate_tokens(user)
        return estimate_cost(self.model, in_tok, max_tokens)

    # ----------------------------------------------------------- main call
    def analyze(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
        cache_system: bool = True,
        skip_budget_check: bool = False,
    ) -> ClaudeResponse:
        """Send a single chat completion and return text + usage.

        ``cache_system=True`` (the default) wraps the system prompt in an
        ephemeral cache_control block. The first call writes the cache; every
        subsequent call within ~5 minutes pays the much cheaper read rate.
        """
        if not skip_budget_check:
            est = self.estimate_call_cost(system, user, max_tokens)
            self.cost.check_budget(est)

        system_blocks = self._build_system(system, cache_system)
        messages = [{"role": "user", "content": user}]

        msg = self._call_with_retry(
            system=system_blocks,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        text = _extract_text(msg)
        usage = Usage.from_response(msg.usage, self.model)
        self.cost.add(usage)
        log.info(
            "claude call: in=%d out=%d cache_w=%d cache_r=%d cost=$%.4f",
            usage.input_tokens, usage.output_tokens,
            usage.cache_write_tokens, usage.cache_read_tokens,
            usage.cost_usd,
        )
        return ClaudeResponse(text=text, usage=usage, raw=msg)

    # ------------------------------------------------------------ internals
    @staticmethod
    def _build_system(system: str, cache: bool) -> list[dict[str, Any]]:
        block: dict[str, Any] = {"type": "text", "text": system}
        if cache:
            block["cache_control"] = {"type": "ephemeral"}
        return [block]

    def _call_with_retry(self, **kwargs: Any) -> Any:
        for attempt in range(self.max_retries + 1):
            try:
                return self._sdk.messages.create(model=self.model, **kwargs)
            except anthropic.APIStatusError as e:
                status = getattr(e, "status_code", None)
                # Retry on rate limit (429) and any 5xx.
                if status == 429 or (status is not None and 500 <= status < 600):
                    if attempt >= self.max_retries:
                        log.error("Anthropic call failed after %d attempts: %s",
                                  attempt, e)
                        raise
                    delay = _backoff_seconds(attempt)
                    log.warning("Retryable HTTP %s, sleeping %.1fs (attempt %d/%d)",
                                status, delay, attempt + 1, self.max_retries)
                    time.sleep(delay)
                    continue
                raise
            except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
                if attempt >= self.max_retries:
                    log.error("Anthropic transport error after %d attempts: %s",
                              attempt, e)
                    raise
                delay = _backoff_seconds(attempt)
                log.warning("Transport error, sleeping %.1fs (attempt %d/%d): %s",
                            delay, attempt + 1, self.max_retries, e)
                time.sleep(delay)


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff with full jitter, capped at DEFAULT_BACKOFF_CAP."""
    raw = min(DEFAULT_BACKOFF_CAP, DEFAULT_BACKOFF_BASE ** (attempt + 1))
    return random.uniform(0.0, raw)


def _extract_text(msg: Any) -> str:
    """Concat all text blocks from an Anthropic message response."""
    parts: list[str] = []
    for block in getattr(msg, "content", []) or []:
        # Anthropic block objects expose ``type`` and ``text`` attrs.
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", "") or "")
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text") or "")
    return "".join(parts).strip()


# ---------------------------------------------------------------- JSON extraction

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```",
                             re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> dict[str, Any] | list[Any]:
    """Best-effort JSON extraction.

    Tries, in order:
      1. ``json.loads`` on the whole string (LLM returned pure JSON).
      2. The first ```json``` fenced block.
      3. The first brace-balanced object or array found by linear scan.

    Raises ``ValueError`` if none of the above produces parseable JSON. The
    error includes a short text preview for log-level debugging.
    """
    s = text.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    m = _FENCED_JSON_RE.search(s)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    chunk = _scan_balanced(s)
    if chunk is not None:
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            pass
    preview = s[:240].replace("\n", " ")
    raise ValueError(f"No parseable JSON in model output; head={preview!r}")


def _scan_balanced(s: str) -> str | None:
    """Return the first brace-balanced ``{...}`` or ``[...]`` substring."""
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = s.find(open_c)
        while start != -1:
            depth = 0
            in_string = False
            escape = False
            for i in range(start, len(s)):
                ch = s[i]
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                elif ch == open_c:
                    depth += 1
                elif ch == close_c:
                    depth -= 1
                    if depth == 0:
                        return s[start:i + 1]
            start = s.find(open_c, start + 1)
    return None
