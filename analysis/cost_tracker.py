"""Token + dollar accounting for every Claude request in a Layer 3 run.

The tracker has three jobs:

1. Sum input / output / cache-read / cache-write tokens reported by
   ``response.usage`` so the operator can see real spend after a run.
2. Estimate the dollar cost of a *prospective* request before it is sent.
3. Refuse to spend above a configurable hard budget (default $1.00).

Pricing is per-million-tokens and varies by model. We keep it as a small
table here so a future model swap is one entry. Numbers are quoted from
Anthropic's published pricing page for Claude Sonnet 4.5.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Model pricing ($ per 1M tokens). Update when Anthropic publishes new rates.
# Keys are the model IDs Anthropic accepts; alias the family so a request
# routed to a specific dated version still resolves.
# ---------------------------------------------------------------------------
PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-5": {
        "input":       3.00,
        "output":     15.00,
        "cache_write": 3.75,
        "cache_read":  0.30,
    },
    "claude-opus-4-7": {
        "input":      15.00,
        "output":     75.00,
        "cache_write":18.75,
        "cache_read":  1.50,
    },
    "claude-haiku-4-5": {
        "input":       1.00,
        "output":      5.00,
        "cache_write": 1.25,
        "cache_read":  0.10,
    },
}


def get_pricing(model: str) -> dict[str, float]:
    """Resolve pricing for a model, falling back to the family root."""
    if model in PRICING:
        return PRICING[model]
    # Strip a trailing dated suffix like ``-20250929`` if present.
    for prefix in PRICING:
        if model.startswith(prefix):
            return PRICING[prefix]
    raise KeyError(f"No pricing entry for model {model!r}")


@dataclass
class Usage:
    """One request's worth of token + cost accounting."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""

    @classmethod
    def from_response(cls, usage_obj: Any, model: str) -> "Usage":
        """Build from an :class:`anthropic.types.Usage` (or dict) object.

        The Anthropic SDK exposes ``input_tokens``, ``output_tokens``,
        ``cache_creation_input_tokens`` and ``cache_read_input_tokens``. Some
        of those fields are ``None`` when caching is not in play, so coerce
        to ``int`` defensively.
        """
        def _get(name: str) -> int:
            value = getattr(usage_obj, name, None)
            if value is None and isinstance(usage_obj, dict):
                value = usage_obj.get(name)
            return int(value or 0)

        u = cls(
            input_tokens=_get("input_tokens"),
            output_tokens=_get("output_tokens"),
            cache_write_tokens=_get("cache_creation_input_tokens"),
            cache_read_tokens=_get("cache_read_input_tokens"),
            model=model,
        )
        u.cost_usd = estimate_cost(
            model,
            u.input_tokens,
            u.output_tokens,
            u.cache_write_tokens,
            u.cache_read_tokens,
        )
        return u


def estimate_cost(model: str, input_tokens: int, output_tokens: int,
                  cache_write_tokens: int = 0, cache_read_tokens: int = 0) -> float:
    """Apply the model's per-1M rates and return dollars."""
    p = get_pricing(model)
    return (
        input_tokens       / 1_000_000 * p["input"]      +
        output_tokens      / 1_000_000 * p["output"]     +
        cache_write_tokens / 1_000_000 * p["cache_write"]+
        cache_read_tokens  / 1_000_000 * p["cache_read"]
    )


@dataclass
class CostTracker:
    """Accumulate spend across a Layer 3 run; veto requests above budget.

    The budget is a hard ceiling; ``check_budget`` raises so that a bad
    estimate cannot accidentally rack up dollars across many small calls.
    """
    budget_usd: float = 1.00
    spent_usd: float = 0.0
    history: list[Usage] = field(default_factory=list)

    # ---- accounting ------------------------------------------------------
    def add(self, usage: Usage) -> None:
        self.history.append(usage)
        self.spent_usd += usage.cost_usd

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.budget_usd - self.spent_usd)

    # ---- gating ----------------------------------------------------------
    def can_afford(self, estimated_cost: float) -> bool:
        return (self.spent_usd + estimated_cost) <= self.budget_usd

    def check_budget(self, estimated_cost: float) -> None:
        """Raise :class:`BudgetExceededError` if a request would blow the cap."""
        if not self.can_afford(estimated_cost):
            raise BudgetExceededError(
                f"Projected cost ${self.spent_usd + estimated_cost:.4f} "
                f"exceeds budget ${self.budget_usd:.2f} "
                f"(spent ${self.spent_usd:.4f}, this call ${estimated_cost:.4f})"
            )

    # ---- reporting -------------------------------------------------------
    def totals(self) -> dict[str, float | int]:
        return {
            "calls":              len(self.history),
            "input_tokens":       sum(u.input_tokens       for u in self.history),
            "output_tokens":      sum(u.output_tokens      for u in self.history),
            "cache_write_tokens": sum(u.cache_write_tokens for u in self.history),
            "cache_read_tokens":  sum(u.cache_read_tokens  for u in self.history),
            "spent_usd":          round(self.spent_usd, 6),
            "budget_usd":         self.budget_usd,
            "remaining_usd":      round(self.remaining_usd, 6),
        }

    def format_summary(self) -> str:
        t = self.totals()
        return (
            f"Cost summary: {t['calls']} calls  "
            f"in={t['input_tokens']:,}  out={t['output_tokens']:,}  "
            f"cache_w={t['cache_write_tokens']:,}  cache_r={t['cache_read_tokens']:,}  "
            f"spent=${t['spent_usd']:.4f}  remaining=${t['remaining_usd']:.4f}"
        )


class BudgetExceededError(RuntimeError):
    """Raised by :meth:`CostTracker.check_budget` when a call would overspend."""
