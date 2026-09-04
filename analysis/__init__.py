"""Mahajan Hedge Fund - Layer 3 (Qualitative Research Engine).

This package owns LLM-driven qualitative analysis. It never produces alpha
through calculation; Layer 2 already ranks the universe quantitatively.
Layer 3's job is to answer the questions a senior analyst would ask once a
ticker is on the conviction list:

    * Does management appear trustworthy?
    * Is capital allocation intelligent?
    * Has the investment thesis improved or deteriorated?
    * Are there accounting red flags?
    * If we owned this company for five years, what concerns us most?

Layer 3 never scores or ranks: the Layer 2 composite is the sole ranking
engine. Outputs are a categorical research overlay (``overlay_analyzer`` /
``overlay_store``, research status PASS / WATCHLIST / REVIEW /
AVOID_RED_FLAG) and a markdown research memo (``report_generator``), both
purely informational for the human at the approvals step.

Design constraints:
    * All API calls go through :mod:`analysis.api_client` (prompt caching,
      retries, JSON extraction).
    * All spend is tracked through :class:`analysis.cost_tracker.CostTracker`
      and aborts above a configurable hard budget.
    * Identical analyses are deduped by SQLite cache keyed on artifact hash
      and prompt version - we never pay twice for the same input.
"""

__version__ = "1.0.0"
__layer__ = 3
__default_model__ = "claude-sonnet-4-5"

# Prompt versions are bumped any time the wording of a system prompt changes
# so the cache layer invalidates stale entries automatically.
PROMPT_VERSIONS = {
    "earnings":   "v1",
    "filing":     "v2",   # v2: reads Item 7 MD&A prose, not the metrics table
    "risk":       "v1",
    "insider":    "v1",
    "overlay":    "v1",   # replaces "thesis"; categorical due-diligence overlay
    "sector":     "v2",   # v2: consumes overlay summaries instead of theses
}
