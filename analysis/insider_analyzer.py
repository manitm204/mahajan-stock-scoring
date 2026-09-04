"""Form 4 insider activity analyzer.

Aggregates trailing-window Form 4 transactions, sends a compact table and
summary stats to Claude, and asks for an interpretation that distinguishes
high-signal open-market buying from routine 10b5-1 plan sales.
"""
from __future__ import annotations

from typing import Any

from data.db import Database

from .base import AnalyzerContext, run_cached
from .data_access import (
    InsiderArtifact,
    insider_artifact,
    format_insider_table,
)
from .prompts import INSIDER_SYSTEM, INSIDER_USER_TEMPLATE

DEFAULT_WINDOW_DAYS = 180
DEFAULT_MAX_TOKENS = 1024


def analyze_insiders(
    ctx: AnalyzerContext,
    artifact: InsiderArtifact,
) -> dict[str, Any] | None:
    table = format_insider_table(artifact.transactions)
    user = INSIDER_USER_TEMPLATE.format(
        ticker=artifact.ticker,
        window_days=artifact.window_days,
        window_start=artifact.window_start,
        window_end=artifact.window_end,
        n_buys=artifact.n_buys,
        n_sells=artifact.n_sells,
        total_buy_value=artifact.total_buy_value,
        total_sell_value=artifact.total_sell_value,
        unique_buyers=artifact.unique_buyers,
        unique_sellers=artifact.unique_sellers,
        transactions_table=table,
    )
    # Hash on the aggregate stats + table so a single late-arriving txn
    # invalidates the cache cleanly.
    payload = {
        "n_buys": artifact.n_buys,
        "n_sells": artifact.n_sells,
        "buy_value": round(artifact.total_buy_value, 2),
        "sell_value": round(artifact.total_sell_value, 2),
        "table": table,
    }
    return run_cached(
        ctx,
        analyzer="insider",
        ticker=artifact.ticker,
        artifact_id=artifact.artifact_id,
        artifact_payload=payload,
        system_prompt=INSIDER_SYSTEM,
        user_prompt=user,
        max_tokens=DEFAULT_MAX_TOKENS,
    )


def analyze(ctx: AnalyzerContext, db: Database, ticker: str,
            window_days: int = DEFAULT_WINDOW_DAYS,
            as_of: str | None = None) -> dict[str, Any] | None:
    art = insider_artifact(db, ticker, window_days=window_days, as_of=as_of)
    if art is None:
        return None
    return analyze_insiders(ctx, art)
