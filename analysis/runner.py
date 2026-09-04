"""High-level orchestration: analyze a single ticker end-to-end.

Stitches together :mod:`analysis.data_access`, the four evidence analyzers,
the research overlay, and :mod:`analysis.report_generator` into a single
``analyze_ticker`` entry point used by both ``run_analysis.py`` and the
dashboard.

The Layer 2 composite is the only score; Layer 3 produces no numbers, just
the categorical overlay and the memo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from data.db import Database
from data.utils import get_logger

from . import earnings_analyzer, filing_analyzer, insider_analyzer
from . import overlay_analyzer, risk_analyzer, sector_analysis
from .base import AnalyzerContext, maybe_run
from .data_access import CompositeArtifact, list_sector
from .quant_context import QuantContext, quant_context
from .report_generator import build_memo, save_memo

log = get_logger("analysis.runner")


@dataclass
class TickerAnalysis:
    """Bundle of every Layer 3 artifact produced for one ticker."""
    ticker: str
    composite: CompositeArtifact | None
    quant_ctx: QuantContext | None = None
    earnings:  dict[str, Any] | None = None
    filing:    dict[str, Any] | None = None
    risk:      dict[str, Any] | None = None
    insider:   dict[str, Any] | None = None
    overlay:   dict[str, Any] | None = None
    sector_view: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)


def analyze_ticker(
    db: Database,
    ctx: AnalyzerContext,
    ticker: str,
    *,
    run_earnings: bool = True,
    run_filing:   bool = True,
    run_risk:     bool = True,
    run_insider:  bool = True,
    run_overlay:  bool = True,
) -> TickerAnalysis:
    """Run every enabled analyzer for ``ticker``; synthesize the overlay."""
    qctx = quant_context(db, ticker)
    if qctx is None:
        log.warning("No Layer 2 composite for %s; skipping", ticker)
        return TickerAnalysis(ticker=ticker, composite=None,
                              notes=[f"no Layer 2 composite for {ticker}"])

    out = TickerAnalysis(ticker=ticker, composite=qctx.composite,
                         quant_ctx=qctx)

    if run_earnings:
        out.earnings = maybe_run(earnings_analyzer.analyze, ctx, db, ticker,
                                 label=f"earnings/{ticker}")
    if run_filing:
        out.filing = maybe_run(filing_analyzer.analyze, ctx, db, ticker,
                               label=f"filing/{ticker}")
    if run_risk:
        out.risk = maybe_run(risk_analyzer.analyze, ctx, db, ticker,
                             label=f"risk/{ticker}")
    if run_insider:
        out.insider = maybe_run(insider_analyzer.analyze, ctx, db, ticker,
                                label=f"insider/{ticker}")
    if run_overlay:
        out.overlay = maybe_run(
            overlay_analyzer.analyze, ctx, db, ticker,
            label=f"overlay/{ticker}",
            qctx=qctx,
            earnings_analysis=out.earnings,
            filing_analysis=out.filing,
            risk_analysis=out.risk,
            insider_analysis=out.insider,
        )
    return out


# ---------------------------------------------------------------------------
# Sector orchestration
# ---------------------------------------------------------------------------
def analyze_sector_full(
    db: Database,
    ctx: AnalyzerContext,
    sector: str,
    *,
    per_ticker_kwargs: dict[str, Any] | None = None,
    run_sector_view: bool = True,
) -> tuple[list[TickerAnalysis], dict[str, Any] | None]:
    """Analyze every ticker in ``sector``, then a cross-company view."""
    per_ticker_kwargs = per_ticker_kwargs or {}
    scores = list_sector(db, sector)
    analyses: list[TickerAnalysis] = []
    for s in scores:
        analyses.append(
            analyze_ticker(db, ctx, s.ticker, **per_ticker_kwargs)
        )
    sector_view = None
    if run_sector_view:
        overlays = {a.ticker: a.overlay for a in analyses}
        sector_view = sector_analysis.analyze_sector(
            ctx, sector, [a.composite for a in analyses if a.composite],
            overlays,
        )
    return analyses, sector_view


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------
def write_report(analysis: TickerAnalysis,
                 sector_view: dict[str, Any] | None = None,
                 run_date: str | None = None) -> str | None:
    """Render and save the markdown memo. Returns the path written."""
    if analysis.quant_ctx is None:
        return None
    memo = build_memo(
        quant_ctx=analysis.quant_ctx,
        overlay=analysis.overlay,
        earnings=analysis.earnings,
        filing=analysis.filing,
        risk=analysis.risk,
        insider=analysis.insider,
        sector_view=sector_view or analysis.sector_view,
    )
    path = save_memo(memo, analysis.ticker, run_date=run_date)
    return str(path)
