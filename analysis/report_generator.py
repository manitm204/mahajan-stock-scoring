"""Generate the research memo from the Layer 2 quant picture + Layer 3 overlay.

One markdown file per ticker, saved to::

    output/reports/YYYY-MM-DD/TICKER.md

Sections (in order):
    1.  Header - research status + quant signal review + composite score
    2.  Quant Summary - why the model likes/dislikes the stock
    3.  Key Factor Drivers - parent scores, strong/weak drivers, sub drivers
    4.  Earnings Call Takeaways
    5.  Filing Analysis
    6.  Risk Analysis
    7.  Insider Activity
    8.  Confirmation vs Contradiction - the overlay's core answer
    9.  What Could Change The Thesis
    10. Final Research View

The Layer 2 composite is the only score in the memo; the overlay contributes
categoricals and evidence, never numbers or recommendations.

The generator is pure formatting - no LLM calls, no DB lookups beyond what
the caller supplies in the bundle.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from data.config import load_config

from .quant_context import (
    STRONG_DRIVER_MIN,
    WEAK_DRIVER_MAX,
    QuantContext,
)

STATUS_LABELS = {
    "PASS":           "PASS - qualitative evidence supports the quant signal",
    "WATCHLIST":      "WATCHLIST - interesting but uncertain",
    "REVIEW":         "REVIEW - contradictions require human review",
    "AVOID_RED_FLAG": "AVOID (RED FLAG) - serious concerns identified",
}


def _bullets(items: list[Any] | None) -> str:
    if not items:
        return "_(none)_"
    out = []
    for it in items:
        if isinstance(it, dict):
            text = it.get("summary") or it.get("why") or it.get("note") or str(it)
        else:
            text = str(it)
        out.append(f"- {text}")
    return "\n".join(out)


def _section(label: str, items: list[Any] | None, lines: list[str]) -> None:
    """Append a bold section label + blank line + bullets to ``lines``."""
    lines.append(f"**{label}:**")
    lines.append("")
    lines.append(_bullets(items))
    lines.append("")


def _kv(label: str, value: Any, default: str = "n/a") -> str:
    if value in (None, "", []):
        value = default
    return f"- **{label}:** {value}"


def _fmt_score(value: float | None) -> str:
    return f"{value:.1f}" if isinstance(value, (int, float)) else "n/a"


def build_memo(
    *,
    quant_ctx: QuantContext,
    overlay:   dict[str, Any] | None,
    earnings:  dict[str, Any] | None,
    filing:    dict[str, Any] | None,
    risk:      dict[str, Any] | None,
    insider:   dict[str, Any] | None,
    sector_view: dict[str, Any] | None = None,
) -> str:
    """Render the markdown memo. Caller decides where to write it."""
    composite = quant_ctx.composite
    t = composite.ticker
    o = overlay or {}
    status = o.get("final_research_status")

    lines: list[str] = []
    title = f"{quant_ctx.company_name} ({t})" if quant_ctx.company_name else t
    lines.append(f"# {title} - Research Memo")
    lines.append(f"_Generated for Mahajan Hedge Fund, as of {composite.as_of_date}_")
    lines.append("")
    lines.append(f"**Research status: {STATUS_LABELS.get(status, '_(no overlay available)_')}**")
    if o.get("quant_signal_review"):
        lines.append(_kv("Quant signal review", o.get("quant_signal_review")))
    lines.append(_kv("Quant composite (official ranking score)",
                     _fmt_score(composite.composite_score)))
    lines.append(_kv("Signal", composite.long_short_flag))
    lines.append("")

    # --- 1. Quant summary --------------------------------------------------
    lines.append("## 1. Quant Summary")
    lines.append(_kv("Sector", composite.sector))
    if quant_ctx.industry:
        lines.append(_kv("Industry", quant_ctx.industry))
    lines.append(_kv("Composite score", _fmt_score(composite.composite_score)))
    if quant_ctx.universe_percentile is not None:
        lines.append(_kv("Universe percentile",
                         f"{quant_ctx.universe_percentile:.0f} of "
                         f"{quant_ctx.universe_size} names"))
    lines.append(_kv("Sector rank", composite.sector_rank))
    if quant_ctx.regime:
        lines.append(_kv("Market regime", quant_ctx.regime))
    ret = quant_ctx.returns
    if ret:
        perf = "  ".join(
            f"{label} {ret[col]:+.1%}"
            for col, label in (("return_20d", "1M"), ("return_60d", "3M"),
                               ("return_252d", "12M"))
            if ret.get(col) is not None
        )
        if perf:
            lines.append(_kv("Recent performance", perf))
    lines.append("")

    # --- 2. Key factor drivers ----------------------------------------------
    lines.append("## 2. Key Factor Drivers")
    if quant_ctx.parent_scores:
        lines.append("| Factor | Score | |")
        lines.append("|---|---|---|")
        for factor, score in sorted(quant_ctx.parent_scores.items(),
                                    key=lambda kv: kv[1], reverse=True):
            tag = ""
            if score >= STRONG_DRIVER_MIN:
                tag = "strong driver"
            elif score <= WEAK_DRIVER_MAX:
                tag = "weak driver"
            lines.append(f"| {factor} | {score:.1f} | {tag} |")
        if quant_ctx.sub_drivers:
            lines.append("")
            lines.append("**Notable sub-factor drivers:**")
            lines.append("")
            for s in quant_ctx.sub_drivers:
                raw = s.get("raw_value")
                raw_txt = f" (raw {raw:.4g})" if isinstance(raw, (int, float)) else ""
                lines.append(f"- {s['factor']}/{s['sub_factor']}: "
                             f"{float(s['score']):.1f}{raw_txt}")
    else:
        lines.append("  _(no parent factor breakdown available)_")
    lines.append("")

    # --- 3. Earnings call takeaways -----------------------------------------
    lines.append("## 3. Earnings Call Takeaways")
    if earnings:
        mc = earnings.get("management_confidence") or {}
        lines.append(_kv("Management tone", mc.get("tone")))
        lines.append(_kv("Evidence", mc.get("evidence")))
        gq = earnings.get("guidance_quality") or {}
        lines.append(_kv("Guidance", gq.get("direction")))
        lines.append(_kv("Guidance credibility", gq.get("credibility")))
        cp = earnings.get("competitive_position") or {}
        lines.append(_kv("Competitive trajectory", cp.get("trajectory")))
        quotes = earnings.get("important_quotes") or []
        if quotes:
            lines.append("**Key quotes:**")
            lines.append("")
            for q in quotes[:4]:
                lines.append(f"- _{q.get('speaker','?')}:_ \"{q.get('quote','')}\" "
                             f"- {q.get('why_it_matters','')}")
        if earnings.get("overall_summary"):
            lines.append(f"\n{earnings['overall_summary']}")
    else:
        lines.append("  _(no earnings call analysis available)_")
    lines.append("")

    # --- 4. Filing analysis --------------------------------------------------
    lines.append("## 4. Filing Analysis")
    if filing:
        lines.append(_kv("Earnings quality score", filing.get("earnings_quality_score")))
        lines.append(_kv("Balance sheet score",    filing.get("balance_sheet_score")))
        lines.append(_kv("Accounting risk",        filing.get("accounting_risk")))
        lines.append("")
        _section("Green flags", filing.get("green_flags"), lines)
        _section("Red flags",   filing.get("red_flags"),   lines)
        if filing.get("overall_summary"):
            lines.append(f"\n{filing['overall_summary']}")
    else:
        lines.append("  _(no filing analysis available)_")
    lines.append("")

    # --- 5. Risk analysis ------------------------------------------------------
    lines.append("## 5. Risk Analysis")
    if risk:
        lines.append(_kv("Boilerplate fraction", risk.get("boilerplate_fraction")))
        lines.append("")
        _section("Newly introduced risks", risk.get("newly_introduced_risks"), lines)
        _section("Removed risks",          risk.get("removed_risks"),          lines)
        _section("Material risks",         risk.get("material_risks"),         lines)
        if risk.get("overall_assessment"):
            lines.append(f"\n{risk['overall_assessment']}")
    else:
        lines.append("  _(no risk-factor analysis available)_")
    lines.append("")

    # --- 6. Insider activity ----------------------------------------------------
    lines.append("## 6. Insider Activity")
    if insider:
        interp = o.get("insider_signal_interpretation")
        if interp and interp != "UNKNOWN":
            lines.append(_kv("Overlay interpretation", interp))
        lines.append(_kv("Analyzer signal", insider.get("signal")))
        lines.append(_kv("Reasoning", insider.get("reasoning")))
        important = insider.get("important_transactions") or []
        if important:
            lines.append("**Notable transactions:**")
            lines.append("")
            for tx in important[:5]:
                lines.append(
                    f"- {tx.get('date','?')} {tx.get('insider','?')} "
                    f"({tx.get('title','?')}) {tx.get('action','?')} "
                    f"${(tx.get('value_usd') or 0):,.0f} - "
                    f"{tx.get('why_it_matters','')}"
                )
    else:
        lines.append("  _(no insider activity in window)_")
    lines.append("")

    # --- 7. Confirmation vs contradiction -----------------------------------
    lines.append("## 7. Confirmation vs Contradiction")
    if overlay:
        lines.append(
            f"Does the qualitative evidence confirm, weaken, or contradict the "
            f"quant signal? **{o.get('quant_signal_review') or 'UNKNOWN'}** "
            f"(absolute view: {o.get('thesis_alignment') or 'UNKNOWN'}; "
            f"qualitative risk: {o.get('qualitative_risk_level') or 'UNKNOWN'})"
        )
        lines.append("")
        _section("Confirming evidence",   o.get("key_confirming_evidence"),   lines)
        _section("Contradicting evidence", o.get("key_contradicting_evidence"), lines)
        _section("Red flags",             o.get("red_flags"),                  lines)
        _section("Open questions",        o.get("open_questions"),             lines)
    else:
        lines.append("  _(no overlay available)_")
    lines.append("")

    # --- 8. What could change the thesis --------------------------------------
    lines.append("## 8. What Could Change The Thesis")
    wct = o.get("what_could_change_thesis") or {}
    if any(wct.get(k) for k in ("bull_breakers", "bear_breakers", "catalysts")):
        _section("Bull thesis breakers", wct.get("bull_breakers"), lines)
        _section("Bear thesis breakers", wct.get("bear_breakers"), lines)
        _section("Catalysts",            wct.get("catalysts"),     lines)
    else:
        lines.append("  _(not assessed)_")
    lines.append("")

    # --- Sector context (optional) ---------------------------------------
    if sector_view:
        lines.append("## 9. Sector Context")
        for k in ("most_attractive_long", "highest_risk_short_candidate",
                  "best_management", "highest_business_quality"):
            v = sector_view.get(k)
            if isinstance(v, dict):
                lines.append(f"- **{k.replace('_', ' ').title()}:** "
                             f"{v.get('ticker','?')} - {v.get('why','')}")
        if sector_view.get("sector_outlook"):
            lines.append(f"\n_Sector outlook:_ {sector_view['sector_outlook']}")
        lines.append("")
        final_section = "10"
    else:
        final_section = "9"

    # --- Final research view -----------------------------------------------
    lines.append(f"## {final_section}. Final Research View")
    lines.append(_final_view(t, composite.long_short_flag, o))
    lines.append("")
    return "\n".join(lines)


def _final_view(ticker: str, flag: str | None, o: dict[str, Any]) -> str:
    """Concise analyst conclusion - status + review, no recommendation."""
    status = o.get("final_research_status")
    if not status:
        return (f"No overlay was produced for {ticker}; the quant signal "
                f"({flag or 'n/a'}) stands unexamined. Run Layer 3 before acting.")
    review = o.get("quant_signal_review") or "UNKNOWN"
    parts = [
        f"**{ticker}** carries a **{status}** research status: the qualitative "
        f"evidence **{review.lower()}** the quant {flag or 'n/a'} signal."
    ]
    biggest = (o.get("open_questions") or [None])[0]
    flags = o.get("red_flags") or []
    if flags:
        parts.append(f" Primary concern: {flags[0]}")
    if biggest:
        parts.append(f" Open question for the analyst: {biggest}")
    parts.append(" The quant composite remains the official ranking; this view "
                 "informs the human approval decision only.")
    return "".join(parts)


def save_memo(memo: str, ticker: str,
              run_date: date | str | None = None,
              output_root: Path | None = None) -> Path:
    """Write the memo to ``output/reports/YYYY-MM-DD/TICKER.md`` and return path."""
    cfg = load_config()
    if output_root is None:
        output_root = cfg.root / "output" / "reports"
    if run_date is None:
        run_date = date.today().isoformat()
    elif isinstance(run_date, date):
        run_date = run_date.isoformat()
    dest = Path(output_root) / run_date / f"{ticker}.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(memo, encoding="utf-8")
    return dest
