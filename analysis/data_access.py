"""Read-only adapter from Layer 1/2 SQLite into Layer 3 analyzer inputs.

Every analyzer needs a small slice of warehouse data: the latest transcript,
8 quarters of fundamental features, the current and prior 10-K Risk Factors
sections, a window of insider transactions, the Layer 2 composite score.
Centralising those queries here keeps the analyzers focused on prompting
and lets the cache hash each artifact deterministically.

NB: no analytics here, just SQL. No HTTP. No mutation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from data.db import Database

# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------
@dataclass
class TranscriptArtifact:
    ticker: str
    fiscal_year: int
    fiscal_quarter: int
    call_date: str | None
    text: str

    @property
    def artifact_id(self) -> str:
        return f"transcript:{self.fiscal_year}Q{self.fiscal_quarter}"


def latest_transcript(db: Database, ticker: str) -> TranscriptArtifact | None:
    row = db.query_one(
        "SELECT ticker, fiscal_year, fiscal_quarter, call_date, transcript_text "
        "FROM transcripts WHERE ticker = ? AND transcript_text IS NOT NULL "
        "ORDER BY fiscal_year DESC, fiscal_quarter DESC LIMIT 1",
        (ticker,),
    )
    if not row or not row["transcript_text"]:
        return None
    return TranscriptArtifact(
        ticker=row["ticker"],
        fiscal_year=int(row["fiscal_year"]),
        fiscal_quarter=int(row["fiscal_quarter"]),
        call_date=row["call_date"],
        text=row["transcript_text"],
    )


# ---------------------------------------------------------------------------
# Fundamentals - 8-quarter slice for forensic accounting
# ---------------------------------------------------------------------------
FUNDAMENTAL_FEATURE_COLS = [
    "revenue_growth_yoy", "revenue_growth_qoq",
    "eps_growth_yoy", "eps_growth_qoq",
    "revenue_cagr_3y", "eps_cagr_3y",
    "roe", "roic",
    "gross_margin", "operating_margin", "net_margin",
    "cfo_to_net_income",
    "asset_turnover",
    "debt_to_equity", "current_ratio", "interest_coverage",
    "net_debt_to_ebitda",
    "fcf_yield",
    "price_to_sales", "ev_to_ebitda",
]


@dataclass
class FundamentalsArtifact:
    ticker: str
    sector: str
    quarters: list[dict[str, Any]]   # newest first

    @property
    def artifact_id(self) -> str:
        if not self.quarters:
            return "fundamentals:empty"
        first = self.quarters[0]["fiscal_date"]
        last = self.quarters[-1]["fiscal_date"]
        return f"fundamentals:{last}_to_{first}"


def fundamentals_window(db: Database, ticker: str, n_quarters: int = 8) -> FundamentalsArtifact | None:
    """Newest ``n_quarters`` of quarterly fundamental features for ``ticker``."""
    cols = ", ".join(["fiscal_date"] + FUNDAMENTAL_FEATURE_COLS)
    rows = db.query(
        f"SELECT {cols} FROM fundamental_features "
        f"WHERE ticker = ? AND period_type = 'quarterly' "
        f"ORDER BY fiscal_date DESC LIMIT ?",
        (ticker, n_quarters),
    )
    if not rows:
        return None
    sector_row = db.query_one(
        "SELECT gics_sector FROM universe WHERE ticker = ?", (ticker,))
    sector = sector_row["gics_sector"] if sector_row else "Unknown"
    return FundamentalsArtifact(
        ticker=ticker,
        sector=sector or "Unknown",
        quarters=[dict(r) for r in rows],
    )


def format_metrics_table(quarters: list[dict[str, Any]]) -> str:
    """Render the 8-quarter feature matrix as a compact text table.

    Token-efficient: column header is the fiscal date, rows are the metric
    names. Values are rounded so the model is not distracted by spurious
    precision and the cache key is stable across reruns.
    """
    if not quarters:
        return "(no quarterly fundamentals available)"
    dates = [q["fiscal_date"] for q in quarters]
    header = "Metric              | " + " | ".join(d[:10] for d in dates)
    lines = [header, "-" * len(header)]
    for col in FUNDAMENTAL_FEATURE_COLS:
        cells = []
        for q in quarters:
            v = q.get(col)
            cells.append(_fmt_num(v))
        lines.append(f"{col:19s} | " + " | ".join(c.rjust(10) for c in cells))
    return "\n".join(lines)


def _fmt_num(v: Any) -> str:
    if v is None:
        return "  -"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)[:10]
    if abs(f) >= 100:
        return f"{f:,.0f}"
    return f"{f:.3f}"


# ---------------------------------------------------------------------------
# 10-K Risk Factors (Item 1A) extraction
# ---------------------------------------------------------------------------
@dataclass
class RiskArtifact:
    ticker: str
    current_filing_date: str
    current_text: str
    prior_filing_date: str | None
    prior_text: str | None

    @property
    def artifact_id(self) -> str:
        prior = self.prior_filing_date or "none"
        return f"risk:{self.current_filing_date}_vs_{prior}"


# These patterns are intentionally permissive - SEC filings vary wildly in
# whitespace/HTML structure. We just need to find a "Risk Factors" section
# that has plausible length (>2k chars) and stop at the next item header.
_ITEM_1A_START = re.compile(
    r"item\s*1a[\.\s]*risk\s*factors", re.IGNORECASE)
_ITEM_1A_STOP = re.compile(
    r"item\s*1b[\.\s]*|item\s*2[\.\s]*properties", re.IGNORECASE)


def extract_risk_section(filing_text: str) -> str | None:
    """Return the cleaned Item 1A. Risk Factors section, or ``None``."""
    if not filing_text:
        return None
    cleaned = strip_html(filing_text)
    start_match = _ITEM_1A_START.search(cleaned)
    if not start_match:
        return None
    body = cleaned[start_match.end():]
    stop_match = _ITEM_1A_STOP.search(body)
    section = body[:stop_match.start()] if stop_match else body
    # Collapse runs of whitespace and strip xbrl/numeric residue lines.
    section = re.sub(r"\s+", " ", section).strip()
    if len(section) < 2000:
        return None
    return section


def strip_html(text: str) -> str:
    """Defensive HTML / XBRL stripper for SEC filing payloads."""
    if not text:
        return ""
    # Drop script/style blocks fully so we don't surface CSS to the model.
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text,
                  flags=re.IGNORECASE | re.DOTALL)
    # Drop every tag.
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode the small handful of HTML entities that show up in 10-Ks.
    text = (text
            .replace("&nbsp;", " ")
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", '"')
            .replace("&#8217;", "'")
            .replace("&#8220;", '"')
            .replace("&#8221;", '"'))
    return text


def risk_artifact(db: Database, ticker: str) -> RiskArtifact | None:
    """Build a RiskArtifact pairing the latest 10-K with the prior 10-K."""
    rows = db.query(
        "SELECT filing_date, filing_text FROM sec_filings "
        "WHERE ticker = ? AND form_type = '10-K' AND filing_text IS NOT NULL "
        "ORDER BY filing_date DESC LIMIT 2",
        (ticker,),
    )
    if not rows:
        return None
    current_section = extract_risk_section(rows[0]["filing_text"])
    if current_section is None:
        return None
    prior_section = None
    prior_date = None
    if len(rows) >= 2:
        prior_section = extract_risk_section(rows[1]["filing_text"])
        prior_date = rows[1]["filing_date"]
    return RiskArtifact(
        ticker=ticker,
        current_filing_date=rows[0]["filing_date"],
        current_text=current_section,
        prior_filing_date=prior_date,
        prior_text=prior_section,
    )


# ---------------------------------------------------------------------------
# 10-K MD&A (Item 7) extraction - the filing analyzer's orthogonal source.
#
# The forensic read used to consume the 8-quarter metrics table, which Layer 2
# has already scored numerically (heavy overlap with the quality factor). We
# instead feed management's own discussion prose - liquidity, segment
# commentary, critical accounting estimates - which the quant factors never
# see. The output schema is unchanged so the overlay / report_generator
# keep working.
# ---------------------------------------------------------------------------
@dataclass
class MDAArtifact:
    ticker: str
    sector: str
    filing_date: str
    form_type: str
    text: str

    @property
    def artifact_id(self) -> str:
        return f"mdna:{self.form_type}:{self.filing_date}"


# MD&A lives under a different item number by form. Require the word
# "management" after the item marker so we don't match the adjacent
# "Item 7A" / "Item 3" market-risk headers. The warehouse holds far more
# 10-Qs than 10-Ks, so quarterly Item 2 is the primary coverage source.
#   10-K -> Item 7   (stop at Item 7A, or Item 8 financial statements)
#   10-Q -> Item 2   (stop at Item 3 market risk, or Item 4 controls)
_MDNA_PATTERNS: dict[str, tuple[re.Pattern[str], re.Pattern[str]]] = {
    "10-K": (
        re.compile(r"item\s*7[\.\s]*management[\'’]?s?\s*discussion", re.IGNORECASE),
        re.compile(r"item\s*7a[\.\s]*|item\s*8[\.\s]*financial", re.IGNORECASE),
    ),
    "10-Q": (
        re.compile(r"item\s*2[\.\s]*management[\'’]?s?\s*discussion", re.IGNORECASE),
        re.compile(r"item\s*3[\.\s]*quantitative|item\s*4[\.\s]*controls", re.IGNORECASE),
    ),
}

_MDNA_MIN_CHARS = 2000
# The bare MD&A phrase, item-number-agnostic - used only by the Tier 2 fallback.
_MDNA_PHRASE = re.compile(
    r"management[\'’]?s?\s+discussion\s+and\s+analysis", re.IGNORECASE)
# A near-universal MD&A subsection heading. The Tier 2 fallback casts a wide
# net, so we require this marker to be present before accepting a span - it
# guarantees we surfaced actual MD&A prose, not financial-statement notes.
_MDNA_MARKER = re.compile(r"liquidity\s+and\s+capital\s+resources", re.IGNORECASE)


def extract_mdna_section(filing_text: str, form_type: str = "10-K") -> str | None:
    """Return the cleaned MD&A section for ``form_type``, or ``None``.

    Tier 1 (the common case): scan every "Item N Management's Discussion"
    header and return the first whose following stop header bounds a full
    (>=2000 char) section. This skips table-of-contents references, which are
    followed almost immediately by the next item header and fall under the
    length gate.

    Tier 2 (fallback): some filers cluster the item *titles* together (a
    part index), so the header is immediately followed by the next item title
    and the real body hides between later stop headers. Only when Tier 1
    finds nothing do we run the wider, marker-guarded recovery.
    """
    pats = _MDNA_PATTERNS.get(form_type)
    if not filing_text or pats is None:
        return None
    start_re, stop_re = pats
    cleaned = strip_html(filing_text)
    for start_match in start_re.finditer(cleaned):
        body = cleaned[start_match.end():]
        stop_match = stop_re.search(body)
        section = body[:stop_match.start()] if stop_match else body
        section = re.sub(r"\s+", " ", section).strip()
        if len(section) >= _MDNA_MIN_CHARS:
            return section
    return _recover_clustered_mdna(cleaned, stop_re)


def _recover_clustered_mdna(cleaned: str, stop_re: re.Pattern[str]) -> str | None:
    """Longest MD&A-looking span for filings that cluster their item titles.

    Considers two candidate families and keeps the longest span that carries
    the Liquidity & Capital Resources heading (so non-MD&A prose is rejected):
      (a) each MD&A phrase -> the first stop header at least MIN chars past it
          (jumps over an adjacent title-cluster stop to the real terminator);
      (b) each span between consecutive stop headers (where a clustered body
          sits when it has no header of its own directly in front of it).
    """
    candidates: list[str] = []
    stops = [m.start() for m in stop_re.finditer(cleaned)] + [len(cleaned)]
    for m in _MDNA_PHRASE.finditer(cleaned):
        a = m.end()
        body = cleaned[a:]
        cut = next((s.start() for s in stop_re.finditer(body)
                    if s.start() >= _MDNA_MIN_CHARS), None)
        candidates.append(body[:cut] if cut is not None else body)
    for i in range(len(stops) - 1):
        candidates.append(cleaned[stops[i]:stops[i + 1]])
    best = ""
    for cand in candidates:
        section = re.sub(r"\s+", " ", cand).strip()
        if (len(section) >= _MDNA_MIN_CHARS
                and _MDNA_MARKER.search(section)
                and len(section) > len(best)):
            best = section
    return best or None


def mdna_artifact(db: Database, ticker: str) -> MDAArtifact | None:
    """Build an MDAArtifact from the latest 10-K or 10-Q MD&A section.

    Prefers the most recent filing of either type; falls through to older
    filings if the newest one has no extractable MD&A (malformed text).
    """
    rows = db.query(
        "SELECT form_type, filing_date, filing_text FROM sec_filings "
        "WHERE ticker = ? AND form_type IN ('10-K', '10-Q') "
        "  AND filing_text IS NOT NULL "
        "ORDER BY filing_date DESC LIMIT 5",
        (ticker,),
    )
    for row in rows:
        section = extract_mdna_section(row["filing_text"], row["form_type"])
        if section is None:
            continue
        sector_row = db.query_one(
            "SELECT gics_sector FROM universe WHERE ticker = ?", (ticker,))
        sector = (sector_row["gics_sector"] if sector_row else "Unknown") or "Unknown"
        return MDAArtifact(
            ticker=ticker,
            sector=sector,
            filing_date=row["filing_date"],
            form_type=row["form_type"],
            text=section,
        )
    return None


# ---------------------------------------------------------------------------
# Insider transactions window
# ---------------------------------------------------------------------------
@dataclass
class InsiderArtifact:
    ticker: str
    window_days: int
    window_start: str
    window_end: str
    transactions: list[dict[str, Any]]
    n_buys: int
    n_sells: int
    total_buy_value: float
    total_sell_value: float
    unique_buyers: int
    unique_sellers: int

    @property
    def artifact_id(self) -> str:
        return f"insider:{self.window_start}_to_{self.window_end}"


def insider_artifact(db: Database, ticker: str, window_days: int = 180,
                     as_of: str | None = None,
                     max_rows: int = 40) -> InsiderArtifact | None:
    """Aggregate Form 4 activity over a trailing window for the analyzer.

    Returns ``None`` if there are no transactions at all in the window.
    """
    if as_of is None:
        end_dt = datetime.utcnow().date()
    else:
        end_dt = datetime.strptime(as_of, "%Y-%m-%d").date()
    start_dt = end_dt - timedelta(days=window_days)
    rows = db.query(
        "SELECT insider_name, insider_title, transaction_type, transaction_code, "
        "       shares, price, value, transaction_date, is_purchase "
        "FROM insider_transactions "
        "WHERE ticker = ? AND transaction_date BETWEEN ? AND ? "
        "ORDER BY transaction_date DESC, value DESC",
        (ticker, start_dt.isoformat(), end_dt.isoformat()),
    )
    if not rows:
        return None
    transactions = [dict(r) for r in rows]
    buys  = [t for t in transactions if t.get("is_purchase") == 1]
    sells = [t for t in transactions if t.get("is_purchase") == 0]
    return InsiderArtifact(
        ticker=ticker,
        window_days=window_days,
        window_start=start_dt.isoformat(),
        window_end=end_dt.isoformat(),
        transactions=transactions[:max_rows],
        n_buys=len(buys),
        n_sells=len(sells),
        total_buy_value=sum((t.get("value") or 0) for t in buys),
        total_sell_value=sum((t.get("value") or 0) for t in sells),
        unique_buyers=len({t["insider_name"] for t in buys if t.get("insider_name")}),
        unique_sellers=len({t["insider_name"] for t in sells if t.get("insider_name")}),
    )


def format_insider_table(transactions: list[dict[str, Any]]) -> str:
    if not transactions:
        return "(no transactions in window)"
    lines = ["date       | insider                     | title         | code | shares       | $value"]
    lines.append("-" * len(lines[0]))
    for t in transactions:
        name  = (t.get("insider_name") or "")[:27]
        title = (t.get("insider_title") or "")[:13]
        code  = (t.get("transaction_code") or "")[:4]
        shares = t.get("shares") or 0
        value  = t.get("value")  or 0
        lines.append(
            f"{t.get('transaction_date','')[:10]:10s} | "
            f"{name:27s} | {title:13s} | {code:4s} | "
            f"{shares:>12,.0f} | {value:>14,.0f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Layer 2 composite score lookup
# ---------------------------------------------------------------------------
@dataclass
class CompositeArtifact:
    ticker: str
    sector: str
    composite_score: float
    sector_rank: int | None
    long_short_flag: str | None
    as_of_date: str


def latest_composite(db: Database, ticker: str) -> CompositeArtifact | None:
    row = db.query_one(
        "SELECT ticker, sector, composite_score, sector_rank, "
        "       long_short_flag, as_of_date "
        "FROM composite_scores WHERE ticker = ? "
        "ORDER BY as_of_date DESC LIMIT 1",
        (ticker,),
    )
    if not row:
        return None
    return CompositeArtifact(
        ticker=row["ticker"],
        sector=row["sector"] or "Unknown",
        composite_score=float(row["composite_score"] or 0),
        sector_rank=row["sector_rank"],
        long_short_flag=row["long_short_flag"],
        as_of_date=row["as_of_date"],
    )


def list_candidates(db: Database, flag: str, limit: int = 25,
                    as_of: str | None = None) -> list[CompositeArtifact]:
    """Top ``limit`` names with ``long_short_flag`` = LONG or SHORT.

    LONG candidates rank by descending composite_score; SHORT by ascending
    (lowest composite = best short).
    """
    flag = flag.upper()
    order = "DESC" if flag == "LONG" else "ASC"
    if as_of is None:
        # Use the most recent scoring date in the table.
        latest = db.query_one(
            "SELECT MAX(as_of_date) AS d FROM composite_scores")
        if not latest or not latest["d"]:
            return []
        as_of = latest["d"]
    rows = db.query(
        f"SELECT ticker, sector, composite_score, sector_rank, "
        f"       long_short_flag, as_of_date "
        f"FROM composite_scores "
        f"WHERE long_short_flag = ? AND as_of_date = ? "
        f"ORDER BY composite_score {order} LIMIT ?",
        (flag, as_of, limit),
    )
    return [
        CompositeArtifact(
            ticker=r["ticker"], sector=r["sector"] or "Unknown",
            composite_score=float(r["composite_score"] or 0),
            sector_rank=r["sector_rank"],
            long_short_flag=r["long_short_flag"],
            as_of_date=r["as_of_date"],
        )
        for r in rows
    ]


def list_sector(db: Database, sector: str,
                as_of: str | None = None) -> list[CompositeArtifact]:
    if as_of is None:
        latest = db.query_one("SELECT MAX(as_of_date) AS d FROM composite_scores")
        if not latest or not latest["d"]:
            return []
        as_of = latest["d"]
    rows = db.query(
        "SELECT ticker, sector, composite_score, sector_rank, "
        "       long_short_flag, as_of_date "
        "FROM composite_scores WHERE sector = ? AND as_of_date = ? "
        "ORDER BY composite_score DESC",
        (sector, as_of),
    )
    return [
        CompositeArtifact(
            ticker=r["ticker"], sector=r["sector"] or "Unknown",
            composite_score=float(r["composite_score"] or 0),
            sector_rank=r["sector_rank"],
            long_short_flag=r["long_short_flag"],
            as_of_date=r["as_of_date"],
        )
        for r in rows
    ]
