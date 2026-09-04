"""Tests for the Layer 3 research overlay: validation, storage, memo, quant context."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.overlay_analyzer import validate_overlay
from analysis.overlay_store import ensure_schema, persist_overlays
from analysis.quant_context import QuantContext, format_quant_context, quant_context
from analysis.report_generator import build_memo
from analysis.data_access import CompositeArtifact
from analysis.runner import TickerAnalysis
from data.db import Database


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def db(tmp_path: Path) -> Database:
    # Database() applies the Layer 1 schema (universe, price_features,
    # fundamental_features, ...); only the Layer 2 tables need creating here.
    d = Database(path=tmp_path / "test.db", wal=False)
    d._conn.executescript("""
        CREATE TABLE IF NOT EXISTS composite_scores (
            as_of_date TEXT, ticker TEXT, sector TEXT, regime TEXT,
            composite_score REAL, sector_rank INTEGER,
            long_short_flag TEXT, computed_at TEXT,
            PRIMARY KEY (as_of_date, ticker)
        );
        CREATE TABLE IF NOT EXISTS parent_factor_scores (
            as_of_date TEXT, ticker TEXT, factor TEXT, score REAL,
            PRIMARY KEY (as_of_date, ticker, factor)
        );
        CREATE TABLE IF NOT EXISTS sub_factor_scores (
            as_of_date TEXT, ticker TEXT, factor TEXT, sub_factor TEXT,
            score REAL, raw_value REAL,
            PRIMARY KEY (as_of_date, ticker, factor, sub_factor)
        );
    """)
    d._conn.commit()
    yield d
    d.close()


AS_OF = "2026-07-01"


def _seed_ticker(db: Database, ticker: str = "AAPL",
                 score: float = 84.0) -> None:
    db._conn.execute(
        "INSERT INTO composite_scores VALUES (?, ?, 'Information Technology', "
        "'bull', ?, 3, 'LONG', '')",
        (AS_OF, ticker, score))
    db._conn.execute(
        "INSERT INTO composite_scores VALUES (?, 'ZZZZ', 'Energy', "
        "'bull', 20.0, 9, 'SHORT', '')", (AS_OF,))
    for factor, s in (("momentum", 91.0), ("growth", 88.0), ("value", 35.0)):
        db._conn.execute(
            "INSERT INTO parent_factor_scores VALUES (?, ?, ?, ?)",
            (AS_OF, ticker, factor, s))
    db._conn.execute(
        "INSERT INTO sub_factor_scores VALUES (?, ?, 'momentum', 'mom_12_1', "
        "94.0, 0.62)", (AS_OF, ticker))
    db._conn.execute(
        "INSERT INTO universe (ticker, company_name, gics_sector, "
        "gics_sub_industry) VALUES (?, 'Apple Inc', "
        "'Information Technology', 'Technology Hardware')", (ticker,))
    db._conn.execute(
        "INSERT INTO price_features (ticker, date, return_20d, return_60d, "
        "return_252d, volatility_20d, distance_from_52w_high) "
        "VALUES (?, ?, 0.042, 0.118, 0.385, 0.22, -0.03)", (ticker, AS_OF))
    db._conn.execute(
        "INSERT INTO fundamental_features (ticker, period_type, fiscal_date, "
        "fcf_yield, ev_to_ebitda, price_to_sales, roe, roic, debt_to_equity, "
        "revenue_growth_yoy, eps_growth_yoy) "
        "VALUES (?, 'quarterly', '2026-03-31', 0.032, 18.5, 6.2, 0.45, 0.31, "
        "1.4, 0.08, 0.11)",
        (ticker,))
    db._conn.commit()


def _valid_overlay() -> dict:
    return {
        "quant_signal_review": "CONFIRMS",
        "thesis_alignment": "BULLISH",
        "qualitative_risk_level": "LOW",
        "business_quality": "HIGH",
        "management_tone": "CONFIDENT",
        "competitive_position": "IMPROVING",
        "accounting_risk": "LOW",
        "filing_risk": "LOW",
        "insider_signal_interpretation": "MODESTLY_POSITIVE",
        "red_flags": [],
        "open_questions": ["Churn in the services segment"],
        "key_confirming_evidence": ["a", "b", "c"],
        "key_contradicting_evidence": [],
        "what_could_change_thesis": {
            "bull_breakers": ["margin collapse"],
            "bear_breakers": ["guide raise"],
            "catalysts": ["WWDC"],
        },
        "final_research_status": "PASS",
        "confidence": 78,
    }


# ---------------------------------------------------------------------------
# validate_overlay
# ---------------------------------------------------------------------------
def test_validate_passes_clean_payload_through():
    out = validate_overlay(_valid_overlay(), ticker="AAPL")
    assert out["final_research_status"] == "PASS"
    assert out["quant_signal_review"] == "CONFIRMS"
    assert out["confidence"] == 78


def test_validate_forces_bad_status_to_review():
    o = _valid_overlay()
    o["final_research_status"] = "BUY"          # not a research status
    out = validate_overlay(o, ticker="AAPL")
    assert out["final_research_status"] == "REVIEW"


def test_validate_missing_status_becomes_review():
    o = _valid_overlay()
    del o["final_research_status"]
    assert validate_overlay(o)["final_research_status"] == "REVIEW"


def test_validate_coerces_unknown_enum_values():
    o = _valid_overlay()
    o["management_tone"] = "euphoric"           # not in enum -> UNKNOWN
    o["thesis_alignment"] = "SIDEWAYS"          # no UNKNOWN member -> None
    out = validate_overlay(o, ticker="AAPL")
    assert out["management_tone"] == "UNKNOWN"
    assert out["thesis_alignment"] is None


def test_validate_normalizes_case_and_spaces():
    o = _valid_overlay()
    o["final_research_status"] = "avoid red_flag"
    o["insider_signal_interpretation"] = "modestly positive"
    out = validate_overlay(o)
    assert out["final_research_status"] == "AVOID_RED_FLAG"
    assert out["insider_signal_interpretation"] == "MODESTLY_POSITIVE"


def test_validate_coerces_list_and_nested_fields():
    o = _valid_overlay()
    o["red_flags"] = "single string not a list"
    o["what_could_change_thesis"] = "nonsense"
    o["confidence"] = "high"
    out = validate_overlay(o)
    assert out["red_flags"] == ["single string not a list"]
    assert out["what_could_change_thesis"] == {
        "bull_breakers": [], "bear_breakers": [], "catalysts": []}
    assert out["confidence"] is None


# ---------------------------------------------------------------------------
# overlay_store
# ---------------------------------------------------------------------------
def _analysis(ticker: str = "AAPL", overlay: dict | None = None) -> TickerAnalysis:
    composite = CompositeArtifact(
        ticker=ticker, sector="Information Technology",
        composite_score=84.0, sector_rank=3,
        long_short_flag="LONG", as_of_date=AS_OF,
    )
    return TickerAnalysis(ticker=ticker, composite=composite,
                          overlay=overlay)


def test_persist_overlays_roundtrip(db: Database):
    overlay = validate_overlay(_valid_overlay())
    n = persist_overlays(db, [_analysis(overlay=overlay)], model="test-model")
    assert n == 1
    row = db.query_one("SELECT * FROM research_overlays WHERE ticker='AAPL'")
    assert row["research_status"] == "PASS"
    assert row["quant_signal_review"] == "CONFIRMS"
    assert row["composite_score"] == 84.0
    assert row["as_of_date"] == AS_OF
    assert json.loads(row["open_questions"]) == ["Churn in the services segment"]
    full = json.loads(row["overlay_json"])
    assert full["what_could_change_thesis"]["catalysts"] == ["WWDC"]


def test_persist_overlays_upserts_on_rerun(db: Database):
    o1 = validate_overlay(_valid_overlay())
    persist_overlays(db, [_analysis(overlay=o1)])
    o2 = dict(o1, final_research_status="WATCHLIST")
    persist_overlays(db, [_analysis(overlay=o2)])
    rows = db.query("SELECT research_status FROM research_overlays "
                    "WHERE ticker='AAPL'")
    assert len(rows) == 1
    assert rows[0]["research_status"] == "WATCHLIST"


def test_persist_overlays_skips_missing_overlay(db: Database):
    ensure_schema(db)
    assert persist_overlays(db, [_analysis(overlay=None)]) == 0


# ---------------------------------------------------------------------------
# quant_context
# ---------------------------------------------------------------------------
def test_quant_context_assembles_all_sections(db: Database):
    _seed_ticker(db)
    qctx = quant_context(db, "AAPL")
    assert qctx is not None
    assert qctx.composite.composite_score == 84.0
    assert qctx.company_name == "Apple Inc"
    assert qctx.parent_scores == {"momentum": 91.0, "growth": 88.0, "value": 35.0}
    assert qctx.universe_size == 2
    assert qctx.universe_percentile == 100.0
    assert qctx.returns["return_20d"] == pytest.approx(0.042)
    assert qctx.valuation["ev_to_ebitda"] == pytest.approx(18.5)
    assert any(s["sub_factor"] == "mom_12_1" for s in qctx.sub_drivers)


def test_quant_context_none_without_composite(db: Database):
    assert quant_context(db, "NOPE") is None


def test_format_quant_context_renders_drivers(db: Database):
    _seed_ticker(db)
    block = format_quant_context(quant_context(db, "AAPL"))
    assert "Composite score: 84.0" in block
    assert "momentum" in block and "strong driver" in block
    assert "value" in block and "weak driver" in block
    assert "Signal: LONG" in block
    assert "12M +38.5%" in block


def test_format_quant_context_tolerates_empty_sections():
    qctx = QuantContext(composite=CompositeArtifact(
        ticker="X", sector="Unknown", composite_score=50.0,
        sector_rank=None, long_short_flag=None, as_of_date=AS_OF))
    block = format_quant_context(qctx)
    assert "Composite score: 50.0" in block   # no crash, minimal block


# ---------------------------------------------------------------------------
# report_generator
# ---------------------------------------------------------------------------
def _qctx(db: Database) -> QuantContext:
    _seed_ticker(db)
    return quant_context(db, "AAPL")


def test_build_memo_full_inputs(db: Database):
    memo = build_memo(
        quant_ctx=_qctx(db),
        overlay=validate_overlay(_valid_overlay()),
        earnings={"management_confidence": {"tone": "confident",
                                            "evidence": "raised guide"},
                  "guidance_quality": {"direction": "raised", "credibility": 80},
                  "competitive_position": {"trajectory": "improving"},
                  "overall_summary": "Strong quarter."},
        filing={"earnings_quality_score": 82, "balance_sheet_score": 75,
                "accounting_risk": 12, "green_flags": ["clean cash flow"],
                "red_flags": []},
        risk={"boilerplate_fraction": 0.8,
              "newly_introduced_risks": [{"summary": "new supplier risk",
                                          "severity": "medium"}],
              "removed_risks": [], "material_risks": []},
        insider={"signal": "BUY", "reasoning": "CFO bought",
                 "important_transactions": []},
    )
    assert "Research status: PASS" in memo
    assert "## 1. Quant Summary" in memo
    assert "## 2. Key Factor Drivers" in memo
    assert "## 7. Confirmation vs Contradiction" in memo
    assert "**CONFIRMS**" in memo
    assert "## 8. What Could Change The Thesis" in memo
    assert "WWDC" in memo
    assert "## 9. Final Research View" in memo
    # No recommendation/score-blend language survives the redesign.
    assert "Recommendation" not in memo
    assert "Blended" not in memo


def test_build_memo_without_overlay_or_analyzers(db: Database):
    memo = build_memo(quant_ctx=_qctx(db), overlay=None, earnings=None,
                      filing=None, risk=None, insider=None)
    assert "no overlay available" in memo
    assert "Quant Summary" in memo
    assert "stands unexamined" in memo
