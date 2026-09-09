"""Centralized SQLite warehouse for Layer 1.

One database, normalized tables, indexed for historical research. Every
module reads/writes through the :class:`Database` helper so the schema stays
authoritative in a single place. Writes are historical and idempotent:
upserts update the latest snapshot of a row identified by its natural key,
append-only tables use ``INSERT OR IGNORE`` so duplicate filings/transactions
never create duplicate rows.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import pandas as pd

from .config import load_config
from .utils import get_logger

# ---------------------------------------------------------------------------
# Schema. Kept as one executescript string for compactness. All DDL is
# IF NOT EXISTS so opening the DB is always safe and idempotent.
# ---------------------------------------------------------------------------
SCHEMA = """
-- Universe & benchmarks -----------------------------------------------------
CREATE TABLE IF NOT EXISTS universe (
    ticker            TEXT PRIMARY KEY,
    company_name      TEXT,
    gics_sector       TEXT,
    gics_sub_industry TEXT,
    date_added        TEXT,
    source            TEXT,
    active            INTEGER DEFAULT 1,
    first_seen        TEXT,
    last_updated      TEXT
);

CREATE TABLE IF NOT EXISTS benchmarks (
    ticker       TEXT PRIMARY KEY,
    name         TEXT,
    category     TEXT,
    last_updated TEXT
);

-- Point-in-time index membership (from FMP constituent-change history).
-- One row per membership spell; end_date NULL = still a member. A ticker can
-- have several spells (removed then re-added). Used by research/backtests so
-- historical scoring only sees names that were actually in the index then.
CREATE TABLE IF NOT EXISTS universe_history (
    ticker       TEXT NOT NULL,
    start_date   TEXT NOT NULL,   -- date added (inclusive)
    end_date     TEXT,            -- date removed (exclusive); NULL = current
    company_name TEXT,
    source       TEXT,
    last_updated TEXT,
    PRIMARY KEY (ticker, start_date)
);
CREATE INDEX IF NOT EXISTS idx_unihist_window ON universe_history(start_date, end_date);

-- Market data ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS daily_prices (
    ticker    TEXT NOT NULL,
    date      TEXT NOT NULL,
    open      REAL,
    high      REAL,
    low       REAL,
    close     REAL,
    adj_close REAL,
    volume    INTEGER,
    source    TEXT,
    PRIMARY KEY (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_prices_ticker ON daily_prices(ticker);
CREATE INDEX IF NOT EXISTS idx_prices_date   ON daily_prices(date);

-- Fundamentals (raw, wide; raw_json keeps every field for the future) -------
CREATE TABLE IF NOT EXISTS fundamentals (
    ticker              TEXT NOT NULL,
    period_type         TEXT NOT NULL,        -- annual | quarterly
    fiscal_date         TEXT NOT NULL,
    currency            TEXT,
    revenue             REAL,
    cost_of_revenue     REAL,
    gross_profit        REAL,
    operating_income    REAL,
    ebit                REAL,
    ebitda              REAL,
    net_income          REAL,
    eps_basic           REAL,
    eps_diluted         REAL,
    rnd_expense         REAL,
    interest_expense    REAL,
    operating_cash_flow REAL,
    free_cash_flow      REAL,
    capex               REAL,
    dividends_paid      REAL,
    buybacks            REAL,
    total_assets        REAL,
    current_assets      REAL,
    cash                REAL,
    total_liabilities   REAL,
    current_liabilities REAL,
    debt                REAL,
    net_debt            REAL,
    working_capital     REAL,
    retained_earnings   REAL,
    shareholder_equity  REAL,
    shares_outstanding  REAL,
    raw_json            TEXT,
    source              TEXT,
    fetched_at          TEXT,
    PRIMARY KEY (ticker, period_type, fiscal_date)
);
CREATE INDEX IF NOT EXISTS idx_fund_ticker ON fundamentals(ticker);
CREATE INDEX IF NOT EXISTS idx_fund_date   ON fundamentals(fiscal_date);

-- Derived features ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS price_features (
    ticker                   TEXT NOT NULL,
    date                     TEXT NOT NULL,
    return_20d               REAL,
    return_60d               REAL,
    return_252d              REAL,
    volatility_20d           REAL,
    distance_from_52w_high   REAL,
    distance_from_52w_low    REAL,
    dollar_volume            REAL,
    relative_volume          REAL,
    beta_vs_spy              REAL,
    sector_relative_strength REAL,
    computed_at              TEXT,
    PRIMARY KEY (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_pf_ticker ON price_features(ticker);
CREATE INDEX IF NOT EXISTS idx_pf_date   ON price_features(date);

CREATE TABLE IF NOT EXISTS fundamental_features (
    ticker             TEXT NOT NULL,
    period_type        TEXT NOT NULL,
    fiscal_date        TEXT NOT NULL,
    revenue_growth_yoy REAL,
    revenue_growth_qoq REAL,
    eps_growth_yoy     REAL,
    eps_growth_qoq     REAL,
    revenue_cagr_3y    REAL,
    eps_cagr_3y        REAL,
    roe                REAL,
    roic               REAL,
    gross_margin       REAL,
    operating_margin   REAL,
    net_margin         REAL,
    cfo_to_net_income  REAL,
    asset_turnover     REAL,
    debt_to_equity     REAL,
    current_ratio      REAL,
    interest_coverage  REAL,
    net_debt_to_ebitda REAL,
    fcf_yield          REAL,
    price_to_sales     REAL,
    ev_to_ebitda       REAL,
    computed_at        TEXT,
    PRIMARY KEY (ticker, period_type, fiscal_date)
);
CREATE INDEX IF NOT EXISTS idx_ff_ticker ON fundamental_features(ticker);

-- SEC filings & insider activity -------------------------------------------
CREATE TABLE IF NOT EXISTS sec_filings (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker           TEXT,
    cik              TEXT,
    form_type        TEXT,
    filing_date      TEXT,
    report_date      TEXT,
    accession_number TEXT,
    primary_doc      TEXT,
    primary_doc_url  TEXT,
    filing_text      TEXT,
    fetched_at       TEXT,
    UNIQUE (accession_number, primary_doc)
);
CREATE INDEX IF NOT EXISTS idx_filings_ticker ON sec_filings(ticker);
CREATE INDEX IF NOT EXISTS idx_filings_date   ON sec_filings(filing_date);
CREATE INDEX IF NOT EXISTS idx_filings_form   ON sec_filings(form_type);

CREATE TABLE IF NOT EXISTS insider_transactions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker           TEXT,
    cik              TEXT,
    accession_number TEXT,
    insider_name     TEXT,
    insider_title    TEXT,
    transaction_type TEXT,
    transaction_code TEXT,
    shares           REAL,
    price            REAL,
    value            REAL,
    transaction_date TEXT,
    ownership_type   TEXT,
    is_purchase      INTEGER,
    fetched_at       TEXT,
    -- Deterministic key: a multi-column UNIQUE would let SQLite treat NULL
    -- prices (grants) as distinct and re-insert them on every run.
    dedup_key        TEXT UNIQUE
);
CREATE INDEX IF NOT EXISTS idx_insider_ticker ON insider_transactions(ticker);
CREATE INDEX IF NOT EXISTS idx_insider_date   ON insider_transactions(transaction_date);

CREATE TABLE IF NOT EXISTS insider_flags (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker           TEXT,
    transaction_date TEXT,
    flag_type        TEXT,
    detail           TEXT,
    accession_number TEXT,
    created_at       TEXT,
    UNIQUE (ticker, transaction_date, flag_type, accession_number)
);
CREATE INDEX IF NOT EXISTS idx_iflag_ticker ON insider_flags(ticker);

-- Beneficial ownership (13D/13G activist stakes) ----------------------------
CREATE TABLE IF NOT EXISTS beneficial_ownership (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker                   TEXT,
    cik                      TEXT,
    filing_date              TEXT,
    accepted_date            TEXT,
    cusip                    TEXT,
    reporting_person         TEXT,
    citizenship              TEXT,
    sole_voting_power        REAL,
    shared_voting_power      REAL,
    sole_dispositive_power   REAL,
    shared_dispositive_power REAL,
    amount_beneficially_owned REAL,
    percent_of_class         REAL,
    reporting_person_type    TEXT,
    filing_type              TEXT,   -- '13D'/'13G' parsed from the filing URL
                                      -- (only detectable post ~2019 XBRL-viewer
                                      -- filings; NULL for older filings)
    url                      TEXT,
    source                   TEXT,
    fetched_at               TEXT,
    dedup_key                TEXT UNIQUE
);
CREATE INDEX IF NOT EXISTS idx_benown_ticker ON beneficial_ownership(ticker);
CREATE INDEX IF NOT EXISTS idx_benown_date   ON beneficial_ownership(filing_date);

-- Congressional trading (STOCK Act disclosures) -----------------------------
CREATE TABLE IF NOT EXISTS congressional_trades (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker                TEXT,
    chamber               TEXT,           -- 'senate' or 'house'
    member_id             TEXT,
    first_name            TEXT,
    last_name             TEXT,
    office                TEXT,
    district              TEXT,
    owner                 TEXT,
    asset_description     TEXT,
    asset_type            TEXT,
    transaction_type      TEXT,
    transaction_date      TEXT,
    disclosure_date       TEXT,           -- PIT gate: public availability, not transaction_date
    amount_range          TEXT,
    capital_gains_over_200 INTEGER,
    link                  TEXT,
    source                TEXT,
    fetched_at            TEXT,
    dedup_key             TEXT UNIQUE
);
CREATE INDEX IF NOT EXISTS idx_congress_ticker ON congressional_trades(ticker);
CREATE INDEX IF NOT EXISTS idx_congress_date   ON congressional_trades(disclosure_date);

-- Institutional holdings (13F) ---------------------------------------------
CREATE TABLE IF NOT EXISTS institutional_holdings (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    fund_name        TEXT,
    cik              TEXT,
    ticker           TEXT,
    cusip            TEXT,
    shares_held      REAL,
    market_value     REAL,
    report_date      TEXT,
    filing_date      TEXT,
    accession_number TEXT,
    fetched_at       TEXT,
    UNIQUE (cik, cusip, report_date, accession_number)
);
CREATE INDEX IF NOT EXISTS idx_13f_ticker ON institutional_holdings(ticker);
CREATE INDEX IF NOT EXISTS idx_13f_report ON institutional_holdings(report_date);
CREATE INDEX IF NOT EXISTS idx_13f_filing ON institutional_holdings(filing_date);
CREATE INDEX IF NOT EXISTS idx_13f_cusip  ON institutional_holdings(cusip);

CREATE TABLE IF NOT EXISTS institutional_signals (
    ticker             TEXT NOT NULL,
    report_date        TEXT NOT NULL,
    fund_count         INTEGER,
    position_increases INTEGER,
    position_decreases INTEGER,
    new_positions      INTEGER,
    closed_positions   INTEGER,
    net_share_change   REAL,
    computed_at        TEXT,
    PRIMARY KEY (ticker, report_date)
);

-- Whole-market 13F ownership summary per symbol/quarter (FMP Ultimate). Deep,
-- universe-wide breadth (thousands of filers) vs the handful of curated EDGAR
-- funds; source for institutional_signals.
CREATE TABLE IF NOT EXISTS institutional_ownership_summary (
    ticker                   TEXT NOT NULL,
    report_date              TEXT NOT NULL,   -- fiscal quarter end
    cik                      TEXT,
    investors_holding        INTEGER,
    investors_holding_change INTEGER,
    number_of_13f_shares     REAL,
    shares_change            REAL,            -- numberOf13FsharesChange
    total_invested           REAL,
    ownership_percent        REAL,
    ownership_percent_change REAL,
    new_positions            INTEGER,
    increased_positions      INTEGER,
    reduced_positions        INTEGER,
    closed_positions         INTEGER,
    put_call_ratio           REAL,
    source                   TEXT,
    fetched_at               TEXT,
    PRIMARY KEY (ticker, report_date)
);
CREATE INDEX IF NOT EXISTS idx_inst_own_report ON institutional_ownership_summary(report_date);

-- Short interest ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS short_interest (
    ticker                   TEXT NOT NULL,
    date                     TEXT NOT NULL,
    shares_short             REAL,
    short_ratio              REAL,
    short_percent_of_float   REAL,
    shares_outstanding       REAL,
    float_shares             REAL,
    short_interest_change    REAL,
    short_percent_float_change REAL,
    days_to_cover            REAL,
    short_squeeze_risk_flag  INTEGER,
    source                   TEXT,
    fetched_at               TEXT,
    PRIMARY KEY (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_short_ticker ON short_interest(ticker);
CREATE INDEX IF NOT EXISTS idx_short_date   ON short_interest(date);

-- Analyst estimates ---------------------------------------------------------
CREATE TABLE IF NOT EXISTS analyst_estimates (
    ticker                 TEXT NOT NULL,
    snapshot_date          TEXT NOT NULL,
    forward_eps            REAL,
    revenue_estimate       REAL,
    consensus_price_target REAL,
    low_price_target       REAL,
    high_price_target      REAL,
    recommendation_mean    REAL,
    number_of_analysts     INTEGER,
    current_price          REAL,
    source                 TEXT,
    fetched_at             TEXT,
    PRIMARY KEY (ticker, snapshot_date)
);
CREATE INDEX IF NOT EXISTS idx_est_ticker ON analyst_estimates(ticker);
CREATE INDEX IF NOT EXISTS idx_est_date   ON analyst_estimates(snapshot_date);

CREATE TABLE IF NOT EXISTS analyst_estimate_features (
    ticker                    TEXT NOT NULL,
    snapshot_date             TEXT NOT NULL,
    forward_eps_revision_30d  REAL,
    forward_eps_revision_60d  REAL,
    forward_eps_revision_90d  REAL,
    price_target_revision_30d REAL,
    price_target_revision_60d REAL,
    price_target_revision_90d REAL,
    price_target_upside_pct   REAL,
    estimate_breadth_change   REAL,
    analyst_count_change      INTEGER,
    computed_at               TEXT,
    PRIMARY KEY (ticker, snapshot_date)
);

-- Analyst grade history + revision features (FMP) ---------------------------
-- grades-historical is genuinely dated point-in-time history, so revision
-- windows are computed from one fetch rather than accumulated daily snapshots.
CREATE TABLE IF NOT EXISTS analyst_grades (
    ticker      TEXT NOT NULL,
    date        TEXT NOT NULL,
    strong_buy  INTEGER,
    buy         INTEGER,
    hold        INTEGER,
    sell        INTEGER,
    strong_sell INTEGER,
    total       INTEGER,
    source      TEXT,
    fetched_at  TEXT,
    PRIMARY KEY (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_grades_ticker ON analyst_grades(ticker);
CREATE INDEX IF NOT EXISTS idx_grades_date   ON analyst_grades(date);

CREATE TABLE IF NOT EXISTS analyst_revision_features (
    ticker                  TEXT NOT NULL,
    snapshot_date           TEXT NOT NULL,
    rating_net_score        REAL,   -- (2*sb + b - s - 2*ss) / total, range [-2, 2]
    rating_change_30d       REAL,   -- net-score change vs ~30d ago (upgrade momentum)
    rating_change_60d       REAL,
    rating_change_90d       REAL,
    pt_consensus            REAL,   -- recent consensus price target
    pt_momentum             REAL,   -- lastMonthAvg / lastQuarterAvg - 1
    total_analysts          INTEGER,
    source                  TEXT,
    computed_at             TEXT,
    PRIMARY KEY (ticker, snapshot_date)
);
CREATE INDEX IF NOT EXISTS idx_revfeat_date ON analyst_revision_features(snapshot_date);

-- Dated per-analyst price-target events (FMP price-target-news) -------------
-- A genuine time series of upgrade/downgrade/target-change events, used to
-- build trailing-30d revision aggregates that the snapshot-only Yahoo path
-- cannot produce.
CREATE TABLE IF NOT EXISTS analyst_price_target_events (
    ticker            TEXT NOT NULL,
    published_date    TEXT NOT NULL,
    analyst_company   TEXT,
    analyst_name      TEXT,
    price_target      REAL,
    adj_price_target  REAL,
    price_when_posted REAL,
    news_title        TEXT,
    news_url          TEXT,
    source            TEXT,
    fetched_at        TEXT,
    UNIQUE (ticker, published_date, analyst_company, price_target)
);
CREATE INDEX IF NOT EXISTS idx_pt_evt_ticker ON analyst_price_target_events(ticker);
CREATE INDEX IF NOT EXISTS idx_pt_evt_date   ON analyst_price_target_events(published_date);

-- Dated individual rating actions (FMP /stable/grades) -----------------------
-- One row per analyst-firm rating action (upgrade / downgrade / initiation /
-- maintain) with the firm identity and the previous -> new grade transition.
-- Unlike analyst_grades (monthly aggregate counts, no identity), this enables
-- firm-selectivity signals: a Buy from a firm that rarely says Buy.
CREATE TABLE IF NOT EXISTS analyst_grade_events (
    ticker           TEXT NOT NULL,
    date             TEXT NOT NULL,
    grading_company  TEXT,
    previous_grade   TEXT,
    new_grade        TEXT,
    action           TEXT,
    source           TEXT,
    fetched_at       TEXT,
    UNIQUE (ticker, date, grading_company, new_grade, action)
);
CREATE INDEX IF NOT EXISTS idx_grade_evt_ticker ON analyst_grade_events(ticker);
CREATE INDEX IF NOT EXISTS idx_grade_evt_date   ON analyst_grade_events(date);

-- Earnings calendar ---------------------------------------------------------
CREATE TABLE IF NOT EXISTS earnings_calendar (
    ticker         TEXT NOT NULL,
    earnings_date  TEXT NOT NULL,
    earnings_time  TEXT,
    fiscal_quarter TEXT,
    eps_estimate   REAL,
    eps_actual     REAL,
    revenue_estimate REAL,
    revenue_actual   REAL,
    source         TEXT,
    fetched_at     TEXT,
    PRIMARY KEY (ticker, earnings_date)
);
CREATE INDEX IF NOT EXISTS idx_earn_ticker ON earnings_calendar(ticker);
CREATE INDEX IF NOT EXISTS idx_earn_date   ON earnings_calendar(earnings_date);

-- Earnings transcripts (NLP-ready) -----------------------------------------
CREATE TABLE IF NOT EXISTS transcripts (
    ticker                    TEXT NOT NULL,
    fiscal_year               INTEGER NOT NULL,
    fiscal_quarter            INTEGER NOT NULL,
    call_date                 TEXT,
    transcript_text           TEXT,
    source                    TEXT,
    fetched_at                TEXT,
    management_tone_score     REAL,
    guidance_sentiment_score  REAL,
    risk_mentions_count       INTEGER,
    margin_pressure_mentions  INTEGER,
    ai_mentions               INTEGER,
    capex_mentions            INTEGER,
    restructuring_mentions    INTEGER,
    PRIMARY KEY (ticker, fiscal_year, fiscal_quarter)
);
CREATE INDEX IF NOT EXISTS idx_trans_ticker ON transcripts(ticker);

-- Historical dividends (per-share cash) — subfactor expansion (Layer 2 research)
CREATE TABLE IF NOT EXISTS historical_dividends (
    ticker           TEXT NOT NULL,
    ex_date          TEXT NOT NULL,      -- ex-dividend date
    payment_date     TEXT,
    record_date      TEXT,
    declaration_date TEXT,
    dividend         REAL,               -- per-share cash dividend
    adj_dividend     REAL,               -- split-adjusted
    source           TEXT,
    fetched_at       TEXT,
    PRIMARY KEY (ticker, ex_date)
);
CREATE INDEX IF NOT EXISTS idx_hdiv_ticker    ON historical_dividends(ticker);
CREATE INDEX IF NOT EXISTS idx_hdiv_ex_date   ON historical_dividends(ex_date);

-- Data quality log & key/value meta ----------------------------------------
CREATE TABLE IF NOT EXISTS data_quality_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT,
    check_name TEXT,
    ticker     TEXT,
    severity   TEXT,
    message    TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_dq_run ON data_quality_log(run_id);

CREATE TABLE IF NOT EXISTS meta (
    key        TEXT PRIMARY KEY,
    value      TEXT,
    updated_at TEXT
);
"""


class Database:
    """Thin wrapper over a sqlite3 connection with upsert/query helpers."""

    def __init__(self, path: Path | str | None = None, wal: bool = True) -> None:
        cfg = load_config()
        self.path = Path(path) if path else cfg.db_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.log = get_logger("db")
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON;")
        if wal:
            self._conn.execute("PRAGMA journal_mode = WAL;")
        self.init_schema()

    # -- lifecycle ----------------------------------------------------------
    def init_schema(self) -> None:
        self._conn.executescript(SCHEMA)
        self._apply_migrations()
        self._conn.commit()

    def _apply_migrations(self) -> None:
        """Idempotent ALTER TABLE migrations for additive column changes.

        SQLite lacks `ADD COLUMN IF NOT EXISTS`, so we probe `PRAGMA
        table_info` and skip columns already present. Schema-extending
        changes go here; structural changes still belong in SCHEMA.
        """
        migrations: list[tuple[str, str, str]] = [
            # FMP price-target-news → trailing-30d revision signals.
            ("analyst_revision_features", "pt_event_count_30d", "INTEGER"),
            ("analyst_revision_features", "pt_upgrade_ratio_30d", "REAL"),
            ("analyst_revision_features", "pt_target_upside_30d", "REAL"),
            # FMP /stable/earnings → revenue surprise fields.
            ("earnings_calendar", "revenue_estimate", "REAL"),
            ("earnings_calendar", "revenue_actual", "REAL"),
        ]
        for table, column, decl in migrations:
            existing = {r["name"] for r in self._conn.execute(
                f"PRAGMA table_info({table})")}
            if column not in existing:
                self._conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    def close(self) -> None:
        self._conn.commit()
        self._conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # -- write helpers ------------------------------------------------------
    def upsert(self, table: str, rows: Sequence[dict[str, Any]],
               conflict: Sequence[str], update: Sequence[str] | None = None) -> int:
        """Insert rows; on conflict over ``conflict`` cols, update ``update`` cols.

        Returns the number of rows submitted. Used for tables keyed by a
        natural key (prices, fundamentals, features, snapshots) so that the
        newest fetch refreshes the stored row without creating duplicates.
        """
        if not rows:
            return 0
        cols = _union_keys(rows)
        if update is None:
            update = [c for c in cols if c not in conflict]
        placeholders = ", ".join(["?"] * len(cols))
        col_sql = ", ".join(cols)
        conflict_sql = ", ".join(conflict)
        if update:
            set_sql = ", ".join(f"{c}=excluded.{c}" for c in update)
            action = f"DO UPDATE SET {set_sql}"
        else:
            action = "DO NOTHING"
        sql = (
            f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders}) "
            f"ON CONFLICT({conflict_sql}) {action}"
        )
        params = [tuple(r.get(c) for c in cols) for r in rows]
        with self.transaction() as conn:
            conn.executemany(sql, params)
        return len(rows)

    def insert_ignore(self, table: str, rows: Sequence[dict[str, Any]]) -> int:
        """Append rows, ignoring those that violate a UNIQUE constraint.

        Returns the count of rows newly inserted (via total_changes delta).
        Used for append-only history: filings, insider txns, 13F holdings.
        """
        if not rows:
            return 0
        cols = _union_keys(rows)
        placeholders = ", ".join(["?"] * len(cols))
        col_sql = ", ".join(cols)
        sql = f"INSERT OR IGNORE INTO {table} ({col_sql}) VALUES ({placeholders})"
        params = [tuple(r.get(c) for c in cols) for r in rows]
        before = self._conn.total_changes
        with self.transaction() as conn:
            conn.executemany(sql, params)
        return self._conn.total_changes - before

    # -- read helpers -------------------------------------------------------
    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return self._conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        return self._conn.execute(sql, params).fetchone()

    def scalar(self, sql: str, params: Sequence[Any] = ()) -> Any:
        row = self._conn.execute(sql, params).fetchone()
        return row[0] if row else None

    def query_df(self, sql: str, params: Sequence[Any] = ()) -> pd.DataFrame:
        return pd.read_sql_query(sql, self._conn, params=list(params))

    def count(self, table: str, where: str = "", params: Sequence[Any] = ()) -> int:
        sql = f"SELECT COUNT(*) FROM {table}"
        if where:
            sql += f" WHERE {where}"
        return int(self.scalar(sql, params) or 0)

    def max_value(self, table: str, column: str, where: str = "",
                  params: Sequence[Any] = ()) -> Any:
        sql = f"SELECT MAX({column}) FROM {table}"
        if where:
            sql += f" WHERE {where}"
        return self.scalar(sql, params)

    # -- meta / data quality ------------------------------------------------
    def set_meta(self, key: str, value: str) -> None:
        self.upsert(
            "meta",
            [{"key": key, "value": value, "updated_at": _now()}],
            conflict=["key"],
        )

    def get_meta(self, key: str) -> str | None:
        row = self.query_one("SELECT value FROM meta WHERE key = ?", (key,))
        return row["value"] if row else None

    def log_quality(self, run_id: str, check_name: str, message: str,
                    ticker: str | None = None, severity: str = "warning") -> None:
        self.insert_ignore_unchecked(
            "data_quality_log",
            {
                "run_id": run_id,
                "check_name": check_name,
                "ticker": ticker,
                "severity": severity,
                "message": message,
                "created_at": _now(),
            },
        )

    def insert_ignore_unchecked(self, table: str, row: dict[str, Any]) -> None:
        cols = list(row.keys())
        placeholders = ", ".join(["?"] * len(cols))
        sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
        with self.transaction() as conn:
            conn.execute(sql, tuple(row.values()))

    # -- universe helpers (shared by many modules) --------------------------
    def universe_tickers(self, active_only: bool = True) -> list[str]:
        sql = "SELECT ticker FROM universe"
        if active_only:
            sql += " WHERE active = 1"
        sql += " ORDER BY ticker"
        return [r["ticker"] for r in self.query(sql)]

    def members_as_of(self, as_of: str) -> list[str]:
        """Point-in-time index members on ``as_of`` (from universe_history).

        Empty list means the history table has not been built yet — callers
        should fall back to (or error on) the static universe explicitly
        rather than silently scoring a survivorship-biased set.
        """
        return [r["ticker"] for r in self.query(
            "SELECT DISTINCT ticker FROM universe_history "
            "WHERE start_date <= ? AND (end_date IS NULL OR end_date > ?) "
            "ORDER BY ticker", (as_of, as_of))]

    def benchmark_tickers(self) -> list[str]:
        return [r["ticker"] for r in self.query(
            "SELECT ticker FROM benchmarks ORDER BY ticker")]

    def all_tradeable_tickers(self) -> list[str]:
        return sorted(set(self.universe_tickers()) | set(self.benchmark_tickers()))


def _union_keys(rows: Sequence[dict[str, Any]]) -> list[str]:
    """Ordered union of keys across rows so heterogeneous dicts (e.g. computed
    feature rows where some keys are set only conditionally) never silently
    drop columns based on the first row's shape."""
    cols: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                cols.append(key)
    return cols


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_db(path: Path | str | None = None) -> Database:
    """Convenience factory honoring config WAL setting."""
    cfg = load_config()
    return Database(path=path, wal=bool(cfg.get("database", "wal_mode", default=True)))


if __name__ == "__main__":
    with get_db() as db:
        tables = db.query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        print(f"Database ready at {db.path}")
        print(f"Tables ({len(tables)}):")
        for t in tables:
            print("  -", t["name"])
