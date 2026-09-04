# Mahajan Hedge Fund — Architecture & Data Dictionary

> Reference document for the platform. Layer 1 (the Data Layer) is built and
> verified; later layers are planned and described here so each one can be
> built against a stable contract — the SQLite warehouse.

---

## 1. What this repo is

Mahajan Hedge Fund is a **long/short equity research platform**. It is built in
strict layers. Each layer has one job, reads from the layer(s) below it, and
never reaches up. The boundary between every layer is the central SQLite
warehouse (`cache/mahajan.db`): Layer 1 writes facts and features into it;
higher layers read those tables and write their own derived outputs back.

This separation is deliberate. It means any layer can be rebuilt, re-run, or
replaced without touching the others, and a data problem can always be traced
to exactly one module.

### Layer roadmap

| Layer | Directory | Responsibility | Status |
|-------|-----------|----------------|--------|
| **1. Data** | `data/` | Ingestion, storage, normalization, feature generation | **Built** |
| 2. Factors | `factors/` | Cross-sectional factor scoring from Layer 1 features | Planned |
| 3. Analysis | `analysis/` | Single-name / thematic deep-dives, AI analysis | Planned |
| 4. Portfolio | `portfolio/` | Ranking, long/short construction, position sizing | Planned |
| 5. Risk | `risk/` | Exposure, beta, concentration, scenario limits | Planned |
| 6. Execution | `execution/` | Order generation / broker integration | Planned |
| 7. Reporting | `reporting/` | Tear sheets, attribution, summaries | Planned |
| — Dashboard | `dashboard/` | UI surface over the layers | Planned |

**Layer 1 does not** score, rank, build portfolios, run AI analysis, execute
trades, or report. Those are explicitly out of scope for `data/` and belong to
the layers above. Layer 1's only product is a clean, historical, normalized
data warehouse plus derived features.

---

## 2. Layer 1 design principles

1. **Historical, never overwritten.** Append-only history tables (`INSERT OR
   IGNORE`) for things that happened once (filings, insider txns, 13F holdings,
   snapshots). Natural-key tables (`ON CONFLICT … DO UPDATE`) for the latest
   state of a fact (prices, fundamentals, features).
2. **Idempotent.** Running the pipeline twice produces zero new rows and zero
   duplicates. Every append-only table has a deterministic uniqueness key.
3. **Incremental.** Modules re-fetch only the recent window and merge; a daily
   run is cheap. Prices re-fetch a small tail so split/dividend adjustments
   propagate.
4. **Independently runnable.** Every module has a `python -m data.<module>`
   entry point and works standalone; `run_data.py` only wires them together.
5. **Provider-agnostic.** Modules ask a registry for a provider and call a
   normalized method. Adding a vendor = one class + one config line, no
   changes to consumers.
6. **Raw kept.** `fundamentals.raw_json` preserves every source field even if
   unused today, so future layers never need a re-fetch.
7. **Quality is non-fatal.** Validation emits warnings into a log table; it
   never crashes a build.

---

## 3. Repository layout

```
mahajan_hedge_fund/
├── config.yaml            # All tunable parameters (no secrets)
├── .env                   # API keys (gitignored): POLYGON / FMP / FRED + SEC contact
├── run_data.py            # Layer 1 orchestrator — 14 ordered stages + summary
├── data/                  # ===== LAYER 1 =====
│   ├── config.py          # Loads config.yaml + .env into a Config object
│   ├── db.py              # Database wrapper + authoritative SCHEMA (all tables)
│   ├── providers.py       # Provider abstraction + priority registry
│   ├── utils.py           # Logger, retry, safe_float/int, date helpers
│   ├── universe.py        # S&P 500 scrape + 26 benchmark/regime tickers
│   ├── market_data.py     # Daily OHLCV (incremental)
│   ├── fundamentals.py    # Income / balance / cashflow (annual + quarterly)
│   ├── features.py        # Derived price + fundamental features
│   ├── short_interest.py  # Short interest + squeeze flags
│   ├── estimates.py       # Analyst estimates + revisions
│   ├── earnings_calendar.py
│   ├── sec_data.py        # EDGAR filings + Form 4 insider parsing/flags
│   ├── institutional.py   # 13F holdings + QoQ signals
│   ├── transcripts.py     # Earnings call transcripts (FMP-gated, NLP-ready)
│   └── quality.py         # Non-fatal data quality checks
├── factors/ analysis/ portfolio/ risk/ execution/ reporting/ dashboard/
│                          # Placeholders for Layers 2–7 (.gitkeep only)
├── cache/                 # mahajan.db, EDGAR CIK map, S&P 500 CSV (gitignored)
├── output/                # run.log (gitignored)
└── docs/ARCHITECTURE.md   # This file
```

---

## 4. The provider framework

`data/providers.py` decouples *what data we want* from *where it comes from*.

```
        ┌─────────────────────────────────────────────┐
        │ ProviderRegistry (priority list per domain) │
        └─────────────────────────────────────────────┘
prices  ──► Polygon  ─(no key)─► yfinance        (fallback always works)
fundamentals ──► FMP ─(no key)─► yfinance
transcripts  ──► FMP  (gated: inactive without FMP_API_KEY)
macro        ──► FRED (gated)
filings      ──► SEC EDGAR (handled directly by sec_data.py)
```

The registry reads the priority list from `config.yaml → providers`, then for
each domain selects the **first provider whose required API key is present** in
`.env`, else falls back. Every provider normalizes to the same method
signature (`get_prices`, `get_statements`, `get_transcript`, `get_series`), so
consuming modules never branch on vendor. The chosen provider per domain is
logged at startup and shown in the run summary.

**Currently active** (no paid keys present): prices + fundamentals = `yfinance`,
filings = `sec`. FMP/FRED/Polygon paths exist and turn on automatically the
moment a key is added to `.env`.

---

## 5. Data dictionary

One database, 18 tables. All dates are ISO `YYYY-MM-DD` strings. Indexes exist
on every `ticker`, `date`, `filing_date`, and `report_date` column used for
lookups. Write semantics column: **UPSERT** = natural-key, latest snapshot;
**APPEND** = `INSERT OR IGNORE` history.

### Reference / universe

**`universe`** — current tradeable names (S&P 500). *UPSERT, key `ticker`.*
| column | type | notes |
|--------|------|-------|
| ticker | TEXT PK | |
| company_name, gics_sector, gics_sub_industry | TEXT | from Wikipedia scrape |
| date_added, source | TEXT | |
| active | INT | 1 = currently a constituent |
| first_seen, last_updated | TEXT | |

**`benchmarks`** — 26 ETFs / indices for beta, sector RS, and regime. *UPSERT, key `ticker`.*
Categories: `market` (SPY, QQQ, IWM, DIA), `sector` (11 × XL*), `factor`
(MTUM, QUAL, VLUE, SIZE), `regime` (VIX, TLT, HYG, LQD), `breadth` (RSP),
`alt` (GLD, USO).

### Market data

**`daily_prices`** — 4 years of OHLCV. *UPSERT, key `(ticker, date)`.*
| column | type | notes |
|--------|------|-------|
| ticker, date | TEXT PK | |
| open, high, low, close, adj_close | REAL | `adj_close` = split/div adjusted |
| volume | INT | |
| source | TEXT | polygon \| yfinance |

### Fundamentals (raw, wide)

**`fundamentals`** — quarterly + annual statements, normalized. *UPSERT, key
`(ticker, period_type, fiscal_date)`.* `period_type ∈ {annual, quarterly}`.
Income (revenue, cost_of_revenue, gross_profit, operating_income, ebit, ebitda,
net_income, eps_basic, eps_diluted, rnd_expense, interest_expense), cash flow
(operating_cash_flow, free_cash_flow, capex, dividends_paid, buybacks), balance
(total_assets, current_assets, cash, total_liabilities, current_liabilities,
debt, net_debt, working_capital, retained_earnings, shareholder_equity,
shares_outstanding). **`raw_json`** keeps the full source payload.

### Derived features

**`price_features`** — *UPSERT, key `(ticker, date)`.* return_20d/60d/252d,
volatility_20d (annualized ×√252), distance_from_52w_high/low, dollar_volume,
relative_volume (vs 20d avg), beta_vs_spy (rolling 252d), sector_relative_strength.

**`fundamental_features`** — *UPSERT, key `(ticker, period_type, fiscal_date)`.*
Growth (revenue/eps × yoy/qoq, revenue/eps_cagr_3y), returns (roe, roic),
margins (gross/operating/net), quality (cfo_to_net_income, asset_turnover),
leverage (debt_to_equity, current_ratio, interest_coverage, net_debt_to_ebitda),
valuation (fcf_yield, price_to_sales, ev_to_ebitda). Quarterly ratios use TTM sums.

### SEC filings & insiders

**`sec_filings`** — 10-K/10-Q/8-K/Form 4 metadata; body text cached for latest
10-K/10-Q only. *APPEND, unique `(accession_number, primary_doc)`.*

**`insider_transactions`** — parsed from Form 4 XML. *APPEND, unique
`dedup_key`.* Fields: insider_name/title, transaction_type/code, shares, price,
value, transaction_date, ownership_type, is_purchase. **`dedup_key`** is a
deterministic `|`-joined identity (accession + insider + date + code + shares +
price + ownership); a plain multi-column UNIQUE would treat NULL-price grants as
distinct and re-insert them every run.

**`insider_flags`** — derived signals: CEO/CFO/Director Purchase, Large Purchase
(≥ $1M), Cluster Buying (≥ 3 insiders). *APPEND, unique `(ticker,
transaction_date, flag_type, accession_number)`.*

### Institutional (13F)

**`institutional_holdings`** — tracked managers (Berkshire, Pershing Square,
Appaloosa, Baupost, Third Point). *APPEND, unique `(cik, cusip, report_date,
accession_number)`.* CUSIP→ticker resolved by normalized issuer-name match;
cusip always stored even when ticker is unresolved. Value units: thousands
pre-2023, dollars after.

**`institutional_signals`** — QoQ deltas per name. *UPSERT, key `(ticker,
report_date)`.* fund_count, position_increases/decreases, new/closed_positions,
net_share_change.

### Market signals

**`short_interest`** — *UPSERT, key `(ticker, date)`.* shares_short,
short_ratio, short_percent_of_float, float_shares, days_to_cover, change
fields, short_squeeze_risk_flag (>20% float short & DTC > 5).

**`analyst_estimates`** — point-in-time snapshots. *UPSERT, key `(ticker,
snapshot_date)`.* forward_eps, revenue_estimate, consensus/low/high_price_target,
recommendation_mean, number_of_analysts, current_price.

**`analyst_estimate_features`** — *UPSERT, key `(ticker, snapshot_date)`.*
forward_eps & price_target revisions over 30/60/90d, price_target_upside_pct,
estimate_breadth_change, analyst_count_change.

**`earnings_calendar`** — *UPSERT, key `(ticker, earnings_date)`.* earnings_time,
fiscal_quarter, eps_estimate, eps_actual.

**`transcripts`** — FMP-gated, **only for explicitly requested candidate
tickers, never the whole universe**. *UPSERT, key `(ticker, fiscal_year,
fiscal_quarter)`.* transcript_text + reserved NULL NLP columns
(management_tone_score, guidance_sentiment_score, risk/margin/ai/capex/
restructuring mention counts) for a future layer to populate.

### Operational

**`data_quality_log`** — run_id, check_name, ticker, severity, message,
created_at. One row per warning per run.

**`meta`** — key/value store (e.g. `universe_last_refresh`) for incremental
bookkeeping.

---

## 6. The pipeline (`run_data.py`)

14 ordered stages, run in dependency order, logging to `output/run.log` and
printing an execution summary:

1. Universe & benchmarks → 2. Provider init → 3. Prices → 4. Fundamentals →
5–6. Price + fundamental features → 7. Short interest → 8. Estimates →
9. Earnings calendar → 10. Transcripts (candidates only) → 11–12. SEC filings +
insiders → 13. 13F → 14. Data quality validation.

**CLI flags:** `--tickers`, `--full`, `--no-benchmarks`, and a `--no-*` switch
per stage (`--no-prices`, `--no-fundamentals`, `--no-features`,
`--no-estimates`, `--no-earnings-calendar`,
`--no-transcripts`, `--no-filings`, `--no-13f`), plus `--transcripts-only`,
`--forms`, and `--days` to tune SEC scope. Short interest is opt-in via
`--short-interest` (Polygon free tier is paced to 5 req/min, ~2.4h for the
full universe); when skipped, the run warns if the stored data is >15 days
old.

```bash
python run_data.py                                 # full universe
python run_data.py --tickers AAPL MSFT NVDA        # subset
python run_data.py --no-filings --no-13f           # skip slow EDGAR stages
python run_data.py --forms 10-K 4 --days 30        # tune SEC scope
python -m data.market_data --tickers AAPL          # any module standalone
```

---

## 7. Data quality

`data/quality.py` runs 7 non-fatal checks after every build and records
warnings in `data_quality_log`: missing/invalid OHLC, negative share counts,
duplicate filings across tickers, duplicate insider transactions, malformed
dates, priced names lacking fundamentals, and stale prices (>10d behind market).
A broken check logs an error and returns 0 — it never crashes the run.

**Verification status:** a 3-ticker end-to-end run (AAPL/MSFT/NVDA) builds clean
with **0 quality warnings**, and a second consecutive run adds **0 new rows**
across all tables — confirming full idempotency.

---

## 8. Notes for Layer 2 (Factors)

When building `factors/`, treat Layer 1 as a read-only contract:

- **Read features, not raw.** `price_features` and `fundamental_features` are
  the intended inputs; raw tables are for traceability and recomputation.
- **Point-in-time correctness.** Use `fiscal_date` / `snapshot_date` /
  `report_date` to avoid look-ahead bias. Snapshots are stored historically for
  exactly this reason.
- **Cross-sectional scoring belongs to Layer 2,** not Layer 1 — Layer 1
  deliberately stores only per-name absolute features (the one exception,
  `sector_relative_strength`, is a raw input, not a rank).
- **Write your outputs to new tables** in the same DB (e.g. `factor_scores`),
  following the same UPSERT/APPEND discipline and adding them to a Layer 2
  schema module — do not mutate Layer 1 tables.
- **Macro/regime data** (FRED) and benchmark series are already available for
  regime-conditional factor models.
