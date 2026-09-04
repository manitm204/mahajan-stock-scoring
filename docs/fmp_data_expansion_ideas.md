# FMP Data Expansion Ideas — 20 Candidate Endpoints

*Written 2026-07-30. A survey of FMP stable-API endpoints we do NOT currently pull, chosen to
complement the parent_selection_v4 composite (8 parents / 23 subfactors) and our PIT-correctness
constraints. Source: https://site.financialmodelingprep.com/developer/docs/stable*

## What we already pull (for reference)

- **FMP** (`/stable`, via `data/providers.py`): `historical-price-eod/full`, `income-statement`,
  `balance-sheet-statement`, `cash-flow-statement`, `earning-call-transcript`, `grades-historical`,
  `grades` (individual rating actions, added 2026-08-03 for #11), `earnings` (added 2026-07-30
  for #1), `price-target-news`, `price-target-summary`, `analyst-estimates`,
  `institutional-ownership/symbol-positions-summary`, `dividends`, `insider-trading/search`
- **Polygon**: daily aggregates (primary prices), FINRA short interest
- **FRED**: VIX (`VIXCLS`)
- **SEC EDGAR**: filings index, insider fallback
- **yfinance**: fallback prices/fundamentals (earnings surprises migrated to FMP `stable/earnings` 2026-07-30)

---

## Fundamentals & valuation depth

### 1. Native earnings surprises — `stable/earnings` (per symbol) + `earnings-surprises-bulk` — ✅ DONE (2026-07-30)
`/stable/earnings` is live (82k rows back to 1985) and has replaced yfinance for
`grw_earnings_surprise`. Historical EPS actual vs. consensus estimate, straight from FMP. Our `grw_earnings_surprise`
subfactor (40% of the growth parent) currently rides on yfinance's `get_earnings_dates`, the
flakiest source in the stack. Migrating to FMP gives one vendor, deeper history, and a bulk
endpoint for the whole universe — plus it enables a proper SUE (standardized unexpected earnings)
and post-earnings-drift subfactor.

### 2. As-reported statements + filing dates — `income-statement-as-reported`, `balance-sheet-statement-as-reported`, `financial-reports-dates`
Numbers exactly as filed with the SEC, before later restatements, with precise filing dates. Given
how much of this project's history is look-ahead bug hunting, this is the structural fix: standard
statement endpoints can silently serve restated figures that didn't exist at the scoring date. Even
used only as an audit layer, diffing as-reported vs. standard statements would quantify how much
restatement contamination our fundamentals tables carry.

### 3. Statement growth series — `income-statement-growth`, `balance-sheet-statement-growth`, `cash-flow-statement-growth`, `financial-growth`
Pre-computed period-over-period growth for every line item (revenue, gross profit, FCF, R&D,
inventory, receivables, debt). The growth parent currently uses only EPS YoY and operating income
growth; this unlocks cheap candidates like inventory-growth-vs-sales-growth (classic
earnings-quality red flag) and asset growth (documented negative predictor) without writing the
diff logic. Slots directly into the subfactor-expansion library as a new candidate battery.

### 4. Owner earnings — `stable/owner-earnings`
FMP's Buffett-style owner earnings (net income + D&A ± working-capital changes − maintenance
capex) per share. Use as a valuation subfactor (owner-earnings yield) or a quality subfactor
(owner earnings vs. reported net income divergence flags accrual-heavy earnings). A genuinely
different earnings definition than anything in the current quality/value families (Piotroski,
Altman, EV/revenue).

### 5. PIT market cap & enterprise value — `historical-market-cap`, `enterprise-values`
Daily historical market capitalization and quarterly EV series (with debt/cash components broken
out). We currently derive market cap and EV ourselves, and both feed live subfactors
(`val_ev_revenue_inv`, Altman Z) — a vendor-provided historical series is both a cross-check
against our shares-outstanding × price derivation and a fix for share-count staleness between
filings.

## New company-level signals

### 6. Revenue product segmentation — `revenue-product-segmentation`
Annual/quarterly revenue broken down by product line. Two candidate signals: revenue concentration
(Herfindahl across segments — single-product companies are fragile, relevant to the loser-avoider
philosophy) and fastest-segment growth (is the growth engine accelerating or is a legacy segment
masking decline?). Nothing in the current stack sees below the top-line revenue number.

### 7. Revenue geographic segmentation — `revenue-geographic-segmentation`
Same idea by region. Gives a foreign-revenue-share exposure per stock, enabling a
dollar-sensitivity/geopolitical risk factor, and lets us check whether the composite is
accidentally loaded on international-revenue names in any regime. Also a natural risk axis for the
Layer 4/5 diversification caps alongside sector.

### 8. Historical employee count — `historical-employee-count`
Headcount over time from 10-K filings. Employee growth is a documented negative predictor
(empire-building), and revenue-per-employee change is a clean efficiency signal — both are
one-line factors once the data exists. An entirely new data dimension: no current table knows
anything about headcount.

### 9. Executive compensation — `executive-compensation`, `executive-compensation-benchmark`
Named-officer pay from DEF 14A proxies, plus an industry benchmark endpoint. The tradeable angle
is a governance subfactor: CEO pay relative to industry peers and relative to performance
(overpaid-CEO-of-underperformer is a classic red flag that fits the loser-veto architecture). Also
a natural structured input for the LLM overlay memos.

### 10. Share float — `shares-float`, `all-shares-float`
Actual free float and float percentage per symbol. The short parent (biggest parent weight at
0.1738 as of the 2026-08-07 B-cap ship) computes short-%-of-float from Polygon short interest divided by our own float
approximation — a real float denominator directly improves `si_short_pct_float` and its 252d
percentile. Float turnover (volume/float) is also a cleaner liquidity/crowding measure than raw
dollar volume.

## Analyst & smart-money intelligence

### 11. Analyst track records — `tipranks-analyst-summary`, `tipranks-pit-by-symbol`, `tipranks-pit-by-analyst`
FMP serves TipRanks analyst-level data: each analyst's historical accuracy and returns, with
point-in-time-by-symbol variants. The revisions parent currently treats every analyst equally;
weighting price-target changes and grade changes by the issuing analyst's track record is the most
direct upgrade available to an already-working factor (revisions carries 0.1515 parent weight
as of the 2026-08-07 B-cap weight ship). The
PIT endpoints matter — they tell us the analyst's reputation *at the time*, not today.

### 12. Grade consensus snapshots — `grades-consensus`, `grades-summary`
Current buy/hold/sell consensus counts and a summarized breakdown per symbol. Beyond feeding the
diffusion subfactor from a second angle, analyst coverage count itself is a signal:
neglected-stock effects, and coverage initiation/abandonment events. Also gives a same-day
snapshot to validate the month-complete-gated `analyst_grades` history against.

### 13. Smart-money-weighted 13F — `filings-extract-with-analytics-by-holder`, `holder-performance-summary`
Per-holder 13F extracts enriched with analytics, plus a performance summary of each institutional
holder. The institutional parent uses raw breadth (investors holding, new positions) where every
filer counts the same; this enables a "quality of holders" variant — changes in positions held by
historically high-performing funds only. Fundamentally different hypothesis than breadth, and the
13F partial-quarter lesson (report_date + 45d gate) already tells us exactly how to ingest it
PIT-safely.

### 14. Beneficial ownership / activist stakes — `acquisition-of-beneficial-ownership`
13D/13G filings — investors crossing the 5% ownership threshold. A new 13D from a known activist
is one of the strongest documented single-event alphas, and unlike quarterly 13Fs these arrive
within days of the trade. Would be our only fast-reacting institutional signal; everything else in
that family moves at quarterly cadence.

### 15. Congressional trading — `senate-trades`, `house-trades`
Stock transactions disclosed by U.S. senators and representatives under the STOCK Act. Treat it
exactly like the insider parent: cluster-buy flags and buy/sell pressure, but from politicians
rather than corporate officers — a completely independent "informed trader" population. PIT
caution mirrors insider ingest: gate on disclosure date (filings can lag trades by up to 45 days),
not transaction date.

### 16. Insider trade statistics — `insider-trade-statistics`
Pre-aggregated quarterly insider metrics per symbol: buy/sell counts, ratios, totals. The insider
parent computes rolling 180d aggregates from raw Form 4 rows; this endpoint provides an
independent aggregation to validate against (that family has already produced one look-ahead bug)
and offers a longer quarterly history for cheap backtest extension.

## Market structure & context

### 17. ETF ownership exposure — `etf-asset-exposure`
For a given stock, every ETF that holds it and at what weight. Aggregate passive-ownership share
is a crowding/flow-sensitivity measure that interacts naturally with the short-interest parent
(high SI + high passive ownership = squeeze mechanics), and index-inclusion changes are tradeable
events. We have no visibility into passive ownership today — 13Fs blend it invisibly into total
institutional holdings.

### 18. Industry-relative valuation — `historical-industry-pe`, `historical-sector-pe`, `industry-pe-snapshot`
Historical average P/E by industry and sector. Our value subfactors are cross-sectional against
the whole universe, which structurally tilts them toward cheap sectors; industry-relative versions
(stock P/E minus its industry's contemporaneous P/E) isolate within-industry cheapness — and our
own Brinson analysis showed 92% of the edge is within-sector selection, so within-industry
normalization is aligned with where the alpha actually lives.

### 19. Sector & industry momentum — `historical-sector-performance`, `industry-performance-snapshot`
Daily performance series by sector and industry. Uses: an industry-momentum overlay (a stock's
industry trend is predictive beyond its own momentum), and a richer regime signal for the
VIX-tilt/weighting machinery — sector-dispersion regimes computed from vendor data instead of
aggregating our own panel. Extends the SPY-signal regime work with a cross-sectional dimension
that study lacked.

### 20. Dilution & event feed — `latest-equity-offering` / `fundraising` (+ `8k-latest`, `mergers-acquisitions-search`)
Equity offering and S-1/424B filings give a shelf-offering/dilution flag — announced share
issuance is a well-documented negative signal and the natural completion of the buyback-yield
subfactor (we currently see repurchases but not offerings until they hit the share count quarters
later). The 8-K and M&A feeds are less quant-factor material but are exactly the structured event
context the LLM overlay memos would benefit from at analysis time.

---

## Priority shortlist

Highest expected value for effort, given the architecture:

1. ~~**#1 earnings surprises** — de-risk a live 0.409-weight subfactor off yfinance~~ ✅ DONE (2026-07-30)
2. ~~**#11 analyst track records** — direct upgrade to a working parent (revisions)~~ ✅ DONE (2026-08-04) —
   TipRanks endpoints are a separate paid add-on (402 on Ultimate), but `/stable/grades` (individual
   rating actions with firm identity, history to ~2012) delivered the same idea firm-level:
   `rev_rating_surprise_90d` (grade vs the issuing firm's own PIT baseline) passed the battery
   (+0.034 in-window, 97% coverage) AND the 2016-2022 true OOS test (~+0.01), and is now the top
   slot in the ratified revisions formula. Full trail: output/analyst_deep_dive/.
   Follow-up closed 2026-08-07: `rev_pt_upgrade_ratio_30d` (dead on clean data after the PT
   look-ahead fix, IC −0.001) removed; user ratified the clean-selector formula
   0.475 rating_surprise + 0.333 raise_and_bullish + 0.192 target_revision_raw, and the
   B-capped engine weights were re-derived (`scripts/derive_engine_weights_b.py`).
3. **#10 share float** — better denominator for the highest-weighted parent (short)
4. **#14 / #15 activist stakes + congressional trades** — genuinely new informed-trader signals
   that reuse the existing insider PIT machinery
5. **#18 industry-relative valuation** — aligns value with the within-sector-selection finding

Per pre-registration discipline: each of these should enter through the subfactor-expansion
candidate library and pass the same IC/IR battery before touching production.
