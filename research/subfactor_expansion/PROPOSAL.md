# Subfactor Expansion — Audit + Proposal

Goal: bring each parent bucket to **8–10 well-designed subfactors** for research,
so parent construction and weight studies have real breadth. This is a
**research library** (not a production swap): the current production `lean` set
stays live; expanded candidates are validated in isolation.

Status: PROPOSAL — no code changed. Awaiting scope confirmation.

---

## 1. Audit of the current library

Universe: **505 S&P 500 tickers**. Layer-1 spans (see `data/db.py`):

| Table | Span | Notes |
|---|---|---|
| `daily_prices` | 2022-06-21 → 2026-07-01 | 4 years — caps long-horizon backtests |
| `fundamentals` (annual) | 2004 → 2026 | 99.8% coverage on income/BS/CFS **except** `dividends_paid` and `working_capital` and `shares_outstanding` (all ~0.6%) — these are structurally under-populated by the current FMP pull |
| `fundamental_features` (quarterly) | 2004 → 2026 | ROIC, ROE, margins, CFO/NI, D/E, current ratio, interest coverage, net-debt/EBITDA all **≥ 94%** — many are computed but not yet emitted as subfactors |
| `short_interest` | 2017-12 → 2026-07 | bi-monthly FINRA; % of float 99%, days-to-cover 100%, Δ 98% |
| `insider_transactions` | 1988 → 2026 (1.46M rows) | all Form-4 codes; 180d window: P (buys) touches 137 tickers, S (sells) 435 |
| `institutional_signals` | 2019-03 → 2026-03 | 5 aggregates (fund count, position ↑/↓, net share Δ, new/closed positions) — used |
| `institutional_ownership_summary` | 2019-03 → 2026-03 | richer whole-market fields (**ownership_percent, ownership_percent_change, put_call_ratio, investors_holding_change, total_invested**) — **currently unused** |
| `analyst_grades` | 2018-12 → 2026-07 | monthly strong_buy/buy/hold/sell/strong_sell counts — **currently unused** as a subfactor (only aggregated to net rating) |
| `analyst_revision_features` | 2018-12 → 2026-07 | rating change 30/60/90d, PT momentum, PT breadth |
| `analyst_estimate_features` | 2026-06-21 → 2026-07-03 | numeric revisions — **only 2 weeks live**, gated by `_MIN_COVERAGE = 0.30` in `factors/revisions.py` |
| `earnings_calendar` | 2008 → 2026 | eps_estimate 99.9%, eps_actual 95.4% — supports SUE / earnings-surprise signals **not yet used** |

Production subfactor counts (config `lean` set; V1 in parens):

| Parent | lean | V1 | Weak spots |
|---|---|---|---|
| Momentum | 4 | 7 | no volume-confirmation, no consistency/win-rate metric |
| Value | 4 | 5 | shareholder-yield collapses (dividends 0.6% coverage); no EY/PE, no EV/FCF, no yield spread |
| Quality | **2** | 9 | only Altman + Piotroski; ROIC/margins/leverage subs verdicted Remove but ROIC is a proven long-horizon anchor |
| Growth | 3 | 6 | no earnings surprise, no gross-profit growth, no margin expansion |
| Revisions | 5 (+4 gated) | 5 | thin numeric-revision layer; no analyst-count Δ, no buy/hold diffusion, no rating dispersion |
| Institutional | 5 | 5 | `institutional_ownership_summary` fields are untouched (Δ ownership %, put/call, holders change) |
| Insider | 3 | 3 | no cluster-buy alone, no CEO/CFO isolated, no insider-selling-pressure signal, no window-length variants |
| Short | 3 | 3 | no percentile-of-history, no rate-of-change acceleration, no squeeze proxy (short-* momentum) |

**Total production subs: 29 (lean) / 44 (V1).** User target: **8×10 = 80**.

Known gaps in raw data:
- **dividends_paid** and **working_capital**: FMP schema populates these unreliably in the current pull — needs a re-fetch of cash-flow statement / historical dividends endpoint to properly build dividend-yield and NWC-based subfactors.
- **borrow fee / utilization**: not offered by FMP or (as far as we can tell) Massive Starter. Only IBKR / S3 Partners provide this. Propose to **mark as unavailable** and not attempt it.
- **estimate history**: only 2 weeks — numeric-revision subs are structurally forward-accruing; don't test on 2 weeks of data.

---

## 2. Proposed subfactors — 8-10 per parent

**Notation.** `src`: LOCAL (already in `data/db.py`), FMP (new endpoint), MASSIVE
(new endpoint). `sign`: `+` = higher raw is better, `−` = invert. `readiness`:
READY (compute today from local), **FETCH** (needs new backfill), *gated*
(compute today but degrade to neutral in early history).

### Momentum (8 subs — 4 kept + 4 new)

| Name | Intuition | Raw fields | src | sign | readiness |
|---|---|---|---|---|---|
| mom_12_1 (existing) | classic 12-1M return, drops short reversal | daily_prices | LOCAL | + | READY |
| mom_6m (existing) | medium horizon | daily_prices | LOCAL | + | READY |
| mom_52w_high_prox (existing) | anchor near highs | price_features.distance_from_52w_high | LOCAL | + | READY |
| mom_vol_adj (existing) | reward smooth trends | 12-1 / vol_20d | LOCAL | + | READY |
| **mom_3m_ex_1m** | 3-1M return (short-horizon momentum with reversal drop) | daily_prices | LOCAL | + | READY |
| **mom_volume_confirmed** | 12-1 return × sign(rel_volume_20d − 1); rewards momentum backed by volume expansion, penalizes drift on shrinking volume | daily_prices, price_features.relative_volume | LOCAL | + | READY |
| **mom_consistency_60d** | share of last 60 daily returns that are positive | daily_prices | LOCAL | + | READY |
| **mom_max_drawdown_252d** | max drawdown over past year; smaller ⇒ smoother trend | daily_prices | LOCAL | − | READY |

*Deferred as unnecessary given the 4y price history:* longer-window momentum
(24M etc.). *Dropped from proposal:* `mom_acceleration` (already verdicted
Remove — negative IR) and `mom_sector_rel_strength` (redundant with 12-1 after
sector percentile rank).

### Value (10 subs — 4 kept + 6 new)

| Name | Intuition | Raw fields | src | sign | readiness |
|---|---|---|---|---|---|
| val_book_to_price (existing) | classic book yield | equity / mcap | LOCAL | + | READY |
| val_ev_ebitda_inv (existing) | operational cheapness | EV / EBITDA | LOCAL | − | READY |
| val_sales_to_ev (existing) | revenue yield | revenue / EV | LOCAL | + | READY |
| val_shareholder_yield (existing) | buyback + dividend return | buybacks + dividends / mcap | LOCAL | + | READY (weak — dividends 0.6%) |
| **val_earnings_yield** | inverse P/E TTM | net_income / mcap | LOCAL | + | READY |
| **val_fcf_yield_clean** | rebuild of val_fcf_yield with sector-neutral outlier clip (last version verdicted Remove — the raw is fine, the *tails* dominated) | fcf / mcap, winsorized 1%/99% within sector | LOCAL | + | READY |
| **val_ev_fcf_inv** | operational cash cheapness | EV / FCF | LOCAL | − | READY |
| **val_ev_revenue_inv** | revenue multiple (natural form) | EV / revenue | LOCAL | − | READY |
| **val_dividend_yield** | dividend / mcap (needs a real cash dividend series) | `historical_stock_dividend` FMP endpoint | **FMP** | + | **FETCH** |
| **val_buyback_yield** | isolated buybacks/mcap (already reliable) | −buybacks / mcap | LOCAL | + | READY |

### Quality (10 subs — 2 kept + 8 new, several restoring V1 metrics with a distributional fix)

The 7 quality metrics verdicted "Remove" earlier had negative IR *in raw-percentile form*. Two things to try in the expansion:
- **Winsorize / trim** raw values within sector before percentile (heavy tails were dominating).
- **Add composite health signals** that pool multiple raw metrics (fewer degrees of freedom, lower turnover).

| Name | Intuition | Raw fields | src | sign | readiness |
|---|---|---|---|---|---|
| qual_altman_z (existing) | distress model | annual BS/IS | LOCAL | + | READY |
| qual_piotroski_f (existing) | 9-test health score | current + prior annual | LOCAL | + | READY |
| **qual_roic** | operator return on invested capital | fundamental_features.roic | LOCAL | + | READY |
| **qual_gross_margin_wins** | gross margin, winsorized 5%/95% within sector | ff.gross_margin | LOCAL | + | READY |
| **qual_operating_margin** | operating margin | ff.operating_margin | LOCAL | + | READY |
| **qual_fcf_margin** | free cash flow / revenue (previously tested weak — retest with winsorization) | fcf / revenue | LOCAL | + | READY |
| **qual_debt_to_ebitda_inv** | leverage relative to earning power (less noisy than D/E in cyclicals) | ff.net_debt_to_ebitda | LOCAL | − | READY |
| **qual_interest_coverage** | ebit / interest expense (capped at 30x within sector to tame infinities) | ff.interest_coverage | LOCAL | + | READY |
| **qual_asset_turnover** | revenue / total assets | ff.asset_turnover | LOCAL | + | READY |
| **qual_earnings_stability_3y** | 1 / std(EPS_ttm) across last 3y — reward smoother earnings | fundamentals.eps_diluted history | LOCAL | + | READY |

### Growth (10 subs — 3 kept + 7 new)

| Name | Intuition | Raw fields | src | sign | readiness |
|---|---|---|---|---|---|
| grw_earnings_yoy (existing) | EPS YoY | ff.eps_growth_yoy | LOCAL | + | READY |
| grw_revenue_cagr_3y (existing) | 3y revenue CAGR | ff.revenue_cagr_3y | LOCAL | + | READY |
| grw_revenue_acceleration (existing) | Δ in YoY growth rate | ff annual history | LOCAL | + | READY |
| **grw_revenue_yoy** | restore (was in V1) — level of latest revenue growth | ff.revenue_growth_yoy | LOCAL | + | READY |
| **grw_gross_profit_growth** | gross_profit YoY (level 2 signal above revenue growth) | fundamentals.gross_profit | LOCAL | + | READY |
| **grw_operating_income_growth** | operating_income YoY | fundamentals.operating_income | LOCAL | + | READY |
| **grw_ebitda_growth** | EBITDA YoY | fundamentals.ebitda | LOCAL | + | READY |
| **grw_fcf_growth_smoothed** | trailing-2yr average FCF growth (dampen single-year noise) | fundamentals.free_cash_flow history | LOCAL | + | READY |
| **grw_margin_expansion_1y** | operating margin (latest) − operating margin (prior year) | ff.operating_margin history | LOCAL | + | READY |
| **grw_earnings_surprise** | (eps_actual − eps_estimate) / |eps_estimate|, latest reported | earnings_calendar | LOCAL | + | READY |

### Revisions (9 subs — 5 kept + 4 new)

| Name | Intuition | Raw fields | src | sign | readiness |
|---|---|---|---|---|---|
| rev_rating_change_30d (existing) | net rating drift 30d | analyst_revision_features | LOCAL | + | READY |
| rev_rating_change_90d (existing) | net rating drift 90d | analyst_revision_features | LOCAL | + | READY |
| rev_pt_momentum (existing) | consensus PT momentum | analyst_revision_features | LOCAL | + | READY |
| rev_pt_target_upside_30d (existing) | implied upside from PT | analyst_revision_features | LOCAL | + | READY |
| rev_pt_upgrade_ratio_30d (existing) | share of PT upgrades | analyst_revision_features | LOCAL | + | READY |
| **rev_analyst_count_change_90d** | Δ in total_analysts across last 90d snapshots | analyst_revision_features.total_analysts history | LOCAL | + | READY |
| **rev_grade_diffusion** | (strong_buy + buy − sell − strong_sell) / total, latest month | analyst_grades | LOCAL | + | READY |
| **rev_grade_diffusion_change_90d** | 90d change in the diffusion index | analyst_grades history | LOCAL | + | READY |
| **rev_forward_eps_revision_90d** | numeric forward-EPS revision 90d (already stored, gated for coverage) | analyst_estimate_features.forward_eps_revision_90d | LOCAL | + | *gated* |

### Institutional (10 subs — 5 kept + 5 new)

| Name | Intuition | Raw fields | src | sign | readiness |
|---|---|---|---|---|---|
| inst_fund_count (existing) | holder breadth | institutional_signals | LOCAL | + | READY |
| inst_net_share_change (existing) | net accumulation | institutional_signals | LOCAL | + | READY |
| inst_new_positions (existing) | new-money breadth | institutional_signals | LOCAL | + | READY |
| inst_multi_fund_open (existing) | ≥2 funds opened | institutional_signals | LOCAL | + | READY |
| inst_high_conviction (existing) | count of position ↑ | institutional_signals | LOCAL | + | READY |
| **inst_ownership_pct_change** | Δ in whole-market 13F ownership % (much finer than the fund-count aggregate) | institutional_ownership_summary.ownership_percent_change | LOCAL | + | READY |
| **inst_investors_holding_change** | Δ in unique holders count | institutional_ownership_summary.investors_holding_change | LOCAL | + | READY |
| **inst_net_flow_qoq** | shares_change × price (latest snapshot vs prior) — accumulation dollars, sign preserved | ownership_summary.shares_change | LOCAL | + | READY |
| **inst_put_call_ratio_inv** | put/call ratio (rising puts = bearish institutional hedging) | ownership_summary.put_call_ratio | LOCAL | − | READY |
| **inst_concentration_pct** | ownership_percent (level) — high sponsorship = confirmation | ownership_summary.ownership_percent | LOCAL | + | READY |

### Insider (10 subs — 3 kept + 7 new)

Insider is the parent with the most nuance risk. Priors from memory: past
audits found that insider buying **as scored today** underperforms — the memory
project note says "insider Drop/invert (ic -0.19)" was recorded during a
coverage-expansion study. That study was on a *rebuilt* window. Two hypotheses:
(1) the sign is genuinely mixed (managers sometimes buy into value traps);
(2) window length matters. Expansion strategy: keep the direction *higher = better*
as the default, but add explicit **selling-pressure** subs so a rising sell tide
can drag the parent even without any buys.

| Name | Intuition | Raw fields | src | sign | readiness |
|---|---|---|---|---|---|
| ins_net_dollar_flow (existing) | signed $ flow, 180d | insider_transactions | LOCAL | + | READY |
| ins_buy_sell_ratio (existing) | buy$ / (buy$+sell$) | insider_transactions | LOCAL | + | READY |
| ins_high_conviction_buy (existing) | CEO/CFO OR cluster OR ≥$1M | insider_transactions | LOCAL | + | READY |
| **ins_purchase_frequency_180d** | count of distinct buy days over trailing 180d | insider_transactions | LOCAL | + | READY |
| **ins_cluster_buyers_180d** | distinct insider names buying (P) | insider_transactions | LOCAL | + | READY |
| **ins_ceo_cfo_buy_dollars** | $ of P transactions by CEO/CFO only | insider_transactions | LOCAL | + | READY |
| **ins_sell_pressure_inv** | sell$ / mcap over 180d (higher = worse) — critical because production `net_dollar_flow` normalizes weakly; this is a percentile of the raw drag | insider_transactions, mcap | LOCAL | − | READY |
| **ins_net_flow_90d** | shorter-window version of net dollar flow (regime sensitivity) | insider_transactions | LOCAL | + | READY |
| **ins_officer_buy_ratio** | officer P$ / total insider P$ (officer conviction share) | insider_transactions | LOCAL | + | READY |
| **ins_no_selling_flag** | binary: zero sells in the window (contrarian conviction — no insider is unloading) | insider_transactions | LOCAL | + | READY |

### Short (9 subs — 3 kept + 6 new; borrow-fee unavailable and flagged)

| Name | Intuition | Raw fields | src | sign | readiness |
|---|---|---|---|---|---|
| si_short_pct_float (existing) | direct short pressure | short_interest | LOCAL | − | READY |
| si_days_to_cover (existing) | days to cover | short_interest | LOCAL | − | READY |
| si_short_interest_change (existing) | Δ short % of float | short_interest | LOCAL | − | READY |
| **si_short_pct_float_percentile_252d** | own-history percentile of pct float over trailing year (contextualizes level) | short_interest | LOCAL | − | READY |
| **si_short_trend_90d** | linear-slope of short % over trailing 90d | short_interest | LOCAL | − | READY |
| **si_short_change_accel** | (Δ pct float latest) − (Δ pct float prior period) — rising rate of change | short_interest | LOCAL | − | READY |
| **si_short_covering_signal** | large negative Δ (top decile of falls) — bullish covering | short_interest | LOCAL | + | READY |
| **si_squeeze_setup_flag** | high short + strong momentum flag (informational; keeps the existing helper as an actual score) | short_interest, momentum_score | LOCAL | + / condit. | READY |
| **si_short_dollars_to_mcap** | shares_short × price / mcap — dollar-normalized short pressure | short_interest, prices, mcap | LOCAL | − | READY |
| ~~borrow_fee / utilization~~ | not in FMP or Massive Starter — flagged unavailable |

---

## 3. New raw-data fetches

| Endpoint | Source | Purpose | Notes |
|---|---|---|---|
| `/historical/stock_dividend` (per ticker, 4y) | FMP | Populate real dividend-yield subfactor | Cash dividends only; skip stock splits. |
| (nothing else) | | | Every other subfactor above is derivable from local tables |

Massive Starter would only add price/volume backfills (we already have 4y). If
you have specific Massive endpoints in mind (e.g., ETF flows, borrow proxy),
name them — the current codebase has no Massive provider integrated.

---

## 4. What this pipeline will *not* try

- **Wire expansions into production `factors/`** — kept isolated in
  `research/subfactor_expansion/library.py` so the daily scoring pipeline is
  unaffected until you decide to promote specific subs (`config.factor_sets`).
- **Long horizons past ~3 years** — prices only go back to 2022-06-21, so
  standalone IC / Q5-Q1 spreads at 12M horizons will have limited windows.
- **Sub-daily / options-flow / borrow-fee** — not available in current
  providers.

---

## 5. Deliverables when built

- `research/subfactor_expansion/library.py` — candidate computations
- `research/subfactor_expansion/validation.py` — coverage / IC / monotonicity / hit-rate / stability
- `scripts/subfactor_expansion/fetch_dividends.py` — the one new backfill (optional)
- `run_subfactor_expansion.py` — CLI: `--audit`, `--build`, `--validate`, `--report`
- `output/subfactor_expansion/REPORT.md` + per-parent CSVs
