# Pairs Trading — Strategy Spec, Findings & Roadmap

*Research window: 2017-H1 – 2026-H1 (19 semiannual OOS windows), built 2026-07-15
over 7 research rounds. Code: `pairtrading/` (`study.py`, `forensics.py`),
runner `run_pairtrading_study.py`, artifacts in `output/pairtrading/`.
Validation discipline throughout: every rule must replicate in BOTH halves of
the sample (2017-21 "discovery" / 2022-26 "holdout") or it is rejected.*

---

## 1. Current strategy (the spec)

**Universe & formation** — PIT S&P 500 (`members_as_of` at each window start,
delisted names included). Every 6 months, form pairs on the trailing 12 months
of adjusted prices: same GICS sector (synonyms normalised), ranked by sum of
squared deviations (SSD) between normalised price paths, one appearance per
ticker, top 20 traded.

**Entry** (all conditions required):
1. Spread ≥ 2.0σ of its formation std (signal at close, execute next close —
   GGR one-day wait).
2. ≤ 2.75σ at execution — deeper dislocations are broken pairs, not bargains.
3. ≥ 40 trading days left in the window — no runway, no trade.
4. Composite tilt gate: composite(long leg) ≥ composite(short leg) at the
   latest PIT rebalance — never short the name the model prefers.
5. Hot-streak stand-down (`RegimeGate`): no new entries while the book's own
   trailing 42-day return exceeds +1%.

**Position & exit:**
- Long the leg below, short the leg above, $1/$1 dollar-neutral.
- Pile-on: if an open trade deepens to 3σ, double it once.
- Exit at |spread| ≤ 0.5σ or a zero-cross, whichever first; forced exit at
  window end or delisting. Costs 10bp per side throughout.

**Capital** — "invested" accounting: capital is spread across open positions
only, floored at 8 slots (max 1/8 of capital per unit position). Reserve
buffer sits in T-bills (not credited in the backtest numbers below).

**Headline stats (net, 9.5y):**

| basis | CAGR | Sharpe | Sortino | hit | max DD | t-stat | trades/yr |
|---|---|---|---|---|---|---|---|
| committed capital (`pileon_hot`) | 2.1% | 0.69 (0.69/0.70 by era) | 0.89 | 73% | −5.6% | 2.14 | ~26 |
| invested capital (adopted) | 3.8% | 0.66 (0.60/0.77) | 0.81 | 73% | −10.4% | ~2.0 | ~26 |

Beta vs SPY: **0.01**, correlation 0.075. Every calendar year 2022-2025
positive; known dips: 2021-H2 and 2026-H1 (see §3).

**Portfolio role** (2017-26, daily-rebalanced blends):

| | CAGR | Sharpe | max DD | worst yr |
|---|---|---|---|---|
| SPY | 13.6% | 0.79 | −34% | −23.0% |
| 80/20 SPY+pairs | 11.9% | 0.83 | −28% | −17.9% |
| 80/20, pairs at 3x (−2%/yr financing) | 12.6% | 0.85 | −28% | −17.4% |
| 50/50 SPY+pairs | 9.0% | 0.93 | −19% | −10.0% |

---

## 2. What we figured out (7 rounds)

**R1 — classic replication.** GGR distance pairs on modern S&P 500 large caps
are ~dead net of costs (Sharpe 0.18; gross ~+1.7%/yr, costs eat ~1.1%) —
matches the Do & Faff decay literature. Cointegration filtering HURTS (swaps
tight pairs for looser ones); naive stop-losses hurt (whipsaw churn).
Composite score gates helped mildly even here.

**R2 — "strict entries" falsified.** A pre-registered discipline bundle
(quality pairs, 2.5σ entry, confirmation, stops, time stop) was decisively
negative (Sharpe −0.66). Ablation: both halves hurt independently. The classic
edge lives in tight-σ, patient, 50+-day trades; selectivity deleted exactly
the trades that made money.

**R3 — trade forensics → first real rules.** Outcome is decided by
convergence-before-window-end (converged: 97% win, +5.0%; window-end: 28%,
−3.3%); winners and losers are nearly indistinguishable *before* entry.
Two margins replicate: entries > 2.75σ lose, entries with < 40d runway lose
→ `ADJUSTED`. Rejected (didn't replicate): sub-2σ pullback entries, VIX,
widening speed, half-life, sector.

**R4 — the score is the edge; exit earlier.** `comp_diff` (composite long −
short) is the ONE score feature separating winners from losers in both eras;
losers disproportionately short names the model likes. Converged trades touch
0.5σ ~a week before the zero-cross, and 27% of window-end losers touch it too
→ 0.5σ take-profit + tilt gate = `TUNED`, first variant positive in both eras.
Parent-level features (short_diff, momentum levels) failed replication — the
composite beats any single parent.

**R5 — user-idea grid.** Correlation floors (0.90/0.95) worse both eras (SSD
already selects co-movers; corr floor admits looser pairs). Absolute score
levels (long ≥ 50th pct, short ≤ 25th) work but are too selective — relative
tilt beats absolute levels. 1.0σ/1.5σ entries don't clear costs; 2σ is the
sweet spot. Pile-on at 3σ: hit rate 66→71%, return ~doubles, Sharpe flat —
an intra-trade leverage dial (adopted as optional).

**R6 — sizing and regime.** comp_diff SIZING rejected before building:
Spearman(comp_diff, payoff) ≈ 0 in both eras among gate-passers — the score
signal is binary. Drawdowns (2021-H2, 2026-H1) are synchronized all-sector
window-end failures (regime breaks), not bad single entries; concurrency
signal failed replication, but the hot-streak stand-down (contrarian on the
book's own trailing P&L) replicates → `RegimeGate` adopted; first t > 2.

**R7 — CAGR push.** The binding constraint was capital, not signal: invested
accounting (fund open trades only) lifts CAGR 2.1% → 3.8% with identical
trades. Rejected: 50-pair book (quality decays past top-20), news-proxy
entry block (big-move divergences here DO converge), dispersion stand-down
(halves trades). Vol-sizing neutral. **No replicating entry signal remains
untested-and-unused; the residual constraint is structural: ~26 trades/yr ×
~1.7% average payoff.**

**Infrastructure gotchas for future work:** the `daily_prices` pivot has
all-NaN holiday rows (off-exchange prints) — `dropna(how="all")` before any
`iloc[0]`-anchored normalisation; stop-outs must block re-entry until the
spread re-enters the band or the sim churns 5x trades; `RegimeGate` needs the
strategy's own daily series (two-pass in `run_grid`).

---

## 3. Known limitations

- **Absolute edge is small**: t ≈ 2.0–2.1 after seven rounds of (disciplined)
  iteration on one dataset; live performance should be expected below
  backtest. This is a diversifier (β = 0.01), not a return engine.
- **Regime-break drawdowns are unfixed**: late-2021 and 2026-H1 are
  synchronized divergence episodes; nothing measurable at entry time flagged
  them (VIX, concurrency, dispersion, scores all failed). Treat ~−10% (invested
  basis) as the strategy's recurring cost of doing business.
- **Multiple-comparisons debt**: the hot-streak threshold (42d/+1%) was 1 of 3
  tested; the era-split guards against gross overfit but ~20 ideas have now
  touched this dataset. The honest prior on live Sharpe is below 0.66.
- No T-bill credit, no borrow-fee modelling on the short legs, monthly-ish
  composite staleness for the gate.

## 4. Next steps

1. **Paper-trade it** (the real test): wire the 20-pair book into the Alpaca
   paper stack alongside the loser-screen sleeve (`run_portfolio` /
   `run_execution --from-approved` pattern). Verify live hit rate tracks ~70%
   and fills at close ± spread assumptions for 2-3 quarters before any capital.
2. **Weekly cadence study** (structural capacity): shorter
   formation/trading cycles (e.g., 6M formation / monthly trading with weekly
   scans) to raise trade count — the only remaining lever for CAGR that isn't
   leverage. New study, real look-ahead care needed.
3. **Mid/small-cap universe** (user vetoed 2026-07-15; revisit only if wanted):
   the literature says the anomaly survived there; needs a PIT membership
   source (FMP has none for S&P 400/600) — Polygon reference-tickers-by-date +
   historical mcap is the plausible route.
4. **Borrow-fee & T-bill realism pass**: credit reserve cash at FRED 3M yield,
   debit hard-to-borrow fees per leg; re-verify the blends.
5. **Sleeve integration**: if paper-trading confirms, the 80/20 SPY+pairs
   (optionally walking the sleeve toward 2-3x) is the deployment shape —
   portable alpha on top of equity beta, not a standalone book.

---

## Addendum 2026-07-18 — relative-value expansion program (see output/relvalue/)

A follow-on program (`relvalue/`, `run_relvalue_study.py`) generalized this
study to cointegration selection, stock-vs-ETF, ETF-ETF, share classes, and
ratio/residual signals on a **repaired total-return price panel** (the study
above ran on `adj_close`, which mixes dividend conventions at the 2022-06-21
source splice — see output/relvalue/FINAL_REPORT.md §1). Key outcomes:
the distance+tilt chassis reproduces (Sharpe 0.60 vs 0.66 here — both
programs mutually validated); adding reversal confirmation reached 0.75 in
eval; the hot-streak gate did NOT replicate on clean data; all other families
failed; and the untouched 2025-07→2026-06 holdout was decisively negative for
every finalist (−6% to −8.5%), confirming the 2026-H1 regime break noted in
§3. **Deployment of any pairs sleeve remains gated on a positive forward
stretch.**
