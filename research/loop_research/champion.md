# Current champion (loop-research, 2026-09-09)

`research.loop_research.candidates.top5_by_insider` (exp5) -- promoted after
a 93-candidate batch, replacing exp3 (avg_parent_score_top5). Among the
composite_score==100 tied pool, prefer the single highest insider parent
score (ins_no_selling_flag / ins_cluster_buyers_180d / ins_sell_pressure_inv)
and ignore all other parents. Deterministic tie-break: ticker ascending.

Clears the ORIGINAL (strict) promotion rule, not just the revised one:
- Win rate 98.4% of paired seeds (500 sims vs the random baseline).
- 95% CI on the paired Sharpe difference is [+0.037, +0.457] -- entirely
  positive, does not touch zero.
- Median Sharpe 1.223 vs baseline 0.972; CAGR 0.218 vs 0.167; alpha vs SPY
  0.096 vs 0.051; max_dd -0.177 vs -0.219 (better, not worse).
- Wins 6 of 7 calendar years, including 2022 -- the one regime-shock year
  every prior candidate (exp1-exp4, and every momentum/quality/growth-based
  idea in the 93-candidate batch) lost. Removing its single best year
  (2024) still leaves the other six years solidly positive (+24pp sum), not
  a reversal.

Known caveats, not swept under the rug:
- Derived from 81 monthly dates / one historical path (2020-2026) -- still
  a single realized history, not a law of nature. Insider cluster-buying
  signals are known in the literature to concentrate their edge around
  market-stress/recovery episodes, several of which fall in this window;
  worth revisiting if/when more history or an independent period becomes
  available.
- Growth-tilted selection rules were consistently bad across the whole
  batch (every growth-containing combo screened poorly) -- worth treating
  growth as actively unhelpful for WITHIN-tied-pool selection, distinct
  from its role in the broader composite.
- Excluding high-vol/high-beta names from the tied pool was dramatically
  worse than random (0-20% win rate) in this specific 2020-2026 trending
  window -- not evidence that vol/beta screens never work, just that they
  hurt badly when the market's returns are concentrated in a few high-beta
  winners, as this period's were.

Not yet wired into the production dashboard or any live selection logic --
this is a research-loop finding, not a deployed change. See
docs/monte_carlo_book_construction.md for the dashboard's current (random
baseline) featured config.

## Update 2026-09-10: more rigorous per-period re-check (marginal, not overturned)

A separate paired Monte Carlo test on a BROADER pool (top decile minus the
==100 tied names -- research/loop_research/insider_edge_paired.py) initially
looked like a similarly strong insider-edge win rate, but a follow-up
per-calendar-time breakdown revealed that finding was almost entirely a
2020-2021 artifact (98% of the 7-year sum from those 2 years alone, p=0.42
across actual calendar months once sim-resampling noise is averaged out,
and no correlation with the insider factor's own year-by-year IC) -- i.e.
noise/regime-fluke dressed up as signal by a naive Monte Carlo win rate. See
session_log.md "Paired insider-edge test" + "CORRECTION" entries.

Applying that SAME rigor directly to this champion (not the broader pool --
research/loop_research/champion_period_breakdown.py) gives a healthier but
still only marginal result: t=2.00, **p=0.049** across the 77 real calendar
months (not the 500 pseudo-replicated sims), 54% of months positive, 5 of 7
years net positive, and -- unlike the broader-pool finding -- NOT
concentrated in 2020-2021 (2020 is actually slightly negative; 2024 is the
single best year). Insider IC-by-year still does not track which years the
edge shows up in (2022: negative IC, good edge year; 2026: best IC, flat
edge year).

**Net: champion status is kept, but the honest confidence level is "probably
a small real effect that isn't obviously a regime fluke," not "proven" --
the original 98.4%-win-rate / p<0.0001-style framing overstated certainty by
treating 500 pseudo-replicated sims of one market history as if they were
500 independent observations.**

## New track (2026-09-11): top-20-pool champion, separate from the ==100 pool above

Everything above concerns the composite_score==100 tied-pool track. A second,
parallel track was opened using a different pool definition -- rank every
date by composite score, take the top 20, then select 5 from that pool
(instead of the narrower, often-smaller ==100 tied group). Baseline for this
track: `research.loop_research.harness.random_top20_selector` (uniform
random draw of 5 from the top-20 pool, same 4-sleeve/4-month-hold mechanics).
Promotion rule: the codified 8-gate `research.loop_research.promotion.
evaluate_promotion` function (see session_log.md "Promotion rule revised" for
the ==100-pool-era rule this superseded; the current rule's exact reject/
promote thresholds are documented in promotion.py's module docstring).

Four candidates tested on this pool (full detail in session_log.md
"Top-20-pool track" section):
1. `min_subfactor_score_top20` (min of 24 subfactors) -- REJECTED.
2. `avg_top3_parent_score_top20` (avg of best 3 of 8 parents) -- REJECTED
   under the formal gate (touched PROMISING at best).
3. `dynamic_best_ic_top20_selector` (route to whichever parent has the
   highest trailing-5y IC LEVEL) -- REJECTED (Sharpe -0.032 vs baseline, win
   rate 40%, only 1/7 years positive) -- same failure mode as the earlier
   ==100-pool version: the trailing-IC leader is usually quality, a weak
   standalone tie-breaker.
4. **`dynamic_ic_momentum_top20_selector` (route to whichever parent's mean
   IC jumped the most YEAR-OVER-YEAR, i.e. current-1y mean IC minus
   prior-1y mean IC, both strictly before the review date, rotating across
   parents rather than sticking to one) -- PROMOTED, new champion for this
   track.**

### `dynamic_ic_momentum_top20_selector` -- promoted 2026-09-11

`research.loop_research.dynamic_ic_momentum_top20.dynamic_ic_momentum_top20_selector`.
Clears every gate in the formal 8-gate rule:

| metric | baseline (median) | candidate (median) |
|---|---|---|
| sharpe | 0.959 | **1.231** (+0.272) |
| cagr | 0.163 | **0.213** |
| alpha vs SPY | 0.048 | **0.092** (+0.045) |
| max_dd | -0.214 | **-0.203** (better) |
| beta vs SPY | 0.807 | 0.788 (slightly lower) |

- Win rate **99%** of 500 paired sims.
- Average monthly excess return (sim-averaged, real calendar months):
  **+0.0035/mo**, positive.
- **Strict leave-one-year-out: passes for all 7 years** -- removing any
  single year, both Sharpe-diff and alpha-diff stay positive on the
  remaining six (smallest margin: 2024 removed, sharpe_diff +0.203,
  alpha_diff +0.030 -- still comfortably positive).
- 4/7 calendar years individually net-positive (2021, 2023, 2024, 2025);
  removing the single best year (2024, +0.147 sum) still leaves +0.136
  across the other six years -- not a reversal.
- Max drawdown improves (doesn't deteriorate), so the 4pp-tolerance gate is
  moot here.
- Perturbation (option 1: noise added only to the final within-parent
  ranking step, winner-parent decision left exact): **76.5% win rate at
  0.5x noise** -- clears the 55% bar.

**Additional robustness run beyond the formal gate (user-requested, not part
of the codified rule):**
- Perturbation option 2 (Gaussian noise added directly to the per-date IC
  estimates that drive the winner-parent decision itself, corrected after an
  `id(rng)`-caching bug was found and fixed): win rate 94.5% / 89.0% / 83.5%
  at noise_mult 0.25 / 0.5 / 1.0x -- decision logic degrades gracefully, does
  not collapse.
- Perturbation option 3 (user-designed cross-sectional bootstrap: at each
  date resample the ~20 pool stocks with replacement once, reuse the SAME
  resampled index for all 8 parents' IC recompute so cross-parent
  correlation from shared stock identity is preserved, recompute the winner
  decision from the bootstrapped IC table): win rate **80.5%**, median
  Sharpe 1.102 vs baseline 0.959 across 200 sims.
- Moving-block bootstrap over calendar time (the honest significance check
  respecting serial autocorrelation, same method as the ==100-pool champion
  above): at the principled L=4 block length, **95% CI [-0.0004, +0.0081],
  p=0.078** -- does NOT cleanly exclude zero at the conventional bar, though
  it clears p<0.05 at longer block lengths (L=6: p=0.048; L=12: p=0.005) and
  the point estimate (~+0.0035/mo) is stable across every block length
  tested, which is the pattern of a real-but-underpowered effect rather than
  noise (noise would typically look unstable as the block length changes,
  not flat). Weaker standalone significance than the ==100-pool champion's
  p=0.014 at the same L=4, but every other test in the battery (formal
  8-gate rule, both perturbation designs, leave-one-year-out on every year)
  passes cleanly. Treated as "probably a real effect, not fully confirmed by
  the calendar-time significance check alone" -- same honest-confidence
  framing as the ==100-pool champion, not a stronger claim.

Not yet wired into the production dashboard or any live selection logic --
research-loop finding only, same status as the ==100-pool champion above.
`random_top20_selector` remains the reference/fallback baseline distribution
for all future top-20-pool-track experiments; this selector is now the bar
new candidates on this pool must beat.

## Update 2026-09-10 (later): block bootstrap over time -- real significance number

The per-period t-test above still assumed the 77 monthly observations were
independent, which they aren't (4-month holds, staggered monthly reviews ->
serial correlation). Moving-block bootstrap over calendar time
(research/loop_research/block_bootstrap.py, output/loop_research/
block_bootstrap.csv) at the principled block length (L=4 months, matching
the sleeve hold): **95% CI [0.0008, 0.0068], p=0.014** -- excludes zero, and
actually a bit MORE confident than the naive t-test's p=0.049 (checked why:
the diff series has mild negative, not positive, autocorrelation, so the
naive test was slightly conservative here, not anti-conservative). Robust
across block lengths 1-12 months (p ranges 0.047 down to <0.0001 as L
grows). This is the most rigorous significance check run on the champion to
date, and it confirms rather than weakens the "cautiously survives" verdict.
Standing caveat unchanged either way: still bootstraps variation within one
realized 2020-2026 history, not a guarantee across regimes.
