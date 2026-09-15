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

## New top-20-pool champion (2026-09-12): insider_revisions_min_top20

After the 101-idea systematic exploration batch and the alpha-adjustment
correction (below), a new candidate is better-supported than
`dynamic_ic_momentum_top20_selector` on every axis and is promoted as the
new top-20-pool champion: `research.loop_research.exploration_batch_100.
metric_insider_revisions_min` (wired via `make_selector`). Rule: within the
top-20-by-composite-rank pool, rank by `min(insider_score, revisions_score)`
descending -- i.e. BOTH insider strength and revisions strength must be
present, not just one (a floor/AND requirement, not a sum or average).

| metric | baseline | candidate |
|---|---|---|
| Sharpe | 0.962 | **1.261** (+0.299, largest of any candidate tested) |
| CAGR | 0.163 | **0.230** |
| Alpha vs SPY | 0.047 | **0.108** (largest of any candidate tested) |
| Beta vs SPY | 0.811 | 0.785 (lower -- not a leverage story) |
| Max drawdown | -0.216 | -0.245 (worse by ~2.9pp -- the one real cost) |

Win rate 99.2%, perturbation win rate 98.6%. **Alpha-adjusted block
bootstrap (2026-09-12 methodology, see below) at L=4: p=0.015, IMPROVING
from the raw p=0.024** -- one of only two candidates in the entire 119-test
sweep (the other being this exact `==100`-pool champion) whose significance
gets stronger, not weaker, once beta is controlled for, because it runs at
lower beta than the baseline. 6 of 7 years positive (only 2022 negative);
best single year supplies only 31.5% of the total -- the least concentrated
result found all session.

**Why it's more convincing than just "insider is good" (which we already
knew):** three sibling combinations of the exact same two parents were also
tested -- `insider_revisions_sum_top20` (raw bootstrap p=0.243),
`insider_revisions_product_top20` (p=0.335), and
`insider_revisions_balance_top20` (smallest gap between the two, p=0.362)
-- all landed far weaker. The MIN/floor logic specifically (both signals
must independently clear a bar) is what works; a generic blend of the same
two parents does not. That sharp separation from three close relatives is
good evidence this isn't just a lucky pick out of a 100+-idea sweep.

Economic rationale: requiring simultaneous insider-buying conviction AND
analyst/revisions momentum is a stronger interaction filter than either
alone -- each factor screens out the other's false positives, rather than
one strong parent (e.g. insider alone) being allowed to compensate for the
other being weak or negative.

`dynamic_ic_momentum_top20_selector` is demoted to a close second: it's
real (alpha-adjusted p=0.060, also improves under beta-adjustment since it
too runs slightly light on beta) but has a more complex, more-moving-parts
mechanism (rolling windows, YoY comparison, rotating parent selection) and
more year-to-year concentration (48% of the edge from one year) than the
new champion's simple static floor. Kept on record as a validated, still-
reasonable alternative, not discarded.

### Flagged but NOT promoted: the "recency / re-rating" theme

Two related ideas -- `new_entrant_top20` (freshly promoted into the top-20
pool) and `scoremom_1m_top20` (composite score risen the most over the
trailing 1 month) -- both surfaced as leading candidates in the 101-idea
batch and share a common theme (very recently strong/improving names
outperform), but **neither is confirmed** once scrutinized the same way as
the champion above:
- `scoremom_1m_top20`: alpha-adjusted bootstrap p WORSENS (0.076 -> 0.121)
  -- it carries meaningfully higher beta than baseline (0.849 vs 0.822), so
  part of its apparent edge is extra market exposure, not selection skill.
  Its rank-based sibling `scorerankchange_1m_top20` shows the identical
  pattern (0.106 -> 0.151, also higher beta).
- `new_entrant_top20`: alpha-adjustment actually helps (p: 0.133 -> 0.059,
  it runs at lower beta) but only 4 of 7 years are positive and 54% of the
  total edge comes from a single year (2022) even after adjustment -- too
  concentrated to trust yet.
- Also notable: the exact same score-momentum IDEA at longer lookbacks (2,
  3, 6, 9, 12 months) collapses hard -- `scoremom_3m_top20` and beyond are
  among the worst-performing candidates in the entire 101-idea batch. The
  1-month version sitting right at the edge of a sharp lookback cliff is
  itself a reason for caution (see session_log.md's "100-idea batch"
  write-up for the full lookback table).

**Status: promising theme, not a promoted rule.** Worth a dedicated,
better-controlled follow-up (e.g. testing whether a beta-neutralized or
smaller, more disciplined version of "recency" survives) before treating
either as a real finding, rather than folding it in alongside the
confirmed champion above.

## Alpha-adjustment correction (2026-09-12): raw excess return doesn't control for beta

User-identified gap: the block-bootstrap significance check (and the
`evaluate_promotion` calendar-time gates -- `avg_monthly_excess`,
`year_sum`, `n_years_positive`, `majority_years`,
`best_year_removed_negative`) all originally used the RAW
`candidate_return - baseline_return` monthly difference. A candidate that
simply carries more market beta than the baseline shows positive "excess
return" almost by construction in a mostly-bullish window -- that's
leverage, not selection skill, and the raw check couldn't distinguish the
two.

Fix, now wired directly into `research/loop_research/promotion.py`'s
`evaluate_promotion` (applies to every future candidate automatically, not
just a one-off script): estimate one full-sample beta per leg (candidate,
baseline) against SPY, and bootstrap the CAPM-residual difference --
`(candidate_return - beta_candidate*SPY_return) - (baseline_return -
beta_baseline*SPY_return)` -- instead of the raw difference. Raw numbers
are still computed and reported alongside (`raw_avg_monthly_excess`,
`raw_diff_by_date`, `beta_candidate_monthly`, `beta_baseline_monthly`) for
transparency.

Reran the top 10 candidates from the 101-idea batch under this correction
(full table: output/loop_research/alpha_adjusted_top10_bootstrap.json).
Clean pattern: every candidate with LOWER beta than baseline got MORE
significant; every candidate with HIGHER beta got weaker, several
dramatically (`subdiff_ins_no_selling_flag_minus_si_short_pct_float_top20`
0.076 -> 0.184; the deliberate negative control
`parents_above_median_contrarian_top20` 0.148 -> 0.437, fully debunked).
Only `insider_revisions_min_top20` and `top5_by_insider` are unambiguously
confirmed once beta is controlled for -- see the champion sections above.
Full narrative + per-candidate table: session_log.md and the published
promotion-audit report (output/loop_research/promotion_audit_report.html).

## 10-idea challenge batch (2026-09-12): champion survives, one near-miss flagged

10 new ideas tested one at a time directly against `insider_revisions_min_top20`
itself (not the random baseline) via `research.loop_research.
exploration_batch_2026_09_12` -- full table in session_log.md. 9 of 10
REJECTED cleanly (adding a 3rd floor leg, chasing insider re-rating or
stability, risk-scaling the floor by volatility, OR-instead-of-AND logic,
subfactor-level granularity, and diluting freshness into a smooth tenure
continuum all failed, several badly). Champion is unchanged.

One near-miss, not promoted: `insider_revisions_min_new_entrant_top20`
(the floor plus a bonus for names freshly promoted into the top-20 pool
this month) clears every mechanical gate except one -- Sharpe +0.143, 100%
win rate, alpha improves, drawdown improves, 63.6% perturbation win rate,
and passes leave-one-year-out for 6 of 7 years. It fails only because
removing the single best year (2022) leaves the cumulative edge at -0.0104
(essentially flat, not a real reversal) -- a much healthier failure mode
than the earlier standalone `new_entrant_top20` (which concentrated 54% of
its edge in 2022 alone). Flagged as PROMISING, worth an alpha-adjusted
block-bootstrap significance check before any promotion decision.

## Noise-robustness deep dive (2026-09-13, user-driven): champion vs. new-entrant

Follow-on user session digging into exactly how the perturbation-robustness
gate works and whether `insider_revisions_min_new_entrant_top20`'s
apparent edge over the champion survives scrutiny. Scripts:
`research.loop_research.perturbation_sweep` (noise-multiple sweep vs.
random), `research.loop_research.bootstrap_two` (alpha-adjusted block
bootstrap for both strategies vs. random), `research.loop_research.
compare_three` (pick-overlap + head-to-head stats for champion /
new-entrant / a pure 1-month score-momentum control).

**Mechanism recap.** The perturbation test (`make_perturbed_selector` in
`exploration_batch_2026_09_11.py`) adds independent Gaussian noise to each
pool stock's ranking score every month: `noise_std = noise_mult * that
month's own cross-sectional std of the score` (default `noise_mult=0.5`).
For the champion this is on the raw ~0-100 `min(insider,revisions)` scale
(typical std ~9.9); for new-entrant it's on the combined `floor_rank +
0.5*is_new` scale (~0-1.5, typical std ~0.49) -- different absolute units,
same *relative* amount of scrambling by construction.

**How much noise actually moves things (81 months x 30 draws/level):**

| noise_mult | champion avg rank change (/20) | champion avg top-5 flips | new-entrant avg rank change (/20) | new-entrant avg top-5 flips |
|---|---|---|---|---|
| 0.5 | 2.30 | 1.27 | 2.09 | 1.11 |
| 1.0 | 3.63 | 2.04 | 3.49 | 1.92 |
| 1.5 | 4.43 | 2.50 | 4.32 | 2.41 |
| 2.0 | 4.92 | 2.76 | 4.84 | 2.69 |
| 3.0 | 5.48 | 3.08 | 5.44 | 3.06 |
| 4.0 | 5.76 | 3.25 | 5.71 | 3.23 |

The two strategies get scrambled by almost identical amounts at every
noise level -- a fair, apples-to-apples comparison of what happens next.

**Noise-multiple sweep, win rate vs. random_top20_selector:**

| noise_mult | champion win rate | new-entrant win rate |
|---|---|---|
| 0.5 | 98.6% | 100.0% |
| 1.0 | 94.4% | 97.5% |
| 1.5 | 90.0% | 93.5% |
| 2.0 | 83.0% | 86.5% |
| 3.0 | 77.2% | 78.5% |
| 4.0 | 69.2% | 71.0% |

Both decay gracefully and monotonically with no cliff -- the signature of
a broad effect, not a knife-edge artifact. New-entrant wins slightly more
often than the champion at every level vs. random.

**But the head-to-head comparison at 1x noise tells a different story.**
Full stats (100 real Monte Carlo sims each, `H.sim_metrics`):

| metric | champion 0x | champion 1x noise (mean) | new-entrant 0x | new-entrant 1x noise (mean) |
|---|---|---|---|---|
| Sharpe | 1.261 | 1.229 (-0.035, -2.8%) | 1.404 | 1.221 (-0.195, -13.9%) |
| Alpha vs SPY | 10.83% | 9.65pp (-1.17pp, -10.8%) | 12.87% | 9.32pp (-3.61pp, -28.0%) |
| CAGR | 22.96% | 21.90% | 26.25% | 21.75% |
| Max DD | -24.48% | -23.79% | -21.15% | -22.21% |

**Key finding: new-entrant loses ~3x the alpha and ~5x the Sharpe that the
champion loses, from an equally-sized noise scramble (3.49-3.63 avg rank
change either way) -- and its post-noise stats end up statistically
indistinguishable from (very slightly worse than) the champion's own
post-noise stats**, despite starting ~2pp of alpha ahead at 0x. Only 4% of
new-entrant's noisy sims beat its own 0x version (vs. 36% for the
champion) -- its clean result sits further out on its own noise
distribution's tail than the champion's does.

**Interpretation:** new-entrant's entire incremental edge over the
champion appears to be riding on a small number of precisely-ordered,
noise-sensitive picks (correctly identifying which handful of *freshly
arrived* names also have strong insider/revisions floors) rather than a
robust structural effect. This is consistent with, and helps explain,
everything else found about this candidate: it shares only ~60% of its
picks with the champion (`compare_three.py` overlap analysis), and its
significance vs. random -- while nominally stronger than the champion's on
a pure alpha-adjusted block-bootstrap basis (p=0.0002 vs p=0.0148 at L=4,
see `bootstrap_two.py` output below) -- does not survive a noise stress
test nearly as well as the champion's own edge does. **Net read: the
simpler mechanism (plain floor, no recency tilt) is the more durable one.**
The bootstrap comparison alone would have argued for treating new-entrant
as the stronger finding; the noise-sensitivity comparison argues the
opposite; taken together, this is a genuine case where standard
significance testing and robustness testing point in different directions,
and the honest conclusion is "not clearly better than the champion,"
not "clearly better."

**Alpha-adjusted block bootstrap vs. random_top20_selector (L=1,3,4,6,12
months, `research.loop_research.bootstrap_two`):**

| block_len | champion p-value | champion CI excludes 0? | new-entrant p-value | new-entrant CI excludes 0? |
|---|---|---|---|---|
| 1 | 0.0126 | yes | <0.0001 | yes |
| 3 | 0.0152 | yes | 0.0004 | yes |
| 4 | 0.0148 | yes | 0.0002 | yes |
| 6 | 0.0232 | yes | 0.0008 | yes |
| 12 | 0.0570 | **no** | 0.0002 | yes |

New-entrant's CI stays clear of zero at every block length tested
(including L=12, where the champion's own CI touches zero and loses
significance) -- on this test alone, new-entrant looks like the more
robust finding. This is the tension the noise-sensitivity analysis above
resolves: new-entrant's total edge over random is real and well-supported,
but a large share of it is likely the same effect the champion already
captures (shared ~60% of picks), amplified -- not a clean, independent,
equally-durable addition on top.

**Status unchanged: champion remains `insider_revisions_min_top20`.**
`insider_revisions_min_new_entrant_top20` is downgraded from "PROMISING,
worth a bootstrap check" to "real edge vs. random, but incremental value
over the champion looks fragile under noise -- not recommended for
promotion without further work" (e.g. per-year decomposition of the
new-entrant-specific incremental contribution, isolated from the shared
floor logic).
