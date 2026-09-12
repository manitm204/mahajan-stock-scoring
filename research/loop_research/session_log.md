# Loop-research session log — staggered-sleeves stock-selection search

Started 2026-09-09. Baseline = production featured config: 5-stock book,
4-month hold, 4 staggered sleeves, full turnover each review (REFRESH_N=5),
uniform random draw from the composite_score==100 pool each time a slot is
filled. Metrics use the fixed `research.autoresearch.evaluate.bench_returns`
(start-labeled, matches `compute_portfolio_returns`) — see
docs/monte_carlo_book_construction.md for why this matters (the prior
mislabeled version reported a spurious negative beta).

Immutable: `research/loop_research/harness.py` (sleeve bookkeeping, T+1 exec,
costs, Monte Carlo seeding, metric definitions). Only the `selector` function
passed into each experiment varies. Every experiment recorded below,
including failures — nothing is deleted after a rejection.

Decision rule: candidate promoted only if median OOS Sharpe improves, wins
>=60% of paired seeds, isn't driven by one year, and CAGR/alpha doesn't come
with materially worse max drawdown. Full record incl. all metrics in
output/loop_research/results.json.

## exp1_min_parent_score_top5
**Hypothesis:** Among the score==100 tied pool, prefer names with no glaring weak spot in any single parent family (rank by min of the 8 V4 parent scores, take top 5) instead of a uniform random draw.
**Params:** {}  |  deterministic=True  |  n_sims=500

| metric | baseline median | candidate median | paired diff (median) |
|---|---|---|---|
| cagr | 0.1669 | 0.1743 | +0.0074 |
| sharpe | 0.9715 | 1.0365 | +0.0650 |
| sortino | 0.9516 | 0.9606 | +0.0090 |
| max_dd | -0.2195 | -0.2245 | -0.0050 |
| avg_turnover | 0.4375 | 0.3975 | -0.0400 |
| spy_beta | 0.8139 | 0.7648 | -0.0492 |
| spy_alpha | 0.0515 | 0.0636 | +0.0122 |
| spy_ir | 0.2319 | 0.2716 | +0.0397 |
| qqq_ir | -0.1942 | -0.1550 | +0.0392 |
| unique_holdings | 156.0000 | 112.0000 | -44.0000 |
| total_return | 1.7988 | 1.9185 | +0.1197 |

**sharpe win rate vs baseline:** 73.8%  (95% CI on paired diff: [-0.1489, +0.2703])


**Follow-up check (exp1):** per-calendar-year breakdown (30-seed baseline average
vs the single deterministic candidate run) shows the edge is NOT driven by one
year -- candidate wins 2021 (+4.4pp), 2023 (+7.7pp), 2026 YTD (+16.6pp), loses
2020 (-4.5pp), 2022 (-6.3pp), 2024 (-5.0pp), 2025 (-4.5pp). Net positive but
genuinely mixed, and the paired-diff 95% CI on Sharpe still crosses zero
([-0.149, +0.270]). **Verdict: inconclusive, not promoted** -- promising
direction (74% win rate, alpha +0.051->+0.064) but doesn't clear the "CI must
not include a materially negative result" bar yet. Kept as a candidate to
revisit if a refinement tightens it.

**Failure-mode dig (motivating exp2):** looked at the 15 worst score==100-pool
forward returns during 2022 (candidate's worst year) with their 8 parent
scores attached. Momentum is the only parent with a clearly negative
correlation to forward return that year (corr=-0.155 vs value +0.007,
quality -0.031, growth -0.038) -- a momentum-crash signature: several of the
worst offenders (MO, COP, WY, AMAT, GOOGL, CE) carried momentum scores 65-95
while their quality scores sat in the 35-55 range. Hypothesis: penalize
momentum running far ahead of quality, rather than just avoiding any single
weak parent (exp1's rule) -- a name can pass exp1's "no glaring weak spot"
filter with quality=53 and still be a momentum-crash candidate.

## exp2_momentum_overextension_penalty_top5
**Hypothesis:** Penalize names whose momentum score runs far ahead of their quality score (momentum-crash risk) among the score==100 pool; rank by (momentum-quality) ascending, take the smallest-gap top 5.
**Params:** {}  |  deterministic=True  |  n_sims=500

| metric | baseline median | candidate median | paired diff (median) |
|---|---|---|---|
| cagr | 0.1669 | 0.1800 | +0.0131 |
| sharpe | 0.9715 | 0.9424 | -0.0291 |
| sortino | 0.9516 | 0.9791 | +0.0275 |
| max_dd | -0.2195 | -0.2089 | +0.0105 |
| avg_turnover | 0.4375 | 0.4287 | -0.0088 |
| spy_beta | 0.8139 | 0.8786 | +0.0647 |
| spy_alpha | 0.0515 | 0.0564 | +0.0049 |
| spy_ir | 0.2319 | 0.3277 | +0.0958 |
| qqq_ir | -0.1942 | -0.0934 | +0.1008 |
| unique_holdings | 156.0000 | 130.0000 | -26.0000 |
| total_return | 1.7988 | 2.0144 | +0.2156 |

**sharpe win rate vs baseline:** 42.4%  (95% CI on paired diff: [-0.2429, +0.1762])


**exp2 verdict: REJECTED.** Win rate only 42.4% (<60% bar), median Sharpe
slightly worse (0.942 vs 0.972 baseline) and CI [-0.243, +0.176] straddles
zero. Interesting tension though: CAGR (+1.3pp), alpha (0.056 vs 0.051), and
max_dd (-0.209 vs -0.219, better) all nominally improved -- the Sharpe/win-rate
loss looks like a volatility story, not a return story. Ranking purely on
(momentum-quality) gap likely narrows the effective candidate set to a
recurring handful of low-momentum names, adding idiosyncratic concentration
risk that a single min-parent-score or averaged-score rule wouldn't. Restored
champion (random baseline) -- no change to production config.

**Next hypothesis (exp3):** average across all 8 parent scores (not just
min, not just a momentum/quality gap) -- a broader, more diversified quality
bar per the pre-registered experiment list ("rank by average parent score"),
which should keep exp1's "no glaring weakness" intuition without the
narrow-concentration failure mode exp2 hit.

## exp3_avg_parent_score_top5
**Hypothesis:** Rank the score==100 pool by the simple average of all 8 V4 parent scores (broader diversification than exp1 min-rule, less narrowing than exp2 momentum/quality gap), take top 5.
**Params:** {}  |  deterministic=True  |  n_sims=500

| metric | baseline median | candidate median | paired diff (median) |
|---|---|---|---|
| cagr | 0.1669 | 0.1705 | +0.0035 |
| sharpe | 0.9715 | 1.0217 | +0.0502 |
| sortino | 0.9516 | 1.0528 | +0.1012 |
| max_dd | -0.2195 | -0.2128 | +0.0067 |
| avg_turnover | 0.4375 | 0.3837 | -0.0538 |
| spy_beta | 0.8139 | 0.7768 | -0.0371 |
| spy_alpha | 0.0515 | 0.0582 | +0.0067 |
| spy_ir | 0.2319 | 0.2507 | +0.0187 |
| qqq_ir | -0.1942 | -0.1850 | +0.0092 |
| unique_holdings | 156.0000 | 116.0000 | -40.0000 |
| total_return | 1.7988 | 1.8556 | +0.0568 |

**sharpe win rate vs baseline:** 69.2%  (95% CI on paired diff: [-0.1636, +0.2555])


**exp3 verdict: PROMISING, clears the win-rate bar but flagged, not yet
promoted.** Win rate 69.2% (>60% bar), median Sharpe 1.022 vs baseline 0.972,
alpha 0.058 vs 0.051, max_dd improved (-0.213 vs -0.219), CAGR +0.4pp. CI on
paired Sharpe diff still touches zero ([-0.164, +0.255]).

Per-year check (30-seed baseline average vs single deterministic candidate):
wins 2021 (+3.0pp), 2025 (+8.5pp), 2026 YTD (+3.9pp), roughly flat 2020/2023/
2024, **loses 2022 (-5.8pp)**. Not "driven by one great year" (it wins in
4-5 of 7), but there is now a THIRD deterministic parent-score-based
candidate (min-rule, momentum/quality-gap rule, average rule) that all show
the same recurring weak spot: 2022 specifically. Working theory: 2022 was a
regime where re-ranking within the composite==100 tied pool by ANY
parent-score-based rule tended to concentrate on names that were more
exposed to the momentum-crash/rate-shock dynamic that year, while the random
baseline's incidental diversification across the tied pool acted as an
accidental hedge. This is now a specific, falsifiable next hypothesis rather
than a vague caveat -- worth testing directly (e.g., a config that keeps
some diversification/randomness alongside the parent-score tilt, or checks
whether a volatility/beta filter specifically fixes the 2022 gap) before
treating exp3 as a real promotion candidate.

**Status: none of exp1/exp2/exp3 promoted to production.** All fail the "no
materially negative result in the CI" bar, and exp3 additionally has a
real, reproducible 2022 weakness worth addressing first. Champion (random
baseline) unchanged.

## exp4_avg_parent_score_weighted_random
**Hypothesis:** Keep randomness (diversification, the suspected 2022 hedge) but tilt the sampling weights toward higher average-parent-score names instead of hard top-5 ranking. Weighted sample w/o replacement, weight=max(avg_parent_score,1).
**Params:** {}  |  deterministic=False  |  n_sims=500

| metric | baseline median | candidate median | paired diff (median) |
|---|---|---|---|
| cagr | 0.1669 | 0.1653 | -0.0016 |
| sharpe | 0.9715 | 0.9692 | -0.0023 |
| sortino | 0.9516 | 0.9535 | +0.0019 |
| max_dd | -0.2195 | -0.2216 | -0.0022 |
| avg_turnover | 0.4375 | 0.4350 | -0.0025 |
| spy_beta | 0.8139 | 0.8079 | -0.0060 |
| spy_alpha | 0.0515 | 0.0514 | -0.0000 |
| spy_ir | 0.2319 | 0.2196 | -0.0123 |
| qqq_ir | -0.1942 | -0.1976 | -0.0034 |
| unique_holdings | 156.0000 | 154.0000 | -2.0000 |
| total_return | 1.7988 | 1.7725 | -0.0262 |

**sharpe win rate vs baseline:** 51.0%  (95% CI on paired diff: [-0.3207, +0.2740])


**exp4 verdict: REJECTED.** Win rate 51.0% -- indistinguishable from a coin
flip -- and every metric (Sharpe 0.969 vs 0.972, CAGR 0.165 vs 0.167, alpha
0.051 vs 0.051, max_dd -0.222 vs -0.219) landed within noise of the pure
random baseline. Softening the parent-score tilt into sampling weights
(rather than hard top-5 ranking) fully washed out exp3's edge -- it did NOT
retain most of the edge while fixing 2022; it retained essentially none of
either. This is informative: the edge in exp1/exp3 seems to require the full
deterministic tilt to show up at all, and that same full-strength tilt is
what produces the 2022 vulnerability. A "soft compromise" is not a free
lunch here -- restored champion (random baseline), no promotion.

**Session status after 4 iterations:** champion unchanged (random baseline).
Leading candidate remains exp3 (avg-parent-score top5, 69.2% win rate,
Sharpe 1.02 vs 0.97, alpha +0.058 vs +0.051) but not promoted -- CI still
touches zero and it shares the recurring 2022 weakness with every other
deterministic tilt tried. exp2 (momentum/quality gap) and exp4 (weighted
random) both rejected outright. See output/loop_research/results.json for
full per-metric records of all four experiments.

---

## Promotion rule revised (2026-09-09, user request)

The original rule vetoed promotion whenever the paired-diff 95% CI touched
negative territory. Problem: the Monte Carlo here only resamples *which
tied name gets drawn* on top of one fixed 7-year price path with exactly
one regime-shock year (2022) -- so that CI conflates two different
uncertainties: (a) selection-draw noise, which shrinks with more sims, and
(b) single-episode regime uncertainty, which cannot shrink no matter how
many sims run, because there is only one 2022 in the data. A hard CI veto
was effectively demanding certainty about a second crash episode that
doesn't exist in this dataset.

**Revised promotion rule** -- candidate replaces champion if:
1. Median OOS Sharpe improves meaningfully.
2. Wins >=60% of paired seeds.
3. The net edge isn't produced entirely by one outlier year (checked via:
   remove the single best year and see whether the remaining years are
   still roughly neutral-to-positive, not a large reversal).
4. CAGR or alpha improves without materially worse max drawdown (moving
   together, not traded off against each other).
5. Costs already included (T+1 lag + 10bps/side baked into every sim).
6. Selection logic stays simple and economically defensible.

The CI is still computed and reported every time, but is no longer a hard
veto -- it's reported as a caveat about regime-uncertainty, not grounds for
automatic rejection.

### Re-evaluation under the revised rule

Per-year paired diffs (candidate - baseline, single deterministic run vs a
30-seed baseline average -- output/loop_research/yearly_breakdown.json):

| year | exp1 diff | exp3 diff |
|---|---|---|
| 2020 | -0.045 | -0.019 |
| 2021 | +0.044 | +0.030 |
| 2022 | -0.063 | -0.058 |
| 2023 | +0.077 | -0.007 |
| 2024 | -0.050 | -0.010 |
| 2025 | -0.045 | +0.085 |
| 2026 (partial) | +0.166 | +0.039 |
| **sum (all years)** | **+0.084** | **+0.060** |
| **sum excl. single best year** | **-0.082** (best=2026) | **-0.025** (best=2025) |

**exp1:** win rate 73.8% (criterion 2: pass), Sharpe/alpha up (criterion 1:
pass), but removing its single best year (2026, only a partial year of
data) flips the net effect to a clear negative (-0.082) -- **criterion 3
fails outright**. exp1's aggregate edge is materially carried by one
partial year. **Verdict: still not promoted**, even under the relaxed rule.

**exp3:** win rate 69.2% (pass), Sharpe 1.02 vs 0.97 / alpha +0.058 vs
+0.051 / max_dd -0.213 vs -0.219, all moving together favorably (criterion
1 & 4: pass). Removing its single best year (2025) brings the remaining six
years to roughly flat (-0.025, much smaller than exp1's -0.082) rather than
a sharp reversal -- **criterion 3: passes, but marginally, not cleanly**.
Logic (average of 8 already-production parent scores) stays simple
(criterion 6: pass). Costs included (criterion 5: pass).

**Verdict: exp3 (avg_parent_score_top5) is PROMOTED as the new champion**
for this stock-selection question under the revised rule -- it clears every
criterion, though criteria 3's "not driven by one year" check is a
marginal pass, not a clean one (2025 does supply most of the net edge, and
2022 remains a known, explained weak spot). This should be treated as
"best idea found so far, adopt with the caveats stated," not "definitively
proven." Champion selector: `research.loop_research.candidates.avg_parent_score_top5`.
Random baseline retained as the fallback/reference distribution for all
future experiments.

## exp5_top5_by_insider
**Hypothesis:** Batch screen (waveC) found this by far the single strongest deterministic tie-break: among the score==100 pool, prefer the single highest insider parent score (no_selling_flag/cluster_buyers/sell_pressure_inv), ignore all other parents.
**Params:** {}  |  deterministic=True  |  n_sims=500

| metric | baseline median | candidate median | paired diff (median) |
|---|---|---|---|
| cagr | 0.1669 | 0.2179 | +0.0510 |
| sharpe | 0.9715 | 1.2227 | +0.2512 |
| sortino | 0.9516 | 1.1897 | +0.2381 |
| max_dd | -0.2195 | -0.1772 | +0.0422 |
| avg_turnover | 0.4375 | 0.4225 | -0.0150 |
| spy_beta | 0.8139 | 0.7995 | -0.0144 |
| spy_alpha | 0.0515 | 0.0962 | +0.0447 |
| spy_ir | 0.2319 | 0.6417 | +0.4098 |
| qqq_ir | -0.1942 | 0.0765 | +0.2707 |
| unique_holdings | 156.0000 | 140.0000 | -16.0000 |
| total_return | 1.7988 | 2.7219 | +0.9231 |

**sharpe win rate vs baseline:** 98.4%  (95% CI on paired diff: [+0.0374, +0.4565])


---

## 93-candidate batch (2026-09-09, user request: "100 more strategies")

Ran a 93-selector batch in named waves (A: parent-aggregation variants,
B: single-parent floor filters, C: single-parent tie-breaks, D/J: thematic
parent combos, E: momentum-overextension variants, F: score trajectory,
G: sector diversification, H/K: risk filters + refinements, I: leave-one-out
averages, K: negative controls/triples/momentum-cap refinements,
L: systematic C(8,2) pair grid) -- full list + rationale in
research/loop_research/batch_candidates.py, every result (win or lose) in
output/loop_research/batch_results.json. Screened at 100 sims (cheap pass);
leaders confirmed at full 500 sims before any promotion decision, per the
tiered-compute plan flagged to the user up front.

### Headline patterns across all 93

- **Insider is by far the strongest single parent.** top5_by_insider:
  98% win rate (100-sim screen), confirmed at 96.4pp Sharpe edge in a
  500-sim/per-year follow-up (see exp5 below). Every combo containing
  insider scored well: insider+quality (96%), quality+institutional+insider
  triple (83%), insider+revisions (85%), floor_insider>=60 (88%).
- **Revisions is the second-strongest.** top5_by_revisions (94%),
  revisions+institutional (94%), revisions+quality (85%).
- **Growth is a consistently BAD signal within this already-elite tied
  pool** -- every combo containing growth screened poorly: top5_by_growth
  alone (1%), momentum+growth (1%), growth+short (4%), value+growth GARP
  (6%), momentum+growth+revisions triple (5%), growth+insider (11%). Checked
  the sub-factor definitions (factors/parent_selection_v4.py) for a sign
  error -- none found (grw_earnings_surprise/grw_earnings_yoy/
  grw_operating_income_growth are all intuitively signed, same construction
  as every other parent). Read as a real, if small-sample (81 monthly
  dates, one historical path), finding: growth-tilting WITHIN a pool that's
  already screened to the top of its sector on the full composite adds
  idiosyncratic earnings-miss risk without much extra edge.
- **Momentum-heavy combos are weak** (momentum+value 13%, momentum+
  institutional 16%, momentum+growth 1%), consistent with exp1-exp4's
  2022 momentum-crash finding.
- **Excluding high vol/beta names from the tied pool is dramatically
  worse, not better** (waveH_exclude_high_vol 0%, waveK exclude_high_vol_
  median 0%, exclude_high_beta 1%, exclude_high_beta_1p0 20%) -- WORSE than
  the deliberate negative controls (worst_avg_parent 20%, worst_min_parent
  46%). Read as regime-specific: 2020-2026 was dominated by a handful of
  high-beta/high-vol winners (2023-2024 mega-cap/AI rally); systematically
  screening them out of a small ~10-15-name tied pool removed most of the
  pool's actual alpha-bearing names. This is a caution about applying a
  defensive vol/beta screen in a strongly trending market, not a universal
  "vol filters don't work" conclusion -- would need a different (pre-2020 or
  down-market-only) window to test that claim properly.
- **max_parent_score (rank by the single BEST parent) does poorly (5%)** --
  consistent with exp1's earlier finding that "no weak spot" (min-based)
  beats "has one standout strength" (max-based).
- **Negative controls behaved as expected** (worst_avg_parent 20%,
  worst_min_parent 46%, both <50%), i.e. below baseline -- a basic sanity
  check that the harness/metric direction is correct.

### exp5: top5_by_insider -- PROMOTED, cleanly

Batch's single strongest screen, confirmed at full 500 sims (identical
harness/metrics as exp1-4):

| metric | baseline (median) | candidate (median) |
|---|---|---|
| sharpe | 0.972 | **1.223** |
| cagr | 0.167 | **0.218** |
| alpha vs SPY | 0.051 | **0.096** |
| max_dd | -0.219 | **-0.177** (better) |
| beta vs SPY | 0.814 | 0.799 |

Win rate **98.4%**, 95% CI on paired Sharpe diff **[+0.037, +0.457] --
entirely positive, clears even the original strict rule**, not just the
revised one. Per-year (30-seed baseline avg vs single deterministic run):
wins 6 of 7 years (2021, **2022** +6.9pp, 2023, 2024, 2025, 2026), only
loses 2020 (-2.2pp, small). Critically this is the FIRST candidate across
all 98 tried (exp1-4 + 93-batch) that wins 2022 -- the regime-shock year
every parent-score-based candidate before it lost. Removing its single best
year (2024, +12.4pp) still leaves +24.1pp across the other six years --
nowhere near a reversal (contrast exp1's -8.2pp reversal and exp3's -2.5pp
marginal one).

**Verdict: exp5 (top5_by_insider) is the new champion**, replacing exp3.
Selector: `research.loop_research.candidates.top5_by_insider` (added, wraps
`batch_candidates.make_single_parent_top5_selector("insider")`). This is
still a research-loop finding on 81 monthly dates / one historical path --
worth a mental caveat that "insider dominates" could partly reflect this
specific window (insider cluster-buying signals are known to work best
around specific market-stress/recovery episodes, several of which fall in
this sample), not a law of nature. But it is the cleanest, best-supported
result the loop has produced.

---

## Perturbation Monte Carlo on exp5 (2026-09-09, user request)

Valid concern raised: `top5_by_insider` is fully deterministic (one fixed
140-ticker history), so the earlier "500 sims, 98.4% win rate" only ever
compared that ONE fixed path against 500 different RANDOM-baseline paths --
it gave the baseline variance, not the candidate any variance of its own.
That statistic can't distinguish "insider score cleanly separates good
picks" from "the actual top-ranked names were a coin-flip-close tie away
from a much worse book, and we got lucky."

Fix: add Gaussian noise (scaled to that date's tied-pool insider-score
cross-sectional std) to the insider score before ranking, independently per
sim, reselect top-5 under the perturbed ranking, at several noise
multiples (research/loop_research/perturbation.py). Also tracked Jaccard
overlap with the exact deterministic top-5 as a diagnostic.

| noise (x pool std) | median Sharpe | win rate vs random baseline | overlap w/ exact top5 |
|---|---|---|---|
| 0 (exact, deterministic) | 1.223 | 98.4% | 100% |
| 0.25x | 1.202 | 96.0% | 74% |
| 0.5x  | 1.177 | 94.8% | 69% |
| 1.0x  | 1.126 | 87.0% | 59% |
| 2.0x  | 1.055 | 74.2% | 53% |
| 4.0x  | 1.011 | 63.0% | 43% |

**Verdict: the edge decays gracefully, does not collapse to a 50% coin flip
even at large noise.** If the win had been riding on a razor-thin near-tie,
win rate would have crashed toward 50% at small noise already (0.25x
already scrambles ~26% of the picks). Instead, even at 1x noise (as large
as the pool's ENTIRE natural insider-score spread that month) win rate is
still 87%; even at 4x noise, where picks are essentially decorrelated from
the exact top-5 (43% overlap), it's still beating baseline 63% of the time.
Interpretation: the effect isn't about these 5 exact tickers being special
-- it's that a noisy, imprecise draw from near the top of the insider
ranking still beats an indiscriminate random draw from the full tied pool.
That's a materially stronger and more believable claim than the original
deterministic-vs-random comparison alone, and meaningfully lowers (without
eliminating) the "this is just one lucky historical path" concern. Full
numbers: output/loop_research/insider_perturbation.json.

Champion status unchanged (exp5, top5_by_insider) -- this analysis
strengthens confidence in it rather than changing the decision.

---

## Pool-robustness test: top100 vs next11 (2026-09-09, user request)

Skeptical check requested: the composite==100 tied pool is already the
best-of-sector by construction -- does "pick the best insider names" still
beat random selection on a LESS pre-screened pool? Built a second pool each
date -- "next11" = the 11 highest-scored names strictly BELOW 100 (same
typical size as the top100 tied group, for a fair comparison) -- and reran
both the 500-sim random-5 Monte Carlo and the deterministic top5-by-insider
rule on it. Chart: output/loop_research/pool_comparison.png. Full curves +
stats: output/loop_research/pool_comparison.json.

| pool | selector | CAGR | Sharpe | Sortino | max_dd | beta | alpha | IR |
|---|---|---|---|---|---|---|---|---|
| top100 | random (median) | 0.167 | 0.972 | 0.952 | -0.219 | 0.814 | 0.051 | 0.232 |
| top100 | insider top5 | **0.218** | **1.223** | **1.190** | **-0.177** | 0.799 | **0.096** | **0.642** |
| next11 | random (median) | 0.158 | 0.924 | 0.934 | -0.250 | 0.832 | 0.040 | 0.166 |
| next11 | insider top5 | 0.178 | 0.957 | 0.949 | -0.251 | 0.909 | 0.049 | 0.367 |

Insider-beats-random-sim win rate: **98.4% on top100** vs **62.6% on
next11**.

**Honest read: the effect direction survives (insider still beats random on
BOTH pools, and IR roughly doubles on both), but the effect SIZE shrinks a
lot outside the elite tied pool.** On next11 the Sharpe edge is +0.033
(0.957 vs 0.924) vs +0.251 on top100 (1.223 vs 0.972) -- about 7x smaller.
Max drawdown, which clearly improved on top100 (-0.177 vs -0.219), is
actually flat-to-slightly-worse on next11 (-0.251 vs -0.250). This is
consistent with a real interpretation, not just "insider works everywhere":
insider cluster-buying is most informative as a TIE-BREAKER among names
that have already cleared every other quality bar (the top100 pool, by
construction, already passed the full composite screen) -- among a broader,
less-screened set it still helps a little, but doesn't compensate for
weaker overall fundamentals the way it can among near-identical elite
names. Confirms exp5 is a genuine but SCOPE-LIMITED finding: it works best
specifically as a tie-breaker within an already-elite pool, not as a
general-purpose insider-momentum strategy.

---

## Decile-spread validation of the composite score itself (2026-09-09, user request)

Question: does the composite score carry real information at all, beyond
the already-tested top100/next11 comparison? Ranked the FULL universe by
composite score each date into 10 equal-sized rank deciles (not raw
score-value bins, since the score is a per-sector percentile with a large
tied mass at 100 -- equal-sized rank deciles avoid decile 1 being
artificially huge). Held a random 10-name book per decile, same staggered-
sleeve mechanics as everywhere else (4 sleeves, 4-month hold, full
turnover), 500 Monte Carlo sims per decile. Chart:
output/loop_research/decile_spread.png. Full numbers:
output/loop_research/decile_spread.json.

| decile | CAGR | Sharpe | Sortino | max_dd | beta | alpha | IR | % beating SPY |
|---|---|---|---|---|---|---|---|---|
| 1 (highest score) | 0.165 | **0.997** | 0.979 | -0.232 | 0.827 | **0.047** | 0.255 | **93.2%** |
| 2 | 0.152 | 0.939 | 0.954 | -0.237 | 0.817 | 0.036 | 0.113 | 82.4% |
| 3 | 0.122 | 0.763 | 0.757 | -0.250 | 0.837 | 0.007 | -0.204 | 17.0% |
| 4 | 0.113 | 0.706 | 0.676 | -0.264 | 0.844 | -0.002 | -0.289 | 5.4% |
| 5 | 0.122 | 0.743 | 0.759 | -0.258 | 0.852 | 0.005 | -0.176 | 16.6% |
| 6 | 0.124 | 0.774 | 0.778 | -0.253 | 0.812 | 0.012 | -0.166 | 19.4% |
| 7 | 0.103 | 0.635 | 0.652 | -0.267 | 0.847 | -0.010 | -0.334 | 2.6% |
| 8 | 0.119 | 0.705 | 0.752 | -0.230 | 0.842 | 0.005 | -0.172 | 14.6% |
| 9 | 0.106 | 0.620 | 0.677 | -0.269 | 0.870 | -0.011 | -0.263 | 3.2% |
| 10 (lowest score) | 0.103 | **0.569** | 0.607 | **-0.301** | 0.954 | **-0.020** | -0.216 | 4.2% |

SPY over this window: Sharpe 0.810, CAGR 0.139 (from the earlier saved
featured-config run) -- shown as a reference line on the chart.

**Read: the score has real signal at the extremes, not a clean monotonic
staircase across all 10 deciles.** Decile 1 and 2 (top 20%) clearly and
consistently separate from everything else -- both beat SPY on the large
majority of sims (93%/82%), both have positive alpha, decile 1 has the
best Sharpe by a wide margin. Decile 10 (bottom 10%) is clearly the worst
on every metric -- lowest Sharpe, most negative alpha, worst (most
negative) max drawdown, highest beta (0.954, closest to unhedged market
exposure), lowest SPY-beat rate. But deciles 3-9 (the "middle 70%") are
compressed and noisy, not a clean descending staircase -- decile 6
(Sharpe 0.774) actually beats deciles 3, 4, 5, 7, 8, 9, non-monotonically.
This is a common and honest pattern for a real-but-imperfect signal: strong
discrimination between "clearly good" (top 20%), "clearly bad" (bottom
10%), and everything in between being roughly interchangeable noise. It
validates that the composite score is doing real work at the top and
bottom, while tempering any claim that it finely ranks the middle of the
universe.

## Paired insider-edge test (2026-09-10, user request)

**Question:** is the insider-tie-break edge (exp5) real, or is it just a lucky
near-tie in the composite==100 pool it was found on? User's proposed design:
at every review date, take the top-decile pool (same equal-sized rank
decile-1 definition as the decile-spread test above) MINUS the composite==100
tied names -- the "good but not elite" ~39-40-name pool. Draw a random 10
from that pool, split top-5 / bottom-5 by insider parent score, run BOTH legs
forward as full staggered-sleeve books (4 sleeves, 4-month hold, full
turnover), 500 sims. Critically, both legs share the exact same random draw
of 10 at every date across the whole path (same rng seed, identical rng call
order, diverging only in the deterministic top-5/bottom-5 split) -- a paired
design that cancels out market-timing/regime noise and isolates the insider
split itself. Baseline for "no edge" is a per-sim diff of 0.
Code: research/loop_research/insider_edge_paired.py. Full numbers:
output/loop_research/insider_edge_paired.json. Chart:
output/loop_research/insider_edge_paired.png.

| metric | insider-5 (median) | other-5 (median) | median diff | win rate (diff>0) |
|---|---|---|---|---|
| CAGR | 0.174 | 0.155 | **+0.021** | 69.4% |
| Sharpe | 0.982 | 0.943 | **+0.051** | 58.2% |
| Alpha (vs SPY) | 0.050 | 0.043 | **+0.009** | 60.0% |

**Read: the edge survives outside the elite pool but is much smaller and
noisier than in the composite==100 pool.** The sign is right on all three
metrics and the win rates are meaningfully above 50% (especially CAGR at
69%), so this isn't nothing -- but a Sharpe win rate of 58% (vs. exp5's
98.4% in the ==100 pool) means roughly 2 in 5 sims still see the "other 5"
(the names insider score did NOT pick) come out ahead on risk-adjusted
terms. Consistent with the earlier top100-vs-next11 pool-comparison finding:
insider's discriminating power is real but concentrated where the rest of
the composite has already done most of the filtering (the ==100 tied pool).
Applied to a broader "good but not elite" pool, it's a genuine, positive,
modest tilt -- not a strong standalone signal. Net effect on the champion:
no change -- exp5 remains the champion for its stated scope (a tie-breaker
within the ==100 pool specifically); this test adds an honest boundary on
how far that finding travels.

**Added 2026-09-10: p-value on the convergence chart.** One-sample t-test of
the per-sim (insider-other) diff against 0, annotated on
output/loop_research/insider_edge_paired_convergence.png:
CAGR mean=+0.0190 t=11.74 p<0.0001; Sharpe mean=+0.0370 t=4.46 p<0.0001;
alpha mean=+0.0073 t=5.19 p<0.0001. CAVEAT: this p-value assumes the 500
sims are independent, but they all replay the same 2020-2026 price history
with overlapping review dates -- effective independent sample size is much
smaller than 500. Read it as "the sign is stable across random draws," not
as a literal frequentist significance level -- the dominant remaining
uncertainty is having only one realized market history (same caveat as
exp5), which resampling the draw cannot address. A block-bootstrap-over-time
test would be the honest way to get a real significance number; not yet run.

**CORRECTION 2026-09-10: per-period breakdown reverses the "significant" read
above.** The 500-sim t-test (p<0.0001) only resamples which random names get
compared -- every sim replays the same 2020-2026 history, so it can't detect
whether the edge is broad-based across time or a one-regime artifact. Fix:
average the per-sim return diff ACROSS sims first (kills the selection-draw
noise, keeps the sim-noise-free monthly time series), then test across the
80 actual calendar months instead of 500 duplicated sims.
Code: same file, `_period_breakdown()`/`_plot_period()`. Chart:
output/loop_research/insider_edge_paired_period.png. Data:
output/loop_research/insider_edge_paired_period.json.

Result: **t=0.81, p=0.42, 51% of months positive -- not distinguishable from
noise.** Per-year sums: 2020 +0.054, 2021 +0.069, 2022 +0.008, 2023 +0.002,
2024 -0.013, 2025 +0.011, 2026 -0.006. 2020+2021 alone account for ~98% of
the entire 7-year cumulative sum (+0.123 of +0.126 total); 2022-2026
combined contribute essentially nothing, and 2 of the last 3 years are net
negative. The cumulative-diff curve is flat-to-declining since mid-2022.

**Revised conclusion: in this broader (decile-1-minus-100) pool, the
insider-tie-break edge looks like a 2020-2021-regime artifact (COVID-
recovery / speculative-momentum era), not a durable signal.** This does NOT
directly overturn exp5 (which operates in the much more elite ==100 pool
and separately passed a "remove best year" robustness check), but it is a
strong caution against generalizing exp5's finding beyond that elite pool,
and a good illustration of why sim-count alone (500, or 5000, or 50000)
cannot fix a single-historical-path problem -- only more independent
calendar time can. Recommended follow-up (not yet run): rerun this same
per-period breakdown on the ORIGINAL exp5 ==100-pool result to check whether
it has the same 2020-2021 concentration.

## Champion (exp5) per-period breakdown, same rigor as the paired-pool test (2026-09-10)

Applied the identical per-calendar-time method that debunked the broader
decile-1-minus-100 finding to the actual champion: top5_by_insider
(candidates.py) within its real composite==100 pool, vs the random baseline,
500 sims averaged per calendar date to cancel the baseline's own resampling
noise (deterministic champion path only has one real path by construction).
IC computed vs 1M forward returns restricted to the same ==100 pool each
date (had to lower period_ic's min_names floor from 20 to 8 -- the ==100
pool is only ~11-12 names, so the default floor silently returned None for
every period on the first pass; fixed before drawing conclusions).
Code: research/loop_research/champion_period_breakdown.py. Chart:
output/loop_research/champion_period_breakdown.png. Table:
output/loop_research/champion_period_breakdown.csv.

| year | champion vs random (sum) | IC that year (==100 pool) | trailing-5yr IC |
|---|---|---|---|
| 2020 | -0.019 | 0.012 | 0.012 |
| 2021 | +0.047 | 0.031 | 0.022 |
| 2022 | +0.059 | -0.049 | -0.003 |
| 2023 | +0.029 | 0.039 | 0.008 |
| 2024 | +0.106 | 0.086 | 0.023 |
| 2025 | +0.070 | 0.032 | 0.027 |
| 2026 | -0.002 | 0.087 | 0.034 |

Overall: t=2.00, p=0.049, 54% of months positive, 5/7 years net positive.

**Read: materially different pattern from the broader-pool finding, and
healthier, but only marginal.** Unlike the decile-1-minus-100 test, the
edge here is NOT concentrated in 2020-2021 (2020 is actually slightly
negative; the single best year is 2024). 5 of 7 years are net positive,
which is a real breadth-across-time result the broader pool did not have.
BUT p=0.049 sits right at the conventional significance line -- marginal,
not strong -- and still rests on one realized 2020-2026 market history (more
independent calendar months than the 500-sim resampling gave us, but still
not a true out-of-sample test). IC-by-year still does not track the edge
(2022: negative IC, good edge year; 2026: best IC, flat edge year), so
"insider IC predicts when the tie-break works" is not supported here either.
**Verdict: exp5 cautiously survives this scrutiny (broad-based, not a
regime fluke like the broader pool) but the significance is marginal, not
strong -- champion status kept, but confidence in it should be described as
"probably a small real effect," not "proven."**

## Block bootstrap over calendar time (2026-09-10) -- the real significance number

The per-period t-test (t=2.00, p=0.049) assumed the 77 monthly diff
observations were independent, but the staggered-sleeve book holds names 4
months with a monthly-staggered review cadence, so adjacent months share
holdings and returns are serially correlated -- a plain t-test on
autocorrelated data can overstate significance. Moving-block bootstrap
(resample contiguous blocks of months with replacement, preserving local
autocorrelation, 10,000 resamples) at block lengths 1/3/4/6/12 months.
Code: research/loop_research/block_bootstrap.py. Table:
output/loop_research/block_bootstrap.csv. Chart:
output/loop_research/block_bootstrap.png.

| block length (months) | 95% CI | p-value (two-sided) |
|---|---|---|
| 1 (iid) | [0.0001, 0.0071] | 0.047 |
| 3 | [0.0006, 0.0069] | 0.014 |
| 4 (= sleeve hold length, principled default) | [0.0008, 0.0068] | 0.014 |
| 6 | [0.0011, 0.0067] | 0.006 |
| 12 | [0.0019, 0.0071] | <0.0001 |

**Result: the champion's edge holds up, and slightly more confidently than
the naive t-test suggested (p=0.014 at L=4 vs the t-test's p=0.049).**
Checked why (expected autocorrelation to widen the CI, not narrow it): the
monthly diff series has mild NEGATIVE autocorrelation (lag-1 ~=-0.14, lag-4
~=-0.12), plausibly because only 1 of 4 sleeves refreshes in most months, so
a weak month partially mean-reverts the next -- meaning the naive t-test was
if anything slightly conservative here, not anti-conservative. This is the
most rigorous significance estimate produced for the champion so far, and it
confirms rather than overturns the "cautiously survives" verdict from the
per-period breakdown above. Standing caveat unchanged: this bootstraps
variation WITHIN one realized 2020-2026 history -- it establishes the edge
is not an artifact of ignoring serial correlation in that history, not that
it is guaranteed to repeat in a different regime.

## Tested and rejected: dynamic "route to highest trailing-5yr IC" tie-break (2026-09-10/11)

Follow-up to the per-parent correlation study: momentum/institutional/growth
showed a weak (not significant, n=6) positive correlation between trailing
IC and paired-diff edge, so tested whether dynamically tie-breaking the
composite==100 pool by whichever of the 8 parents currently has the
highest trailing-5-year IC (no look-ahead: only ICs from review dates
strictly before the current one, expanding window capped at 5y, falls back
to insider before any history exists) beats always using insider (the
champion). Code: research/loop_research/dynamic_best_ic.py. Data/chart:
output/loop_research/dynamic_best_ic.json / .png.

**Result: rejected. Worse than both the random baseline and the champion.**
vs random baseline: t=0.17, p=0.87 (indistinguishable from noise). vs
champion (top5_by_insider): t=-1.21, p=0.23, and behind the champion in 4 of
6 years (2021-2024 all negative, only ahead in 2020/2025/2026).

**Why:** winner-frequency breakdown across ~81 review dates: quality won
the "highest trailing IC" contest 52 times (64%), institutional 18,
insider only 6, value 4, revisions 1. Quality is one of the WEAKEST
standalone tie-break parents (p=0.90 in the parent-comparison test) despite
frequently having the best trailing IC -- IC (general forecasting power
across the whole score distribution) and tie-break power (differentiating
among an already-elite, near-perfect-score pool) are different properties
and don't line up. Insider, the one parent with a real marginal tie-break
edge, rarely has the single highest trailing IC at any given moment, so a
"pick the current IC leader" rule mostly avoids using it. Confirms and
extends the earlier finding that IC-by-year does not predict when a
parent's tie-break edge shows up. Champion unchanged: top5_by_insider
(exp5) remains the best-supported choice.

---

## Top-20-pool track (2026-09-11, user request): new baseline + codified promotion rule

Opened a second, parallel track using a different pool definition: instead
of the composite==100 tied pool (which can be small, ~10-15 names, and is
already screened to a near-perfect score), rank every date by composite
score, take the top 20, and draw 5 from that broader pool. New baseline:
`research.loop_research.harness.random_top20_selector` (uniform random,
same 4-sleeve/4-month-hold/T+1/10bps mechanics as everywhere else).

Also codified, at the user's request, a reusable promotion rule as one
function -- `research.loop_research.promotion.evaluate_promotion` -- rather
than deciding each candidate ad hoc. Final negotiated rule (see
promotion.py's module docstring for the authoritative version):
**REJECT** if median Sharpe improvement <0.025, sim win rate <50%, average
monthly excess return (sim-averaged, real calendar months) <0, or removing
the single best year makes the cumulative excess return negative.
**PROMOTE** if median Sharpe improvement >=0.05 AND win rate >=60% AND
avg monthly excess >0 AND wins >=4/7 calendar years AND strict
leave-one-year-out (for every year removed, both Sharpe-diff and
alpha-diff vs baseline stay positive on the remaining six) AND max
drawdown doesn't worsen by more than 4pp AND full-period alpha improves
AND perturbed-ranking win rate stays >=55%. **Otherwise: PROMISING.**
Validated by reproducing known results: `top5_by_insider` -> PROMOTE
(matches champion.md's numbers), `momentum_overextension_penalty_top5` ->
REJECT (matches the historical 42.4% win rate).

Four candidates tested against the top-20-pool random baseline:

### min_subfactor_score_top20 -- REJECTED
Rank the top-20 pool by the MINIMUM across all 24 individual subfactor
scores (finer-grained version of the old min-parent-score idea), take top
k. Result: did not clear the promotion gates (see
output/loop_research/promotion_min_subfactor_score_top20.json for full
numbers) -- rejected, same family of failure as narrow single-metric
floors tried on the ==100 pool.

### avg_top3_parent_score_top20 -- REJECTED (PROMISING at best)
Within the top-20 pool, average each ticker's 3 HIGHEST parent scores (out
of 8) instead of requiring uniform strength across all 8 or penalizing any
one weak spot -- rewards standout strengths. Compact record:
output/loop_research/experiments/avg_top3_parent_score_top20.json. Did not
clear the full 8-gate promote bar.

### dynamic_best_ic_top20_selector -- REJECTED
Re-run of the old ==100-pool "route to whichever parent currently has the
highest trailing-5-year IC LEVEL" idea, adapted to the top-20 pool (no
look-ahead: only ICs strictly before the review date). Winner-frequency
across 81 dates: quality 43, insider 18, revisions 14, growth 6 -- same
failure mode as the ==100-pool version (quality wins the IC-level contest
most often despite being a weak standalone tie-breaker). Result: Sharpe
-0.032 vs baseline, win rate 40%, only 1/7 years positive -- REJECTED,
worse than the random baseline itself.

### dynamic_ic_momentum_top20_selector -- PROMOTED, new champion for this track
**Hypothesis (user's, 2026-09-11):** instead of routing to whichever parent
has the highest trailing IC LEVEL (rejected above), route to whichever
parent's mean IC has JUMPED THE MOST year-over-year -- current-1y mean IC
minus prior-1y mean IC, both computed strictly before the review date, so
the selector rotates across parents based on whose predictive power is
currently IMPROVING rather than sticking to whichever parent has the
highest slow-moving historical average. Falls back to insider before 2
full years of trailing history exist.
Code: `research.loop_research.dynamic_ic_momentum_top20`. Full report:
output/loop_research/promotion_dynamic_ic_momentum_top20_selector.json.
Compact record (all tests consolidated):
output/loop_research/experiments/dynamic_ic_yoy_momentum_top20.json.

| metric | baseline (median) | candidate (median) |
|---|---|---|
| sharpe | 0.959 | **1.231** (+0.272) |
| cagr | 0.163 | **0.213** |
| alpha vs SPY | 0.048 | **0.092** |
| max_dd | -0.214 | **-0.203** (better) |
| beta vs SPY | 0.807 | 0.788 |

Win rate **99%** of 500 paired sims. Avg monthly excess return (sim-
averaged, real calendar months): +0.0035, positive. Strict
leave-one-year-out passes for **all 7 years** (both Sharpe-diff and
alpha-diff stay positive on the remaining six no matter which year is
removed; smallest margin is removing 2024: sharpe_diff +0.203, alpha_diff
+0.030). 4/7 individual years net-positive (2021, 2023, 2024, 2025);
removing the single best year (2024) still leaves +0.136 across the other
six -- not a reversal. Max drawdown improves. Perturbation (noise on the
final within-parent ranking step, option 1): 76.5% win rate at 0.5x noise,
clears the 55% bar. **Clears every gate in the formal rule: PROMOTE.**

**Extra robustness beyond the formal rule (user-requested):**
- Perturbation option 2 (noise added directly to the per-date IC estimates
  driving the winner-parent DECISION, not just the final ranking): 94.5% /
  89.0% / 83.5% win rate at noise_mult 0.25/0.5/1.0x. (An earlier run of
  this test showed a spurious 19.5% collapse at 1.0x noise; traced to an
  `id(rng)`-keyed cache dict silently serving a stale prior-sim result once
  Python recycled a garbage-collected `random.Random` object's memory
  address -- fixed by caching on the rng object's own attribute instead,
  which ties the cache lifetime to that specific object with no id-reuse
  risk. The corrected numbers above supersede the collapse finding, which
  should be disregarded.)
- Perturbation option 3 (user-designed cross-sectional bootstrap: at each
  date, resample the ~20 pool stocks with replacement ONCE and reuse that
  same resampled index for all 8 parents' IC recomputation, preserving the
  real cross-parent correlation from shared stock identity, rather than
  adding independent per-parent noise): win rate **80.5%**, median Sharpe
  1.102 vs baseline 0.959 (200 sims). Same `id(rng)`-caching bug was present
  here too and fixed the same way before this result was trusted.
- Moving-block bootstrap over calendar time (same method as
  block_bootstrap.py, applied to this candidate's diff series --
  output/loop_research/dynamic_ic_yoy_block_bootstrap.csv): at the
  principled L=4 block length, **95% CI [-0.0004, +0.0081], p=0.078** --
  does not cleanly exclude zero at the conventional 95% bar, though it does
  clear p<0.05 at longer block lengths (L=6: 0.048, L=12: 0.005) and the
  point estimate (~+0.0035/mo) is essentially IDENTICAL across every block
  length tested (1/3/4/6/12 months) -- the signature of a real, if
  underpowered, effect rather than noise (noise typically looks unstable as
  block length changes, not flat). Weaker standalone significance than the
  ==100-pool champion's p=0.014 at the same L=4, given only ~7 years of one
  realized market history to test against. Discussed explicitly with the
  user: a conventional p<0.05 bar demands a level of certainty this
  domain's data (one realized historical path, no way to generate
  independent additional calendar time) cannot supply even for a genuinely
  real effect, so this check is treated as a graded confidence signal
  alongside the rest of the battery, not a standalone hard gate -- the
  candidate is promoted on the strength of the full battery (formal 8-gate
  rule + both perturbation designs + leave-one-year-out on every year), with
  the calendar-time significance explicitly flagged as "supportive, not yet
  confirmatory" rather than papered over.

**Verdict: dynamic_ic_momentum_top20_selector is PROMOTED as the new
champion for the top-20-pool track.** `random_top20_selector` remains the
reference/fallback baseline distribution for future top-20-pool
experiments; this selector is now the bar new candidates on this pool must
beat. See champion.md for the consolidated summary.

### worst_week_return_top20 -- REJECTED
**Hypothesis (user, 2026-09-11):** short-term oversold/mean-reversion --
within the top-20 pool, prefer names that fell the most over the trailing
~1 week (5 trading days), instead of any score-based tilt. No look-ahead
(current price vs price 5 trading days earlier, both realized by the review
date). Code: `research.loop_research.candidates.worst_week_return_top20`.
Full report: output/loop_research/promotion_worst_week_return_top20.json.

Result: median Sharpe **-0.044** vs baseline (0.917 vs 0.962), win rate
**35.8%** (loses more than it wins), beta UP +0.11 while alpha DOWN -0.006
(more market risk for less compensation -- the wrong-direction sanity
pattern), leave-one-year-out fails 5 of 7 years, and removing the single
best year (2026 YTD) flips the cumulative effect negative. **REJECTED**
outright -- worse than random. Reading: names in this already-elite top-20
pool that just had their worst week are, in this sample, more often
reacting to genuine bad news (earnings miss, guidance cut) than presenting
an oversold discount -- "falling knife" dominates "oversold bounce" here,
consistent with the batch study's earlier finding that momentum/price-
action-chasing signals underperform quality/fundamentals-based tie-breaks
in this window.

### dynamic_best_subfactor_ic_top20 -- REJECTED (one-year concentration)
**Hypothesis (user, 2026-09-11):** finer-grained version of the rejected
parent-level "route to highest trailing IC" idea -- instead of choosing
among the 8 aggregate parents, go through all 24 individual subfactors
(factors.parent_selection_v4.SELECTED_SUBS) and route to whichever single
SUBFACTOR has the highest trailing-5y IC on the top-20 pool, then rank/pick
by that subfactor's own score. Same no-look-ahead convention as the parent
version. Code: `research.loop_research.dynamic_best_subfactor_ic_top20`.
Full report: output/loop_research/promotion_dynamic_best_subfactor_ic_top20_selector.json.
Compact record: output/loop_research/experiments/dynamic_best_subfactor_ic_top20.json.
Block bootstrap: output/loop_research/dynamic_subfactor_ic_block_bootstrap.csv.

Winner-frequency across 81 dates (unlike the parent-level version, quality
never wins here): `ins_no_selling_flag` 34, `rev_target_revision_raw_30d`
27, `rev_rating_surprise_90d` 5, `qual_altman_z` 4, `inst_net_share_change`
4, `rev_raise_and_bullish_90d` 4, `inst_investors_holding_change` 3 -- two
insider/revisions subfactors dominate, both already known strong
tie-breakers, so this avoided the parent-level failure mode.

**Initially looked like the strongest candidate tested on this pool**:
median Sharpe **+0.245** (1.207 vs 0.962), win rate **97.6%**, leave-one-
year-out passes cleanly for all 7 years, alpha +0.044, max_dd improves,
perturbation (noise on final ranking, winner-subfactor decision left exact)
**88.6%** win rate. Clears every gate in the formal 8-gate rule: **PROMOTE**
by the mechanical rule.

**But the block bootstrap over calendar time told a different story than
it did for the momentum champion.** At the principled L=4 block length:
95% CI [-0.0013, +0.0077], **p=0.166** -- far from significant. Worse, the
p-value got WORSE, not better, as block length grew (L=1: 0.144, L=4:
0.166, L=6: 0.168, L=12: 0.218) -- the opposite of the momentum candidate's
pattern (flat point estimate, p improving with longer blocks, the
signature of a real-but-underpowered effect). A p-value that deteriorates
as more calendar-time structure is honored is a signature of a bursty,
concentrated effect, not a broadly distributed one -- prompted a full
per-year dig (user request).

**Per-year breakdown confirmed a near-textbook one-year artifact:**

| year | sum | mean monthly | % months positive |
|---|---|---|---|
| 2020 | -0.035 | -0.003 | 42% |
| 2021 | -0.049 | -0.004 | 50% |
| 2022 | +0.021 | +0.002 | 75% |
| 2023 | +0.045 | +0.004 | 58% |
| 2024 | +0.064 | +0.005 | 50% |
| **2025** | **+0.216** | **+0.018** | 83% |
| 2026 | -0.014 | -0.002 | 50% |

Cumulative excess return is NEGATIVE for four straight years (2020-2023:
-0.035 -> -0.085 -> -0.064 -> -0.019), only turns clearly positive in 2025,
and 2026 gives most of it back to flat. **2025 alone supplies ~87% of the
entire 7-year cumulative edge** (+0.216 of +0.247 total) -- removing 2025
leaves only +0.032, barely above zero (this is the same number the
promotion report's `remainder_after_best_year_removed` gate already
reported, 0.0315, but the mechanical gate only checks sign, not magnitude,
so it passed on a technicality). Naive t-test on the full series: t=1.42,
p=0.160 -- not significant even before accounting for autocorrelation.

**Investigated what actually drove 2025 (user request), three checks, none
support a genuine signal-discovery story:**
1. **No new subfactor in 2025.** The winner rotation used the exact same
   two subfactors that dominate 2022-2024 (`rev_target_revision_raw_30d`
   7/12 months, `ins_no_selling_flag` 5/12) -- the selection logic did
   nothing different that year.
2. **The chosen subfactor's IC wasn't unusually strong in 2025 either.**
   Trailing IC driving the picks: `rev_target_revision_raw_30d` was 0.086
   in 2025 vs a HIGHER 0.113 in 2024 (which had a much smaller edge, +0.064)
   -- if trailing IC drove the edge size, 2024 should have been the bigger
   year, not 2025. Realized (same-period) IC of whichever subfactor was in
   use each month averaged +0.050 in 2025 -- essentially tied with 2022
   (+0.036) and 2023 (+0.020). 2024's realized chosen-subfactor IC was
   deeply NEGATIVE (-0.113) yet 2024 still had positive excess return --
   there is no consistent relationship between realized signal quality and
   excess return size in this data, which undercuts any causal story.
3. **Not a single-stock fluke either.** 2025 holdings are a diversified
   ~48-name set (AMAT most persistent at 11 of ~48 sleeve-months, ~23%,
   then CAH/TJX/BNY/ADBE/CEG/CAT/WFC) -- no one idiosyncratic mega-winner.
   Within-year it's front-loaded (top 3 months = Mar/Aug/Sep = 62% of the
   2025 total) but spread across a real portfolio, not one lucky trade.

**Verdict: REJECTED, despite passing every mechanical gate in the formal
promotion rule.** Reads as regime luck (2025 happened to be a good year for
the kind of large, quality/revisions/insider-tilted names this top-20 pool
selects) rather than the subfactor-rotation mechanism discovering or
exploiting anything real -- there is no mechanistic link between "how the
rule behaved" or "how good the signal was" and the size of the payoff that
year, which is exactly the opposite of what a genuine, replicable edge
should look like. This is the same failure mode, and the same diagnostic
(per-calendar-time breakdown) that caught the earlier decile-1-minus-100
insider-edge false positive -- a good illustration of why the mechanical
8-gate rule alone is not sufficient and the block-bootstrap +
per-year-breakdown follow-up matters. Champion for the top-20-pool track
remains unchanged: `dynamic_ic_momentum_top20_selector`.

## 10-idea exploration batch (2026-09-11, user request): deliberately unrelated mechanisms

User asked for 10 brand-new ideas, unrelated to anything tried so far, each
run through the full pipeline (8-gate promotion rule with perturbation
baked in, then block bootstrap for anything clearing the mechanical rule),
looped automatically. Code: `research.loop_research.exploration_batch_2026_09_11`
(10 `metric_fn`s + a generic `make_selector`/`make_perturbed_selector` that
turns any per-ticker ranking metric into a full selector, so every idea
gets the same perturbation test for free). Driver + full results:
output/loop_research/exploration_batch_2026_09_11_summary.json, per-
candidate detail in output/loop_research/promotion_<name>.json and
output/loop_research/experiments/<name>.json. Baseline throughout:
`random_top20_selector`.

Deliberately spanned mechanisms never tried on this pool before: price-based
risk (volatility, beta, risk-adjusted momentum), score time-series dynamics
(score momentum, score stability), pool-membership dynamics (new entrant,
tenure), a naive behavioral control (nominal share price), trend-following
(200-day moving-average gap), and a categorical breadth count (parents
above the day's cross-sectional median) -- as opposed to every earlier
candidate, which ranked by a single already-existing score quantity.

| experiment | sharpe_improve | win_rate | perturbation_win_rate | decision |
|---|---|---|---|---|
| low_realized_vol_top20 | -0.259 | 1.2% | 2.8% | REJECT |
| low_beta_top20 | -0.310 | 0.4% | 4.6% | REJECT |
| risk_adjusted_momentum_top20 | +0.056 | 68.6% | 67.6% | REJECT |
| score_momentum_top20 | -0.250 | 1.6% | 11.6% | REJECT |
| score_stability_top20 | -0.176 | 7.8% | 45.8% | REJECT |
| **new_entrant_top20** | **+0.195** | **94.8%** | **74.2%** | **PROMOTE** (mechanically) |
| longest_tenure_top20 | -0.084 | 24.0% | 28.6% | REJECT |
| low_nominal_price_top20 | -0.039 | 38.0% | 59.4% | REJECT |
| above_200dma_gap_top20 | -0.054 | 33.8% | 59.4% | REJECT |
| **parent_breadth_top20** | **+0.250** | **97.8%** | **80.4%** | **PROMOTE** (mechanically) |

**Headline pattern: defensive/low-risk tilts are actively harmful in this
pool, again.** `low_realized_vol_top20` and `low_beta_top20` are the two
worst results of the entire session (win rates 1.2% and 0.4%) -- a third,
independent confirmation (after the 93-candidate batch's vol/beta-exclusion
finding and the general "excluding high-beta names is dramatically worse"
pattern) that this 2020-2026 window's return is concentrated in higher-
beta/higher-vol names within the top-20 pool, and defensively screening
them out is actively destructive, not neutral.

**Momentum/re-rating-chasing signals continue to fail.** `score_momentum_top20`
(chase names whose composite score just rose) and `above_200dma_gap_top20`
(chase names trending furthest above their long-run average) both lost
clearly -- consistent with every prior price-momentum/re-rating-chasing idea
tested this session (worst_week_return_top20, the earlier momentum-
overextension work).

**Sanity check passed: the naive behavioral control correctly failed.**
`low_nominal_price_top20` (lowest raw share price, no economic rationale)
came back REJECT (win rate 38%) as it should -- useful confirmation the
framework isn't rubber-stamping arbitrary rankings.

**Pool-membership dynamics split sharply: freshness beats persistence.**
`new_entrant_top20` (prefer names JUST promoted into the top-20 pool this
month) mechanically PROMOTEs; `longest_tenure_top20` (prefer names that
have been in the pool longest) REJECTs (win rate 24%) -- the pool's edge, if
any, looks like it favors re-rating/newly-arrived names over stable
long-tenured members, the opposite of what score_stability_top20 (also
rejected) would have predicted.

### The two mechanical PROMOTEs, put through the same scrutiny that caught the subfactor-IC false positive

Both `new_entrant_top20` and `parent_breadth_top20` clear every gate in the
formal rule, including perturbation (74.2% and 80.4% win rate respectively
-- comfortably above the 55% bar). Neither clears the block bootstrap at
the conventional bar, so both got the same per-year concentration dig that
caught the earlier subfactor-IC false positive.

**new_entrant_top20:** block bootstrap at L=4: 95% CI [-0.0008, +0.0059],
**p=0.133**. Per-year sums: 2020 +0.006, 2021 +0.052, **2022 +0.139**, 2023
-0.023, 2024 +0.072, 2025 -0.015, 2026 -0.026. **2022 alone supplies 68% of
the total edge (+0.139 of +0.205)**, and the other years flip sign
inconsistently (positive 2021/2024, negative 2023/2025/2026) rather than
building a broad multi-year trend. This is the same failure signature (one
dominant year, sign-flipping elsewhere) that sank `dynamic_best_subfactor_ic_top20`,
just less extreme (68% vs 87% concentration). **Verdict: PROMISING but
fragile, NOT promoted -- reads as another one-year-leaning result, not a
confirmed edge.**

**parent_breadth_top20:** block bootstrap at L=4: 95% CI [-0.0018,
+0.0059], **p=0.300** -- weaker naive significance than new_entrant, but a
materially healthier per-year shape. Per-year sums: 2020 -0.035, 2021
-0.025, 2022 +0.007, 2023 +0.047, **2024 +0.107**, 2025 +0.032, 2026 +0.052.
Cumulative excess is negative through 2020-2021, then turns positive in
2022 and **stays positive for five straight years running** (2022-2026,
every single one) -- 2024 is the largest single contributor (58% of the
total) but is not propping up an otherwise negative or sign-flipping
series, unlike new_entrant_top20 or the rejected subfactor-IC candidate.
**Verdict: PROMISING, more credible than new_entrant_top20 -- looks like a
real but currently underpowered multi-year trend (weak t-stat/bootstrap
because the effect is small relative to noise, not because it's a
concentrated fluke), worth revisiting as more calendar time accrues rather
than either promoting now or dismissing outright.**

**Net for this batch: no change to the top-20-pool champion**
(`dynamic_ic_momentum_top20_selector`, whose block-bootstrap picture at the
same L=4 -- p=0.078, flat point estimate across block lengths -- is still
meaningfully more convincing than either mechanical PROMOTE here). Both
`new_entrant_top20` and `parent_breadth_top20` are kept on record as
PROMISING candidates (parent_breadth the stronger of the two) rather than
promoted or discarded.
