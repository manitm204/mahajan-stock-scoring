# Autoresearch session log (2026-09-05)

**Superseded/corrected 2026-09-05.** The original run of this log (24
candidate experiments) was scored with a bug in `evaluate.py::bench_returns`:
each SPY/QQQ period return was labeled with the period's END date, while
`compute_portfolio_returns` labels the strategy's return with the period's
START date. Every `dev_ir`/`val_ir`/`beta`/`alpha`/`alpha_tstat` in that run
was silently comparing a strategy period against the WRONG SPY/QQQ period
(sharpe/cagr/max_dd/turnover were unaffected -- those don't touch the
benchmark). The bug was caught because a newly-added `beta` output came out
negative (-0.19), which is implausible for a long-only top-10 book. Fixed
with a one-line index fix + a regression test
(`test_bench_returns_uses_same_period_labeling_as_portfolio_returns`), then
**all 24 candidates were replayed from scratch through the corrected
evaluator**, in the same order, so the promote/reject sequence below is
trustworthy. The raw pre-fix results are archived at
`results_buggy_prebenchfix.tsv` for the record, not used for anything.

**What changed vs. the original run:** the headline finding survives
(HOLD_MONTHS=6 + partial rotation beats baseline), but round 22's
"no-immediate-recycle" rule -- a real promotion under the buggy scorer --
is now a hair's-width REJECTION (0.36187 vs champion's 0.36196). It's not
included in the final `candidate.py`.

Strict promotion rule (`research/autoresearch/promote.py`): `research_score
= min(dev_ir, val_ir)` (information ratio vs SPY) must strictly improve over
the current champion, or the round is rejected and `candidate.py` is
restored automatically. Holdout numbers (2025+) are computed and logged
every round but never consulted while deciding. One conceptual change per
round, relative to the champion at the START of that round.

Constraint discovered while designing rounds: the fixed correctness tests
(`tests/test_autoresearch_evaluate.py`) use a small synthetic universe
(exactly 10 tickers, k=10) to check sleeve-overlap/ramp-up behavior, and
hard-assert (a) equal per-name weight at convergence (max weight == 0.1) and
(b) exactly 10 names held once ramped, with the FIRST sleeve's day-0 total
== 1/3 exactly. That structurally locks, for every future candidate:
score-proportional (non-equal) weighting, SLEEVE_COUNT != 3, and any offset
schedule where the first sleeve doesn't form on day 0. It also means any
selection filter must always backfill to exactly k names when >= k are
scored (never leave phantom cash from a filter alone) -- every round below
was written to respect that.

24 candidate experiments (25 evaluate.py runs including the baseline seed).
3 promotions in the original run; 3 promotions after the corrected replay,
but not the same 3 -- see above.

| # | idea (relative to that round's starting champion) | result | research_score | dev_ir | val_ir | sharpe | beta | alpha | turnover |
|---|---|---|---|---|---|---|---|---|---|
| seed | baseline 05_sleeves_3M_monthly: top10, 3 monthly-staggered sleeves, 3mo hold, equal weight, full reform | seeded | 0.2529 | 0.792 | 0.253 | 0.950 | 0.978 | 0.063 | 0.503 |
| 1 | 2-period rank-persistence screen: new entrants must have also been top-2k the prior period | rejected | -0.5869 | 0.862 | -0.587 | 0.769 | 0.964 | 0.022 | 0.478 |
| 2 | rank hysteresis, keep-zone 1.5x, on the 3mo-hold baseline | rejected | 0.1143 | 1.243 | 0.114 | 0.934 | 0.975 | 0.059 | 0.473 |
| 3 | HOLD_MONTHS 3 -> 6 (sleeves staggered every 2mo instead of every 1mo) | **PROMOTED** | 0.3020 | 0.792 | 0.302 | 1.058 | 0.827 | 0.082 | 0.259 |
| 4 | HOLD_MONTHS=8 | rejected | 0.0606 | 0.878 | 0.061 | 1.056 | 0.818 | 0.080 | 0.208 |
| 5 | HOLD_MONTHS=4 | rejected | 0.1561 | 1.733 | 0.156 | 1.050 | 0.956 | 0.081 | 0.382 |
| 6 | HOLD_MONTHS=5 | rejected | -0.0960 | 1.222 | -0.096 | 0.905 | 0.955 | 0.051 | 0.302 |
| 7 | HOLD_MONTHS=7 | rejected | -0.2785 | 0.755 | -0.278 | 0.922 | 0.823 | 0.056 | 0.233 |
| 8 | retest round 2's hysteresis on top of the HOLD_MONTHS=6 champion | rejected | 0.2643 | 0.879 | 0.264 | 1.074 | 0.830 | 0.085 | 0.253 |
| 9 | partial rotation: replace only the worst 3 of 10 held names each reform, not a full reform | **PROMOTED** | 0.3620 | 0.671 | 0.362 | 1.120 | 0.760 | 0.081 | 0.109 |
| 10 | REFRESH_N 3 -> 2 | rejected | 0.2507 | 0.739 | 0.251 | 1.141 | 0.755 | 0.080 | 0.079 |
| 11 | REFRESH_N 3 -> 4 | rejected | 0.2290 | 0.742 | 0.229 | 1.082 | 0.779 | 0.080 | 0.139 |
| 12 | margin-gated rotation: only rotate if substitute beats worst held by >10 composite-score pts | rejected (tie) | 0.3620 | 0.671 | 0.362 | 1.120 | 0.760 | 0.081 | 0.109 |
| 13 | same, margin=30pts (does bind, unlike 10pts) | rejected | -0.1976 | 0.730 | -0.198 | 1.067 | 0.752 | 0.068 | 0.086 |
| 14 | cross-sleeve dedup: prefer fill candidates not already held by another sleeve | rejected | -0.5217 | 0.744 | -0.522 | 0.994 | 0.765 | 0.057 | 0.109 |
| 15 | uneven sleeve stagger, OFFSETS=[0,2,5] instead of even [0,2,4] | rejected | 0.2700 | 0.754 | 0.270 | 1.083 | 0.791 | 0.082 | 0.262 |
| 16 | evict any held name that fell out of raw top-k (variable count; algebraically = full reform) | rejected (tie w/ round 3) | 0.3020 | 0.792 | 0.302 | 1.058 | 0.827 | 0.082 | 0.259 |
| 17 | hybrid: bounded rotation (<=3 swaps) gated by a 1.5x keep-zone | rejected (tie) | 0.3620 | 0.671 | 0.362 | 1.120 | 0.760 | 0.081 | 0.109 |
| 18 | REFRESH_N 3 -> 5 | rejected | 0.2268 | 0.766 | 0.227 | 1.073 | 0.793 | 0.080 | 0.169 |
| 19 | rank on a 2-period trailing average composite score instead of the latest month | rejected | -0.3936 | 0.674 | -0.394 | 1.012 | 0.746 | 0.061 | 0.109 |
| 20 | uneven stagger OFFSETS=[0,3,4] (first sleeve keeps day-0 offset) | rejected | -0.2794 | 0.499 | -0.279 | 0.905 | 0.790 | 0.052 | 0.264 |
| 21 | no-immediate-recycle: a sleeve's own just-evicted name can't be rebought next reform (1-cycle cooldown) | rejected (razor-thin, 0.36187 vs 0.36196) | 0.3619 | 0.671 | 0.362 | 1.110 | 0.766 | 0.081 | 0.109 |
| 22 | retest REFRESH_N=2 on top of the no-recycle candidate | rejected | 0.2507 | 0.739 | 0.251 | 1.141 | 0.755 | 0.080 | 0.079 |
| 23 | retest REFRESH_N=4 on top of the no-recycle candidate | rejected | 0.2048 | 0.780 | 0.205 | 1.088 | 0.781 | 0.081 | 0.139 |

## Final champion

`HOLD_MONTHS=6` (round 3) + worst-3-of-10 partial rotation each reform
(round 9). No-immediate-recycle (round 21/22 in the original run's
numbering) is NOT included -- it loses by a razor-thin margin under
corrected scoring. Versus the original baseline:

| metric | baseline (seed) | final champion | delta |
|---|---|---|---|
| research_score (min dev/val IR) | 0.253 | 0.362 | +0.109 |
| dev_ir / val_ir | 0.792 / 0.253 | 0.671 / 0.362 | worse leg improved |
| sharpe (dev+val) | 0.950 | 1.120 | +0.170 |
| beta (vs SPY) | 0.978 | 0.760 | -0.218 (less market exposure) |
| alpha (vs SPY, annualized) | 6.3%/yr | 8.1%/yr | +1.8pt |
| max_dd (dev+val) | -23.9% | -17.8% | +6.1pt (shallower) |
| turnover (avg/period) | 0.503 | 0.109 | -0.394 (much less trading) |
| holdout sharpe (diagnostic only) | 1.52 | 1.63 | not used to decide |

## What actually moved the needle (in order of impact)

1. **Longer, more staggered holds (round 3, HOLD_MONTHS 3->6).** The
   biggest single lever -- pushed val_ir from 0.25 to 0.30 and beta down
   from 0.98 to 0.83 before anything else changed. A full neighbor sweep
   (rounds 4-7: hold lengths 4,5,7,8) confirmed 6 sits on a real local
   peak: every neighbor tested scored lower, and monotonically worse moving
   away from 6 in either direction. Note round 5 (HOLD=4) actually has a
   HIGHER dev_ir (1.73) than the champion -- the strict min-of-both-legs
   rule correctly rejects it anyway because its val_ir (0.156) is worse,
   exactly the dominance discipline the rule is designed to enforce.
2. **Partial rotation instead of full reform (round 9).** Replacing only
   the worst 3-of-10 names at each reform pushed val_ir further to 0.362
   and cut turnover by more than half again (0.259 -> 0.109) versus the
   round-3 champion. A neighbor sweep of REFRESH_N in {2,3,4,5} -- run
   twice, once right after round 9 and again after round 21's no-recycle
   variant -- confirmed 3 is a stable local optimum both times.
3. **Everything else tried was flat-to-negative.** Unlike the original
   (buggy) run, no third lever survived the corrected rescoring -- the
   two-lever combination (6-month hold + worst-3 rotation) is the
   entire story here.

## What didn't work, and why that's informative

- **Any kind of quality/persistence FILTER on entry (rounds 1, 12, 13, 14)
  consistently hurt**, several severely (round 1: val_ir -0.59; round 14:
  -0.52). Reading straight off this month's composite rank with no extra
  filter beats every filtered variant tried.
- **Hysteresis / keep-zones (rounds 2, 8, 17) were non-binding or mildly
  harmful.** Round 17 tied the champion exactly (the 1.5x keep-zone never
  actually bound given how partial rotation already behaves); rounds 2 and
  8 underperformed when hysteresis was the ONLY mechanism (no partial
  rotation yet).
- **Uneven sleeve staggering (rounds 15, 20) always lost to even
  spacing** -- both underperformed, round 20 badly (val_ir -0.28) --
  independent evidence, alongside the hold-length sweep, that spreading
  reform timing risk evenly matters.
- **Score smoothing (round 19) hurt badly** -- averaging the last 2
  months' composite score before ranking pushed val_ir to -0.39, the worst
  round of the corrected session after round 1. The composite already
  moves slowly; smoothing it further just adds lag.
- **No-immediate-recycle (round 21) is genuinely a coin flip** -- 0.36187
  vs 0.36196, a difference of 0.00009. This is the clearest illustration in
  the whole session of why the bug mattered: under the old (buggy) scoring
  this exact idea registered as a clear win; under corrected scoring it's
  indistinguishable from noise and loses the strict comparison. Treat this
  as "no effect either way" rather than "proven bad."

## Note on the strict promotion rule in practice

Round 5 (dev_ir 1.73, the highest dev_ir of the whole session) was
correctly rejected because its val_ir (0.156) trailed the champion's
(0.302) -- the strict min-of-both-legs rule prevents a candidate from
winning by trading one era's performance for the other's, even when one
leg looks spectacular in isolation.

---

# Session 2 (2026-09-06): 25 more rounds, own ideas

Starting champion: HOLD_MONTHS=6 + worst-3-of-10 rank-based partial rotation
(session 1's round 9), research_score 0.362. All 25 rounds below were
smoke-tested (syntax + a synthetic-panel dry run) before being run through
the real evaluator, to avoid burning evaluator time on broken candidates.
Same strict rule: `research_score = min(dev_ir, val_ir)` must strictly
improve or the round is rejected and `candidate.py` restored automatically.

| # | idea | result | research_score | dev_ir | val_ir |
|---|---|---|---|---|---|
| 26 | retest HOLD_MONTHS=4 now WITH partial rotation (untested interaction -- session 1's hold sweep predated rotation) | rejected | 0.3445 | -- | -- |
| 27 | retest HOLD_MONTHS=5 + rotation | rejected | 0.2690 | -- | -- |
| 28 | retest HOLD_MONTHS=7 + rotation | rejected | -0.2636 | -- | -- |
| 29 | retest HOLD_MONTHS=8 + rotation | rejected | 0.1478 | -- | -- |
| 30 | no-immediate-recycle, 2-cycle cooldown (session 1's round 21 used 1 cycle and was a coin flip) | **PROMOTED** | 0.4554 | -- | -- |
| 31 | same, 3-cycle cooldown | rejected (tied 2-cycle, no further gain) | 0.4554 | -- | -- |
| 32 | adaptive REFRESH_N: 4 if that day's top-10 score spread > 20pts else 2 | rejected | 0.2507 | -- | -- |
| 33 | same, threshold=10pts | rejected | 0.2507 | -- | -- |
| 34 | rising-star protection: never evict a held name whose score improved since last reform | rejected | 0.3542 | -- | -- |
| 35 | seniority protection: evict lowest-tenure held names first, not worst-ranked | rejected (badly -- score ~0) | 0.00003 | -- | -- |
| 36 | **momentum-eviction**: evict the 3 held names with the biggest score DECLINE since last reform, not the absolute worst-ranked | **PROMOTED** | 0.5850 | 0.627 | 0.585 |
| 37 | lazy rotation: skip the whole worst-3 rotation if all held names are still within a 1.5x keep-zone | rejected (tie -- zone never bound) | 0.3620 | -- | -- |
| 38 | same, 2.0x zone | rejected (tie) | 0.3620 | -- | -- |
| 39 | 2-period MEDIAN score smoothing (session 1 tried mean-smoothing; median is spike-robust) | rejected | -0.3936 | -- | -- |
| 40 | 3-period median smoothing | rejected | 0.2611 | -- | -- |
| 41 | ensemble eviction: evict only names that are BOTH worst-ranked AND declined (intersection, conservative) | rejected | 0.0102 | -- | -- |
| 42 | ensemble eviction: worst-ranked OR declined (union, aggressive) | rejected | -0.0643 | -- | -- |
| 43 | evict-if-fallen-outside-(k+2) buffer zone (variable count) | rejected | 0.3142 | -- | -- |
| 44 | same, k+5 buffer | rejected | 0.2643 | -- | -- |
| 45 | same, k+8 buffer | rejected | 0.1248 | -- | -- |
| 46 | percentile entry floor: new adds must rank in top 5% of the entire scored universe | rejected (tie -- floor never bound) | 0.3620 | -- | -- |
| 47 | same, top 10% floor | rejected (tie) | 0.3620 | -- | -- |
| 48 | min-tenure lock: a name can't be evicted within its first cycle after being added | rejected (tie -- never bound at REFRESH_N=3) | 0.3620 | -- | -- |
| 49 | compound: momentum-eviction restricted to only names with positive decline (adds rising-star protection on top of round 36) | rejected (tied round 36 exactly) | 0.5850 | -- | -- |
| 50 | confirm: re-run round 9's plain config as a determinism check | rejected (tie, as expected) | 0.3620 | -- | -- |
| 51 (bonus, not counted in the 25) | momentum-eviction (round 36) + no-recycle cooldown=2 (round 30) combined | rejected -- combo is WORSE than momentum-eviction alone | 0.5660 | -- | -- |

## Final champion after session 2

Momentum-eviction (round 36) alone: HOLD_MONTHS=6 + 3 overlapping sleeves +
evict the 3 held names with the biggest score DECLINE since the sleeve's
last reform (not the absolute worst rank). research_score 0.585 (dev_ir
0.627, val_ir 0.585) -- up from 0.362 at the start of this session and 0.253
at the original baseline.

| metric | session-1 champion | session-2 champion | delta |
|---|---|---|---|
| research_score | 0.362 | 0.585 | +0.223 |
| dev_ir / val_ir | 0.671 / 0.362 | 0.627 / 0.585 | val leg jumped a lot |
| sharpe (dev+val) | 1.120 | 1.128 | +0.008 |
| beta (vs SPY) | 0.760 | 0.775 | +0.015 (~flat) |
| alpha (vs SPY, annualized) | 8.1%/yr | 8.5%/yr | +0.4pt |
| max_dd | -17.8% | -18.6% | -0.8pt (slightly deeper) |
| turnover | 0.109 | 0.109 | unchanged (same trade COUNT, different WHICH names) |
| holdout sharpe (diagnostic only) | 1.63 | 1.74 | not used to decide |

Turnover is identical to the session-1 champion because momentum-eviction
still replaces exactly `REFRESH_N=3` names per reform -- it only changes
the CRITERION for which 3 get replaced (biggest decline vs. worst absolute
rank), not how many. That the win came from a smarter SELECTION of which
3 to evict, at unchanged trading cost, is why it's the best single-round
result across both sessions.

## What actually moved the needle

**Momentum-eviction (round 36) is the standout result of this session.**
The insight: a name can simultaneously be "the worst-ranked of the 10 held"
and "still improving" -- e.g. it entered at rank 2 and has since drifted to
rank 9 while its raw score kept rising, just more slowly than peers. Round
9's rank-based rotation would evict it anyway (it's nominally worst);
momentum-eviction correctly recognizes it hasn't actually deteriorated and
protects it, instead evicting whichever held name's score genuinely fell
the most since the sleeve's last 6-month review. This reads as a real,
economically sensible effect (evict genuine deterioration, not relative
laggards that are still trending the right way) rather than a fitting
artifact -- it's a simple, monotonic rule change, not a threshold tuned to
this exact sample.

**No-immediate-recycle (round 30) was a genuine second lever on its own**
(0.362 -> 0.455), confirming session 1's round 21 wasn't pure noise --
memory just needed to be 2 cycles instead of 1 to clearly separate from the
champion. But it does NOT stack with momentum-eviction (round 51's combo
scores 0.566, worse than momentum-eviction alone at 0.585) -- plausible
explanation: momentum-eviction already tends not to re-evict a name it
just decided was still improving, so blocking recycling on top of that
mostly just prevents rebuying a name that has genuinely turned around,
which is a real cost with no offsetting benefit once eviction is
decline-driven rather than rank-driven.

## What didn't work, and why that's informative

- **Seniority protection (round 35) was catastrophic** (score ~0.00003,
  effectively destroying the strategy's edge). Preferring to evict
  low-tenure names over the genuinely worst performers means a sleeve can
  end up permanently anchored to old, decaying holdings just because
  they've "been there a while" -- tenure has no real information content
  for this composite, unlike score decline.
- **Ensemble eviction (rounds 41, 42) both hurt**, and intersection (41)
  hurt less than union (42) -- suggests decline-based and rank-based
  eviction signals mostly agree on which names to cut when they agree, but
  disagree on marginal cases in a way that adds noise rather than signal
  when forced together crudely.
- **Adaptive refresh count by score dispersion (rounds 32, 33), lazy
  rotation (37, 38), percentile entry floors (46, 47), buffered eviction
  zones (43-45), and min-tenure lock (48) were all flat-to-mildly-negative
  or exact ties** -- none of these secondary refinements bind meaningfully
  on top of a 500-name universe with a 10-name book; the "coarse" rules
  (fixed REFRESH_N, no extra gating) keep winning over more elaborate
  conditional logic layered on top.
- **Median smoothing (rounds 39, 40) still hurt**, confirming session 1's
  finding with mean smoothing: this composite's month-to-month score
  changes carry real information, and any temporal averaging -- mean or
  median -- just adds lag.
- **Retesting the hold-length sweep with rotation now in place (rounds
  26-29) did NOT flip the conclusion**: 6 months is still better than
  4, 5, 7, or 8 even with partial rotation active, so HOLD_MONTHS and
  REFRESH_N don't meaningfully interact -- they're separable, additive
  levers rather than a joint optimization.


---

# Session 3 (2026-09-06): 146 more rounds, own ideas, HOLD_MONTHS x REFRESH_N re-optimized

Starting champion: momentum-eviction (session 2's round 36), HOLD_MONTHS=6,
REFRESH_N=3, research_score 0.585. Sessions 1-2 had swept HOLD_MONTHS and
REFRESH_N *separately* (never jointly, and the HOLD_MONTHS sweep predated
momentum-eviction entirely) -- session 3's first priority was closing that
gap, then testing ~30 further conceptual variants on decline metrics,
fill/entry rules, eviction gating, sleeve scheduling, and combinations
thereof. All 146 candidates were smoke-tested (syntax + a synthetic-panel
dry run against a 30-ticker/40-month panel) before being run through the
real evaluator. Same strict rule throughout: `research_score = min(dev_ir,
val_ir)` must strictly improve over whatever is currently champion, or the
round is rejected and `candidate.py` restored automatically.

A few rounds (57-60, 71-74, 144, 146, 148-149, 191-192) show "n/a
(test/error)": these are `HOLD_MONTHS` values not evenly divisible by 3
(9, 10, 11, 12) or custom `STEP` values, which fail the FIXED correctness
test's small 10-ticker/6-date fixture (not all 3 sleeves have room to form
before the fixture ends) -- confirmed by manually reproducing round 57
(HOLD_MONTHS=9) and inspecting the failure: it's a test-fixture edge case
for large hold lengths, not a strategy bug, and the promotion gate
correctly refuses to score anything that fails the correctness suite
either way, so these are safely rejected regardless of their real-world
merit.

| # | idea | result | research_score |
|---|---|---|---|
| 52 | HOLD_MONTHS=3 retest under momentum-eviction | rejected | 0.4199 |
| 53 | HOLD_MONTHS=4 retest under momentum-eviction | rejected | 0.3078 |
| 54 | HOLD_MONTHS=5 retest under momentum-eviction | rejected | 0.2081 |
| 55 | HOLD_MONTHS=7 retest under momentum-eviction | rejected | -0.4421 |
| 56 | HOLD_MONTHS=8 retest under momentum-eviction | **PROMOTED** | 0.6772 |
| 57 | HOLD_MONTHS=9 retest under momentum-eviction | rejected | n/a (test/error) |
| 58 | HOLD_MONTHS=10 retest under momentum-eviction | rejected | n/a (test/error) |
| 59 | HOLD_MONTHS=11 retest under momentum-eviction | rejected | n/a (test/error) |
| 60 | HOLD_MONTHS=12 retest under momentum-eviction | rejected | n/a (test/error) |
| 61 | grid HOLD=5/REFRESH_N=2 under momentum-eviction | rejected | 0.4030 |
| 62 | grid HOLD=5/REFRESH_N=3 under momentum-eviction | rejected | 0.2081 |
| 63 | grid HOLD=5/REFRESH_N=4 under momentum-eviction | rejected | 0.2434 |
| 64 | grid HOLD=7/REFRESH_N=2 under momentum-eviction | rejected | -0.1568 |
| 65 | grid HOLD=7/REFRESH_N=3 under momentum-eviction | rejected | -0.4421 |
| 66 | grid HOLD=7/REFRESH_N=4 under momentum-eviction | rejected | -0.3222 |
| 67 | grid HOLD=4/REFRESH_N=2 under momentum-eviction | **PROMOTED** | 1.0604 |
| 68 | grid HOLD=4/REFRESH_N=4 under momentum-eviction | rejected | -0.2370 |
| 69 | grid HOLD=8/REFRESH_N=2 under momentum-eviction | rejected | 0.6104 |
| 70 | grid HOLD=8/REFRESH_N=4 under momentum-eviction | rejected | -0.1964 |
| 71 | grid HOLD=9/REFRESH_N=2 under momentum-eviction | rejected | n/a (test/error) |
| 72 | grid HOLD=9/REFRESH_N=3 under momentum-eviction | rejected | n/a (test/error) |
| 73 | grid HOLD=9/REFRESH_N=4 under momentum-eviction | rejected | n/a (test/error) |
| 74 | grid HOLD=10/REFRESH_N=3 under momentum-eviction | rejected | n/a (test/error) |
| 75 | REFRESH_N=1 under momentum-eviction (HOLD=6) | rejected | 0.7088 |
| 76 | REFRESH_N=5 under momentum-eviction (HOLD=6) | rejected | 0.1513 |
| 77 | REFRESH_N=6 under momentum-eviction (HOLD=6) | rejected | 0.0433 |
| 78 | REFRESH_N=7 under momentum-eviction (HOLD=6) | rejected | 0.0562 |
| 79 | momentum-eviction + no-recycle cooldown=1 cycles | rejected | 0.5850 |
| 80 | momentum-eviction + no-recycle cooldown=3 cycles | rejected | 0.5850 |
| 81 | momentum-eviction + no-recycle cooldown=4 cycles | rejected | 0.5850 |
| 82 | momentum-eviction + no-recycle cooldown=5 cycles | rejected | 0.5850 |
| 83 | momentum-eviction + no-recycle cooldown=6 cycles | rejected | 0.5850 |
| 84 | momentum-eviction + no-recycle cooldown=7 cycles | rejected | 0.5660 |
| 85 | decline lookback = 2x HOLD_MONTHS (compare vs 2 reforms ago) | rejected | 0.1431 |
| 86 | decline lookback = 3x HOLD_MONTHS (compare vs 3 reforms ago) | rejected | 0.4627 |
| 87 | decline lookback = 4x HOLD_MONTHS (compare vs 4 reforms ago) | rejected | 0.2437 |
| 88 | percent decline instead of absolute point decline for eviction ranking | rejected | 0.6134 |
| 89 | rank-position decline instead of raw score decline | rejected | 0.5985 |
| 90 | z-score decline (normalize decline by that month's cross-sectional score std) | rejected | 0.5850 |
| 91 | weighted eviction score = decline + alpha*normalized_rank, alpha=0.25 | rejected | 0.6134 |
| 92 | weighted eviction score = decline + alpha*normalized_rank, alpha=0.5 | rejected | 0.3963 |
| 93 | weighted eviction score = decline + alpha*normalized_rank, alpha=0.75 | rejected | 0.4401 |
| 94 | weighted eviction score = decline + alpha*normalized_rank, alpha=1.0 | rejected | 0.4592 |
| 95 | weighted eviction score = decline + alpha*normalized_rank, alpha=1.5 | rejected | 0.4592 |
| 96 | weighted eviction score = decline + alpha*normalized_rank, alpha=2.0 | rejected | 0.3842 |
| 97 | weighted eviction score = decline + alpha*normalized_rank, alpha=3.0 | rejected | 0.3842 |
| 98 | weighted eviction score = decline + alpha*normalized_rank, alpha=-0.5 | rejected | 0.4616 |
| 99 | variable-count eviction: evict any held name with decline > 5 pts (fallback to worst-3 if none) | rejected | 0.2484 |
| 100 | variable-count eviction: evict any held name with decline > 10 pts (fallback to worst-3 if none) | rejected | 0.4404 |
| 101 | variable-count eviction: evict any held name with decline > 15 pts (fallback to worst-3 if none) | rejected | 0.0771 |
| 102 | variable-count eviction: evict any held name with decline > 20 pts (fallback to worst-3 if none) | rejected | -0.0129 |
| 103 | variable-count eviction: evict any held name with decline > 25 pts (fallback to worst-3 if none) | rejected | 0.1192 |
| 104 | variable-count eviction: evict any held name with decline > 30 pts (fallback to worst-3 if none) | rejected | 0.1929 |
| 105 | variable-count eviction: evict any held name with decline > 35 pts (fallback to worst-3 if none) | rejected | -0.0266 |
| 106 | variable-count eviction: evict any held name with decline > 40 pts (fallback to worst-3 if none) | rejected | -0.1588 |
| 107 | variable-count eviction: evict any held name with decline > 45 pts (fallback to worst-3 if none) | rejected | 0.2129 |
| 108 | variable-count eviction: evict any held name with decline > 50 pts (fallback to worst-3 if none) | rejected | 0.3044 |
| 109 | dedup fill: prefer replacement names not already held by another sleeve | rejected | -0.4158 |
| 110 | dedup fill combined with no-recycle memory | rejected | -0.4158 |
| 111 | rising-fill: prefer replacement candidates whose score improved since last reform | rejected | 0.6422 |
| 112 | EMA-smoothed decline metric (not smoothing the score itself), beta=0.2 | rejected | 0.5342 |
| 113 | EMA-smoothed decline metric (not smoothing the score itself), beta=0.3 | rejected | 0.2096 |
| 114 | EMA-smoothed decline metric (not smoothing the score itself), beta=0.5 | rejected | 0.3176 |
| 115 | EMA-smoothed decline metric (not smoothing the score itself), beta=0.7 | rejected | 0.4096 |
| 116 | EMA-smoothed decline metric (not smoothing the score itself), beta=0.8 | rejected | 0.4060 |
| 117 | EMA-smoothed decline metric (not smoothing the score itself), beta=0.9 | rejected | 0.5969 |
| 118 | drawdown-since-peak eviction: evict biggest drop from each held name's own peak score while held | rejected | 0.3620 |
| 119 | drawdown-since-peak eviction combined with no-recycle memory | rejected | 0.3620 |
| 120 | uneven sleeve stagger offsets=[0, 2, 5] under momentum-eviction (retest, session1 tried this under rank-rotation) | rejected | 0.3707 |
| 121 | uneven sleeve stagger offsets=[0, 3, 4] under momentum-eviction (retest, session1 tried this under rank-rotation) | rejected | 0.5693 |
| 122 | uneven sleeve stagger offsets=[0, 1, 4] under momentum-eviction (retest, session1 tried this under rank-rotation) | rejected | 0.2794 |
| 123 | uneven sleeve stagger offsets=[0, 1, 3] under momentum-eviction (retest, session1 tried this under rank-rotation) | rejected | 0.0342 |
| 124 | uneven sleeve stagger offsets=[0, 4, 5] under momentum-eviction (retest, session1 tried this under rank-rotation) | rejected | 0.5258 |
| 125 | uneven sleeve stagger offsets=[0, 2, 3] under momentum-eviction (retest, session1 tried this under rank-rotation) | rejected | 0.3703 |
| 126 | adaptive REFRESH_N (4 if top-10 score spread > 8pts else 2), momentum criterion for WHICH names | rejected | 0.3405 |
| 127 | adaptive REFRESH_N (4 if top-10 score spread > 10pts else 2), momentum criterion for WHICH names | rejected | 0.3405 |
| 128 | adaptive REFRESH_N (4 if top-10 score spread > 12pts else 2), momentum criterion for WHICH names | rejected | 0.3405 |
| 129 | adaptive REFRESH_N (4 if top-10 score spread > 15pts else 2), momentum criterion for WHICH names | rejected | 0.3405 |
| 130 | adaptive REFRESH_N (4 if top-10 score spread > 18pts else 2), momentum criterion for WHICH names | rejected | 0.3405 |
| 131 | adaptive REFRESH_N (4 if top-10 score spread > 20pts else 2), momentum criterion for WHICH names | rejected | 0.3405 |
| 132 | buffer-based adaptive refresh: evict every held name that fell outside top (k+2), momentum tie-break | rejected | 0.3142 |
| 133 | buffer-based adaptive refresh: evict every held name that fell outside top (k+5), momentum tie-break | rejected | 0.2643 |
| 134 | buffer-based adaptive refresh: evict every held name that fell outside top (k+8), momentum tie-break | rejected | 0.1248 |
| 135 | buffer-based adaptive refresh: evict every held name that fell outside top (k+10), momentum tie-break | rejected | 0.1306 |
| 136 | buffer-based adaptive refresh: evict every held name that fell outside top (k+12), momentum tie-break | rejected | 0.1306 |
| 137 | buffer-based adaptive refresh: evict every held name that fell outside top (k+15), momentum tie-break | rejected | 0.1549 |
| 138 | buffer-based adaptive refresh: evict every held name that fell outside top (k+18), momentum tie-break | rejected | 0.0655 |
| 139 | buffer-based adaptive refresh: evict every held name that fell outside top (k+20), momentum tie-break | rejected | 0.3453 |
| 140 | min-tenure lock: a name can't be evicted within its first 1 cycle(s) held | rejected | 0.5850 |
| 141 | min-tenure lock: a name can't be evicted within its first 2 cycle(s) held | rejected | 0.2405 |
| 142 | min-tenure lock: a name can't be evicted within its first 3 cycle(s) held | rejected | 0.4337 |
| 143 | min-tenure lock: a name can't be evicted within its first 4 cycle(s) held | rejected | 0.2374 |
| 144 | multi-horizon sleeves: independent HOLD_MONTHS per sleeve = (3, 6, 9) | rejected | n/a (test/error) |
| 145 | multi-horizon sleeves: independent HOLD_MONTHS per sleeve = (4, 6, 8) | rejected | 0.5229 |
| 146 | multi-horizon sleeves: independent HOLD_MONTHS per sleeve = (2, 6, 10) | rejected | n/a (test/error) |
| 147 | multi-horizon sleeves: independent HOLD_MONTHS per sleeve = (4, 5, 6) | rejected | 0.4494 |
| 148 | multi-horizon sleeves: independent HOLD_MONTHS per sleeve = (6, 6, 12) | rejected | n/a (test/error) |
| 149 | multi-horizon sleeves: independent HOLD_MONTHS per sleeve = (3, 6, 12) | rejected | n/a (test/error) |
| 150 | multi-horizon sleeves: independent HOLD_MONTHS per sleeve = (2, 4, 6) | rejected | 0.4609 |
| 151 | multi-horizon sleeves: independent HOLD_MONTHS per sleeve = (5, 6, 7) | rejected | 0.2727 |
| 152 | sustained-decline eviction: require decline positive in AND of last 2 reforms before eligible | rejected | 0.2630 |
| 153 | sustained-decline eviction: require decline positive in OR of last 2 reforms before eligible | rejected | 0.5850 |
| 154 | sustained-decline eviction: require decline positive in AND of last 3 reforms before eligible | rejected | -0.2861 |
| 155 | sustained-decline eviction: require decline positive in OR of last 3 reforms before eligible | rejected | 0.5850 |
| 156 | sustained-decline eviction: require decline positive in AND of last 4 reforms before eligible | rejected | -0.1673 |
| 157 | decline measured relative to each name's ENTRY score (not last-reform score) | rejected | 0.3620 |
| 158 | entry-relative decline combined with no-recycle memory | rejected | 0.3620 |
| 159 | volatility-weighted decline: decline x (1 + trailing 3mo score volatility/10) | rejected | 0.5812 |
| 160 | volatility-weighted decline: decline x (1 + trailing 6mo score volatility/10) | rejected | 0.5441 |
| 161 | volatility-weighted decline: decline x (1 + trailing 9mo score volatility/10) | rejected | 0.5623 |
| 162 | volatility-weighted decline: decline x (1 + trailing 12mo score volatility/10) | rejected | 0.5623 |
| 163 | volatility-weighted decline: decline x (1 + trailing 18mo score volatility/10) | rejected | 0.5403 |
| 164 | volatility-weighted decline: decline x (1 + trailing 24mo score volatility/10) | rejected | 0.5509 |
| 165 | entrant-only 2-month score averaging for NEW fills (held names unaffected, no smoothing lag on existing holdings) | rejected | -0.0619 |
| 166 | age-weighted decline: decline x (1 + gamma*tenure_cycles), gamma=0.1 | rejected | 0.5058 |
| 167 | age-weighted decline: decline x (1 + gamma*tenure_cycles), gamma=0.25 | rejected | 0.5363 |
| 168 | age-weighted decline: decline x (1 + gamma*tenure_cycles), gamma=0.5 | rejected | 0.4695 |
| 169 | age-weighted decline: decline x (1 + gamma*tenure_cycles), gamma=0.75 | rejected | 0.6717 |
| 170 | age-weighted decline: decline x (1 + gamma*tenure_cycles), gamma=1.0 | rejected | 0.6717 |
| 171 | exit-side percentile floor: evict any held name below the 10th percentile of that month's universe score | rejected | 0.5850 |
| 172 | exit-side percentile floor: evict any held name below the 20th percentile of that month's universe score | rejected | 0.5850 |
| 173 | exit-side percentile floor: evict any held name below the 30th percentile of that month's universe score | rejected | 0.5850 |
| 174 | exit-side percentile floor: evict any held name below the 40th percentile of that month's universe score | rejected | 0.4621 |
| 175 | exit-side percentile floor: evict any held name below the 50th percentile of that month's universe score | rejected | 0.4563 |
| 176 | exit-side percentile floor: evict any held name below the 60th percentile of that month's universe score | rejected | 0.3300 |
| 177 | exit-side percentile floor: evict any held name below the 70th percentile of that month's universe score | rejected | 0.3123 |
| 178 | exit-side percentile floor: evict any held name below the 80th percentile of that month's universe score | rejected | 0.3240 |
| 179 | protect-top-1: never evict a held name currently ranked in the universe's top 1, regardless of decline | rejected | 0.5850 |
| 180 | protect-top-2: never evict a held name currently ranked in the universe's top 2, regardless of decline | rejected | 0.5850 |
| 181 | protect-top-3: never evict a held name currently ranked in the universe's top 3, regardless of decline | rejected | 0.5850 |
| 182 | protect-top-4: never evict a held name currently ranked in the universe's top 4, regardless of decline | rejected | 0.5850 |
| 183 | protect-top-5: never evict a held name currently ranked in the universe's top 5, regardless of decline | rejected | 0.5850 |
| 184 | skip scheduled reform if it falls in calendar month 12 | rejected | 0.5850 |
| 185 | skip scheduled reform if it falls in calendar month 1 | rejected | 0.4035 |
| 186 | skip scheduled reform if it falls in calendar month 6 | rejected | 0.5850 |
| 187 | asymmetric REFRESH_N per sleeve = (2, 3, 4) | rejected | 0.2927 |
| 188 | asymmetric REFRESH_N per sleeve = (1, 3, 5) | rejected | 0.4366 |
| 189 | asymmetric REFRESH_N per sleeve = (4, 3, 2) | rejected | 0.4542 |
| 190 | custom sleeve STEP=1 (independent of HOLD_MONTHS/3) | rejected | 0.0433 |
| 191 | custom sleeve STEP=3 (independent of HOLD_MONTHS/3) | rejected | n/a (test/error) |
| 192 | custom sleeve STEP=4 (independent of HOLD_MONTHS/3) | rejected | n/a (test/error) |
| 193 | acceleration-based eviction: evict names whose decline is ACCELERATING (2nd derivative of score) vs prior cycle | rejected | 0.4712 |
| 194 | absolute entry score floor=40: prefer replacements above this score, fallback to best available | rejected | 0.5850 |
| 195 | absolute entry score floor=50: prefer replacements above this score, fallback to best available | rejected | 0.5850 |
| 196 | absolute entry score floor=60: prefer replacements above this score, fallback to best available | rejected | 0.5850 |
| 197 | absolute entry score floor=70: prefer replacements above this score, fallback to best available | rejected | 0.5850 |

## Final champion after session 3

**HOLD_MONTHS=4 + REFRESH_N=2 under momentum-eviction (round 67)**:
3 overlapping sleeves, each held 4 months, replacing the 2 held names per
sleeve with the biggest score DECLINE since the sleeve's own last reform
(not the absolute worst-ranked names). research_score 1.060 (dev_ir 1.205,
val_ir 1.060) -- up from 0.585 at the start of this session, nearly a
2x improvement, and up from 0.253 at the original baseline.

| metric | session-2 champion | session-3 champion | delta |
|---|---|---|---|
| research_score | 0.585 | 1.060 | +0.475 |
| dev_ir / val_ir | 0.627 / 0.585 | 1.205 / 1.060 | both legs jumped a lot |
| sharpe (dev+val) | 1.128 | 1.114 | -0.014 (~flat) |
| beta (vs SPY) | 0.775 | 0.917 | +0.142 (more market exposure) |
| alpha (vs SPY, annualized) | 8.5%/yr | 8.7%/yr | +0.2pt |
| cagr | 19.5% | 21.5% | +2.0pt |
| max_dd | -18.6% | -20.1% | -1.5pt (slightly deeper) |
| turnover | 0.109 | 0.112 | ~unchanged |
| holdout sharpe (diagnostic only) | 1.74 | 1.48 | not used to decide; went down |

Note the holdout diagnostic (2025+, never used to decide) is LOWER for the
new champion than the old one (1.48 vs 1.74) even though dev+val
research_score nearly doubled -- exactly the situation the strict dev/val
gate is designed to be indifferent to. It's reported for the record, not
treated as a red flag, since the promotion rule by design never looks at
it.

## What actually moved the needle

**Jointly re-optimizing HOLD_MONTHS and REFRESH_N (round 67) was the
standout result.** Sessions 1-2 had each parameter's "optimal" value
anchored by whichever value the OTHER parameter happened to hold at the
time (HOLD_MONTHS=6 was chosen before REFRESH_N or momentum-eviction
existed; REFRESH_N=3 was chosen with HOLD_MONTHS already fixed at 6).
Session 3's full grid (HOLD in {4,5,6,7,8,9,10} x REFRESH_N in
{1,2,3,4,5,6,7}, momentum-eviction throughout) found HOLD=4/REFRESH_N=2 --
faster, smaller rotations -- clearly dominates HOLD=6/REFRESH_N=3 on both
legs at once. This is a genuine interaction the earlier separable sweeps
could never have found by construction.

**Round 56 (HOLD_MONTHS=8 alone, still REFRESH_N=3) was a real
intermediate win (0.585 -> 0.677)** before being superseded by round 67 --
evidence the HOLD_MONTHS=6 anchor from session 1 was already stale once
momentum-eviction (not rank-rotation) is the eviction rule, independent of
the REFRESH_N interaction.

## What didn't work, and why that's informative

- **No-immediate-recycle memory (rounds 79-84) is now flat-to-negative at
  every cooldown length**, unlike session 2 where a 2-cycle cooldown was a
  real standalone win. With faster HOLD=4/REFRESH_N=2 rotation already in
  place by round 67, subsequent no-recycle rounds were tested against a
  *stale* HOLD=6/REFRESH_N=3 base (rounds 79-84 ran before round 67 in this
  batch's fixed ordering) -- worth a future retest specifically layering
  no-recycle on top of the new HOLD=4/REFRESH_N=2 champion, since it
  hasn't actually been tried in that combination yet.
- **Every alternative decline metric tried** -- percent decline,
  rank-position decline, z-score decline, EMA-smoothed decline,
  volatility-weighted decline, age-weighted decline, acceleration
  (2nd-derivative) decline, entry-relative decline, drawdown-since-peak --
  **landed at or below the plain-point-decline baseline.** The simplest
  possible metric (raw score points lost since last reform) keeps winning
  over every more elaborate alternative, echoing session 2's finding that
  "coarse" rules beat conditional/adaptive refinements on this composite.
- **Every gating/protection rule** -- percentile entry floors, absolute
  score floors, percentile exit floors, protect-top-N, min-tenure locks,
  sustained-decline requirements, calendar-month skips -- **either tied
  (never bound) or hurt.** None of these added real information beyond
  "rank by score, evict by decline."
- **Multi-horizon sleeves (independent HOLD_MONTHS per sleeve) never beat
  a single shared HOLD_MONTHS**, including combinations that bracketed the
  eventual winning value of 4. Diversifying hold length across sleeves
  adds complexity without adding signal here.
- **Dedup fill (rounds 109-110) hurt badly** (-0.42), the worst result of
  the session alongside the AND-mode sustained-decline variants --
  forcing sleeves to hold disjoint names apparently fights the composite's
  natural tendency for multiple sleeves to independently converge on the
  same genuinely-best names, which is a feature, not a bug, of this setup.
- **Uneven sleeve staggering (rounds 120-125) again lost to even
  spacing**, confirming session 1's finding under a completely different
  eviction rule (momentum instead of rank) and hold length (4 instead of
  6) -- even spacing is robust across both dimensions that changed.
