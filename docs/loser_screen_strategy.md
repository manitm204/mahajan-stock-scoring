# The Loser-Screen Strategy — Results & Decisions

> **⚠ INVALIDATED 2026-07-15 — ghost-member PIT bug.** Every backtest number
> in this document was computed on window caches contaminated by non-members:
> `build_parent_panel` reindexed each date's frame to the all-time union
> universe, and `composite_from_parents` NEUTRAL-filled the resulting all-NaN
> ghost rows, so ~200 non-members per date (delisted zombies AND
> future index entrants = look-ahead) received neutral composites, passed the
> top-75% screen, and were structurally unvetoable. Books were roughly HALF
> ghosts (vpos6_20 "267 names" → ~148 real). Fixed same day (compose.py +
> backtest.py, `tests/test_ghost_members.py`); caches rebuilt PIT-clean.
> **Clean re-run (`output/loserscreen_veto2_pit/`): mix_screen25 Sharpe 0.820
> (loses to SPY by 1.0%/yr); vpos6_20 Sharpe 0.892, +0.3%/yr vs SPY,
> +1.2%/yr vs screen (t = 1.07, 10/19 windows, beats 94.5% of null); vqs20
> t = 0.63, 10/19 — FAILS its pre-registered bar.** The headline edge below
> was largely a ghost artifact. The clean read is now CONSISTENT with the
> independent walk-forward validation (no OOS alpha vs SPY). Strategy is NOT
> validated; do not wire to any account. Historical numbers below are kept
> for the record only.

*Research concluded 2026-07-14 (superseded — see banner). Status at the time:
validated in backtest, NOT yet wired to production. All numbers net of 10
bps/side transaction costs, walk-forward over 19 semiannual PIT windows
(2017-H1 → 2026-H1), rolling-5y model re-selection, 110 tradable months.*

---

## The strategy (final specification)

1. **Universe**: all scored names with market-cap coverage (~620, PIT S&P-type
   membership).
2. **Signal**: the production quant composite (parent-selection V4, rolling-5y
   walk-forward construction). Used for exactly one thing:
3. **Screen**: **drop the bottom 25% of names by composite rank.** Hold the
   other ~75% (~465 names). No winner-picking, no rank weighting.
4. **Parent veto** (added 2026-07-14, = the ratified **vpos6_20**): also
   drop any survivor in the bottom-20% per-date percentile of **any parent
   except value and institutional** (vetoes on momentum, quality, growth,
   revisions, insider, short; NaN never vetoes, degenerate parents skipped).
   Leaves ~267 names.
5. **Weighting**: **50/50 blend of equal weight and market-cap weight** per
   name, renormalised.
6. **Rebalance** (updated 2026-07-15, see "Operations" below): **two
   sleeves of half the capital, each fully reformed every 6 months, offset
   by 3 months; no trading between reforms.** (All backtest results in
   this doc were measured on the original monthly-reform construction;
   the cadence studies showed slower reform costs nothing pre-tax.)
7. **Leverage**: none (decision below).

*See "The parent-veto extension" below for the veto evidence and its
data-mining caveat, and `docs/strategy_current_and_roadmap.md` for the
operating summary + research roadmap status.*

## Headline result ($10,000, Jan 2017 → Jun 2026)

| | vpos6_20 (current) | mix_screen25 | SPY | QQQ |
|---|---|---|---|---|
| Final value | **$55,825** | $40,286 | $32,821 | $59,115 |
| CAGR | **20.6%** | 16.4% | 13.8% | 21.4% |
| Sharpe | **1.12** | 0.97 | 0.88 | 1.09 |
| Max drawdown | −27.6% | −27.3% | −28.4% | −34.3% |
| Beta vs SPY | 1.09 | 1.04 | 1.00 | — |
| CAPM alpha | **+5.0%/yr** | +1.8%/yr (t=1.62) | — | — |
| Names held | ~267 | ~466 | 503 | ~100 |

(vpos6_20 numbers carry the in-sample selection caveat below; mix_screen25
and the +1.8%/yr alpha are the fully pre-registered floor.)

Beat SPY on returns in **7 of 10 years**; losses never worse than −0.4pp/yr,
wins up to +11pp (2020). Edge concentrates in down years (2018, 2020, 2022) —
the screen can't make good names better, it keeps the worst names out when the
tide goes out. QQQ won this decade on a concentrated tech bet we deliberately
don't hold.

## Why we believe it: the evidence trail

Three **pre-registered** confirmations (bar declared before each first run):

| test | result | window wins | t-stat |
|---|---|---|---|
| EW screen20 vs EW broad (v1) | Sharpe 0.901 vs 0.824 — PASS | 15/19 | 2.72 |
| cap screen20 vs cap broad (v2) | 0.969 vs 0.932 — PASS | 12/19 | 1.62 |
| mix screen20 vs mix broad (v3) | 0.954 vs 0.896 — PASS | **17/19** | 2.43 |
| parent-veto vqs20 vs mix_screen25 (v4) | 1.006 vs 0.967 — PASS (incl. random-null check) | 13/19 | 2.77 |

Supporting findings (exploratory, mapped not tuned):

- **Depth**: edge lives in dropping the worst 20–30%; decays at 40%,
  *harmful* at 50%+ (deleting the model's neutral zone). 25% = plateau midpoint.
- **The model is a loser-avoider, not a winner-picker**: the excluded bottom-20%
  book earns ~5pp/yr less than market; the incumbent top-decile book *trails*
  SPY; rank-tilting weight toward the model's favorites is significantly
  harmful (−2.3%/yr, t=−3.8) — consistent with hump-shaped OOS quantiles.
- **EW-vs-cap is a regime bet, not alpha**: cap's full-period Sharpe edge is
  entirely 2023–24 mega-cap concentration (EW won 6/10 years incl. 2022).
  The 50/50 blend sits between them in every single year — it hedges the bet.
- **cap-weighted broad book ≈ SPY** (corr 0.997, TE 1.4%/yr) — validates the
  whole pipeline end-to-end.
- Of the +2.6pp/yr over SPY: ~1.0–1.3pp is the screen (validated), the rest is
  EW-half diversification + universe residue. Calibrate forward expectations
  to the validated part.

## The parent-veto extension (v4/v4b, 2026-07-14 — vpos6_20 since ratified as the working spec)

Motivation: the composite is an *average*, so one terrible parent can hide
behind mid-pack strength elsewhere — and a smaller book was wanted without
winner-picking (which the tilt tests proved harmful). Veto rule: within the
screen25 survivors, drop any name in the bottom-X percentile of a veto
parent's per-date cross-sectional rank (NaN never vetoes; degenerate parents
— short pre-2018, revisions pre-2019 — are skipped that date).

**Pre-registered confirmation (vqs20 — veto on quality OR short at 20%, the
only two parents with positive OOS IC):** PASSED all three checks, including
a new one — beating the 90th percentile of 200 *size-matched random null
books* (same number of names dropped per date, persistent random choice,
zero information). vqs20 beat 99% of null draws; the null itself averaged
Sharpe 0.963 ≈ screen25, proving fewer names per se add nothing. Mechanism
confirmed: the ~75 vetoed names lose −6.3%/yr vs screen25 (t=−3.5).

| book | veto parents | names | Sharpe | CAGR | active vs screen25 | null draws beaten |
|---|---|---|---|---|---|---|
| mix_screen25 | (none) | ~466 | 0.967 | 16.4% | — | — |
| **vqs20** (pre-registered) | quality, short | ~391 | 1.006 | 17.4% | +0.9%/yr, t=2.77, 13/19 | 99% |
| vall20 (exploratory) | all 8 | ~225 | 1.021 | 18.4% | +1.9%/yr, t=1.62, 12/19 | 95% |
| vpos4_20 (exploratory) | qual/short/insider/growth | ~307 | 1.079 | 19.6% | +2.8%/yr, t=3.57, 13/19 | 100% |
| **vpos6_20** (exploratory) | all except value, institutional | ~267 | **1.117** | **20.6%** | **+3.8%/yr, t=3.69, 14/19** | **100%** |

Decomposition (leave-one-out on vall20): the **value and institutional
vetoes are the drag** (removing either raises Sharpe); insider, momentum and
short carry the effect; quality is neutral inside the union (redundant with
the others despite working solo). vpos6_20 fixes vall20's flaw — it beats
screen25 in *every* block including 2023-2025, where vall20 lost.

Credibility, stated plainly:
- vqs20 is adoption-grade: parents chosen from prior OOS evidence, bar
  declared before the run, fourth confirmed pass.
- vpos4/vpos6 are **in-sample selections** — chosen by looking at the same
  110 months' diagnostics. The t-stats are inflated by selection; the
  size-matched nulls bound the luck story but cannot remove the selection
  effect. In their favour: solo diagnostics, leave-one-out and subsets all
  tell one coherent story, and excluding the institutional veto has real
  prior support (the factor was independently found degenerate). Excluding
  the *value* veto means "never exclude expensive stocks" — plausibly a
  2017-2026 growth-decade artifact, the weakest link in vpos6.
- Only forward (paper-account) months can promote vpos6 to trusted.

Outputs: `output/loserscreen_veto/` (v4 REPORT.md + verdict),
`output/loserscreen_veto2/` (decomposition + `veto_dashboard.png`).

## The closing battery (2026-07-14) — well confirmed dry, spec confirmed

Four final studies closed the research cycle. Net effect on the spec: zero
changes — which is itself the finding.

**1. Five new veto candidates (v5, pre-registered) — ALL FAILED.**
Net share issuance, asset growth, Sloan accruals, idiosyncratic vol and
MAX5 lottery, each as one extra veto on the base (definitions + per-cell
bar in `loserscreen/__init__.py` v5). Best loser: accruals, Sharpe 1.102
vs base 1.117, beating only 39% of its size-matched null. Two lessons with
teeth: (i) the volatility vetoes are *actively harmful* in this large-cap
universe (solo on screen25: idio_vol t=−3.2, max5 t=−3.6 — the lottery
anomaly is a small-cap effect; here the volatile names are the winners);
(ii) the fundamental flags are already subsumed by the composite + parent
vetoes. The null also showed random ~210-name drops from the base average
Sharpe ~1.11 (q90 ~1.15) — "smaller book, higher Sharpe" happens by luck
constantly, which is why every candidate had to beat it and none did.
→ `output/loserscreen_cands/REPORT.md`.

**2. Sector-concentration diagnostic — benign, one watch item.**
Average sector tilts of the veto book vs broad are within ±4pp, HHI barely
above broad → no sector caps. Watch item: the Communication Services
active weight has trended to **+12pp at the latest 2026 dates** — eyeball
sector actives at each live rebalance; revisit caps only if any sector
persists above ~+10pp. (Data note: the sector map mixes two taxonomies —
"Health Care"/"Healthcare" etc. — pre-existing, left unchanged.)
→ `output/loserscreen_cands/sector_diag.png`.

**3. Brinson-Fachler decomposition — the edge is stock selection, not
sector rotation.** Of vpos6_20's +4.1%/yr gross active vs mix_screen25,
**+3.8%/yr (92%) is within-sector selection (t=4.16)**; sector allocation
is +0.3%/yr (t=0.75, noise). Selection is positive in 10 of 12 sectors.
This is the single strongest rebuttal to the data-mining worry: a mined
artifact has no reason to show broad, uniform within-sector edge. It also
retires the sector-cap idea for good (caps would constrain the noise term).
→ `output/loserscreen_veto2/brinson.png`.

**4. SPY 200d-SMA trend filter (pre-registered) — FAILED, not adopted.**
Halving exposure below the 200-day average passed the Sharpe check (1.223
vs 1.117) and the drawdown check (−21.7% vs −27.6%) but lost **6 of the 8
windows where it acted** — tiny wins in the two slow bears (+0.2pp,
+1.0pp), large losses in every recovery (−3.5 to −6.3pp) — and costs
2.7pp/yr of CAGR ($55.8k → $45.2k). The Sharpe gain is variance shrinkage
riding on one bear (2022). Per the declared rule: a drawdown tool, not a
Sharpe tool — shelved next to the leverage options for a future
risk-preference change. (The binary Faber 0/1 variant was strictly worse
on wealth: $35.2k.) → `output/loserscreen_final/trendfilter.png`.

## Leverage findings (kept in the back pocket, not adopted)

**Decision: run unlevered.** The unlevered book has the highest Sharpe (0.97)
of every configuration tested; leverage scales wealth but always costs some
Sharpe (financing spread) and drawdown. If leverage is ever wanted, the two
vetted options:

| option | mean expo | final value | CAGR | Sharpe | max DD |
|---|---|---|---|---|---|
| **static 1.25× margin** | 1.25 | $50,406 | 19.3% | 0.927 | −33.8% |
| **vol-target rv 1.0–1.5×** | 1.33 | $50,897 | 19.4% | 0.940 | −29.7% |
| (vix 1.0–1.5×, best Sharpe) | 1.21 | $46,977 | 18.4% | 0.980 | −28.1% |

- Vol-target rule: `exposure = clip(20% / trailing-63d-SPY-realized-vol, 1.0, 1.5)`
  at each monthly rebalance (inverse *volatility*, not volume). The original
  0.75-floor version passed its pre-registered bar vs static (Sharpe tie,
  −4.6pp maxDD, 12/19); the floor-1.0 + VIX variants were post-hoc refinements —
  directionally sensible, hold the exact numbers loosely.
- Caveat on rv 1.0–1.5 vs static: its higher terminal value rides on higher
  mean exposure (1.33 vs 1.25); the honest advantages are the shallower
  drawdown and per-unit efficiency, not the +$500.
- Margin financing modelled at fed funds + 1%; retail brokers often charge
  more — re-check the real rate before ever using this. Futures overlay
  (cheap beta) loses to margin (levered alpha) as long as the model's edge
  over SPY exceeds ~0.7pp/yr. Taxable-account delevering realizes gains
  (not modelled).

## Operations: cadence, taxes and sleeves (2026-07-15)

The user's pre-launch concern — monthly reform of ~267 names means
transaction costs, short-term capital-gains tax and constant management —
was studied with a lot-level simulator (HIFO lot selection, ST 24% / LT 15%,
loss carryforward, year-end tax payment, 10 bps/side; dividend taxes and
wash sales ignored on both sides). Findings, each with its script:

**1. Rebalance cadence is free pre-tax** (`loserscreen_cadence_phases.py`,
all reform phases): monthly $556k vs quarterly mean $577k vs semiannual
mean $571k vs annual mean $552k per $100k — equal within noise. The signal
moves slowly; trading it 12x/yr instead of 2x adds nothing. Monthly's only
real advantage is immunity to phase luck. Annual is where decay starts
(worst-phase Sharpe 0.97). Note: the lot simulator reads ~0.02 Sharpe below
the research engine (cash frictions, delisted-proceeds drag) — internally
consistent, use it only for policy comparisons.

**2. Taxes are the dominant friction, and the edge survives them**
(`loserscreen_ops_tax.py`): after full liquidation, every cadence beats
after-tax buy-and-hold SPY ($294k) — monthly $421k, semiannual $487k.
Monthly reform hands ~29% of gains to tax vs SPY's deferred 15%; semiannual
converts most gains to long-term and saves ~$65k per $100k over the decade.
An IRA eliminates the entire issue.

**3. Condensing to ~100 names does NOT reduce friction** (same study):
a sector-stratified top-cap 100-name subset loses ~1.5pp/yr (forfeits the
equal-weight small-name half of the edge) while the tax/cost percentages
barely move. Friction scales with turnover, not name count. Rejected.

**4. Two-sleeve stagger — ADOPTED** (`loserscreen_sleeves_events.py`):
two half-books reformed semiannually, 3 months apart. Mean outcome
identical (mathematically guaranteed), calendar-phase luck spread cut ~4x
($189k → $47k pre-tax, $129k → $33k after-tax per $100k), Sharpe a hair
better, per-dollar tax profile unchanged. Free insurance.

**5. Mid-cycle "emergency exits" — REJECTED** (same study): selling held
names between reforms when decisively excluded (bottom-15% composite or
bottom-10% veto parent — hysteresis vs the 25%/20% reform lines) fired
~110 sells/yr, cost ~0.6pp/yr of wealth, and reintroduced monthly scoring
and trading. Its Sharpe bump (1.09 → 1.14) is the idle-cash-buffer effect
— the same variance-shrink signature as the rejected trend filter, not
skill.

**Adopted operating policy: two sleeves, semiannual reform per sleeve,
no trading in between — four rebalancing sessions a year, each touching
half the portfolio.**

## Known caveats

- Backtest, one decade, one path. Three confirmations are correlated views of
  the same 110 months, not independent discoveries.
- Later refinements (screen25, floor-1.0, VIX dial) were chosen after seeing
  results — plateau-interpolations, not fresh validations.
- The insider look-ahead bug in `factors/utils.py` was fixed 2026-07-14;
  research-path loaders were already clean, so these results are unaffected.
- Forward validation (paper account) is the only remaining evidence that
  counts. Every month not collected is gone forever.

## Reproduce

```
python run_loserscreen_study.py                    # v1 EW (pre-registered)
python run_loserscreen_study.py --books v2         # cap + depth curve
python run_loserscreen_study.py --books mix        # 50/50 blend (pre-registered)
python run_loserscreen_study.py --books tilt       # rank-tilt (exploratory)
python run_loserscreen_study.py --books wmix       # weighting grid (exploratory)
python scripts/loserscreen_yearly.py               # cap-vs-EW by year
python scripts/loserscreen_vs_spy.py               # chosen book vs SPY
python scripts/loserscreen_vs_bench.py             # + QQQ, 3x SPY
python scripts/loserscreen_leverage.py             # 1.25/1.5x margin vs futures
python scripts/loserscreen_voltarget.py            # vol targeting (pre-registered)
python scripts/loserscreen_voltarget2.py           # dial/floor grid (exploratory)
python run_loserscreen_study.py --books veto --out output/loserscreen_veto    # v4 (pre-registered)
python run_loserscreen_study.py --books veto2 --out output/loserscreen_veto2  # decomposition
python scripts/loserscreen_veto_charts.py          # veto dashboard (needs veto2 outputs)
python run_loserscreen_study.py --books cands --out output/loserscreen_cands  # v5 (pre-registered, all FAIL)
python scripts/loserscreen_sector_diag.py          # sector concentration diagnostic
python scripts/loserscreen_brinson.py              # allocation-vs-selection decomposition
python scripts/loserscreen_trendfilter.py          # 200d trend filter (pre-registered, FAIL)
python scripts/loserscreen_ops_tax.py              # lot-level tax/cost study + 100-name condense
python scripts/loserscreen_cadence_phases.py       # reform cadence x phase grid (pre-tax)
python scripts/loserscreen_sleeves_events.py       # two-sleeve stagger + emergency exits
```

Outputs live in `output/loserscreen*/` (each REPORT.md carries its verbatim
pre-registration and verdict) and `output/loserscreen_final/` (charts).
Code: `loserscreen/` package. Window caches: `cache/vixtilt/`.

## Next steps

1. ~~Decide the base construction~~ **Decided (2026-07-14): vpos6_20** —
   knowingly data-mined, vqs20 is the pre-registered fallback if it lags
   forward. ~~Decide operations~~ **Decided (2026-07-15): two sleeves,
   semiannual reform per sleeve, offset 3 months, no mid-cycle trading.**
   Wire it into the portfolio layer at 1.0× — forward months are the real
   test.
2. Start the LLM avoid-list forward ledger (AVOID_RED_FLAG ∪ quant screen is
   the plausible next alpha source; only validatable forward).
3. No further backtest permutations on this dataset — the veto family is
   mapped; anything further is re-slicing the same decade.
