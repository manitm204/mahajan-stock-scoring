# Current Strategy & Research Roadmap

> **⚠ INVALIDATED 2026-07-15 — ghost-member PIT bug.** The backtests behind
> this spec held ~50% non-member "ghost" names (neutral-filled, unvetoable,
> including future index entrants = look-ahead). On PIT-clean rebuilt caches
> the edge largely disappears: vpos6_20 Sharpe 0.892 (was 1.117), +0.3%/yr vs
> SPY, veto uplift t = 1.07; vqs20 fails its pre-registered bar. See the
> banner in `docs/loser_screen_strategy.md` for details. **Do NOT wire to the
> paper account.** Roadmap paused pending a decision on the clean results.

*As of 2026-07-14 (superseded — see banner). This is the operating summary —
the full evidence trail (pre-registrations, verdicts, caveats) lives in
`docs/loser_screen_strategy.md`.*

---

## The current approach (user-ratified working spec)

The book is **vpos6_20**, built in five steps every month:

1. **Universe**: all scored names with PIT market-cap coverage (~620,
   S&P-type membership). Staying with this universe for now — expansion
   considered but deferred.
2. **Composite screen**: rank by the production quant composite
   (parent-selection V4) and **drop the bottom 25%** — keep the top 75%
   (~466 names). The composite is used only to exclude losers, never to
   pick or overweight winners (tilt tests proved that harmful).
3. **Parent veto**: additionally drop any survivor whose parent score is in
   the **bottom 20% percentile (per date) of ANY parent except value and
   institutional** — i.e. vetoes on momentum, quality, growth, revisions,
   insider, short. Missing values never veto; degenerate parents (short
   pre-2018, revisions pre-2019) are skipped. Leaves **~267 names**.
4. **Weighting**: **50/50 blend of equal weight and market-cap weight**,
   renormalised. No rank tilting.
5. **Rebalance** (updated 2026-07-15 after the friction studies): **two
   sleeves of half the capital, each fully reformed every 6 months, offset
   by 3 months** (e.g. sleeve A in Jan/Jul, sleeve B in Apr/Oct). Scoring
   runs only on reform dates; **no trading between reforms** (mid-cycle
   emergency exits tested and rejected). Rationale: cadence is free
   pre-tax, semiannual roughly halves the tax bill vs monthly, and the
   stagger cuts the calendar-phase luck spread ~4x at zero expected cost.
   In a tax-free account (IRA), monthly or quarterly is equally fine.
6. **Leverage: none.** Shelved, not dead. If ever wanted, the two vetted
   back-pocket options are static 1.25× margin or the vol-target dial
   `clip(20% / trailing-63d-SPY-vol, 1.0, 1.5)`.

Backtest record (2017-2026, net): **Sharpe 1.117, CAGR 20.6%,
+3.8%/yr over mix_screen25 (t=3.69), beats 100% of 200 size-matched random
nulls, wins 14/19 walk-forward windows, beats the no-veto book in every
market block.**

**Honesty clause (do not delete):** the value/institutional exclusion was
chosen *after* seeing the single-parent diagnostics — in-sample selection on
the same 110 months, so the t-stat is inflated. The pre-registered,
adoption-grade fallback is **vqs20** (veto on quality+short only, ~391
names, Sharpe 1.006). vpos6_20 is adopted knowingly; forward paper-account
months are the only thing that can promote it to trusted. If it lags vqs20
forward, fall back without ceremony.

## Research cycle results (all planned tests now RUN — cycle closed 2026-07-14)

### 1. New veto candidates — RUN 2026-07-14, ALL FIVE FAILED

Five classic "bad company" flags (net share issuance, asset growth, Sloan
accruals, idiosyncratic vol, MAX5 lottery), each pre-registered as one
additional veto on the current base (v5 in `loserscreen/__init__.py`,
results in `output/loserscreen_cands/REPORT.md`). **Every candidate made
the base worse** — none passed any of the three checks:

| candidate | base+veto Sharpe (base 1.117) | window wins | vs null |
|---|---|---|---|
| accruals | 1.102 (best loser) | 13/19 | beat only 39% of draws |
| net_issuance | 1.050 | 5/19 | 5% |
| asset_growth | 1.025 | 8/19 | 2% |
| idio_vol | 0.966 | 2/19 | 1% |
| max5 | 0.904 | 4/19 | 0% |

What we learned (worth the run):
- **The vol-based vetoes are actively harmful in this universe** (solo on
  screen25: idio_vol t=−3.2, max5 t=−3.6) — the lottery/low-vol anomaly is
  a small/micro-cap effect; in an S&P-type universe, vetoing volatile names
  deletes winners. This kills the "inverse-vol weighting" construction idea
  too, or at least demands it clear a high bar.
- The fundamental flags (issuance, asset growth, accruals) are roughly
  neutral standalone but **worse than random drops** on top of the base —
  the composite + parent vetoes already capture what they capture.
- The size-matched null was essential: random ~210-name drops from the base
  averaged near Sharpe 1.11 with q90 ≈ 1.15, so "smaller book, higher
  Sharpe" happens by luck all the time. No candidate came close to its null
  q90.
- Consequence: the exclusion-signal well on THIS universe and decade is
  now, credibly, dry. The spec stands unchanged (vpos6_20).

### 2. Construction ideas — all resolved

- **Trend/drawdown filter — RUN 2026-07-14, FAILED its bar, NOT adopted**
  (`scripts/loserscreen_trendfilter.py` → `output/loserscreen_final/
  trendfilter.png`; pre-registered: halve exposure when SPY < 200d SMA,
  cash at fed funds, 10bps on switches). Sharpe check passed (1.223 vs
  1.117) and maxDD passed (−21.7% vs −27.6%), **but it lost 6 of the 8
  windows where it actually acted** — small wins in the two slow bears
  (2018-H2 +0.2pp, 2022-H1 +1.0pp), big losses in every recovery/whipsaw
  (2019-H1 −5.3pp, 2023-H2 −4.9pp, 2026-H1 −6.3pp). The full-period Sharpe
  gain is a volatility-shrink effect that costs 2.7pp/yr of CAGR
  ($55.8k → $45.2k) and rides on essentially one bear (2022). Verdict per
  the declared rule: it is a drawdown tool, not a Sharpe tool — shelved
  next to the leverage options for a future risk-preference change, not
  part of the spec.
- **Inverse-vol weighting within survivors**: the alpha lives in down years;
  weighting survivors by inverse volatility leans into where the edge is.
  *Prior weakened by the v5 result (vol-based vetoes were harmful here) —
  low priority now.*
- **Sector-concentration diagnostic — RUN 2026-07-14**
  (`scripts/loserscreen_sector_diag.py` → `output/loserscreen_cands/
  sector_diag.png`): historical tilts are benign (avg active weights vs the
  broad book within ±4pp; HHI only modestly above broad), so **no hard
  sector caps** — but the Communication Services overweight has TRENDED UP
  and sits at **+12pp at the most recent dates** (the veto currently clears
  comm names while flagging financials/health-care). Live-book rule of
  thumb: eyeball sector actives at each rebalance; revisit caps only if a
  single sector's active weight persists above ~+10pp. Data note: the
  sector map mixes two taxonomies (e.g. "Health Care" vs "Healthcare",
  "Financials" vs "Financial Services") — pre-existing production behavior,
  slightly fragments composite peer groups; not changed.
- **Brinson decomposition — RUN 2026-07-14**
  (`scripts/loserscreen_brinson.py` → `output/loserscreen_veto2/brinson.png`):
  of vpos6_20's +4.1%/yr gross active vs mix_screen25, **+3.8%/yr (92%) is
  within-sector selection (t=4.16)** and only +0.3%/yr is sector allocation
  (t=0.75, noise). Selection is positive in 10 of 12 sectors (Cons Disc
  +1.0pp, Financials +0.9pp, Tech/Comm +0.6pp each) — the edge is broad
  loser-avoidance inside sectors, NOT a lucky sector-rotation decade. This
  materially strengthens vpos6_20's credibility despite its in-sample
  selection, and confirms sector caps are unnecessary (capping would only
  suppress the noise component).

### 2b. Operational friction & taxes — RUN 2026-07-15

Lot-level simulation (`scripts/loserscreen_ops_tax.py` →
`output/loserscreen_final/ops_tax.png`): $100k taxable account, HIFO lots,
ST 24% / LT 15%, losses carried forward, 10 bps/side, 2017-2026; policies =
monthly/quarterly/semiannual/annual full reform + changes-only; books =
full 267 names and a condensed 100-name sector-stratified top-cap variant.
Benchmark: after-tax buy-and-hold SPY = $294k.

| after-tax, after-liquidation ($100k start) | 267 names | 100 names |
|---|---|---|
| monthly | $421k | $367k |
| quarterly | $436k | $370k |
| **semiannual** | **$487k** | $372k |
| annual | $451k | $403k |
| changes-only | $427k | $356k |

Findings:
1. **Monthly reform is unnecessary.** Pre-tax finals are statistically flat
   across cadences ($549-632k on 267 names) — the screen signal is slow.
   Slower cadences cut orders from ~300/month to ~350 twice a year and
   leave more gains long-term. Claim honestly: cadence doesn't cost
   performance; don't over-read semiannual's top slot (one path).
2. **Every policy beats after-tax SPY by a wide margin** (+$127k to +$193k)
   — the edge survives full short-term-gains taxation.
3. **Condensing to 100 names does NOT reduce friction meaningfully and
   costs ~1.5pp/yr of return** (top-cap sampling loses the EW small-name
   half of the edge; taxes don't shrink proportionally). Fewer names only
   buys operational simplicity (13-45 orders/mo).
4. Changes-only trading disappoints (churns more than quarterly reform).
5. Taxes are the biggest friction: monthly reform pays ~29% of gains vs
   SPY's deferred 15% — an IRA/tax-advantaged account eliminates this
   entirely and restores the pre-tax column.

**Operating guidance adopted: keep 267 names; rebalance quarterly-to-
semiannually in a taxable account (monthly is fine in an IRA); do not
condense below ~250 for friction reasons.**

Cadence-phase follow-up (2026-07-15, `scripts/loserscreen_cadence_phases.py`,
pre-tax, all reform phases): monthly $556k / CAGR 20.6% vs quarterly mean
$577k (3 phases, $541-622k) vs semiannual mean $571k (6 phases, **$481-671k**)
vs annual mean $552k (12 phases). Means are equal within noise — monthly
buys NO pre-tax return, only consistency: slow cadences carry phase luck
(which calendar months you happen to reform on was worth ±$95k on $100k
historically, unknowable in advance). Annual is where signal decay starts
to show (worst-phase Sharpe 0.966). Verdict: semiannual's after-tax
advantage is real and its phase risk is symmetric luck — guidance stands.

Sleeves + emergency-exit follow-up (2026-07-15,
`scripts/loserscreen_sleeves_events.py`):
- **Two-sleeve stagger ADOPTED**: two half-books reformed 3 months apart.
  Identical mean outcome (mathematically must be), phase-luck spread cut
  ~4x ($189k → $47k pre-tax; $129k → $33k after-tax on $100k), Sharpe a
  hair better (phase diversification), same per-dollar tax profile. Free
  insurance; only cost is tracking two sleeves.
- **Emergency exits REJECTED**: selling held names mid-cycle when
  decisively excluded (bottom-15% composite or bottom-10% veto parent,
  proceeds to cash until next reform) fired ~110 sells/yr — not "a couple
  things" — cost ~0.6pp/yr of wealth ($571k → $540k pre-tax mean), and its
  Sharpe bump (1.09 → 1.14) is the cash-buffer effect, not skill (same
  variance-shrink signature as the rejected trend filter). It also
  reintroduces monthly scoring + ~9 trades/month, defeating the point of
  semiannual. Operating spec: **two sleeves, semiannual per sleeve, no
  mid-cycle trading.**

### 3. Local-LLM report reader (idea stage — design constraints first)

Goal: replace/augment the paid-API LLM overlay with a free local model that
reads filings and flags governance red flags (fits the avoid-list ledger).
Two look-ahead traps, one obvious and one subtle:

- Obvious: the model may only read documents **published on or before** the
  formation date.
- Subtle and unfixable: **any modern LLM already knows these companies'
  outcomes from its training data.** It knows which banks failed and which
  stocks collapsed, regardless of what document it is shown. Therefore an
  LLM overlay can NEVER be honestly backtested on this decade — it is
  forward-ledger-only validation, same as the existing paid overlay.

### 4. Deferred (considered, not now)

- **Universe expansion** beyond the ~620-name SPY-type universe (more
  breadth, stronger down-cap exclusion edges) — user prefers staying with
  the current universe for now.
- **Pre-2016 / GFC backfill** — validation-only if ever done (PIT data
  quality degrades back there).
- **Paid external data** (options IV/skew, borrow fees, transcripts) — only
  after the free ideas are exhausted.

## Bottom line — the best portfolio we know how to build (2026-07-14)

**vpos6_20, unlevered, exactly as specified at the top of this doc.**
Backtest: Sharpe 1.12, CAGR 20.6%, $10k → $55.8k (2017-2026), beta 1.09,
CAPM alpha +5.0%/yr, ~267 names. The fully pre-registered floor under it is
vqs20 (Sharpe 1.006) and mix_screen25 (0.967, alpha +1.8%/yr t=1.62).

Everything else was tested and rejected or shelved, each by a declared rule:
- 5 new veto candidates: all failed; vol-based vetoes actively harmful here.
- Sector caps: unnecessary (tilts benign; Brinson shows 92% of the edge is
  within-sector selection, t=4.16 — the strongest evidence the veto edge is
  real and not sector-rotation luck).
- Trend filter: drawdown tool, not a Sharpe tool (loses 6/8 active windows,
  −2.7pp/yr CAGR) — shelved.
- Leverage: shelved (static 1.25× or vol-target 1.0-1.5× if ever wanted).
- Rank-tilting toward the model's favorites: proven harmful (t=−3.8). The
  composite is a loser-avoider; its entire edge is exclusion.

Operations (settled 2026-07-15 by the friction studies): **two sleeves,
semiannual reform per sleeve offset 3 months, no mid-cycle trading, ~267
names kept** — condensing to 100 costs ~1.5pp/yr and saves no meaningful
friction; monthly rebalancing adds no pre-tax return and roughly doubles
the tax bill; emergency exits are a cash-drag trap; the strategy beats
after-tax buy-and-hold SPY under every policy tested.

The exclusion-signal well on this universe and decade is dry. The remaining
work is not research: **wire vpos6_20 to the Alpaca paper account at 1.0×
with the two-sleeve semiannual structure** (watch item: Comm Services
active weight, currently ~+12pp) and start the LLM avoid-list forward
ledger. Forward months are the only evidence that can still change
anything.

## Standing rules

1. Every confirmatory test gets its bar written down before the first run.
2. Concentrated books must beat a size-matched random null, not just the
   reference — fewer names by luck is not a finding.
3. No more re-slicing of the existing eight parents on this dataset.
4. The paper account is the referee. Wire the current spec at 1.0× and let
   forward months accumulate — that data cannot be bought later.
