# Hierarchical, Regime-Aware Production Factor Model — Research Prototype

**Status:** Approved design, not yet implemented.
**Scope:** Research-only. No production file (`factors/`, `config.yaml`, live scoring) is modified.

## Objective

Build a research-only prototype for a hierarchical, regime-aware factor model that:

- creates a stable core of historically reliable subfactors,
- applies controlled adjustments using recent performance and probabilistic
  VIX-regime evidence,
- avoids rebuilding itself from scratch every month, overreacting to small
  High-VIX samples, or forcing weak subfactors into a parent.

The guiding principle: long-run OOS evidence determines trust; recent evidence
captures strengthening or decay; VIX probabilities provide a limited regime
tilt; shrinkage prevents small-sample overreaction; correlation controls
redundancy; hysteresis prevents excessive feature churn.

## Context: what already exists

This repo already has most of the raw machinery this prototype needs, built
across a series of prior research studies (see `research/walkforward/`):

- `research/walkforward/factor_persistence.py` (run via `run_factor_persistence.py`)
  computes per-calendar-year IC/IR/spread/hit-rate/coverage for every subfactor
  and parent, plus hard VIX-regime buckets, and writes
  `output/factor_persistence/stability_ranking_subfactors.csv` and
  `subfactor_annual_detail.csv` — the two artifacts named in the objective.
  This is a **per-year**, full-history report; it is not structured for
  point-in-time "what did we know as of month M" lookups.
- `research/parent_selection.py` already implements rank → greedy-diversify
  (cross-sectional Spearman R² < 0.60 against every selected sub) → weight
  (∝ positive score, capped 50%, renormalized) selection within a parent
  bucket — this is structurally Section 4 of the objective, just fed a
  different score.
- `research/walkforward/selection.py::select_config` reproduces the live V4
  selection chain on an arbitrary training window with no look-ahead
  (`splits.py` caps training rebalances so no selection forward-return window
  crosses the test boundary).
- `research/walkforward/compose.py` has the production-faithful composite
  blend (`frozen_composite`, `build_parent_panel`, `ic_ir_weights`).
- `research/walkforward/vix_overlay.py` already does **hard**-bucket
  VIX-regime parent-weight tilting (full-tilt / conservative / positive-only)
  with a water-fill capping helper.
- `research/walkforward/splits.py` already provides rolling-5Y / rolling-2Y /
  expanding train-window policies and semiannual test windows — the existing
  "rolling5y wins" finding (see project memory) is exactly variant A below.
- `research/walkforward/portfolio.py` / `analysis.py` already compute the
  full CAGR/Sharpe/Sortino/max-drawdown/SPY-relative/turnover metric suite
  used by every walk-forward report in this repo.

**This prototype is therefore an extension of that lineage, not a from-scratch
build.** Nothing existing is modified; new files are added alongside it. The
genuinely new pieces are: smooth/probabilistic VIX regimes (replacing today's
hard cutoffs), shrinkage-toward-long-run, the Predictive/Reliability/
Production scoring formulas, the CORE/REGIME_DEPENDENT/WATCHLIST/EXCLUDED
classification, monthly-evidence-with-quarterly-hysteresis membership state,
and capped 70/30 parent-weight smoothing — plus a new walk-forward harness
that compares all of it against the existing baseline.

## Architecture

New files, one per concern, following the existing one-file-per-concern
convention already used in `research/walkforward/` (`selection.py`,
`compose.py`, `splits.py`, `vix_overlay.py` each own one piece):

| File | Responsibility |
|---|---|
| `research/walkforward/regime_probability.py` | Smooth VIX regime membership + shrinkage math (pure functions, no panel dependency) |
| `research/walkforward/regime_aware_evidence.py` | Monthly PIT evidence cache + `as_of(cutoff)` slicing (long-run, recent-24M, regime-bucketed) |
| `research/walkforward/regime_aware_scoring.py` | Predictive/Reliability/Production scores + CORE/REGIME_DEPENDENT/WATCHLIST/EXCLUDED classification |
| `research/walkforward/regime_aware_selection.py` | Per-parent subfactor selection (reuses `parent_selection.py`'s R² diversification gate, fed the new Production Score) + intra-parent weights |
| `research/walkforward/regime_aware_parents.py` | Parent utility score, 70/30 base/adaptive weight blend, monthly/quarterly change caps, hysteresis state machine |
| `research/walkforward/regime_aware_walkforward.py` | The continuous monthly-stepping A/B/C/D loop, snapshotting `FrozenConfig` at each semiannual OOS boundary |
| `run_regime_aware_study.py` | Entry point script, matching `run_vix_overlay_study.py` / `run_factor_persistence.py` |

Outputs land in `output/regime_aware/` with a `REGIME_AWARE_REPORT.md`,
matching the existing `output/vix_overlay/VIX_OVERLAY_REPORT.md` style.

**No existing file is modified.** `factors/parent_selection_v4.py` and
`config.yaml` are untouched; this is purely additive research code.

## 1. Evidence Engine + Smooth VIX Regimes

### Monthly PIT cache

`regime_aware_evidence.build_monthly_cache(panel, matrix, vix)` produces one
row per `(sub_factor, rebal_date)` — the panel is already monthly, so this is
the finest useful grain. Reuses `factor_persistence.py`'s `_signal_ic` /
`_spread` / `_coverage` helper logic, but keeps each month's raw observation
instead of collapsing it into an annual mean:

```
sub_factor, date, vix_level, ic_3M, ic_6M, spread_3M_raw, spread_6M_raw, n_names
```

`as_of(cache, cutoff)` derives everything PIT-safe, using only
`date <= cutoff`:

- **long_run**: expanding mean/std of `mean_ic = (ic_3M + ic_6M) / 2` from
  panel inception (`DATA_START = "2015-06-30"` in `splits.py`) → cutoff. This
  is deliberately the same starting point the existing rolling-5Y baseline's
  training window uses, **not** `FIRST_TEST_YEAR = 2017` (which marks when
  OOS *test* windows begin, not when evidence should start accumulating) —
  anchoring long-run evidence at 2017 would leave zero history at the very
  first 2017-H1 test boundary.
- **recent_24M**: mean over the trailing 24 months ending at cutoff, or
  however many months are available if fewer than 24 have elapsed since
  panel inception (degrades gracefully to ≈`long_run` in the first two years,
  same underlying data).
- **annual rollup**: group by calendar year (2015 is a partial year, same
  convention `splits.py` already uses for the partial 2026 year) → % positive
  years, and a year-level persistence IR (mean of annual IC / std of annual
  IC — the same definition already used as `persistence_ir_3M` in
  `factor_persistence.py`).

### Smooth VIX regime membership

`regime_probability.py` replaces the hard `vix < 15 / 15–25 / > 25` cutoffs
in `vix_regime_study.py` with a soft 3-way split using two logistic sigmoids
centered at the same boundaries (15, 25), transition width `w = 4`:

```
s1 = sigmoid((vix - 15) / w)
s2 = sigmoid((vix - 25) / w)
p_low = 1 - s1
p_med = s1 - s2
p_high = s2
```

This always sums to 1 and is always non-negative (`s1 >= s2` since 15 < 25
and sigmoid is monotonic). A value exactly at a boundary splits ~50/50
across it; a value at the bucket center (e.g. VIX=20) gets >90% into that
regime. `w` is a named constant, adjustable.

### Shrinkage

For each regime `R`, using probability-weighted sums over all
`date <= cutoff`:

```
n_eff_R    = Σ_d p_R(vix_d)                              # effective sample size
regime_ic_R = Σ_d p_R(vix_d) · ic_d / n_eff_R
λ_R         = n_eff_R / (n_eff_R + k)                     # k = 24 default
shrunk_ic_R = λ_R · regime_ic_R + (1 - λ_R) · long_run_ic
```

### Expected IC

```
expected_ic(cutoff) = 0.50 · long_run_ic
                     + 0.25 · recent_24M_ic
                     + 0.25 · Σ_R p_R(vix_cutoff) · shrunk_ic_R
```

## 2. Scoring & Classification

### Per-subfactor metrics (all derived from Section 1)

- Expected IC (blended formula above)
- IC-IR = long-run mean_ic / std of the **monthly** per-date IC series, both
  computed over the same panel-inception → cutoff window as `long_run`
- Q5-Q1 spread = long-run mean annualized spread
- Hit Rate = long-run fraction of months with IC > 0
- Positive-Year % = fraction of calendar years with positive mean IC
- Persistence-IR = mean(annual IC) / std(annual IC) — year-level stability,
  distinct from the monthly IC-IR above
- Spread-Consistency = fraction of years whose Q5-Q1 spread shares the sign
  of the long-run mean spread
- Coverage = long-run mean fraction of universe scored
- Sample-Confidence = percentile rank of months-of-history-available

### Formulas

Within each parent, percentile-rank each metric across that parent's
candidates (reusing `parent_selection.py`'s existing `_pct_rank` helper
verbatim):

```
Predictive  = 0.40·rank(ExpectedIC) + 0.25·rank(IC-IR) + 0.25·rank(Q5-Q1) + 0.10·rank(HitRate)
Reliability = 0.35·rank(PosYear%) + 0.25·rank(PersistenceIR) + 0.20·rank(SpreadConsist)
            + 0.10·rank(Coverage) + 0.10·rank(SampleConf)
Production  = Predictive × Reliability
```

### Direction preservation

Percentile rank alone cannot guarantee a materially negative Expected IC is
never rescued (the worst of a bad bunch still ranks somewhere within [0,1]).
So selection eligibility is gated **before** ranking: a candidate is only
eligible if `ExpectedIC > 0` at the current cutoff — mirroring the existing
`if not (ic[c] > 0): stop` rule already in `parent_selection.py`.

### Classification

Absolute (not parent-relative), generalizing the existing
`PERSISTENT` / `PERSISTENTLY_NEGATIVE` / `NOISY` / `MIXED` flags in
`factor_persistence.py` and folding in the `regime_dep` column that
framework already computes:

| Flag | Rule |
|---|---|
| `EXCLUDED` | long-run mean IC < 0 **and** positive-years ≤ 40% **and** ≥3 years of history |
| `CORE` | long-run mean IC ≥ 0.01 **and** positive-years ≥ 60% **and** regime-dependence ≤ 0.03 |
| `REGIME_DEPENDENT` | long-run mean IC ≥ 0 but regime-dependence > 0.03 |
| `WATCHLIST` | everything else (mixed sign, thin history, near-zero long-run IC) |

`regime-dependence` = std across the 3 regimes of the **raw, unshrunk**
`regime_ic_R` (not `shrunk_ic_R`) — shrinkage pulls thin-sample regimes
toward the long-run mean, which would artificially understate how much a
factor's performance actually swings by regime. This mirrors the `regime_dep`
column `factor_persistence.py` already computes from unshrunk regime buckets.

Thresholds are named constants, adjustable. `EXCLUDED` subfactors are never
eligible for selection even if `ExpectedIC` briefly turns positive.
Everything else is eligible subject to the `ExpectedIC > 0` gate.

## 3. Subfactor Selection, Parent Weighting & Hysteresis

### Subfactor selection within a parent

Reuses `parent_selection.py::select_subfactors()`'s R²-diversification loop,
ranking by **Production Score** and gating on `ExpectedIC > 0` instead of the
old blended Sub-factor Score / raw `mean_ic`. One new rule: after tentatively
adding a candidate, rebuild the parent composite including it and recompute
its Expected IC / IC-IR / Q5-Q1; if any drops by more than a **10% relative
tolerance** versus the parent without that candidate, reject it and stop.

### Weighting

Same proportional-to-positive-score / cap-50% / renormalize mechanism as
today (`parent_weights()`), plus a new **10% minimum-weight floor**: any
selected sub whose proportional share would fall below 10% is floored to 10%
and the set renormalizes.

### Parent utility score

Identical Predictive×Reliability construction as Section 2, computed on the
constructed parent composite, ranked across the 8 parents.

### Base weight (the "70%")

Reuses the existing rolling-5Y machinery unmodified: `select_config()` +
`ic_ir_weights(cap=0.25)` on a trailing 5-year training window ending at
cutoff. This is exactly variant A's construction.

### Adaptive weight (the "30%")

Water-fill by Parent Utility (reusing the `_water_fill` helper already in
`vix_overlay.py`), restricted to parents with positive Expected IC, capped
at 25%, using recent-24M + current probabilistic-VIX evidence.

### Blend & guardrails

```
raw = 0.70·base + 0.30·adaptive                                  # cap 25%, water-fill renormalize
if expected_ic[parent] <= 0: raw[parent] = 0, redistribute        # no allocation to negative-IC parents
clip to [prev_month_w - 2pp, prev_month_w + 2pp]                  # monthly cap
clip to [quarter_start_w - 5pp, quarter_start_w + 5pp]            # quarterly cap
water-fill renormalize to sum to 1
```

`prev_month_w` is the realized weight from the immediately preceding month.
`quarter_start_w` is the realized weight snapshotted once at the start of
the current **calendar quarter** (Jan/Apr/Jul/Oct) and held fixed as that
quarter's reference point — so three consecutive ±2pp monthly moves in the
same direction (up to ±6pp) are still capped at ±5pp by the quarterly band
by the third month. The same calendar-quarter boundaries gate subfactor
membership changes below.

The 2–3% diversification floor is **not** applied to the primary model — it
is an optional `min_weight_floor` parameter, run as separately labeled
variants (`C+2%floor`, `C+3%floor`) in the walk-forward comparison.

### Hysteresis (subfactor membership)

New state carried month-over-month across the whole 2015→2026 simulation —
nothing today persists selection state between windows.

- Each `(parent, subfactor)` tracks `above_streak` / `below_streak`
  counters, updated monthly based on whether the one-shot greedy algorithm
  would currently select it and whether `ExpectedIC > 0`.
- **Entry**, evaluated only at quarter boundaries: a candidate joins only if
  it has been eligible-and-selectable for **2 consecutive months**.
- **Exit**, evaluated only at quarter boundaries: a member leaves if
  `ExpectedIC < 0` or its score fell below the exit threshold for **3
  consecutive months**.
- **Immediate exit** (bypasses hysteresis): a sign-inversion flag
  (`POSSIBLE_SIGN_INVERSION`, reusing the existing `horizon_flag` logic) or
  coverage collapsing below 50%.
- Between quarter boundaries, the selected **set** is frozen, but
  intra-parent **weights** among already-selected subs still update monthly
  as evidence updates — only the set is quarterly-sticky.

## 4. Walk-Forward Harness & Deliverables

### Structure

Variants B/C/D run as **one continuous monthly simulation** starting at
panel inception (`DATA_START = "2015-06-30"`) through 2026-06 — `recent_24M`
simply degrades to a shorter window in the first two years, as noted in
Section 1 — carrying hysteresis state and previous weights forward
month-over-month — unlike the existing per-window `select_config`, which
reselects from scratch each window with no memory. At each of the same 19
semiannual boundaries used by every existing regime study
(`semiannual_policy_splits("rolling5y")`), the current
`(sub_weights, parent_weights)` is snapshotted into a `FrozenConfig` and
applied to that window's unseen test rebalances via the existing,
**unmodified** `frozen_composite()` — so no look-ahead is possible and the
OOS scoring machinery is identical to every prior study in this repo.

### Variants

| Variant | Definition |
|---|---|
| **A** | Existing rolling-5Y baseline — calls `select_config()` unmodified per window. No new code. |
| **B** | Stable, no VIX — monthly loop, `ExpectedIC = long_run` only, hysteresis on, capped weight changes on. |
| **C** | Full proposed model (Sections 1–3). |
| **D** | Dynamic-monthly ablation — C's machinery, but `ExpectedIC = recent_24M` only (no long-run anchor, no shrinkage), hysteresis disabled (membership re-evaluated every month), no month-over-month weight cap. |
| **C+2%floor / C+3%floor** | C with the diversification floor engaged, run as separate labeled variants. |

Six named runs total, plus one `k=12` sensitivity note on C — a small
predefined grid, not a sweep.

### Metrics per window

Reuses `analysis.composite_ic`, `analysis.quantile_analysis`,
`pf.simulate(...).metrics` unmodified — the exact fields `vix_overlay.py`
already reports: IC/IC-IR, Q1-Q5 profile, Q5-Q1 spread, hit rate,
monotonicity, CAGR, Sharpe, Sortino, max drawdown, SPY-relative
alpha/IR/relative-drawdown, portfolio turnover. Plus two new churn metrics
computed from the state log: mean absolute month-over-month parent-weight
change, and subfactor entry/exit event counts per year.

### Deliverables → files

| # | Deliverable | Output |
|---|---|---|
| 1 | Master subfactor evidence table | `output/regime_aware/subfactor_evidence_latest.csv` (+ full monthly cache CSV) |
| 2 | VIX probabilities + shrunk regimes | Table in report, computed at latest cutoff |
| 3 | Selected/rejected subs + reasons | `output/regime_aware/selection_decisions.csv` (same shape as `parent_selection.py`'s `decisions`) |
| 4 | Intra-parent weights & formulas | `output/regime_aware/parent_formulas.csv` |
| 5 | Parent evidence/utility/weights | `output/regime_aware/parent_evidence.csv` |
| 6 | Monthly/quarterly weight & membership history | `output/regime_aware/state_history.csv` + churn chart |
| 7 | Walk-forward comparison | `output/regime_aware/REGIME_AWARE_REPORT.md` (per-window + full-period tables, A/B/C/D/+floor) |
| 8 | Recommendation | Narrative section closing the report |

## Out of scope

- Any modification to `factors/`, `config.yaml`, or live scoring.
- Wiring this model into `run_scoring.py` / production execution.
- A parameter sweep beyond the small grid listed above.
- Monthly-cadence OOS *performance measurement* (performance is reported on
  the existing 19-window semiannual grid for comparability with prior
  studies; only internal evidence/state updates monthly).
