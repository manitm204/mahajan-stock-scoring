# Monte Carlo: does random selection among tied names matter?

*Research note, 2026-09-09. Both scripts read the production EQEFF composite
score panel; neither is wired into the live scoring/portfolio pipeline.*

## The question

At any review date the top of the composite score is a plateau: roughly a
dozen names are tied at `composite_score == 100` (see
`research/autoresearch/candidate.py`'s docstring on `BOOK_SIZE=11`). Two book
constructions pick a subset of that tied pool with a deterministic rule
(momentum-decline eviction for the staggered-sleeve family, a trailing-stop /
time-cap / rank-floor exit for the managed-book family). This asks: if you
replace the deterministic pick with a **uniform random draw from the same
tied pool**, holding everything else (sleeve structure, exit rule, rebalance
cadence) fixed, how much of the strategy's edge survives?

Each script runs N independent random replays and reports them as a
spaghetti chart (median + p10/p90 band) against SPY and QQQ, plus a
CAGR/Sharpe/Sortino/max-drawdown/beta/alpha table.

## The two scripts

**`scripts/monte_carlo_random_book.py`** — staggered-sleeve family
(`--sleeves` sleeves, each holding `--n` names, reviewed every `--hold`
months, evicting `--refresh` names at random per review instead of by
momentum decline). Default (3 sleeves, n=4, hold=4, refresh=3) matches
`21_loopeng_book4_hold4_evict3` from `scripts/generate_strategy_lab_comparison.py`.
`--sleeves` must divide evenly into a sensible stagger for `--hold` (e.g.
hold=4/sleeves=3 or hold=4/sleeves=4 both give genuinely distinct review
phases; hold=2/sleeves=3 does not — two of the three sleeves land on
identical review dates, see the dashboard config history for that bug).

```
python scripts/monte_carlo_random_book.py --n 5 --hold 4 --refresh 5 --sleeves 4 --sims 2500 \
    --out output/monte_carlo_random_book/n5_hold4_refresh5_sleeves4_sims2500.json
```

**`scripts/monte_carlo_managed_book.py`** — single managed book, `--k`
positions, exit rule = 9-month time cap OR 10% trailing stop OR falling into
the worst quartile of the scored universe (1-week minimum hold), entries
drawn at random from the score==100 pool instead of top-k-by-score. Also
supports `--mode all100` (deterministic: hold every name scored exactly 100,
no fixed k, no randomness — a sanity check, not a real book).

```
python scripts/monte_carlo_managed_book.py --mode random --k 5 --sims 200 \
    --out output/monte_carlo_managed_book/k5_results.json
```

## Results

*Updated 2026-09-09 (third pass) — composite scores walk-forward through
2026-09-04 (see "Walk-forward extension" below), the bench_returns
date-alignment bug is fixed (see "Beta/alpha bug found and fixed" below),
and the featured sleeve config's sim count was raised from 500 to 2500 for a
tighter Monte Carlo distribution (numbers barely moved — Sharpe 0.966→0.963,
alpha 0.0510→0.0504 — confirming 500 sims was already stable, this is extra
margin, not a correction). The featured sleeve config also changed from
n=5/hold=2mo/3-sleeves (which had a phase-collision bug — two of the three
sleeves reviewed on identical dates) to n=5/hold=4mo/4-sleeves, a genuinely
staggered config.*

### Staggered sleeves, n=5 / hold=4mo / refresh=5 / 4 sleeves, 2500 sims

| | Portfolio (mean of sims) | SPY | QQQ |
|---|---|---|---|
| CAGR | **16.6%** | 13.9% | 19.2% |
| Sharpe | 0.96 | 0.81 | 0.91 |
| Sortino | 0.95 | 0.73 | 0.99 |
| Max drawdown | -22.0% | -27.0% | -35.4% |
| Beta vs SPY | **+0.81** | 1.00 | 1.13 |
| Alpha vs SPY (ann.) | +5.0% | — | +3.5% |

Median final return 1.78x vs SPY's 1.38x; **89.8% of the 2500 random-draw
sims beat SPY**, 11.1% beat QQQ. Random selection among the tied pool, with
this sleeve/hold/refresh structure, comfortably clears SPY most of the
time — but note the beta is a real, solidly positive ~0.8 (see the bug
writeup below for why an earlier version of this table showed beta ≈ -0.10):
part of the outperformance is just being long the market at slightly less
than 1x beta with a persistently-strong stock-picking pool on top, not a
market-neutral effect.

### Managed book, k=5, cap9/trail10%/rankfloor75%/minhold1wk, 200 sims

| | Portfolio (mean of sims) | SPY | QQQ |
|---|---|---|---|
| CAGR | **10.4%** | 14.4% | 20.1% |
| Sharpe | 0.66 | 0.85 | 0.97 |
| Sortino | 0.74 | 0.79 | 1.03 |
| Max drawdown | -20.1% | -28.4% | -34.3% |
| Beta vs SPY | 0.65 | 1.00 | 1.11 |
| Alpha vs SPY (ann.) | +1.5% | — | +3.9% |

Median final return 0.85x vs SPY's 1.39x — **only 16% of sims beat SPY**.
Opposite conclusion from the sleeve family: with random entries, this
exit-rule-driven single book trails SPY most of the time. Whatever edge the
deterministic top-k entry has in this family looks like it depends on
*which* names get picked, not just on picking from the tied pool at all.
(This config's beta/alpha were never affected by the bug below — its own
return series and benchmark series happen to use a matching, already-correct
labeling convention; verified numerically, see git history for
`scripts/sleeve_beta_robustness.py`.)

## Beta/alpha bug found and fixed (2026-09-09)

The sleeve config's beta was originally reported as **-0.10** (R² ≈ 0.01) —
implausible for a random long-only draw from a large-cap S&P-ish universe.
A dedicated robustness check (`scripts/sleeve_beta_robustness.py`, daily/
weekly HAC regressions, rolling beta, calendar-year splits, downside beta)
found the true beta is **~0.81-0.91**, positive and stable across every
frequency and time window tested, and traced the discrepancy to a real bug:

`research/autoresearch/evaluate.py::compute_portfolio_returns()` labels each
period's return by the date it **starts**; `scripts/run_strategy_sweep.py`'s
`bench_returns()` labeled SPY/QQQ's return by the date it **ends** (plain
`.pct_change()`). `research/walkforward/portfolio.py::benchmark_stats()`
aligns the two series by raw index label, so every period was silently
paired with the wrong month's benchmark return. `research/autoresearch/evaluate.py`
already had its own correctly start-labeled `bench_returns()` (with a
regression test whose docstring says it guards against exactly this —
"a bug that briefly shipped and produced an implausible negative SPY beta"),
but `scripts/monte_carlo_random_book.py` (and several other scripts) were
still importing the older, unfixed duplicate from `run_strategy_sweep.py`.

Fixed by switching the affected scripts to import `bench_returns` from
`research.autoresearch.evaluate` instead: `scripts/monte_carlo_random_book.py`,
`scripts/ablation_monte_carlo_fine_grid.py`,
`scripts/ablation_monte_carlo_random_book.py`. `scripts/generate_strategy_lab_comparison.py`
mixes both return conventions (one leg via `monthly_returns`, one via
`compute_portfolio_returns`) and now computes two separately-aligned
benchmark series, one per leg. Scripts using only `monthly_returns` for
their own return series (`research/loop_engineering/harness.py`,
`scripts/run_strategy_followup.py`, `scripts/monte_carlo_managed_book.py`)
were already correctly self-consistent with the old `bench_returns` and were
**not** changed — flipping them too would have introduced the same bug in
the opposite direction. All 20 tests in `tests/test_autoresearch_evaluate.py`
still pass.

Not yet regenerated: `output/loop_engineering_v3/strategy_lab_comparison.json`
(the `21_loopeng_book4_hold4_evict3` leg was affected the same way; that
research artifact no longer feeds a dashboard page but still has stale
beta/R² if anyone reads the raw JSON).

## Walk-forward extension (2026-09-09)

The composite score cache the Monte Carlo scripts read
(`output/crowding/weight_config_study/comp_10configs.pkl`, built by
`scripts/full_pit_backtest_10configs.py`) was capped at 2026-06-30 by two
things, both fixed:

1. The underlying candidate-score panel
   (`cache/subfactor_expansion/cand_panel_*_monthly_v2.pkl`) only had scored
   dates through June 2026. Extended to 2026-09-04 (latest available price
   data) with `scripts/extend_subfactor_panel.py`, which scores only the new
   monthly dates (PIT, independent per date) and merges them into the
   existing cache instead of rebuilding 11 years of history.
2. `full_pit_backtest_10configs.py` hard-capped applied dates at
   `OOS_END = "2026-06-30"` (imported from `scripts/full_pit_backtest_eqeff.py`,
   a constant shared by a dozen other scripts) and explicitly skipped
   `2026-07-01` as a derivation point. Fixed locally in
   `full_pit_backtest_10configs.py` only — a proper new derivation point now
   fires at 2026-07-01 (5-year trailing window `2021-07-01..2026-07-01`,
   same as every other 6-month step), and the apply-date cap is now the
   panel's own last date instead of the shared hardcoded constant, so the
   shared module and its other consumers are untouched.

Verified before rerunning: `run_selection()` (subfactor selection) and the
10 weight-derivation methods (including EQEFF) are pure functions of the
trailing window's own `ScorePanel` and forward-return dict — no global
state, no reuse of production weights, forward returns truncated to
`d + horizon <= t` before scoring. Confirmed by reading
`research/parent_selection.py::run_selection` and its callees
(`subfactor_performance`, `correlation_matrix`, `inventory`) end to end.
Subfactor selection and parent weights do change window to window (verified
in the derivation log — 8 parents selected every window, weights vary), so
this is a true walk-forward re-derivation, not a fixed-weight extension.

## Other saved configs

Not featured on the dashboard, kept for reference. **Beta/alpha inside these
JSONs predate the bench_returns fix above and are wrong** for every
`monte_carlo_random_book.py` output below (CAGR/Sharpe/Sortino/max-DD/final-
return numbers are unaffected — only beta/alpha use `bench_returns`); rerun
with the same flags to get correct beta/alpha if you need them:

- `output/monte_carlo_random_book/n5_hold2_refresh5_sims500.json` — n=5,
  hold=2mo, 3 sleeves (the phase-collision config — two sleeves land on
  identical review dates at this hold/sleeve-count combo).
- `output/monte_carlo_random_book/n5_hold2_refresh5_sleeves2_sims500.json` —
  n=5, hold=2mo, 2 sleeves (the corrected version of the above).
- `output/monte_carlo_random_book/n5_hold3_refresh5_sleeves3_sims500.json` —
  n=5, hold=3mo, 3 sleeves.
- `output/monte_carlo_random_book/n5_hold4_refresh5_sims500.json` — n=5,
  hold=4mo, 3 sleeves (pre-`--sleeves`-flag run).
- `output/monte_carlo_random_book/n5_hold4_refresh5_sleeves4_sims500.json` —
  superseded by the featured `..._sims2500.json` (2026-09-09, more sims for
  a tighter distribution); beta/alpha in the 500-sim file are already
  correct (post-fix), just kept fewer sims — safe to delete if you want.
- `output/monte_carlo_random_book/results.json` — original book4/hold4/evict3
  defaults, pre-CLI-args version of the script, 200 sims.
- `output/monte_carlo_random_book/sweep/hold{1..8}_sleeves{1..8}.json` — the
  hold-months-vs-sleeve-count sweep (200 sims each), see the sweep chart.

**Not affected by the bug** (already used a self-consistent, already-correct
labeling convention — see the "Beta/alpha bug" section above):

- `output/monte_carlo_managed_book/all100_results.json` — deterministic
  hold-everything-at-100 sanity check.
- `output/monte_carlo_managed_book/k5_value15_results.json` — k=5 with the
  v2 champion's value-parent entry floor (`--min-value-pct 15`) added.

## Caveats

- All runs share `--seed 0` as the base seed (each sim offsets by its index),
  so re-running with the same args reproduces the same draws.
- These are single-strategy-family robustness checks, not new pre-registered
  studies — no forward/OOS claim is being made here.
- The two families aren't apples-to-apples (different rebalance cadence,
  different exit logic, different position count over time for the managed
  book), so "sleeves beat SPY more often than the managed book does" is not
  itself evidence that one family is better — it's evidence about how much
  each family's deterministic rule matters.

## Reproduce

```
# only needed when the panel/composite cache has gone stale again:
python scripts/extend_subfactor_panel.py --end <latest trading date>   # then bump the 3 PANEL_PKL constants
python scripts/full_pit_backtest_10configs.py                          # rebuild comp_10configs.pkl (delete it first to force a rerun)

python scripts/monte_carlo_random_book.py --n 5 --hold 4 --refresh 5 --sleeves 4 --sims 2500 --out output/monte_carlo_random_book/n5_hold4_refresh5_sleeves4_sims2500.json
python scripts/monte_carlo_managed_book.py --mode random --k 5 --sims 200 --out output/monte_carlo_managed_book/k5_results.json
```

Dashboard: `dashboard/pages/6_Monte_Carlo.py` reads both JSONs directly (no
aggregation script — each JSON already carries its own sim curves,
percentile bands, and stats table).
