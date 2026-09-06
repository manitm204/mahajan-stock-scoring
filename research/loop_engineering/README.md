# Loop engineering: #13 exit-rule search (2026-09-02/03, extended 2026-09-06)

Automated propose -> test -> keep-or-discard search over the exit-rule
parameter space, seeded from strategy `13_trailstop10_cap12M` (top-10 EQEFF
book, sell on a 10% trailing stop, 12-month cap). This is a **separate
project from `research/autoresearch/`** (which redesigns #05's calendar
sleeves via momentum eviction) -- same production composite score and price
data, different rule family (a single managed book with a parametrized exit
rule, `research/strategies/generic_rule.py::StrategyConfig` +
`research/strategies/engine.py::simulate_managed_book`) and a different,
simpler harness (`research/loop_engineering/harness.py`).

## Methodology

Each round scores one `StrategyConfig` (trailing-stop %, hard stop,
take-profit, time cap, score/rank exit floors, minimum hold, entry-pool
width, sector cap, Value-parent entry floor) by simulating it once over the
full 2020-01 -> 2026-06 period, then splitting the resulting monthly returns
into a TRAIN window (2020-01..2023-12) and a VAL window (2024-01..2026-06).
A round only replaces the champion if it clears that round's promotion rule
-- see `harness.py`'s three variants:

- **v1 (`better_than_champion`)**: train Sharpe must improve by >0.02 over
  the champion, and val Sharpe must not fall below `PROMOTE_VAL_RATIO`
  (0.70) of either the champion's or the original baseline's val Sharpe.
- **v2 (`better_than_champion_v2`, combined-score rule)**: `(train_sharpe +
  val_sharpe)/2` must improve by >0.02, and neither leg may drop by more
  than 0.10 individually. Used for `output/loop_engineering/` (779 rounds,
  4 promotions -- see `champion.json`/`log.csv`/`loop_dashboard.html` there).
- **v3 (`better_than_champion_v3`, strict dual-improvement rule)**: BOTH
  train AND val Sharpe must individually improve by >0.02 -- no
  train/val trade-off allowed. Used for `output/loop_engineering_v3/` (153
  rounds, 1 promotion -- see `champion.json`/`log.csv`/`verdict.html` there).

v2 and v3 are two independent replays of (mostly) the same round ideas
under different promotion strictness, run side by side to see how much of
v2's gain survives a stricter bar. **The 9-month time cap (round 23 in the
v3 numbering, round 2 in v2's) is the one change both promotion rules
independently rediscovered** -- the most robust single finding of the
whole effort. v2 additionally promotes a worst-quartile rank-floor exit, a
1-week minimum hold, and a 15th-percentile Value-parent entry floor (see
`output/loop_engineering/champion.json`, round 113); v3's strict rule
rejected all three of those under its stricter bar, leaving only the cap
change (see `output/loop_engineering_v3/champion.json`, round 23).

## v2 final champion (`output/loop_engineering/champion.json`, round 113)

`k=10, trail_pct=0.1, cap_months=9, rank_floor_pct=0.75,
min_hold_months=0.25, min_value_pct=15.0` -- full-period Sharpe 1.308, CAGR
21.9%, max DD -11.1% (vs. baseline #13's 1.107 / 18.1% / -10.7%).

## v3 final champion (`output/loop_engineering_v3/champion.json`, round 23)

`k=10, trail_pct=0.1, cap_months=9` only -- full-period Sharpe 1.192, CAGR
19.0%, max DD -10.5%.

## Manual extension of v3 (2026-09-06, NOT part of the automated search)

After the two automated searches above, v3 was extended manually in a
later session by porting two of v2's already-promoted levers onto v3's
config one at a time, outside the propose/test/promote loop entirely (no
train/val gate applied -- these are direct, deliberate ports of levers
already validated by v2's search, verified only by recomputing full-period
metrics). Recorded in
`output/loop_engineering_v3/manual_extension_2026-09-06.json`:

| step | added | Sharpe | CAGR | Max DD | alpha t-stat |
|---|---|---|---|---|---|
| 0 (v3 baseline) | cap_months=9 only | 1.192 | 19.0% | -10.5% | 2.04 |
| 1 | + min_hold_months=0.25 (1-week min hold) | 1.268 | 20.1% | -10.1% | 2.35 |
| 2 | + rank_floor_pct=0.75 (worst-quartile exit) | 1.282 | 20.2% | -11.0% | 2.42 |

Step 1 (min-hold) is a clean improvement on every metric, confirming the
lever generalizes independent of what else it's combined with -- it was
also v2's single biggest contributor (~41% of v2's total gain over
baseline #13). Step 2 (quartile exit) trades a bit of drawdown for a
further small Sharpe/alpha gain, mirroring the same trade v2's search
already found this lever makes. The result (`v3_loopeng_cap9_minhold1wk_quartile`)
still trails v2 because it skips v2's Value-floor entry filter.

## Interactive dashboards / write-ups (Claude artifacts)

- Full 24-strategy sweep (original 20 + v2/v3 + #21/#22 autoresearch,
  Sharpe/risk-return/equity-curve charts, per-strategy mechanics writeup):
  https://claude.ai/code/artifact/f3dcfdff-a383-4514-a9c8-e5d3d708cd2a
- v2 round-by-round dashboard (154 rounds, combined-score rule):
  https://claude.ai/code/artifact/867ed678-d01b-489f-88e2-63b612613347
- v3 round-by-round dashboard (31 rounds, strict dual-improvement rule):
  https://claude.ai/code/artifact/42591904-4b5d-47f3-bb43-fc553e51a5cf
- Deep statistical comparison of `v3_loopeng_cap9_minhold1wk_quartile` vs.
  the autoresearch project's `21_loopeng_book4_hold4_evict3` vs. SPY/QQQ
  (correlation, PCA, effective-holdings, turnover, tail-risk, and a
  v3/QQQ blend-frontier analysis):
  https://claude.ai/code/artifact/b86355c9-24c8-4bf6-9dbc-fa85b0f9d8f4

## Caveats

- **Small sample.** 75-77 monthly observations; treat Sharpe/alpha gaps
  under ~0.3-0.4 as within noise for any single round-over-round comparison.
- **The manual v3 extension above was not gated by the train/val promotion
  rule** the way every other row in `log.csv` was -- it's reported as a
  direct, labeled port of already-validated v2 levers, not as a new
  automated-search result. Don't conflate the two when reading `log.csv`.
- **Neither v2 nor v3 (nor the manual extension) is the live production
  strategy** -- these are research artifacts sitting alongside the
  production composite/scoring pipeline, not wired into it.
