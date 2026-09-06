# Autoresearch program: top-10 EQEFF holding/rebalancing strategy

A Karpathy-style propose -> evaluate -> log loop for iterating on how a fixed
composite score (production EQEFF, per-sector percentile, fully PIT) gets
turned into a traded top-10 book. This replaces the one-shot 20-strategy
sweep in `research/strategies/` (rules.py / engine.py / run_strategy_sweep.py)
with a structure a research agent can iterate on unsupervised, with a hard
boundary around what it's allowed to touch.

## The boundary

**Mutable, by the research agent:** `candidate.py` only. One function,
`generate_targets(context) -> pd.DataFrame`. Any ranking, selection,
weighting, holding-period, rebalancing, risk-management, or exit idea goes
here.

**Fixed, never touched by the agent:**
- `evaluate.py` -- data loading, dev/val/holdout boundaries, T+1 execution,
  10bps/side costs, return reconstruction, metric definitions, the
  research-score formula, and the test gate.
- `tests/test_autoresearch_evaluate.py` -- correctness/leakage tests that
  must pass before evaluate.py will print a score.

The interface between them is deliberately narrow: `candidate.py` receives a
frozen `Context` with exactly three fields (`rebal_dates`, `comp`, `k`) --
no prices, no returns, no performance numbers of any kind, for any period.
It returns a long-format weight table; `evaluate.py` treats that table as
the only trustworthy output and independently recomputes everything else
(trades, costs, T+1 execution, returns, metrics). A candidate cannot see how
well it performed, so there's no channel for it to overfit to results it was
never shown -- holdout included.

## Data window and splits

Fixed window: 2020-01 -> 2026-06 (78 monthly PIT rebalance dates, same cache
as the original sweep: `output/crowding/weight_config_study/comp_10configs.pkl`
+ `cache/subfactor_expansion/cand_panel_2015-06-30_2026-06-30_monthly_v2.pkl`).

| Split    | Range                 | Role                                         |
|----------|-----------------------|-----------------------------------------------|
| dev      | 2020-01 -> 2022-12    | primary fit/iterate signal                    |
| val      | 2023-01 -> 2024-12    | out-of-sample check within the same loop      |
| holdout  | 2025-01 -> 2026-06    | **locked**. Diagnostic only. Never gates decisions |

Annual cuts of the holdout may be printed for a human to eyeball regime
behavior, but no single year is a pass/fail gate -- only `research_score`
(dev/val) decides whether a candidate is an improvement.

## research_score

```
research_score = min(dev_ir, val_ir)      # information ratio vs SPY
```

Worst-of, not average-of: a candidate that looks great in one era and
mediocre in the other is not preferred over one that's consistently decent
in both. This is the single number the loop optimizes; `sharpe`, `cagr`,
`beta`/`alpha` (both vs SPY, over the same dev+val window), `max_dd`,
`alpha_tstat`, `turnover` are reported alongside for context but are not
the objective.

## Loop protocol

1. Propose a change to `candidate.py` (and only that file).
2. Run `python -m research.autoresearch.evaluate`.
   - It runs the fixed test suite first. Any failure aborts with no score.
   - It then prints one JSON line: `research_score`, `dev_ir`, `val_ir`,
     `sharpe`, `cagr`, `beta`, `alpha`, `max_dd`, `alpha_tstat`, `turnover`,
     plus a nested `_diagnostic_only_not_gated` block with holdout numbers.
   - It appends one row to `results.tsv` (timestamped, hashed to the exact
     `candidate.py` content that produced it).
3. Decide accept/reject/iterate using `research_score` (and `dev_ir`/`val_ir`
   individually, to catch a candidate that's only winning by juicing one
   split). **Never read `_diagnostic_only_not_gated` to make this decision.**
4. Keep or revert `candidate.py` accordingly; go to 1.

Every run is logged in `results.tsv` regardless of outcome, so the search
history is auditable after the fact -- including rejected candidates, which
matters for noticing if a later "win" is just re-finding a config already
tried and discarded.

## Safety invariants (enforced by tests, not convention)

- `Context` has exactly the fields `rebal_dates`, `comp`, `k` -- checked by a
  structural test so a future edit to `evaluate.py` can't quietly widen the
  interface without the test suite noticing.
- `context.comp` is a read-only `MappingProxyType` of read-only `pd.Series`
  -- a candidate cannot mutate shared state and have it leak into a later
  call or another candidate's run.
- `generate_targets` output is validated independently: dates must come from
  `context.rebal_dates` (nothing later can be smuggled in), weights must be
  non-negative and sum to <= 1.0 per date (long-only, no leverage).
- T+1 execution and the transaction-cost model live only in `evaluate.py`'s
  `compute_portfolio_returns`; `candidate.py` has no access to prices at all,
  so it cannot even attempt to reason about intraday/close timing.
- `candidate.py` never imports `evaluate.py` (checked by a test), so it has
  no name-based handle to monkeypatch fixed constants like the cost rate or
  the holdout boundary.

## Baseline candidate (current `candidate.py`)

`05_sleeves_3M_monthly` from the original sweep: top-10 by composite score,
three overlapping monthly sleeves each held untouched for 3 months, equal
weight (1/30 per name once fully ramped), no exit rule. See `results.tsv`
for its dev/val/holdout numbers under the corrected (T+1, cost-independent
reconciliation) harness -- these differ slightly from the original
`output/strategy_sweep/summary.csv` row of the same name because that sweep
executed at the same close used to rank names (fixed 2026-09-05 in
`research/strategies/engine.py`'s `simulate_calendar_sleeves`, and never
repeated in this harness in the first place).
