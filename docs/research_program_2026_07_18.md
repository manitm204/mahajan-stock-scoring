# Gap-Filling Research Program — Final Report (2026-07-18)

User-approved scope: do NOT re-run the ~360-config construction search already in the
repo; instead harden the harness, consolidate baselines, test only genuinely novel
signal families under pre-registered bars, and run an adversarial validation battery
on the locked finalist. Objective basis: report pre-tax and after-tax; **after-tax
decisive** (taxable account).

---

## 1. What was audited and fixed (Phase A)

- **Holdout contamination found and disclosed.** `research/ablation/engine.py`
  computes 2024–26 `alpha_holdout` for every grid config and
  `report_full_grid` picked the finalist as *highest holdout alpha* — so the
  "untouched 2024–26 holdout" was in fact consulted during selection. All docs now
  call it a **discounted** check; the only clean gate is forward (paper) months.
- **Stale doc claims corrected**: the insider look-ahead bug is FIXED in production
  (`factors/utils.py` bounds windows at `min(as_of, today)`;
  `tests/test_insider_construction.py` guards it); prices span 2014-06→2026-07
  (736 tickers incl. delisted), not "4 years".
- **Harness additions** (all tested, `tests/test_ablation_engine_lag.py`):
  - `exec_lag_days` on `AblationConfig` — T+N delayed execution.
  - QQQ-relative columns (`beta_qqq`, `alpha_qqq`, `ex_qqq`) on every row +
    generic `benchmark_row()`.
  - `research/ablation/diagnostics.py` — paired block-bootstrap CIs, drop-best
    fragility, rolling alpha.
  - `simulate_config(..., scores=...)` override so alternative signals reuse the
    identical costed pipeline; `keep_series=True` returns the return series.
- **Reproducibility rescue**: the after-tax/leverage/options scripts that generated
  `output/ablation/` artifacts lived only in a session scratchpad; now committed under
  `scripts/finalist_toolkit/` with a README.

## 2. Consolidated baselines (Phase B) — `output/leaderboard/BASELINES.md`

Pre-tax, monthly grid, net 10 bps/side, 2017-01→2026-06:

| | net CAGR | Sharpe | maxDD | alpha vs SPY (t) | alpha search | alpha holdout* |
|---|--:|--:|--:|--:|--:|--:|
| SPY | 13.8% | 0.88 | −28% | — | — | — |
| QQQ | 21.4% | 1.09 | −34% | +5.5% (2.1) | +6.6% | **−0.9%** |
| EW universe (PIT) | 9.9% | 0.62 | −28% | −3.5% (−1.7) | −3.0% | 0.0% |
| CAP universe (PIT) | 13.2% | 0.83 | −30% | −0.8% (−1.4) | −1.2% | −0.1% |
| **Finalist** (25/cap5/capmatch/vix) | **16.3%** | **0.97** | −33% | +2.0% (1.4) | +1.4% | +3.1% |
| Finalist **T+1** | 16.3% | 0.92 | −31% | +1.7% (1.2) | +1.2% | +2.6% |
| Finalist **25 bps** | 15.7% | 0.94 | −33% | +1.4% (1.0) | +0.8% | +2.5% |

\* discounted holdout — consulted during prior selection.

After-tax (13-mo hold, 4 sleeves, HIFO, ST 32%/LT 15%): finalist 13.9% CAGR /
$330k walk-away vs SPY 12.4% / $292k vs **QQQ 19.4% / $508k**. Equal-weighting the
universe was the *worst* baseline of the decade (8.9% after tax) — the cap-weight
tilt inside the finalist matters.

**New facts**: the edge survives T+1 execution (−0.3pp alpha) and 25 bps costs
(−0.6pp) — it is not a same-close or cost artifact. QQQ's alpha was entirely
search-era (holdout −0.9%).

## 3. Novel signal families (Phase C) — pre-registered, all FAILED

`output/novel_signals/PREREGISTRATION.md` (bars written before any run: search-alpha
t≥1.5, both sub-eras ≥0, Sharpe ≥ level-composite reference; construction locked).

10 configs: Δ-composite (k∈{3,6} × weight {0.3,0.5,1.0}) and weakest-link/product
interactions (value∧momentum, quality∧revisions). **Zero passed.**

- Pure score-momentum is decisively harmful: −2.5%/yr search alpha, Sharpe 0.68.
- value∧momentum AND-selection: −2.1 to −3.5%/yr (value traps are not fixed by
  requiring momentum in this universe; you just concentrate in a weird corner).
- quality∧revisions matched the reference Sharpe (0.96–0.97 vs 0.95) at **35% lower
  turnover** but with search-alpha t ≈ 0.4 — not promotable; noted as the one lead
  worth a pre-registered second look if turnover ever becomes binding.

Multiple-testing ledger for this program: **+10 configs (Phase C) + 6 neighbors +
3 sensitivities (Phases B/D) = 19 new runs** on top of the repo's ~360 prior configs.

## 4. Adversarial validation of the finalist (Phase D) — `output/validation/VALIDATION.md`

- **Search-window (pre-2024) alpha t = 0.81 — not statistically significant.**
  Bootstrap 90% CI on search alpha: −1.7%..+4.6% (P[alpha<0] = 23%).
- Full-period bootstrap: alpha CI −0.4%..+4.6%, P[alpha<0] = 8% — suggestive only.
- Fragility: +2.5%/yr excess falls to +1.3% without 2023, +1.2% without the best 3
  months. Not one-trade-driven, but 2023 carries half the edge.
- Consistency is the strongest evidence: rolling 36-mo alpha positive in **96%** of
  windows (min −0.9%).
- Stress: 2018Q4 −1.3%/yr, COVID crash **+15.1%/yr**, calendar-2022 **−5.8%/yr**,
  2022–23 sub-era +2.2%/yr. The "Era-2 edge" is really a **2023–26 edge**; the 2022
  bear itself was lost.
- Parameter plateau: neighbors (top20/30, cap/ewcap, no-tilt) all Sharpe 0.91–0.97,
  search alpha spread 1.9pp. **Exception: `sector=none` flips search alpha negative**
  — the cap_match sector overlay is load-bearing. The VIX tilt adds ~+0.5pp.

## 5. Recommended strategies

All three share the signal/selection core: walk-forward composite (rolling-5y,
3M/6M selection), **top 25% / cap5 / cap_match / VIX tilt**, 13-month hold across
4 staggered sleeves, 10 bps/side, T+1-robust. They differ only in the risk wrapper:

1. **Best risk-adjusted (default): the plain finalist book.** After-tax 13.9–14.1%
   CAGR / ~0.91 Sharpe, −32.6% maxDD, beta 0.95. Fails when: momentum/quality factor
   crowding unwinds, or a 2022-style rate-shock bear (it lost that year).
2. **Higher-growth: finalist at 1.25× margin + 30% covered-call overwrite** (needs
   ~$190k+; XSP): in-sample 16.9% / 0.94 Sharpe / −37.8% DD. Without the account size,
   plain 1.25× (16.4% / 0.86 / −40%). Fails when: leveraged drawdown tolerance is
   overestimated, or a melt-up caps the overwrite while margin costs bite.
3. **Drawdown-controlled: 70/30 book-1.25× + XSP call-credit-spread sleeve**
   (floor ~$15–30k): 12.5% / 0.93 / −28.2% DD, beta 0.80. A de-risker, not a
   money-maker; fails in sustained melt-ups (short-call sleeve bleeds).

Options-wrapper numbers are in-sample with ~2 vol events; discount the short-vol
Sharpe. See `output/ablation/PORTFOLIO_SUMMARY.md` + `STARTING_THE_PORTFOLIO.md`.

## 6. Attribution

Prior Brinson work: ~92% of selection edge is within-sector stock selection, not
sector allocation. This program adds: the sector-match overlay and (to a lesser
degree) the VIX tilt are the two construction pieces with positive marginal search
alpha; beta ≈ 1.02 vs SPY (no hidden beta); eff-N ≈ 52 (no concentration engine);
excess is 2023-heavy but not month-concentrated; EW-vs-cap decomposition shows the
book's cap-weight tilt was worth ~+3pp/yr vs equal-weighting this decade.

## 7. Reproduction

```bash
python scripts/baselines_leaderboard.py     # Phase B tables (uses cached WF run)
python scripts/novel_signals_study.py       # Phase C (pre-registered, 10 configs)
python scripts/validation_battery.py        # Phase D battery + neighbors
python -m pytest tests/ -q                  # incl. PIT regression + T+1 tests
# construction grid (pre-existing): python run_ablation.py --stage full
# after-tax grid (pre-existing):    python scripts/finalist_toolkit/tax_rebalance.py
```

## 8. Final critical review

**Robust:** PIT correctness of the harness (regression-tested); the finalist's
*consistency* (96% of rolling windows, T+1- and cost-robust, flat parameter plateau);
the after-tax structural choices (13-mo/4-sleeve dominates monthly rebalancing for a
taxable account); the negative results (Δ-score, factor-AND selection, loser vetoes,
trend filters — all confidently dead).

**Tentative:** the SPY edge itself. Pre-2024 alpha t = 0.81; the honest statement is
"a plausible +1.5–2.5%/yr pre-tax edge vs SPY, consistent but not statistically
demonstrated, with corroborating (but contaminated) 2024–26 alpha of +3.1%."
**QQQ is not beaten** after tax over any full window; the claim is SPY-plus, not
market-best growth.

**What could invalidate:** the 2024–26 corroboration is contaminated by selection;
2023 carries half the excess; the composite's OOS IC is +0.005 (weak); factor
crowding or a rates-driven bear (2022 repeat) turns the edge negative; short-vol
wrappers are undersampled.

**What would raise confidence:** 12+ months of paper-account forward performance
(the pre-registered clean gate); dividend-explicit total-return data (adj_close
proxy currently defers dividend tax — slightly flatters benchmarks after tax, so the
finalist's after-tax SPY margin is, if anything, understated); extending prices past
mid-2026 to create a genuinely untouched holdout.

**Bottom line:** no strategy in this repo demonstrably beats QQQ. The finalist is a
defensible SPY-plus for a taxable account — deploy at 1× (wrapper optional per risk
appetite), gate any leverage on the paper-trading record, and stop mining this
sample: 379 logged configs have extracted what there is to extract.
