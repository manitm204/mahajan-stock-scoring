# Finalist toolkit — after-tax / leverage / options companion scripts

Rescued 2026-07-18 from the 2026-07-17/18 session scratchpad (they generated the
artifacts in `output/ablation/` — `PORTFOLIO_SUMMARY.md` §7 references them — but had
never been committed to the repo, which made those results non-reproducible).

Preserved as-is (paths may assume repo root as CWD; run with `python scripts/finalist_toolkit/<x>.py`):

| Script | Produces |
|---|---|
| `tax_rebalance.py` | Lot-level after-tax simulator (13-mo hold, 4 sleeves, HIFO, ST 32%/LT 15%) — core of `tax_ablation_13mo_4sleeve.csv` |
| `final_stats.py` | Finalist metric tables (`finalist_metrics.md`) |
| `start2020.py` | 2020-start comparison + `start2020_curve.png` |
| `leverage_kelly.py`, `lev15_compare.py`, `lev15_aftertax.py` | Leverage/Kelly sweeps + curves |
| `covered_call.py`, `put_hedge.py`, `credit_spread_sweep.py`, `spread_mgmt.py`, `options_variants.py`, `combined_credit.py` | Options overlay studies (§5 of PORTFOLIO_SUMMARY) |
| `mini_ablation.py`, `simple_ablation.py`, `effn_table.py`, `hz_experiment.py`, `tax_picks.py`, `r2.py`, `impact_delisting_fix.py` | Supporting one-offs |
