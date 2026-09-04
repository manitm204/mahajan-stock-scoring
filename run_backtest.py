"""Mahajan Hedge Fund — Layer 3 (Backtesting) entry point.

Replays the Layer 2 composite through history with strict point-in-time
discipline and reports whether the top-ranked longs outperform, the bottom-ranked
shorts underperform, and the strategy beats SPY after realistic costs.

Examples:
    python run_backtest.py                                   # default run
    python run_backtest.py --strategy top5_strict
    python run_backtest.py --strategy market_neutral --rebalance weekly
    python run_backtest.py --strategy long_only_top5 --no-transaction-costs
    python run_backtest.py --start-date 2022-01-01 --end-date 2026-06-01

Default:
    --start-date 2022-01-01 --rebalance monthly --strategy top5_entry_top10_exit
"""
from __future__ import annotations

import argparse

from backtesting import BacktestConfig, run_backtest
from backtesting import metrics as bt_metrics
from backtesting import plots as bt_plots
from backtesting import reports as bt_reports
from backtesting.portfolio_rules import STRATEGIES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mahajan Hedge Fund — Layer 3 point-in-time backtest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--start-date", default="2022-01-01", help="Backtest start (YYYY-MM-DD)")
    p.add_argument("--end-date", default="2026-06-01", help="Backtest end (YYYY-MM-DD)")
    p.add_argument("--strategy", default="top5_entry_top10_exit", choices=sorted(STRATEGIES),
                   help="Strategy variant to run")
    p.add_argument("--rebalance", default="monthly", choices=["weekly", "monthly", "quarterly"],
                   help="Rebalance frequency")
    p.add_argument("--transaction-costs", action=argparse.BooleanOptionalAction, default=True,
                   help="Apply transaction + borrow costs (--no-transaction-costs to disable)")
    p.add_argument("--transaction-cost-bps", type=float, default=10.0,
                   help="One-way transaction cost in basis points")
    p.add_argument("--short-borrow-cost-annual-bps", type=float, default=300.0,
                   help="Annual short borrow cost in basis points")
    p.add_argument("--weights", default="config", choices=["config", "regime", "ic", "engine"],
                   help="Composite weighting: config (static), regime (VIX-conditional), "
                        "ic (walk-forward Information Coefficient learned weights), or "
                        "engine (Factor Weight Engine: stable baseline + small, "
                        "confidence-scaled, regime-learned overlay)")
    p.add_argument("--ic-min-periods", type=int, default=6,
                   help="Rebalances of IC history before learned weights replace the static "
                        "fallback (only used with --weights ic)")
    p.add_argument("--ic-metric", default="mean", choices=["mean", "ir"],
                   help="IC ranking score: 'mean' (trailing mean IC) or 'ir' "
                        "(information ratio = mean / std of trailing ICs). IR penalizes "
                        "factors that are noisy and is the overfit-resistant default for "
                        "any concentrated bet.")
    p.add_argument("--ic-max-weight", type=float, default=1.0,
                   help="Per-factor cap in the IC composite (e.g. 0.4 limits any single "
                        "factor to 40%%). Excess is redistributed proportionally to the "
                        "remaining factors. 1.0 = no cap.")
    p.add_argument("--ic-min-weight", type=float, default=0.0,
                   help="Per-factor floor in the IC composite (e.g. 0.05 forces every "
                        "learnable factor to at least 5%%). Combined with --ic-max-weight, "
                        "this prevents the walk-forward weighter from over-concentrating "
                        "on a single parent. 0.0 = no floor (default).")
    p.add_argument("--engine-overlay-strength", type=float, default=0.30,
                   help="(--weights engine) max fraction of total weight the adaptive "
                        "overlay may reallocate, before the confidence shrink. Small by "
                        "design so the philosophy baseline stays dominant.")
    p.add_argument("--engine-min-weight", type=float, default=0.02,
                   help="(--weights engine) per-factor floor")
    p.add_argument("--engine-max-weight", type=float, default=0.40,
                   help="(--weights engine) per-factor cap")
    p.add_argument("--engine-smoothing", type=float, default=0.50,
                   help="(--weights engine) EWMA weight on the previous vector "
                        "(0=none, closer to 1 = slower weight transitions)")
    p.add_argument("--engine-min-periods", type=int, default=4,
                   help="(--weights engine) completed periods of history before the "
                        "overlay engages (below this the baseline is used)")
    p.add_argument("--engine-regime-blend", type=float, default=0.50,
                   help="(--weights engine) max weight on the current-regime "
                        "effectiveness view vs the all-regime view")
    p.add_argument("--drop-parents", nargs="+", default=[],
                   metavar="PARENT",
                   help="Parent factor keys to remove entirely (weight forced to 0, "
                        "remaining parents renormalized). Valid keys: momentum, value, "
                        "quality, growth, revisions, insider, short, institutional.")
    p.add_argument("--holdout-months", type=int, default=0,
                   help="Hold out the last N months as an out-of-sample evaluation. "
                        "The IC weighter still walks forward through OOS (PIT-safe); "
                        "IS and OOS metrics are reported and saved separately so the "
                        "OOS segment is an untouched check on the weighting recipe.")
    p.add_argument("--position-weighting", default="equal", choices=["equal", "score"],
                   help="Within-side position sizing: 'equal' (1/N) or 'score' "
                        "(longs sized ∝ composite_score, shorts ∝ 100-composite_score). "
                        "Each side still sums to 1.0 so 50/50 gross is preserved.")
    p.add_argument("--sub-factor-set", default=None,
                   help="Named sub-factor set from config.factor_sets (e.g. 'v2'). "
                        "Default (unset) uses every sub-factor each parent emits "
                        "(V1 / production). A named set filters per-parent "
                        "sub-factors before they roll up into the parent score.")
    p.add_argument("--regime-weights", action="store_true",
                   help="(Alias) use VIX regime-conditional weights; same as --weights regime")
    p.add_argument("--no-reporting-lag", action="store_true",
                   help="Disable the fundamentals/13F availability lag (NOT point-in-time safe)")
    p.add_argument("--output-dir", default="output/backtests", help="Where to write outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Default the sub-factor set to the production config value unless overridden.
    from data.config import load_config
    from factors.pipeline import active_sub_factor_set
    args.sub_factor_set = active_sub_factor_set(load_config(), args.sub_factor_set)
    weights_mode = "regime" if args.regime_weights else args.weights
    config = BacktestConfig(
        strategy=args.strategy,
        rebalance=args.rebalance,
        start_date=args.start_date,
        end_date=args.end_date,
        transaction_costs=args.transaction_costs,
        transaction_cost_bps=args.transaction_cost_bps,
        short_borrow_cost_annual_bps=args.short_borrow_cost_annual_bps,
        reporting_lag=not args.no_reporting_lag,
        use_regime=(weights_mode == "regime"),
        weights_mode=weights_mode,
        ic_min_periods=args.ic_min_periods,
        ic_metric=args.ic_metric,
        ic_max_weight=args.ic_max_weight,
        ic_min_weight=args.ic_min_weight,
        holdout_months=args.holdout_months,
        position_weighting=args.position_weighting,
        sub_factor_set=args.sub_factor_set,
        drop_parents=tuple(args.drop_parents),
        engine_overlay_strength=args.engine_overlay_strength,
        engine_min_weight=args.engine_min_weight,
        engine_max_weight=args.engine_max_weight,
        engine_smoothing=args.engine_smoothing,
        engine_min_periods=args.engine_min_periods,
        engine_regime_blend=args.engine_regime_blend,
        output_dir=args.output_dir,
    )

    weights_tag = config.weights_mode
    if config.weights_mode == "ic":
        weights_tag = f"ic[{config.ic_metric},cap={config.ic_max_weight}]"
    elif config.weights_mode == "engine":
        weights_tag = (f"engine[overlay={config.engine_overlay_strength},"
                       f"smooth={config.engine_smoothing}]")
    holdout_tag = (f" | holdout={config.holdout_months}mo"
                   if config.holdout_months > 0 else "")
    set_tag = (f" | factors={config.sub_factor_set}"
               if config.sub_factor_set else "")
    pos_tag = (f" | sizing={config.position_weighting}"
               if config.position_weighting != "equal" else "")
    print(f"Running {config.strategy} | {config.rebalance} | "
          f"{config.start_date}..{config.end_date} | "
          f"costs={'on' if config.transaction_costs else 'off'} | "
          f"weights={weights_tag}{set_tag}{pos_tag}{holdout_tag} ...")

    result = run_backtest(config)
    metrics = bt_metrics.compute_metrics(result)
    bt_reports.save_csvs(result, metrics, config.output_dir)
    bt_plots.generate_all(result, config.output_dir)
    print(bt_reports.console_summary(result, metrics, config.output_dir))


if __name__ == "__main__":
    main()
