"""Mahajan Hedge Fund — Factor Weight Engine entry point.

Walks the monthly rebalance grid with strict point-in-time discipline, scoring
the universe at each date (Layer 2), classifying the market regime (VIX + SPY vs
its 200-day MA), and feeding both into :class:`factors.weight_engine.
FactorWeightEngine`. The engine blends the philosophy-driven baseline weights
(``config.factors.default_weights``) with a small, confidence-scaled overlay that
tilts toward factors whose IC / hit rate / factor spread have held up recently —
learned separately by regime — and smooths the result across rebalances.

The point is NOT to maximize trailing return (that just chases momentum in a
bull market); it is to keep a stable, explainable, regime-aware factor mix.

Note on factor naming: this codebase's parent factors are momentum, value,
quality, growth, revisions (analyst earnings-estimate revisions — the "earnings"
expectation factor), insider, short and institutional. The engine operates over
whatever parents are configured; several (revisions/short/institutional) have
only shallow PIT history, so their measured effectiveness — and therefore their
overlay tilt — stays near zero historically by construction.

Outputs (under --output-dir, default output/weight_engine/):
    weights_timeline.csv   one row per rebalance: regime, confidence, weights
    latest_decision.json   full diagnostics for the most recent date
    latest_metrics.csv     per-factor windowed IC / hit / spread / effectiveness

Examples:
    python run_weight_engine.py
    python run_weight_engine.py --overlay-strength 0.4 --smoothing 0.6
    python run_weight_engine.py --start-date 2023-01-01 --end-date 2026-06-01
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from backtesting import data_loader as dl
from data.config import load_config
from data.db import get_db
from factors.market_regime import regime_from_db
from factors.pipeline import active_sub_factor_set, score_universe
from factors.weight_engine import FactorWeightEngine, WeightDecision


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mahajan Hedge Fund — regime-aware Factor Weight Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--start-date", default="2022-01-01", help="Grid start (YYYY-MM-DD)")
    p.add_argument("--end-date", default="2026-06-01", help="Grid end (YYYY-MM-DD)")
    p.add_argument("--rebalance", default="monthly",
                   choices=["weekly", "monthly", "quarterly"], help="Rebalance frequency")
    p.add_argument("--overlay-strength", type=float, default=0.30,
                   help="Max fraction of total weight the adaptive overlay may move "
                        "from the weakest to the strongest factors (before the "
                        "confidence shrink). Small by design.")
    p.add_argument("--min-weight", type=float, default=0.02, help="Per-factor floor")
    p.add_argument("--max-weight", type=float, default=0.40, help="Per-factor cap")
    p.add_argument("--smoothing", type=float, default=0.50,
                   help="EWMA weight on the previous vector (0=no smoothing, "
                        "closer to 1 = slower transitions)")
    p.add_argument("--min-periods", type=int, default=8,
                   help="Completed periods of IC history before the overlay engages "
                        "(below this the engine returns the baseline). Raised 4→8 "
                        "after the weight-stability sweep (output/weight_stability): a "
                        "longer warm-up lowers turnover and lifts composite IC — the "
                        "thin-history overlay at 4 added churn without predictive power.")
    p.add_argument("--regime-blend", type=float, default=0.50,
                   help="Max weight on the current-regime effectiveness view vs the "
                        "all-regime view (scaled down when same-regime history is thin)")
    p.add_argument("--low-vix", type=float, default=15.0, help="VIX < this = low-vol")
    p.add_argument("--high-vix", type=float, default=25.0, help="VIX > this = high-vol")
    p.add_argument("--sub-factor-set", default=None,
                   help="Named sub-factor set from config.factor_sets (e.g. 'lean'). "
                        "The engine then learns weights from the FILTERED parent "
                        "scores. Unset = the factors.sub_factor_set production default.")
    p.add_argument("--output-dir", default="output/weight_engine", help="Output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    baseline = {k: float(v) for k, v in
                cfg.get("factors", "default_weights", default={}).items()}
    keys = list(baseline.keys())
    if not keys:
        raise SystemExit("No factors.default_weights in config.")
    set_name = active_sub_factor_set(cfg, args.sub_factor_set)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with get_db() as db:
        universe = db.universe_tickers()
        trading = dl.trading_calendar(db)
        matrix = dl.load_price_matrix(db, universe, args.start_date, args.end_date)
        if matrix.empty:
            raise SystemExit("No price data in the requested range.")
        rebal_dates = [d for d in dl.generate_rebalance_dates(
            trading, args.start_date, args.end_date, args.rebalance) if d in matrix.index]
        if len(rebal_dates) < args.min_periods + 1:
            raise SystemExit(
                f"Need at least {args.min_periods + 1} rebalances; got {len(rebal_dates)}.")

        engine = FactorWeightEngine(
            keys, baseline, matrix,
            overlay_strength=args.overlay_strength,
            min_weight=args.min_weight, max_weight=args.max_weight,
            smoothing=args.smoothing, min_periods=args.min_periods,
            regime_blend=args.regime_blend)

        print(f"Walking {len(rebal_dates)} {args.rebalance} rebalances "
              f"{rebal_dates[0]}..{rebal_dates[-1]} over {len(keys)} factors "
              f"(sub_factor_set={set_name or 'full'}) ...")

        timeline: list[dict] = []
        last: WeightDecision | None = None
        for i, d in enumerate(rebal_dates):
            su = score_universe(d, db=db, cfg=cfg, reporting_lag=True,
                                sub_factor_set=set_name)
            regime = regime_from_db(db, d, su.vix,
                                    low_vix=args.low_vix, high_vix=args.high_vix)
            dec = engine.weights_for(d, regime)
            engine.record(d, su.frame, regime)
            last = dec
            row = {"date": d, "regime": dec.regime,
                   "confidence": round(dec.confidence, 4), "applied": dec.applied}
            row.update({k: round(dec.weights[k], 4) for k in keys})
            timeline.append(row)
            print(f"  [{i+1:>2}/{len(rebal_dates)}] {d}  regime={dec.regime:<8} "
                  f"conf={dec.confidence:.2f}  {dec.applied}")

        assert last is not None
        pd.DataFrame(timeline).to_csv(out_dir / "weights_timeline.csv", index=False)
        _write_latest(last, baseline, keys, out_dir)
        print(_console_summary(last, baseline, keys, out_dir))


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def _num(x: float) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 6)


def _write_latest(dec: WeightDecision, baseline, keys, out_dir: Path) -> None:
    metric_rows = []
    for k in keys:
        m = dec.metrics.get(k)
        row = {"factor": k, "weight": _num(dec.weights[k]),
               "baseline": _num(baseline.get(k, 0.0)),
               "delta": _num(dec.weights[k] - baseline.get(k, 0.0)),
               "effectiveness": _num(dec.effectiveness.get(k, float("nan"))),
               "tilt": _num(dec.tilt.get(k, 0.0))}
        if m is not None:
            row.update({
                "ic_recency": _num(m.ic_recency), "hit_recency": _num(m.hit_recency),
                "spread_recency": _num(m.spread_recency),
                "ic_consistency": _num(m.ic_consistency), "n_obs": m.n_obs})
            for w in (1, 3, 6, 12):
                row[f"ic_{w}m"] = _num(m.ic_by_window.get(w, float("nan")))
                row[f"hit_{w}m"] = _num(m.hit_by_window.get(w, float("nan")))
        metric_rows.append(row)
    pd.DataFrame(metric_rows).to_csv(out_dir / "latest_metrics.csv", index=False)

    rs = dec.regime_state
    payload = {
        "date": dec.date,
        "regime": dec.regime,
        "regime_inputs": {"vix": _num(rs.vix), "vix_bucket": rs.vix_bucket,
                          "spy_to_200dma": _num(rs.spy_to_200dma),
                          "spy_above_200dma": rs.spy_above_200dma},
        "confidence": _num(dec.confidence),
        "confidence_terms": {k: _num(v) for k, v in dec.confidence_terms.items()},
        "overlay_strength_effective": _num(dec.overlay_strength_effective),
        "n_periods": dec.n_periods, "n_regime_periods": dec.n_regime_periods,
        "applied": dec.applied,
        "weights": {k: _num(dec.weights[k]) for k in keys},
        "baseline": {k: _num(baseline.get(k, 0.0)) for k in keys},
        "overlay": {k: _num(dec.overlay.get(k, 0.0)) for k in keys},
        "notes": rs.notes,
    }
    (out_dir / "latest_decision.json").write_text(json.dumps(payload, indent=2))


def _console_summary(dec: WeightDecision, baseline, keys, out_dir: Path) -> str:
    rs = dec.regime_state
    vix = f"{rs.vix:.1f}" if rs.vix is not None else "n/a"
    trend = ("above" if rs.spy_above_200dma else "below") if rs.spy_above_200dma is not None else "n/a"
    lines = ["", "=" * 70,
             f" FACTOR WEIGHT ENGINE — {dec.date}", "=" * 70,
             f" Regime:     {dec.regime}  (VIX {vix} / {rs.vix_bucket or 'n/a'}, "
             f"SPY {trend} 200dma)",
             f" Confidence: {dec.confidence:.2f}  "
             f"[{', '.join(f'{k}={v:.2f}' for k, v in dec.confidence_terms.items())}]",
             f" Overlay:    strength {dec.overlay_strength_effective:.3f} "
             f"(= {dec.applied}); periods={dec.n_periods}, regime_periods={dec.n_regime_periods}",
             "-" * 70,
             f" {'factor':<14}{'weight':>9}{'base':>9}{'delta':>9}{'ic_rec':>9}{'hit':>7}",
             "-" * 70]
    for k in sorted(keys, key=lambda x: dec.weights[x], reverse=True):
        m = dec.metrics.get(k)
        ic = f"{m.ic_recency:+.3f}" if m and not math.isnan(m.ic_recency) else "  n/a"
        hit = f"{m.hit_recency:.2f}" if m and not math.isnan(m.hit_recency) else " n/a"
        delta = dec.weights[k] - baseline.get(k, 0.0)
        lines.append(f" {k:<14}{dec.weights[k]:>9.3f}{baseline.get(k, 0.0):>9.3f}"
                     f"{delta:>+9.3f}{ic:>9}{hit:>7}")
    lines += ["-" * 70,
              f" sum weights = {sum(dec.weights.values()):.4f}",
              f" Saved: {out_dir}/weights_timeline.csv, latest_metrics.csv, latest_decision.json",
              "=" * 70]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
