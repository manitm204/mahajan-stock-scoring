"""Mahajan Hedge Fund — Layer 2 (Factor Scoring Engine) entry point.

Reads only the Layer 1 SQLite warehouse, scores the full universe with a
sector-neutral, missing-data-robust factor framework, and writes the results
back as new Layer 2 tables plus a CSV. No external APIs are ever called, so every
run is reproducible from stored data.

Cross-sectional ranking requires the whole universe, so scoring always runs over
all names; ``--ticker``/``--long-only``/``--short-only`` filter the *view*
(console + CSV), not the computation.

Examples:
    python run_scoring.py                       # score latest, full universe
    python run_scoring.py --ticker AAPL         # full run, show only AAPL
    python run_scoring.py --date 2026-06-01      # point-in-time (backtest)
    python run_scoring.py --regime-weights       # VIX regime-conditional weights
    python run_scoring.py --long-only            # CSV/console: LONG candidates
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pandas as pd

from data.config import load_config
from data.db import get_db
from data.utils import get_logger

from factors import score_factor
from factors.base import persist_results
from factors.composite import build_composite
from factors.crowding import detect_crowding
from factors.pipeline import (active_sub_factor_set, _resolve_factor_set,
                              resolve_factor_registry)
from factors.regime_weights import resolve_weights
from factors.short_interest import squeeze_warnings
from factors.utils import DataContext


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mahajan Hedge Fund — Layer 2 factor scoring")
    p.add_argument("--ticker", help="Restrict the displayed/exported view to one ticker")
    p.add_argument("--date", help="Score as-of this date (YYYY-MM-DD); default = latest")
    p.add_argument("--regime-weights", action="store_true",
                   help="Apply VIX regime-conditional composite weights")
    p.add_argument("--engine-weights", action=argparse.BooleanOptionalAction, default=None,
                   help="Use the Factor Weight Engine weights (factors.engine_weights). "
                        "Overrides the factors.use_engine_weights config toggle; "
                        "takes precedence over --regime-weights when on.")
    p.add_argument("--sub-factor-set", default=None,
                   help="Named sub-factor set from config.factor_sets (e.g. 'lean'). "
                        "Overrides the factors.sub_factor_set production default; "
                        "pass 'full' (or v1/none/all) to force the full V1 set.")
    p.add_argument("--long-only", action="store_true", help="View only LONG candidates")
    p.add_argument("--short-only", action="store_true", help="View only SHORT candidates")
    p.add_argument("--no-store", action="store_true", help="Skip writing scores to the DB")
    p.add_argument("--no-csv", action="store_true", help="Skip writing the CSV export")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    log = get_logger("scoring", log_file=cfg.log_file)
    started = datetime.now(timezone.utc)

    min_obs = int(cfg.get("factors", "min_sector_obs", default=5))
    insider_days = int(cfg.get("factors", "insider_window_days", default=90))

    with get_db() as db:
        ctx = DataContext(db=db, as_of=args.date, insider_window_days=insider_days)
        as_of = ctx.as_of
        log.info("Scoring %d names as of %s", len(ctx.universe), as_of)

        # 1–3. Sub-factor + parent factor scores -----------------------------
        set_name = active_sub_factor_set(cfg, args.sub_factor_set)
        allowlist_map = _resolve_factor_set(cfg, set_name)
        registry, weight_map = resolve_factor_registry(set_name)
        results = [score_factor(f, ctx, min_obs=min_obs,
                                sub_factor_allowlist=(weight_map.get(f.key)
                                                      or allowlist_map.get(f.key)))
                   for f in registry]
        log.info("Scored %d factors (sub_factor_set=%s)", len(results), set_name or "full")

        # 4. Regime weights --------------------------------------------------
        decision = resolve_weights(cfg, ctx, use_regime=args.regime_weights,
                                   use_engine=args.engine_weights)
        log.info("Weights=%s (VIX=%s, regime=%s)",
                 decision.applied, decision.vix, decision.regime)

        # 5–7. Composite, sector re-rank, LONG/SHORT labels ------------------
        composite = build_composite(results, decision.weights, ctx.sectors(), cfg, min_obs=min_obs)

        # 8. Crowding detection ---------------------------------------------
        crowd = detect_crowding(ctx, results, composite, cfg)
        momentum_parent = next(r.parent for r in results if r.key == "momentum")
        squeeze = squeeze_warnings(ctx, momentum_parent)

        # 9. Export ----------------------------------------------------------
        full = _assemble(results, composite)
        missing = _missing_data_warnings(results)
        computed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if not args.no_store:
            persist_results(db, as_of, results, composite, decision.applied, computed_at)
            log.info("Persisted scores to DB for %s", as_of)
        if not args.no_csv:
            path = _export_csv(cfg, full)
            log.info("Wrote %s", path)

    _print_summary(full, composite, crowd, squeeze, decision, as_of, args, started, missing)


def _assemble(results, composite: pd.DataFrame) -> pd.DataFrame:
    """Wide, explainable frame: sector + sub-factor + parent + composite columns."""
    sub_all = pd.concat([r.sub_scores for r in results], axis=1)
    parent_cols = [f"{r.key}_score" for r in results]
    ordered = (
        ["sector"]
        + list(sub_all.columns)
        + parent_cols
        + ["composite_raw", "composite_score", "sector_rank", "long_short_flag"]
    )
    full = composite.join(sub_all)
    full = full.reindex(columns=[c for c in ordered if c in full.columns])
    full.index.name = "ticker"
    return full.sort_values("composite_score", ascending=False)


def _export_csv(cfg, full: pd.DataFrame):
    rel = cfg.get("factors", "output", "csv", default="output/scored_universe_latest.csv")
    path = cfg.root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    full.round(2).to_csv(path)
    return path


def _apply_view(full: pd.DataFrame, args) -> pd.DataFrame:
    view = full
    if args.long_only:
        view = view[view["long_short_flag"] == "LONG"]
    if args.short_only:
        view = view[view["long_short_flag"] == "SHORT"]
    if args.ticker:
        view = view[view.index == args.ticker.upper()]
    return view


def _missing_data_warnings(results, threshold: float = 0.5) -> list[str]:
    """Sub-factors whose raw inputs are mostly missing (scored neutral)."""
    out = []
    for res in results:
        if res.raw.empty:
            continue
        for col in res.raw.columns:
            frac = float(res.raw[col].isna().mean())
            if frac >= threshold:
                out.append(f"{res.key}.{col}: {frac:.0%} missing")
    return out


def _print_summary(full, composite, crowd, squeeze, decision, as_of, args, started, missing) -> None:
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    view = _apply_view(full, args)
    counts = composite["long_short_flag"].value_counts()
    cols = ["sector", "composite_score", "long_short_flag"]

    lines: list[str] = []
    add = lines.append
    add("")
    add("=" * 72)
    add(f" MAHAJAN HEDGE FUND — LAYER 2 FACTOR SCORING  (as of {as_of})".center(72, "="))
    add("=" * 72)
    add(f" Universe scored : {len(full)}    Weights: {decision.applied}    "
        f"VIX: {decision.vix if decision.vix is not None else 'n/a'} ({decision.regime})")
    add(f" Classification  : LONG {int(counts.get('LONG', 0))}  "
        f"SHORT {int(counts.get('SHORT', 0))}  WATCHLIST {int(counts.get('WATCHLIST', 0))}")

    if args.ticker:
        add("-" * 72)
        if view.empty:
            add(f" Ticker {args.ticker.upper()} not found in scored universe.")
        else:
            add(f" {args.ticker.upper()} detail:")
            add(view.round(2).to_string())
    else:
        longs = full[full["long_short_flag"] == "LONG"].head(5)
        shorts = full[full["long_short_flag"] == "SHORT"].sort_values("composite_score").head(5)
        add("-" * 72)
        add(" TOP 5 LONG CANDIDATES")
        add(longs[cols].round(2).to_string() if len(longs) else "   (none)")
        add("-" * 72)
        add(" TOP 5 SHORT CANDIDATES")
        add(shorts[cols].round(2).to_string() if len(shorts) else "   (none)")

    add("-" * 72)
    add(" DEGENERATE FACTORS (insufficient data to differentiate)")
    add("   " + (", ".join(crowd.degenerate_factors) if crowd.degenerate_factors else "(none)"))

    add("-" * 72)
    add(" CROWDING / RISK WARNINGS")
    risk = [w for w in crowd.warnings if not w.startswith("degenerate_factor")]
    if risk:
        for w in risk:
            add(f"   - {w}")
    else:
        add("   (none)")
    if squeeze:
        add(f"   - short_squeeze_warning: high short interest + strong momentum: "
            f"{', '.join(squeeze[:15])}{'...' if len(squeeze) > 15 else ''}")

    add("-" * 72)
    add(" MISSING DATA WARNINGS (sub-factors with mostly absent inputs -> neutral)")
    if missing:
        for m in missing:
            add(f"   - {m}")
    else:
        add("   (none)")

    add("=" * 72)
    add(f" Done in {elapsed:.1f}s")
    add("=" * 72)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
