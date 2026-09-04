"""The point-in-time backtest loop.

For each rebalance date the engine (1) scores the whole universe using only data
filed on/before that date, (2) selects long/short holdings per the strategy's
enter/exit rule, then (3) realizes the forward return earned until the next
rebalance from adjusted-close prices, net of optional transaction and borrow
costs. It emits everything the metrics/plots/reports layers need: an equity
curve, per-rebalance positions, round-trip trades, a rebalance log, name-level
forward returns (for hit rates), and per-date factor quintile returns.

No-look-ahead is structural: returns for the period ``[d, e]`` are computed from
prices at ``d`` and ``e`` only, and selection at ``d`` consumes only
``score_universe(d, reporting_lag=True)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data.config import Config, load_config
from data.db import get_db
from factors import ALL_FACTORS
from factors.market_regime import regime_from_db
from factors.pipeline import score_universe
from factors.weight_engine import FactorWeightEngine

from . import data_loader as dl
from .portfolio_rules import (
    STRATEGIES, StrategySpec, equal_weights, score_weights, select_positions,
)
from .weighting import ICWeighter

SERIES = ["long_basket", "short_basket", "long_short", "spy"]
FACTOR_KEYS = [f.key for f in ALL_FACTORS]


@dataclass
class BacktestConfig:
    strategy: str = "top5_entry_top10_exit"
    rebalance: str = "monthly"
    start_date: str = "2022-01-01"
    end_date: str = "2026-06-01"
    transaction_costs: bool = True
    transaction_cost_bps: float = 10.0
    short_borrow_cost_annual_bps: float = 300.0
    reporting_lag: bool = True
    use_regime: bool = False
    weights_mode: str = "config"        # config | regime | ic | engine
    ic_min_periods: int = 6             # IC warm-up before learned weights apply
    ic_metric: str = "mean"             # "mean" or "ir" (info ratio) — IC mode only
    ic_max_weight: float = 1.0          # per-factor cap (1.0 = no cap) — IC mode only
    holdout_months: int = 0             # last N months reported as OOS holdout
    position_weighting: str = "equal"   # "equal" (1/N per side) or "score" (∝ composite)
    sub_factor_set: str | None = None   # None = V1 (all sub-factors); "v2" filters
    drop_parents: tuple[str, ...] = ()  # parent factor keys forced to weight 0
    ic_min_weight: float = 0.0          # per-factor floor in IC composite (0 = no floor)
    # Factor Weight Engine (weights_mode="engine"): baseline + confidence-scaled,
    # regime-learned overlay. See factors.weight_engine.FactorWeightEngine.
    engine_overlay_strength: float = 0.30
    engine_min_weight: float = 0.02
    engine_max_weight: float = 0.40
    engine_smoothing: float = 0.50
    engine_min_periods: int = 4
    engine_regime_blend: float = 0.50
    output_dir: str = "output/backtests"


@dataclass
class BacktestResult:
    config: BacktestConfig
    spec: StrategySpec
    rebalance_dates: list[str]
    equity_curve: pd.DataFrame          # date x {long_basket, short_basket, long_short, spy}
    period_returns: pd.DataFrame        # period_end x same series
    positions: pd.DataFrame
    trades: pd.DataFrame
    rebalance_log: pd.DataFrame
    forward_returns: pd.DataFrame       # name-period level (hit rates)
    factor_quintiles: pd.DataFrame      # per-date quintile forward returns
    weights_log: pd.DataFrame = field(default_factory=pd.DataFrame)  # per-date factor weights
    pit_notes: list[str] = field(default_factory=list)
    holdout_start: str | None = None    # first rebalance treated as OOS, if any

    @property
    def primary_series(self) -> str:
        return self.spec.primary_series


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_backtest(config: BacktestConfig, cfg: Config | None = None) -> BacktestResult:
    cfg = cfg or load_config()
    spec = STRATEGIES.get(config.strategy)
    if spec is None:
        raise ValueError(
            f"Unknown strategy {config.strategy!r}. Choices: {sorted(STRATEGIES)}")

    with get_db() as db:
        universe = db.universe_tickers()
        trading_dates = dl.trading_calendar(db)
        matrix = dl.load_price_matrix(db, universe, config.start_date, config.end_date)
        if matrix.empty:
            raise RuntimeError("No price data in the requested date range.")

        rebal_dates = dl.generate_rebalance_dates(
            trading_dates, config.start_date, config.end_date, config.rebalance)
        rebal_dates = [d for d in rebal_dates if d in matrix.index]
        if len(rebal_dates) < 2:
            raise RuntimeError(
                "Need at least two rebalance dates; widen the date range or use a "
                "finer rebalance frequency.")
        terminal = dl.last_trading_on_or_before(trading_dates, config.end_date)
        if terminal not in matrix.index:
            terminal = matrix.index[-1]

        engine = _Engine(config, cfg, spec, db, matrix, rebal_dates, terminal)
        return engine.run()


# ---------------------------------------------------------------------------
# Engine internals
# ---------------------------------------------------------------------------
class _Engine:
    def __init__(self, config, cfg, spec, db, matrix, rebal_dates, terminal):
        self.config = config
        self.cfg = cfg
        self.spec = spec
        self.db = db
        self.matrix = matrix
        self.rebal_dates = rebal_dates
        self.terminal = terminal

        self.txn_frac = (config.transaction_cost_bps / 1e4) if config.transaction_costs else 0.0
        self.borrow_annual = (config.short_borrow_cost_annual_bps / 1e4) if config.transaction_costs else 0.0

        self._positions: list[dict] = []
        self._periods: list[dict] = []
        self._fwd: list[dict] = []
        self._quintiles: list[dict] = []
        self._rebal_log: list[dict] = []
        self._trades: list[dict] = []
        self._ledger: dict[tuple[str, str], dict] = {}
        self._pit_notes: list[str] = []
        self._note_counts: dict[str, int] = {}
        self._weights_log: list[dict] = []

        # Parents the caller has chosen to remove entirely (zero weight). They
        # are stripped from the IC learnable key list and from the static weight
        # override so the composite is built purely from the remaining parents.
        self._dropped = {k for k in config.drop_parents if k in FACTOR_KEYS}
        unknown = set(config.drop_parents) - set(FACTOR_KEYS)
        if unknown:
            raise ValueError(
                f"--drop-parents references unknown parent key(s): {sorted(unknown)}. "
                f"Valid keys: {FACTOR_KEYS}")
        self._active_keys = [k for k in FACTOR_KEYS if k not in self._dropped]

        # VIX regime thresholds (shared with the static regime-weight tables) so
        # the Factor Weight Engine buckets volatility the same way the rest of
        # the system does.
        self._low_vix = float(cfg.get("factors", "regime", "low_vol_max", default=15.0))
        self._high_vix = float(cfg.get("factors", "regime", "high_vol_min", default=25.0))

        # Walk-forward IC weighting is opt-in (weights_mode="ic"); otherwise the
        # static/regime weight vector from resolve_weights is used as before.
        # weights_mode="engine" uses the baseline+overlay Factor Weight Engine.
        self._ic = None
        self._we = None
        if config.weights_mode == "engine":
            self._we = FactorWeightEngine(
                self._active_keys, self._active_baseline(), matrix,
                overlay_strength=config.engine_overlay_strength,
                min_weight=config.engine_min_weight,
                max_weight=config.engine_max_weight,
                smoothing=config.engine_smoothing,
                min_periods=config.engine_min_periods,
                regime_blend=config.engine_regime_blend,
            )
        elif config.weights_mode == "ic":
            default_w = {k: float(v) for k, v in
                         cfg.get("factors", "default_weights", default={}).items()}
            default_w = {k: v for k, v in default_w.items() if k not in self._dropped}
            # Renormalize so the warmup/fallback vector still sums to 1 after
            # dropping parents; otherwise the IC weighter would log weights
            # summing to (1 - dropped_share) until learned weights kick in.
            _total = sum(default_w.values())
            if _total > 0:
                default_w = {k: v / _total for k, v in default_w.items()}
            self._ic = ICWeighter(
                self._active_keys, default_w, matrix,
                min_periods=config.ic_min_periods,
                metric=config.ic_metric,
                max_weight=config.ic_max_weight,
                min_weight=config.ic_min_weight,
            )

        # OOS holdout: any rebalance date >= holdout_start is tagged "oos" so the
        # IS/OOS split summaries can be computed at report time. The IC weighter
        # itself is unaffected — it keeps walking forward, which is the honest
        # evaluation (the recipe, not the weight values, is what is being held
        # out for inspection).
        self._holdout_start: str | None = self._resolve_holdout_start(
            config.holdout_months, rebal_dates)

    def run(self) -> BacktestResult:
        prev_long: set[str] = set()
        prev_short: set[str] = set()
        prev_long_w: dict[str, float] = {}
        prev_short_w: dict[str, float] = {}

        n = len(self.rebal_dates)
        for i, d in enumerate(self.rebal_dates):
            we_decision = None
            regime = None
            if self._we is not None:
                decision = None
                regime = self._regime_for(d)
                we_decision = self._we.weights_for(d, regime)
                su = score_universe(
                    d, db=self.db, cfg=self.cfg,
                    reporting_lag=self.config.reporting_lag,
                    weights_override=we_decision.weights,
                    sub_factor_set=self.config.sub_factor_set,
                )
            elif self._ic is not None:
                decision = self._ic.weights_for(d)
                su = score_universe(
                    d, db=self.db, cfg=self.cfg,
                    reporting_lag=self.config.reporting_lag,
                    weights_override=decision["weights"],
                    sub_factor_set=self.config.sub_factor_set,
                )
            else:
                decision = None
                if self._dropped:
                    # Static/regime weights minus the dropped parents, renormalized.
                    base = self._static_weights_for(d)
                    kept = {k: v for k, v in base.items() if k not in self._dropped}
                    total = sum(kept.values())
                    override = ({k: v / total for k, v in kept.items()}
                                if total > 0 else None)
                    su = score_universe(
                        d, db=self.db, cfg=self.cfg,
                        reporting_lag=self.config.reporting_lag,
                        weights_override=override,
                        sub_factor_set=self.config.sub_factor_set,
                    )
                else:
                    su = score_universe(
                        d, db=self.db, cfg=self.cfg,
                        reporting_lag=self.config.reporting_lag,
                        use_regime=self.config.use_regime,
                        sub_factor_set=self.config.sub_factor_set,
                    )
            for note in su.pit_notes:
                self._note_counts[note] = self._note_counts.get(note, 0) + 1
            ranked = su.frame
            self._record_weights(d, su, decision, we_decision)
            if self._we is not None:
                self._we.record(d, ranked, regime)
            elif self._ic is not None:
                cols = [f"{k}_score" for k in self._active_keys
                        if f"{k}_score" in ranked.columns]
                self._ic.record(d, ranked[cols])
            long_names, short_names = select_positions(self.spec, ranked, prev_long, prev_short)

            px_d = self.matrix.loc[d]
            self._reconcile("long", long_names, px_d, d, i)
            self._reconcile("short", short_names, px_d, d, i)

            if self.config.position_weighting == "score":
                new_long_w = score_weights(long_names, ranked, "long")
                new_short_w = score_weights(short_names, ranked, "short")
            else:
                new_long_w = equal_weights(long_names)
                new_short_w = equal_weights(short_names)
            self._record_positions(d, ranked, long_names, short_names, new_long_w, new_short_w, px_d)

            period_end = self.rebal_dates[i + 1] if i + 1 < n else self.terminal
            if period_end and period_end > d:
                self._run_period(
                    d, period_end, ranked, long_names, short_names,
                    new_long_w, new_short_w, prev_long_w, prev_short_w)

            prev_long, prev_short = set(long_names), set(short_names)
            prev_long_w, prev_short_w = new_long_w, new_short_w

        # Close anything still open at the terminal settlement date.
        self._close_all(self.terminal, n)
        return self._assemble(su)

    # -- one holding period -------------------------------------------------
    def _run_period(self, d, e, ranked, long_names, short_names,
                    new_long_w, new_short_w, prev_long_w, prev_short_w):
        px0, px1 = self.matrix.loc[d], self.matrix.loc[e]
        spy_ret = _ret(px0.get(dl.SPY), px1.get(dl.SPY))
        holding_days = (pd.Timestamp(e) - pd.Timestamp(d)).days
        borrow = self.borrow_annual * (holding_days / 365.25)

        long_gross, long_recs, long_missing = _basket_eval(
            px0, px1, long_names, "long", weights=new_long_w)
        short_gross, short_recs, short_missing = _basket_eval(
            px0, px1, short_names, "short", weights=new_short_w)

        long_traded = _turnover(prev_long_w, new_long_w)
        short_traded = _turnover(prev_short_w, new_short_w)
        long_net = long_gross - long_traded * self.txn_frac
        short_net = short_gross - short_traded * self.txn_frac - borrow

        has_short = self.spec.short_enabled
        ls_net = (0.5 * long_net + 0.5 * short_net) if has_short else long_net

        # Reported one-way turnover as a fraction of *gross* book. ``long_traded``
        # / ``short_traded`` are sum|Δw| with each side's weights summing to 1, so
        # a two-sided 50/50 book scales each side by 0.5 before the 0.5*sum|Δw|
        # one-way convention; a long-only book is the full 0.5*sum|Δw|. (Costs are
        # applied per side above, on each side's own capital, so they are
        # unaffected by this reporting scale.)
        portfolio_turnover = (
            0.25 * (long_traded + short_traded) if has_short else 0.5 * long_traded)

        phase = self._phase_for(d)
        self._periods.append({
            "rebalance_date": d, "period_end": e,
            "long_basket": long_net,
            "short_basket": short_net if has_short else np.nan,
            "long_short": ls_net,
            "spy": spy_ret,
            "phase": phase,
        })

        # Name-level forward returns for hit-rate analytics.
        for rec in long_recs:
            self._fwd.append({
                "rebalance_date": d, "period_end": e, "ticker": rec["ticker"],
                "side": "long", "stock_fwd": rec["stock_fwd"], "profit": rec["profit"],
                "spy_fwd": spy_ret,
            })
        for rec in short_recs:
            self._fwd.append({
                "rebalance_date": d, "period_end": e, "ticker": rec["ticker"],
                "side": "short", "stock_fwd": rec["stock_fwd"], "profit": rec["profit"],
                "spy_fwd": spy_ret,
            })

        self._record_quintiles(d, e, ranked, px0, px1)

        self._rebal_log.append({
            "rebalance_date": d, "period_end": e, "holding_days": holding_days,
            "n_long": len(long_names), "n_short": len(short_names),
            "turnover": portfolio_turnover,
            "missing_prices": long_missing + short_missing,
            "long_return": long_net, "short_return": short_net if has_short else np.nan,
            "long_short_return": ls_net, "spy_return": spy_ret,
            "phase": phase,
        })

    def _record_quintiles(self, d, e, ranked, px0, px1):
        fwd_all = (px1 / px0) - 1.0
        cr = ranked["composite_raw"].dropna()
        if cr.nunique() < 5:
            return
        try:
            q = pd.qcut(cr, 5, labels=[1, 2, 3, 4, 5])
        except ValueError:
            return
        row = {"rebalance_date": d, "period_end": e}
        means = {}
        for lab in (1, 2, 3, 4, 5):
            names = q.index[q == lab]
            r = fwd_all.reindex(names).dropna()
            means[lab] = float(r.mean()) if len(r) else np.nan
            row[f"q{lab}_forward_return"] = means[lab]
        row["q5_minus_q1_spread"] = (
            means[5] - means[1]
            if not (np.isnan(means[5]) or np.isnan(means[1])) else np.nan)
        self._quintiles.append(row)

    def _active_baseline(self) -> dict[str, float]:
        """Config default weights restricted to active parents, renormalized to 1."""
        base = {k: float(v) for k, v in
                self.cfg.get("factors", "default_weights", default={}).items()
                if k not in self._dropped}
        total = sum(base.values())
        return {k: v / total for k, v in base.items()} if total > 0 else base

    def _regime_for(self, d: str):
        """Point-in-time market regime at ``d`` (VIX + SPY vs 200dma)."""
        row = self.db.query_one(
            "SELECT close FROM daily_prices WHERE ticker='VIX' AND date <= ? "
            "ORDER BY date DESC LIMIT 1", (d,))
        vix = float(row["close"]) if row and row["close"] is not None else None
        return regime_from_db(self.db, d, vix,
                              low_vix=self._low_vix, high_vix=self._high_vix)

    def _static_weights_for(self, _d: str) -> dict[str, float]:
        """Resolve the static (or regime) weight vector before any drop filter.

        Used only when ``drop_parents`` is set with weights_mode != 'ic' so the
        override passed to ``score_universe`` has the dropped parents removed.
        """
        if self.config.use_regime:
            # Regime weights depend on the VIX as-of the date; resolve_weights
            # reads it from a DataContext. Build a lightweight context.
            from factors.utils import DataContext
            from factors.regime_weights import resolve_weights
            ctx = DataContext(db=self.db, as_of=_d,
                              insider_window_days=int(self.cfg.get(
                                  "factors", "insider_window_days", default=90)),
                              reporting_lag=self.config.reporting_lag)
            try:
                decision = resolve_weights(self.cfg, ctx, use_regime=True)
                return {k: float(v) for k, v in decision.weights.items()}
            finally:
                ctx.close()
        return {k: float(v) for k, v in
                self.cfg.get("factors", "default_weights", default={}).items()}

    def _record_weights(self, d, su, decision, we_decision=None):
        """Log the composite weight vector (and trailing IC stats) used at ``d``."""
        if we_decision is not None:
            row = {"rebalance_date": d, "applied": we_decision.applied,
                   "phase": self._phase_for(d),
                   "regime": we_decision.regime,
                   "vix": we_decision.regime_state.vix
                          if we_decision.regime_state.vix is not None else np.nan,
                   "n_ic_periods": we_decision.n_periods,
                   "confidence": we_decision.confidence}
            for k in FACTOR_KEYS:
                row[f"{k}_w"] = float(su.weights.get(k, np.nan))
                m = we_decision.metrics.get(k)
                row[f"{k}_ic"] = m.ic_recency if m is not None else np.nan
            self._weights_log.append(row)
            return

        applied = decision["applied"] if decision is not None else su.applied
        row = {"rebalance_date": d, "applied": applied,
               "phase": self._phase_for(d),
               "regime": getattr(su, "regime", np.nan),
               "vix": getattr(su, "vix", np.nan),
               "n_ic_periods": decision["n_ic_periods"] if decision is not None else np.nan}
        for k in FACTOR_KEYS:
            row[f"{k}_w"] = float(su.weights.get(k, np.nan))
        if decision is not None:
            for k in FACTOR_KEYS:
                row[f"{k}_ic"] = decision["mean_ic"].get(k, np.nan)
                row[f"{k}_ic_std"] = decision["std_ic"].get(k, np.nan)
        self._weights_log.append(row)

    # -- phase / holdout ----------------------------------------------------
    def _resolve_holdout_start(self, months: int, rebal_dates: list[str]) -> str | None:
        """First rebalance date treated as OOS, or None if the holdout is off.

        Uses the actual rebalance grid so the boundary is always a rebalance
        date the run touched; ``months`` is measured back from the final
        rebalance. If ``months`` is non-positive, holdout is disabled.
        """
        if months <= 0 or len(rebal_dates) < 2:
            return None
        last = pd.Timestamp(rebal_dates[-1])
        cutoff = last - pd.DateOffset(months=months)
        oos = [d for d in rebal_dates if pd.Timestamp(d) >= cutoff]
        if not oos or len(oos) == len(rebal_dates):
            return None  # zero IS or zero OOS — refuse to split
        return oos[0]

    def _phase_for(self, d: str) -> str:
        if self._holdout_start is None:
            return "is"
        return "oos" if pd.Timestamp(d) >= pd.Timestamp(self._holdout_start) else "is"

    # -- positions / trade ledger ------------------------------------------
    def _record_positions(self, d, ranked, long_names, short_names, lw, sw, px_d):
        for side, names, weights in (("long", long_names, lw), ("short", short_names, sw)):
            for t in names:
                row = ranked.loc[t] if t in ranked.index else None
                self._positions.append({
                    "rebalance_date": d, "ticker": t, "side": side,
                    "weight": weights.get(t, np.nan),
                    "sector": row["sector"] if row is not None else None,
                    "composite_score": float(row["composite_score"]) if row is not None else np.nan,
                    "composite_raw": float(row["composite_raw"]) if row is not None else np.nan,
                    "rank": int(row["rank"]) if row is not None else None,
                    "rank_from_bottom": int(row["rank_from_bottom"]) if row is not None else None,
                    "entry_price": _clean(px_d.get(t)),
                })

    def _reconcile(self, side, new_names, px_d, date, idx):
        new_set = set(new_names)
        for key in list(self._ledger):
            t, s = key
            if s == side and t not in new_set:
                self._close_trade(t, side, self._ledger.pop(key), date, _clean(px_d.get(t)), idx)
        for t in new_names:
            key = (t, side)
            if key not in self._ledger:
                self._ledger[key] = {"entry_date": date, "entry_price": _clean(px_d.get(t)), "entry_idx": idx}

    def _close_all(self, date, idx):
        px = self.matrix.loc[date]
        for key in list(self._ledger):
            t, side = key
            self._close_trade(t, side, self._ledger.pop(key), date, _clean(px.get(t)), idx)

    def _close_trade(self, ticker, side, entry, exit_date, exit_price, exit_idx):
        ep, xp = entry["entry_price"], exit_price
        if ep is None or xp is None or ep <= 0 or xp <= 0:
            ret = np.nan
        else:
            ret = (xp / ep - 1.0) if side == "long" else (ep / xp - 1.0)
        self._trades.append({
            "ticker": ticker, "side": side,
            "entry_date": entry["entry_date"], "entry_price": ep,
            "exit_date": exit_date, "exit_price": xp,
            "return": ret,
            "holding_days": (pd.Timestamp(exit_date) - pd.Timestamp(entry["entry_date"])).days,
            "holding_periods": exit_idx - entry["entry_idx"],
        })

    # -- assemble result ----------------------------------------------------
    def _finalize_notes(self) -> list[str]:
        """Flatten per-date PIT notes into a deduped list.

        A note that fired on every rebalance (a structural assumption) is shown
        as-is; one that fired on only some (e.g. a data source that became
        available partway through) is annotated with its coverage so the
        limitation's scope is explicit.
        """
        total = len(self.rebal_dates)
        notes = []
        for note, cnt in self._note_counts.items():
            notes.append(note if cnt >= total else f"{note} [{cnt}/{total} rebalances]")
        return notes

    def _assemble(self, last_su) -> BacktestResult:
        periods = pd.DataFrame(self._periods)
        equity = self._build_equity(periods)
        if not periods.empty:
            # period_returns keeps phase alongside the SERIES so downstream
            # reporting can split IS vs OOS without re-deriving it.
            keep = SERIES + (["phase"] if "phase" in periods.columns else [])
            periods = periods.set_index("period_end")[keep]

        result = BacktestResult(
            config=self.config,
            spec=self.spec,
            rebalance_dates=self.rebal_dates,
            equity_curve=equity,
            period_returns=periods,
            positions=pd.DataFrame(self._positions),
            trades=pd.DataFrame(self._trades),
            rebalance_log=pd.DataFrame(self._rebal_log),
            forward_returns=pd.DataFrame(self._fwd),
            factor_quintiles=pd.DataFrame(self._quintiles),
            weights_log=pd.DataFrame(self._weights_log),
            pit_notes=self._finalize_notes(),
            holdout_start=self._holdout_start,
        )
        return result

    def _build_equity(self, periods: pd.DataFrame) -> pd.DataFrame:
        active = {"long_basket": True, "long_short": True, "spy": True,
                  "short_basket": self.spec.short_enabled}
        d0 = self.rebal_dates[0]
        rows = [{"date": d0, **{k: (1.0 if active[k] else np.nan) for k in SERIES}}]
        cum = {k: 1.0 for k in SERIES}
        for _, p in periods.iterrows():
            row = {"date": p["period_end"]}
            for k in SERIES:
                r = p[k]
                if not active[k] or pd.isna(r):
                    row[k] = np.nan
                else:
                    cum[k] *= (1.0 + r)
                    row[k] = cum[k]
            rows.append(row)
        return pd.DataFrame(rows).set_index("date")


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------
def _bad(p) -> bool:
    return p is None or pd.isna(p) or p <= 0


def _ret(p0, p1) -> float:
    return np.nan if _bad(p0) or _bad(p1) else float(p1 / p0 - 1.0)


def _clean(p):
    return None if p is None or pd.isna(p) else float(p)


def _basket_eval(px0, px1, names, side, weights=None):
    """Weighted basket return plus per-name records and missing count.

    ``weights`` is a {ticker: weight} dict that sums to 1.0 over the held
    names; pass ``None`` for the historical equal-weight behaviour. Names with
    missing prices drop out and the remaining weights are renormalized so the
    basket stays fully invested when one or two names have a price gap.
    """
    recs, missing = [], 0
    pairs: list[tuple[str, float]] = []  # (ticker, single-name profit)
    for t in names:
        p0, p1 = px0.get(t), px1.get(t)
        if _bad(p0) or _bad(p1):
            missing += 1
            continue
        stock_fwd = float(p1 / p0 - 1.0)
        profit = stock_fwd if side == "long" else float(p0 / p1 - 1.0)
        pairs.append((t, profit))
        recs.append({"ticker": t, "stock_fwd": stock_fwd, "profit": profit})
    if not pairs:
        return 0.0, recs, missing
    if weights is None:
        gross = float(np.mean([p for _, p in pairs]))
    else:
        kept = {t: float(weights.get(t, 0.0)) for t, _ in pairs}
        total = sum(kept.values())
        if total <= 0:
            gross = float(np.mean([p for _, p in pairs]))
        else:
            gross = float(sum((kept[t] / total) * p for t, p in pairs))
    return gross, recs, missing


def _turnover(prev_w: dict, new_w: dict) -> float:
    """Traded fraction = sum of absolute weight changes (0 = no change, 2 = full flip)."""
    keys = set(prev_w) | set(new_w)
    return float(sum(abs(new_w.get(k, 0.0) - prev_w.get(k, 0.0)) for k in keys))
