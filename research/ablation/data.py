"""One-time data bundle for the construction ablation.

Everything expensive lives here and is computed once per session: the walk-forward
scoring run, PIT market caps (quarterly shares_outstanding lagged 45 days — the
scoring layer's reporting-lag convention — times same-day price), trailing volatility,
per-date parent percentile ranks, the VIX spot series, and (lazily, on first use) the
VIX-tilted composite. Every ablation config downstream is pure re-weighting.
"""
from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from backtesting import data_loader as dl
from factors.composite import NEUTRAL, _normalize_parents
from factors.utils import sector_percentile
from factors.vix_tilt import apply_vix_tilt
from research.forward_returns import realize_delistings
from research.walkforward.compose import build_parent_panel
from research.walkforward.runner import run_splits
from research.walkforward.selection import slice_panel
from research.walkforward.splits import resolve_splits

SHARES_LAG_DAYS = 10   # short-interest publish lag (float_shares observed w/ each SI report)


@dataclass
class AblationData:
    matrix: pd.DataFrame                 # daily px, delistings realized (ffilled)
    sectors: pd.Series
    run: object                          # RunResult: pooled_scores/parent_scores/splits_data
    rebal_dates: list                    # monthly grid usable for simulation
    caps: pd.DataFrame                   # rebal date x ticker PIT market cap
    vol: pd.DataFrame                    # rebal date x ticker trailing 1y ann. vol
    vix: pd.Series                       # date -> VIX close
    parent_ranks: dict                   # {parent: {date: Series pct 0-100}}
    panel: object = None                 # ScorePanel (kept for the VIX-tilt recompute)
    _tilt_scores: dict | None = field(default=None, repr=False)

    def vix_spot(self, d: str) -> float | None:
        s = self.vix[self.vix.index <= d]
        return float(s.iloc[-1]) if len(s) else None

    def tilt_scores(self) -> dict:
        """Composite re-blended per date with VIX-tilted parent weights (frozen
        literature rule). Computed once, cached."""
        if self._tilt_scores is None:
            t0 = time.time()
            out = {}
            for sd in self.run.splits_data:
                cfg = sd["config"]
                test_rebals = list(sd["scores"].keys())
                ppanel = build_parent_panel(slice_panel(self.panel, test_rebals),
                                            cfg.sub_weights)
                for d in test_rebals:
                    frame = ppanel.scores.get(d)
                    if frame is None:
                        continue
                    pw = apply_vix_tilt(cfg.parent_weights, self.vix_spot(d))[0]
                    cols = [p for p in pw if p in frame.columns]
                    blend = _normalize_parents(frame[cols], "zscore", 20.0)
                    comp = pd.Series(0.0, index=frame.index)
                    used = 0.0
                    for k in cols:
                        comp += pw[k] * blend[k].fillna(NEUTRAL)
                        used += pw[k]
                    if used > 0:
                        comp /= used
                    secs = self.sectors.reindex(frame.index).fillna("Unknown")
                    out[d] = sector_percentile(comp, secs, higher_is_better=True, min_obs=5)
            self._tilt_scores = out
            print(f"  [setup] VIX-tilted composite built ({time.time()-t0:.0f}s)", flush=True)
        return self._tilt_scores


def _pit_caps(db, matrix: pd.DataFrame, rebal_dates: list) -> pd.DataFrame:
    """PIT free-float market cap at each rebalance: latest ``float_shares`` observed
    on/before ``d - 10d`` (short-interest publish lag), times the (delisting-realized)
    price at ``d``. Free-float shares are the S&P index-weighting basis, so this is the
    benchmark-aligned size — ``fundamentals.shares_outstanding`` is unpopulated here."""
    df = db.query_df(
        "SELECT ticker, date, float_shares FROM short_interest "
        "WHERE float_shares IS NOT NULL AND float_shares > 0"
    )
    sh = df.pivot_table(index="date", columns="ticker",
                        values="float_shares", aggfunc="last")
    sh.index = pd.to_datetime(sh.index)
    # ffill carries the latest float forward; bfill covers the pre-first-observation
    # head (mainly 2017, before short-interest float history begins). bfill cannot leak
    # return info: float is only a sizing input, never a selection signal, and it is a
    # slow-moving reference quantity — so an early book sized on end-2017 float is a
    # benign approximation, not look-ahead into prices.
    sh = sh.sort_index().ffill().bfill()
    rows = {}
    for d in rebal_dates:
        cutoff = pd.Timestamp(d) - pd.Timedelta(days=SHARES_LAG_DAYS)
        # Latest row on/before the cutoff already holds each ticker's newest float
        # (index is ffilled); DataFrame.asof is unusable — it needs an all-non-NaN row.
        upto = sh.loc[:cutoff]
        shares = upto.iloc[-1] if len(upto) else pd.Series(dtype=float)
        rows[d] = shares.reindex(matrix.columns) * matrix.loc[d]
    caps = pd.DataFrame(rows).T
    cov = caps.loc[rebal_dates[-1]].notna().mean()
    print(f"  [setup] PIT caps built ({cov:.0%} ticker coverage on last date)", flush=True)
    if cov < 0.5:
        raise RuntimeError(
            f"PIT cap coverage {cov:.0%} — cap-weighted configs would silently "
            f"degrade to equal weight; refusing to run.")
    return caps


def load_ablation_data(panel, db, start: str, end: str,
                       splits: str = "rolling5y") -> AblationData:
    t0 = time.time()
    matrix_raw = dl.load_price_matrix(db, panel.universe, start, end)
    sectors = dl.global_sectors(db)
    vixdf = db.query_df(
        "SELECT date, close FROM daily_prices WHERE ticker='VIX' ORDER BY date")
    vix = pd.Series(vixdf["close"].values, index=vixdf["date"].astype(str))
    print(f"  [setup] prices/sectors/VIX loaded ({time.time()-t0:.0f}s)", flush=True)

    cache = Path("cache") / f"ablation_run_{splits}.pkl"
    if cache.exists():
        with cache.open("rb") as fh:
            run = pickle.load(fh)
        print(f"  [setup] walk-forward run loaded from cache ({cache}) — delete it "
              f"after any scoring/panel change", flush=True)
    else:
        t0 = time.time()
        print(f"  [setup] walk-forward scoring run (splits={splits}) — the slow "
              f"part, ~10-20 min ...", flush=True)
        run = run_splits(panel, resolve_splits(splits), matrix_raw, sectors,
                         verbose=False)
        with cache.open("wb") as fh:
            pickle.dump(run, fh)
        print(f"  [setup] walk-forward done ({time.time()-t0:.0f}s, "
              f"{len(run.pooled_scores)} scored dates) — cached to {cache}",
              flush=True)

    matrix = realize_delistings(matrix_raw)
    rebal_dates = [d for d in sorted(run.pooled_scores) if d in matrix.index]

    caps = _pit_caps(db, matrix, rebal_dates)
    ann = matrix.pct_change(fill_method=None).rolling(252, min_periods=60).std() \
        * np.sqrt(252)
    vol = ann.loc[rebal_dates]
    parent_ranks = {
        p: {d: s.rank(pct=True) * 100.0 for d, s in per_date.items()}
        for p, per_date in run.parent_scores.items()
    }
    print(f"  [setup] vol + parent ranks ready — {len(rebal_dates)} rebalances, "
          f"{len(run.parent_scores)} parents", flush=True)
    return AblationData(matrix=matrix, sectors=sectors, run=run,
                        rebal_dates=rebal_dates, caps=caps, vol=vol, vix=vix,
                        parent_ranks=parent_ranks, panel=panel)
