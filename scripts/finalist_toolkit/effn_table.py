"""One consolidated table: Portfolio 1x / +VIX tilt / +VIX tilt 1.25x / SPY / QQQ.
Columns: CAGR, vol, Sharpe, Sortino, maxDD, beta(vs SPY), effective N.
Reuses the cached walk-forward run (cache/ablation_run_rolling5y.pkl)."""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/manit/Desktop/fun_projects/mahajan_hedge_fund")

import numpy as np
import pandas as pd

from run_walkforward import PANEL_START, PRICE_END, _load_panel, get_db
from research.ablation import load_ablation_data
from research.ablation.engine import (AblationConfig, apply_exclusion, select_book,
                                      base_weights, sector_overlay, cap_within_sectors)
from backtesting.data_loader import SPY
from research.walkforward.portfolio import _sortino, _cagr

FFR = {2017: 0.0100, 2018: 0.0183, 2019: 0.0216, 2020: 0.0038, 2021: 0.0008,
       2022: 0.0168, 2023: 0.0503, 2024: 0.0510, 2025: 0.0433, 2026: 0.0390}
SPREAD = 0.005
ERA = "2021-12-31"

FIN = dict(top_pct=0.10, exclusion="none", weighting="cap5", sector="cap_match")


def sim_series(data, cfg):
    """Mirror simulate_config's loop; return (net monthly series, mean effN)."""
    scores = data.tilt_scores() if cfg.vix_tilt else data.run.pooled_scores
    form = data.rebal_dates
    net, effs, ncount = {}, [], []
    prev_w, prev_names = pd.Series(dtype=float), []
    for i in range(len(form) - 1):
        d, nxt = form[i], form[i + 1]
        sc = apply_exclusion(scores[d].dropna(), d, data.parent_ranks, cfg.exclusion)
        names = select_book(sc, cfg.top_pct, cfg.exit_pct, prev_names)
        if not names:
            continue
        w = base_weights(cfg, names, sc, d, data)
        w = sector_overlay(w, cfg, scores[d].dropna().index.tolist(), d, data)
        w = w / w.sum()
        if cfg.final_cap is not None:
            w = cap_within_sectors(w, data.sectors, cfg.final_cap)
        ret = (data.matrix.loc[nxt].reindex(w.index)
               / data.matrix.loc[d].reindex(w.index) - 1.0)
        ok = ret.notna(); w = w[ok] / w[ok].sum(); ret = ret[ok]
        g = float((w * ret).sum())
        traded = float((w - prev_w.reindex(w.index.union(prev_w.index)).fillna(0.0))
                       .abs().sum())
        net[nxt] = g - traded * cfg.cost_bps / 1e4
        effs.append(1.0 / float((w ** 2).sum())); ncount.append(len(w))
        drift = w * (1.0 + ret); prev_w = drift / drift.sum(); prev_names = names
    print(f"  [{cfg.name}] avg holdings={np.mean(ncount):.0f}  effN={np.mean(effs):.0f}")
    return pd.Series(net).sort_index(), float(np.mean(effs))


def bench_series(data, ticker):
    px = data.matrix[ticker].reindex(data.rebal_dates)
    return (px / px.shift(1) - 1.0).dropna()


def cap_effn(data, subset_top=None):
    """Average effective-N of a cap-weighted benchmark proxy on our universe.
    subset_top=None -> full scored universe (SPY proxy); int -> top-N by cap (QQQ proxy)."""
    effs = []
    for d in data.rebal_dates:
        if d not in data.run.pooled_scores:
            continue
        if d not in data.caps.index:
            continue
        uni = data.run.pooled_scores[d].dropna().index
        caps = data.caps.loc[d].reindex(uni).dropna()
        caps = caps[caps > 0]
        if subset_top:
            caps = caps.nlargest(subset_top)
        if caps.sum() <= 0 or len(caps) < 2:
            continue
        w = caps / caps.sum()
        effs.append(1.0 / float((w ** 2).sum()))
    return float(np.mean(effs))


def metrics(r, spy, ppy=12.0):
    r = r.dropna()
    n = len(r)
    cagr = _cagr(r, ppy)
    vol = r.std(ddof=1) * np.sqrt(ppy)
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(ppy)
    sortino = _sortino(r, ppy)
    eq = (1 + r).cumprod(); mdd = (eq / eq.cummax() - 1).min()
    b = spy.reindex(r.index)
    df = pd.concat([r, b], axis=1).dropna()
    beta = np.cov(df.iloc[:, 0], df.iloc[:, 1], ddof=1)[0, 1] / np.var(df.iloc[:, 1], ddof=1)
    return dict(cagr=cagr, vol=vol, sharpe=sharpe, sortino=sortino, mdd=mdd, beta=beta)


def lever(r, L):
    ffr = pd.Series(pd.to_datetime(r.index).year.map(FFR), index=r.index)
    return L * r - (L - 1.0) * (ffr + SPREAD) / 12.0


def main():
    panel = _load_panel(False)
    with get_db() as db:
        data = load_ablation_data(panel, db, PANEL_START, PRICE_END, splits="rolling5y")

    spy = bench_series(data, SPY)
    qqq = bench_series(data, "QQQ") if "QQQ" in data.matrix.columns else None

    FIN15 = {**FIN, "top_pct": 0.15}
    r_base, eff_base = sim_series(data, AblationConfig(name="p10", vix_tilt=True, **FIN))
    r_tilt, eff_tilt = sim_series(data, AblationConfig(name="p15", vix_tilt=True, **FIN15))
    r_lev = lever(r_tilt, 1.25)

    idx = r_tilt.index
    spy_a = spy.reindex(idx); qqq_a = qqq.reindex(idx) if qqq is not None else None
    eff_spy = cap_effn(data); eff_qqq = cap_effn(data, subset_top=100)

    rows = []
    def add(name, r, effn, span=None):
        rr = r if span is None else r[r.index.astype(str).to_series().between(*span).values]
        m = metrics(rr.reindex(rr.index), spy.reindex(rr.index))
        m.update(name=name, effn=effn); rows.append(m)

    def full_table(span, label):
        print(f"\n{'='*88}\n{label}\n{'='*88}")
        recs = []
        for name, r, effn in [("Portfolio top10% + tilt (ref)", r_base, eff_base),
                              ("Portfolio top15% + tilt", r_tilt, eff_tilt),
                              ("Portfolio top15% + tilt, 1.25x", r_lev, eff_tilt),
                              ("SPY", spy_a, eff_spy)] + (
                              [("QQQ", qqq_a, eff_qqq)] if qqq is not None else []):
            r2 = r.reindex(idx)
            if span:
                mask = r2.index.astype(str)
                r2 = r2[(mask >= span[0]) & (mask <= span[1])]
            m = metrics(r2, spy.reindex(r2.index))
            recs.append((name, m, effn))
        hdr = f"{'':32}{'CAGR':>8}{'Vol':>8}{'Sharpe':>8}{'Sortino':>9}{'maxDD':>9}{'Beta':>7}{'effN':>7}"
        print(hdr)
        for name, m, effn in recs:
            print(f"{name:32}{m['cagr']*100:7.1f}%{m['vol']*100:7.1f}%{m['sharpe']:8.2f}"
                  f"{m['sortino']:9.2f}{m['mdd']*100:8.1f}%{m['beta']:7.2f}{effn:7.0f}")

    full_table(None, "FULL PERIOD 2017-2026")
    full_table(("0000", ERA), "ERA 1  (2017 - 2021)")
    full_table((ERA, "9999"), "ERA 2  (2022 - 2026)")
    print(f"\n[note] effN for SPY/QQQ are cap-weight proxies on our universe "
          f"(SPY=full universe, QQQ=top-100 by cap). Leverage does not change effN.")


if __name__ == "__main__":
    main()
