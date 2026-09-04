"""Unit smoke tests for the construction-ablation engine (research/ablation).

Synthetic 8-ticker world, deterministic prices. Pins: every weighting scheme returns
weights that sum to 1 with no negatives, the 5% waterfill respects its cap, sector
overlays hit their targets, exclusions drop/demote the right names, hysteresis holds
incumbents, and simulate_config runs end-to-end producing sane costed metrics.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from research.ablation.engine import (AblationConfig, apply_exclusion, base_weights,
                                      cap_within_sectors, sector_overlay, select_book,
                                      simulate_config, _waterfill_cap)

TICKERS = list("ABCDEFGH")
N_MONTHS = 30


def _data():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2020-01-31", periods=N_MONTHS, freq="ME").strftime("%Y-%m-%d")
    steps = 1.0 + rng.normal(0.005, 0.03, size=(N_MONTHS, len(TICKERS)))
    px = pd.DataFrame(100 * np.cumprod(steps, axis=0), index=dates, columns=TICKERS)
    px["SPY"] = 100 * np.cumprod(1.0 + rng.normal(0.006, 0.02, N_MONTHS))
    scores = {d: pd.Series(rng.uniform(0, 100, len(TICKERS)), index=TICKERS)
              for d in dates}
    ranks = {p: {d: s.rank(pct=True) * 100 for d, s in scores.items()}
             for p in ("quality", "momentum")}
    caps = pd.DataFrame({t: 1e9 * (i + 1) for i, t in enumerate(TICKERS)},
                        index=dates)
    vol = pd.DataFrame(np.tile(0.2 + 0.02 * np.arange(len(TICKERS)), (N_MONTHS, 1)),
                       index=dates, columns=TICKERS)
    sectors = pd.Series(["Tech", "Tech", "Tech", "Tech", "Fin", "Fin", "Health",
                         "Health"], index=TICKERS)
    run = SimpleNamespace(pooled_scores=scores, parent_scores=ranks, splits_data=[])
    return SimpleNamespace(matrix=px, sectors=sectors, run=run,
                           rebal_dates=list(dates), caps=caps, vol=vol,
                           parent_ranks=ranks, tilt_scores=lambda: scores)


DATA = _data()
D = DATA.rebal_dates[10]
SCORE = DATA.run.pooled_scores[D]


@pytest.mark.parametrize("wt", ["ew", "cap", "cap5", "ewcap", "rank_lin", "rank_sqrt",
                                "score_vol", "inv_vol", "erc", "softmax"])
def test_every_weighting_is_a_valid_book(wt):
    cfg = AblationConfig(name="t", weighting=wt)
    w = base_weights(cfg, TICKERS, SCORE, D, DATA)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all() and len(w) == len(TICKERS)


def test_waterfill_respects_cap():
    w = pd.Series([0.6, 0.2, 0.1, 0.05, 0.05], index=list("VWXYZ"))
    out = _waterfill_cap(w, 0.30)
    assert out.max() <= 0.30 + 1e-6
    assert out.sum() == pytest.approx(1.0)


def test_sector_overlay_hits_count_neutral_targets():
    cfg = AblationConfig(name="t", sector="neutral")
    w = pd.Series(1 / len(TICKERS), index=TICKERS)
    out = sector_overlay(w, cfg, TICKERS, D, DATA)
    by_sec = out.groupby(DATA.sectors).sum()
    tgt = DATA.sectors.value_counts(normalize=True)
    for s in tgt.index:
        assert by_sec[s] == pytest.approx(tgt[s])


def test_sector_overlay_blend_match_averages_spy_and_qqq_targets(monkeypatch):
    import research.ablation.engine as eng
    members = ["A", "B", "E"]           # 2 Tech + 1 Fin, caps 1e9/2e9/5e9
    monkeypatch.setattr(eng, "_qqq_members", lambda: members)
    cfg = AblationConfig(name="t", sector="blend_match")
    w = pd.Series(1 / len(TICKERS), index=TICKERS)
    out = sector_overlay(w, cfg, TICKERS, D, DATA)
    assert out.sum() == pytest.approx(1.0)

    ucaps = DATA.caps.loc[D]
    spy = ucaps.groupby(DATA.sectors).sum() / ucaps.sum()
    mc = ucaps.reindex(members)
    qqq = mc.groupby(DATA.sectors.reindex(members)).sum() / mc.sum()
    tgt = (0.5 * spy + 0.5 * qqq.reindex(spy.index).fillna(0.0))
    by_sec = out.groupby(DATA.sectors).sum()
    for s in tgt.index:
        assert by_sec[s] == pytest.approx(tgt[s] / tgt.sum())


def test_exclusion_drops_and_demotes():
    hard = apply_exclusion(SCORE, D, DATA.parent_ranks, "any_p10")
    flagged = (pd.DataFrame({p: DATA.parent_ranks[p][D] for p in DATA.parent_ranks})
               < 10).any(axis=1)
    assert set(hard.index) == set(SCORE.index[~flagged])
    soft = apply_exclusion(SCORE, D, DATA.parent_ranks, "soft_p10")
    assert len(soft) == len(SCORE)                       # demoted, not dropped
    assert (soft[flagged] == SCORE[flagged] - 10.0).all()


def test_hysteresis_holds_incumbents():
    s = pd.Series({"A": 90, "B": 80, "C": 70, "D": 60, "E": 50, "F": 40, "G": 30,
                   "H": 20})
    fresh = select_book(s, 0.25, None, [])
    assert fresh == ["A", "B"]
    held = select_book(s, 0.25, 0.50, ["C", "D"])        # incumbents inside top 50%
    assert set(held) == {"C", "D"}
    churned = select_book(s, 0.25, 0.50, ["G", "H"])     # incumbents fell out
    assert churned == ["A", "B"]


def test_cap_within_sectors_enforces_ceiling_and_preserves_sector_totals():
    # Tech sleeve = 60% total with a 40% name; Fin sleeve = 40% with a 30% name.
    w = pd.Series({"A": 0.40, "B": 0.12, "C": 0.08, "E": 0.30, "F": 0.10},
                  index=["A", "B", "C", "E", "F"])
    sectors = pd.Series({"A": "Tech", "B": "Tech", "C": "Tech", "E": "Fin", "F": "Fin"})
    out = cap_within_sectors(w, sectors, 0.25)            # feasible: 3*.25>.60, 2*.25>.40
    assert out.max() <= 0.25 + 1e-9                       # hard ceiling holds
    assert out.sum() == pytest.approx(1.0)
    # sector totals unchanged: Tech stays 0.60, Fin stays 0.40
    by = out.groupby(sectors).sum()
    assert by["Tech"] == pytest.approx(0.60)
    assert by["Fin"] == pytest.approx(0.40)


def test_final_cap_config_binds_on_real_book():
    hard = simulate_config(DATA, AblationConfig(name="hard", weighting="cap",
                                                sector="cap_match", final_cap=0.30))
    soft = simulate_config(DATA, AblationConfig(name="soft", weighting="cap",
                                                sector="cap_match"))
    assert np.isfinite(hard["net_cagr"])
    # the hard cap must diversify: effective N can only rise (or hold)
    assert hard["eff_n"] >= soft["eff_n"] - 1e-6


def test_simulate_config_end_to_end_costs_reduce_returns():
    free = simulate_config(DATA, AblationConfig(name="free", cost_bps=0.0))
    paid = simulate_config(DATA, AblationConfig(name="paid", cost_bps=50.0))
    for r in (free, paid):
        assert np.isfinite(r["net_cagr"]) and np.isfinite(r["sharpe"])
        assert r["avg_names"] == 2                       # top 25% of 8
    assert paid["net_cagr"] < free["net_cagr"]
    assert free["gross_cagr"] == pytest.approx(free["net_cagr"])
