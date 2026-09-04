from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research import compute_forward_returns
from research.panel import ScorePanel
from research.subset_selection import correlation_matrix
from research.walkforward.regime_aware_evidence import build_monthly_cache
from research.walkforward.regime_aware_walkforward import (
    VARIANTS, _apply_floor, _apply_variant_ic, _incremental_correlation_matrix,
    _incremental_forward_returns, _incremental_parent_cache, _is_quarter_boundary,
    _mean_abs_weight_change, _percentile_regime_columns, _percentile_vix_thresholds,
    _possible_sign_inversion, _ParentCacheState, _regime_only_ic,
    _subfactor_churn_per_year, run_monthly_variant, run_walkforward_comparison,
)
from research.walkforward.regime_probability import (
    REGIME_ORDER as _REGIME_ORDER, regime_probabilities as _regime_probabilities,
    shrink_regime_ic as _shrink_regime_ic,
)


def test_is_quarter_boundary():
    assert _is_quarter_boundary("2020-01-31")
    assert _is_quarter_boundary("2020-04-30")
    assert _is_quarter_boundary("2020-07-31")
    assert _is_quarter_boundary("2020-10-31")
    assert not _is_quarter_boundary("2020-02-29")
    assert not _is_quarter_boundary("2020-06-30")


def test_apply_variant_ic_long_run_mode():
    evidence = pd.DataFrame({"expected_ic": [0.05], "long_run_mean_ic": [0.02],
                             "recent_24m_ic": [0.09]})
    out = _apply_variant_ic(evidence, "long_run")
    assert out["expected_ic"].iloc[0] == pytest.approx(0.02)


def test_apply_variant_ic_recent_mode_falls_back_to_long_run():
    evidence = pd.DataFrame({"expected_ic": [0.05], "long_run_mean_ic": [0.02],
                             "recent_24m_ic": [np.nan]})
    out = _apply_variant_ic(evidence, "recent")
    assert out["expected_ic"].iloc[0] == pytest.approx(0.02)


def test_apply_variant_ic_full_mode_is_unchanged():
    evidence = pd.DataFrame({"expected_ic": [0.05], "long_run_mean_ic": [0.02],
                             "recent_24m_ic": [0.09]})
    out = _apply_variant_ic(evidence, "full")
    assert out["expected_ic"].iloc[0] == pytest.approx(0.05)


def _regime_evidence_row() -> dict:
    return {
        "long_run_mean_ic": 0.02, "recent_24m_ic": 0.09,
        "regime_ic_low": 0.05, "regime_n_eff_low": 40,
        "regime_ic_medium": 0.01, "regime_n_eff_medium": 20,
        "regime_ic_high": -0.02, "regime_n_eff_high": 5,
    }


def test_regime_only_ic_matches_hand_computed_probability_weighted_shrinkage():
    row = pd.Series(_regime_evidence_row())
    vix_now = 18.0
    out = _regime_only_ic(row, vix_now)

    probs = _regime_probabilities(vix_now)
    expected = sum(
        probs[r] * _shrink_regime_ic(row[f"regime_ic_{r.split(' ')[0].lower()}"],
                                     row["long_run_mean_ic"],
                                     row[f"regime_n_eff_{r.split(' ')[0].lower()}"])
        for r in _REGIME_ORDER)
    assert out == pytest.approx(expected)


def test_regime_only_ic_nan_long_run_returns_nan():
    row = pd.Series({**_regime_evidence_row(), "long_run_mean_ic": float("nan")})
    assert np.isnan(_regime_only_ic(row, 18.0))


def test_apply_variant_ic_regime_only_mode():
    evidence = pd.DataFrame({"expected_ic": [0.05], **{k: [v] for k, v in _regime_evidence_row().items()}})
    out = _apply_variant_ic(evidence, "regime_only", vix_now=18.0)
    expected = _regime_only_ic(evidence.iloc[0], 18.0)
    assert out["expected_ic"].iloc[0] == pytest.approx(expected)


def test_apply_variant_ic_regime_recent_mode_averages_regime_and_recent():
    evidence = pd.DataFrame({"expected_ic": [0.05], **{k: [v] for k, v in _regime_evidence_row().items()}})
    out = _apply_variant_ic(evidence, "regime_recent", vix_now=18.0)
    regime = _regime_only_ic(evidence.iloc[0], 18.0)
    assert out["expected_ic"].iloc[0] == pytest.approx(0.5 * regime + 0.5 * 0.09)


def test_apply_variant_ic_regime_longrun_mode_averages_regime_and_longrun():
    evidence = pd.DataFrame({"expected_ic": [0.05], **{k: [v] for k, v in _regime_evidence_row().items()}})
    out = _apply_variant_ic(evidence, "regime_longrun", vix_now=18.0)
    regime = _regime_only_ic(evidence.iloc[0], 18.0)
    assert out["expected_ic"].iloc[0] == pytest.approx(0.5 * regime + 0.5 * 0.02)


def test_regime_only_ic_low_high_override_changes_result():
    row = pd.Series(_regime_evidence_row())
    vix_now = 22.0     # inside the fixed Medium band but past a lower percentile "high"
    default = _regime_only_ic(row, vix_now)
    overridden = _regime_only_ic(row, vix_now, low=10.0, high=15.0)
    assert default != pytest.approx(overridden)


def test_percentile_vix_thresholds_matches_hand_computed_quantiles():
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    vix = pd.Series(range(1, 101), index=dates, dtype=float)   # 1..100, evenly spread
    cutoff = dates[59].date().isoformat()                       # first 60 obs: 1..60

    low, high = _percentile_vix_thresholds(vix, cutoff, panel_inception="2020-01-01")

    hist = vix.loc[:dates[59]]
    assert low == pytest.approx(hist.quantile(0.20))
    assert high == pytest.approx(hist.quantile(0.80))


def test_percentile_vix_thresholds_falls_back_when_no_history():
    from research.walkforward.regime_probability import REGIME_HIGH, REGIME_LOW
    vix = pd.Series([20.0], index=pd.to_datetime(["2020-01-01"]))
    low, high = _percentile_vix_thresholds(vix, "2019-01-01", panel_inception="2020-01-01")
    assert (low, high) == (REGIME_LOW, REGIME_HIGH)


def test_percentile_regime_columns_reclassifies_using_given_thresholds():
    raw = pd.DataFrame({
        "sub_factor": ["s1", "s1"], "date": ["2020-01-31", "2020-02-29"],
        "vix_level": [10.0, 30.0], "ic_3M": [0.10, -0.10], "ic_6M": [0.10, -0.10],
    })
    low, high = 15.0, 25.0
    out = _percentile_regime_columns(raw, "2020-02-29", low, high,
                                     panel_inception="2020-01-01")
    row = out.set_index("sub_factor").loc["s1"]

    p1, p2 = _regime_probabilities(10.0, low=low, high=high), _regime_probabilities(30.0, low=low, high=high)
    ic1, ic2 = 0.10, -0.10
    for key, r in zip(("low", "medium", "high"), _REGIME_ORDER):
        n_eff = p1[r] + p2[r]
        expected_ic = (p1[r] * ic1 + p2[r] * ic2) / n_eff if n_eff > 1e-9 else float("nan")
        assert row[f"regime_n_eff_{key}"] == pytest.approx(n_eff)
        assert row[f"regime_ic_{key}"] == pytest.approx(expected_ic)


def test_percentile_regime_columns_empty_when_no_history_in_range():
    raw = pd.DataFrame({"sub_factor": ["s1"], "date": ["2020-01-31"],
                        "vix_level": [10.0], "ic_3M": [0.1], "ic_6M": [0.1]})
    out = _percentile_regime_columns(raw, "2019-01-01", 15.0, 25.0)
    assert out.empty
    assert "regime_ic_low" in out.columns


def test_apply_floor_raises_small_positive_weights():
    weights = {"p1": 0.01, "p2": 0.50, "p3": 0.49}
    out = _apply_floor(weights, 0.02)
    assert out["p1"] == pytest.approx(0.02)
    assert abs(sum(out.values()) - 1.0) < 1e-9


def test_apply_floor_no_op_when_all_above_floor():
    weights = {"p1": 0.5, "p2": 0.5}
    assert _apply_floor(weights, 0.02) == weights


def test_possible_sign_inversion_flags_negative_both_horizons():
    p_scored = pd.DataFrame({
        "sub_factor": ["a", "b", "c", "d"],
        "long_run_mean_ic": [-0.05, -0.05, 0.01, -0.05],
        "recent_24m_ic": [-0.06, 0.01, -0.05, -0.06],
        "expected_ic": [-0.03, -0.03, -0.03, -0.01],
    })
    out = _possible_sign_inversion(p_scored)
    assert list(out) == [True, False, False, False]


def _toy_state_log() -> pd.DataFrame:
    """4 months x 2 parents. p1/p2 weights each move by known absolute steps
    (0.05, 0.05, 0.10 per parent -- 6 diffs pooled, mean 0.4/6); active_subs
    membership changes by a known number of entry/exit events per parent
    (p1: 0+1+1=2, p2: 1+0+3=4 -- 6 events total over 90 days)."""
    cutoffs = ["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30"]
    p1_w = [0.6, 0.65, 0.60, 0.70]
    p2_w = [0.4, 0.35, 0.40, 0.30]
    p1_subs = ["subA,subB", "subA,subB", "subA", "subA,subC"]
    p2_subs = ["subX", "subX,subY", "subX,subY", "subZ"]
    rows = []
    for i, c in enumerate(cutoffs):
        rows.append({"cutoff": c, "variant": "C", "parent": "p1",
                     "weight": p1_w[i], "active_subs": p1_subs[i]})
        rows.append({"cutoff": c, "variant": "C", "parent": "p2",
                     "weight": p2_w[i], "active_subs": p2_subs[i]})
    return pd.DataFrame(rows)


def test_mean_abs_weight_change_computes_pooled_mean():
    log = _toy_state_log()
    out = _mean_abs_weight_change(log, "C", "2020-04-30")
    assert out == pytest.approx(0.4 / 6)


def test_mean_abs_weight_change_unlogged_variant_returns_nan():
    log = _toy_state_log()
    assert np.isnan(_mean_abs_weight_change(log, "B", "2020-04-30"))


def test_mean_abs_weight_change_single_month_returns_nan():
    log = _toy_state_log()
    single = log[log["cutoff"] == "2020-01-31"]
    assert np.isnan(_mean_abs_weight_change(single, "C", "2020-01-31"))


def test_subfactor_churn_per_year_computes_annualised_events():
    log = _toy_state_log()
    out = _subfactor_churn_per_year(log, "C", "2020-04-30")
    days = (pd.Timestamp("2020-04-30") - pd.Timestamp("2020-01-31")).days
    assert out == pytest.approx(6 / (days / 365.25))


def test_subfactor_churn_per_year_unlogged_variant_returns_nan():
    log = _toy_state_log()
    assert np.isnan(_subfactor_churn_per_year(log, "B", "2020-04-30"))


def test_subfactor_churn_per_year_single_month_returns_nan():
    log = _toy_state_log()
    single = log[log["cutoff"] == "2020-01-31"]
    assert np.isnan(_subfactor_churn_per_year(single, "C", "2020-01-31"))


UNIVERSE2 = [f"T{i}" for i in range(30)]
DATES2 = pd.date_range("2018-01-31", periods=30, freq="ME").strftime("%Y-%m-%d").tolist()


def _toy_multi_parent_panel(dates: list[str], universe: list[str]) -> ScorePanel:
    """Two parents, each with one positive-IC sub ("*_up", ranks ascending with
    ticker order) and one negative-IC sub ("*_down", ranks descending) -- gives each
    parent exactly one eligible candidate, so selection/hysteresis behaviour stays
    deterministic and easy to reason about."""
    scores = {}
    for k, d in enumerate(dates):
        base = np.linspace(10, 90, len(universe)) + k
        scores[d] = pd.DataFrame({
            "p1_up": base, "p1_down": base[::-1],
            "p2_up": base, "p2_down": base[::-1],
        }, index=universe)
    return ScorePanel(rebal_dates=list(dates), scores=scores,
                      parent_keys=["p1", "p2"],
                      sub_by_parent={"p1": ["p1_up", "p1_down"], "p2": ["p2_up", "p2_down"]},
                      universe=list(universe))


def _toy_matrix2(dates: list[str], universe: list[str]) -> pd.DataFrame:
    idx = pd.date_range(dates[0], periods=len(dates) + 8, freq="ME").strftime("%Y-%m-%d")
    data = {t: (1.01 + i * 0.002) ** np.arange(len(idx)) for i, t in enumerate(universe)}
    return pd.DataFrame(data, index=idx)


def test_incremental_correlation_matrix_matches_full_recompute():
    dates = DATES2[:12]
    expected = correlation_matrix(_toy_multi_parent_panel(dates, UNIVERSE2))

    panel = _toy_multi_parent_panel(DATES2, UNIVERSE2)
    cache: dict = {}
    result = None
    for i in range(1, len(dates) + 1):     # simulate the walk-forward's growing window
        result = _incremental_correlation_matrix(panel, dates[:i], cache)

    pd.testing.assert_frame_equal(result.sort_index(axis=0).sort_index(axis=1),
                                  expected.sort_index(axis=0).sort_index(axis=1))
    assert set(cache) == set(dates)


def test_incremental_correlation_matrix_reuses_cached_dates():
    panel = _toy_multi_parent_panel(DATES2, UNIVERSE2)
    cache: dict = {}
    _incremental_correlation_matrix(panel, DATES2[:5], cache)
    cached = cache[DATES2[0]].copy()

    # If the helper recomputed an already-cached date instead of reusing it, this
    # mutation (which makes every column constant -- unusable, nunique() == 1)
    # would flip the cached entry to the "unusable" sentinel (None).
    panel.scores[DATES2[0]] = panel.scores[DATES2[0]] * 0 + 50.0
    _incremental_correlation_matrix(panel, DATES2[:5], cache)

    pd.testing.assert_frame_equal(cache[DATES2[0]], cached)


def test_incremental_forward_returns_matches_full_recompute():
    dates = DATES2[:15]
    matrix = _toy_matrix2(DATES2, UNIVERSE2)
    horizons = {"3M": 3, "6M": 6}
    expected = compute_forward_returns(matrix, dates, horizons)

    cache: dict = {}
    result = None
    for i in range(1, len(dates) + 1):
        result = _incremental_forward_returns(matrix, dates[:i], horizons, cache)

    for h in horizons:
        assert set(result[h]) == set(expected[h])
        for d in expected[h]:
            pd.testing.assert_series_equal(
                result[h][d].sort_index(), expected[h][d].sort_index())


def test_incremental_parent_cache_appends_new_dates_when_weights_unchanged():
    from research.walkforward.compose import build_parent_panel
    from research.walkforward.selection import slice_panel

    def _norm(df):
        return df.sort_values(["sub_factor", "date"]).reset_index(drop=True)

    panel = _toy_multi_parent_panel(DATES2, UNIVERSE2)
    matrix = _toy_matrix2(DATES2, UNIVERSE2)
    vix = pd.Series(20.0, index=pd.to_datetime(matrix.index))
    sub_weights = {"p1": {"p1_up": 1.0}, "p2": {"p2_up": 1.0}}

    state = _ParentCacheState()
    parent_panel_1 = build_parent_panel(slice_panel(panel, DATES2[:10]), sub_weights)
    _incremental_parent_cache(parent_panel_1, matrix, vix, sub_weights, state)

    parent_panel_2 = build_parent_panel(slice_panel(panel, DATES2[:12]), sub_weights)
    out2 = _incremental_parent_cache(parent_panel_2, matrix, vix, sub_weights, state)

    expected_full = build_monthly_cache(parent_panel_2, matrix, vix)
    pd.testing.assert_frame_equal(_norm(out2), _norm(expected_full))
    assert state.dates == set(DATES2[:12])


def test_incremental_parent_cache_rebuilds_when_weights_change():
    from research.walkforward.compose import build_parent_panel
    from research.walkforward.selection import slice_panel

    def _norm(df):
        return df.sort_values(["sub_factor", "date"]).reset_index(drop=True)

    panel = _toy_multi_parent_panel(DATES2, UNIVERSE2)
    matrix = _toy_matrix2(DATES2, UNIVERSE2)
    vix = pd.Series(20.0, index=pd.to_datetime(matrix.index))
    dates = DATES2[:10]

    state = _ParentCacheState()
    weights_1 = {"p1": {"p1_up": 1.0}, "p2": {"p2_up": 1.0}}
    parent_panel_1 = build_parent_panel(slice_panel(panel, dates), weights_1)
    _incremental_parent_cache(parent_panel_1, matrix, vix, weights_1, state)

    weights_2 = {"p1": {"p1_up": 0.6, "p1_down": 0.4}, "p2": {"p2_up": 1.0}}
    parent_panel_2 = build_parent_panel(slice_panel(panel, dates), weights_2)
    out = _incremental_parent_cache(parent_panel_2, matrix, vix, weights_2, state)

    expected = build_monthly_cache(parent_panel_2, matrix, vix)
    pd.testing.assert_frame_equal(_norm(out), _norm(expected))


def test_run_monthly_variant_smoke_variant_c(capsys):
    panel = _toy_multi_parent_panel(DATES2, UNIVERSE2)
    matrix = _toy_matrix2(DATES2, UNIVERSE2)
    vix = pd.Series(20.0, index=pd.to_datetime(matrix.index))
    sub_cache = build_monthly_cache(panel, matrix, vix)
    boundary = DATES2[25]

    snapshots = run_monthly_variant(panel, matrix, vix, sub_cache, VARIANTS["C"],
                                    test_boundaries=[boundary], verbose=True)

    assert boundary in snapshots
    cfg = snapshots[boundary]
    assert abs(sum(cfg.parent_weights.values()) - 1.0) < 1e-6
    assert all(w == w for w in cfg.parent_weights.values())     # no NaN
    for subw in cfg.sub_weights.values():
        if subw:
            assert abs(sum(subw.values()) - 1.0) < 1e-6

    out = capsys.readouterr().out
    assert "[C] month 1/" in out


def test_run_monthly_variant_smoke_variant_d_no_hysteresis():
    panel = _toy_multi_parent_panel(DATES2, UNIVERSE2)
    matrix = _toy_matrix2(DATES2, UNIVERSE2)
    vix = pd.Series(20.0, index=pd.to_datetime(matrix.index))
    sub_cache = build_monthly_cache(panel, matrix, vix)
    boundary = DATES2[25]

    snapshots = run_monthly_variant(panel, matrix, vix, sub_cache, VARIANTS["D"],
                                    test_boundaries=[boundary])

    assert boundary in snapshots
    assert abs(sum(snapshots[boundary].parent_weights.values()) - 1.0) < 1e-6


def test_run_monthly_variant_populates_state_log_every_month():
    panel = _toy_multi_parent_panel(DATES2, UNIVERSE2)
    matrix = _toy_matrix2(DATES2, UNIVERSE2)
    vix = pd.Series(20.0, index=pd.to_datetime(matrix.index))
    sub_cache = build_monthly_cache(panel, matrix, vix)
    boundary = DATES2[25]
    state_log: list[dict] = []

    run_monthly_variant(panel, matrix, vix, sub_cache, VARIANTS["C"],
                        test_boundaries=[boundary], state_log=state_log)

    dates_up_to_boundary = [d for d in DATES2 if d <= boundary]
    assert len(state_log) == len(dates_up_to_boundary) * len(panel.parent_keys)
    log_df = pd.DataFrame(state_log)
    assert set(log_df["cutoff"].unique()) == set(dates_up_to_boundary)
    assert set(log_df["parent"].unique()) == set(panel.parent_keys)
    assert (log_df["variant"] == "C").all()


def test_run_walkforward_comparison_smoke():
    panel = _toy_multi_parent_panel(DATES2, UNIVERSE2)
    matrix = _toy_matrix2(DATES2, UNIVERSE2)
    vix = pd.Series(20.0, index=pd.to_datetime(matrix.index))
    sectors = pd.Series("Sector1", index=UNIVERSE2)

    out, state_log = run_walkforward_comparison(panel, matrix, vix, sectors,
                                                first_test_year=2019, last_end="2019-12-31",
                                                verbose=False)

    assert not out.empty
    assert set(out["variant"]) == set(VARIANTS) | {"A"}
    assert {"window", "ic_6m", "cagr", "sharpe", "max_drawdown",
            "spy_beta", "spy_alpha", "mean_abs_weight_chg",
            "subfactor_churn_per_year"}.issubset(out.columns)
    assert out["ic_6m"].notna().any()

    assert not state_log.empty
    assert {"cutoff", "variant", "parent", "weight", "active_subs"}.issubset(state_log.columns)


def test_run_walkforward_comparison_respects_variant_names_and_no_baseline():
    panel = _toy_multi_parent_panel(DATES2, UNIVERSE2)
    matrix = _toy_matrix2(DATES2, UNIVERSE2)
    vix = pd.Series(20.0, index=pd.to_datetime(matrix.index))
    sectors = pd.Series("Sector1", index=UNIVERSE2)

    out, _ = run_walkforward_comparison(
        panel, matrix, vix, sectors, first_test_year=2019, last_end="2019-12-31",
        verbose=False, variant_names=["B", "C", "24M"], include_baseline_a=False)

    assert not out.empty
    assert set(out["variant"]) == {"B", "C", "24M"}


def test_run_monthly_variant_shares_select_config_cache_across_calls(monkeypatch):
    import research.walkforward.regime_aware_walkforward as raw

    panel = _toy_multi_parent_panel(DATES2, UNIVERSE2)
    matrix = _toy_matrix2(DATES2, UNIVERSE2)
    vix = pd.Series(20.0, index=pd.to_datetime(matrix.index))
    sub_cache = build_monthly_cache(panel, matrix, vix)
    boundary = DATES2[10]
    walked_months = len([d for d in DATES2 if d <= boundary])

    calls: list = []
    real_select_config = raw.select_config

    def _spy(*args, **kwargs):
        calls.append(1)
        return real_select_config(*args, **kwargs)

    monkeypatch.setattr(raw, "select_config", _spy)

    cache: dict = {}
    raw.run_monthly_variant(panel, matrix, vix, sub_cache, VARIANTS["B"],
                            test_boundaries=[boundary], select_config_cache=cache)
    assert len(calls) == walked_months

    raw.run_monthly_variant(panel, matrix, vix, sub_cache, VARIANTS["C"],
                            test_boundaries=[boundary], select_config_cache=cache)
    assert len(calls) == walked_months     # second variant reused the cache -- zero new calls


def test_run_monthly_variant_threads_recent_months_into_as_of(monkeypatch):
    """VariantSpec.recent_months must reach BOTH as_of() call sites (sub-level and
    parent-level) so a "5Y"/"12M" variant's window actually differs from the
    default 24M -- this is the one part of the new ablation battery that's easy to
    silently wire up wrong (as_of()'s "recent_24m_ic" column keeps its name
    regardless of the window used, so a wiring bug wouldn't show up as a crash)."""
    import research.walkforward.regime_aware_walkforward as raw

    panel = _toy_multi_parent_panel(DATES2, UNIVERSE2)
    matrix = _toy_matrix2(DATES2, UNIVERSE2)
    vix = pd.Series(20.0, index=pd.to_datetime(matrix.index))
    sub_cache = build_monthly_cache(panel, matrix, vix)
    boundary = DATES2[5]

    seen: list[int] = []
    real_as_of = raw.as_of

    def _spy(*args, **kwargs):
        seen.append(kwargs.get("recent_months"))
        return real_as_of(*args, **kwargs)

    monkeypatch.setattr(raw, "as_of", _spy)
    raw.run_monthly_variant(panel, matrix, vix, sub_cache, VARIANTS["5Y"],
                            test_boundaries=[boundary])
    assert seen and all(m == 60 for m in seen)


def test_run_monthly_variant_threads_base_weight_frac_into_blend(monkeypatch):
    import research.walkforward.regime_aware_walkforward as raw

    panel = _toy_multi_parent_panel(DATES2, UNIVERSE2)
    matrix = _toy_matrix2(DATES2, UNIVERSE2)
    vix = pd.Series(20.0, index=pd.to_datetime(matrix.index))
    sub_cache = build_monthly_cache(panel, matrix, vix)
    boundary = DATES2[5]

    seen: list[float] = []
    real_blend = raw.blend_parent_weights

    def _spy(*args, **kwargs):
        seen.append(kwargs.get("base_weight_frac"))
        return real_blend(*args, **kwargs)

    monkeypatch.setattr(raw, "blend_parent_weights", _spy)
    raw.run_monthly_variant(panel, matrix, vix, sub_cache, VARIANTS["B"],
                            test_boundaries=[boundary])
    assert seen and all(f == 0.50 for f in seen)


def test_new_ablation_variants_run_end_to_end():
    panel = _toy_multi_parent_panel(DATES2, UNIVERSE2)
    matrix = _toy_matrix2(DATES2, UNIVERSE2)
    vix = pd.Series(20.0, index=pd.to_datetime(matrix.index))
    sub_cache = build_monthly_cache(panel, matrix, vix)
    boundary = DATES2[20]

    for name in ("5Y", "12M", "VIXOnly", "VIX+12M", "VIX+LongRun"):
        snapshots = run_monthly_variant(panel, matrix, vix, sub_cache, VARIANTS[name],
                                        test_boundaries=[boundary])
        assert boundary in snapshots
        cfg = snapshots[boundary]
        assert abs(sum(cfg.parent_weights.values()) - 1.0) < 1e-6
        assert all(w == w for w in cfg.parent_weights.values())     # no NaN


_CACHE = Path("cache/subfactor_expansion/cand_panel_2015-06-30_2026-06-30_monthly_v2.pkl")


@pytest.mark.slow
@pytest.mark.skipif(not _CACHE.exists(), reason="candidate panel cache not built")
def test_run_monthly_variant_on_real_panel_produces_valid_weights():
    """DB-guarded integration check: the continuous monthly loop (variant C) runs
    end to end on the real cached candidate panel without crashing, and produces a
    well-formed FrozenConfig at the latest available cutoff.

    This intentionally steps through the ENTIRE ~130-month history once (the
    monthly loop is stateful and can't skip ahead), calling select_config every
    month for the rolling-5Y base weight -- so this test is genuinely slow
    (several minutes). That per-month select_config cost is a known follow-up
    optimisation (caching/memoizing across nearby months), flagged in the design
    spec's "Out of scope" section, not something this plan fixes.
    """
    from backtesting import data_loader as dl
    from data.db import get_db
    from research.subfactor_expansion.panel import load_cached_panel as eload
    from research.walkforward.regime_aware_walkforward import VARIANTS, run_monthly_variant
    from research.walkforward.vix_regime_study import load_vix_series
    from run_walkforward import _adapt

    panel = _adapt(eload(_CACHE))
    with get_db() as db:
        matrix = dl.load_price_matrix(db, panel.universe, "2015-06-30", "2026-07-06")
        vix = load_vix_series(db)

    sub_cache = build_monthly_cache(panel, matrix, vix)
    cutoff = panel.rebal_dates[-1]

    snapshots = run_monthly_variant(panel, matrix, vix, sub_cache, VARIANTS["C"],
                                    test_boundaries=[cutoff], verbose=True)

    assert cutoff in snapshots
    cfg = snapshots[cutoff]
    assert cfg.parent_weights
    assert abs(sum(cfg.parent_weights.values()) - 1.0) < 1e-6
    assert max(cfg.parent_weights.values()) <= 0.25 + 1e-6
    assert all(w == w for w in cfg.parent_weights.values())     # no NaN
    for parent, subw in cfg.sub_weights.items():
        if subw:
            assert abs(sum(subw.values()) - 1.0) < 1e-6
            # The 50% single-subfactor cap only constrains parents with >=2
            # selected subs -- with exactly one selected sub, the cap-then-
            # renormalise-to-1.0 steps in parent_subfactor_weights necessarily
            # send it back to 100% (there's no second sub to hold the other
            # 50%), and the design spec explicitly allows selection to stop at
            # one sub ("do not force two or three subfactors").
            if len(subw) >= 2:
                assert max(subw.values()) <= 0.50 + 1e-6
