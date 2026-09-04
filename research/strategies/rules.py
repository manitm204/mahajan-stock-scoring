"""The 20 pre-registered holding/exit strategies (user request 2026-09-02),
all built on the SAME production EQEFF composite score -- see engine.py for
the two simulators these dispatch to.

#17/#18 use a VALUE-PARENT-CHEAPNESS PROXY for "analyst price-target upside,"
per the user's explicit choice (2026-09-02): the `analyst_estimates` /
`analyst_estimate_features` tables only hold consensus price targets back to
2026-06-21 (~2 months), not the multi-year history needed for this backtest,
so a literal PT-gap rule is not PIT-testable over 2020-2026. Instead, at
entry we set an assumed target upside proportional to the name's Value-parent
percentile that day: implied_upside = (value_pct / 100) * PROXY_MAX_UPSIDE,
with PROXY_MAX_UPSIDE = 0.40 a fixed, undtuned assumption (roughly the size
of a typical analyst PT gap) -- NOT calibrated on outcome data. This is an
approximation of "how much upside the market's cheapness read implies," not
a real analyst forecast; treat #17/#18 results as illustrative of the RULE
STRUCTURE (partial vs. full target, with a time cap), not as validated PT
strategies.
"""
from __future__ import annotations

from .engine import _months_between, _rank_pct

PROXY_MAX_UPSIDE = 0.40


def _pct_band_rule(up: float | None, down: float | None, cap_months: float | None):
    def rule(pos, today, price, is_review, comp_today) -> bool:
        chg = price / pos.entry_price - 1.0
        if up is not None and chg >= up:
            return True
        if down is not None and chg <= down:
            return True
        if cap_months is not None and _months_between(pos.entry_date, today) >= cap_months:
            return True
        return False
    return rule


def _trailing_stop_rule(dd_pct: float, cap_months: float):
    def rule(pos, today, price, is_review, comp_today) -> bool:
        dd = price / pos.peak_price - 1.0
        if dd <= -abs(dd_pct):
            return True
        if _months_between(pos.entry_date, today) >= cap_months:
            return True
        return False
    return rule


def _pt_proxy_rule(frac_of_target: float, cap_months: float):
    def rule(pos, today, price, is_review, comp_today) -> bool:
        implied_upside = (pos.entry_value_pct / 100.0) * PROXY_MAX_UPSIDE \
            if pos.entry_value_pct == pos.entry_value_pct else 0.0
        target_chg = implied_upside * frac_of_target
        chg = price / pos.entry_price - 1.0
        if chg >= target_chg:
            return True
        if _months_between(pos.entry_date, today) >= cap_months:
            return True
        return False
    return rule


def _score_deterioration_rule(cap_months: float, median_score: float = 50.0):
    def rule(pos, today, price, is_review, comp_today) -> bool:
        if is_review and comp_today is not None:
            sc = comp_today.get(pos.ticker)
            if sc is not None and sc == sc and sc < median_score:
                return True
        if _months_between(pos.entry_date, today) >= cap_months:
            return True
        return False
    return rule


def _rank_hysteresis_rule(exit_pct: float):
    def rule(pos, today, price, is_review, comp_today) -> bool:
        if not is_review or comp_today is None:
            return False
        rp = _rank_pct(comp_today, pos.ticker)
        return rp is None or rp > exit_pct
    return rule


def _trailing_stop_combo_rule(trail_pct: float, cap_months: float,
                              hard_stop_pct: float | None = None,
                              score_floor: float | None = None):
    """Generalized version of #13: trailing stop off the peak, PLUS optionally
    a hard stop-loss measured from the entry price (not the peak) and/or a
    "sell if the composite score craters" review-time exit. Any one firing
    closes the position; the time cap is always the backstop."""
    def rule(pos, today, price, is_review, comp_today) -> bool:
        if price / pos.peak_price - 1.0 <= -abs(trail_pct):
            return True
        if hard_stop_pct is not None and price / pos.entry_price - 1.0 <= -abs(hard_stop_pct):
            return True
        if score_floor is not None and is_review and comp_today is not None:
            sc = comp_today.get(pos.ticker)
            if sc is not None and sc == sc and sc < score_floor:
                return True
        if _months_between(pos.entry_date, today) >= cap_months:
            return True
        return False
    return rule


# --------------------------------------------------------------------------- #
# Strategy registry: name -> (engine kind, kwargs for that engine)
# --------------------------------------------------------------------------- #
CALENDAR_STRATS = {
    "01_monthly_1M":            dict(hold_months=1, sleeve_count=1),
    "02_quarterly_3M":          dict(hold_months=3, sleeve_count=1),
    "03_semiannual_6M":         dict(hold_months=6, sleeve_count=1),
    "04_annual_12M":            dict(hold_months=12, sleeve_count=1),
    "05_sleeves_3M_monthly":    dict(hold_months=3, sleeve_count=3),
    "06_sleeves_6M_monthly":    dict(hold_months=6, sleeve_count=6),
    "07_sleeves_12M_monthly":   dict(hold_months=12, sleeve_count=12),
    "08_sleeves_6M_2x_offset3": dict(hold_months=6, sleeve_count=2),
    "20_buyhold_control":       dict(hold_months=None, sleeve_count=1),
}

MANAGED_STRATS = {
    "09_pm10_cap6M":       _pct_band_rule(0.10, -0.10, 6),
    "10_pm15_cap12M":      _pct_band_rule(0.15, -0.15, 12),
    "11_letwinrun_p25_m10_cap12M": _pct_band_rule(0.25, -0.10, 12),
    "12_sellwin_p10_m20_cap6M":    _pct_band_rule(0.10, -0.20, 6),
    "13_trailstop10_cap12M":       _trailing_stop_rule(0.10, 12),
    "14_pm10_cap3M":               _pct_band_rule(0.10, -0.10, 3),
    "15_rankhyst_p25":             _rank_hysteresis_rule(0.25),
    "16_rankhyst_p40":             _rank_hysteresis_rule(0.40),
    "17_ptproxy_half_cap12M":      _pt_proxy_rule(0.50, 12),
    "18_ptproxy_full_cap6M":       _pt_proxy_rule(1.00, 6),
    "19_scoredeterioration_cap12M": _score_deterioration_rule(12),
}

# --------------------------------------------------------------------------- #
# Follow-up round (user request 2026-09-02): trailing-stop threshold
# sensitivity + stop-loss / score-crash combos layered on top of #13.
# --------------------------------------------------------------------------- #
FOLLOWUP_STRATS = {
    "21_trailstop01_cap12M":  _trailing_stop_combo_rule(0.01, 12),
    "22_trailstop05_cap12M":  _trailing_stop_combo_rule(0.05, 12),
    "23_trailstop20_cap12M":  _trailing_stop_combo_rule(0.20, 12),
    "24_trail10_hardstop20_cap12M":         _trailing_stop_combo_rule(0.10, 12, hard_stop_pct=0.20),
    "25_trail10_scorecrash25_cap12M":       _trailing_stop_combo_rule(0.10, 12, score_floor=25.0),
    "26_trail10_hardstop20_scorecrash25_cap12M": _trailing_stop_combo_rule(
        0.10, 12, hard_stop_pct=0.20, score_floor=25.0),
}
