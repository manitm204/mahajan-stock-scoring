"""VIX-relative overlay: percentiles, regime utilities, weight blending.

All statistics here consume *training-window* data only: the VIX distribution is the
daily closes inside ``[train_start, train_end]``, comparable observations are training
stat rebalances, and parent IC/IR/Q5-Q1 come from the boundary-truncated 1M forward
returns cached in :class:`vixtilt.baseline.WindowBaseline`. The only test-period input
is the spot VIX known on the rebalance date itself.

Regime utility per parent (over comparable observations):
    0.50 · rank(mean IC) + 0.25 · rank(IC-IR) + 0.25 · rank(mean Q5-Q1 spread)
ranks scaled to [0, 1] (worst parent → 0). Utilities → weights ∝ utility, capped at
25 % per parent, renormalised. Thin samples (n < ``MIN_REGIME_OBS``) shrink the regime
utility toward the full-training utility by n / (n + MIN_REGIME_OBS); n = 0 falls back
to baseline weights. Blend: adjusted = (1-s)·baseline + s·regime.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .baseline import WindowBaseline

PARENT_CAP = 0.25
MIN_REGIME_OBS = 12
MIN_NAMES = 20           # min cross-section for a per-date IC / spread
NO_TILT_LO, NO_TILT_HI = 0.30, 0.70   # dead zone: no tilt in the broad middle
TAIL_LO, TAIL_HI = 0.30, 0.70        # comparable-obs tail definition (same side)
VIX_LOW, VIX_HIGH = 15.0, 25.0       # fixed-bucket thresholds (existing convention)


# --------------------------------------------------------------------------- #
# VIX helpers
# --------------------------------------------------------------------------- #
def vix_spot(vix: pd.Series, date: str) -> tuple[str, float]:
    """Last VIX close on/before ``date`` → (close_date, value). Asserts no look-ahead."""
    sub = vix[vix.index <= pd.Timestamp(date)]
    if sub.empty:
        return date, float("nan")
    asof = sub.index[-1]
    assert asof <= pd.Timestamp(date), "VIX spot taken from the future"
    return asof.date().isoformat(), float(sub.iloc[-1])


def train_vix_values(vix: pd.Series, train_start: str, train_end: str) -> np.ndarray:
    sub = vix[(vix.index >= pd.Timestamp(train_start))
              & (vix.index <= pd.Timestamp(train_end))]
    assert sub.empty or sub.index.max() <= pd.Timestamp(train_end), \
        "training VIX distribution leaks past train_end"
    return sub.to_numpy(dtype=float)


def percentile_of(values: np.ndarray, x: float) -> float:
    if len(values) == 0 or not np.isfinite(x):
        return float("nan")
    return float(np.mean(values <= x))


def fixed_bucket(x: float) -> str:
    if not np.isfinite(x):
        return "Unknown"
    if x < VIX_LOW:
        return "Low"
    if x <= VIX_HIGH:
        return "Medium"
    return "High"


def pctile_strength(pct: float, scale: float = 1.0) -> float:
    """Tiered overlay strength from the training-relative VIX percentile."""
    if not np.isfinite(pct):
        return 0.0
    if NO_TILT_LO <= pct <= NO_TILT_HI:
        s = 0.0
    elif 0.15 <= pct < NO_TILT_LO or NO_TILT_HI < pct <= 0.85:
        s = 0.10
    elif 0.05 <= pct < 0.15 or 0.85 < pct <= 0.95:
        s = 0.20
    else:
        s = 0.30
    return min(1.0, s * scale)


# --------------------------------------------------------------------------- #
# Parent regime statistics → utility → weights
# --------------------------------------------------------------------------- #
def _spearman_ic(scores: pd.Series, fwd: pd.Series) -> float:
    df = pd.concat([scores.rename("s"), fwd.rename("f")], axis=1).dropna()
    if len(df) < MIN_NAMES or df["s"].nunique() < 2:
        return float("nan")
    ic = df["s"].corr(df["f"], method="spearman")
    return float(ic) if pd.notna(ic) else float("nan")


def _q5q1_spread(scores: pd.Series, fwd: pd.Series) -> float:
    df = pd.concat([scores.rename("s"), fwd.rename("f")], axis=1).dropna()
    if len(df) < MIN_NAMES:
        return float("nan")
    r = df["s"].rank(method="first")
    q = pd.qcut(r, 5, labels=False)
    return float(df["f"][q == 4].mean() - df["f"][q == 0].mean())


def parent_stats(dates: list[str], parent_train: dict[str, pd.DataFrame],
                 fwd1m: dict[str, pd.Series], parents: list[str]) -> pd.DataFrame:
    """Per-parent mean IC / IC-IR / mean Q5-Q1 over the given training dates (1M fwd)."""
    ics: dict[str, list[float]] = {p: [] for p in parents}
    sps: dict[str, list[float]] = {p: [] for p in parents}
    for d in dates:
        frame, fwd = parent_train.get(d), fwd1m.get(d)
        if frame is None or fwd is None:
            continue
        for p in parents:
            if p not in frame.columns:
                continue
            ic = _spearman_ic(frame[p], fwd)
            sp = _q5q1_spread(frame[p], fwd)
            if np.isfinite(ic):
                ics[p].append(ic)
            if np.isfinite(sp):
                sps[p].append(sp)
    rows = []
    for p in parents:
        arr = np.array(ics[p], dtype=float)
        mean_ic = float(arr.mean()) if len(arr) else float("nan")
        std = float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")
        ir = mean_ic / std if np.isfinite(std) and std > 1e-9 else float("nan")
        spread = float(np.mean(sps[p])) if sps[p] else float("nan")
        rows.append({"parent": p, "mean_ic": mean_ic, "ic_ir": ir,
                     "q5q1": spread, "n_obs": len(arr)})
    return pd.DataFrame(rows).set_index("parent")


def _rank01(s: pd.Series) -> pd.Series:
    """Ranks rescaled to [0, 1] (worst → 0, best → 1); NaN → neutral 0.5."""
    r = s.rank(method="average")
    n = r.notna().sum()
    if n <= 1:
        return pd.Series(0.5, index=s.index)
    return ((r - 1) / (n - 1)).fillna(0.5)


def utility_from_stats(stats: pd.DataFrame) -> pd.Series:
    return (0.50 * _rank01(stats["mean_ic"])
            + 0.25 * _rank01(stats["ic_ir"])
            + 0.25 * _rank01(stats["q5q1"]))


def cap_renorm(weights: dict[str, float], cap: float = PARENT_CAP) -> dict[str, float]:
    """Water-fill weights to the per-parent cap and renormalise to 1."""
    w = {p: float(weights[p]) for p in sorted(weights) if weights[p] > 0}
    if not w:
        return {}
    tot = sum(w.values())
    w = {p: x / tot for p, x in w.items()}
    for _ in range(20):
        over = [p for p, x in w.items() if x > cap + 1e-12]
        if not over:
            break
        excess = sum(w[p] - cap for p in over)
        for p in over:
            w[p] = cap
        under = [p for p, x in w.items() if x < cap - 1e-12]
        pool = sum(w[p] for p in under)
        if not under or pool <= 1e-12:
            break
        for p in under:
            w[p] += excess * w[p] / pool
    tot = sum(w.values())
    return {p: x / tot for p, x in w.items()}


def regime_weights_from_utility(utility: pd.Series) -> dict[str, float]:
    pos = utility[utility > 0]
    if pos.empty:
        return {}
    return cap_renorm(pos.to_dict())


# --------------------------------------------------------------------------- #
# Variants
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VariantSpec:
    name: str
    kind: str            # "baseline" | "fixed" | "pctile" | "literature"
    scale: float = 1.0   # tier multiplier for pctile variants
    fixed_strength: float = 0.30
    trigger: str = "pct"  # literature variants: "pct" (training pctile) | "level" (VIX pts)
    desc: str = ""
    # lit_band transition knots (sensitivity grid varies these; defaults = production)
    lo_full: float = 13.0
    lo_edge: float = 15.0
    hi_edge: float = 23.0
    hi_full: float = 27.0


DEFAULT_VARIANTS: list[VariantSpec] = [
    VariantSpec("baseline",     "baseline",
                desc="frozen rolling-5y parent weights, no VIX input"),
    VariantSpec("fixed30",      "fixed", fixed_strength=0.30,
                desc="Low/Med/High spot-VIX bucket (15/25); regime weights from "
                     "same-bucket training obs; strength 30 %"),
    VariantSpec("pctile_mild",  "pctile", scale=0.5,
                desc="training-relative VIX percentile tiers ×0.5 (0/5/10/15 %)"),
    VariantSpec("pctile_std",   "pctile", scale=1.0,
                desc="percentile tiers: 30-70→0 %, 15-30/70-85→10 %, 5-15/85-95→20 %, "
                     "tails→30 %"),
    VariantSpec("pctile_strong","pctile", scale=2.0,
                desc="percentile tiers ×2 (0/20/40/60 %)"),
]

# Pre-registered literature rule (fixed multipliers, no estimation, no tuning):
# high VIX: momentum ×0.5, freed weight → quality+value pro-rata;
# low VIX:  momentum ×1.25, value ×0.75; middle: baseline untouched.
LIT_MOM_CUT, LIT_MOM_BOOST, LIT_VAL_CUT = 0.5, 1.25, 0.75
LIT_PCT_LO, LIT_PCT_HI = 0.15, 0.85
LIT_LVL_LO, LIT_LVL_HI = 15.0, 25.0

LIT_VARIANTS: list[VariantSpec] = [
    DEFAULT_VARIANTS[0],
    VariantSpec("lit_pct", "literature", trigger="pct",
                desc="literature rule (mom ×0.5→quality/value in high, mom ×1.25 / "
                     "value ×0.75 in low); trigger = spot VIX <15th / >85th training "
                     "percentile"),
    VariantSpec("lit_fix", "literature", trigger="level",
                desc="same literature rule; trigger = spot VIX <15 / >25 points"),
]

# Smooth (linear-ramp) literature rule — no thresholds, no estimated parameters:
# momentum multiplier 1.0 at VIX<=15 ramping to 0.5 at VIX 25 and 0.0 at VIX>=35
# (freed weight -> quality+value pro-rata); low side ramps to the full x1.25 momentum /
# x0.75 value boost between VIX 15 and 10.
SMOOTH_HI_START, SMOOTH_HI_END = 15.0, 35.0     # mult 1.0 -> 0.0 (0.5 at 25)
SMOOTH_LO_START, SMOOTH_LO_END = 15.0, 10.0     # boost 0 -> full


def smooth_multipliers(v: float) -> tuple[float, float]:
    """(momentum_mult, value_mult) as continuous piecewise-linear functions of VIX."""
    if not np.isfinite(v):
        return 1.0, 1.0
    if v >= SMOOTH_HI_START:
        frac = min(1.0, (v - SMOOTH_HI_START) / (SMOOTH_HI_END - SMOOTH_HI_START))
        return 1.0 - frac, 1.0
    frac = min(1.0, (SMOOTH_LO_START - v) / (SMOOTH_LO_START - SMOOTH_LO_END))
    return (1.0 + frac * (LIT_MOM_BOOST - 1.0),
            1.0 - frac * (1.0 - LIT_VAL_CUT))


def band_multipliers(v: float, lo_full: float = 13.0, lo_edge: float = 15.0,
                     hi_edge: float = 23.0, hi_full: float = 27.0) -> tuple[float, float]:
    """Smoothed *step* rule: identical to the lit_fix decisions outside narrow
    transition bands. High side: mult 1.0 below ``hi_edge``, linear to the 0.5 floor
    at ``hi_full`` (never lower). Low side: full ×1.25/×0.75 boost below ``lo_full``,
    fading to none at ``lo_edge``. Defaults are the production knots."""
    if not np.isfinite(v):
        return 1.0, 1.0
    if v >= hi_edge:
        frac = min(1.0, (v - hi_edge) / (hi_full - hi_edge))
        return 1.0 - 0.5 * frac, 1.0
    if v <= lo_edge:
        frac = min(1.0, (lo_edge - v) / (lo_edge - lo_full))
        return (1.0 + frac * (LIT_MOM_BOOST - 1.0),
                1.0 - frac * (1.0 - LIT_VAL_CUT))
    return 1.0, 1.0


# Sensitivity grid around the production knots — for ROBUSTNESS assessment only
# (pre-committed: production keeps 13-15/23-27 regardless of which cell is argmax;
# the grid can only demote the rule if the chosen cell turns out to be an outlier).
SENS_VARIANTS: list[VariantSpec] = [DEFAULT_VARIANTS[0]] + [
    VariantSpec(f"band_{lf:g}-{le:g}_{he:g}-{hf:g}", "lit_band",
                lo_full=lf, lo_edge=le, hi_edge=he, hi_full=hf,
                desc=f"lit_band knots: low {lf:g}→{le:g}, high {he:g}→{hf:g}")
    for (lf, le) in ((12.0, 14.0), (13.0, 15.0), (14.0, 16.0))
    for (he, hf) in ((21.0, 25.0), (23.0, 27.0), (25.0, 29.0))
]

# Width grid: same anchors, wider/narrower transition bands (incl. bands that reach
# into the 15-23 neutral zone). Robustness-only, same pre-commitment as SENS_VARIANTS.
SENSW_VARIANTS: list[VariantSpec] = [DEFAULT_VARIANTS[0]] + [
    VariantSpec(f"band_{lf:g}-{le:g}_{he:g}-{hf:g}", "lit_band",
                lo_full=lf, lo_edge=le, hi_edge=he, hi_full=hf,
                desc=f"lit_band knots: low {lf:g}→{le:g}, high {he:g}→{hf:g}")
    for (lf, le) in ((13.0, 15.0), (12.0, 16.0), (11.0, 15.0))
    for (he, hf) in ((23.0, 27.0), (20.0, 27.0), (21.0, 29.0))
]

SMOOTH_VARIANTS: list[VariantSpec] = [
    DEFAULT_VARIANTS[0],
    VariantSpec("lit_fix", "literature", trigger="level",
                desc="step literature rule; trigger = spot VIX <15 / >25 points"),
    VariantSpec("lit_smooth", "lit_smooth",
                desc="linear-ramp literature rule: momentum mult 1.0@VIX15 → 0.5@25 → "
                     "0.0@35 (freed → quality/value); low side ramps to ×1.25 mom / "
                     "×0.75 value between VIX 15 and 10"),
    VariantSpec("lit_band", "lit_band",
                desc="smoothed STEP rule: flat ×1.0 below VIX 23, ramp to the ×0.5 "
                     "floor at 27 (never below); full low-VIX boost below 13 fading "
                     "out at 15 — step decisions everywhere else"),
]


@dataclass
class RebalanceDecision:
    """Everything the overlay decided at one (window, variant, test rebalance)."""
    date: str
    vix_date: str
    vix: float
    vix_pct: float           # percentile vs training VIX distribution
    bucket: str              # fixed Low/Medium/High bucket of spot
    strength: float
    n_comparable: int
    shrink_lambda: float     # 1.0 = no shrink, 0.0 = fully full-sample utility
    fallback: str            # "" | "no_tilt" | "no_obs" | "no_positive_utility"
    weights: dict[str, float]
    l1_shift: float          # 0.5 * sum |adjusted - baseline|


class WindowOverlay:
    """Per-window overlay engine: precomputes training-side statistics once, then
    answers weight queries per (variant, test rebalance)."""

    def __init__(self, wb: WindowBaseline, vix: pd.Series):
        self.wb = wb
        self.parents = wb.parents
        w = wb.window
        self.train_vix = train_vix_values(vix, w.train_start, w.train_end)
        self.vix = vix
        # Spot VIX + percentile/bucket per training stat rebalance (training-only).
        self.stat_vix: dict[str, float] = {}
        self.stat_pct: dict[str, float] = {}
        self.stat_bucket: dict[str, str] = {}
        for d in wb.stat_rebals:
            _, v = vix_spot(vix, d)
            self.stat_vix[d] = v
            self.stat_pct[d] = percentile_of(self.train_vix, v)
            self.stat_bucket[d] = fixed_bucket(v)
        # Full-training utility (the shrink target) — computed once.
        self.full_stats = parent_stats(wb.stat_rebals, wb.parent_train,
                                       wb.fwd1m_train, self.parents)
        self.full_utility = utility_from_stats(self.full_stats)
        self._regime_cache: dict[str, tuple[pd.Series, int]] = {}

    # ---- comparable observations -------------------------------------------------- #
    def _comparable_dates(self, key: str) -> list[str]:
        if key == "tail_hi":
            return [d for d in self.wb.stat_rebals if self.stat_pct.get(d, np.nan) >= TAIL_HI]
        if key == "tail_lo":
            return [d for d in self.wb.stat_rebals if self.stat_pct.get(d, np.nan) <= TAIL_LO]
        # fixed buckets
        return [d for d in self.wb.stat_rebals if self.stat_bucket.get(d) == key]

    def _regime_utility(self, key: str) -> tuple[pd.Series, int]:
        """Shrunk regime utility for a comparable-observation key + sample size."""
        if key in self._regime_cache:
            return self._regime_cache[key]
        dates = self._comparable_dates(key)
        n = len(dates)
        if n == 0:
            out = (pd.Series(dtype=float), 0)
        else:
            stats = parent_stats(dates, self.wb.parent_train, self.wb.fwd1m_train,
                                 self.parents)
            u = utility_from_stats(stats)
            if n < MIN_REGIME_OBS:
                lam = n / (n + MIN_REGIME_OBS)
                u = lam * u + (1 - lam) * self.full_utility
            out = (u, n)
        self._regime_cache[key] = out
        return out

    # ---- decision per (variant, rebalance) ---------------------------------------- #
    def decide(self, variant: VariantSpec, date: str) -> RebalanceDecision:
        base = dict(self.wb.parent_weights)
        vdate, v = vix_spot(self.vix, date)
        assert vdate <= date, "spot VIX from the future"
        pct = percentile_of(self.train_vix, v)
        bucket = fixed_bucket(v)

        def _dec(strength, n, lam, fallback, weights):
            l1 = 0.5 * sum(abs(weights.get(p, 0.0) - base.get(p, 0.0))
                           for p in set(weights) | set(base))
            return RebalanceDecision(date, vdate, v, pct, bucket, strength, n, lam,
                                     fallback, weights, l1)

        if variant.kind == "baseline":
            return _dec(0.0, 0, 1.0, "", base)

        if variant.kind == "literature":
            if variant.trigger == "pct":
                regime = "high" if pct > LIT_PCT_HI else \
                    ("low" if pct < LIT_PCT_LO else "mid")
            else:
                regime = "high" if v > LIT_LVL_HI else \
                    ("low" if v < LIT_LVL_LO else "mid")
            if regime == "mid" or not base:
                return _dec(0.0, 0, 1.0, "no_tilt", base)
            w = dict(base)
            if regime == "high":
                freed = w.get("momentum", 0.0) * (1.0 - LIT_MOM_CUT)
                if "momentum" in w:
                    w["momentum"] *= LIT_MOM_CUT
                qv_tot = w.get("quality", 0.0) + w.get("value", 0.0)
                for p in ("quality", "value"):
                    share = (w.get(p, 0.0) / qv_tot) if qv_tot > 0 else 0.5
                    w[p] = w.get(p, 0.0) + freed * share
            else:
                if "momentum" in w:
                    w["momentum"] *= LIT_MOM_BOOST
                if "value" in w:
                    w["value"] *= LIT_VAL_CUT
            return _dec(1.0, 0, 1.0, "", cap_renorm(w))

        if variant.kind in ("lit_smooth", "lit_band"):
            mom_m, val_m = (smooth_multipliers(v) if variant.kind == "lit_smooth"
                            else band_multipliers(v, variant.lo_full, variant.lo_edge,
                                                  variant.hi_edge, variant.hi_full))
            if (abs(mom_m - 1.0) < 1e-12 and abs(val_m - 1.0) < 1e-12) or not base:
                return _dec(0.0, 0, 1.0, "no_tilt", base)
            w = dict(base)
            if mom_m < 1.0:                     # high side: cut momentum, free -> Q+V
                freed = w.get("momentum", 0.0) * (1.0 - mom_m)
                if "momentum" in w:
                    w["momentum"] *= mom_m
                qv_tot = w.get("quality", 0.0) + w.get("value", 0.0)
                for p in ("quality", "value"):
                    share = (w.get(p, 0.0) / qv_tot) if qv_tot > 0 else 0.5
                    w[p] = w.get(p, 0.0) + freed * share
            else:                               # low side: boost momentum, trim value
                if "momentum" in w:
                    w["momentum"] *= mom_m
                if "value" in w:
                    w["value"] *= val_m
            return _dec(abs(mom_m - 1.0), 0, 1.0, "", cap_renorm(w))

        if variant.kind == "fixed":
            strength, key = variant.fixed_strength, bucket
        else:  # pctile
            strength = pctile_strength(pct, variant.scale)
            if strength <= 0.0:
                return _dec(0.0, 0, 1.0, "no_tilt", base)
            key = "tail_hi" if pct > NO_TILT_HI else "tail_lo"

        utility, n = self._regime_utility(key)
        if n == 0:
            return _dec(0.0, 0, 1.0, "no_obs", base)
        lam = 1.0 if n >= MIN_REGIME_OBS else n / (n + MIN_REGIME_OBS)
        rw = regime_weights_from_utility(utility)
        if not rw:
            return _dec(0.0, n, lam, "no_positive_utility", base)
        # sorted → deterministic fp summation order downstream (tie-stable books)
        keys = sorted(set(base) | set(rw))
        adj = {p: (1 - strength) * base.get(p, 0.0) + strength * rw.get(p, 0.0)
               for p in keys}
        adj = {p: x for p, x in adj.items() if x > 1e-12}
        tot = sum(adj.values())
        adj = {p: x / tot for p, x in adj.items()}
        return _dec(strength, n, lam, "", adj)
