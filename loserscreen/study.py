"""Book construction, simulation and aggregation for the loser-screen study.

One composite (the frozen per-window baseline), several book rules. Simulation
mirrors ``vixtilt.backtest.simulate`` but takes precomputed name lists so the
book rule is explicit; tie-breaks are deterministic (sorted index + mergesort),
which matters when comparing ~0.02 Sharpe effects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from vixtilt import backtest as bt
from vixtilt.baseline import CACHE_DIR, assert_window_pit, load_or_build
from vixtilt.windows import Window, block_of, semiannual_windows

COST_PER_SIDE = 0.0010
SPY = "SPY"


# --------------------------------------------------------------------------- #
# Book rules — all take the composite Series, return a name list
# --------------------------------------------------------------------------- #
def _ranked(score: pd.Series, ascending: bool) -> pd.Index:
    s = score.dropna()
    return s.sort_index().sort_values(ascending=ascending, kind="mergesort").index


def broad_book(score: pd.Series) -> list[str]:
    return list(_ranked(score, ascending=False))


def screened_book(score: pd.Series, drop_pct: float) -> list[str]:
    ranked = _ranked(score, ascending=False)
    k = int(round(len(ranked) * drop_pct))
    return list(ranked[: len(ranked) - k]) if k else list(ranked)


def bottom_book(score: pd.Series, pct: float) -> list[str]:
    ranked = _ranked(score, ascending=True)
    k = max(1, int(round(len(ranked) * pct)))
    return list(ranked[:k])


@dataclass(frozen=True)
class BookSpec:
    name: str
    desc: str
    weighting: str = "ew"          # "ew" | "cap" | "mix" | "tilt" | "tilt2"
    # mix = 0.5·ew + 0.5·cap; tilt/tilt2 = mix scaled by composite-rank quintile
    # (mild ×0.5..×1.5 / strong ×1..×5), renormalised.
    blend: tuple[float, float, float] | None = None
    # generic (ew, cap, tilt) shares; overrides `weighting` when set. The tilt
    # leg is the mild quintile ladder applied to equal weights.
    base_rule: str | None = None   # explicit book rule for blend books
    veto_parents: tuple[str, ...] | None = None  # v4: drop names in the bottom
    veto_pct: float = 0.0                        # veto_pct of ANY veto parent
    veto_complement: bool = False                # keep the vetoed names instead

    @property
    def rule(self) -> str:
        if self.base_rule:
            return self.base_rule
        for pfx in ("cap_", "mix_", "tilt2_", "tilt_"):
            if self.name.startswith(pfx):
                return self.name[len(pfx):]
        return self.name

    def names(self, score: pd.Series) -> list[str]:
        r = self.rule
        if r == "broad":
            return broad_book(score)
        if r.startswith("screen"):
            return screened_book(score, int(r[6:]) / 100.0)
        if r.startswith("bottom"):
            return bottom_book(score, int(r[6:]) / 100.0)
        if r.startswith("top"):
            return bt.top_names(score, int(r[3:]) / 100.0)
        raise ValueError(self.name)


BOOKS = [
    BookSpec("broad", "all scored names, equal weight (reference)"),
    BookSpec("screen10", "broad minus bottom 10% by composite"),
    BookSpec("screen20", "broad minus bottom 20% (PRE-REGISTERED PRIMARY)"),
    BookSpec("screen30", "broad minus bottom 30%"),
    BookSpec("bottom20", "the excluded bottom 20% as a book (diagnostic)"),
    BookSpec("top20", "incumbent top-20% construction (context)"),
]

_DEPTHS = (10, 20, 30, 40, 50, 60, 70)
BOOKS_V2 = (
    [BookSpec("broad", "all scored names, equal weight (reference)")]
    + [BookSpec(f"screen{d}", f"EW broad minus bottom {d}% (depth curve)")
       for d in _DEPTHS]
    + [BookSpec("top20", "incumbent EW top-20% (curve endpoint context)")]
    + [BookSpec("cap_broad", "cap-weighted, mcap-covered subset (reference)",
                weighting="cap")]
    + [BookSpec(f"cap_screen{d}",
                "cap-weighted minus bottom 20% (PRE-REGISTERED v2 PRIMARY)"
                if d == 20 else f"cap-weighted minus bottom {d}% (depth curve)",
                weighting="cap")
       for d in _DEPTHS]
    + [BookSpec("cap_top20", "cap-weighted top-20% (context)", weighting="cap")]
)

_TILT_MULTS = {"tilt": (0.5, 0.75, 1.0, 1.25, 1.5),    # worst→best quintile
               "tilt2": (1.0, 2.0, 3.0, 4.0, 5.0)}

BOOKS_TILT = [
    BookSpec("mix_broad", "50/50 blend, unscreened (reference)", weighting="mix"),
    BookSpec("mix_screen20", "blend minus bottom 20% (v3-validated)",
             weighting="mix"),
    BookSpec("mix_screen30", "blend minus bottom 30%", weighting="mix"),
    BookSpec("mix_screen40", "blend minus bottom 40% (depth curve)",
             weighting="mix"),
    BookSpec("mix_screen50", "blend minus bottom 50% (depth curve)",
             weighting="mix"),
    BookSpec("tilt_broad", "unscreened, mild quintile tilt ×0.5..×1.5",
             weighting="tilt"),
    BookSpec("tilt2_broad", "unscreened, strong quintile tilt ×1..×5",
             weighting="tilt2"),
    BookSpec("tilt_screen20", "screen20 + mild quintile tilt", weighting="tilt"),
    BookSpec("tilt2_screen20", "screen20 + strong quintile tilt",
             weighting="tilt2"),
]

def _blend_spec(name: str, ew: float, cap: float, tilt: float) -> BookSpec:
    return BookSpec(name, f"screen30, {ew:.2f}·EW + {cap:.2f}·cap + {tilt:.2f}·tilt",
                    blend=(ew, cap, tilt), base_rule="screen30")


BOOKS_WMIX = [
    _blend_spec("s30_ew", 1, 0, 0),                    # reference
    _blend_spec("s30_cap", 0, 1, 0),
    _blend_spec("s30_tilt", 0, 0, 1),
    _blend_spec("s30_ew50cap50", 0.5, 0.5, 0),
    _blend_spec("s30_ew75cap25", 0.75, 0.25, 0),
    _blend_spec("s30_ew25cap75", 0.25, 0.75, 0),
    _blend_spec("s30_ew50tilt50", 0.5, 0, 0.5),
    _blend_spec("s30_cap50tilt50", 0, 0.5, 0.5),
    _blend_spec("s30_thirds", 1 / 3, 1 / 3, 1 / 3),
    _blend_spec("s30_ew50cap25tilt25", 0.5, 0.25, 0.25),
]

def veto_set(frame: pd.DataFrame, parents: tuple[str, ...],
             pct: float) -> set[str]:
    """Names in the bottom `pct` percentile of ANY veto parent (per-date rank
    over all scored names). NaN never vetoes; degenerate parents are skipped."""
    out: set[str] = set()
    for p in parents:
        col = frame[p].dropna() if p in frame.columns else pd.Series(dtype=float)
        if col.nunique() < 10:
            continue
        r = col.rank(pct=True)
        out |= set(r.index[r <= pct])
    return out


def _veto_filter(b: BookSpec, names: list[str],
                 frame: pd.DataFrame | None) -> list[str]:
    if not b.veto_parents or frame is None:
        return names
    vetoed = veto_set(frame, b.veto_parents, b.veto_pct)
    if b.veto_complement:
        return [n for n in names if n in vetoed]
    return [n for n in names if n not in vetoed]


_PARENTS = ("momentum", "value", "quality", "growth", "revisions",
            "institutional", "insider", "short")


def _veto_spec(name: str, desc: str, parents: tuple[str, ...], pct: float,
               complement: bool = False) -> BookSpec:
    return BookSpec(name, desc, weighting="mix", base_rule="screen25",
                    veto_parents=parents, veto_pct=pct,
                    veto_complement=complement)


BOOKS_VETO = (
    [BookSpec("mix_broad", "50/50 blend, unscreened (context)", weighting="mix"),
     BookSpec("mix_screen25", "ratified construction (reference)",
              weighting="mix"),
     _veto_spec("vqs20", "screen25 minus bottom-20% quality OR short "
                         "(PRE-REGISTERED v4 PRIMARY)", ("quality", "short"), 0.20),
     _veto_spec("vqs10", "quality/short veto at 10% (curve point)",
                ("quality", "short"), 0.10),
     _veto_spec("vall10", "any-parent veto at 10%", _PARENTS, 0.10),
     _veto_spec("vall20", "any-parent veto at 20%", _PARENTS, 0.20),
     _veto_spec("vqs20_cut", "the names vqs20 removes (mechanism check)",
                ("quality", "short"), 0.20, complement=True)]
    + [_veto_spec(f"v1_{p}20", f"screen25 minus bottom-20% {p} only (diagnostic)",
                  (p,), 0.20) for p in _PARENTS]
)

_VETO2_SUBSETS = {
    "vpos4_20": ("quality", "short", "insider", "growth"),
    "vpos6_20": tuple(p for p in _PARENTS
                      if p not in ("value", "institutional")),
}

BOOKS_VETO2 = (
    # EXPLORATORY decomposition of the all-parent veto — subsets are chosen
    # from the v4 single-parent diagnostics (in-sample), so no bar applies.
    [BookSpec("mix_screen25", "ratified construction (reference)",
              weighting="mix"),
     _veto_spec("vqs20", "v4-validated quality/short veto (context)",
                ("quality", "short"), 0.20),
     _veto_spec("vall20", "any-parent veto at 20%", _PARENTS, 0.20)]
    + [_veto_spec(f"vno_{p}20", f"any-parent veto at 20% EXCEPT {p} "
                                "(leave-one-out)",
                  tuple(x for x in _PARENTS if x != p), 0.20)
       for p in _PARENTS]
    + [_veto_spec("vpos4_20", "veto on the 4 individually-positive parents "
                              "(quality/short/insider/growth — in-sample pick)",
                  _VETO2_SUBSETS["vpos4_20"], 0.20),
       _veto_spec("vpos6_20", "any-parent veto minus value & institutional "
                              "(the two individually-harmful vetoes)",
                  _VETO2_SUBSETS["vpos6_20"], 0.20)]
)

_CANDS = ("net_issuance", "asset_growth", "accruals", "idio_vol", "max5")

BOOKS_CANDS = (
    # v5 new-candidate battery (pre-registered): each candidate as one extra
    # veto on the ratified base; solos + all-five are exploratory context.
    [BookSpec("mix_screen25", "screen-only context", weighting="mix"),
     _veto_spec("vpos6_20", "ratified base (REFERENCE)",
                _VETO2_SUBSETS["vpos6_20"], 0.20)]
    + [_veto_spec(f"vp6_{c}20", f"base + {c} veto at 20% (PRIMARY cell)",
                  _VETO2_SUBSETS["vpos6_20"] + (c,), 0.20) for c in _CANDS]
    + [_veto_spec(f"solo_{c}20", f"{c} as the only veto on screen25 "
                                 "(EXPLORATORY standalone)",
                  (c,), 0.20) for c in _CANDS]
    + [_veto_spec("vp6_allnew20", "base + all five candidates (EXPLORATORY)",
                  _VETO2_SUBSETS["vpos6_20"] + _CANDS, 0.20)]
)

BOOKS_MIX = [
    BookSpec("broad", "EW reference"),
    BookSpec("screen20", "EW screen20 reference"),
    BookSpec("cap_broad", "cap reference", weighting="cap"),
    BookSpec("cap_screen20", "cap screen20 reference", weighting="cap"),
    BookSpec("mix_broad", "50/50 EW-cap blend, covered subset (reference)",
             weighting="mix"),
    BookSpec("mix_screen10", "blend minus bottom 10% (curve point)",
             weighting="mix"),
    BookSpec("mix_screen20", "blend minus bottom 20% (PRE-REGISTERED v3 PRIMARY)",
             weighting="mix"),
    BookSpec("mix_screen30", "blend minus bottom 30% (curve point)",
             weighting="mix"),
]


# --------------------------------------------------------------------------- #
# Simulation over precomputed books (rules identical across variants)
# --------------------------------------------------------------------------- #
def simulate_books(books: dict[str, pd.Series], matrix: pd.DataFrame,
                   cost_per_side: float) -> pd.DataFrame:
    """``books``: formation date -> weight Series (already normalised to 1)."""
    dates = [d for d in sorted(books) if d in matrix.index]
    rows: list[dict] = []
    prev = pd.Series(dtype=float)
    for i, d in enumerate(dates[:-1]):
        nxt = dates[i + 1]
        w = books[d]
        if w.empty:
            continue
        names = list(w.index)
        px0, px1 = matrix.loc[d], matrix.loc[nxt]
        ret = (px1.reindex(names) / px0.reindex(names)) - 1.0
        ok = ret.notna()
        if not ok.any():
            continue
        ww = w[ok] / w[ok].sum()
        gross = float((ww * ret[ok]).sum())
        idx = prev.index.union(w.index)
        traded = float((w.reindex(idx).fillna(0.0) - prev.reindex(idx).fillna(0.0))
                       .abs().sum())
        cost = cost_per_side * traded
        spy = float(px1[SPY] / px0[SPY] - 1.0) if SPY in px0.index and px0[SPY] > 0 \
            else float("nan")
        rows.append({"date": d, "next": nxt, "gross": gross, "net": gross - cost,
                     "turnover": 0.5 * traded, "cost": cost, "spy": spy,
                     "n_names": int(ok.sum())})
        prev = w
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(
        columns=["next", "gross", "net", "turnover", "cost", "spy", "n_names"])


# --------------------------------------------------------------------------- #
# Study
# --------------------------------------------------------------------------- #
@dataclass
class ScreenResult:
    books: list[BookSpec]
    windows: list[Window]
    portfolios: dict[str, pd.DataFrame] = field(default_factory=dict)
    window_of_date: dict[str, str] = field(default_factory=dict)
    pit_checks: pd.DataFrame = field(default_factory=pd.DataFrame)
    book_sizes: pd.DataFrame = field(default_factory=pd.DataFrame)
    weights: dict[str, dict[str, pd.Series]] = field(default_factory=dict)
    # book name -> {formation date -> weight Series} (needed by the v4 null)


def _book_weights(b: BookSpec, score: pd.Series, mcap: pd.Series | None,
                  frame: pd.DataFrame | None = None) -> pd.Series:
    """Weight Series for one book on one date (empty when not buildable)."""
    if b.blend is not None:
        # all blend books live on the mcap-covered subset (fair comparison,
        # even when the cap share is zero)
        if mcap is None:
            return pd.Series(dtype=float)
        score = score[score.index.isin(mcap.index)]
        names = _veto_filter(b, b.names(score), frame)
        if not names:
            return pd.Series(dtype=float)
        ew = pd.Series(1.0 / len(names), index=sorted(names))
        mc = mcap.reindex(ew.index)
        cap = mc / mc.sum()
        s = score.reindex(ew.index)
        q = pd.qcut(s.rank(method="first"), 5, labels=False)
        tilt = ew * q.map(dict(enumerate(_TILT_MULTS["tilt"])))
        tilt = tilt / tilt.sum()
        a_ew, a_cap, a_tilt = b.blend
        w = a_ew * ew + a_cap * cap + a_tilt * tilt
        return w / w.sum()
    if b.weighting in ("cap", "mix", "tilt", "tilt2"):
        if mcap is None:
            return pd.Series(dtype=float)
        score = score[score.index.isin(mcap.index)]
    names = _veto_filter(b, b.names(score), frame)
    if not names:
        return pd.Series(dtype=float)
    ew = pd.Series(1.0 / len(names), index=sorted(names))
    if b.weighting == "ew":
        return ew
    mc = mcap.reindex(names)
    cap = (mc / mc.sum()).sort_index()
    if b.weighting == "cap":
        return cap
    mix = 0.5 * ew + 0.5 * cap
    if b.weighting == "mix":
        return mix
    # quintile tilt within the held book (deterministic tie-break via sorted
    # index + rank method="first")
    s = score.reindex(sorted(names))
    q = pd.qcut(s.rank(method="first"), 5, labels=False)
    mult = q.map(dict(enumerate(_TILT_MULTS[b.weighting])))
    w = mix * mult
    return w / w.sum()


def run_study(panel, matrix: pd.DataFrame, sectors: pd.Series, *,
              books: list[BookSpec] = BOOKS,
              mcaps: dict[str, pd.Series] | None = None,
              extra_cols: dict[str, pd.DataFrame] | None = None,
              first_test_year: int = 2017, last_end: str = "2026-06-30",
              cache_dir: Path = CACHE_DIR, rebuild_cache: bool = False,
              cost_per_side: float = COST_PER_SIDE,
              verbose: bool = True) -> ScreenResult:
    if any(b.weighting == "cap" for b in books) and not mcaps:
        raise ValueError("cap-weighted books need mcaps")
    wins = semiannual_windows(first_test_year, last_end)
    res = ScreenResult(books=list(books), windows=wins)
    weight_lists: dict[str, dict[str, pd.Series]] = {b.name: {} for b in books}
    pit_rows: list[dict] = []
    size_rows: list[dict] = []

    if verbose:
        print(f"Building baseline composites over {len(wins)} windows…")
    for win in wins:
        wb = load_or_build(panel, matrix, win, cache_dir=cache_dir,
                           rebuild=rebuild_cache, verbose=verbose)
        for chk in assert_window_pit(wb):
            pit_rows.append({"window": win.label, "check": chk, "status": "PASS"})
        for d in wb.test_rebals:
            frame = wb.parent_test.get(d)
            if frame is None:
                continue
            res.window_of_date[d] = win.label
            score = bt.composite_from_parents(frame, wb.parent_weights, sectors)
            if extra_cols and d in extra_cols:
                # candidate veto columns (v5) — never enter the composite,
                # only readable by veto_set
                frame = frame.join(extra_cols[d])
            mcap = (mcaps or {}).get(d)
            for b in books:
                w = _book_weights(b, score, mcap, frame)
                weight_lists[b.name][d] = w
                row = {"window": win.label, "date": d, "book": b.name,
                       "n_names": len(w)}
                if b.weighting == "cap" and mcap is not None and len(w):
                    covered = mcap[mcap.index.isin(score.dropna().index)]
                    row["cap_frac_held"] = float(
                        mcap.reindex(w.index).sum() / covered.sum())
                    row["n_covered"] = int(score.dropna().index.isin(
                        mcap.index).sum())
                size_rows.append(row)

    res.pit_checks = pd.DataFrame(pit_rows)
    res.book_sizes = pd.DataFrame(size_rows)
    res.weights = weight_lists
    if verbose:
        print("Simulating books…")
    for b in books:
        res.portfolios[b.name] = simulate_books(weight_lists[b.name], matrix,
                                                cost_per_side)
    return res


def _mix_weights(names: list[str], mcap: pd.Series) -> pd.Series:
    if not names:
        return pd.Series(dtype=float)
    ew = pd.Series(1.0 / len(names), index=sorted(names))
    mc = mcap.reindex(ew.index)
    cap = mc / mc.sum()
    return 0.5 * ew + 0.5 * cap


def random_null(res: ScreenResult, primary: str, ref: str,
                matrix: pd.DataFrame, mcaps: dict[str, pd.Series], *,
                n_draws: int = 200, seed: int = 20260714,
                cost_per_side: float = COST_PER_SIDE,
                verbose: bool = True) -> pd.Series:
    """Random-characteristic null (v4 pre-registration, check c).

    Each draw fixes one uniform u per ticker, then per date drops the same
    NUMBER of names from the `ref` book as `primary` dropped, lowest u first
    (mix-weighted). Same size and persistence as the veto, zero information.
    Returns the full-period net Sharpe of each draw."""
    import numpy as np
    ref_w, prim_w = res.weights[ref], res.weights[primary]
    tickers = sorted({t for w in ref_w.values() for t in w.index})
    rng = np.random.default_rng(seed)
    sharpes = []
    for i in range(n_draws):
        u = pd.Series(rng.random(len(tickers)), index=tickers)
        books = {}
        for d, w in ref_w.items():
            k = len(prim_w.get(d, ()))
            if not k:
                continue
            keep = u.reindex(sorted(w.index)) \
                .sort_values(ascending=False, kind="mergesort").index[:k]
            books[d] = _mix_weights(list(keep), mcaps[d])
        r = simulate_books(books, matrix, cost_per_side)["net"].dropna()
        sharpes.append(float(r.mean() / r.std(ddof=1) * np.sqrt(12)))
        if verbose and (i + 1) % 25 == 0:
            print(f"  null draw {i + 1}/{n_draws}")
    return pd.Series(sharpes, name="null_sharpe")


# --------------------------------------------------------------------------- #
# Aggregation (window / block / full), mirroring vixtilt's grouping
# --------------------------------------------------------------------------- #
def grouping_metrics(res: ScreenResult, group_windows: dict[str, set[str]],
                     leg: str = "net") -> pd.DataFrame:
    rows = []
    for gname, wset in group_windows.items():
        for b in res.books:
            pf = res.portfolios.get(b.name)
            if pf is None or pf.empty:
                continue
            keep = [d for d in pf.index if res.window_of_date.get(d) in wset]
            sub = pf.loc[keep]
            if sub.empty:
                continue
            m = bt.perf_metrics(sub[leg], sub["spy"], sub["turnover"])
            m.update({"grouping": gname, "book": b.name,
                      "avg_n_names": float(sub["n_names"].mean())})
            rows.append(m)
    return pd.DataFrame(rows)


def group_defs(res: ScreenResult) -> dict[str, dict[str, set[str]]]:
    all_w = {w.label for w in res.windows}
    per_window = {w.label: {w.label} for w in res.windows}
    blocks: dict[str, set[str]] = {}
    for w in res.windows:
        blocks.setdefault(block_of(w.label), set()).add(w.label)
    return {"window": per_window, "block": blocks, "full": {"full": all_w}}
