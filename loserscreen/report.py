"""REPORT.md + CSV outputs for the loser-screen study, incl. the pre-registered verdict."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import study as st

FULL_COLS = ["book", "avg_n_names", "cagr", "sharpe", "sortino", "ann_vol",
             "max_drawdown", "hit_rate", "avg_turnover", "spy_cagr",
             "spy_excess_cagr", "spy_alpha", "spy_beta", "spy_ir"]


def _fmt(df: pd.DataFrame) -> str:
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    header = "| " + " | ".join(d.columns) + " |"
    sep = "|" + "|".join(" --- " for _ in d.columns) + "|"
    body = "\n".join("| " + " | ".join(str(v) for v in r) + " |"
                     for r in d.itertuples(index=False))
    return "\n".join([header, sep, body])


def window_wins(res: st.ScreenResult, a: str = "screen20", b: str = "broad",
                leg: str = "net") -> pd.DataFrame:
    """Per-window pooled net total return of `a` vs `b` (the pre-registered win metric)."""
    rows = []
    pa, pb = res.portfolios[a], res.portfolios[b]
    for w in res.windows:
        da = [d for d in pa.index if res.window_of_date.get(d) == w.label]
        db = [d for d in pb.index if res.window_of_date.get(d) == w.label]
        if not da or not db:
            continue
        ra = float((1.0 + pa.loc[da, leg]).prod() - 1.0)
        rb = float((1.0 + pb.loc[db, leg]).prod() - 1.0)
        rows.append({"window": w.label, a: ra, b: rb, "diff": ra - rb,
                     "win": ra > rb})
    return pd.DataFrame(rows)


def verdict(full: pd.DataFrame, wins: pd.DataFrame, a_name: str = "screen20",
            b_name: str = "broad") -> tuple[bool, str]:
    f = full.set_index("book")
    sh_s, sh_b = f.loc[a_name, "sharpe"], f.loc[b_name, "sharpe"]
    n_win, n_tot = int(wins["win"].sum()), len(wins)
    a = sh_s > sh_b
    b = n_win >= 10
    lines = [
        f"(a) full-period net Sharpe {a_name} {sh_s:.3f} vs {b_name} {sh_b:.3f} → "
        f"{'PASS' if a else 'FAIL'}",
        f"(b) window wins {n_win}/{n_tot} (bar: >=10/19) → {'PASS' if b else 'FAIL'}",
    ]
    return a and b, "\n".join(f"- {ln}" for ln in lines)


def active_stats(res: st.ScreenResult, a: str, b: str) -> dict:
    """Annualised active return of `a` over `b` on the common monthly grid + t-stat."""
    import numpy as np
    pa, pb = res.portfolios[a]["net"], res.portfolios[b]["net"]
    act = (pa - pb).dropna()
    if len(act) < 2:
        return {}
    te = float(act.std(ddof=1))
    ann, ann_te = float(act.mean()) * 12, te * (12 ** 0.5)
    ir = ann / ann_te if ann_te > 0 else float("nan")
    t = float(act.mean() / (te / np.sqrt(len(act)))) if te > 0 else float("nan")
    return {"pair": f"{a} - {b}", "n_months": len(act), "active_ann": ann,
            "te_ann": ann_te, "ir": ir, "t_stat": t,
            "hit_rate": float((act > 0).mean())}


def write_all(res: st.ScreenResult, out_dir: Path, cost_per_side: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = st.group_defs(res)
    full_net = st.grouping_metrics(res, groups["full"], leg="net")
    full_gross = st.grouping_metrics(res, groups["full"], leg="gross")
    block_net = st.grouping_metrics(res, groups["block"], leg="net")
    win_net = st.grouping_metrics(res, groups["window"], leg="net")
    wins = window_wins(res)

    full_net.to_csv(out_dir / "full_net.csv", index=False)
    full_gross.to_csv(out_dir / "full_gross.csv", index=False)
    block_net.to_csv(out_dir / "block_net.csv", index=False)
    win_net.to_csv(out_dir / "window_net.csv", index=False)
    wins.to_csv(out_dir / "window_wins.csv", index=False)
    res.pit_checks.to_csv(out_dir / "pit_checks.csv", index=False)
    res.book_sizes.to_csv(out_dir / "book_sizes.csv", index=False)

    passed, vlines = verdict(full_net, wins)
    import loserscreen
    md = [
        "# Loser-screen walk-forward study",
        "",
        "## Pre-registration (verbatim, fixed before the first run)",
        "", "```", loserscreen.__doc__.strip(), "```", "",
        f"## VERDICT: **{'PASS' if passed else 'FAIL'}**", "", vlines, "",
        f"Costs: {cost_per_side * 1e4:.0f} bps/side on traded notional. "
        f"{res.pit_checks.shape[0]} PIT assertions pass.",
        "",
        "## Full period (net)", "", _fmt(full_net[FULL_COLS]), "",
        "## Full period (gross)", "", _fmt(full_gross[FULL_COLS]), "",
        "## Blocks (net)", "",
        _fmt(block_net[["grouping"] + FULL_COLS].sort_values(["grouping", "book"])), "",
        "## Per-window: screen20 vs broad (net total return — the win metric)", "",
        _fmt(wins), "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(md))


def spy_tracking(res: st.ScreenResult, book: str) -> dict:
    """How index-like is this book? Correlation/beta/TE/excess vs SPY."""
    import numpy as np
    pf = res.portfolios[book]
    df = pf[["net", "spy"]].dropna()
    p, b = df["net"], df["spy"]
    beta = float(((p - p.mean()) * (b - b.mean())).sum()
                 / ((b - b.mean()) ** 2).sum())
    act = p - b
    return {"book": book, "corr": float(p.corr(b)), "beta": beta,
            "te_ann": float(act.std(ddof=1)) * np.sqrt(12),
            "excess_ann": float(act.mean()) * 12}


def write_mix(res: st.ScreenResult, out_dir: Path, cost_per_side: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = st.group_defs(res)
    full_net = st.grouping_metrics(res, groups["full"], leg="net")
    win_net = st.grouping_metrics(res, groups["window"], leg="net")
    wins_mix = window_wins(res, "mix_screen20", "mix_broad")

    full_net.to_csv(out_dir / "full_net.csv", index=False)
    win_net.to_csv(out_dir / "window_net.csv", index=False)
    wins_mix.to_csv(out_dir / "window_wins_mix.csv", index=False)
    res.pit_checks.to_csv(out_dir / "pit_checks.csv", index=False)
    res.book_sizes.to_csv(out_dir / "book_sizes.csv", index=False)

    passed, vlines = verdict(full_net, wins_mix, "mix_screen20", "mix_broad")
    act = pd.DataFrame([s for s in (active_stats(res, "screen20", "broad"),
                                    active_stats(res, "cap_screen20", "cap_broad"),
                                    active_stats(res, "mix_screen20", "mix_broad"))
                        if s])
    act.to_csv(out_dir / "active_stats.csv", index=False)
    trk = pd.DataFrame([spy_tracking(res, b)
                        for b in ("broad", "cap_broad", "mix_broad")])
    trk.to_csv(out_dir / "spy_tracking.csv", index=False)

    import loserscreen
    md = [
        "# Loser-screen v3 — 50/50 EW-cap blend",
        "",
        "## Pre-registration (verbatim, fixed before the first run)",
        "", "```", loserscreen.__doc__.strip(), "```", "",
        f"## CONFIRMATORY VERDICT (mix_screen20 vs mix_broad): "
        f"**{'PASS' if passed else 'FAIL'}**", "", vlines, "",
        f"Costs: {cost_per_side * 1e4:.0f} bps/side. "
        f"{res.pit_checks.shape[0]} PIT assertions pass.",
        "",
        "## Active return significance (all three weightings)", "", _fmt(act), "",
        "## Broad-book tracking vs SPY", "", _fmt(trk), "",
        "## Full period (net)", "", _fmt(full_net[FULL_COLS]), "",
        "## Per-window: mix_screen20 vs mix_broad (net total return)", "",
        _fmt(wins_mix), "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(md))


def write_explore(res: st.ScreenResult, out_dir: Path,
                  cost_per_side: float, ref: str = "mix_broad") -> None:
    """EXPLORATORY report — no pre-registered bar; everything vs `ref`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = st.group_defs(res)
    full_net = st.grouping_metrics(res, groups["full"], leg="net")
    win_net = st.grouping_metrics(res, groups["window"], leg="net")
    full_net.to_csv(out_dir / "full_net.csv", index=False)
    win_net.to_csv(out_dir / "window_net.csv", index=False)
    res.pit_checks.to_csv(out_dir / "pit_checks.csv", index=False)
    res.book_sizes.to_csv(out_dir / "book_sizes.csv", index=False)

    rows = []
    for b in res.books:
        if b.name == ref:
            continue
        a = active_stats(res, b.name, ref)
        if a:
            w = window_wins(res, b.name, ref)
            a["window_wins"] = f"{int(w['win'].sum())}/{len(w)}"
            rows.append(a)
    act = pd.DataFrame(rows)
    act.to_csv(out_dir / "active_stats.csv", index=False)

    md = [
        "# Loser-screen — EXPLORATORY variants (no pre-registered bar)",
        "",
        "These cells are for mapping, not selection: any configuration that looks",
        "best here would need a fresh pre-registered confirmation before being",
        "treated as real (multiple-comparison risk).",
        "",
        f"Reference book: `{ref}`. Costs: {cost_per_side * 1e4:.0f} bps/side. "
        f"{res.pit_checks.shape[0]} PIT assertions pass.",
        "",
        f"## Active vs {ref}", "", _fmt(act), "",
        "## Full period (net)", "", _fmt(full_net[FULL_COLS]), "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(md))


def write_veto(res: st.ScreenResult, out_dir: Path, cost_per_side: float,
               null_sharpes: pd.Series) -> bool:
    """v4 parent-veto report: three-check verdict (Sharpe, windows, random null)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = st.group_defs(res)
    full_net = st.grouping_metrics(res, groups["full"], leg="net")
    block_net = st.grouping_metrics(res, groups["block"], leg="net")
    win_net = st.grouping_metrics(res, groups["window"], leg="net")
    wins = window_wins(res, "vqs20", "mix_screen25")

    full_net.to_csv(out_dir / "full_net.csv", index=False)
    block_net.to_csv(out_dir / "block_net.csv", index=False)
    win_net.to_csv(out_dir / "window_net.csv", index=False)
    wins.to_csv(out_dir / "window_wins_veto.csv", index=False)
    null_sharpes.to_csv(out_dir / "null_sharpes.csv", index=False)
    res.pit_checks.to_csv(out_dir / "pit_checks.csv", index=False)
    res.book_sizes.to_csv(out_dir / "book_sizes.csv", index=False)

    rows = []
    for b in res.books:
        if b.name in ("mix_screen25", "mix_broad"):
            continue
        a = active_stats(res, b.name, "mix_screen25")
        if a:
            w = window_wins(res, b.name, "mix_screen25")
            a["window_wins"] = f"{int(w['win'].sum())}/{len(w)}"
            rows.append(a)
    act = pd.DataFrame(rows)
    act.to_csv(out_dir / "active_stats.csv", index=False)

    f = full_net.set_index("book")
    sh_v, sh_r = f.loc["vqs20", "sharpe"], f.loc["mix_screen25", "sharpe"]
    n_win, n_tot = int(wins["win"].sum()), len(wins)
    null_q90 = float(null_sharpes.quantile(0.90))
    null_pctile = float((null_sharpes < sh_v).mean())
    checks = [
        (sh_v > sh_r, f"(a) full-period net Sharpe vqs20 {sh_v:.3f} vs "
                      f"mix_screen25 {sh_r:.3f}"),
        (n_win >= 10, f"(b) window wins {n_win}/{n_tot} (bar: >=10/19)"),
        (sh_v >= null_q90, f"(c) vqs20 Sharpe {sh_v:.3f} vs null 90th pct "
                           f"{null_q90:.3f} (beats {null_pctile:.0%} of "
                           f"{len(null_sharpes)} draws)"),
    ]
    passed = all(ok for ok, _ in checks)
    vlines = "\n".join(f"- {msg} → {'PASS' if ok else 'FAIL'}"
                       for ok, msg in checks)

    import loserscreen
    md = [
        "# Loser-screen v4 — parent-veto within screen25",
        "",
        "## Pre-registration (verbatim, fixed before the first run)",
        "", "```", loserscreen.__doc__.strip(), "```", "",
        f"## CONFIRMATORY VERDICT (vqs20 vs mix_screen25): "
        f"**{'PASS' if passed else 'FAIL'}**", "", vlines, "",
        f"Costs: {cost_per_side * 1e4:.0f} bps/side. "
        f"{res.pit_checks.shape[0]} PIT assertions pass. "
        f"Null draws: n={len(null_sharpes)}, mean "
        f"{float(null_sharpes.mean()):.3f}, 90th pct {null_q90:.3f}.",
        "",
        "## Active vs mix_screen25 (vqs20 confirmatory; all others EXPLORATORY)",
        "", _fmt(act), "",
        "## Full period (net)", "", _fmt(full_net[FULL_COLS]), "",
        "## Per-window: vqs20 vs mix_screen25 (net total return)", "",
        _fmt(wins), "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(md))
    return passed


def write_veto2(res: st.ScreenResult, out_dir: Path, cost_per_side: float,
                nulls: dict[str, pd.Series]) -> None:
    """EXPLORATORY veto decomposition: leave-one-out + in-sample subsets, with
    size-matched random nulls for the concentrated books. No bar."""
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = st.group_defs(res)
    full_net = st.grouping_metrics(res, groups["full"], leg="net")
    block_net = st.grouping_metrics(res, groups["block"], leg="net")
    win_net = st.grouping_metrics(res, groups["window"], leg="net")
    full_net.to_csv(out_dir / "full_net.csv", index=False)
    block_net.to_csv(out_dir / "block_net.csv", index=False)
    win_net.to_csv(out_dir / "window_net.csv", index=False)
    res.pit_checks.to_csv(out_dir / "pit_checks.csv", index=False)
    res.book_sizes.to_csv(out_dir / "book_sizes.csv", index=False)

    rows = []
    for b in res.books:
        if b.name == "mix_screen25":
            continue
        a = active_stats(res, b.name, "mix_screen25")
        if a:
            w = window_wins(res, b.name, "mix_screen25")
            a["window_wins"] = f"{int(w['win'].sum())}/{len(w)}"
            rows.append(a)
    act = pd.DataFrame(rows)
    act.to_csv(out_dir / "active_stats.csv", index=False)

    f = full_net.set_index("book")
    nrows = []
    for name, ns in nulls.items():
        sh = float(f.loc[name, "sharpe"])
        nrows.append({"book": name, "sharpe": sh,
                      "avg_n_names": float(f.loc[name, "avg_n_names"]),
                      "null_mean": float(ns.mean()),
                      "null_q90": float(ns.quantile(0.90)),
                      "pct_draws_beaten": float((ns < sh).mean()),
                      "n_draws": len(ns)})
        ns.to_csv(out_dir / f"null_sharpes_{name}.csv", index=False)
    nulltab = pd.DataFrame(nrows)
    nulltab.to_csv(out_dir / "null_table.csv", index=False)

    md = [
        "# Loser-screen v4b — parent-veto decomposition (EXPLORATORY)",
        "",
        "No pre-registered bar. The subset books (vpos4_20, vpos6_20) were",
        "chosen FROM the v4 single-parent diagnostics — in-sample selection on",
        "the same 110 months. Any subset adopted from this table needs to be",
        "treated as data-mined; the size-matched nulls bound the luck story",
        "but cannot remove the selection effect.",
        "",
        f"Reference: `mix_screen25`. Costs: {cost_per_side * 1e4:.0f} bps/side. "
        f"{res.pit_checks.shape[0]} PIT assertions pass.",
        "",
        "## Size-matched random nulls (200 same-size persistent-random drops)",
        "", _fmt(nulltab), "",
        "## Active vs mix_screen25", "", _fmt(act), "",
        "## Full period (net)", "", _fmt(full_net[FULL_COLS]), "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(md))


def write_cands(res: st.ScreenResult, out_dir: Path, cost_per_side: float,
                nulls: dict[str, pd.Series]) -> dict[str, bool]:
    """v5 new-candidate battery: per-candidate three-check verdicts vs vpos6_20."""
    ref = "vpos6_20"
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = st.group_defs(res)
    full_net = st.grouping_metrics(res, groups["full"], leg="net")
    block_net = st.grouping_metrics(res, groups["block"], leg="net")
    win_net = st.grouping_metrics(res, groups["window"], leg="net")
    full_net.to_csv(out_dir / "full_net.csv", index=False)
    block_net.to_csv(out_dir / "block_net.csv", index=False)
    win_net.to_csv(out_dir / "window_net.csv", index=False)
    res.pit_checks.to_csv(out_dir / "pit_checks.csv", index=False)
    res.book_sizes.to_csv(out_dir / "book_sizes.csv", index=False)

    def _act_table(base: str, names: list[str]) -> pd.DataFrame:
        rows = []
        for n in names:
            a = active_stats(res, n, base)
            if a:
                w = window_wins(res, n, base)
                a["window_wins"] = f"{int(w['win'].sum())}/{len(w)}"
                rows.append(a)
        return pd.DataFrame(rows)

    primaries = [f"vp6_{c}20" for c in st._CANDS]
    act = _act_table(ref, primaries + ["vp6_allnew20"])
    act.to_csv(out_dir / "active_stats.csv", index=False)
    act_solo = _act_table("mix_screen25", [f"solo_{c}20" for c in st._CANDS])
    act_solo.to_csv(out_dir / "active_stats_solo.csv", index=False)

    f = full_net.set_index("book")
    sh_ref = float(f.loc[ref, "sharpe"])
    verdicts: dict[str, bool] = {}
    vrows = []
    for c in st._CANDS:
        name = f"vp6_{c}20"
        sh = float(f.loc[name, "sharpe"])
        wins = window_wins(res, name, ref)
        n_win = int(wins["win"].sum())
        ns = nulls[name]
        q90 = float(ns.quantile(0.90))
        ok = sh > sh_ref and n_win >= 10 and sh >= q90
        verdicts[c] = ok
        ns.to_csv(out_dir / f"null_sharpes_{name}.csv", index=False)
        vrows.append({"candidate": c, "sharpe": sh, "ref_sharpe": sh_ref,
                      "window_wins": f"{n_win}/{len(wins)}",
                      "null_q90": q90,
                      "pct_draws_beaten": float((ns < sh).mean()),
                      "avg_n_names": float(f.loc[name, "avg_n_names"]),
                      "verdict": "PASS" if ok else "FAIL"})
    vtab = pd.DataFrame(vrows)
    vtab.to_csv(out_dir / "verdicts.csv", index=False)

    import loserscreen
    md = [
        "# Loser-screen v5 — new-candidate vetoes on the ratified base",
        "",
        "## Pre-registration (verbatim, fixed before the first run)",
        "", "```", loserscreen.__doc__.strip(), "```", "",
        "## PER-CANDIDATE VERDICTS (bar: Sharpe > base AND >=10/19 wins AND "
        ">= null q90)", "", _fmt(vtab), "",
        f"Costs: {cost_per_side * 1e4:.0f} bps/side. "
        f"{res.pit_checks.shape[0]} PIT assertions pass. Base: {ref} "
        f"Sharpe {sh_ref:.3f}.",
        "",
        "## Active vs vpos6_20 (primary cells + all-five)", "", _fmt(act), "",
        "## Solo vetoes on mix_screen25 (EXPLORATORY standalone strength)",
        "", _fmt(act_solo), "",
        "## Full period (net)", "", _fmt(full_net[FULL_COLS]), "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(md))
    return verdicts


def write_v2(res: st.ScreenResult, out_dir: Path, cost_per_side: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = st.group_defs(res)
    full_net = st.grouping_metrics(res, groups["full"], leg="net")
    block_net = st.grouping_metrics(res, groups["block"], leg="net")
    win_net = st.grouping_metrics(res, groups["window"], leg="net")
    wins_cap = window_wins(res, "cap_screen20", "cap_broad")
    wins_ew = window_wins(res, "screen20", "broad")

    full_net.to_csv(out_dir / "full_net.csv", index=False)
    block_net.to_csv(out_dir / "block_net.csv", index=False)
    win_net.to_csv(out_dir / "window_net.csv", index=False)
    wins_cap.to_csv(out_dir / "window_wins_cap.csv", index=False)
    wins_ew.to_csv(out_dir / "window_wins_ew.csv", index=False)
    res.pit_checks.to_csv(out_dir / "pit_checks.csv", index=False)
    res.book_sizes.to_csv(out_dir / "book_sizes.csv", index=False)

    passed, vlines = verdict(full_net, wins_cap, "cap_screen20", "cap_broad")
    act = pd.DataFrame([s for s in (active_stats(res, "screen20", "broad"),
                                    active_stats(res, "cap_screen20", "cap_broad"))
                        if s])
    act.to_csv(out_dir / "active_stats.csv", index=False)

    # Depth curve: Sharpe / excess-vs-reference by drop depth, per weighting.
    f = full_net.set_index("book")
    rows = []
    for d in (0, 10, 20, 30, 40, 50, 60, 70):
        row = {"drop_pct": d}
        for pfx, ref in (("", "broad"), ("cap_", "cap_broad")):
            name = f"{pfx}broad" if d == 0 else f"{pfx}screen{d}"
            if name in f.index:
                row[f"{pfx or 'ew_'}sharpe"] = f.loc[name, "sharpe"]
                row[f"{pfx or 'ew_'}cagr"] = f.loc[name, "cagr"]
        rows.append(row)
    curve = pd.DataFrame(rows)
    curve.to_csv(out_dir / "depth_curve.csv", index=False)

    import loserscreen
    md = [
        "# Loser-screen v2 — cap-weighted confirmation + depth curve",
        "",
        "## Pre-registration (verbatim, fixed before the first run)",
        "", "```", loserscreen.__doc__.strip(), "```", "",
        f"## CONFIRMATORY VERDICT (cap_screen20 vs cap_broad): "
        f"**{'PASS' if passed else 'FAIL'}**", "", vlines, "",
        f"Costs: {cost_per_side * 1e4:.0f} bps/side. "
        f"{res.pit_checks.shape[0]} PIT assertions pass.",
        "",
        "## Active return significance", "", _fmt(act), "",
        "## Depth curve (exploratory — mapping, not tuning)", "", _fmt(curve), "",
        "## Full period (net)", "", _fmt(full_net[FULL_COLS]), "",
        "## Per-window: cap_screen20 vs cap_broad (net total return)", "",
        _fmt(wins_cap), "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(md))
