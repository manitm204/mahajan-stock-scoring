"""LLM pilot phase 2 — production research-overlay replication, PIT.

Production prompts VERBATIM (OVERLAY_SYSTEM + OVERLAY_USER_TEMPLATE) and
production validate_overlay for enum coercion. The quant block is rebuilt
point-in-time from the clean window caches: composite (sector percentile),
universe percentile, the 8 parent scores with production's 70/40
strong/weak driver tags, and PIT price stats. filing_analysis = the phase-1
filing JSONs; earnings/risk/insider = null (production-tolerated).

Resume-safe: one JSON per (cohort, ticker) under output/llm_pilot/overlay/.

Usage: python scripts/llm_pilot_overlay_score.py   (~2h for 239 names)
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from analysis.overlay_analyzer import validate_overlay
from analysis.prompts import OVERLAY_SYSTEM, OVERLAY_USER_TEMPLATE
from analysis.quant_context import STRONG_DRIVER_MIN, WEAK_DRIVER_MAX
from backtesting import data_loader as dl
from data.db import get_db
from run_loserscreen_study import PANEL_START, PRICE_END, _load_panel
from scripts.llm_pilot_score import MODEL, api_key, call, pace
from vixtilt import backtest as bt
from vixtilt.baseline import load_or_build
from vixtilt.windows import semiannual_windows

OUT = Path("output/llm_pilot")
OVERLAY_DIR = OUT / "overlay"
FORMATIONS = {"2024": "2024-06-28", "2025": "2025-06-30"}
MAX_TOKENS = 3072   # production overlay DEFAULT_MAX_TOKENS


def pit_price_stats(px: pd.Series, d0: str) -> dict[str, float]:
    hist = px.dropna()
    hist = hist[hist.index <= d0]
    if len(hist) < 60:
        return {}
    out = {}
    for label, n in (("return_20d", 20), ("return_60d", 60),
                     ("return_252d", 252)):
        if len(hist) > n:
            out[label] = float(hist.iloc[-1] / hist.iloc[-n - 1] - 1)
    tail = hist.iloc[-252:]
    out["distance_from_52w_high"] = float(hist.iloc[-1] / tail.max() - 1)
    r = hist.iloc[-21:].pct_change().dropna()
    if len(r) >= 15:
        out["volatility_20d"] = float(r.std(ddof=1) * np.sqrt(252))
    return out


def quant_block(ticker: str, d0: str, frame: pd.DataFrame, comp: pd.Series,
                sector: str, px: pd.Series) -> str:
    lines = [f"Company: {ticker}", f"Sector: {sector}", f"As of: {d0}"]
    score = float(comp[ticker])
    pct = float(comp.rank(pct=True)[ticker] * 100)
    lines.append(f"Composite score: {score:.1f}/100  (universe percentile "
                 f"{pct:.0f} of {comp.notna().sum()} names)  Signal: LONG candidate")
    lines.append("")
    lines.append("Parent factor scores (0-100, sector-relative percentile; "
                 "higher = better):")
    parents = frame.loc[ticker].dropna()
    for factor, s in parents.sort_values(ascending=False).items():
        tag = ("  <- strong driver" if s >= STRONG_DRIVER_MIN else
               "  <- weak driver" if s <= WEAK_DRIVER_MAX else "")
        lines.append(f"  {factor:14s} {s:5.1f}{tag}")
    stats = pit_price_stats(px, d0)
    if stats:
        lines.append("")
        lines.append("Recent price action:")
        for k, v in stats.items():
            lines.append(f"  {k:24s} {v:+.1%}")
    return "\n".join(lines)


def main() -> int:
    key = api_key()
    cohorts = pd.read_csv(OUT / "cohorts.csv", dtype={"cohort": str})
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    panel = _load_panel()
    rebals = list(panel.rebal_dates)
    with get_db() as db:
        matrix = dl.load_price_matrix(db, panel.universe, PANEL_START, PRICE_END)
        sectors = dl.global_sectors(db)
    frames, comps = {}, {}
    for c, d0 in FORMATIONS.items():
        win = next(w for w in semiannual_windows(2017, "2026-06-30")
                   if d0 in w.test_rebals(rebals))
        wb = load_or_build(panel, matrix, win, verbose=False)
        frames[c] = wb.parent_test[d0]
        comps[c] = bt.composite_from_parents(frames[c], wb.parent_weights,
                                             sectors)

    todo = []
    for r in cohorts.itertuples():
        out_path = OVERLAY_DIR / f"{r.cohort}_{r.ticker}.json"
        filing_path = OUT / "scores" / f"{r.cohort}_{r.ticker}.json"
        if out_path.exists() or not filing_path.exists():
            continue
        filing = json.load(open(filing_path))
        if "error" in filing:
            continue
        todo.append((r, filing, out_path))
    print(f"{len(todo)} overlay calls to make", flush=True)

    import time
    t0 = time.time()
    n_ok = n_fail = 0
    for i, (r, filing, out_path) in enumerate(todo, 1):
        filing.pop("_meta", None)
        block = quant_block(r.ticker, FORMATIONS[r.cohort], frames[r.cohort],
                            comps[r.cohort], r.sector,
                            matrix.get(r.ticker, pd.Series(dtype=float)))
        user = OVERLAY_USER_TEMPLATE.format(
            quant_context_block=block,
            earnings_json="null",
            filing_json=json.dumps(filing, separators=(",", ":")),
            risk_json="null",
            insider_json="null")
        data = None
        for _ in range(2):
            data, hdrs = call(key, OVERLAY_SYSTEM, user, max_tokens=MAX_TOKENS)
            if data is not None:
                break
        if data is None:
            data = {"error": "unparseable"}
            n_fail += 1
        else:
            data = validate_overlay(data, ticker=r.ticker)
            data["_meta"] = {"cohort": r.cohort, "ticker": r.ticker,
                             "model": MODEL, "phase": "overlay"}
            n_ok += 1
        json.dump(data, open(out_path, "w"), indent=1)
        pace(hdrs)
        if i % 20 == 0:
            rate = i / ((time.time() - t0) / 60)
            print(f"  {i}/{len(todo)} ok={n_ok} fail={n_fail} "
                  f"{rate:.1f}/min eta {(len(todo) - i) / rate / 60:.1f}h",
                  flush=True)
    print(f"done: ok={n_ok} fail={n_fail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
