"""Two-sleeve staggered semiannual reform + emergency-exit overlay.

Policies compared on vpos6_20 (lot-level sim, 10 bps/side; taxed runs use
HIFO, ST 24% / LT 15%):
  single      — semiannual full reform, one book (6 phases).
  sleeves     — two $50k sleeves reformed 3 months apart (each dollar still
                trades semiannually; averages the phase lottery). 3 offsets.
  single+em   — semiannual + monthly EMERGENCY EXITS: a held name is sold
                mid-cycle only if DECISIVELY excluded — bottom 15% by
                composite (vs the 25% screen line) or bottom 10% of a veto
                parent (vs the 20% veto line). Exits only, no mid-cycle
                buys; proceeds sit in cash until the sleeve's next reform
                (conservative).
  sleeves+em  — both.

Usage: python scripts/loserscreen_sleeves_events.py   (~4min)
Outputs: output/loserscreen_final/sleeves_events.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from backtesting import data_loader as dl
from data.db import get_db
from loserscreen import study as st
from loserscreen.mcap import market_caps
from run_loserscreen_study import PANEL_START, PRICE_END, _load_panel
from scripts.loserscreen_ops_tax import (BOOK, LT_RATE, OUT, ST_RATE,
                                         START_CAPITAL, simulate)

EMERG_BOOK = st.BookSpec("emerg", "still-acceptable zone (hysteresis)",
                         weighting="mix", base_rule="screen15",
                         veto_parents=st._VETO2_SUBSETS["vpos6_20"],
                         veto_pct=0.10)


def metrics(values: list[float], start: float, tax_paid: float,
            liq_final: float) -> dict:
    v = pd.Series(values)
    r = v.pct_change().dropna()
    years = len(v) / 12
    return {"final_pre_liq": float(v.iloc[-1]),
            "final_liquidated": liq_final,
            "cagr": (float(v.iloc[-1]) / start) ** (1 / years) - 1,
            "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(12)),
            "tax_paid": tax_paid}


def main() -> int:
    panel = _load_panel()
    with get_db() as db:
        matrix = dl.load_price_matrix(db, panel.universe, PANEL_START, PRICE_END)
        sectors = dl.global_sectors(db)
        mcaps = market_caps(db, list(panel.rebal_dates), panel.universe, matrix)
    res = st.run_study(panel, matrix, sectors, books=[BOOK, EMERG_BOOK],
                       mcaps=mcaps, verbose=False)
    w = res.weights[BOOK.name]
    emerg = {d: set(v.index) for d, v in res.weights["emerg"].items()}
    dates = [d for d in sorted(w) if d in matrix.index and len(w[d])]

    rows = []
    for taxed in (False, True):
        s, l = (ST_RATE, LT_RATE) if taxed else (0.0, 0.0)

        for em, tag in ((None, "single"), (emerg, "single+em")):
            for ph in range(6):
                r = simulate(dates, w, matrix, 6, False, s, l, phase=ph,
                             emergency=em)
                m = metrics(r["values"], START_CAPITAL, r["tax_paid"],
                            r["final_liquidated"])
                rows.append({"policy": tag, "taxed": taxed, "phase": ph,
                             "em_sells_per_yr":
                                 r["emergency_sells"] / (len(dates) / 12),
                             **m})

        for em, tag in ((None, "sleeves"), (emerg, "sleeves+em")):
            for ph in range(3):
                a = simulate(dates, w, matrix, 6, False, s, l, phase=ph,
                             start=START_CAPITAL / 2, emergency=em)
                b = simulate(dates, w, matrix, 6, False, s, l, phase=ph + 3,
                             start=START_CAPITAL / 2, emergency=em)
                vals = [x + y for x, y in zip(a["values"], b["values"])]
                m = metrics(vals, START_CAPITAL,
                            a["tax_paid"] + b["tax_paid"],
                            a["final_liquidated"] + b["final_liquidated"])
                rows.append({"policy": tag, "taxed": taxed, "phase": ph,
                             "em_sells_per_yr":
                                 (a["emergency_sells"] + b["emergency_sells"])
                                 / (len(dates) / 12),
                             **m})

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(OUT / "sleeves_events.csv", index=False)

    for taxed in (False, True):
        sub = df[df.taxed == taxed]
        key = "final_liquidated" if taxed else "final_pre_liq"
        agg = sub.groupby("policy").agg(
            n=("phase", "count"),
            final_mean=(key, "mean"), final_min=(key, "min"),
            final_max=(key, "max"), sharpe_mean=("sharpe", "mean"),
            em_sells_yr=("em_sells_per_yr", "mean"),
            tax=("tax_paid", "mean"),
        ).reindex(["single", "sleeves", "single+em", "sleeves+em"])
        agg["spread"] = agg.final_max - agg.final_min
        pd.set_option("display.float_format", "{:,.2f}".format)
        print(f"=== {'AFTER-TAX (liquidated)' if taxed else 'PRE-TAX'} ===")
        print(agg.to_string())
        print()
    print(f"wrote {OUT}/sleeves_events.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
