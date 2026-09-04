"""LLM pilot — resolve no_cik manifest rows (delisted tickers) via FMP.

EDGAR's ticker->CIK map only lists current registrants, so names delisted
since formation (acquisitions etc.) fail in llm_pilot_filings.py. Dropping
them would be survivorship bias — this patch resolves their CIKs through the
FMP stable /profile endpoint and re-runs the same pick_filing logic, then
rewrites the manifest in place.

Usage: python scripts/llm_pilot_fix_ciks.py   (after llm_pilot_filings.py)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import requests

from data.config import load_config
from data.db import get_db
from data.sec_data import EdgarClient
from scripts.llm_pilot_filings import MANIFEST, pick_filing


def fmp_cik(ticker: str, key: str) -> str | None:
    r = requests.get("https://financialmodelingprep.com/stable/profile",
                     params={"symbol": ticker, "apikey": key}, timeout=30)
    data = r.json()
    if isinstance(data, list) and data and data[0].get("cik"):
        return str(data[0]["cik"]).zfill(10)
    return None


def main() -> int:
    m = pd.read_csv(MANIFEST, dtype={"cohort": str})
    bad = m[m.status == "no_cik"]
    if bad.empty:
        print("no no_cik rows — nothing to do")
        return 0
    key = None
    for line in open(".env"):
        if line.startswith("FMP_API_KEY="):
            key = line.strip().split("=", 1)[1].strip()
    if not key:
        raise SystemExit("FMP_API_KEY not found in .env")

    cfg = load_config()
    client = EdgarClient(cfg)
    ciks = {t: fmp_cik(t, key) for t in bad.ticker.unique()}
    print("resolved:", {t: c for t, c in ciks.items() if c},
          "| unresolved:", [t for t, c in ciks.items() if not c])

    subs_cache: dict[str, dict | None] = {}
    with get_db() as db:
        for i, r in bad.iterrows():
            cik = ciks.get(r.ticker)
            if not cik:
                continue
            if cik not in subs_cache:
                subs_cache[cik] = client.submissions(cik)
            subs = subs_cache[cik]
            if not subs:
                m.loc[i, "status"] = "no_submissions"
                continue
            picked = pick_filing(db, client, cik, subs, r.ticker, r.formation)
            if picked is None:
                m.loc[i, "status"] = "no_mdna"
                continue
            for k, v in picked.items():
                m.loc[i, k] = v

    m.to_csv(MANIFEST, index=False)
    for c, grp in m.groupby("cohort"):
        ok = (grp.status == "ok").sum()
        print(f"cohort {c}: {ok}/{len(grp)} ok ({ok / len(grp):.1%}); "
              f"fails: {grp[grp.status != 'ok'].status.value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
