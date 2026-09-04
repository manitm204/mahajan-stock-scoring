"""LLM pilot — EDGAR backfill of point-in-time filings for the two cohorts.

For each (cohort, ticker) in output/llm_pilot/cohorts.csv, fetch the most
recent 10-K filed ON OR BEFORE the formation date whose MD&A (Item 7) is
extractable; fallback: most recent 10-Q (Item 2). Filing text is stored
append-only in the existing ``sec_filings`` table; a manifest maps each
(cohort, ticker) to its chosen accession.

Resume-safe: pairs already in the manifest are skipped on re-run.

Usage: python scripts/llm_pilot_filings.py
Output: output/llm_pilot/filings_manifest.csv (+ rows in sec_filings)
"""
from __future__ import annotations

import csv
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from analysis.data_access import extract_mdna_section
from data.config import load_config
from data.db import get_db
from data.sec_data import ARCHIVE_BASE, MAX_TEXT_CHARS, EdgarClient

OUT = Path("output/llm_pilot")
MANIFEST = OUT / "filings_manifest.csv"
FIELDS = ["cohort", "formation", "ticker", "form_type", "filing_date",
          "accession_number", "mdna_chars", "status"]
# newest-first tries per pair: two 10-Ks then two 10-Qs (an older filing is
# still PIT-valid; a 10-K with unextractable MD&A falls through)
MAX_TRIES = {"10-K": 2, "10-Q": 2}


def _all_filings(subs: dict) -> list[dict]:
    recent = subs.get("filings", {}).get("recent", {})
    keys = ("form", "accessionNumber", "filingDate", "reportDate",
            "primaryDocument")
    cols = {k: recent.get(k, []) for k in keys}
    return [{k: (cols[k][i] if i < len(cols[k]) else None) for k in keys}
            for i in range(len(cols["form"]))]


def pick_filing(db, client: EdgarClient, cik: str, subs: dict, ticker: str,
                formation: str) -> dict | None:
    """Newest extractable-MD&A filing at or before formation; stores text."""
    filings = _all_filings(subs)
    for form in ("10-K", "10-Q"):
        cands = sorted((f for f in filings
                        if f["form"] == form and f.get("filingDate")
                        and f["filingDate"] <= formation),
                       key=lambda f: f["filingDate"], reverse=True)
        for f in cands[:MAX_TRIES[form]]:
            acc, doc = f["accessionNumber"], f.get("primaryDocument") or ""
            if not doc:
                continue
            row = db.query_one(
                "SELECT filing_text FROM sec_filings WHERE accession_number=? "
                "AND primary_doc=?", (acc, doc))
            text = row["filing_text"] if row else None
            if text is None:
                text = client.doc_text(cik, acc, doc)
                if not text:
                    continue
                text = text[:MAX_TEXT_CHARS]
                db.insert_ignore("sec_filings", [{
                    "ticker": ticker, "cik": cik, "form_type": form,
                    "filing_date": f["filingDate"],
                    "report_date": f.get("reportDate"),
                    "accession_number": acc, "primary_doc": doc,
                    "primary_doc_url": ARCHIVE_BASE.format(
                        cik=int(cik), acc=acc.replace("-", "")) + doc,
                    "filing_text": text,
                    "fetched_at": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"),
                }])
            mdna = extract_mdna_section(text, form)
            if mdna:
                return {"form_type": form, "filing_date": f["filingDate"],
                        "accession_number": acc, "mdna_chars": len(mdna),
                        "status": "ok"}
    return None


def main() -> int:
    cohorts = pd.read_csv(OUT / "cohorts.csv",
                          dtype={"cohort": str})[
        ["cohort", "formation", "ticker"]].drop_duplicates()
    done: set[tuple[str, str]] = set()
    if MANIFEST.exists():
        m = pd.read_csv(MANIFEST, dtype={"cohort": str})
        done = set(zip(m.cohort, m.ticker))
    todo = [r for r in cohorts.itertuples()
            if (r.cohort, r.ticker) not in done]
    print(f"{len(cohorts)} pairs, {len(done)} done, {len(todo)} to fetch")

    cfg = load_config()
    client = EdgarClient(cfg)
    new_file = not MANIFEST.exists()
    subs_cache: dict[str, dict | None] = {}
    with get_db() as db, open(MANIFEST, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for i, r in enumerate(todo, 1):
            base = {"cohort": r.cohort, "formation": r.formation,
                    "ticker": r.ticker}
            cik = client.resolve_cik(r.ticker)
            if not cik:
                w.writerow({**base, "form_type": "", "filing_date": "",
                            "accession_number": "", "mdna_chars": 0,
                            "status": "no_cik"})
                fh.flush()
                continue
            if cik not in subs_cache:
                subs_cache[cik] = client.submissions(cik)
            subs = subs_cache[cik]
            if not subs:
                w.writerow({**base, "form_type": "", "filing_date": "",
                            "accession_number": "", "mdna_chars": 0,
                            "status": "no_submissions"})
                fh.flush()
                continue
            picked = pick_filing(db, client, cik, subs, r.ticker, r.formation)
            if picked is None:
                picked = {"form_type": "", "filing_date": "",
                          "accession_number": "", "mdna_chars": 0,
                          "status": "no_mdna"}
            w.writerow({**base, **picked})
            fh.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}")

    m = pd.read_csv(MANIFEST, dtype={"cohort": str})
    for c, grp in m.groupby("cohort"):
        ok = (grp.status == "ok").sum()
        print(f"cohort {c}: {ok}/{len(grp)} ok "
              f"({ok / len(grp):.1%}); fails: "
              f"{grp[grp.status != 'ok'].status.value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
