"""LLM pilot — score cohort filings with gpt-oss-120b via the Cerebras API.

Production prompts VERBATIM (analysis/prompts.py FILING_SYSTEM +
FILING_USER_TEMPLATE), production MD&A extraction and edge-preserving
truncation; declared deviations (docs/llm_pilot_design.md amendments):
12,000-char MD&A cap (vs 45,000 in production) and host/model =
Cerebras gpt-oss-120b (documented June 2024 cutoff, strict boundary probe
PASSED — output/llm_pilot/cutoff_probe_gptoss.json). Groq free tier's
~100k-token/day cap made the original Llama 3.3 70B route a 2-week run;
all filings are re-scored uniformly on this one model/host/prompt size.

Resume-safe: one JSON per (cohort, ticker) under output/llm_pilot/scores/;
existing files are skipped. 429s are logged and retried with bounded sleeps.

Usage: python scripts/llm_pilot_score.py   (~1-2h for ~239 filings)
"""
from __future__ import annotations

import json
import re
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import requests

from analysis.base import truncate_preserving_edges
from analysis.data_access import extract_mdna_section
from analysis.prompts import FILING_SYSTEM, FILING_USER_TEMPLATE
from data.db import get_db

OUT = Path("output/llm_pilot")
SCORES = OUT / "scores"
MODEL = "gpt-oss-120b"
URL = "https://api.cerebras.ai/v1/chat/completions"
KEY_NAME = "CEREBRAS_API_KEY"
MAX_MDNA_CHARS = 8_000      # declared pilot deviation (production: 45_000);
                            # sized to Cerebras' 1M-token/day free budget
MAX_TOKENS = 4096           # reasoning tokens count against the completion
RETRIES_PER_FILING = 2      # 1 retry on unparseable JSON, per pre-registration


def api_key() -> str:
    for line in open(Path(__file__).resolve().parent.parent / ".env"):
        if line.startswith(f"{KEY_NAME}="):
            return line.strip().split("=", 1)[1].strip().strip("\"'")
    raise SystemExit(f"{KEY_NAME} not found in .env")


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def call(key: str, system: str, user: str,
         max_tokens: int = MAX_TOKENS) -> tuple[dict | None, dict]:
    """One chat call; returns (parsed json or None, rate headers)."""
    while True:
        try:
            r = requests.post(URL, headers={"Authorization": f"Bearer {key}"},
                          json={"model": MODEL,
                                "messages": [
                                    {"role": "system", "content": system},
                                    {"role": "user", "content": user}],
                                "max_tokens": max_tokens,
                                "temperature": 0,
                                "response_format": {"type": "json_object"}},
                          timeout=180)
        except (requests.Timeout, requests.ConnectionError) as e:
            print(f"  network error ({type(e).__name__}) — sleeping 30s",
                  flush=True)
            time.sleep(30)
            continue
        hdrs = {k.lower(): v for k, v in r.headers.items()}
        if r.status_code == 429:
            wait = min(float(hdrs.get("retry-after", 20)) + 1, 120)
            print(f"  429 (retry-after={hdrs.get('retry-after')}) — "
                  f"sleeping {wait:.0f}s", flush=True)
            time.sleep(wait)
            continue
        if r.status_code >= 500:
            print(f"  {r.status_code} from Groq — sleeping 15s", flush=True)
            time.sleep(15)
            continue
        r.raise_for_status()
        return parse_json(r.json()["choices"][0]["message"]["content"]), hdrs


def pace(hdrs: dict) -> None:
    """Cerebras free tier: 5 req/min, 150 req/HOUR (binding), 1M tokens/day.
    Fixed 25s spacing → ~144 req/h; hard limits fall to the 429 retry loop."""
    time.sleep(25)


def main() -> int:
    key = api_key()
    manifest = pd.read_csv(OUT / "filings_manifest.csv", dtype={"cohort": str})
    manifest = manifest[manifest.status == "ok"]
    # score only pairs in the CURRENT cohort book (manifest may carry rows
    # from a superseded cohort build)
    live = set(map(tuple, pd.read_csv(OUT / "cohorts.csv",
                                      dtype={"cohort": str})[
        ["cohort", "ticker"]].itertuples(index=False)))
    manifest = manifest[[(r.cohort, r.ticker) in live
                         for r in manifest.itertuples()]]
    SCORES.mkdir(parents=True, exist_ok=True)
    todo = [r for r in manifest.itertuples()
            if not (SCORES / f"{r.cohort}_{r.ticker}.json").exists()]
    print(f"{len(manifest)} filings, {len(manifest) - len(todo)} scored, "
          f"{len(todo)} to go", flush=True)

    sectors = pd.read_csv(OUT / "cohorts.csv", dtype={"cohort": str}).set_index(
        ["cohort", "ticker"]).sector.to_dict()

    t0 = time.time()
    n_ok = n_fail = 0
    with get_db() as db:
        for i, r in enumerate(todo, 1):
            row = db.query_one(
                "SELECT filing_text FROM sec_filings WHERE accession_number=? "
                "AND primary_doc IS NOT NULL AND filing_text IS NOT NULL "
                "LIMIT 1", (r.accession_number,))
            mdna = extract_mdna_section(row["filing_text"], r.form_type) \
                if row else None
            out_path = SCORES / f"{r.cohort}_{r.ticker}.json"
            if not mdna:
                json.dump({"error": "no_mdna"}, open(out_path, "w"))
                n_fail += 1
                continue
            mdna = truncate_preserving_edges(mdna, MAX_MDNA_CHARS)
            user = FILING_USER_TEMPLATE.format(
                ticker=r.ticker,
                sector=sectors.get((r.cohort, r.ticker), "Unknown"),
                form_type=r.form_type, filing_date=r.filing_date,
                mdna_text=mdna)
            data = None
            for _ in range(RETRIES_PER_FILING):
                data, hdrs = call(key, FILING_SYSTEM, user)
                if data is not None:
                    break
            if data is None:
                data = {"error": "unparseable"}
                n_fail += 1
            else:
                data["_meta"] = {"cohort": r.cohort, "ticker": r.ticker,
                                 "form_type": r.form_type,
                                 "filing_date": r.filing_date,
                                 "accession": r.accession_number,
                                 "model": MODEL, "mdna_chars": len(mdna)}
                n_ok += 1
            json.dump(data, open(out_path, "w"), indent=1)
            pace(hdrs)
            if i % 20 == 0:
                rate = i / ((time.time() - t0) / 60)
                eta = (len(todo) - i) / rate
                print(f"  {i}/{len(todo)} ok={n_ok} fail={n_fail} "
                      f"{rate:.1f}/min eta {eta / 60:.1f}h", flush=True)

    print(f"done: ok={n_ok} fail={n_fail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
