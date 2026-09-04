"""SEC EDGAR module — filings & Form 4 insider transactions.

Uses the public EDGAR REST APIs with a descriptive User-Agent and a rate
limiter that stays under SEC's ~10 req/s fair-access limit. Retrieves the
latest 10-K / 10-Q (with text), recent 8-K metadata, and parses Form 4
insider transactions, then derives insider flags (CEO/CFO/Director purchase,
large purchase, cluster buying). All storage is append-only and de-duplicated.

Run standalone:
    python -m data.sec_data --tickers AAPL --days 90
    python -m data.sec_data --forms 10-K 4 --tickers MSFT
"""
from __future__ import annotations

import argparse
import json
import warnings
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone

import requests
from bs4 import BeautifulSoup

try:  # quiet bs4 when a filing's primary doc is XML rather than HTML
    from bs4 import XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except ImportError:  # pragma: no cover
    pass

from .config import load_config
from .db import Database, get_db
from .providers import ProviderRegistry
from .utils import RateLimiter, get_logger, retry, safe_float, to_iso_date

log = get_logger("sec_data")

EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/"
MAX_TEXT_CHARS = 600_000          # cap stored filing text
TEXT_FORMS = {"10-K", "10-Q"}     # forms for which we fetch & store body text


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _dedup_key(accession: str, txn: dict) -> str:
    """Deterministic identity for an insider transaction so re-runs never
    duplicate it (NULL price grants included)."""
    parts = [accession, txn.get("insider_name") or "", txn.get("transaction_date") or "",
             txn.get("transaction_code") or "", f"{txn.get('shares')}",
             f"{txn.get('price')}", txn.get("ownership_type") or ""]
    return "|".join(parts)


def _headers(cfg) -> dict:
    ua = cfg.get("sec", "user_agent", default="Mahajan Research research@example.com")
    return {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}


class EdgarClient:
    """Rate-limited EDGAR HTTP client with ticker->CIK resolution."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        rate = float(cfg.get("sec", "max_requests_per_second", default=8))
        self.limiter = RateLimiter(rate)
        self.headers = _headers(cfg)
        self._cik_map: dict[str, str] | None = None
        self.cache_file = cfg.path("paths", "cache_dir", default="cache") / "edgar_ciks.json"

    def _get(self, url: str, as_json: bool = True):
        self.limiter.wait()

        def _fetch():
            r = requests.get(url, headers=self.headers, timeout=30)
            r.raise_for_status()
            return r.json() if as_json else r.content

        return retry(_fetch, logger=log, what=f"EDGAR GET {url}")

    def cik_map(self) -> dict[str, str]:
        if self._cik_map is not None:
            return self._cik_map
        data = None
        if self.cache_file.exists():
            try:
                data = json.loads(self.cache_file.read_text())
            except Exception:  # noqa: BLE001
                data = None
        if data is None:
            data = self._get(EDGAR_TICKERS_URL)
            if data:
                self.cache_file.write_text(json.dumps(data))
        mapping: dict[str, str] = {}
        for entry in (data or {}).values():
            tkr = str(entry.get("ticker", "")).upper().replace(".", "-")
            cik = str(entry.get("cik_str", "")).zfill(10)
            if tkr:
                mapping[tkr] = cik
        self._cik_map = mapping
        return mapping

    def resolve_cik(self, ticker: str) -> str | None:
        return self.cik_map().get(ticker.upper())

    def submissions(self, cik: str) -> dict | None:
        return self._get(SUBMISSIONS_URL.format(cik=cik))

    def doc_text(self, cik: str, accession: str, primary_doc: str) -> str | None:
        acc = accession.replace("-", "")
        url = ARCHIVE_BASE.format(cik=int(cik), acc=acc) + primary_doc
        content = self._get(url, as_json=False)
        if not content:
            return None
        try:
            soup = BeautifulSoup(content, "lxml")
            text = soup.get_text(" ", strip=True)
        except Exception:  # noqa: BLE001
            text = content.decode("utf-8", errors="ignore")
        return text[:MAX_TEXT_CHARS]

    def form4_xml(self, cik: str, accession: str, primary_doc: str) -> bytes | None:
        acc = accession.replace("-", "")
        base = ARCHIVE_BASE.format(cik=int(cik), acc=acc)
        # Raw ownership XML lives in the filing folder; strip any xsl/ prefix.
        candidate = primary_doc.split("/")[-1] if primary_doc else ""
        if candidate.endswith(".xml"):
            content = self._get(base + candidate, as_json=False)
            if content and b"ownershipDocument" in content:
                return content
        # Fall back to the folder index to locate the ownership xml.
        index = self._get(base + "index.json")
        if not index:
            return None
        for item in index.get("directory", {}).get("item", []):
            name = item.get("name", "")
            if name.endswith(".xml") and not name.lower().startswith(("xsl", "r")):
                content = self._get(base + name, as_json=False)
                if content and b"ownershipDocument" in content:
                    return content
        return None


# ---------------------------------------------------------------------------
# Form 4 parsing
# ---------------------------------------------------------------------------
def parse_form4(xml_bytes: bytes) -> dict | None:
    """Parse an ownership (Form 4) XML into owner + transaction records."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return None

    def ftext(node, path) -> str | None:
        return node.findtext(path) if node is not None else None

    owner_name = root.findtext(".//{*}reportingOwnerId/{*}rptOwnerName")
    rel = root.find(".//{*}reportingOwnerRelationship")
    is_director = (ftext(rel, "{*}isDirector") or "0").strip() in ("1", "true")
    is_officer = (ftext(rel, "{*}isOfficer") or "0").strip() in ("1", "true")
    is_ten = (ftext(rel, "{*}isTenPercentOwner") or "0").strip() in ("1", "true")
    officer_title = ftext(rel, "{*}officerTitle")
    titles = []
    if officer_title:
        titles.append(officer_title)
    if is_director:
        titles.append("Director")
    if is_ten:
        titles.append("10% Owner")
    title = ", ".join(titles) if titles else ("Officer" if is_officer else None)

    txns = []
    for tx in root.findall(".//{*}nonDerivativeTransaction"):
        code = tx.findtext(".//{*}transactionCoding/{*}transactionCode")
        ad = tx.findtext(".//{*}transactionAcquiredDisposedCode/{*}value")
        shares = safe_float(tx.findtext(".//{*}transactionShares/{*}value"))
        price = safe_float(tx.findtext(".//{*}transactionPricePerShare/{*}value"))
        tdate = to_iso_date(tx.findtext(".//{*}transactionDate/{*}value"))
        own = tx.findtext(".//{*}directOrIndirectOwnership/{*}value")
        if shares is None or tdate is None:
            continue
        txns.append({
            "insider_name": owner_name,
            "insider_title": title,
            "transaction_code": (code or "").strip(),
            "transaction_type": {"A": "Acquired", "D": "Disposed"}.get(
                (ad or "").strip(), None),
            "shares": shares,
            "price": price,
            "value": (shares * price) if (shares and price) else None,
            "transaction_date": tdate,
            "ownership_type": {"D": "Direct", "I": "Indirect"}.get(
                (own or "").strip(), None),
            "is_purchase": int((code or "").strip().upper() == "P"),
        })
    return {"owner_name": owner_name, "title": title, "is_director": is_director,
            "transactions": txns}


# ---------------------------------------------------------------------------
# Insider flags
# ---------------------------------------------------------------------------
def derive_insider_flags(db: Database, ticker: str, cfg,
                         lookback_days: int | None = None) -> int:
    large_usd = float(cfg.get("sec", "insider_large_purchase_usd", default=1_000_000))
    cluster_min = int(cfg.get("sec", "cluster_buy_min_insiders", default=3))
    days = lookback_days if lookback_days is not None else int(
        cfg.get("sec", "recent_filing_days", default=90))
    cutoff = (date.today() - timedelta(days=days)).isoformat()

    purchases = db.query(
        "SELECT insider_name, insider_title, shares, price, value, transaction_date, "
        "accession_number FROM insider_transactions "
        "WHERE ticker = ? AND is_purchase = 1 AND transaction_date >= ?",
        (ticker, cutoff))
    flags: list[dict] = []
    now = _now()
    for p in purchases:
        title = (p["insider_title"] or "").upper()
        base = {"ticker": ticker, "transaction_date": p["transaction_date"],
                "accession_number": p["accession_number"], "created_at": now}
        if "CEO" in title or "CHIEF EXECUTIVE" in title:
            flags.append({**base, "flag_type": "CEO Purchase", "detail": p["insider_name"]})
        if "CFO" in title or "CHIEF FINANCIAL" in title:
            flags.append({**base, "flag_type": "CFO Purchase", "detail": p["insider_name"]})
        if "DIRECTOR" in title:
            flags.append({**base, "flag_type": "Director Purchase", "detail": p["insider_name"]})
        if p["value"] and p["value"] >= large_usd:
            flags.append({**base, "flag_type": "Large Purchase",
                          "detail": f"${p['value']:,.0f}"})

    distinct_buyers = {p["insider_name"] for p in purchases if p["insider_name"]}
    if len(distinct_buyers) >= cluster_min:
        flags.append({"ticker": ticker, "transaction_date": cutoff,
                      "flag_type": "Cluster Buying", "accession_number": "",
                      "detail": f"{len(distinct_buyers)} insiders buying", "created_at": now})

    return db.insert_ignore("insider_flags", flags) if flags else 0


# ---------------------------------------------------------------------------
# FMP insider backfill — multi-year Form-4 history
# ---------------------------------------------------------------------------
# FMP's `transactionType` is the full descriptive code (e.g. "P-Purchase",
# "S-Sale", "M-Exempt"). Split on the first '-' to recover the single-letter
# code the existing flag logic and `is_purchase` derivation expect.
_TXN_TYPE_MAP = {"A": "Acquired", "D": "Disposed"}
_OWNERSHIP_MAP = {"D": "Direct", "I": "Indirect"}


def _accession_from_url(url: str | None) -> str:
    """Recover the dash-formatted accession number from a Form-4 filing URL.

    Example: https://www.sec.gov/Archives/edgar/data/320193/000114036126025622/0001140361-26-025622-index.htm
            -> "0001140361-26-025622"
    Returns "" when the URL is missing or unparsable; dedup_key still works
    since it composes multiple fields.
    """
    if not url:
        return ""
    last = url.rsplit("/", 1)[-1]
    if last.endswith("-index.htm"):
        return last[: -len("-index.htm")]
    return ""


def _fmp_row_to_txn(symbol: str, row: dict) -> dict | None:
    raw_type = (row.get("transactionType") or "").strip()
    code_letter = raw_type.split("-", 1)[0].strip().upper() if raw_type else ""
    shares = safe_float(row.get("securitiesTransacted"))
    tdate = to_iso_date(row.get("transactionDate"))
    if shares is None or not tdate:
        return None
    price = safe_float(row.get("price"))
    # FMP returns 0 for grants and exempt transactions; treat as unknown so
    # the derived `value` stays NULL (matches the EDGAR-path semantics).
    if price == 0:
        price = None
    ad = (row.get("acquisitionOrDisposition") or "").strip().upper()
    own = (row.get("directOrIndirect") or "").strip().upper()
    return {
        "ticker": symbol,
        "cik": str(row.get("companyCik") or ""),
        "accession_number": _accession_from_url(row.get("url")),
        "insider_name": row.get("reportingName"),
        "insider_title": row.get("typeOfOwner"),
        "transaction_type": _TXN_TYPE_MAP.get(ad),
        "transaction_code": code_letter,
        "shares": shares,
        "price": price,
        "value": (shares * price) if (shares and price) else None,
        "transaction_date": tdate,
        "ownership_type": _OWNERSHIP_MAP.get(own),
        "is_purchase": int(code_letter == "P"),
    }


def update_insiders_from_fmp(db: Database, tickers: list[str],
                             registry: ProviderRegistry,
                             max_pages: int = 5,
                             flag_lookback_days: int = 365 * 4) -> dict:
    """Backfill insider_transactions + insider_flags via FMP.

    Independent of the SEC EDGAR loop; safe to run before or after it.
    `flag_lookback_days` widens the flag derivation window so a deep
    backfill produces flags across the backtest history, not just the
    rolling 90-day cutoff used by the original EDGAR path.
    """
    cfg = load_config()
    provider = registry.insiders()
    if provider is None or not hasattr(provider, "get_insider_trades"):
        log.warning("No FMP provider for insiders; skipping FMP backfill")
        return {"insider_txns": 0, "flags": 0, "failed": list(tickers)}

    now = _now()
    totals = {"insider_txns": 0, "flags": 0}
    failed: list[str] = []
    for i, ticker in enumerate(tickers, 1):
        try:
            raw = provider.get_insider_trades(ticker, max_pages=max_pages)
            if not raw:
                continue
            rows: list[dict] = []
            for r in raw:
                txn = _fmp_row_to_txn(ticker, r)
                if txn is None:
                    continue
                txn["fetched_at"] = now
                txn["dedup_key"] = _dedup_key(txn["accession_number"], txn)
                rows.append(txn)
            if rows:
                totals["insider_txns"] += db.insert_ignore(
                    "insider_transactions", rows)
            totals["flags"] += derive_insider_flags(
                db, ticker, cfg, lookback_days=flag_lookback_days)
        except Exception as exc:  # noqa: BLE001
            log.warning("FMP insider backfill failed for %s: %s", ticker, exc)
            failed.append(ticker)
        if i % 25 == 0 or i == len(tickers):
            log.info("FMP insiders: %d/%d (%d txns, %d flags)",
                     i, len(tickers), totals["insider_txns"], totals["flags"])
    totals["failed"] = failed
    return totals


# ---------------------------------------------------------------------------
# Per-ticker processing
# ---------------------------------------------------------------------------
def _recent_filings(subs: dict, forms: list[str], cutoff: str) -> list[dict]:
    recent = subs.get("filings", {}).get("recent", {})
    keys = ("form", "accessionNumber", "filingDate", "reportDate",
            "primaryDocument", "primaryDocDescription")
    cols = {k: recent.get(k, []) for k in keys}
    n = len(cols["form"])
    out = []
    for i in range(n):
        form = cols["form"][i]
        fdate = cols["filingDate"][i] if i < len(cols["filingDate"]) else None
        if form not in forms or (fdate and fdate < cutoff):
            continue
        out.append({k: (cols[k][i] if i < len(cols[k]) else None) for k in keys})
    return out


def process_ticker(db: Database, client: EdgarClient, ticker: str, forms: list[str],
                   days: int, fetch_text: bool, parse_insiders: bool) -> dict:
    stats = {"filings": 0, "insider_txns": 0, "flags": 0}
    cik = client.resolve_cik(ticker)
    if not cik:
        log.debug("No CIK for %s", ticker)
        return stats
    subs = client.submissions(cik)
    if not subs:
        return stats
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    filings = _recent_filings(subs, forms, cutoff)

    now = _now()
    filing_rows: list[dict] = []
    text_fetched = {"10-K": False, "10-Q": False}
    for f in filings:
        form = f["form"]
        text = None
        if (fetch_text and form in TEXT_FORMS and not text_fetched.get(form)
                and f.get("primaryDocument")):
            text = client.doc_text(cik, f["accessionNumber"], f["primaryDocument"])
            text_fetched[form] = True  # only latest of each
        acc = f["accessionNumber"]
        doc = f.get("primaryDocument") or ""
        url = (ARCHIVE_BASE.format(cik=int(cik), acc=acc.replace("-", "")) + doc
               if doc else None)
        filing_rows.append({
            "ticker": ticker, "cik": cik, "form_type": form,
            "filing_date": f.get("filingDate"), "report_date": f.get("reportDate"),
            "accession_number": acc, "primary_doc": doc, "primary_doc_url": url,
            "filing_text": text, "fetched_at": now,
        })

        if parse_insiders and form == "4" and doc:
            xml = client.form4_xml(cik, acc, doc)
            if xml:
                parsed = parse_form4(xml)
                if parsed and parsed["transactions"]:
                    txn_rows = [{
                        "ticker": ticker, "cik": cik, "accession_number": acc,
                        "fetched_at": now, "dedup_key": _dedup_key(acc, t),
                        **t} for t in parsed["transactions"]]
                    stats["insider_txns"] += db.insert_ignore(
                        "insider_transactions", txn_rows)

    if filing_rows:
        stats["filings"] = db.insert_ignore("sec_filings", filing_rows)
    if parse_insiders:
        stats["flags"] = derive_insider_flags(db, ticker, client.cfg)
    return stats


def update_sec(db: Database, tickers: list[str], forms: list[str] | None = None,
               days: int | None = None, fetch_text: bool = True,
               registry: ProviderRegistry | None = None,
               insider_backfill_pages: int = 5) -> dict:
    cfg = load_config()
    forms = forms or cfg.get("sec", "default_form_types", default=["10-K", "10-Q", "8-K", "4"])
    days = days or int(cfg.get("sec", "recent_filing_days", default=90))
    registry = registry or ProviderRegistry(cfg)
    client = EdgarClient(cfg)
    if not client.cik_map():
        log.error("Could not load EDGAR ticker->CIK map; skipping SEC")
        return {"filings": 0, "insider_txns": 0, "flags": 0, "failed": tickers}

    # Route insider parsing: prefer FMP (multi-year history); fall back to
    # EDGAR Form-4 parsing if no FMP key. Regardless, the EDGAR loop still
    # ingests 10-K/10-Q/8-K filings + text.
    insider_provider = registry.insiders()
    use_fmp_insiders = (insider_provider is not None
                        and hasattr(insider_provider, "get_insider_trades"))
    parse_insiders = ("4" in forms) and not use_fmp_insiders

    totals = {"filings": 0, "insider_txns": 0, "flags": 0}
    failed: list[str] = []
    for i, ticker in enumerate(tickers, 1):
        try:
            s = process_ticker(db, client, ticker, forms, days, fetch_text, parse_insiders)
            for k in totals:
                totals[k] += s[k]
        except Exception as exc:  # noqa: BLE001
            log.warning("SEC processing failed for %s: %s", ticker, exc)
            failed.append(ticker)
        if i % 10 == 0 or i == len(tickers):
            log.info("SEC: %d/%d (%d filings, %d insider txns)",
                     i, len(tickers), totals["filings"], totals["insider_txns"])

    if use_fmp_insiders and "4" in forms:
        fmp = update_insiders_from_fmp(
            db, tickers, registry, max_pages=insider_backfill_pages)
        totals["insider_txns"] += fmp["insider_txns"]
        totals["flags"] += fmp["flags"]
        failed.extend(fmp["failed"])

    totals["failed"] = failed
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch SEC filings & insider txns")
    parser.add_argument("--tickers", nargs="*")
    parser.add_argument("--forms", nargs="*", help="Form types (default from config)")
    parser.add_argument("--days", type=int, help="Lookback window in days")
    parser.add_argument("--no-text", action="store_true", help="Skip filing body text")
    parser.add_argument("--insider-backfill-pages", type=int, default=5,
                        help="FMP insider-trading pages per ticker (1000 txns/page)")
    parser.add_argument("--insiders-only", action="store_true",
                        help="Skip the EDGAR filing loop; only run the FMP "
                             "insider backfill (used for departed PIT names)")
    args = parser.parse_args()
    cfg = load_config()
    get_logger("sec_data", log_file=cfg.log_file)
    with get_db() as db:
        tickers = [t.upper() for t in args.tickers] if args.tickers else db.universe_tickers()
        if args.insiders_only:
            registry = ProviderRegistry(cfg)
            stats = update_insiders_from_fmp(
                db, tickers, registry, max_pages=args.insider_backfill_pages)
            stats.setdefault("filings", 0)
        else:
            stats = update_sec(db, tickers, forms=args.forms, days=args.days,
                               fetch_text=not args.no_text,
                               insider_backfill_pages=args.insider_backfill_pages)
    print(f"Filings stored      : {stats['filings']}")
    print(f"Insider txns parsed : {stats['insider_txns']}")
    print(f"Insider flags       : {stats['flags']}")
    print(f"Failed              : {len(stats['failed'])}")


if __name__ == "__main__":
    main()
