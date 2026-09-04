"""Institutional holdings module — 13F filings.

Tracks a configured set of investment managers (Berkshire, Pershing Square,
Appaloosa, Baupost, Third Point, ...) by parsing their 13F-HR information
tables from EDGAR. Holdings are stored per (fund, cusip, report period) and
CUSIPs are best-effort resolved to tickers via issuer-name matching against
the universe. Quarter-over-quarter signals (fund count, increases, decreases,
new/closed positions, net share change) are derived per ticker.

Run standalone:
    python -m data.institutional --quarters 4
    python -m data.institutional --funds "Berkshire Hathaway"
"""
from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone

from .config import load_config
from .db import Database, get_db
from .sec_data import ARCHIVE_BASE, EdgarClient
from .utils import get_logger, safe_float, safe_int, to_iso_date

log = get_logger("institutional")

_SUFFIXES = {"INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY",
             "LTD", "LIMITED", "PLC", "LP", "LLC", "HOLDINGS", "HLDGS", "GROUP",
             "THE", "COM", "CL", "CLASS", "A", "B", "C", "NEW", "DEL", "PAR",
             "COMMON", "STOCK", "SHARES", "&"}
VALUE_DOLLAR_THRESHOLD = "2023-01-01"  # 13F values reported in whole dollars after


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize_name(name: str) -> str:
    name = re.sub(r"[^A-Z0-9 ]", " ", (name or "").upper())
    tokens = [t for t in name.split() if t and t not in _SUFFIXES]
    return "".join(tokens)


def build_name_index(db: Database) -> dict[str, str]:
    """Map normalized issuer name -> ticker for CUSIP resolution."""
    idx: dict[str, str] = {}
    for r in db.query("SELECT ticker, company_name FROM universe"):
        key = _normalize_name(r["company_name"] or r["ticker"])
        if key:
            idx.setdefault(key, r["ticker"])
    return idx


def parse_info_table(xml_bytes: bytes) -> list[dict]:
    """Parse a 13F information table XML into holdings, aggregated per CUSIP.

    Large managers report the same security across many infoTable rows (one per
    investment manager / voting-authority bucket). A fund's true position is the
    SUM of those rows, so we aggregate by CUSIP -- not doing so understates big
    positions by an order of magnitude. Option lines (``putCall`` set) are
    excluded so share counts stay clean equity holdings.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []
    agg: dict[str, dict] = {}
    for it in root.findall(".//{*}infoTable"):
        if (it.findtext("{*}putCall") or "").strip():
            continue  # derivative position, not a share holding
        cusip = (it.findtext("{*}cusip") or "").strip().upper()
        shares = safe_int(it.findtext(".//{*}sshPrnamt"))
        if not cusip or shares is None:
            continue
        value = safe_float(it.findtext("{*}value"))
        h = agg.get(cusip)
        if h is None:
            agg[cusip] = {"name": it.findtext("{*}nameOfIssuer"), "cusip": cusip,
                          "value": value, "shares": shares}
        else:
            h["shares"] += shares
            if value is not None:
                h["value"] = (h["value"] or 0) + value
    return list(agg.values())


def _find_info_table_xml(client: EdgarClient, cik: str, accession: str) -> bytes | None:
    acc = accession.replace("-", "")
    base = ARCHIVE_BASE.format(cik=int(cik), acc=acc)
    index = client._get(base + "index.json")
    if not index:
        return None
    candidates = []
    for item in index.get("directory", {}).get("item", []):
        name = item.get("name", "")
        if name.endswith(".xml") and not name.lower().startswith(("xsl", "primary_doc")):
            candidates.append(name)
    # Prefer obvious info-table names first.
    candidates.sort(key=lambda n: ("infotable" not in n.lower(), n))
    for name in candidates:
        content = client._get(base + name, as_json=False)
        if content and b"infoTable" in content:
            return content
    return None


def _recent_13f(subs: dict, quarters: int) -> list[dict]:
    recent = subs.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    out = []
    for i, form in enumerate(forms):
        if not form.startswith("13F-HR"):
            continue
        out.append({
            "accession": recent["accessionNumber"][i],
            "filing_date": recent["filingDate"][i],
            "report_date": (recent["reportDate"][i]
                            if i < len(recent.get("reportDate", [])) else None),
        })
        if len(out) >= quarters:
            break
    return out


def process_fund(db: Database, client: EdgarClient, fund_name: str, cik: str,
                 quarters: int, name_index: dict[str, str]) -> int:
    subs = client.submissions(cik)
    if not subs:
        return 0
    filings = _recent_13f(subs, quarters)
    now = _now()
    stored = 0
    for f in filings:
        xml = _find_info_table_xml(client, cik, f["accession"])
        if not xml:
            continue
        holdings = parse_info_table(xml)
        in_dollars = (f["filing_date"] or "") >= VALUE_DOLLAR_THRESHOLD
        rows = []
        for h in holdings:
            mv = h["value"]
            if mv is not None and not in_dollars:
                mv *= 1000  # pre-2023 values reported in thousands
            ticker = name_index.get(_normalize_name(h["name"]))
            rows.append({
                "fund_name": fund_name, "cik": cik, "ticker": ticker,
                "cusip": h["cusip"], "shares_held": h["shares"],
                "market_value": mv, "report_date": to_iso_date(f["report_date"]),
                "filing_date": to_iso_date(f["filing_date"]),
                "accession_number": f["accession"], "fetched_at": now,
            })
        if rows:
            # Upsert (not insert-ignore) so a re-parse corrects the shares/value
            # of an already-stored filing in place.
            stored += db.upsert(
                "institutional_holdings", rows,
                conflict=["cik", "cusip", "report_date", "accession_number"],
                update=["fund_name", "ticker", "shares_held", "market_value",
                        "filing_date", "fetched_at"])
    return stored


def compute_signals(db: Database) -> int:
    """Derive per-ticker quarter-over-quarter 13F signals."""
    report_dates = [r["report_date"] for r in db.query(
        "SELECT DISTINCT report_date FROM institutional_holdings "
        "WHERE ticker IS NOT NULL AND report_date IS NOT NULL ORDER BY report_date")]
    now = _now()
    written = 0
    for i, rd in enumerate(report_dates):
        prev_rd = report_dates[i - 1] if i > 0 else None
        cur = _holdings_by_ticker(db, rd)
        prev = _holdings_by_ticker(db, prev_rd) if prev_rd else {}
        rows = []
        for ticker, funds in cur.items():
            inc = dec = new = closed = 0
            net = 0.0
            prev_funds = prev.get(ticker, {})
            for fund, shares in funds.items():
                p = prev_funds.get(fund)
                if p is None:
                    new += 1
                elif shares > p:
                    inc += 1
                elif shares < p:
                    dec += 1
                net += shares - (p or 0)
            for fund, shares in prev_funds.items():
                if fund not in funds:
                    closed += 1
                    net -= shares
            rows.append({
                "ticker": ticker, "report_date": rd, "fund_count": len(funds),
                "position_increases": inc, "position_decreases": dec,
                "new_positions": new, "closed_positions": closed,
                "net_share_change": net, "computed_at": now,
            })
        if rows:
            written += db.upsert("institutional_signals", rows,
                                 conflict=["ticker", "report_date"])
    return written


def _holdings_by_ticker(db: Database, report_date: str | None) -> dict[str, dict]:
    if not report_date:
        return {}
    out: dict[str, dict] = {}
    for r in db.query(
            "SELECT ticker, fund_name, SUM(shares_held) s FROM institutional_holdings "
            "WHERE report_date = ? AND ticker IS NOT NULL GROUP BY ticker, fund_name",
            (report_date,)):
        out.setdefault(r["ticker"], {})[r["fund_name"]] = r["s"] or 0
    return out


def update_institutional(db: Database, funds: dict[str, str] | None = None,
                         quarters: int | None = None) -> dict:
    cfg = load_config()
    funds = funds or cfg.get("institutional", "funds", default={})
    quarters = quarters or int(cfg.get("institutional", "default_quarters", default=4))
    client = EdgarClient(cfg)
    name_index = build_name_index(db)

    stored = 0
    failed: list[str] = []
    for fund_name, cik in funds.items():
        try:
            n = process_fund(db, client, fund_name, str(cik).zfill(10),
                             quarters, name_index)
            stored += n
            log.info("13F %s: %d holdings stored", fund_name, n)
        except Exception as exc:  # noqa: BLE001
            log.warning("13F failed for %s: %s", fund_name, exc)
            failed.append(fund_name)
    signals = compute_signals(db)
    log.info("13F complete: %d holdings, %d signals", stored, signals)
    return {"holdings": stored, "signals": signals, "failed": failed}


# ---------------------------------------------------------------------------
# FMP whole-market ownership summary (preferred: deep, universe-wide)
# ---------------------------------------------------------------------------
_QUARTER_ENDS = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
# 13F-HR filings are due 45 days after quarter end. Before that deadline FMP's
# summary only aggregates the early filers (e.g. 244 of ~6,400 for MSFT
# 2026-Q2), so QoQ deltas like numberOf13FsharesChange are wildly wrong.
FILING_DEADLINE_DAYS = 45


def _quarter_grid(start: str, end: str) -> list[tuple[int, int, str]]:
    """(year, quarter, report_date) for quarter-ends spanning [start, end]."""
    sy, sq = int(start[:4]), (int(start[5:7]) - 1) // 3 + 1
    ey, eq = int(end[:4]), (int(end[5:7]) - 1) // 3 + 1
    out: list[tuple[int, int, str]] = []
    y, q = sy, sq
    while (y, q) <= (ey, eq):
        out.append((y, q, f"{y}-{_QUARTER_ENDS[q]}"))
        q, y = (1, y + 1) if q == 4 else (q + 1, y)
    return out


def _map_summary(ticker: str, report_date: str, r: dict, now: str) -> dict:
    return {
        "ticker": ticker, "report_date": report_date, "cik": r.get("cik"),
        "investors_holding": safe_int(r.get("investorsHolding")),
        "investors_holding_change": safe_int(r.get("investorsHoldingChange")),
        "number_of_13f_shares": safe_float(r.get("numberOf13Fshares")),
        "shares_change": safe_float(r.get("numberOf13FsharesChange")),
        "total_invested": safe_float(r.get("totalInvested")),
        "ownership_percent": safe_float(r.get("ownershipPercent")),
        "ownership_percent_change": safe_float(r.get("ownershipPercentChange")),
        "new_positions": safe_int(r.get("newPositions")),
        "increased_positions": safe_int(r.get("increasedPositions")),
        "reduced_positions": safe_int(r.get("reducedPositions")),
        "closed_positions": safe_int(r.get("closedPositions")),
        "put_call_ratio": safe_float(r.get("putCallRatio")),
        "source": "fmp", "fetched_at": now,
    }


def compute_signals_from_ownership(db: Database) -> int:
    """Populate ``institutional_signals`` from the FMP ownership summary.

    Maps whole-market breadth onto the columns the institutional factor reads,
    so the factor and its point-in-time reader work unchanged but now see
    thousands of filers per name instead of a curated dozen funds.
    """
    now = _now()
    rows = db.query(
        "SELECT ticker, report_date, investors_holding, increased_positions, "
        "reduced_positions, new_positions, closed_positions, shares_change "
        "FROM institutional_ownership_summary")
    out = [{
        "ticker": r["ticker"], "report_date": r["report_date"],
        "fund_count": r["investors_holding"],
        "position_increases": r["increased_positions"],
        "position_decreases": r["reduced_positions"],
        "new_positions": r["new_positions"],
        "closed_positions": r["closed_positions"],
        "net_share_change": r["shares_change"],
        "computed_at": now,
    } for r in rows]
    if not out:
        return 0
    return db.upsert("institutional_signals", out, conflict=["ticker", "report_date"])


def fmp_update_ownership(db: Database, tickers: list[str], start: str = "2019-01-01",
                        end: str | None = None, registry=None) -> dict:
    """Backfill per-symbol quarterly 13F ownership summaries from FMP Ultimate."""
    from .providers import ProviderRegistry
    cfg = load_config()
    registry = registry or ProviderRegistry(cfg)
    provider = registry.fundamentals()
    if provider is None or not hasattr(provider, "get_institutional_ownership"):
        log.warning("No FMP provider for institutional ownership; skipping")
        return {"rows": 0, "signals": 0, "failed": tickers}
    end = end or datetime.now(timezone.utc).date().isoformat()
    grid = _quarter_grid(start, end)
    # Only ingest quarters past their 13F filing deadline: a pre-deadline
    # summary covers just the early filers, and the `have` skip-set below would
    # freeze that partial vintage in place forever.
    complete = [g for g in grid if
                (date.fromisoformat(g[2]) + timedelta(days=FILING_DEADLINE_DAYS)
                 ).isoformat() <= end]
    if len(complete) < len(grid):
        skipped = [g[2] for g in grid if g not in complete]
        log.info("13F ownership: skipping incomplete quarter(s) %s "
                 "(filing deadline not passed)", ", ".join(skipped))
    grid = complete
    now = _now()
    stored = 0
    failed: list[str] = []
    # Skip (ticker, report_date) pairs already stored so deep re-runs only pay
    # for the missing quarters (e.g. 2016-2018 for current names).
    have = {(r["ticker"], r["report_date"]) for r in db.query(
        "SELECT ticker, report_date FROM institutional_ownership_summary")}
    for i, ticker in enumerate(tickers, 1):
        rows = []
        for (y, q, rd) in grid:
            if (ticker, rd) in have:
                continue
            try:
                r = provider.get_institutional_ownership(ticker, y, q)
            except Exception as exc:  # noqa: BLE001
                log.warning("13F ownership %s %dQ%d failed: %s", ticker, y, q, exc)
                continue
            if r:
                rows.append(_map_summary(ticker, rd, r, now))
        if rows:
            stored += db.upsert("institutional_ownership_summary", rows,
                                conflict=["ticker", "report_date"])
        else:
            failed.append(ticker)
        if i % 25 == 0 or i == len(tickers):
            log.info("13F ownership: %d/%d tickers (%d summary rows)",
                     i, len(tickers), stored)
    signals = compute_signals_from_ownership(db)
    log.info("13F ownership complete: %d summary rows, %d signals", stored, signals)
    return {"rows": stored, "signals": signals, "failed": failed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch 13F institutional holdings")
    parser.add_argument("--source", choices=["fmp", "edgar"], default="fmp",
                        help="fmp = whole-market ownership summary (deep, universe-"
                             "wide); edgar = curated per-fund holdings")
    parser.add_argument("--funds", nargs="*", help="EDGAR: subset of fund names")
    parser.add_argument("--quarters", type=int, help="EDGAR: number of recent quarters")
    parser.add_argument("--start", default="2019-01-01", help="FMP: backfill start date")
    parser.add_argument("--tickers", nargs="*", help="FMP: restrict to tickers")
    args = parser.parse_args()
    cfg = load_config()
    get_logger("institutional", log_file=cfg.log_file)
    with get_db() as db:
        if args.source == "fmp":
            tickers = [t.upper() for t in args.tickers] if args.tickers else db.universe_tickers()
            stats = fmp_update_ownership(db, tickers, start=args.start)
            print(f"Ownership rows  : {stats['rows']}")
            print(f"Signals computed: {stats['signals']}")
            print(f"Tickers w/o data: {len(stats['failed'])}")
        else:
            all_funds = cfg.get("institutional", "funds", default={})
            funds = ({f: all_funds[f] for f in args.funds if f in all_funds}
                     if args.funds else all_funds)
            stats = update_institutional(db, funds=funds, quarters=args.quarters)
            print(f"Holdings stored : {stats['holdings']}")
            print(f"Signals computed: {stats['signals']}")
            print(f"Failed funds    : {len(stats['failed'])}")


if __name__ == "__main__":
    main()
