"""SQLite cache for Layer 3 analyzer responses.

A run can analyze 25 long + 25 short candidates across half a dozen analyzers
- 300+ Claude calls. Most artifacts (a transcript, a 10-K risk section, an
8-quarter financials table) do not change between runs. Caching by artifact
hash means we pay for them once and then re-use the JSON forever, subject
to:

  * TTL          - default 30 days, defends against silent prompt drift
  * artifact hash - if the underlying text changes, the entry is invalid
  * prompt version - bumped in :mod:`analysis` whenever wording changes
  * model        - a model swap re-runs everything

The cache file is independent of the Layer 1 warehouse so analytics work
cannot corrupt research data.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from data.config import load_config
from data.utils import get_logger

log = get_logger("analysis.cache")

SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis_cache (
    analyzer       TEXT NOT NULL,
    ticker         TEXT NOT NULL,
    artifact_id    TEXT NOT NULL,
    artifact_hash  TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    model          TEXT NOT NULL,
    response_json  TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    expires_at     TEXT NOT NULL,
    PRIMARY KEY (analyzer, ticker, artifact_id, prompt_version, model)
);
CREATE INDEX IF NOT EXISTS idx_cache_expires ON analysis_cache(expires_at);
"""


def hash_artifact(payload: str | bytes | dict | list) -> str:
    """Stable SHA-256 of the input. Dicts/lists are JSON-serialized first."""
    if isinstance(payload, (dict, list)):
        payload = json.dumps(payload, sort_keys=True, default=str)
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CacheKey:
    analyzer: str
    ticker: str
    artifact_id: str
    artifact_hash: str
    prompt_version: str
    model: str


class AnalysisCache:
    """SQLite-backed key/value store for analyzer outputs."""

    def __init__(self, path: Path | str | None = None, ttl_days: int = 30) -> None:
        cfg = load_config()
        self.path = Path(path) if path else (cfg.root / "cache" / "analysis_cache.db")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(days=ttl_days)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        with closing(self._conn.cursor()) as cur:
            cur.executescript(SCHEMA)
        self._conn.commit()

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self._conn.commit()
        self._conn.close()

    def __enter__(self) -> "AnalysisCache":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- core operations ---------------------------------------------------
    def get(self, key: CacheKey) -> dict[str, Any] | None:
        """Return the cached response or ``None`` if absent/expired/stale.

        Invalidates on:
          * row missing or expired (TTL elapsed)
          * stored ``artifact_hash`` differs from caller's hash
        """
        row = self._conn.execute(
            "SELECT artifact_hash, response_json, expires_at FROM analysis_cache "
            "WHERE analyzer=? AND ticker=? AND artifact_id=? "
            "AND prompt_version=? AND model=?",
            (key.analyzer, key.ticker, key.artifact_id,
             key.prompt_version, key.model),
        ).fetchone()
        if row is None:
            return None
        if row["artifact_hash"] != key.artifact_hash:
            log.debug("Cache MISS (hash changed): %s / %s / %s",
                      key.analyzer, key.ticker, key.artifact_id)
            return None
        expires = datetime.fromisoformat(row["expires_at"])
        if datetime.now(timezone.utc) > expires:
            log.debug("Cache MISS (expired): %s / %s / %s",
                      key.analyzer, key.ticker, key.artifact_id)
            return None
        try:
            return json.loads(row["response_json"])
        except json.JSONDecodeError:
            log.warning("Corrupt cache row for %s/%s/%s; ignoring",
                        key.analyzer, key.ticker, key.artifact_id)
            return None

    def put(self, key: CacheKey, response: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc)
        expires = now + self.ttl
        self._conn.execute(
            "INSERT OR REPLACE INTO analysis_cache "
            "(analyzer, ticker, artifact_id, artifact_hash, prompt_version, "
            " model, response_json, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (key.analyzer, key.ticker, key.artifact_id, key.artifact_hash,
             key.prompt_version, key.model,
             json.dumps(response, default=str),
             now.isoformat(timespec="seconds"),
             expires.isoformat(timespec="seconds")),
        )
        self._conn.commit()

    def purge_expired(self) -> int:
        """Drop rows whose TTL has lapsed; returns count removed."""
        cur = self._conn.execute(
            "DELETE FROM analysis_cache WHERE expires_at < ?",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"),),
        )
        self._conn.commit()
        return cur.rowcount or 0

    def stats(self) -> dict[str, int]:
        n_total = self._conn.execute(
            "SELECT COUNT(*) FROM analysis_cache").fetchone()[0]
        n_expired = self._conn.execute(
            "SELECT COUNT(*) FROM analysis_cache WHERE expires_at < ?",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"),),
        ).fetchone()[0]
        return {"total": n_total, "expired": n_expired}
