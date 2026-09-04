"""Shared utilities: logging, rate limiting, retries, and safe coercion.

Kept dependency-light so every module can import it without pulling in the
heavier data libraries.
"""
from __future__ import annotations

import logging
import math
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, TypeVar

T = TypeVar("T")

_LOG_CONFIGURED = False


def get_logger(name: str = "mahajan", log_file: Path | str | None = None,
               level: int = logging.INFO) -> logging.Logger:
    """Return the shared project logger, configuring file+console once."""
    global _LOG_CONFIGURED
    logger = logging.getLogger("mahajan")
    if not _LOG_CONFIGURED:
        logger.setLevel(level)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        logger.addHandler(console)
        if log_file is not None:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(path, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        logger.propagate = False
        _LOG_CONFIGURED = True
    # Always return a namespaced child so messages show their origin module.
    return logger.getChild(name) if name and name != "mahajan" else logger


class RateLimiter:
    """Simple blocking rate limiter (token-bucket-ish, min interval based).

    Used for SEC EDGAR where we must stay under ~10 requests/second.
    """

    def __init__(self, max_per_second: float) -> None:
        self.min_interval = 1.0 / max_per_second if max_per_second > 0 else 0.0
        self._last = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.monotonic()


def retry(fn: Callable[[], T], *, attempts: int = 3, base_delay: float = 1.0,
          logger: logging.Logger | None = None, what: str = "operation") -> T | None:
    """Call ``fn`` with exponential backoff. Returns None if all attempts fail."""
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we deliberately swallow & retry
            last_exc = exc
            if logger:
                logger.warning("%s failed (attempt %d/%d): %s", what, i + 1, attempts, exc)
            if i < attempts - 1:
                time.sleep(base_delay * (2 ** i))
    if logger and last_exc:
        logger.error("%s failed permanently: %s", what, last_exc)
    return None


# -- safe coercion helpers --------------------------------------------------

def safe_float(value: Any) -> float | None:
    """Coerce to float, mapping NaN/inf/None/empty to None."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def safe_int(value: Any) -> int | None:
    f = safe_float(value)
    return int(f) if f is not None else None


def to_iso_date(value: Any) -> str | None:
    """Normalize many date representations to an ISO ``YYYY-MM-DD`` string."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        # Accept already-ISO and common variants; let pandas-free parsing handle it.
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(value[:len(fmt) + 2], fmt).date().isoformat()
            except ValueError:
                continue
        # Last resort: take the leading date-looking token.
        return value[:10] if len(value) >= 10 else None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    # pandas Timestamp / numpy datetime64 expose isoformat or str.
    try:
        return value.date().isoformat()  # type: ignore[attr-defined]
    except AttributeError:
        try:
            return str(value)[:10]
        except Exception:  # noqa: BLE001
            return None


def divide(numerator: Any, denominator: Any) -> float | None:
    """Safe division returning None on zero/invalid inputs."""
    n = safe_float(numerator)
    d = safe_float(denominator)
    if n is None or d is None or d == 0:
        return None
    return n / d


def chunked(seq: Sequence[T], size: int) -> Iterable[list[T]]:
    for i in range(0, len(seq), size):
        yield list(seq[i:i + size])


def pct_change(current: Any, previous: Any) -> float | None:
    c = safe_float(current)
    p = safe_float(previous)
    if c is None or p is None or p == 0:
        return None
    return (c - p) / abs(p)
