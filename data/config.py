"""Configuration loader for Layer 1.

Loads ``config.yaml`` and ``.env`` once and exposes a lightweight,
attribute/dict accessor used by every module. Centralizing this keeps
provider keys and tunable parameters out of the module code.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Project root = parent of the `data/` package directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"


class Config:
    """Read-only view over config.yaml plus resolved paths and env keys."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw
        # Load .env into os.environ (does not override already-set vars).
        load_dotenv(ENV_PATH)

    # -- generic access -----------------------------------------------------
    def get(self, *keys: str, default: Any = None) -> Any:
        """Nested lookup: ``cfg.get("sec", "user_agent")``."""
        node: Any = self._raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def section(self, name: str) -> dict[str, Any]:
        return dict(self._raw.get(name, {}))

    # -- resolved filesystem paths -----------------------------------------
    @property
    def root(self) -> Path:
        return PROJECT_ROOT

    def path(self, *keys: str, default: str | None = None) -> Path:
        """Resolve a configured relative path against the project root."""
        value = self.get(*keys, default=default)
        if value is None:
            raise KeyError(f"No path configured for {keys}")
        p = Path(value)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def db_path(self) -> Path:
        return self.path("database", "path", default="cache/mahajan.db")

    @property
    def log_file(self) -> Path:
        return self.path("paths", "log_file", default="output/run.log")

    # -- API keys -----------------------------------------------------------
    @staticmethod
    def env(name: str, default: str | None = None) -> str | None:
        value = os.environ.get(name, default)
        # Treat empty strings as "not set" so blank .env lines do not enable a provider.
        return value if value else default

    def has_key(self, name: str) -> bool:
        return bool(self.env(name))

    def ensure_dirs(self) -> None:
        """Create cache/output directories so first run never fails on I/O."""
        for key in ("cache_dir", "output_dir"):
            rel = self.get("paths", key)
            if rel:
                (PROJECT_ROOT / rel).mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def load_config() -> Config:
    """Load (and cache) the project configuration."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.yaml not found at {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = Config(raw)
    cfg.ensure_dirs()
    return cfg


if __name__ == "__main__":
    c = load_config()
    print(f"Project root : {c.root}")
    print(f"Database     : {c.db_path}")
    print(f"Log file     : {c.log_file}")
    print("Provider keys present:")
    for key in ("POLYGON_API_KEY", "FMP_API_KEY", "FRED_API_KEY"):
        print(f"  {key:18s}: {'yes' if c.has_key(key) else 'no'}")
