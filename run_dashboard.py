"""Launch the Layer 7 Streamlit dashboard.

This is a one-line wrapper so the dashboard starts the same way as the
other Layer entry points (``python run_*.py``). Streamlit handles all
the routing, hot reload, and multi-page navigation; we just hand it
``dashboard/app.py`` and forward CLI args.

Examples::

    python run_dashboard.py
    python run_dashboard.py --server.port 8502 --server.headless true
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        print(
            "streamlit is not installed. Run `pip install streamlit plotly` "
            "and retry.",
            file=sys.stderr,
        )
        return 2

    app = Path(__file__).parent / "dashboard" / "app.py"
    sys.argv = ["streamlit", "run", str(app), *sys.argv[1:]]
    return stcli.main()


if __name__ == "__main__":
    sys.exit(main())
