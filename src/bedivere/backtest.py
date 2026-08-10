"""`python -m bedivere.backtest` — see bedivere.cli.backtest.

A shim so the command reads the way it should. The composition root is
`bedivere.run.backtest.run_backtest`; the argument parsing and archiving are
`bedivere.cli.backtest`.
"""

from __future__ import annotations

import sys

from bedivere.cli.backtest import main

if __name__ == "__main__":
    sys.exit(main())
