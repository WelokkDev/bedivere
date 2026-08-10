"""`python -m bedivere.live` — see bedivere.cli.live.

A shim so the command reads the way it should. The composition root is
`bedivere.run.live.run_live`; the argument parsing, supervision and archiving
are `bedivere.cli.live`.
"""

from __future__ import annotations

import sys

from bedivere.cli.live import main

if __name__ == "__main__":
    sys.exit(main())
