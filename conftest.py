"""Make the src-layout package importable without installation, so a fresh
clone can run `pytest` (and the example) before any `pip install -e .`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
