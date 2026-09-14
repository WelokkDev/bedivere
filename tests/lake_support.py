"""Whether the `lake` extra is installed, so a base-install clone sees green.

Two mechanisms: `pytest.importorskip` at the TOP of a module that imports lake
code at module scope (it must run before those imports, or collection fails
rather than skipping), and `requires_lake` for one test inside a module that is
otherwise dependency-free.
"""

from __future__ import annotations

import importlib.util

import pytest

LAKE_MODULES = ("duckdb", "numpy", "pandas")

HAS_LAKE = all(importlib.util.find_spec(name) is not None for name in LAKE_MODULES)

SKIP_REASON = "the lake extra is not installed (pip install 'bedivere[lake]')"

requires_lake = pytest.mark.skipif(not HAS_LAKE, reason=SKIP_REASON)
