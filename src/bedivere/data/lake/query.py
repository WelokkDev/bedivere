"""SQL over the lake, without writing a script.

DuckDB reads the Parquet in place, and the Hive layout makes the partition keys
(dataset, symbol, series, tf, session) columns for free:

    bedivere-data sql "
      SELECT session, count(*) bars, min(ts), max(ts)
      FROM bars WHERE symbol='NQ' AND tf='1s' AND session >= '2026-06-01'
      GROUP BY session ORDER BY session"

Two views are registered: `bars` (every bar plus the partition keys and a
readable `ts_utc`) and `condition` (the vendor's per-UTC-date quality feed).
Read-only by convention — writing goes through `lake.writer`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import duckdb

LAKE_ROOT_ENV: Final[str] = "BEDIVERE_LAKE_ROOT"

COVERAGE_SQL: Final[str] = """
SELECT dataset, symbol, series, tf,
       count(DISTINCT session) AS session_days,
       count(*)                AS bars,
       min(session)            AS first_session,
       max(session)            AS last_session
FROM bars
GROUP BY ALL
ORDER BY dataset, symbol, series, tf
"""


def lake_root(root: Path | None = None) -> Path:
    """Where the lake lives: an explicit path, then `$BEDIVERE_LAKE_ROOT`, then
    `./data/lake`.

    The default is relative to the working directory, not the installed package:
    a lake belongs to a research project, so two projects are two lakes.
    """
    if root is not None:
        return Path(root)
    env = os.environ.get(LAKE_ROOT_ENV)
    if env:
        return Path(env).resolve()
    return Path("data") / "lake"


def connect(root: Path | None = None) -> duckdb.DuckDBPyConnection:
    """An in-memory DuckDB with `bars` and `condition` views over the lake."""
    base = lake_root(root)
    conn = duckdb.connect(":memory:")

    bars_dir = base / "bars"
    if _has_parquet(bars_dir):
        glob = (bars_dir / "**" / "*.parquet").as_posix()
        conn.execute(
            f"""
            CREATE VIEW bars AS
            SELECT *, to_timestamp(ts) AS ts_utc
            FROM read_parquet('{glob}', hive_partitioning = true)
            """
        )
    else:
        # An empty lake must still answer queries, or every caller needs its
        # own "is there anything there yet" branch.
        conn.execute(
            "CREATE VIEW bars AS SELECT NULL::BIGINT ts, NULL::VARCHAR symbol WHERE false"
        )

    cond_dir = base / "_meta" / "condition"
    if _has_parquet(cond_dir):
        glob = (cond_dir / "**" / "*.parquet").as_posix()
        conn.execute(
            f"""
            CREATE VIEW condition AS
            SELECT * FROM read_parquet('{glob}', hive_partitioning = true)
            """
        )
    else:
        conn.execute(
            "CREATE VIEW condition AS "
            "SELECT NULL::VARCHAR date, NULL::VARCHAR condition WHERE false"
        )
    return conn


def _has_parquet(directory: Path) -> bool:
    return directory.is_dir() and next(directory.rglob("*.parquet"), None) is not None


def sql(query: str, root: Path | None = None) -> list[tuple[object, ...]]:
    with connect(root) as conn:
        return conn.execute(query).fetchall()
