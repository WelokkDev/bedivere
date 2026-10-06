"""SQL over the lake, without writing a script.

DuckDB reads the Parquet in place, and the Hive layout makes the partition keys
(dataset, symbol, series, tf, session) columns for free:

    bedivere-data sql "
      SELECT session, count(*) bars, min(ts), max(ts)
      FROM bars WHERE symbol='NQ' AND tf='1s' AND session >= '2026-06-01'
      GROUP BY session ORDER BY session"

Three views are registered: `bars` (every bar plus the partition keys and a
readable `ts_utc`), `volume_bars` (the event-bar datasets, with theirs) and
`condition` (the vendor's per-UTC-date quality feed). Each exists, empty, when
the lake holds nothing for it. Read-only by convention — writing goes through
`lake.writer`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import duckdb

from bedivere.data.lake.schema import BAR_COLUMNS

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
    """An in-memory DuckDB with `bars`, `volume_bars` and `condition` views
    over the lake."""
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
        # own "is there anything there yet" branch. Same columns as the stored view.
        columns = ", ".join(f'NULL::{c.duck_type} AS "{c.name}"' for c in BAR_COLUMNS)
        keys = _partition_keys("dataset", "series", "session", "symbol", "tf")
        conn.execute(
            f'CREATE VIEW bars AS SELECT {columns}, {keys}, NULL::TIMESTAMPTZ AS "ts_utc" WHERE false'
        )

    from bedivere.data.lake.volume_store import COLUMNS

    event_dir = base / "event_bars"
    if _has_parquet(event_dir):
        glob = (event_dir / "**" / "*.parquet").as_posix().replace("'", "''")
        conn.execute(
            "CREATE VIEW volume_bars AS SELECT * "
            f"FROM read_parquet('{glob}', hive_partitioning=true)"
        )
    else:
        # Quoted and with AS: the parser rejects `close` and `session` as bare aliases.
        columns = ", ".join(f'NULL::{kind} AS "{name}"' for name, kind in COLUMNS.items())
        keys = _partition_keys("dataset", "definition", "series", "session", "symbol")
        conn.execute(f"CREATE VIEW volume_bars AS SELECT {columns}, {keys} WHERE false")

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


def _partition_keys(*names: str) -> str:
    """Partition-key columns for an empty stand-in view, as hive partitioning
    returns them: in name order, with `session` a DATE."""
    return ", ".join(
        f'NULL::{"DATE" if name == "session" else "VARCHAR"} AS "{name}"' for name in names
    )


def _has_parquet(directory: Path) -> bool:
    return directory.is_dir() and next(directory.rglob("*.parquet"), None) is not None


def sql(query: str, root: Path | None = None) -> list[tuple[object, ...]]:
    with connect(root) as conn:
        return conn.execute(query).fetchall()
