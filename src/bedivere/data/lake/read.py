"""The read seam — where the lake becomes a `CandleSource`'s answer.

Same contract as every other source on purpose: `(start_unix, end_unix]`,
close-stamped, ascending, `list[Candle]` — so moving a backtest onto the lake
changes the spec's `data` block and nothing in the engine.

Reading needs no session geometry: a bar's partition label is either its own UTC
date or the next one, so widening the label window a day each side and then
filtering on `ts` is exact. `read_table` additionally exposes the provenance
columns `Candle` has no room for, `instrument_id` above all.

Two disciplines every query here shares, and they are load-bearing:

  * `hive_partitioning=false` with the columns projected EXPLICITLY. With hive
    detection on, `SELECT *` picks up dataset/symbol/series/tf/session as
    columns, and a rewrite would bake the partition keys into the file.
  * `DESCRIBE` before any data flows, so a file whose dtypes have drifted
    refuses instead of casting silently.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final

import duckdb

from bedivere.core.types import Candle, Timeframe
from bedivere.data.lake.layout import SeriesId, session_path, stored_labels
from bedivere.data.lake.schema import (
    BAR_SELECT,
    META_SOURCE_END_NS,
    META_SOURCE_START_NS,
    BarBatch,
    check_described_schema,
)


class LakeMissingError(Exception):
    """No partitions on disk for the requested (series, timeframe)."""


# `datetime.fromtimestamp` raises outside the year 1..9999 range, and callers
# legitimately pass an open-ended sentinel to mean "everything this series
# holds". No real session label sorts outside these bounds, so clamping keeps
# that spelling working and still selects every stored day.
_MIN_UNIX: Final[int] = -62_135_596_800  # 0001-01-01T00:00:00Z
_MAX_UNIX: Final[int] = 253_402_300_799  # 9999-12-31T23:59:59Z


def _utc_date(unix_sec: int) -> date:
    return datetime.fromtimestamp(min(max(unix_sec, _MIN_UNIX), _MAX_UNIX), tz=UTC).date()


_shared_con: duckdb.DuckDBPyConnection | None = None


def _con() -> duckdb.DuckDBPyConnection:
    """One in-memory connection per process. Nothing here holds state between
    calls, and per-call connects would only add latency to the per-window reads
    a sparse replay makes."""
    global _shared_con
    if _shared_con is None:
        _shared_con = duckdb.connect(":memory:")
    return _shared_con


def _sql_path(path: Path) -> str:
    return "'" + path.as_posix().replace("'", "''") + "'"


def _sql_paths(paths: list[Path]) -> str:
    return "[" + ", ".join(_sql_path(p) for p in paths) + "]"


def _checked_bar_relation(paths: list[Path]) -> str:
    """The FROM-clause for `paths`, after pinning their stored dtypes."""
    rel = f"read_parquet({_sql_paths(paths)}, hive_partitioning=false)"
    described_raw = _con().execute(f"DESCRIBE SELECT {BAR_SELECT} FROM {rel}").fetchall()
    described = [(str(row[0]), str(row[1])) for row in described_raw]
    context = (
        paths[0].as_posix()
        if len(paths) == 1
        else f"{len(paths)} files under {paths[0].parent.parent}"
    )
    check_described_schema(described, context=context)
    return rel


def candidate_labels(
    root: Path, sid: SeriesId, timeframe: Timeframe, start_unix: int, end_unix: int
) -> list[str]:
    """Stored session-day labels that could hold bars in `(start, end]`.

    A day either side of the UTC span, because a session label is the date its
    session CLOSES on and its bars begin the previous evening.
    """
    lo = _utc_date(start_unix - 86_400).isoformat()
    hi = _utc_date(end_unix + 86_400).isoformat()
    return [label for label in stored_labels(root, sid, timeframe) if lo <= label <= hi]


def latest_stored_ts(root: Path, sid: SeriesId, timeframe: Timeframe) -> int | None:
    """Close-stamp of the newest bar on disk, or None when nothing is stored.

    `max(ts)` over the last partition only, which DuckDB answers from the file's
    footer statistics — and unlike reading those statistics directly, it stays
    correct for a file whose statistics were never written.
    """
    labels = stored_labels(root, sid, timeframe)
    if not labels:
        return None
    path = session_path(root, sid, timeframe, labels[-1])
    row = _con().execute(
        f"SELECT max(ts) FROM read_parquet({_sql_path(path)}, hive_partitioning=false)"
    ).fetchone()
    if row is None or row[0] is None:
        return None
    return int(row[0])


def read_table(
    root: Path,
    sid: SeriesId,
    timeframe: Timeframe,
    start_unix: int,
    end_unix: int,
) -> BarBatch:
    """Bars in `(start_unix, end_unix]` with every stored column, ascending.

    Raises when the (series, timeframe) has no partitions AT ALL, because that
    is nearly always a wrong series or a wrong lake root and `[]` would hide it
    behind a backtest that takes no trades. A gap INSIDE a series that does
    exist is legitimate, and comes back as an empty batch.
    """
    labels = stored_labels(root, sid, timeframe)
    if not labels:
        raise LakeMissingError(
            f"no lake partitions for {sid} {timeframe.value} under {root} — "
            "wrong series, wrong timeframe, or nothing ingested yet"
        )

    paths = [
        path
        for label in candidate_labels(root, sid, timeframe, start_unix, end_unix)
        if (path := session_path(root, sid, timeframe, label)).is_file()
    ]
    if not paths:
        return BarBatch()

    rel = _checked_bar_relation(paths)
    rows = _con().execute(
        f"SELECT {BAR_SELECT} FROM {rel} WHERE ts > ? AND ts <= ? ORDER BY ts",
        [start_unix, end_unix],
    ).fetchall()
    return BarBatch.from_rows(rows)  # pyright: ignore[reportArgumentType]


def read_partition(path: Path) -> BarBatch:
    rel = _checked_bar_relation([path])
    rows = _con().execute(f"SELECT {BAR_SELECT} FROM {rel} ORDER BY ts").fetchall()
    return BarBatch.from_rows(rows)  # pyright: ignore[reportArgumentType]


def partition_kv(path: Path) -> dict[str, str]:
    """The Parquet footer's key-value metadata, utf-8 decoded — where the
    lineage `write_day` stamped into the partition lives."""
    rows = _con().execute(
        f"SELECT decode(key), decode(value) FROM parquet_kv_metadata({_sql_path(path)})"
    ).fetchall()
    return {str(k): str(v) for k, v in rows}


def source_range(path: Path) -> tuple[int, int] | None:
    """The `[start, end)` Unix-nanosecond range the partition's source declared
    it covered, or None when its footer records none."""
    meta = partition_kv(path)
    try:
        start, end = int(meta[META_SOURCE_START_NS]), int(meta[META_SOURCE_END_NS])
    except (KeyError, ValueError):
        return None
    return (start, end) if 0 <= start < end else None


def partition_row_count(path: Path) -> int:
    """Row count from the footer — no bar data is read."""
    row = _con().execute(
        f"SELECT num_rows FROM parquet_file_metadata({_sql_path(path)})"
    ).fetchone()
    if row is None:
        raise LakeMissingError(f"no parquet footer at {path}")
    return int(row[0])


def read_bars(
    root: Path,
    sid: SeriesId,
    timeframe: Timeframe,
    start_unix: int,
    end_unix: int,
    *,
    drop_synthetic: bool = False,
) -> list[Candle]:
    """Bars in `(start_unix, end_unix]` as engine `Candle`s, ascending.

    Nothing writes synthetic bars today; `drop_synthetic` lets a consumer that
    must not see manufactured prices say so rather than trust that.
    """
    return read_table(root, sid, timeframe, start_unix, end_unix).candles(
        drop_synthetic=drop_synthetic
    )


def roll_boundaries(table: BarBatch) -> list[int]:
    """Timestamps at which `instrument_id` changes.

    Vendor continuous series are unadjusted, so price gaps at a roll: a backtest
    that trades one of these stamps is trading an artefact.
    """
    ids = table.instrument_id
    ts = table.ts
    return [ts[i] for i in range(1, len(ids)) if ids[i] != ids[i - 1]]
