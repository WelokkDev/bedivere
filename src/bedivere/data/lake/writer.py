"""The ONE write seam for lake partitions — validated, atomic, verified.

Every partition that reaches disk goes through `write_day`: value-level
invariants (strictly-ascending close stamps, finite prices, non-negative volume,
uint32 instrument ids) refused loudly and never repaired, footer provenance
stamped in, temp-then-rename, and a read-back check before the rename so a file
that cannot be re-read with the declared schema never becomes visible.

Whole-file replacement is the only write mode. A partition is always re-derivable
from an archived `.dbn.zst`, and appending would let two ingests of overlapping
ranges interleave rows into a file no single input can explain.

Writes are one vectorized `COPY TO` per partition; row-at-a-time inserts are the
one reliable way to make DuckDB slow.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import duckdb
import numpy as np
import pandas as pd  # pyright: ignore[reportMissingTypeStubs]

from bedivere.core.types import Timeframe
from bedivere.data.lake.layout import SeriesId, session_path
from bedivere.data.lake.read import partition_kv, read_partition
from bedivere.data.lake.schema import (
    AS_TRADED,
    BAR_SELECT,
    META_DATASET,
    META_PRICE_BASIS,
    META_SERIES,
    META_SESSION,
    META_SOURCE,
    META_SYMBOL,
    META_TIMEFRAME,
    BarBatch,
)

_UINT32_MAX: Final[int] = 2**32 - 1


class LakeWriteError(Exception):
    """The batch (or the file it produced) failed the seam's checks."""


_shared_con: duckdb.DuckDBPyConnection | None = None


def _con() -> duckdb.DuckDBPyConnection:
    global _shared_con
    if _shared_con is None:
        _shared_con = duckdb.connect(":memory:")
    return _shared_con


def _sql_str(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def partition_metadata(
    sid: SeriesId,
    timeframe: Timeframe,
    label: str,
    source: str,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Provenance for the Parquet footer.

    `extra` carries what the path cannot (a batch job id, a download hash) but
    cannot override the base identity, so a caller cannot relabel a file as a
    series it is not.
    """
    meta = dict(extra or {})
    meta.update(
        {
            META_DATASET: sid.dataset,
            META_SYMBOL: sid.symbol,
            META_SERIES: sid.series,
            META_TIMEFRAME: timeframe.value,
            META_SESSION: label,
            # Fixed, not a parameter: the adjusted view is derived at read
            # time and never reaches a footer.
            META_PRICE_BASIS: AS_TRADED,
            META_SOURCE: source,
        }
    )
    return meta


def _validate(batch: BarBatch, context: str) -> None:
    batch.validate_shape()
    if len(batch) == 0:
        raise LakeWriteError(
            f"{context}: refusing to write an EMPTY partition — a zero-bar file would "
            "make stored_labels count a day that holds nothing"
        )

    ts = np.asarray(batch.ts, dtype=np.int64)
    if len(batch) > 1:
        steps = np.diff(ts)
        if not bool((steps > 0).all()):
            at = int(np.argmax(steps <= 0))
            raise LakeWriteError(
                f"{context}: close stamps must be strictly ascending, but "
                f"ts[{at}]={ts[at]} is followed by ts[{at + 1}]={ts[at + 1]} — "
                "interleaved instruments or an unsorted batch"
            )
    if bool((ts <= 0).any()):
        raise LakeWriteError(f"{context}: non-positive close stamp in batch")

    for name, values in (
        ("open", batch.open),
        ("high", batch.high),
        ("low", batch.low),
        ("close", batch.close),
        ("volume", batch.volume),
    ):
        arr = np.asarray(values, dtype=np.float64)
        if not bool(np.isfinite(arr).all()):
            at = int(np.argmin(np.isfinite(arr)))
            raise LakeWriteError(f"{context}: non-finite {name} at ts={batch.ts[at]}")
    if bool((np.asarray(batch.volume, dtype=np.float64) < 0).any()):
        raise LakeWriteError(f"{context}: negative volume in batch")

    iid = np.asarray(batch.instrument_id, dtype=np.int64)
    if bool((iid < 0).any()) or bool((iid > _UINT32_MAX).any()):
        raise LakeWriteError(f"{context}: instrument_id outside uint32 range")


def _frame(batch: BarBatch) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": np.asarray(batch.ts, dtype=np.int64),
            "open": np.asarray(batch.open, dtype=np.float64),
            "high": np.asarray(batch.high, dtype=np.float64),
            "low": np.asarray(batch.low, dtype=np.float64),
            "close": np.asarray(batch.close, dtype=np.float64),
            "volume": np.asarray(batch.volume, dtype=np.float64),
            "instrument_id": np.asarray(batch.instrument_id, dtype=np.uint32),
            "synthetic": np.asarray(batch.synthetic, dtype=np.bool_),
        }
    )


def _copy_to(tmp: Path, batch: BarBatch, metadata: dict[str, str]) -> None:
    kv = ", ".join(f"{_sql_str(k)}: {_sql_str(v)}" for k, v in metadata.items())
    con = _con()
    con.register("bedivere_write_batch", _frame(batch))
    try:
        con.execute(
            f"COPY (SELECT {BAR_SELECT} FROM bedivere_write_batch) TO {_sql_str(tmp.as_posix())} "
            f"(FORMAT PARQUET, COMPRESSION 'zstd', KV_METADATA {{{kv}}})"
        )
    finally:
        con.unregister("bedivere_write_batch")


def _verify_written(tmp: Path, batch: BarBatch, source: str, context: str) -> None:
    """Read the temp file back before it becomes visible — `read_partition`
    re-pins the dtypes, so a write that changed a column's type is caught here
    rather than by the next backtest."""
    back = read_partition(tmp)
    if len(back) != len(batch) or back.ts[0] != batch.ts[0] or back.ts[-1] != batch.ts[-1]:
        raise LakeWriteError(
            f"{context}: read-back mismatch — wrote {len(batch)} bars "
            f"[{batch.ts[0]}..{batch.ts[-1]}], file holds {len(back)} "
            f"[{back.ts[0] if back.ts else '-'}..{back.ts[-1] if back.ts else '-'}]"
        )
    tag = partition_kv(tmp).get(META_SOURCE, "")
    if tag != source:
        raise LakeWriteError(
            f"{context}: footer {META_SOURCE} read back as {tag!r}, expected {source!r}"
        )


def write_day(
    root: Path,
    sid: SeriesId,
    timeframe: Timeframe,
    label: str,
    batch: BarBatch,
    source: str,
    extra_metadata: dict[str, str] | None = None,
) -> Path:
    """Write one session-day partition, replacing any previous copy.

    Temp-then-rename, because `stored_labels` counts any existing `bars.parquet`
    as a stored day. The `.tmp` suffix also keeps the in-progress file invisible
    to the `*.parquet` globs the SQL views use.
    """
    context = f"{sid} {timeframe.value} {label}"
    _validate(batch, context)
    path = session_path(root, sid, timeframe, label)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        _copy_to(tmp, batch, partition_metadata(sid, timeframe, label, source, extra_metadata))
        _verify_written(tmp, batch, source, context)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path
