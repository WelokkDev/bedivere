"""Versioned event-bar Parquet, separate from the engine's time-bar schema."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import duckdb
import numpy as np
import pandas as pd  # pyright: ignore[reportMissingTypeStubs]

from bedivere.core.session_days import SessionDay, SessionDays
from bedivere.data.lake.layout import SeriesId, session_path
from bedivere.data.lake.read import partition_kv
from bedivere.data.lake.schema import AS_TRADED, BarSchemaError
from bedivere.data.lake.volume import NS, VolumeBar, VolumeSpec

# Explicit casts keep the all-null approximate VWAP column a DOUBLE as well.
COLUMNS = {
    "bar_id": "BIGINT",
    "start_ns": "BIGINT",
    "end_ns": "BIGINT",
    "available_ns": "BIGINT",
    "open": "DOUBLE",
    "high": "DOUBLE",
    "low": "DOUBLE",
    "close": "DOUBLE",
    "volume": "BIGINT",
    "instrument_id": "UINTEGER",
    "input_count": "BIGINT",
    "vwap": "DOUBLE",
    "is_partial": "BOOLEAN",
}
SELECT = ", ".join(COLUMNS)

SAME_TRADE_SECONDS: Final[str] = "same-trades-1s:"
"""Prefix of a `trades-1s` partition's recorded source, and of no other: it
tells rebuilt seconds from stored ones."""


def _sql_str(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def content_hash(bars: list[VolumeBar]) -> str:
    records = [asdict(bar) for bar in bars]
    for record in records:
        for name, kind in COLUMNS.items():
            if kind == "DOUBLE" and record[name] is not None:
                record[name] = float(record[name])
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def volume_path(
    root: Path,
    sid: SeriesId,
    spec: VolumeSpec,
    days: SessionDays,
    label: str,
) -> Path:
    # Reuse the validated session label and series identity path components.
    from bedivere.core.types import Timeframe

    time_path = session_path(root, sid, Timeframe.S1, label)
    return (
        root
        / "event_bars"
        / f"dataset={sid.dataset}"
        / f"symbol={sid.symbol}"
        / f"series={sid.series}"
        / f"definition={spec.key(days)}"
        / time_path.parent.name
        / "bars.parquet"
    )


def validate_bars(bars: list[VolumeBar], spec: VolumeSpec, day: SessionDay) -> None:
    if not bars:
        raise ValueError("refusing to write an empty event-bar partition")
    last_end = day.start_unix * NS
    last_available = last_end
    nearest = spec.boundary == "nearest_second"
    for i, bar in enumerate(bars):
        for name in ("bar_id", "start_ns", "end_ns", "available_ns", "volume", "input_count"):
            value = getattr(bar, name)
            if type(value) is not int or not 0 <= value < 2**63:
                raise ValueError(f"{name} must be a non-negative int64 integer")
        if type(bar.is_partial) is not bool or type(bar.instrument_id) is not int:
            raise ValueError("invalid partial flag or instrument type")
        if bar.bar_id != i or bar.instrument_id != bars[0].instrument_id:
            raise ValueError("bar IDs must be consecutive and the instrument constant")
        if not 0 < bar.instrument_id < 2**32 or bar.input_count <= 0:
            raise ValueError("invalid instrument or input count")
        if not last_end <= bar.start_ns <= bar.end_ns <= bar.available_ns <= day.end_unix * NS:
            raise ValueError("invalid event-bar timestamp order or session bounds")
        if bar.available_ns < last_available:
            raise ValueError("event-bar availability must be non-decreasing")
        last_available = bar.available_ns
        if nearest and any(ts % NS for ts in (bar.start_ns, bar.end_ns, bar.available_ns)):
            raise ValueError("nearest_second boundaries and availability must be whole seconds")
        if spec.source == "trades" and bar.end_ns >= day.end_unix * NS:
            raise ValueError("trade bars must end before session close")
        last_end = bar.end_ns
        if not 0 < bar.volume < 2**63:
            raise ValueError("bar volume must fit a positive int64")
        if bar.is_partial:
            if i != len(bars) - 1 or (not nearest and bar.volume >= spec.threshold):
                raise ValueError("only the last bar can be an undersized remainder")
            if bar.available_ns != day.end_unix * NS:
                raise ValueError("a remainder is only available at session close")
        elif not nearest and (bar.volume < spec.threshold or bar.available_ns != bar.end_ns):
            raise ValueError("full bar has not reached its threshold or has wrong availability")
        elif spec.boundary == "split_trade" and bar.volume != spec.threshold:
            raise ValueError("split_trade full bars must equal the threshold")
        if not all(math.isfinite(p) for p in (bar.open, bar.high, bar.low, bar.close)):
            raise ValueError("non-finite OHLC price")
        if not bar.low <= min(bar.open, bar.close) <= max(bar.open, bar.close) <= bar.high:
            raise ValueError("inconsistent OHLC prices")
        if spec.source == "trades":
            if bar.vwap is None or not math.isfinite(bar.vwap):
                raise ValueError("trade bars require a finite VWAP")
        elif bar.vwap is not None:
            raise ValueError("OHLCV-derived bars cannot claim exact VWAP")


def read_volume_partition(path: Path) -> list[VolumeBar]:
    with duckdb.connect(":memory:") as con:
        relation = f"read_parquet({_sql_str(path.as_posix())}, hive_partitioning=false)"
        described = con.execute(f"DESCRIBE SELECT {SELECT} FROM {relation}").fetchall()
        if [(row[0], row[1]) for row in described] != list(COLUMNS.items()):
            raise BarSchemaError(f"{path}: event-bar schema does not match")
        rows: list[Any] = con.execute(f"SELECT {SELECT} FROM {relation} ORDER BY bar_id").fetchall()
    for row in rows:
        if any(value is None for name, value in zip(COLUMNS, row, strict=True) if name != "vwap"):
            raise BarSchemaError(f"{path}: NULL in required event-bar column")
    return [VolumeBar(*row) for row in rows]


def _rebuilt_seconds(source: str) -> bool:
    return source.startswith(SAME_TRADE_SECONDS)


def checked_volume_partition(
    root: Path,
    sid: SeriesId,
    spec: VolumeSpec,
    days: SessionDays,
    day: SessionDay,
) -> list[VolumeBar]:
    """Verify identity, session geometry, schema, values, and content integrity."""
    return checked_volume_day(root, sid, spec, days, day)[0]


def checked_volume_day(
    root: Path,
    sid: SeriesId,
    spec: VolumeSpec,
    days: SessionDays,
    day: SessionDay,
) -> tuple[list[VolumeBar], str]:
    """`checked_volume_partition`, plus the source the footer records.

    That source must mark rebuilt seconds exactly when the definition is
    `trades-1s`: older builds stored both kinds at the `ohlcv-1s` path.
    """
    path = volume_path(root, sid, spec, days, day.label)
    meta = partition_kv(path)
    expected = {
        "bedivere.dataset": sid.dataset,
        "bedivere.symbol": sid.symbol,
        "bedivere.series": sid.series,
        "bedivere.session": day.label,
        "bedivere.bar_definition": spec.definition(days),
        "bedivere.session_start_ns": str(day.start_unix * NS),
        "bedivere.session_end_ns": str(day.end_unix * NS),
        "bedivere.price_basis": AS_TRADED,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise BarSchemaError(f"{path}: inconsistent {key}")
    source = meta.get("bedivere.source", "")
    if _rebuilt_seconds(source) != (spec.source == "trades-1s"):
        if spec.source == "trades-1s":
            raise BarSchemaError(f"{path}: inconsistent bedivere.source")
        raise BarSchemaError(
            f"{path}: built from one-second bars rebuilt from trades (--compare-1s), but "
            f"stored under source={spec.source}; such bars are now stored under "
            "source=trades-1s. Rebuild the comparison, then delete this partition"
        )
    bars = read_volume_partition(path)
    validate_bars(bars, spec, day)
    if meta.get("bedivere.content_sha256") != content_hash(bars):
        raise BarSchemaError(f"{path}: missing or mismatched content checksum; rebuild partition")
    return bars, source


def _metadata(
    sid: SeriesId,
    spec: VolumeSpec,
    days: SessionDays,
    day: SessionDay,
    bars: list[VolumeBar],
    source: str,
) -> dict[str, str]:
    if _rebuilt_seconds(source) != (spec.source == "trades-1s"):
        raise ValueError(
            f"a source=trades-1s partition records a source starting {SAME_TRADE_SECONDS!r}, "
            f"and no other does; got {source!r} under source={spec.source}"
        )
    return {
        "bedivere.dataset": sid.dataset,
        "bedivere.symbol": sid.symbol,
        "bedivere.series": sid.series,
        "bedivere.session": day.label,
        "bedivere.price_basis": AS_TRADED,
        "bedivere.source": source,
        "bedivere.bar_definition": spec.definition(days),
        "bedivere.session_start_ns": str(day.start_unix * NS),
        "bedivere.session_end_ns": str(day.end_unix * NS),
        "bedivere.content_sha256": content_hash(bars),
    }


def _check_replaceable(path: Path, metadata: Mapping[str, str]) -> None:
    if path.exists():
        previous = partition_kv(path)
        for key in (
            "bedivere.bar_definition",
            "bedivere.session_start_ns",
            "bedivere.session_end_ns",
        ):
            if previous.get(key) != metadata[key]:
                raise ValueError(
                    f"{path}: existing partition has different session geometry or policy"
                )
        # A partition from before `trades-1s` existed, when both kinds shared this path.
        if _rebuilt_seconds(previous.get("bedivere.source", "")) and not _rebuilt_seconds(
            metadata["bedivere.source"]
        ):
            raise ValueError(
                f"{path}: existing partition was built from one-second bars rebuilt from "
                "trades (--compare-1s), which are now stored under source=trades-1s; a build "
                "from another source does not replace it. Rebuild the comparison, then delete "
                "this partition"
            )


def _encode(bars: list[VolumeBar], metadata: Mapping[str, str], directory: Path) -> Path:
    """A read-back-verified Parquet file in `directory`, owned by the caller."""
    frame = pd.DataFrame([asdict(bar) for bar in bars])
    # Preserve integers without floating-point round trips (timestamps exceed 2**53).
    for name, sql_type in COLUMNS.items():
        dtype = {
            "BIGINT": np.int64,
            "UINTEGER": np.uint32,
            "BOOLEAN": np.bool_,
            "DOUBLE": np.float64,
        }[sql_type]
        frame[name] = np.asarray([getattr(b, name) for b in bars], dtype=dtype)
    fd, name = tempfile.mkstemp(prefix="bars-", suffix=".tmp", dir=directory)
    os.close(fd)
    tmp = Path(name)
    kv = ", ".join(f"{_sql_str(k)}: {_sql_str(v)}" for k, v in metadata.items())
    projection = ", ".join(f"{name}::{kind} AS {name}" for name, kind in COLUMNS.items())
    try:
        with duckdb.connect(":memory:") as con:
            con.register("volume_batch", frame)
            con.execute(
                f"COPY (SELECT {projection} FROM volume_batch) TO {_sql_str(tmp.as_posix())} "
                f"(FORMAT PARQUET, COMPRESSION 'zstd', KV_METADATA {{{kv}}})"
            )
        if read_volume_partition(tmp) != bars or partition_kv(tmp) != metadata:
            raise ValueError("event-bar read-back verification failed")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return tmp


def write_volume_day(
    root: Path,
    sid: SeriesId,
    spec: VolumeSpec,
    days: SessionDays,
    day: SessionDay,
    bars: list[VolumeBar],
    *,
    source: str,
) -> Path:
    validate_bars(bars, spec, day)
    path = volume_path(root, sid, spec, days, day.label)
    metadata = _metadata(sid, spec, days, day, bars, source)
    _check_replaceable(path, metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _encode(bars, metadata, path.parent)
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


@dataclass(frozen=True, slots=True)
class StagedPartition:
    """A validated, read-back-verified partition not yet published."""

    staged: Path
    """A temporary file inside the caller's staging directory."""

    path: Path
    """Where `publish_staged` will place it."""

    metadata: Mapping[str, str]


def stage_volume_day(
    root: Path,
    sid: SeriesId,
    spec: VolumeSpec,
    days: SessionDays,
    day: SessionDay,
    bars: list[VolumeBar],
    *,
    source: str,
    staging: Path,
) -> StagedPartition:
    """`write_volume_day` without the final rename. `staging` must be on the
    lake's filesystem, and is the caller's to remove."""
    validate_bars(bars, spec, day)
    path = volume_path(root, sid, spec, days, day.label)
    metadata = _metadata(sid, spec, days, day, bars, source)
    _check_replaceable(path, metadata)
    return StagedPartition(_encode(bars, metadata, staging), path, metadata)


def publish_staged(staged: StagedPartition) -> Path:
    """Atomically replace one partition with its staged file."""
    _check_replaceable(staged.path, staged.metadata)
    staged.path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged.staged, staged.path)
    return staged.path


def volume_inventory(root: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for directory in sorted((root / "event_bars").glob("dataset=*/symbol=*/series=*/definition=*")):
        paths = sorted(directory.glob("session=*/bars.parquet"))
        if not paths:
            continue
        meta = partition_kv(paths[-1])
        entries.append(
            {
                "dataset": meta["bedivere.dataset"],
                "symbol": meta["bedivere.symbol"],
                "series": meta["bedivere.series"],
                "days": len(paths),
                "definition": json.loads(meta["bedivere.bar_definition"]),
                "path": str(directory),
            }
        )
    return entries
