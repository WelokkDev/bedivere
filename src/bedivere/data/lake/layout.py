"""Where a bar lives, and what identifies it.

Provenance is the PATH. Hive-style, so DuckDB discovers the partitions
unaided:

    <root>/bars/dataset=GLBX.MDP3/symbol=NQ/series=v.0/tf=1s/session=2026-06-15/bars.parquet

The feed, the contract-selection rule and the timeframe are partition LEVELS
rather than columns, because a series that rolls on volume and one that rolls
on open interest hold different contracts on the same dates and can print
prices a whole roll gap apart. Keyed `(symbol, timeframe, timestamp)` in one
flat table, whichever wrote last wins and nothing downstream can tell.
Partitioned, they cannot collide at all.

`session` is the SESSION-DAY label, not the UTC date — partitioning on UTC
would split each trading session at 00:00 UTC, mid-evening-session. The
vendor's condition feed is the one place UTC dates appear, and
`bedivere.data.quality` is where the two calendars are reconciled.

Paths and identities only, so nothing that merely locates a file drags in
DuckDB. The table's shape lives in `schema.py`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from bedivere.core.types import Timeframe

BARS_FILENAME: Final[str] = "bars.parquet"

# `instrument_id` for a bar whose source cannot name its contract. Nothing
# writes it today — both `write_day` callers know the id — but vendor ids are
# positive, so 0 cannot collide, and an unknown left visible cannot hide a roll
# from `roll_boundaries` the way a plausible-looking id would.
UNKNOWN_INSTRUMENT: Final[int] = 0

# Partition tokens end up as directory names. Anything outside this set could
# escape the lake root or produce a path that differs across platforms.
_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Session-day labels are the ISO dates `SessionDay.label` carries.
_LABEL_RE: Final[re.Pattern[str]] = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class RollRule(StrEnum):
    """Databento's continuous-contract roll rules: `[ROOT].[RULE].[RANK]`.

    All three rank on the PREVIOUS day's figures, which is what makes them
    causal — the ranking was knowable before the session opened.
    """

    CALENDAR = "c"
    OPEN_INTEREST = "n"
    VOLUME = "v"


@dataclass(frozen=True, slots=True)
class SeriesId:
    """The identity of one price series — everything above the timeframe.

    Two that differ in any field are different series and must never be
    compared or concatenated; the path is what enforces it.
    """

    dataset: str
    """Vendor dataset code, e.g. `GLBX.MDP3`."""

    symbol: str
    """Root/asset symbol, e.g. `NQ`."""

    series: str
    """How a tradeable instrument is selected over time.

    `v.0` / `n.0` / `c.1`  — the vendor's own continuous symbology
    `local.v.0`            — a continuous series WE ranked (see below)
    `raw.NQZ5`             — one specific outright contract
    """

    def __post_init__(self) -> None:
        for name, value in (
            ("dataset", self.dataset),
            ("symbol", self.symbol),
            ("series", self.series),
        ):
            if not _TOKEN_RE.match(value):
                raise ValueError(
                    f"SeriesId.{name} must match {_TOKEN_RE.pattern} (got {value!r}) — "
                    "partition tokens become directory names"
                )

    def __str__(self) -> str:
        return f"{self.dataset}:{self.symbol}:{self.series}"


def continuous(dataset: str, symbol: str, roll: RollRule, rank: int) -> SeriesId:
    """A vendor-served continuous series. `rank` is zero-indexed: 0 is front month."""
    if rank < 0:
        raise ValueError(f"continuous rank must be >= 0 (got {rank})")
    return SeriesId(dataset=dataset, symbol=symbol, series=f"{roll.value}.{rank}")


def local_continuous(dataset: str, symbol: str, roll: RollRule, rank: int) -> SeriesId:
    """A continuous series WE assembled — `local.v.0`, not `v.0`.

    The two need not agree: the vendor ranks over the whole venue, we rank only
    over the contracts in the file we bought. Naming ours `v.0` would let a later
    vendor download silently overwrite a series it does not equal.
    """
    if rank < 0:
        raise ValueError(f"continuous rank must be >= 0 (got {rank})")
    return SeriesId(dataset=dataset, symbol=symbol, series=f"local.{roll.value}.{rank}")


def raw_contract(dataset: str, symbol: str, contract: str) -> SeriesId:
    return SeriesId(dataset=dataset, symbol=symbol, series=f"raw.{contract}")


def to_symbol(sid: SeriesId) -> str:
    """The vendor request symbol: `NQ.v.0` for continuous, `NQZ5` for raw.

    Refusing `local.` series here is what stops a locally-derived identity being
    handed to a vendor fetch and coming back as if it were vendor history.
    """
    if sid.series.startswith("raw."):
        return sid.series.removeprefix("raw.")
    if sid.series.startswith("local."):
        raise ValueError(
            f"{sid} is assembled locally and has no vendor symbol — "
            "request the vendor's own continuous series instead"
        )
    return f"{sid.symbol}.{sid.series}"


def series_dir(root: Path, sid: SeriesId, timeframe: Timeframe) -> Path:
    return (
        root
        / "bars"
        / f"dataset={sid.dataset}"
        / f"symbol={sid.symbol}"
        / f"series={sid.series}"
        / f"tf={timeframe.value}"
    )


def session_path(root: Path, sid: SeriesId, timeframe: Timeframe, label: str) -> Path:
    if not _LABEL_RE.match(label):
        raise ValueError(f"session-day label must be ISO yyyy-mm-dd (got {label!r})")
    return series_dir(root, sid, timeframe) / f"session={label}" / BARS_FILENAME


def label_of(session_dir: Path) -> str:
    name = session_dir.name
    if not name.startswith("session="):
        raise ValueError(f"not a session partition directory: {session_dir}")
    return name.removeprefix("session=")


def stored_labels(root: Path, sid: SeriesId, timeframe: Timeframe) -> list[str]:
    """Labels present on disk, ascending. ISO dates sort lexicographically."""
    base = series_dir(root, sid, timeframe)
    if not base.is_dir():
        return []
    return sorted(
        label_of(child)
        for child in base.iterdir()
        if child.is_dir() and (child / BARS_FILENAME).is_file()
    )
