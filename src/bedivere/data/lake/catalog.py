"""What the lake holds, and whether it holds it completely.

A resample killed halfway leaves every file valid and every read succeeding —
the series just stops early, and the backtest reports fewer trades rather than
an error.

The invariant is LINEAGE-based, not "every timeframe spans the same range":
timeframes legitimately differ when they come from different inputs (a 1h
archive covering sixteen years, a 1s archive covering five). What always holds
is narrower —

    a resampled timeframe covers EXACTLY the session-days its SOURCE covers

— and `write_day` stamps `bedivere.source` into every footer, so the lineage is
recorded in the data rather than remembered by whoever ran the ingest.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bedivere.core.types import Timeframe
from bedivere.data.lake.layout import BARS_FILENAME, SeriesId, series_dir, stored_labels
from bedivere.data.lake.read import partition_kv
from bedivere.data.lake.resample import SOURCE_TAG
from bedivere.data.lake.schema import META_SOURCE

RESAMPLE_PREFIX = f"{SOURCE_TAG}:"


@dataclass(frozen=True, slots=True)
class LakeSeries:
    """One (series, timeframe) present on disk."""

    series: SeriesId
    timeframe: Timeframe
    source: str
    """The `bedivere.source` footer tag: a DBN filename, or `resample:<tf>`."""

    session_days: int

    @property
    def derived_from(self) -> Timeframe | None:
        """The timeframe this was resampled from, or None if directly ingested."""
        if not self.source.startswith(RESAMPLE_PREFIX):
            return None
        return Timeframe(self.source.removeprefix(RESAMPLE_PREFIX))


def _tag(root: Path, sid: SeriesId, timeframe: Timeframe, label: str) -> str:
    path = series_dir(root, sid, timeframe) / f"session={label}" / BARS_FILENAME
    return partition_kv(path).get(META_SOURCE, "")


def inventory(root: Path) -> list[LakeSeries]:
    """Every (series, timeframe) in the lake, with its lineage.

    Reads one Parquet footer per timeframe — metadata only, so it stays cheap
    no matter how much data is stored.
    """
    bars = root / "bars"
    out: list[LakeSeries] = []
    if not bars.is_dir():
        return out

    for dataset_dir in sorted(bars.glob("dataset=*")):
        for symbol_dir in sorted(dataset_dir.glob("symbol=*")):
            for series_path in sorted(symbol_dir.glob("series=*")):
                sid = SeriesId(
                    dataset=dataset_dir.name.removeprefix("dataset="),
                    symbol=symbol_dir.name.removeprefix("symbol="),
                    series=series_path.name.removeprefix("series="),
                )
                for tf_dir in sorted(series_path.glob("tf=*")):
                    timeframe = Timeframe(tf_dir.name.removeprefix("tf="))
                    labels = stored_labels(root, sid, timeframe)
                    if not labels:
                        continue
                    out.append(
                        LakeSeries(
                            series=sid,
                            timeframe=timeframe,
                            source=_tag(root, sid, timeframe, labels[-1]),
                            session_days=len(labels),
                        )
                    )
    return out


def coverage_gaps(root: Path) -> list[str]:
    """Human-readable descriptions of every incomplete resampled timeframe.

    Empty means every derived series covers exactly the days its source does.
    """
    problems: list[str] = []
    entries = inventory(root)
    days_by_key: dict[tuple[str, str, str, str], set[str]] = {
        (
            entry.series.dataset,
            entry.series.symbol,
            entry.series.series,
            entry.timeframe.value,
        ): set(stored_labels(root, entry.series, entry.timeframe))
        for entry in entries
    }

    for entry in entries:
        parent = entry.derived_from
        if parent is None:
            continue
        sid = entry.series
        child_key = (sid.dataset, sid.symbol, sid.series, entry.timeframe.value)
        parent_key = (sid.dataset, sid.symbol, sid.series, parent.value)
        parent_days = days_by_key.get(parent_key)
        if parent_days is None:
            problems.append(
                f"{sid} {entry.timeframe.value}: derived from {parent.value}, "
                "which is not in the lake"
            )
            continue
        child_days = days_by_key[child_key]
        missing = parent_days - child_days
        extra = child_days - parent_days
        if missing or extra:
            detail: list[str] = []
            if missing:
                detail.append(f"{len(missing)} missing (e.g. {sorted(missing)[0]})")
            if extra:
                detail.append(f"{len(extra)} not in source (e.g. {sorted(extra)[0]})")
            problems.append(
                f"{sid} {entry.timeframe.value} (from {parent.value}): "
                f"{len(child_days)} of {len(parent_days)} session-days — " + ", ".join(detail)
            )
    return problems
