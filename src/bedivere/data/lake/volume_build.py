"""Local-only volume-bar builders, and a trades versus one-second comparison.

The comparison builds both representations from the same selected trades, so
it isolates aggregation loss; its approximation is stored as `trades-1s`,
never where `derive_volume` writes `ohlcv-1s`. Archive builds stage each
session and publish only after the read validates through EOF.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import replace
from datetime import UTC, datetime
from itertools import groupby
from pathlib import Path

from bedivere.core.session_days import SessionDay, SessionDays
from bedivere.core.types import Timeframe
from bedivere.data.lake.ingest import IngestError as IngestError
from bedivere.data.lake.layout import SeriesId, session_path
from bedivere.data.lake.read import partition_kv, read_partition, source_range
from bedivere.data.lake.trades import TradeIssue, TradeRead, open_trade_archive
from bedivere.data.lake.trades import fingerprint as _fingerprint
from bedivere.data.lake.volume import (
    NS,
    ApproxBoundary,
    VolumeAccumulator,
    VolumeBar,
    VolumeInput,
    VolumeSpec,
    summarize,
)
from bedivere.data.lake.volume_store import (
    SAME_TRADE_SECONDS,
    StagedPartition,
    checked_volume_partition,
    publish_staged,
    stage_volume_day,
    volume_path,
    write_volume_day,
)

# How a build report's source was checked in this run.
SOURCE_FINGERPRINT = "fingerprint"
"""Resumed: the archive's hash matched the stored source; nothing was decoded."""

SOURCE_FULL_READ = "full_read"
"""Every archive record was decoded and the read validated through EOF."""


def time_inputs(root: Path, sid: SeriesId, tf: Timeframe, day: SessionDay) -> Iterator[VolumeInput]:
    table = read_partition(session_path(root, sid, tf, day.label))
    for i, ts in enumerate(table.ts):
        if table.synthetic[i]:
            raise ValueError("synthetic time bars cannot be used to build volume bars")
        volume = table.volume[i]
        if not volume.is_integer() or not 0 <= volume <= 2**53:
            raise ValueError("source volume must be an exactly represented non-negative integer")
        yield VolumeInput(
            (ts - tf.period_seconds) * NS,
            ts * NS,
            table.open[i],
            table.high[i],
            table.low[i],
            table.close[i],
            int(volume),
            table.instrument_id[i],
        )


def _selected_trades(read: TradeRead, days: SessionDays) -> Iterator[tuple[SessionDay, VolumeInput]]:
    """Selected trades in archive order. An ineligible one fails the build
    instead of being dropped."""
    by_label = {day.label: day for day in days.days}
    invalid = TradeIssue.NON_TRADE_ACTION | TradeIssue.UNDEFINED_PRICE | TradeIssue.ZERO_SIZE
    for batch in read:
        for trade in batch:
            if trade.issues & invalid:
                raise ValueError("invalid trade action, price, or size")
            if trade.issues & TradeIssue.BAD_TS_RECV:
                raise ValueError("trade has F_BAD_TS_RECV; receive-time bars would be unreliable")
            assert trade.session is not None and trade.price is not None
            yield (
                by_label[trade.session],
                VolumeInput.trade(trade.ts_recv, trade.price, trade.size, trade.instrument_id),
            )


def one_second_inputs(trades: Iterable[VolumeInput]) -> Iterator[VolumeInput]:
    """Exact receive-time OHLCV from the same trades; no empty seconds emitted."""
    for second, grouped in groupby(trades, key=lambda row: row.start_ns // NS):
        rows = iter(grouped)
        first = next(rows)
        high, low, close, volume = first.high, first.low, first.close, first.volume
        for row in rows:
            if row.instrument_id != first.instrument_id:
                raise ValueError("one-second aggregation cannot mix contracts")
            high, low, close = max(high, row.high), min(low, row.low), row.close
            volume += row.volume
        yield VolumeInput(
            second * NS,
            (second + 1) * NS,
            first.open,
            high,
            low,
            close,
            volume,
            first.instrument_id,
        )


def _report(
    root: Path,
    sid: SeriesId,
    spec: VolumeSpec,
    days: SessionDays,
    day: SessionDay,
    bars: list[VolumeBar],
    *,
    written: bool,
    status: str,
    source_check: str | None = None,
) -> dict[str, object]:
    return {
        "session": day.label,
        "source": spec.source,
        "boundary": spec.boundary,
        "threshold": spec.threshold,
        "path": str(volume_path(root, sid, spec, days, day.label)),
        "written": written,
        "status": status,
        **({"source_check": source_check} if source_check is not None else {}),
        **summarize(bars, spec.threshold),
    }


def _check_session(sid: SeriesId, day: SessionDay, bars: list[VolumeBar], input_volume: int) -> None:
    if not bars:
        raise ValueError(f"{day.label}: no positive-volume inputs for {sid}")
    if sum(b.volume for b in bars) != input_volume:
        raise ValueError("volume conservation failed")


def _publish(
    root: Path,
    sid: SeriesId,
    spec: VolumeSpec,
    days: SessionDays,
    day: SessionDay,
    bars: list[VolumeBar],
    input_volume: int,
    source: str,
    dry_run: bool,
) -> dict[str, object]:
    _check_session(sid, day, bars, input_volume)
    if not dry_run:
        write_volume_day(root, sid, spec, days, day, bars, source=source)
    status = "computed" if dry_run else "written"
    return _report(root, sid, spec, days, day, bars, written=not dry_run, status=status)


class _Staging:
    """One build's staging directory: inside the lake root, so publishing is a
    rename, and outside the trees readers glob."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._created: list[Path] = []
        self._directory: Path | None = None
        self.partitions: list[StagedPartition] = []

    def directory(self) -> Path:
        if self._directory is None:
            parent = self._root
            while not parent.exists():
                self._created.append(parent)
                parent = parent.parent
            self._root.mkdir(parents=True, exist_ok=True)
            self._directory = Path(tempfile.mkdtemp(prefix=".volume-staging-", dir=self._root))
        return self._directory

    def publish(self) -> None:
        for partition in self.partitions:
            publish_staged(partition)

    def discard(self) -> None:
        if self._directory is not None:
            shutil.rmtree(self._directory, ignore_errors=True)
        for directory in self._created:  # deepest first; never one holding anything
            try:
                directory.rmdir()
            except OSError:
                break


def _stage(
    root: Path,
    sid: SeriesId,
    spec: VolumeSpec,
    days: SessionDays,
    day: SessionDay,
    bars: list[VolumeBar],
    input_volume: int,
    source: str,
    staging: _Staging | None,
) -> dict[str, object]:
    _check_session(sid, day, bars, input_volume)
    if staging is not None:
        staging.partitions.append(
            stage_volume_day(
                root, sid, spec, days, day, bars, source=source, staging=staging.directory()
            )
        )
    status = "written" if staging is not None else "computed"
    return _report(
        root,
        sid,
        spec,
        days,
        day,
        bars,
        written=staging is not None,
        status=status,
        source_check=SOURCE_FULL_READ,
    )


def _cached_report(
    root: Path,
    sid: SeriesId,
    spec: VolumeSpec,
    days: SessionDays,
    day: SessionDay,
    source: str,
    source_check: str | None = None,
) -> dict[str, object] | None:
    path = volume_path(root, sid, spec, days, day.label)
    if not path.is_file():
        return None
    meta = partition_kv(path)
    if meta.get("bedivere.source") != source or "bedivere.content_sha256" not in meta:
        return None
    bars = checked_volume_partition(root, sid, spec, days, day)
    return _report(
        root,
        sid,
        spec,
        days,
        day,
        bars,
        written=False,
        status="unchanged",
        source_check=source_check,
    )


def _utc(ns: int) -> str:
    return datetime.fromtimestamp(ns // NS, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _require_whole_sessions(root: Path, sid: SeriesId, tf: Timeframe, days: SessionDays) -> None:
    """Every stored session's source must have covered it from open to close.

    A footer recording no source range is refused too: the bars cannot show
    what their source left out.
    """
    partial: list[str] = []
    for day in days.days:
        path = session_path(root, sid, tf, day.label)
        if not path.is_file():
            continue  # reported as missing where it is read
        covered = source_range(path)
        opened, closed = day.start_unix * NS, day.end_unix * NS
        if covered is None:
            partial.append(f"{day.label}: its partition records no source range")
        elif covered[0] > opened or covered[1] < closed:
            partial.append(
                f"{day.label}: session {_utc(opened)} .. {_utc(closed)}, source "
                f"{_utc(covered[0])} .. {_utc(covered[1])}"
            )
    if partial:
        shown = partial[:8] + ([f"... and {len(partial) - 8} more"] if len(partial) > 8 else [])
        raise ValueError(
            f"stored {tf.value} bars do not cover {len(partial)} requested session(s) from open "
            "to close, and volume bars built from part of a session would be stamped with all "
            "of it:\n  " + "\n  ".join(shown) + "\nRequest only sessions the ingested archive "
            "covers in full (a download bounded at UTC midnight starts and ends inside one), and "
            "re-ingest partitions that record no source range from their archive."
        )


def derive_volume(
    root: Path,
    sid: SeriesId,
    spec: VolumeSpec,
    days: SessionDays,
    *,
    dry_run: bool = False,
    resume: bool = False,
) -> list[dict[str, object]]:
    """Volume bars from each session's stored one-second bars (`ohlcv-1s`).

    Nothing is built unless every requested session's 1s source covered it in full.
    """
    if not spec.source.startswith("ohlcv-"):
        raise ValueError("building from stored bars requires source=ohlcv-1s")
    tf = Timeframe(spec.source.removeprefix("ohlcv-"))
    if not days.days:
        raise ValueError("at least one session-day is required")
    _require_whole_sessions(root, sid, tf, days)
    reports: list[dict[str, object]] = []
    for day in days.days:
        source = _fingerprint(session_path(root, sid, tf, day.label))
        if resume and (existing := _cached_report(root, sid, spec, days, day, source)) is not None:
            reports.append(existing)
            continue
        accumulator = VolumeAccumulator(spec, day)
        bars: list[VolumeBar] = []
        total = 0
        for row in time_inputs(root, sid, tf, day):
            total += row.volume
            bars.extend(accumulator.push(row))
        bars.extend(accumulator.finish())
        reports.append(_publish(root, sid, spec, days, day, bars, total, source, dry_run))
    return reports


def _aggregate_session(
    trades: Iterable[VolumeInput],
    spec: VolumeSpec,
    day: SessionDay,
    compare_1s: bool,
    approximate_spec: VolumeSpec,
) -> tuple[list[VolumeBar], list[VolumeBar], int]:
    accumulator = VolumeAccumulator(spec, day)
    bars: list[VolumeBar] = []
    total = 0

    def consume() -> Iterator[VolumeInput]:
        nonlocal total
        for trade in trades:
            total += trade.volume
            bars.extend(accumulator.push(trade))
            yield trade

    approximate = VolumeAccumulator(approximate_spec, day)
    approximate_bars: list[VolumeBar] = []
    if compare_1s:
        for second in one_second_inputs(consume()):
            approximate_bars.extend(approximate.push(second))
        approximate_bars.extend(approximate.finish())
    else:
        for _ in consume():
            pass
    bars.extend(accumulator.finish())
    return bars, approximate_bars, total


def build_volume_archive(
    archive: Path,
    root: Path,
    sid: SeriesId,
    spec: VolumeSpec,
    days: SessionDays,
    *,
    compare_1s: bool = False,
    approx_boundary: ApproxBoundary = "whole_trade",
    dry_run: bool = False,
    resume: bool = False,
) -> list[dict[str, object]]:
    """Trade-built volume bars, optionally with their `trades-1s` approximation.

    The archive is hashed before anything is decoded. A resumed session whose
    partitions already record it is returned without decoding; otherwise
    results are staged, and published only after the read validates through
    EOF. Publication is atomic per partition, not across them.
    """
    if spec.source != "trades":
        raise ValueError("archive building requires source=trades")
    if approx_boundary not in ("whole_trade", "nearest_second"):
        raise ValueError("approx_boundary must be whole_trade or nearest_second")
    if not compare_1s and approx_boundary != "whole_trade":
        raise ValueError("approx_boundary requires compare_1s")
    with open_trade_archive(archive) as opened:
        source = opened.fingerprint
        reports: list[dict[str, object]] = []
        approximate_spec = replace(spec, source="trades-1s", boundary=approx_boundary)
        source_by_day: dict[str, str] = {}
        pending: list[SessionDay] = []
        for day in days.days:
            day_source = source
            if not sid.series.startswith("raw."):
                # A changed contract selection invalidates the output, same archive or not.
                day_source += ":selection:" + _fingerprint(
                    session_path(root, sid, Timeframe.S1, day.label)
                )
            source_by_day[day.label] = day_source
            expected = [(spec, day_source)]
            if compare_1s:
                expected.append((approximate_spec, SAME_TRADE_SECONDS + day_source))
            existing = (
                [
                    _cached_report(root, sid, definition, days, day, origin, SOURCE_FINGERPRINT)
                    for definition, origin in expected
                ]
                if resume
                else []
            )
            if existing and all(report is not None for report in existing):
                reports.extend(report for report in existing if report is not None)
            else:
                pending.append(day)
        if not days.days:
            raise ValueError("at least one session-day is required")
        if not pending:
            return reports
        build_days = replace(days, days=tuple(pending))
        staging = None if dry_run else _Staging(root)
        try:
            read = opened.read(sid, sessions=build_days, lake_root=root)
            seen: set[str] = set()
            for day, grouped in groupby(
                _selected_trades(read, build_days), key=lambda pair: pair[0]
            ):
                seen.add(day.label)
                bars, approximate_bars, total = _aggregate_session(
                    (trade for _, trade in grouped),
                    spec,
                    day,
                    compare_1s,
                    approximate_spec,
                )
                day_source = source_by_day[day.label]
                reports.append(_stage(root, sid, spec, days, day, bars, total, day_source, staging))
                if compare_1s:
                    reports.append(
                        _stage(
                            root,
                            sid,
                            approximate_spec,
                            days,
                            day,
                            approximate_bars,
                            total,
                            SAME_TRADE_SECONDS + day_source,
                            staging,
                        )
                    )
            # Exhausted, so validated: a failure would have raised above.
            if not read.validated:
                raise IngestError(f"{archive.name}: trade read ended without validating")
            for label, stamp in read.source.selection_dependencies:
                if source_by_day[label] != f"{source}:selection:{stamp}":
                    raise ValueError(f"{label}: reference 1s partition changed during the build")
            missing = {day.label for day in pending} - seen
            if missing:
                raise ValueError(
                    f"no selected trades for requested sessions: {', '.join(sorted(missing))}"
                )
            if staging is not None:
                staging.publish()
        finally:
            if staging is not None:
                staging.discard()
    return sorted(
        reports, key=lambda report: (str(report["session"]), report["source"] != "trades")
    )
