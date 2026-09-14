"""Coarsening one lake timeframe into a multiple of itself.

Vendors sell a handful of native timeframes — `ohlcv-1s`, `-1m`, `-1h`, `-1d` —
so every rung between those is built here. For OHLCV the reduction is EXACT:
first open, max of maxes, min of mins, last close, summed volume. A derived 5m
bar is the bar a native 5m feed would have printed.

Two deliberate non-choices:

Not a generic wall-clock resampler. Off-the-shelf resampling floors on the wall
clock, but bedivere's buckets start at the session-day OPEN, and for some
periods the two coincide — which is what would make the bug hard to find later.
`bucket_index_of` is the arithmetic `MarketView` already uses.

Not a fresh reduction. `aggregate_bucket` is imported rather than reimplemented,
so lake-derived bars are bit-identical to what `MarketView` folds in memory,
stamp included. Both stamp a bucket at its last constituent bar, which is off
the period grid whenever a bucket's last fine bar is missing — frequent at a 1s
base. Grid-stamping only the lake would break that agreement to fix nothing.

Empty buckets emit nothing. Forward-filling one would replay the previous bar's
high/low range into a quiet stretch, manufacturing signals for any strategy that
reads wicks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from bedivere.core.aggregator import aggregate_bucket
from bedivere.core.buckets import bucket_index_of
from bedivere.core.session_days import SessionDay, SessionDays
from bedivere.core.types import Candle, Timeframe
from bedivere.data.lake.layout import SeriesId, session_path, stored_labels
from bedivere.data.lake.read import read_partition
from bedivere.data.lake.schema import BarBatch
from bedivere.data.lake.writer import write_day


class ResampleError(Exception):
    """The requested coarsening is not a clean multiple, or spans a roll."""


@dataclass(frozen=True, slots=True)
class ResampleReport:
    series: SeriesId
    source: Timeframe
    target: Timeframe
    days_written: int
    bars_in: int
    bars_out: int

    def describe(self) -> str:
        return (
            f"{self.series} {self.source.value} -> {self.target.value}: "
            f"{self.bars_in:,} -> {self.bars_out:,} bars "
            f"across {self.days_written} session-days"
        )


def check_ratio(source: Timeframe, target: Timeframe) -> int:
    """Bars of `source` per bar of `target`. Raises unless it divides evenly."""
    src = source.period_seconds
    dst = target.period_seconds
    if dst <= src:
        raise ResampleError(
            f"target {target.value} ({dst}s) must be coarser than source {source.value} ({src}s)"
        )
    if dst % src != 0:
        raise ResampleError(
            f"{source.value} ({src}s) does not divide {target.value} ({dst}s) evenly — "
            "a partial bucket would silently borrow from its neighbour"
        )
    return dst // src


def resample_table(table: BarBatch, day: SessionDay, target: Timeframe) -> BarBatch:
    """Coarsen one session-day's bars. Input must be ascending by `ts`.

    A bucket spanning two `instrument_id`s is REFUSED: a roll landed mid-bucket,
    and the bar would blend two instruments' prices into one OHLC.
    """
    if len(table) == 0:
        return BarBatch()

    period = target.period_seconds
    out = BarBatch()

    group: list[Candle] = []
    group_instrument: int | None = None
    group_synthetic = False
    group_index: int | None = None

    def emit() -> None:
        if not group or group_instrument is None:
            return
        bar = aggregate_bucket(group)
        out.append(
            bar.timestamp,
            bar.open,
            bar.high,
            bar.low,
            bar.close,
            bar.volume,
            group_instrument,
            synthetic=group_synthetic,
        )

    for i, ts in enumerate(table.ts):
        index = bucket_index_of(ts, day.start_unix, period)
        if group_index is not None and index != group_index:
            emit()
            group = []
            group_instrument = None
            group_synthetic = False
        group_index = index
        if group_instrument is None:
            group_instrument = table.instrument_id[i]
        elif group_instrument != table.instrument_id[i]:
            raise ResampleError(
                f"{day.label}: {target.value} bucket {index} spans instruments "
                f"{group_instrument} and {table.instrument_id[i]} — a roll landed mid-bucket"
            )
        group_synthetic = group_synthetic or table.synthetic[i]
        group.append(
            Candle(
                timestamp=ts,
                open=table.open[i],
                high=table.high[i],
                low=table.low[i],
                close=table.close[i],
                volume=table.volume[i],
            )
        )
    emit()

    return out


SOURCE_TAG: Final[str] = "resample"


# What each RAW (vendor-ingested) timeframe feeds, in order. Chained rather than
# fanned out from the raw series: resampling is associative, so the result is
# identical either way and only the first rung pays for reading the big one.
#
# It lives here rather than in whoever fills the lake because "5m" must mean the
# same thing for every ingest.
DERIVE_CHAINS: Final[dict[Timeframe, list[tuple[Timeframe, Timeframe]]]] = {
    Timeframe.S1: [
        (Timeframe.S1, Timeframe.S15),
        (Timeframe.S15, Timeframe.M5),
        (Timeframe.M5, Timeframe.M15),
        (Timeframe.M15, Timeframe.M30),
        # The hourly rungs continue this chain rather than starting a second
        # one, so a 1s archive alone fills the whole intraday enum. The
        # `Timeframe.H1` key below is for an archive bought natively at 1h.
        (Timeframe.M30, Timeframe.H1),
        (Timeframe.H1, Timeframe.H2),
        (Timeframe.H2, Timeframe.H4),
    ],
    Timeframe.M5: [
        (Timeframe.M5, Timeframe.M15),
        (Timeframe.M15, Timeframe.M30),
        (Timeframe.M30, Timeframe.H1),
        (Timeframe.H1, Timeframe.H2),
        (Timeframe.H2, Timeframe.H4),
    ],
    Timeframe.H1: [
        (Timeframe.H1, Timeframe.H2),
        (Timeframe.H2, Timeframe.H4),
    ],
}


def resample_series(
    root: Path,
    sid: SeriesId,
    source: Timeframe,
    target: Timeframe,
    days: SessionDays,
    *,
    labels: list[str] | None = None,
    dry_run: bool = False,
) -> ResampleReport:
    """Coarsen every stored session-day of `source` into `target`.

    `labels` restricts the work to specific session-days; the default is every
    day present on disk for the source timeframe.
    """
    check_ratio(source, target)
    day_by_label = {day.label: day for day in days.days}
    todo = labels if labels is not None else stored_labels(root, sid, source)

    days_written = bars_in = bars_out = 0
    for label in todo:
        day = day_by_label.get(label)
        if day is None:
            raise ResampleError(
                f"no handed-over session-day for label {label!r} — "
                "session geometry must cover every day being resampled"
            )
        src_path = session_path(root, sid, source, label)
        if not src_path.is_file():
            continue
        table = read_partition(src_path)
        coarse = resample_table(table, day, target)
        if len(coarse) == 0:
            continue
        if not dry_run:
            write_day(root, sid, target, label, coarse, source=f"{SOURCE_TAG}:{source.value}")
        days_written += 1
        bars_in += len(table)
        bars_out += len(coarse)

    return ResampleReport(
        series=sid,
        source=source,
        target=target,
        days_written=days_written,
        bars_in=bars_in,
        bars_out=bars_out,
    )
