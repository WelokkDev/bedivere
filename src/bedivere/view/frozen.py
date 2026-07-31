"""The frozen view — pull-shaped as-of snapshots, and the test oracle.

A FrozenView is the as-of snapshot read at a single bar-close instant: the
entry-TF series plus, per higher timeframe, the closed-bucket `completed` cut
and the forming-bar-inclusive `asOfView` cut. build_frozen_source precomputes
per-TF buckets once and serves view_at(as_of) slices; the frozen-source tests
pin byte-equality between the two construction paths, and the market-view
tests pin equality against the incremental (push-shaped) MarketView at every
instant — two independent constructions that must agree.

Session-days arrive already resolved (see bedivere.core.session_days).
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

from bedivere.core.aggregator import aggregate_bucket
from bedivere.core.buckets import bucket_index_of, bucket_period_end
from bedivere.core.session_days import SessionDays
from bedivere.core.types import Candle, Timeframe

__all__ = [
    "FrozenSource",
    "FrozenView",
    "bucket_index_of",  # re-export: moved to bedivere.core.buckets (single home)
    "build_frozen_source",
    "build_frozen_view",
]


@dataclass(frozen=True, slots=True)
class FrozenView:
    # Entry-TF bars with timestamp <= asOf.
    primary: list[Candle]
    # Per HTF: bars whose bucket CLOSED at <= asOf — the only bars safe for
    # zone detection.
    completed: dict[Timeframe, list[Candle]]
    # Per HTF: `completed` plus a trailing forming bar (NOT flagged partial).
    as_of_view: dict[Timeframe, list[Candle]]


def _reject_daily(tf: Timeframe, fn: str) -> None:
    if tf == Timeframe.D1:
        raise ValueError(
            f'{fn}: "1d" is out of scope — daily bars are session rollups, not intraday buckets. Pass a subset of 15m/30m/1h/2h/4h.'
        )


@dataclass(slots=True)
class _Bucket:
    bars: list[Candle]
    sd_start: int
    sd_end: int
    index: int


def build_frozen_view(
    primary_bars: list[Candle],
    as_of: int,
    timeframes: list[Timeframe],
    days: SessionDays,
) -> FrozenView:
    """The as-of multi-timeframe snapshot for a single decision instant."""
    primary = sorted((b for b in primary_bars if b.timestamp <= as_of), key=lambda b: b.timestamp)

    completed: dict[Timeframe, list[Candle]] = {}
    as_of_view: dict[Timeframe, list[Candle]] = {}

    for tf in timeframes:
        _reject_daily(tf, "buildFrozenView")
        completed_bars, forming_bar = _bucket_as_of(primary, as_of, tf.period_seconds, days)
        completed[tf] = completed_bars
        as_of_view[tf] = [*completed_bars, forming_bar] if forming_bar else [*completed_bars]

    return FrozenView(primary=primary, completed=completed, as_of_view=as_of_view)


def _bucket_as_of(
    primary: list[Candle],
    as_of: int,
    period_seconds: int,
    days: SessionDays,
) -> tuple[list[Candle], Candle | None]:
    """Bucket `primary` (already <= asOf) with the exact aggregator math,
    then split by bucket period-end vs asOf. A bucket's period-end is its
    natural grid close clamped to the session-day end."""
    buckets: dict[str, _Bucket] = {}
    resolve_day = days.make_resolver()

    for c in primary:
        sd = resolve_day(c.timestamp)
        # Bars outside any session-day are dropped from HTF aggregation but
        # stay in `primary`.
        if sd is None:
            continue
        index = bucket_index_of(c.timestamp, sd.start_unix, period_seconds)
        key = f"{sd.label}|{index}"
        bucket = buckets.get(key)
        if bucket is None:
            bucket = _Bucket(bars=[], sd_start=sd.start_unix, sd_end=sd.end_unix, index=index)
            buckets[key] = bucket
        bucket.bars.append(c)

    completed_bars: list[Candle] = []
    forming_bar: Candle | None = None
    forming_stamp = float("-inf")

    for bucket in buckets.values():
        bucket.bars.sort(key=lambda b: b.timestamp)
        candle = aggregate_bucket(bucket.bars)
        period_end = bucket_period_end(bucket.sd_start, bucket.sd_end, bucket.index, period_seconds)
        if period_end <= as_of:
            completed_bars.append(candle)
        elif candle.timestamp > forming_stamp:
            # At most one bucket can straddle asOf; the max-timestamp guard is
            # belt-and-suspenders.
            forming_bar = candle
            forming_stamp = candle.timestamp

    completed_bars.sort(key=lambda b: b.timestamp)
    return completed_bars, forming_bar


@dataclass(slots=True)
class _SourceBucket:
    period_end: int  # natural grid close clamped to the session-day end
    candle: Candle  # aggregate of ALL the bucket's bars (valid once completed)
    bars: list[Candle]  # the bucket's sub-bars, ascending
    first_ts: int


class FrozenSource:
    """Precomputed per-TF bucket lists over the full primary series; reuse
    view_at for every decision instant instead of rebuilding from scratch."""

    def __init__(
        self,
        primary_bars: list[Candle],
        timeframes: list[Timeframe],
        days: SessionDays,
    ) -> None:
        self._timeframes = list(timeframes)
        self._primary = sorted(primary_bars, key=lambda b: b.timestamp)
        self._primary_stamps = [b.timestamp for b in self._primary]
        self._buckets_by_tf: dict[Timeframe, list[_SourceBucket]] = {}
        self._period_ends_by_tf: dict[Timeframe, list[int]] = {}
        for tf in self._timeframes:
            _reject_daily(tf, "buildFrozenSource")
            buckets = _bucketize(self._primary, tf.period_seconds, days)
            self._buckets_by_tf[tf] = buckets
            self._period_ends_by_tf[tf] = [b.period_end for b in buckets]

    def view_at(self, as_of: int) -> FrozenView:
        primary_slice = self._primary[: bisect_right(self._primary_stamps, as_of)]

        completed: dict[Timeframe, list[Candle]] = {}
        as_of_view: dict[Timeframe, list[Candle]] = {}

        for tf in self._timeframes:
            buckets = self._buckets_by_tf[tf]
            period_ends = self._period_ends_by_tf[tf]
            c = bisect_right(period_ends, as_of)  # buckets with periodEnd <= asOf

            completed_bars = [buckets[i].candle for i in range(c)]

            # The single straddling bucket is the forming bar, re-aggregated
            # from ONLY its sub-bars <= asOf.
            forming_bar: Candle | None = None
            if c < len(buckets):
                straddle = buckets[c]
                if straddle.first_ts <= as_of:
                    sub = [b for b in straddle.bars if b.timestamp <= as_of]
                    if sub:
                        forming_bar = aggregate_bucket(sub)

            completed[tf] = completed_bars
            as_of_view[tf] = [*completed_bars, forming_bar] if forming_bar else [*completed_bars]

        return FrozenView(primary=primary_slice, completed=completed, as_of_view=as_of_view)


def build_frozen_source(
    primary_bars: list[Candle],
    timeframes: list[Timeframe],
    days: SessionDays,
) -> FrozenSource:
    return FrozenSource(primary_bars, timeframes, days)


def _bucketize(
    primary: list[Candle],
    period_seconds: int,
    days: SessionDays,
) -> list[_SourceBucket]:
    """Full-primary bucketization, ordered by periodEnd ascending."""
    grouped: dict[str, _Bucket] = {}
    resolve_day = days.make_resolver()
    for c in primary:
        sd = resolve_day(c.timestamp)
        if sd is None:
            continue
        index = bucket_index_of(c.timestamp, sd.start_unix, period_seconds)
        key = f"{sd.label}|{index}"
        b = grouped.get(key)
        if b is None:
            b = _Bucket(bars=[], sd_start=sd.start_unix, sd_end=sd.end_unix, index=index)
            grouped[key] = b
        b.bars.append(c)

    out: list[_SourceBucket] = []
    for b in grouped.values():
        b.bars.sort(key=lambda bb: bb.timestamp)
        period_end = bucket_period_end(b.sd_start, b.sd_end, b.index, period_seconds)
        out.append(
            _SourceBucket(
                period_end=period_end,
                candle=aggregate_bucket(b.bars),
                bars=b.bars,
                first_ts=b.bars[0].timestamp,
            )
        )
    out.sort(key=lambda sb: sb.period_end)
    return out
