"""Batch session-aligned aggregation of close-stamped bars into HTF bars.

Takes already-resolved SessionDays (bedivere.core.session_days) — no session
policy here. The bucket reduction is the same math the frozen view and the
incremental MarketView use, so all three construction paths agree
bit-for-bit; the equivalence tests pin that.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from typing import Literal

from bedivere.core.buckets import bucket_index_of
from bedivere.core.session_days import SessionDayResolver, SessionDays
from bedivere.core.types import Candle, Timeframe

AlignmentStrategy = Literal["session_aligned_with_stubs", "wall_clock_utc"]

# Derived targets only: "1d" is excluded and raw streams short-circuit.
_PERIOD_MINUTES: dict[Timeframe, int] = {
    Timeframe.M30: 30,
    Timeframe.H1: 60,
    Timeframe.H2: 120,
    Timeframe.H4: 240,
}


@dataclass(frozen=True, slots=True)
class AggregateOptions:
    days: SessionDays
    alignment: AlignmentStrategy
    # The caller's current instant (unix seconds) for partial-bar marking.
    # Required: the clock doctrine bans wall-clock defaults.
    now: int


def aggregate_candles(
    candles: list[Candle],
    target_timeframe: Timeframe,
    options: AggregateOptions,
) -> list[Candle]:
    """Aggregates 15-minute close-stamped candles into the target timeframe
    using the handed-over session-day boundaries. Output is close-stamped at
    the LAST underlying bar of each bucket — the convention charting
    platforms use for displayed HTF bars."""
    if target_timeframe == Timeframe.D1:
        raise ValueError(
            'aggregate_candles: target "1d" is not supported by the intraday aggregator — daily bars are session rollups, out of scope here.'
        )
    if target_timeframe in (Timeframe.S1, Timeframe.S15, Timeframe.M5, Timeframe.M15):
        return _mark_partial([_strip_partial(c) for c in candles], options)

    if options.alignment not in ("session_aligned_with_stubs", "wall_clock_utc"):
        raise ValueError(f'aggregateCandles: alignment "{options.alignment}" is not implemented')

    period_seconds = _PERIOD_MINUTES[target_timeframe] * 60
    buckets: dict[str, list[Candle]] = {}
    resolve_day = options.days.make_resolver()

    for candle in candles:
        sd = resolve_day(candle.timestamp)
        if sd is None:
            print(
                f'[aggregator] dropping bar at {candle.timestamp} — not in any handed-over session-day for template "{options.days.template}"',
                file=sys.stderr,
            )
            continue
        bucket_index = bucket_index_of(candle.timestamp, sd.start_unix, period_seconds)
        key = f"{sd.label}|{bucket_index}"
        buckets.setdefault(key, []).append(candle)

    result: list[Candle] = []
    for group in buckets.values():
        group.sort(key=lambda c: c.timestamp)
        result.append(aggregate_bucket(group))

    result.sort(key=lambda c: c.timestamp)
    return _mark_partial(result, options, resolve_day)


def aggregate_bucket(group: list[Candle]) -> Candle:
    """One bucket's bars -> a single close-stamped candle. THE reduction —
    every construction path (batch, frozen, incremental) shares it."""
    last = group[-1]
    volume = 0.0
    for c in group:
        volume += c.volume
    return Candle(
        timestamp=last.timestamp,
        open=group[0].open,
        high=max(c.high for c in group),
        low=min(c.low for c in group),
        close=last.close,
        volume=volume,
    )


def _strip_partial(c: Candle) -> Candle:
    if c.partial is None:
        return c
    return replace(c, partial=None)


def _mark_partial(
    sorted_bars: list[Candle],
    options: AggregateOptions,
    resolve_day: SessionDayResolver | None = None,
) -> list[Candle]:
    """Sets partial=True on the LAST candle when its session-day's expected
    end is in the future. Advisory and in-memory only."""
    if not sorted_bars:
        return sorted_bars
    if resolve_day is None:
        resolve_day = options.days.make_resolver()
    last = sorted_bars[-1]
    sd = resolve_day(last.timestamp)
    if sd is None:
        return sorted_bars
    if sd.end_unix > options.now:
        sorted_bars[-1] = replace(last, partial=True)
    return sorted_bars
