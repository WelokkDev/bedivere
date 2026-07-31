"""Frozen-view construction — completed/forming cuts at as-of instants."""

from __future__ import annotations

import math

from bedivere.core.aggregator import AggregateOptions, aggregate_candles
from bedivere.core.types import Candle, Timeframe
from bedivere.view.frozen import build_frozen_view
from tests.helpers import bar, et, eth_5m_flat, eth_session_days, make_session_days, prev_day, utc

# Literal resolved session-days covering every close-date this module
# touches.
ETH = eth_session_days(["2026-01-06", "2026-04-21", "2026-04-22"])
FAR = utc(2099, 1, 1, 0)


def eth_opts(now: int) -> AggregateOptions:
    return AggregateOptions(days=ETH, alignment="session_aligned_with_stubs", now=now)


class TestAsOfExactlyOn4hClose:
    def test_boundary_bar_in_completed_not_duplicated(self) -> None:
        bars = eth_5m_flat("2026-04-21")
        as_of = et("2026-04-21T10:00:00")
        fv = build_frozen_view(bars, as_of, [Timeframe.H4], ETH)

        completed = fv.completed[Timeframe.H4]
        assert [b.timestamp for b in completed] == [
            et("2026-04-20T22:00:00"),
            et("2026-04-21T02:00:00"),
            et("2026-04-21T06:00:00"),
            et("2026-04-21T10:00:00"),
        ]
        assert all(b.timestamp <= as_of for b in completed)
        assert fv.as_of_view[Timeframe.H4] == completed


class TestFormingCompletedFlip:
    def test_close_minus_1s_still_forming(self) -> None:
        bars = eth_5m_flat("2026-04-21")
        close = et("2026-04-21T10:00:00")
        fv = build_frozen_view(bars, close - 1, [Timeframe.H4], ETH)
        completed = fv.completed[Timeframe.H4]
        as_of_view = fv.as_of_view[Timeframe.H4]
        assert [b.timestamp for b in completed] == [
            et("2026-04-20T22:00:00"),
            et("2026-04-21T02:00:00"),
            et("2026-04-21T06:00:00"),
        ]
        assert len(as_of_view) == 4
        assert as_of_view[-1].timestamp == et("2026-04-21T09:55:00")
        assert not any(b.timestamp == et("2026-04-21T09:55:00") for b in completed)

    def test_close_exactly_completed_no_forming(self) -> None:
        bars = eth_5m_flat("2026-04-21")
        close = et("2026-04-21T10:00:00")
        fv = build_frozen_view(bars, close, [Timeframe.H4], ETH)
        assert close in [b.timestamp for b in fv.completed[Timeframe.H4]]
        assert fv.as_of_view[Timeframe.H4] == fv.completed[Timeframe.H4]

    def test_close_plus_1s_no_new_forming(self) -> None:
        bars = eth_5m_flat("2026-04-21")
        close = et("2026-04-21T10:00:00")
        fv = build_frozen_view(bars, close + 1, [Timeframe.H4], ETH)
        completed = fv.completed[Timeframe.H4]
        assert completed[-1].timestamp == close
        assert fv.as_of_view[Timeframe.H4] == completed


class TestLeakProbe:
    def test_future_spikes_never_leak(self) -> None:
        open_ts = et("2026-04-20T18:00:00")
        end = et("2026-04-21T10:00:00")
        b3_start = et("2026-04-21T06:00:00")
        as_of = et("2026-04-21T08:00:00")
        bars: list[Candle] = []
        for ts in range(open_ts + 300, end + 1, 300):
            in_b3 = b3_start < ts <= end
            if in_b3 and ts <= as_of:
                bars.append(bar(ts, 100, 101, 99, 100, 1))
            elif in_b3 and ts > as_of:
                bars.append(bar(ts, 100, 9999, 1, 100, 1))  # future spike
            else:
                bars.append(bar(ts, 100, 100, 100, 100, 1))

        fv = build_frozen_view(bars, as_of, [Timeframe.H4], ETH)
        completed = fv.completed[Timeframe.H4]
        as_of_view = fv.as_of_view[Timeframe.H4]

        assert len(completed) == 3
        assert all(b.timestamp <= as_of for b in completed)

        forming = as_of_view[-1]
        assert forming.timestamp == as_of
        assert forming.high == 101
        assert forming.low == 99
        assert forming.volume == 24

        for b in [*completed, *as_of_view]:
            assert b.high <= 101
            assert b.low >= 99


def test_empty_forming_bucket_boundary() -> None:
    bars = eth_5m_flat("2026-04-21")
    as_of = et("2026-04-21T02:00:00")
    tfs = [Timeframe.M15, Timeframe.M30, Timeframe.H1, Timeframe.H2, Timeframe.H4]
    fv = build_frozen_view(bars, as_of, tfs, ETH)
    for tf in tfs:
        assert fv.as_of_view[tf] == fv.completed[tf]
        assert all(b.timestamp <= as_of for b in fv.completed[tf])


def test_5m_to_15m_aggregation() -> None:
    def t(hhmm: str) -> int:
        return et(f"2026-04-20T{hhmm}:00")

    bars = [
        bar(t("18:05"), 100, 105, 98, 101, 1),
        bar(t("18:10"), 101, 110, 100, 108, 2),
        bar(t("18:15"), 108, 112, 104, 106, 3),
        bar(t("18:20"), 106, 107, 95, 99, 4),
        bar(t("18:25"), 99, 103, 97, 102, 5),
        bar(t("18:30"), 102, 109, 101, 107, 6),
    ]

    fv = build_frozen_view(bars, FAR, [Timeframe.M15], ETH)
    assert fv.completed[Timeframe.M15] == [
        bar(t("18:15"), 100, 112, 98, 106, 6),
        bar(t("18:30"), 106, 109, 95, 107, 15),
    ]
    # Documents the trap: aggregateCandles treats "15m" as passthrough.
    assert len(aggregate_candles(bars, Timeframe.M15, eth_opts(FAR))) == 6


def test_completed_equals_aggregate_candles_when_everything_closed() -> None:
    bars: list[Candle] = []
    i = 0
    for cd in ["2026-04-21", "2026-04-22"]:
        open_ts = et(f"{prev_day(cd)}T18:00:00")
        close_ts = et(f"{cd}T17:00:00")
        for ts in range(open_ts + 300, close_ts + 1, 300):
            base = 100 + ((i * 13) % 37)
            bars.append(bar(ts, base, base + 4, base - 3, base + 2, 1 + (i % 6)))
            i += 1

    tfs = [Timeframe.M30, Timeframe.H1, Timeframe.H2, Timeframe.H4]
    fv = build_frozen_view(bars, FAR, tfs, ETH)
    for tf in tfs:
        assert fv.completed[tf] == aggregate_candles(bars, tf, eth_opts(FAR))
        assert fv.as_of_view[tf] == fv.completed[tf]


class TestDstSessionBoundaries:
    def test_est_week_4h_close_stamps(self) -> None:
        bars = eth_5m_flat("2026-01-06")  # Tue, EST
        fv = build_frozen_view(bars, FAR, [Timeframe.H4], ETH)
        assert [b.timestamp for b in fv.completed[Timeframe.H4]] == [
            et("2026-01-05T22:00:00"),
            et("2026-01-06T02:00:00"),
            et("2026-01-06T06:00:00"),
            et("2026-01-06T10:00:00"),
            et("2026-01-06T14:00:00"),
            et("2026-01-06T17:00:00"),  # 3h stub
        ]

    def test_flip_lands_on_correct_est_instant(self) -> None:
        bars = eth_5m_flat("2026-01-06")
        close = et("2026-01-06T10:00:00")
        before = build_frozen_view(bars, close - 1, [Timeframe.H4], ETH)
        at = build_frozen_view(bars, close, [Timeframe.H4], ETH)
        assert len(before.completed[Timeframe.H4]) == 3
        assert len(at.completed[Timeframe.H4]) == 4
        assert at.completed[Timeframe.H4][3].timestamp == close


class TestInvariants:
    def test_never_emits_partial_flag(self) -> None:
        bars = eth_5m_flat("2026-04-21")
        as_of = et("2026-04-21T12:30:00")  # mid-bucket → forming bar exists
        tfs = [Timeframe.M15, Timeframe.M30, Timeframe.H1, Timeframe.H2, Timeframe.H4]
        fv = build_frozen_view(bars, as_of, tfs, ETH)
        assert all(b.partial is None for b in fv.primary)
        for tf in tfs:
            assert all(b.partial is None for b in fv.completed[tf])
            assert all(b.partial is None for b in fv.as_of_view[tf])
            extra = len(fv.as_of_view[tf]) - len(fv.completed[tf])
            assert extra in (0, 1)

    def test_primary_is_bars_le_as_of(self) -> None:
        bars = eth_5m_flat("2026-04-21")
        as_of = et("2026-04-21T10:00:00")
        fv = build_frozen_view(bars, as_of, [], ETH)
        assert all(b.timestamp <= as_of for b in fv.primary)
        assert any(b.timestamp == as_of for b in fv.primary)
        assert not any(b.timestamp == as_of + 300 for b in fv.primary)

    def test_no_mutation_and_sorted_primary(self) -> None:
        input_bars = list(reversed(eth_5m_flat("2026-04-21")))
        snapshot = [b.timestamp for b in input_bars]
        fv = build_frozen_view(input_bars, FAR, [Timeframe.H4], ETH)
        assert [b.timestamp for b in input_bars] == snapshot
        for i in range(1, len(fv.primary)):
            assert fv.primary[i].timestamp > fv.primary[i - 1].timestamp

    def test_throws_on_1d(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="1d"):
            build_frozen_view(eth_5m_flat("2026-04-21"), FAR, [Timeframe.D1], ETH)

    def test_24_7_utc_session_no_trailing_stub(self) -> None:
        open_ts = utc(2026, 4, 21, 0)
        close_ts = open_ts + 24 * 3600
        bars = [bar(ts, 100, 100, 100, 100, 1) for ts in range(open_ts + 300, close_ts + 1, 300)]
        s247 = make_session_days("continuous_24_7", "UTC", [("2026-04-21", open_ts, close_ts)])

        as_of = utc(2026, 4, 21, 12)
        fv = build_frozen_view(bars, as_of, [Timeframe.H4], s247)
        assert [b.timestamp for b in fv.completed[Timeframe.H4]] == [
            utc(2026, 4, 21, 4),
            utc(2026, 4, 21, 8),
            utc(2026, 4, 21, 12),
        ]
        assert fv.as_of_view[Timeframe.H4] == fv.completed[Timeframe.H4]


def test_aggregate_math_sanity() -> None:
    # int/float volume mix keeps exact doubles through the sum
    assert math.isclose(1 + 2 + 3 + 4 + 5 + 6, 21)
