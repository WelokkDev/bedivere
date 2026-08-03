"""LiveBarStream — push-fed hygiene: order, duplicates, stamps, endings."""

from __future__ import annotations

import threading

import pytest

from bedivere.core.types import Timeframe
from bedivere.streams.live import LiveBarStream
from tests.helpers import bar


class FakeClock:
    """Settable clock for live tests (no wall reads anywhere). `advance` is
    the loop's duck-typed hook — during a run, time follows the bars, the
    same way ReplayClock does."""

    def __init__(self, start_unix: int) -> None:
        self.unix = start_unix

    def advance(self, ts_unix: int) -> None:
        self.unix = ts_unix

    def now_unix(self) -> int:
        return self.unix

    def now_ms(self) -> int:
        return self.unix * 1000


T0 = 1_784_930_400


def _stream(clock: FakeClock, *, end: int = T0 + 3600, history: int = 0) -> LiveBarStream:
    hist = [bar(T0 + i * 300, 1, 2, 0.5, 1.5) for i in range(1, history + 1)]
    return LiveBarStream(
        symbol="DEMO",
        timeframe=Timeframe.M5,
        clock=clock,
        window_end_unix=end,
        history=hist,
        idle_poll_s=0.01,
    )


def test_history_then_pushed_bars_with_receive_stamps() -> None:
    clock = FakeClock(T0)
    s = _stream(clock, history=2)
    clock.unix = T0 + 901  # "now" when the live bar arrives
    s.push(bar(T0 + 900, 1, 2, 0.5, 1.5))
    s.close()
    events = list(s)
    assert [e.ts for e in events] == [T0 + 300, T0 + 600, T0 + 900]
    # History carries no receive stamp (replay semantics)...
    assert events[0].received_at_ms is None and events[1].received_at_ms is None
    # ...the pushed bar is stamped at PUSH time, not at close time.
    assert events[2].received_at_ms == (T0 + 901) * 1000
    assert s.history_bars == 2 and s.live_bars == 1


def test_duplicates_and_regressions_dropped_and_counted() -> None:
    clock = FakeClock(T0)
    s = _stream(clock, history=1)  # last = T0+300
    s.push(bar(T0 + 600, 1, 2, 0.5, 1.5))
    s.push(bar(T0 + 600, 1, 2, 0.5, 1.5))  # redelivery
    s.push(bar(T0 + 300, 1, 2, 0.5, 1.5))  # regression
    s.close()
    events = list(s)
    assert [e.ts for e in events] == [T0 + 300, T0 + 600]
    assert s.duplicates == 2


def test_backfill_flag_rides_through() -> None:
    clock = FakeClock(T0)
    s = _stream(clock)
    s.push(bar(T0 + 300, 1, 2, 0.5, 1.5), backfill=True)
    s.push(bar(T0 + 600, 1, 2, 0.5, 1.5))
    s.close()
    events = list(s)
    assert [e.backfill for e in events] == [True, False]
    assert s.backfill_bars == 1 and s.live_bars == 1


def test_ends_when_window_end_reached() -> None:
    clock = FakeClock(T0)
    s = _stream(clock, end=T0 + 600)
    s.push(bar(T0 + 300, 1, 2, 0.5, 1.5))
    s.push(bar(T0 + 600, 1, 2, 0.5, 1.5))
    s.push(bar(T0 + 900, 1, 2, 0.5, 1.5))  # never consumed; ends at end
    events = list(s)
    assert [e.ts for e in events] == [T0 + 300, T0 + 600]


def test_bar_past_window_end_closes_without_yielding_it() -> None:
    clock = FakeClock(T0)
    s = _stream(clock, end=T0 + 450)
    s.push(bar(T0 + 300, 1, 2, 0.5, 1.5))
    s.push(bar(T0 + 900, 1, 2, 0.5, 1.5))  # beyond the end
    events = list(s)
    assert [e.ts for e in events] == [T0 + 300]


def test_silent_feed_past_grace_ends_the_stream() -> None:
    clock = FakeClock(T0)
    s = _stream(clock, end=T0 + 300)
    # Nothing ever pushed; clock is already past end + period + slack.
    clock.unix = T0 + 300 + 300 + 121
    assert list(s) == []


def test_close_from_another_thread_ends_a_blocked_iterator() -> None:
    clock = FakeClock(T0)
    s = _stream(clock)
    out: list[int] = []

    def consume() -> None:
        for e in s:
            out.append(e.ts)

    t = threading.Thread(target=consume)
    t.start()
    s.push(bar(T0 + 300, 1, 2, 0.5, 1.5))
    s.close()
    t.join(timeout=5)
    assert not t.is_alive()
    assert out == [T0 + 300]
    assert s.push(bar(T0 + 600, 1, 2, 0.5, 1.5)) is False  # closed refuses


def test_single_use_and_history_validation() -> None:
    clock = FakeClock(T0)
    s = _stream(clock)
    s.close()
    list(s)
    with pytest.raises(RuntimeError, match="single-use"):
        list(s)
    with pytest.raises(ValueError, match="not after"):
        LiveBarStream(
            symbol="DEMO",
            timeframe=Timeframe.M5,
            clock=clock,
            window_end_unix=T0 + 600,
            history=[bar(T0 + 600, 1, 2, 0.5, 1.5), bar(T0 + 300, 1, 2, 0.5, 1.5)],
        )
