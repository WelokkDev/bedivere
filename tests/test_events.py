"""bedivere.engine.events — total ordering (ts, priority_class, seq)."""

from __future__ import annotations

import random

import pytest

from bedivere.core.types import Timeframe
from bedivere.engine.events import (
    BarEvent,
    CommandEvent,
    Event,
    EventQueue,
    PriorityClass,
    TimerEvent,
)
from tests.helpers import bar


def _bar_event(ts: int) -> BarEvent:
    return BarEvent(ts=ts, symbol="NQ", timeframe=Timeframe.M5, candle=bar(ts, 1, 2, 0.5, 1.5))


def test_priority_classes_are_fixed() -> None:
    # venue/fill < market-data < timer < command — pinned.
    assert (
        PriorityClass.VENUE
        < PriorityClass.MARKET_DATA
        < PriorityClass.TIMER
        < PriorityClass.COMMAND
    )
    assert BarEvent.priority_class == PriorityClass.MARKET_DATA
    assert TimerEvent.priority_class == PriorityClass.TIMER
    assert CommandEvent.priority_class == PriorityClass.COMMAND


def test_bar_event_pins_ts_to_close_stamp() -> None:
    with pytest.raises(ValueError, match="candle.timestamp"):
        BarEvent(ts=101, symbol="NQ", timeframe=Timeframe.M5, candle=bar(100, 1, 2, 0.5, 1.5))


def test_same_instant_arbitration_order() -> None:
    q = EventQueue()
    q.push(CommandEvent(ts=100, name="stop"))
    q.push(TimerEvent(ts=100, name="session-end"))
    q.push(_bar_event(100))
    popped = [q.pop(), q.pop(), q.pop()]
    assert isinstance(popped[0], BarEvent)  # no venue event pushed; data first
    assert isinstance(popped[1], TimerEvent)
    assert isinstance(popped[2], CommandEvent)
    assert q.pop() is None


def test_fifo_within_same_instant_and_class() -> None:
    q = EventQueue()
    timers = [TimerEvent(ts=50, name=f"t{i}") for i in range(20)]
    for t in timers:
        q.push(t)
    assert [q.pop() for _ in timers] == timers


def test_total_order_property_random_push_order() -> None:
    """Any push order pops as the stable sort by (ts, priority_class, push order)."""
    rng = random.Random(20260725)
    for _ in range(50):
        events: list[Event] = []
        for i in range(rng.randrange(1, 120)):
            ts = rng.randrange(0, 8)
            cls = rng.randrange(0, 3)
            if cls == 0:
                events.append(_bar_event(ts))
            elif cls == 1:
                events.append(TimerEvent(ts=ts, name=f"t{i}"))
            else:
                events.append(CommandEvent(ts=ts, name=f"c{i}"))

        q = EventQueue()
        seqs = [q.push(e) for e in events]
        assert seqs == sorted(seqs)  # monotone enqueue counter

        expected = [
            e
            for _, _, _, e in sorted(
                (e.ts, int(e.priority_class), s, e) for s, e in zip(seqs, events, strict=True)
            )
        ]
        drained: list[Event] = []
        while len(q):
            popped = q.pop()
            assert popped is not None
            drained.append(popped)
        assert drained == expected


def test_peek_does_not_remove() -> None:
    q = EventQueue()
    e = _bar_event(7)
    q.push(e)
    assert q.peek() is e
    assert len(q) == 1
    assert q.pop() is e
    assert q.peek() is None
