"""Event types + the total-ordering doctrine.

Every event carries `(ts, priority_class)`; the queue assigns `seq` — a
monotone enqueue counter — so the triple `(ts, priority_class, seq)` is a
TOTAL order: deterministic arbitration at one instant, FIFO within one
instant+class. Priority classes are fixed:

    venue/fill < market-data < timer < command

Venue-before-data is the Nautilus rule that structurally kills "react to
the bar that filled you" lookahead: fills and acks settle before the
strategy sees the bar that caused them. Timers fire after the data for the
same instant; commands last.

Venue-side events (acks/fills) are defined with the order intents in
`bedivere.engine.intents` — they carry the six `ts_*` stamps. This module
owns the classes, the data/timer/command event shapes, and the queue.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar, Protocol

from bedivere.core.types import Candle, Timeframe


class PriorityClass(IntEnum):
    """Fixed arbitration order for events at the same instant."""

    VENUE = 0
    MARKET_DATA = 1
    TIMER = 2
    COMMAND = 3


class Event(Protocol):
    """Anything the engine loop can consume: an instant + arbitration class.

    `ts` is unix seconds (bars are close-stamped; venue events stamp their
    effective time). `priority_class` is fixed per event TYPE — a ClassVar,
    so it can never vary per instance."""

    priority_class: ClassVar[PriorityClass]

    @property
    def ts(self) -> int: ...


@dataclass(frozen=True, slots=True)
class BarEvent:
    """One CLOSED bar of a stream's timeframe. `ts` is the close-stamp and
    must equal `candle.timestamp` — a mismatch is a construction bug."""

    priority_class: ClassVar[PriorityClass] = PriorityClass.MARKET_DATA

    ts: int
    symbol: str
    timeframe: Timeframe
    candle: Candle
    # Live-feed replay/backfill bars are marked so decision code can skip
    # them (feed hygiene); replay streams always emit False.
    backfill: bool = False
    # When the process RECEIVED this bar (unix ms). Live streams stamp it at
    # arrival — the honest latency measurement; None (replay) means the
    # close instant stands in for it.
    received_at_ms: int | None = None

    def __post_init__(self) -> None:
        if self.ts != self.candle.timestamp:
            raise ValueError(
                f"BarEvent ts {self.ts} != candle.timestamp {self.candle.timestamp}"
            )


@dataclass(frozen=True, slots=True)
class TimerEvent:
    """A scheduled instant (session-end flatten, etc.). Fires after the
    market data for the same instant, per the ordering doctrine."""

    priority_class: ClassVar[PriorityClass] = PriorityClass.TIMER

    ts: int
    name: str


@dataclass(frozen=True, slots=True)
class CommandEvent:
    """External control (pause / flatten / stop) — arbitrated last at any
    instant."""

    priority_class: ClassVar[PriorityClass] = PriorityClass.COMMAND

    ts: int
    name: str


class EventQueue:
    """Total-order min-queue over `(ts, priority_class, seq)`.

    `seq` is assigned at push — a monotone counter, never reused — so pops
    are deterministic for any push order and FIFO within one
    (instant, class). The seq tiebreak also means the heap never compares
    two Event payloads."""

    __slots__ = ("_heap", "_seq")

    def __init__(self) -> None:
        self._heap: list[tuple[int, int, int, Event]] = []
        self._seq = 0

    def push(self, event: Event) -> int:
        """Enqueue and return the assigned seq."""
        seq = self._seq
        self._seq += 1
        heapq.heappush(self._heap, (event.ts, int(event.priority_class), seq, event))
        return seq

    def pop(self) -> Event | None:
        """Remove and return the next event in total order, or None."""
        if not self._heap:
            return None
        return heapq.heappop(self._heap)[3]

    def peek(self) -> Event | None:
        if not self._heap:
            return None
        return self._heap[0][3]

    def __len__(self) -> int:
        return len(self._heap)
