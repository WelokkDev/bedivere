"""One deterministic clock, injected everywhere.

`Clock` is a protocol with NO default: any component that needs the current
instant takes a Clock argument. `ReplayClock` is the backtest clock — time
is whatever the event loop last advanced to, so a replayed run is
bit-reproducible. `LiveClock` reads the wall and is the ONLY place in
bedivere allowed to (tests/test_wall_clock_gate.py enforces the ban
everywhere else).

Milliseconds exist for the latency stamps: backtest stamps are modelled
(ReplayClock has whole-second resolution, ms = s * 1000); live stamps are
measured (LiveClock has real ms resolution). Same field names either way.
"""

from __future__ import annotations

import time as _time  # allow-listed wall-clock access: LiveClock internals only
from typing import Protocol


class Clock(Protocol):
    """The one time source. Implementations: ReplayClock, LiveClock."""

    def now_unix(self) -> int:
        """Current instant, whole unix seconds."""
        ...

    def now_ms(self) -> int:
        """Current instant, unix milliseconds (latency stamps)."""
        ...


class ReplayClock:
    """Deterministic clock driven by the event loop: `advance(ts)` on every
    event, monotone non-decreasing. A regression means the stream violated
    the total ordering — that is a bug upstream, so it raises instead of
    clamping."""

    __slots__ = ("_now_unix",)

    def __init__(self, start_unix: int) -> None:
        self._now_unix = start_unix

    def advance(self, ts_unix: int) -> None:
        if ts_unix < self._now_unix:
            raise ValueError(
                f"ReplayClock regression: advance({ts_unix}) after {self._now_unix} — the event stream is out of order"
            )
        self._now_unix = ts_unix

    def now_unix(self) -> int:
        return self._now_unix

    def now_ms(self) -> int:
        return self._now_unix * 1000


class LiveClock:
    """Wall-clock time for live runs. The only wall-clock reader in bedivere."""

    __slots__ = ()

    def now_unix(self) -> int:
        return int(_time.time())

    def now_ms(self) -> int:
        return _time.time_ns() // 1_000_000
