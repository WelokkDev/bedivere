"""LiveBarStream — the push-fed BarStream for live and shadow sessions.

bedivere still has no data-fetching layer: YOUR feed adapter (a websocket
callback, a broker SDK thread, a file tailer) calls `push()` as closed bars
arrive, and the engine loop consumes the stream exactly as it consumes a
replay. The stream owns feed hygiene so the loop doesn't have to:

  - `history` bars are yielded first (warm-up, replay-style), then pushed
    bars in arrival order.
  - Every pushed bar is stamped `received_at_ms` from the injected clock at
    push time — the honest close-to-receipt latency measurement.
  - Duplicates and regressions are DROPPED and counted, not raised: live
    feeds redeliver, and a redelivery is a fact about the feed, not a bug
    in the engine. (Replay keeps its strict raise — disorder in a file you
    control IS a bug.)
  - Bars your adapter marks `backfill=True` (catch-up after a reconnect)
    advance engine state but never reach the strategy.
  - The stream ends when a bar reaches `window_end_unix`, when `close()` is
    called, or when the clock passes the window end with a silent feed
    (one bar period + slack of grace) — so a dead feed ends the run instead
    of hanging it.

The injected Clock is the only time source (LiveClock in production, a fake
in tests) — this module never reads the wall directly.
"""

from __future__ import annotations

import queue
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

from bedivere.core.clock import Clock
from bedivere.core.types import Candle, Timeframe
from bedivere.engine.events import BarEvent

# Silent-feed grace past the window end: one bar period + this slack.
_END_GRACE_SLACK_S = 120

_CLOSE = object()


def _no_status(_text: str) -> None:
    return None


@dataclass(frozen=True, slots=True)
class _Pushed:
    candle: Candle
    backfill: bool
    received_at_ms: int


class LiveBarStream:
    """Single-use iterator of BarEvents for one (symbol, timeframe), fed by
    `push()` from any thread."""

    def __init__(
        self,
        *,
        symbol: str,
        timeframe: Timeframe,
        clock: Clock,
        window_end_unix: int,
        history: Sequence[Candle] = (),
        max_queue: int = 4096,
        idle_poll_s: float = 1.0,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        for prev, cur in zip(history, history[1:], strict=False):
            if cur.timestamp <= prev.timestamp:
                raise ValueError(
                    f"LiveBarStream: history bar {cur.timestamp} is not after {prev.timestamp} — "
                    "history is data you prepared; sort it"
                )
        self._symbol = symbol
        self._tf = timeframe
        self._clock = clock
        self._window_end = window_end_unix
        self._history = list(history)
        self._q: queue.Queue[object] = queue.Queue(maxsize=max_queue)
        self._idle_poll_s = idle_poll_s
        self._status = on_status or _no_status
        self._closed = False
        self._consumed = False
        # Observability counters (surfaced in the live result envelope).
        self.history_bars = 0
        self.live_bars = 0
        self.backfill_bars = 0
        self.duplicates = 0
        self.dropped = 0

    # ---------- feed side (any thread) ----------

    def push(self, candle: Candle, *, backfill: bool = False) -> bool:
        """Hand one closed bar to the stream; returns False when it was
        dropped (stream closed, or queue full — counted and loud, because a
        stalled consumer must not silently lose bars)."""
        if self._closed:
            return False
        item = _Pushed(candle=candle, backfill=backfill, received_at_ms=self._clock.now_ms())
        try:
            self._q.put_nowait(item)
            return True
        except queue.Full:
            self.dropped += 1
            self._status(f"queue full — dropped bar {candle.timestamp}")
            return False

    def close(self) -> None:
        """Graceful end from any thread: the iterator finishes after the
        bars already queued. Idempotent."""
        if self._closed:
            return
        self._closed = True
        try:
            self._q.put_nowait(_CLOSE)
        except queue.Full:
            # The consumer will still see _closed on its next idle poll.
            self._status("queue full at close — stream will end on the next poll")

    # ---------- engine side ----------

    def __iter__(self) -> Iterator[BarEvent]:
        if self._consumed:
            raise RuntimeError("LiveBarStream is single-use — construct a new one per run")
        self._consumed = True

        last = 0
        for candle in self._history:
            self.history_bars += 1
            last = candle.timestamp
            yield self._event(candle, backfill=False, received_at_ms=None)

        end_grace_s = self._tf.period_seconds + _END_GRACE_SLACK_S

        while last < self._window_end:
            try:
                item = self._q.get(timeout=self._idle_poll_s)
            except queue.Empty:
                if self._closed:
                    self._status("stream closed — ending after queued bars drained")
                    return
                if self._clock.now_unix() > self._window_end + end_grace_s:
                    self._status("window ended with a silent feed — stream closed")
                    return
                continue

            if item is _CLOSE:
                self._status("stream closed — ending run")
                return
            assert isinstance(item, _Pushed)
            ts = item.candle.timestamp
            if ts <= last:
                self.duplicates += 1
                continue
            if ts > self._window_end:
                self._status(f"bar {ts} is past the window end — ending run")
                return
            if item.backfill:
                self.backfill_bars += 1
            else:
                self.live_bars += 1
            last = ts
            yield self._event(
                item.candle, backfill=item.backfill, received_at_ms=item.received_at_ms
            )

    def _event(self, candle: Candle, *, backfill: bool, received_at_ms: int | None) -> BarEvent:
        return BarEvent(
            ts=candle.timestamp,
            symbol=self._symbol,
            timeframe=self._tf,
            candle=candle,
            backfill=backfill,
            received_at_ms=received_at_ms,
        )
