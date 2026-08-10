"""The FeedAdapter port — where live bars come from.

bedivere ships no market-data connection, and a spec that names one has to
name YOURS. A feed adapter is the small object in between: the composition
builds the `LiveBarStream` (it owns the symbol, timeframe, window, clock and
warm-up history), hands it over, and the adapter pushes closed bars into it
until told to stop.

    adapter.start(stream)   # begin pushing; must not block
    adapter.stop()          # stop pushing and release resources

That is the whole contract. Everything the adapter needs to build itself
arrives as a `FeedContext`, so a factory named in a spec

    "feed": {"factory": "my_pkg.feed:build", "options": {"account": "..."}}

is called as `build(ctx, **options)` and can be written without importing a
single bedivere composition detail.

`ReplayFeed` is the one adapter that ships, and it exists because a shadow
session has to be runnable before you have written a feed: it replays a
`CandleSource` through the SAME LiveBarStream a real feed pushes into, with
real receive stamps and the real end-of-stream behaviour. It is a rehearsal
of the live path, not a backtest — the bars are historical, but every other
moving part is the live one.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from bedivere.core.clock import Clock
from bedivere.core.types import Candle, Timeframe
from bedivere.data.port import CandleSource
from bedivere.streams.live import LiveBarStream


@dataclass(frozen=True, slots=True)
class FeedContext:
    """Everything an adapter needs to know about the run it is feeding.

    `source` is the run's candle source when the spec configured one, so an
    adapter that needs to backfill a reconnect gap has somewhere to read from
    without inventing a second data path.
    """

    symbol: str
    timeframe: Timeframe
    window_start_unix: int
    window_end_unix: int
    clock: Clock
    source: CandleSource | None = None


class FeedAdapter(Protocol):
    def start(self, stream: LiveBarStream) -> None:
        """Begin pushing closed bars. MUST return promptly — the caller's next
        move is to run the engine loop, which is the thing consuming the
        stream. Do the work on your own thread."""
        ...

    def stop(self) -> None:
        """Stop pushing and release the connection. Idempotent; called on
        every exit path including a failed run."""
        ...


class ReplayFeed:
    """Replay a `CandleSource` into a LiveBarStream from a background thread.

    Bars in `(window_start, window_end]` are pushed in order, then the stream
    is CLOSED — a rehearsal that never ends is a rehearsal you cannot put in
    a test. `delay_s` paces the push (0.0 = as fast as the consumer drains).

    Note what is NOT simulated: the clock. Whatever Clock the composition
    injected is the one the run uses, so if you replay yesterday's bars under
    a LiveClock the sim venue will correctly refuse to fill orders "decided"
    a day after the data — that refusal is the model being honest, not a bug.
    Replay a window that is ahead of now, or inject a clock that rides the
    bars (see examples/shadow_session.py).
    """

    def __init__(self, ctx: FeedContext, *, delay_s: float = 0.0) -> None:
        if ctx.source is None:
            raise ValueError(
                "ReplayFeed needs a candle source — give the spec a `data` block, "
                "or name a feed adapter that fetches its own bars"
            )
        self._ctx = ctx
        self._source = ctx.source
        self._delay_s = delay_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, stream: LiveBarStream) -> None:
        if self._thread is not None:
            raise RuntimeError("ReplayFeed.start() called twice — build one per run")
        bars = self._source.candles(
            self._ctx.symbol,
            self._ctx.timeframe,
            self._ctx.window_start_unix,
            self._ctx.window_end_unix,
        )
        self._thread = threading.Thread(
            target=self._push_all,
            args=(stream, bars),
            name="bedivere-replay-feed",
            daemon=True,
        )
        self._thread.start()

    def _push_all(self, stream: LiveBarStream, bars: Sequence[Candle]) -> None:
        try:
            for candle in bars:
                if self._stop.is_set() or not stream.push(candle):
                    return  # asked to stop, or the stream closed under us
                if self._delay_s > 0 and self._stop.wait(self._delay_s):
                    return
        except Exception as e:  # noqa: BLE001 — a dead feeder must not die silently
            sys.stderr.write(f"[feed] replay failed: {e}\n")
        finally:
            # The rehearsal is over: end the stream so the run archives
            # instead of waiting out its window against a feed that is done.
            stream.close()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)


def build_replay_feed(ctx: FeedContext, *, delay_s: float = 0.0) -> FeedAdapter:
    """Spec factory for `ReplayFeed`:

        "feed": {"factory": "bedivere.streams.feed:build_replay_feed",
                 "options": {"delay_s": 0.02}}

    `options` keys are the factory's own keyword-argument names, passed
    through verbatim — they belong to YOUR function, so bedivere does not
    rename them on the way in.
    """
    return ReplayFeed(ctx, delay_s=delay_s)
