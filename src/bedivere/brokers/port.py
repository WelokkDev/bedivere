"""The Broker port — one execution path, swapped client.

`submit_bracket` is deliberately the ONLY entry API a strategy sees: the
two-step reality (entry order → fill → protection, with a naked window) is
implemented INSIDE each broker identically, so the naked-window risk is
modelled in backtest rather than discovered live.

Every method is a COMMAND: its outcome arrives as `OrderEvent`s via `drain`,
never as a return value. That is not style — a live adapter must not block
the engine loop on a venue reply, and a broker that answered synchronously
would model a latency of zero.

Implementations: `SimBroker` (backtest, bedivere.brokers.sim); a live
adapter implements this same port against a real venue (not included).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from bedivere.core.clock import Clock
from bedivere.core.pricing import InstrumentSpec
from bedivere.engine.events import BarEvent
from bedivere.engine.intents import BracketIntent, OrderEvent


@dataclass(frozen=True, slots=True)
class BrokerContext:
    """What a spec-named broker factory is handed: `build(ctx, **options)`.

    The account the adapter points at is deliberately NOT here — the engine
    has no way to know one and no business declaring it.
    """

    instrument: InstrumentSpec
    symbol: str
    clock: Clock
    mode: str  # the operator's label for this run ("paper", "funded", ...)


class Broker(Protocol):
    def preflight(self) -> str | None:
        """A one-line reason why this run must NOT start, or None when the
        venue is clean. `run_live` calls it before the first bar.

        A real adapter checks for an existing open position on the symbol,
        working orders orphaned by a dead run, and a stale position feed
        (which would invalidate the other two checks). `SimBroker` returns
        None — an in-process venue starts empty by construction.

        FAIL CLOSED: a snapshot that could not be fetched, timed out, or
        would not parse must return a reason exactly as loudly as a dirty
        one. "I could not tell" and "it is clean" are opposite answers, and
        an adapter that returns None on a failed query has silently turned
        the first into the second.
        """
        ...

    def submit_bracket(self, intent: BracketIntent) -> int:
        """Accept a bracket; returns the id venue events reference. Stamps
        ts_submitted/ts_acked per the broker's model. The id is a handle, NOT
        an acceptance — a refused bracket comes back as `cancelled` with a
        reason on a later `drain`."""
        ...

    def change(self, bracket_id: int, *, stop_ticks: int | None = None, target_ticks: int | None = None) -> None:
        """Amend the protective legs. On an OPEN bracket they are working
        orders and this is a real amend; on a PENDING one they do not exist
        yet, so this sets the levels it will arm on fill — what a strategy
        tracking a still-forming extreme needs, and what a real venue does
        too (there is nothing to send)."""
        ...

    def cancel(self, bracket_id: int) -> None:
        """Cancel a not-yet-open bracket. Confirmed by a `cancelled` event on
        a later drain, never by returning."""
        ...

    def flatten(self, symbol: str) -> None:
        """Market out of any open position and cancel working protection."""
        ...

    def drain(self, event: BarEvent | None) -> list[OrderEvent]:
        """Advance the venue with one market-data event; returns the venue
        events that settled, in deterministic order.

        `event=None` is the SETTLEMENT POLL: deliver what is queued, advance
        nothing, stamp nothing. The last thing a run does — flatten at
        shutdown — happens after the stream is exhausted, and without a poll
        the events it produces never reach the portfolio, so the run's own
        final trade goes unrecorded.
        """
        ...
