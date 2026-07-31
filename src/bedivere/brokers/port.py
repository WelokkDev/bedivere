"""The Broker port — one execution path, swapped client.

`submit_bracket` is deliberately the ONLY entry API a strategy sees: the
two-step reality (entry order → fill → protection, with a naked window) is
implemented INSIDE each broker identically, so the naked-window risk is
modelled in backtest rather than discovered live. `drain` delivers venue
events into the loop BEFORE the strategy sees the bar that caused them
(venue-first).

Implementations: `SimBroker` (backtest, bedivere.brokers.sim); a live
adapter implements this same port against a real venue (not included).
"""

from __future__ import annotations

from typing import Protocol

from bedivere.engine.events import BarEvent
from bedivere.engine.intents import BracketIntent, OrderEvent


class Broker(Protocol):
    def submit_bracket(self, intent: BracketIntent) -> int:
        """Accept a bracket; returns the bracket id (venue events reference
        it). Stamps ts_submitted/ts_acked per the broker's model."""
        ...

    def change(self, bracket_id: int, *, stop_ticks: int | None = None, target_ticks: int | None = None) -> None:
        """Amend working protection legs."""
        ...

    def cancel(self, bracket_id: int) -> None:
        """Cancel a not-yet-open bracket (pending entry)."""
        ...

    def flatten(self, symbol: str) -> None:
        """Market out of any open position and cancel working protection."""
        ...

    def drain(self, event: BarEvent) -> list[OrderEvent]:
        """Advance the venue with one market-data event; returns the venue
        events that settled, in deterministic order."""
        ...
