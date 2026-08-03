"""Journal → Notifier routing: WHICH steps message you is config here.

The strategy narrates every step through its journal; a NotificationRouter
installed as a journal sink forwards the configured kinds to a Notifier —
so re-pointing alerts (console today, a Discord channel tomorrow) is a
config edit, never a strategy change.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from typing import get_args

from bedivere.engine.intents import OrderEventKind
from bedivere.notify.format import format_event
from bedivere.notify.port import Notifier

# The venue-event kinds the ENGINE journals automatically. Strategy kinds
# are whatever your strategy emits — the engine cannot enumerate them.
VENUE_KINDS: frozenset[str] = frozenset(get_args(OrderEventKind))

DEFAULT_KINDS: frozenset[str] = frozenset(
    {"entry_fill", "stop_fill", "target_fill", "flatten_fill"}
)


class NotificationRouter:
    """Callable journal sink: filter by kind, format, send. Never raises
    into the caller — the engine loop is upstream.

    Pass `known_kinds` (your strategy's full kind vocabulary) to get typo
    protection: a requested kind outside VENUE_KINDS ∪ known_kinds raises at
    construction, so a misspelled kind cannot silently alert on nothing.
    Without it, any kind is accepted."""

    def __init__(
        self,
        kinds: Iterable[str],
        notifier: Notifier,
        *,
        formatter: Callable[[dict[str, object]], str] | None = None,
        known_kinds: Iterable[str] | None = None,
    ) -> None:
        wanted = frozenset(kinds)
        if known_kinds is not None:
            known = VENUE_KINDS | frozenset(known_kinds)
            unknown = sorted(wanted - known)
            if unknown:
                raise ValueError(
                    f"unknown notify kind(s) {unknown} — known kinds: {sorted(known)}"
                )
        self.kinds = wanted
        self._notifier = notifier
        self._format = formatter or format_event
        self.routed = 0

    def __call__(self, event: dict[str, object]) -> None:
        kind = event.get("kind")
        if not isinstance(kind, str) or kind not in self.kinds:
            return
        try:
            self._notifier.send(self._format(event))
            self.routed += 1
        except Exception as e:  # noqa: BLE001 — notification may not break the loop
            sys.stderr.write(f"[notify] routing {kind} failed: {e}\n")
