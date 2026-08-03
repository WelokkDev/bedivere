"""The Notifier port — transport-agnostic outbound messages.

Two rules bind every implementation:

  - send() must never block or raise: the engine loop sits upstream, and a
    slow webhook must not stall a trading decision.
  - Delivery failures are LOUD but contained: dropped messages are counted
    and reported to stderr, never silently discarded and never re-raised
    into the loop.

close() flushes best-effort and is idempotent.
"""

from __future__ import annotations

import sys
from typing import Protocol


class Notifier(Protocol):
    def send(self, text: str) -> None:
        """Fire-and-forget. Must return promptly and never raise."""
        ...

    def close(self) -> None:
        """Flush what can be flushed, release resources. Idempotent."""
        ...


class NullNotifier:
    """The silent default: swallows everything."""

    def send(self, text: str) -> None:  # noqa: ARG002 — port shape
        return

    def close(self) -> None:
        return


class ConsoleNotifier:
    """Stderr sink (stdout is reserved for result JSON)."""

    def __init__(self, prefix: str = "[notify]") -> None:
        self._prefix = prefix

    def send(self, text: str) -> None:
        try:
            sys.stderr.write(f"{self._prefix} {text}\n")
        except Exception:  # noqa: BLE001, S110 — a broken stderr must not kill the loop
            pass

    def close(self) -> None:
        return
