"""Journal event → phone-sized message line.

Per-kind templates for the venue moments that matter, plus a generic
fallback so ANY strategy-emitted kind renders immediately. Formatters are
total: a malformed event degrades to the fallback line, never an exception.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

# Context keys that carry no message value; the fallback line skips them.
_NOISY_KEYS = frozenset({"kind", "ts", "symbol", "stamps", "paramsHash", "mode"})


def _s(e: dict[str, object], key: str, default: str = "?") -> str:
    v = e.get(key)
    return default if v is None else str(v)


def _when(e: dict[str, object]) -> str:
    """UTC wall-clock minute of the event's `ts` (rendering a GIVEN instant
    — not a wall-clock read)."""
    ts = e.get("ts")
    if isinstance(ts, int) and not isinstance(ts, bool):
        return datetime.fromtimestamp(ts, UTC).strftime("%H:%M") + "Z"
    return "?"


def _ticks(e: dict[str, object], key: str) -> str:
    v = e.get(key)
    if isinstance(v, int) and not isinstance(v, bool):
        return f"{v}t"
    return "?"


def _fmt_entry_fill(e: dict[str, object]) -> str:
    return f"▶ {_s(e, 'symbol')} ENTRY {_when(e)} — {_s(e, 'direction')} {_s(e, 'qty')} @ {_ticks(e, 'priceTicks')}"


def _fmt_protection(e: dict[str, object]) -> str:
    return (
        f"🛡 {_s(e, 'symbol')} BRACKET {_when(e)} — "
        f"stop {_ticks(e, 'stopTicks')} · target {_ticks(e, 'targetTicks')}"
    )


def _fmt_exit(emoji: str, label: str) -> Callable[[dict[str, object]], str]:
    def fmt(e: dict[str, object]) -> str:
        ambiguous = " · ambiguous bar" if e.get("ambiguous") else ""
        return f"{emoji} {label} {_s(e, 'symbol')} {_when(e)} @ {_ticks(e, 'priceTicks')}{ambiguous}"

    return fmt


_TEMPLATES: dict[str, Callable[[dict[str, object]], str]] = {
    "entry_fill": _fmt_entry_fill,
    "protection_placed": _fmt_protection,
    "stop_fill": _fmt_exit("🛑", "STOP"),
    "target_fill": _fmt_exit("✅", "TARGET"),
    "flatten_fill": _fmt_exit("⏹", "FLATTEN"),
}


def _fallback(e: dict[str, object]) -> str:
    parts = [
        f"{k}={e[k]}"
        for k in sorted(e)
        if k not in _NOISY_KEYS and not isinstance(e[k], dict | list)
    ]
    detail = " · ".join(parts) if parts else "(no detail)"
    symbol = e.get("symbol")
    head = f"ℹ {symbol} " if symbol is not None else "ℹ "
    return f"{head}{_s(e, 'kind')} {_when(e)} — {detail}"


def format_event(event: dict[str, object]) -> str:
    """One message line for a context-merged journal event. Total: a
    malformed event degrades to the fallback line, never an exception.
    Venue events get compact templates (tick-denominated — the journal is
    integer-exact); every strategy-emitted kind renders via the fallback,
    so alerting on a brand-new kind needs no formatter work. Pass your own
    `formatter` to NotificationRouter for richer, price-denominated text."""
    kind = event.get("kind")
    template = _TEMPLATES.get(kind) if isinstance(kind, str) else None
    if template is not None:
        try:
            return template(event)
        except Exception:  # noqa: BLE001 — degrade, never raise
            pass
    try:
        return _fallback(event)
    except Exception:  # noqa: BLE001
        return f"ℹ journal event {kind!r}"
