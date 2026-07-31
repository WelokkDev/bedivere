"""Resolved session-days as data — the engine's ONLY session surface.

Session POLICY (open/close times, DST, the holiday calendar) is resolved
UPSTREAM — by bedivere.sessions or by your own calendar code — into plain
rows the whole engine consumes:

    {"template": "cme_us_index_futures_eth",
     "timezone": "America/New_York",
     "days": [{"label": "2026-07-21",
               "startUnix": 1784930400,
               "endUnix": 1785013200}, ...]}

This module is the receiving dock: parse those rows and answer containment
lookups over them. It never COMPUTES a boundary — boundary derivation lives
solely in bedivere.sessions; if engine code elsewhere appears to derive a
session boundary from wall-clock rules, that is a defect by definition.

Also home to the session-grid arithmetic (expected close-stamps for a
resolved day) — pure arithmetic on received bounds, sibling to the
aggregator's bucket flooring.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from bedivere.core.types import Timeframe

SessionDayResolver = Callable[[int], "SessionDay | None"]


@dataclass(frozen=True, slots=True)
class SessionDay:
    """One resolved session-day: (start_unix, end_unix] — start exclusive
    (the open instant), end inclusive (the close-stamp of the last
    in-session bar)."""

    label: str
    start_unix: int
    end_unix: int


@dataclass(frozen=True, slots=True)
class SessionDays:
    """The handed-over day set for one (template, window). `days` is
    ascending by start_unix and pairwise disjoint — parse_session_days
    enforces both."""

    template: str
    timezone: str
    days: tuple[SessionDay, ...]

    def day_containing(self, unix_sec: int) -> SessionDay | None:
        """The day with start_unix < unix_sec <= end_unix, or None (bar in
        a maintenance break / weekend gap / outside the handed window)."""
        # Rightmost day with start_unix < unix_sec.
        i = bisect_left([d.start_unix for d in self.days], unix_sec) - 1
        if i < 0:
            return None
        d = self.days[i]
        return d if unix_sec <= d.end_unix else None

    def make_resolver(self) -> SessionDayResolver:
        """One-entry-memo resolver for per-bar loops (consecutive bars almost
        always share a session-day). Sound because days are disjoint."""
        starts = [d.start_unix for d in self.days]
        last: SessionDay | None = None

        def resolve(unix_sec: int) -> SessionDay | None:
            nonlocal last
            if last is not None and last.start_unix < unix_sec <= last.end_unix:
                return last
            i = bisect_left(starts, unix_sec) - 1
            if i < 0:
                return None
            d = self.days[i]
            if unix_sec <= d.end_unix:
                last = d
                return d
            return None

        return resolve

    def overlapping(self, from_unix: int, to_unix: int) -> list[SessionDay]:
        """Days whose (start, end] interval overlaps [from_unix, to_unix]:
        start < to and end >= from."""
        return [d for d in self.days if d.start_unix < to_unix and d.end_unix >= from_unix]


def parse_session_days(payload: object) -> SessionDays:
    """Parse and validate a resolved session payload. Loud on malformed
    input — a silently-mended day list would defeat the single-source-of-
    policy contract."""
    if not isinstance(payload, dict):
        raise ValueError("session payload must be a JSON object")
    payload_d = cast(dict[str, object], payload)
    template = payload_d.get("template")
    timezone = payload_d.get("timezone")
    days_raw = payload_d.get("days")
    if not isinstance(template, str) or len(template) == 0:
        raise ValueError("session.template must be a non-empty string")
    if not isinstance(timezone, str) or len(timezone) == 0:
        raise ValueError("session.timezone must be a non-empty string")
    if not isinstance(days_raw, list):
        raise ValueError("session.days must be an array")

    days: list[SessionDay] = []
    for i, entry in enumerate(cast(list[object], days_raw)):
        if not isinstance(entry, dict):
            raise ValueError(f"session.days[{i}] must be an object")
        entry_d = cast(dict[str, object], entry)
        label = entry_d.get("label")
        start = entry_d.get("startUnix")
        end = entry_d.get("endUnix")
        if not isinstance(label, str) or len(label) == 0:
            raise ValueError(f"session.days[{i}].label must be a non-empty string")
        if isinstance(start, bool) or not isinstance(start, int):
            raise ValueError(f"session.days[{i}].startUnix must be an integer")
        if isinstance(end, bool) or not isinstance(end, int):
            raise ValueError(f"session.days[{i}].endUnix must be an integer")
        if end <= start:
            raise ValueError(f"session.days[{i}] has endUnix <= startUnix")
        days.append(SessionDay(label=label, start_unix=start, end_unix=end))

    days.sort(key=lambda d: d.start_unix)
    for a, b in zip(days, days[1:], strict=False):
        if b.start_unix < a.end_unix:
            raise ValueError(
                f'session.days overlap: "{a.label}" (ends {a.end_unix}) and "{b.label}" (starts {b.start_unix})'
            )

    return SessionDays(template=template, timezone=timezone, days=tuple(days))


# ---------- session-grid arithmetic (ex core/cache/validator.py) ----------


def expected_bar_count(sd: SessionDay, tf: Timeframe) -> int:
    """Expected bar count for one (session-day, timeframe): full periods
    within (startUnix, endUnix] plus a trailing stub when the period doesn't
    evenly divide the session duration."""
    period = tf.period_seconds  # raises on "1d"
    duration = sd.end_unix - sd.start_unix
    return duration // period + (1 if duration % period != 0 else 0)


def expected_close_stamps(sd: SessionDay, tf: Timeframe) -> list[int]:
    """The close-stamps session geometry demands for one (session-day, tf):
    full periods from the session start, plus a stub at the close when the
    period doesn't divide the duration."""
    period_seconds = tf.period_seconds
    duration = sd.end_unix - sd.start_unix
    has_stub = duration % period_seconds != 0
    full_bar_count = expected_bar_count(sd, tf) - (1 if has_stub else 0)
    expected = [sd.start_unix + i * period_seconds for i in range(1, full_bar_count + 1)]
    if has_stub:
        expected.append(sd.end_unix)
    return expected
