"""Session-day derivation — the ONE place boundaries are computed.

Everything else in bedivere consumes already-resolved `SessionDays` rows
(bedivere.core.session_days). This module is where those rows may come
from: a `DailySessionSpec` (local open/close wall-clock times in an IANA
timezone, close-day weekdays, holiday skip list) expanded over an explicit
date range. If you have your own calendar source, skip this module entirely
and hand rows to `parse_session_days` / `SessionDays` yourself.

DST is handled by zoneinfo: each boundary is a wall-clock instant resolved
in the spec's timezone, so a session spanning a transition simply has a
different duration in seconds — the engine's bucket math is built for that
(trailing stub buckets at the session close). Boundary times that fall
inside an ambiguous fall-back hour resolve with PEP 495 fold=0 (the first
occurrence); typical exchange boundaries (17:00/18:00) never land there.

No wall-clock reads here — date ranges are explicit inputs, never
"today". (tests/test_wall_clock_gate.py holds this module to that.)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from bedivere.core.session_days import SessionDay, SessionDays

_WEEKDAY_NAMES = "Mon Tue Wed Thu Fri Sat Sun".split()


@dataclass(frozen=True, slots=True)
class DailySessionSpec:
    """One session per close-date. `open_time`/`close_time` are "HH:MM"
    local wall-clock in `timezone`; when open_time >= close_time the session
    opens on the PREVIOUS calendar day (e.g. CME equity-index futures open
    18:00 the prior evening and close 17:00). `close_weekdays` are the
    weekdays (Monday=0) a session may CLOSE on — the open day is free to be
    a non-close day (a Monday close opens on Sunday)."""

    timezone: str
    open_time: str
    close_time: str
    close_weekdays: tuple[int, ...] = (0, 1, 2, 3, 4)
    holidays: frozenset[str] = frozenset()  # "YYYY-MM-DD" close-dates to skip
    template: str = "custom_daily"

    def __post_init__(self) -> None:
        _parse_hhmm(self.open_time, "open_time")
        _parse_hhmm(self.close_time, "close_time")
        for wd in self.close_weekdays:
            if not 0 <= wd <= 6:
                raise ValueError(f"close_weekdays entries must be 0..6 (Monday=0), got {wd}")
        if len(set(self.close_weekdays)) != len(self.close_weekdays):
            raise ValueError("close_weekdays must not repeat")


def build_session_days(
    spec: DailySessionSpec,
    first_close_date: str,
    last_close_date: str,
) -> SessionDays:
    """Expand a spec into resolved SessionDays rows for every eligible
    close-date in [first_close_date, last_close_date] (inclusive, both
    "YYYY-MM-DD"). Weekends-by-spec and listed holidays are skipped. Raises
    if any produced day is empty or overlaps its neighbor — a spec that
    yields impossible sessions must fail loudly, not truncate quietly."""
    first = _parse_date(first_close_date, "first_close_date")
    last = _parse_date(last_close_date, "last_close_date")
    if last < first:
        raise ValueError(
            f"last_close_date {last_close_date} is before first_close_date {first_close_date}"
        )
    zone = ZoneInfo(spec.timezone)
    open_h, open_m = _parse_hhmm(spec.open_time, "open_time")
    close_h, close_m = _parse_hhmm(spec.close_time, "close_time")
    opens_previous_day = (open_h, open_m) >= (close_h, close_m)

    days: list[SessionDay] = []
    d = first
    while d <= last:
        label = d.isoformat()
        if d.weekday() in spec.close_weekdays and label not in spec.holidays:
            open_day = d - timedelta(days=1) if opens_previous_day else d
            start = _local_unix(open_day, open_h, open_m, zone)
            end = _local_unix(d, close_h, close_m, zone)
            if end <= start:
                raise ValueError(
                    f"session {label}: close {end} is not after open {start} — check open_time/close_time"
                )
            if days and start < days[-1].end_unix:
                raise ValueError(
                    f'sessions overlap: "{days[-1].label}" ends {days[-1].end_unix}, "{label}" starts {start}'
                )
            days.append(SessionDay(label=label, start_unix=start, end_unix=end))
        d += timedelta(days=1)

    if not days:
        raise ValueError(
            f"no eligible close-dates in [{first_close_date}, {last_close_date}] for "
            f"close_weekdays={tuple(_WEEKDAY_NAMES[w] for w in spec.close_weekdays)} and the given holidays"
        )
    return SessionDays(template=spec.template, timezone=spec.timezone, days=tuple(days))


def cme_futures_sessions(
    first_close_date: str,
    last_close_date: str,
    holidays: frozenset[str] | set[str] = frozenset(),
) -> SessionDays:
    """CME US equity-index futures ETH: opens 18:00 America/New_York the
    prior evening, closes 17:00, Monday–Friday close-dates (a Monday session
    opens Sunday evening). Exchange holidays are NOT built in — pass the
    close-dates you want skipped; a session the exchange shortened or
    cancelled that you didn't list will simply have no bars, and the warm-up
    gate / your own bar counts are the honest check."""
    spec = DailySessionSpec(
        timezone="America/New_York",
        open_time="18:00",
        close_time="17:00",
        close_weekdays=(0, 1, 2, 3, 4),
        holidays=frozenset(holidays),
        template="cme_us_index_futures_eth",
    )
    return build_session_days(spec, first_close_date, last_close_date)


# ---------- parsing / resolution helpers ----------


def _parse_hhmm(text: str, what: str) -> tuple[int, int]:
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(f'{what} must be "HH:MM" 24-hour, got {text!r}')
    try:
        h, m = int(parts[0]), int(parts[1])
    except ValueError as e:
        raise ValueError(f'{what} must be "HH:MM" 24-hour, got {text!r}') from e
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"{what} out of range: {text!r}")
    return h, m


def _parse_date(text: str, what: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError as e:
        raise ValueError(f'{what} must be "YYYY-MM-DD", got {text!r}') from e


def _local_unix(d: date, hour: int, minute: int, zone: ZoneInfo) -> int:
    """The unix instant of a local wall-clock time on a calendar date.
    fold=0 (PEP 495 default) picks the first occurrence inside an ambiguous
    fall-back hour; a spring-forward gap resolves per zoneinfo's projection.
    Typical exchange boundaries never land in either window."""
    return int(datetime(d.year, d.month, d.day, hour, minute, tzinfo=zone).timestamp())
