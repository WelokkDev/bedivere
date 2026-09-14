"""Turning a vendor's per-day quality feed into a go/no-go on a backtest window.

Reconciling two calendars is the whole reason this module exists. Databento
reports per UTC calendar date; bedivere reasons in session-days, which for CME
index futures open 18:00 ET and close 17:00 ET the next afternoon — so one
session straddles two UTC dates, and a join on `date == label` would check the
wrong day about half the time and pass.

A session-day therefore inherits the WORST condition of every UTC date it
touches: `degraded` says something is wrong somewhere in that date without
saying where, so a session overlapping it cannot be vouched for.
"""

from __future__ import annotations

from collections.abc import Container, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from bedivere.core.session_days import SessionDay, SessionDays
from bedivere.data.vendor.databento.condition import Condition, DayCondition


class DataQualityError(Exception):
    """A window contains session-days whose data is not trustworthy."""


@dataclass(frozen=True, slots=True)
class SessionDayQuality:
    """One session-day's verdict, with the evidence that produced it."""

    label: str
    utc_dates: tuple[str, ...]
    """Every UTC date the session-day's (start, end] interval touches."""

    worst: Condition | None
    """Worst condition across `utc_dates`, or None when any is UNKNOWN.

    None is not "fine": the feed has no row for at least one date this session
    covers, so nothing has vouched for it.
    """

    flagged: tuple[DayCondition, ...]
    """The non-`available` rows behind the verdict, for reporting."""

    unknown_dates: tuple[str, ...]
    """Covered dates with no row in the feed."""

    @property
    def clean(self) -> bool:
        return self.worst is Condition.AVAILABLE and not self.unknown_dates

    def describe(self) -> str:
        if self.clean:
            return f"{self.label}: available"
        parts = [f"{row.date} {row.condition.value}" for row in self.flagged]
        parts.extend(f"{date} unknown" for date in self.unknown_dates)
        return f"{self.label}: " + ", ".join(parts)


def utc_dates_of(day: SessionDay) -> tuple[str, ...]:
    """UTC dates touched by a session-day's (start, end] interval.

    Start is EXCLUSIVE, so the first instant is `start_unix + 1`: using
    `start_unix` would pull in the previous UTC date for a session opening
    exactly at midnight UTC.
    """
    first = datetime.fromtimestamp(day.start_unix + 1, tz=UTC).date()
    last = datetime.fromtimestamp(day.end_unix, tz=UTC).date()
    out: list[str] = []
    cursor = first
    while cursor <= last:
        out.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return tuple(out)


def assess(
    days: SessionDays | Iterable[SessionDay], conditions: Sequence[DayCondition]
) -> list[SessionDayQuality]:
    """Verdict per session-day, in the order the days were given."""
    by_date = {row.date: row for row in conditions}
    day_list = days.days if isinstance(days, SessionDays) else tuple(days)

    out: list[SessionDayQuality] = []
    for day in day_list:
        dates = utc_dates_of(day)
        flagged: list[DayCondition] = []
        unknown: list[str] = []
        worst: Condition | None = Condition.AVAILABLE
        for date in dates:
            row = by_date.get(date)
            if row is None:
                unknown.append(date)
                worst = None
                continue
            if not row.condition.usable:
                flagged.append(row)
            if worst is not None and row.condition.severity > worst.severity:
                worst = row.condition
        # An unknown date poisons the verdict whatever the known dates said.
        if unknown:
            worst = None
        out.append(
            SessionDayQuality(
                label=day.label,
                utc_dates=dates,
                worst=worst,
                flagged=tuple(flagged),
                unknown_dates=tuple(unknown),
            )
        )
    return out


def require_clean(
    assessments: Sequence[SessionDayQuality],
    *,
    allow: Container[Condition] = (),
    allow_unknown: bool = False,
) -> None:
    """Raise unless every session-day is trustworthy.

    `allow` takes explicit `Condition` members rather than a bare `force=True`,
    so a run that tolerates degraded days names what it tolerated.
    """
    bad = [
        a
        for a in assessments
        if not a.clean
        and not (a.worst is not None and a.worst in allow)
        and not (a.worst is None and allow_unknown)
    ]
    if not bad:
        return
    detail = "\n  ".join(a.describe() for a in bad[:20])
    more = f"\n  ... and {len(bad) - 20} more" if len(bad) > 20 else ""
    raise DataQualityError(
        f"{len(bad)} of {len(assessments)} session-days are not clean:\n  {detail}{more}\n"
        "Pass allow=[Condition.DEGRADED] (or allow_unknown=True) to proceed anyway."
    )


def clean_labels(assessments: Sequence[SessionDayQuality]) -> list[str]:
    """Labels of the session-days that are trustworthy, ascending."""
    return [a.label for a in assessments if a.clean]
