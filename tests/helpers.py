"""Shared fixtures/utilities for the bedivere test suite.

Session-day fixtures here are LITERAL resolved rows (the same shape
bedivere.core.session_days parses) — tests construct boundaries with
zoneinfo for convenience, which is fixture construction, not engine policy.
Fixture instants deliberately avoid the ambiguous fall-back hour.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from bedivere.core.session_days import SessionDay, SessionDays
from bedivere.core.types import Candle

ET = "America/New_York"
_ET_ZONE = ZoneInfo(ET)


def bar(ts: int, o: float, h: float, low: float, c: float, v: float = 0) -> Candle:
    return Candle(timestamp=ts, open=o, high=h, low=low, close=c, volume=v)


def utc(y: int, mo: int, d: int, h: int, mi: int = 0, s: int = 0) -> int:
    return int(datetime(y, mo, d, h, mi, s, tzinfo=UTC).timestamp())


def et(iso: str) -> int:
    """Unix seconds for an ET wall-clock instant "YYYY-MM-DDTHH:MM[:SS]",
    DST-correct for unambiguous instants (all fixtures avoid the fall-back
    repeat hour)."""
    date_part, time_part = iso.split("T")
    y, mo, d = (int(p) for p in date_part.split("-"))
    hh, mm, *rest = (int(p) for p in time_part.split(":"))
    ss = rest[0] if rest else 0
    return int(datetime(y, mo, d, hh, mm, ss, tzinfo=_ET_ZONE).timestamp())


def prev_day(iso: str) -> str:
    y, mo, d = (int(p) for p in iso.split("-"))
    dt = datetime(y, mo, d, tzinfo=UTC) - timedelta(days=1)
    return f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}"


# ---------- literal SessionDays fixtures ----------


def make_session_days(
    template: str, timezone: str, days: list[tuple[str, int, int]]
) -> SessionDays:
    """Literal resolved rows: (label, startUnix, endUnix) triples."""
    ordered = sorted(days, key=lambda t: t[1])
    return SessionDays(
        template=template,
        timezone=timezone,
        days=tuple(SessionDay(label=lbl, start_unix=s, end_unix=e) for lbl, s, e in ordered),
    )


def eth_session_days(close_dates: list[str]) -> SessionDays:
    """CME ETH session-days for the given close-date labels: open prev
    18:00 ET, close 17:00 ET."""
    return make_session_days(
        "cme_us_index_futures_eth",
        ET,
        [
            (cd, et(f"{prev_day(cd)}T18:00:00"), et(f"{cd}T17:00:00"))
            for cd in close_dates
        ],
    )


def eth_5m_flat(close_date: str) -> list[Candle]:
    """Full flat 5m series for one CME ETH session-day (open prev 18:00 ET,
    close 17:00 ET) — 276 bars."""
    open_ts = et(f"{prev_day(close_date)}T18:00:00")
    close_ts = et(f"{close_date}T17:00:00")
    return [bar(ts, 100, 100, 100, 100, 1) for ts in range(open_ts + 300, close_ts + 1, 300)]


def candles_equal(a: list[Candle], b: list[Candle]) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b, strict=True):
        if (
            x.timestamp != y.timestamp
            or x.open != y.open
            or x.high != y.high
            or x.low != y.low
            or x.close != y.close
            or x.volume != y.volume
            or x.partial != y.partial
        ):
            return False
    return True
