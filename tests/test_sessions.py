"""bedivere.sessions — derived session-days must equal hand-resolved rows."""

from __future__ import annotations

import pytest

from bedivere.sessions import DailySessionSpec, build_session_days, cme_futures_sessions
from tests.helpers import et, eth_session_days


def test_cme_preset_reproduces_hand_resolved_rows() -> None:
    built = cme_futures_sessions("2026-07-14", "2026-07-16")
    fixture = eth_session_days(["2026-07-14", "2026-07-15", "2026-07-16"])
    assert built.template == fixture.template
    assert built.timezone == fixture.timezone
    assert built.days == fixture.days  # labels + exact unix boundaries


def test_monday_session_opens_sunday_evening() -> None:
    # 2026-07-13 is a Monday; its ETH session opens Sunday 18:00 ET.
    days = cme_futures_sessions("2026-07-13", "2026-07-13")
    assert len(days.days) == 1
    d = days.days[0]
    assert d.label == "2026-07-13"
    assert d.start_unix == et("2026-07-12T18:00:00")
    assert d.end_unix == et("2026-07-13T17:00:00")


def test_weekend_dates_produce_no_sessions() -> None:
    # Fri 2026-07-10 .. Mon 2026-07-13: Sat/Sun are not close-days.
    days = cme_futures_sessions("2026-07-10", "2026-07-13")
    assert [d.label for d in days.days] == ["2026-07-10", "2026-07-13"]


def test_holidays_skip_by_close_date() -> None:
    days = cme_futures_sessions("2026-07-14", "2026-07-16", holidays={"2026-07-15"})
    assert [d.label for d in days.days] == ["2026-07-14", "2026-07-16"]


def test_dst_spring_forward_lands_in_the_weekend_gap() -> None:
    # US DST begins Sunday 2026-03-08 02:00 ET — inside the Fri-close →
    # Sun-open gap, never inside a session. Friday runs on EST, Monday's
    # session opens Sunday 18:00 EDT; the real-seconds gap is exactly 48h
    # (49 wall-clock hours minus the skipped hour).
    days = cme_futures_sessions("2026-03-06", "2026-03-09")
    fri, mon = days.days
    assert fri.label == "2026-03-06" and mon.label == "2026-03-09"
    assert mon.start_unix - fri.end_unix == 48 * 3600
    # Both sessions are the normal 23h in real seconds.
    assert fri.end_unix - fri.start_unix == 23 * 3600
    assert mon.end_unix - mon.start_unix == 23 * 3600


def test_same_day_sessions_when_open_before_close() -> None:
    spec = DailySessionSpec(timezone="America/New_York", open_time="09:30", close_time="16:00")
    days = build_session_days(spec, "2026-07-14", "2026-07-15")
    assert days.days[0].start_unix == et("2026-07-14T09:30:00")
    assert days.days[0].end_unix == et("2026-07-14T16:00:00")


def test_back_to_back_24h_sessions_touch_but_stay_disjoint() -> None:
    # open == close ("17:00"/"17:00") → 24h sessions where one day's end
    # IS the next day's start. (start, end] semantics keep them disjoint:
    # the boundary instant belongs to the earlier day.
    spec = DailySessionSpec(timezone="America/New_York", open_time="17:00", close_time="17:00")
    days = build_session_days(spec, "2026-07-14", "2026-07-15")
    d1, d2 = days.days
    assert d1.end_unix == d2.start_unix
    assert days.day_containing(d1.end_unix) is d1
    assert days.day_containing(d1.end_unix + 1) is d2


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="HH:MM"):
        DailySessionSpec(timezone="America/New_York", open_time="6pm", close_time="17:00")
    with pytest.raises(ValueError, match="0..6"):
        DailySessionSpec(
            timezone="America/New_York",
            open_time="18:00",
            close_time="17:00",
            close_weekdays=(7,),
        )
    with pytest.raises(ValueError, match="before first"):
        cme_futures_sessions("2026-07-16", "2026-07-14")
    with pytest.raises(ValueError, match="no eligible"):
        cme_futures_sessions("2026-07-11", "2026-07-12")  # Sat..Sun only
