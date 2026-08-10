"""The spec envelope: what it accepts, what it refuses, and how a live run
derives what a backtest states.

The refusals matter more than the acceptances here. A spec is a document a
human writes at 6am, and every field that could silently mean something other
than what was intended is a trade nobody chose.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from bedivere.config.spec import (
    SpecError,
    archived_spec,
    load_spec,
    parse_spec,
    resolve_backtest_window,
    resolve_instrument,
    resolve_live_window,
    resolve_sessions,
)
from bedivere.core.types import Timeframe
from tests.helpers import et

BASE: dict[str, Any] = {
    "strategy": "tests.strategy_fixture:PLUGIN",
    "symbol": "DEMO",
    "config": {"kind": "ramp"},
    "baseTimeframe": "5m",
    "derivedTimeframes": ["30m"],
    "session": {
        "template": "cme_us_index_futures_eth",
        "timezone": "America/New_York",
        "openTime": "18:00",
        "closeTime": "17:00",
        "firstCloseDate": "2026-07-14",
        "lastCloseDate": "2026-07-16",
    },
    "instruments": {"DEMO": {"tickSize": 0.25, "pointValue": 20}},
    "window": {"firstTradeDate": "2026-07-15", "lastTradeDate": "2026-07-16"},
    "sim": {
        "latencyMs": 250,
        "halfSpreadTicks": 1,
        "commissionCentsPerSidePerContract": 105,
        "seed": 7,
        "deferProtectionOneBar": False,
    },
}


def spec(**overrides: Any) -> dict[str, Any]:
    return {**json.loads(json.dumps(BASE)), **overrides}


# ---------- envelope validation ----------


def test_a_valid_envelope_round_trips() -> None:
    parsed = parse_spec(spec())
    assert parsed.base_timeframe is Timeframe.M5
    assert parsed.derived_timeframes == (Timeframe.M30,)
    assert parsed.sim.seed == 7


def test_an_unknown_envelope_key_is_refused() -> None:
    with pytest.raises(SpecError, match="baseTimefrmae"):
        parse_spec(spec(baseTimefrmae="5m"))


def test_costs_have_no_defaults() -> None:
    """Same rule as run_backtest: a frictionless run must be visible in the
    file, never assumed by the loader."""
    payload = spec()
    del payload["sim"]["halfSpreadTicks"]
    with pytest.raises(SpecError, match="halfSpreadTicks"):
        parse_spec(payload)


def test_the_symbol_must_have_an_instrument() -> None:
    with pytest.raises(SpecError, match="instruments has no entry"):
        parse_spec(spec(symbol="OTHER"))


def test_derived_timeframes_must_be_coarser_than_the_base() -> None:
    with pytest.raises(SpecError, match="not coarser"):
        parse_spec(spec(baseTimeframe="30m", derivedTimeframes=["5m"]))


def test_snake_case_is_accepted_alongside_camel() -> None:
    payload = spec()
    payload["base_timeframe"] = payload.pop("baseTimeframe")
    assert parse_spec(payload).base_timeframe is Timeframe.M5


def test_a_session_cannot_be_both_rules_and_rows() -> None:
    payload = spec()
    payload["session"]["days"] = [{"label": "2026-07-14", "startUnix": 1, "endUnix": 2}]
    with pytest.raises(SpecError, match="pick one form"):
        parse_spec(payload)


def test_a_window_needs_both_halves_of_one_pair() -> None:
    with pytest.raises(SpecError, match="both halves"):
        parse_spec(spec(window={"firstTradeDate": "2026-07-15"}))


def test_a_data_block_must_match_its_source() -> None:
    with pytest.raises(SpecError, match="needs `factory`"):
        parse_spec(spec(data={"source": "python", "path": "x.csv"}))
    with pytest.raises(SpecError, match="needs `path`"):
        parse_spec(spec(data={"source": "csv"}))


def test_load_spec_names_the_file_it_could_not_read(tmp_path: Path) -> None:
    with pytest.raises(SpecError, match="not valid JSON"):
        broken = tmp_path / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        load_spec(broken)


# ---------- resolution ----------


def test_instrument_validation_happens_at_the_spec_boundary() -> None:
    """tickSize x pointValue that is not whole cents is a mis-specified
    instrument, and it must fail before a run starts, not mid-fill."""
    with pytest.raises(SpecError, match="whole cents"):
        resolve_instrument(parse_spec(spec(instruments={"DEMO": {"tickSize": 0.001, "pointValue": 1}})))


def test_a_backtest_window_resolves_from_session_day_labels() -> None:
    parsed = parse_spec(spec())
    days = resolve_sessions(parsed)
    start, end = resolve_backtest_window(parsed, days)
    assert start == days.days[1].start_unix
    assert end == days.days[-1].end_unix


def test_a_window_label_that_is_not_a_session_day_is_refused() -> None:
    parsed = parse_spec(spec(window={"firstTradeDate": "2026-07-18", "lastTradeDate": "2026-07-16"}))
    with pytest.raises(SpecError, match="not a session-day"):
        resolve_backtest_window(parsed, resolve_sessions(parsed))


def test_a_backtest_without_a_window_is_refused() -> None:
    payload = spec()
    del payload["window"]
    parsed = parse_spec(payload)
    with pytest.raises(SpecError, match="needs a `window`"):
        resolve_backtest_window(parsed, resolve_sessions(parsed))


def test_a_backtest_calendar_may_not_be_derived_from_now() -> None:
    payload = spec()
    del payload["session"]["firstCloseDate"]
    with pytest.raises(SpecError, match="required for a backtest"):
        resolve_sessions(parse_spec(payload))


def test_a_live_calendar_is_expanded_around_now() -> None:
    """The property that lets one file serve both environments: no dates in
    the session block, and the calendar still contains today."""
    payload = spec()
    del payload["session"]["firstCloseDate"]
    del payload["session"]["lastCloseDate"]
    payload["session"]["lookbackDays"] = 4
    now = et("2026-07-15T10:00")
    days = resolve_sessions(parse_spec(payload), now_unix=now)
    assert days.day_containing(now) is not None


def test_a_live_window_runs_from_now_to_the_session_close() -> None:
    days = resolve_sessions(parse_spec(spec()))
    now = et("2026-07-15T10:00")
    start, end = resolve_live_window(days, now_unix=now)
    assert start == now
    assert end == et("2026-07-15T17:00")


def test_until_shortens_the_live_window() -> None:
    days = resolve_sessions(parse_spec(spec()))
    now = et("2026-07-15T10:00")
    assert resolve_live_window(days, now_unix=now, until="15:55")[1] == et("2026-07-15T15:55")


def test_until_after_the_close_is_refused() -> None:
    days = resolve_sessions(parse_spec(spec()))
    with pytest.raises(SpecError, match="after the session-day close"):
        resolve_live_window(days, now_unix=et("2026-07-15T10:00"), until="23:30")


def test_until_in_the_past_is_refused() -> None:
    days = resolve_sessions(parse_spec(spec()))
    with pytest.raises(SpecError, match="not in the future"):
        resolve_live_window(days, now_unix=et("2026-07-15T10:00"), until="09:00")


def test_a_live_run_outside_any_session_is_refused() -> None:
    days = resolve_sessions(parse_spec(spec()))
    with pytest.raises(SpecError, match="no session-day contains"):
        resolve_live_window(days, now_unix=et("2026-07-15T17:30"))


# ---------- provenance ----------


def test_the_archived_spec_bakes_in_overrides_and_stamps() -> None:
    """The archived spec must re-run identically with no --set flags and no
    dependence on what "today" meant."""
    parsed = parse_spec(spec())
    archived = archived_spec(parsed, resolved_config={"kind": "ramp", "qty": 4}, window=(10, 20))
    assert archived["config"] == {"kind": "ramp", "qty": 4}
    assert archived["window"] == {"startUnix": 10, "endUnix": 20}
    # ...and it is still a valid spec.
    reparsed = parse_spec(archived)
    assert resolve_backtest_window(reparsed, resolve_sessions(reparsed)) == (10, 20)
