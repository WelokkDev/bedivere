"""bedivere.data.quality — reconciling the vendor's UTC dates with session-days."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bedivere.data.quality import (
    DataQualityError,
    assess,
    clean_labels,
    require_clean,
    utc_dates_of,
)
from bedivere.data.vendor.databento.condition import (
    Condition,
    DayCondition,
    load_condition_json,
    parse_conditions,
    read_conditions,
    write_conditions,
)
from tests.helpers import eth_session_days


def cond(date: str, condition: Condition, modified: str | None = "2026-01-01") -> DayCondition:
    return DayCondition(date=date, condition=condition, last_modified=modified)


def test_a_session_day_spans_two_utc_dates() -> None:
    # An 18:00 ET open -> 17:00 ET close crosses 00:00 UTC mid-session.
    day = eth_session_days(["2026-06-15"]).days[0]
    assert utc_dates_of(day) == ("2026-06-14", "2026-06-15")


def test_one_degraded_utc_date_poisons_two_session_days() -> None:
    # The Jun-16 session OPENS on Jun-15 UTC, so a degraded Jun-15 taints both.
    days = eth_session_days(["2026-06-15", "2026-06-16", "2026-06-17"])
    conditions = [
        cond("2026-06-14", Condition.AVAILABLE),
        cond("2026-06-15", Condition.DEGRADED),
        cond("2026-06-16", Condition.AVAILABLE),
        cond("2026-06-17", Condition.AVAILABLE),
    ]
    verdicts = {a.label: a for a in assess(days, conditions)}
    assert verdicts["2026-06-15"].worst is Condition.DEGRADED
    assert verdicts["2026-06-16"].worst is Condition.DEGRADED
    assert verdicts["2026-06-17"].clean
    assert clean_labels(assess(days, conditions)) == ["2026-06-17"]


def test_worst_condition_wins() -> None:
    days = eth_session_days(["2026-06-15"])
    verdict = assess(
        days, [cond("2026-06-14", Condition.DEGRADED), cond("2026-06-15", Condition.MISSING)]
    )[0]
    assert verdict.worst is Condition.MISSING
    assert len(verdict.flagged) == 2
    assert "degraded" in verdict.describe() and "missing" in verdict.describe()


def test_a_missing_row_poisons_the_verdict_rather_than_passing() -> None:
    days = eth_session_days(["2026-06-15"])
    verdict = assess(days, [cond("2026-06-14", Condition.AVAILABLE)])[0]
    assert verdict.worst is None
    assert verdict.unknown_dates == ("2026-06-15",)
    assert not verdict.clean


def test_the_gate_is_fail_closed_and_the_allowlist_is_explicit() -> None:
    days = eth_session_days(["2026-06-15"])
    conditions = [cond("2026-06-14", Condition.AVAILABLE), cond("2026-06-15", Condition.DEGRADED)]
    verdicts = assess(days, conditions)
    with pytest.raises(DataQualityError, match="degraded"):
        require_clean(verdicts)
    require_clean(verdicts, allow=[Condition.DEGRADED])

    unknown = assess(days, [])
    with pytest.raises(DataQualityError):
        require_clean(unknown, allow=[Condition.DEGRADED])
    require_clean(unknown, allow_unknown=True)


def test_severity_ordering() -> None:
    order = [Condition.AVAILABLE, Condition.DEGRADED, Condition.PENDING, Condition.MISSING]
    assert [c.severity for c in order] == sorted(c.severity for c in order)
    assert Condition.AVAILABLE.usable
    assert not any(c.usable for c in order[1:])


def test_parse_rejects_an_unknown_condition_rather_than_dropping_it() -> None:
    with pytest.raises(ValueError, match="not a known Databento condition"):
        parse_conditions([{"date": "2026-06-15", "condition": "probably_fine"}])
    with pytest.raises(ValueError, match="must be an object"):
        parse_conditions(["2026-06-15"])
    with pytest.raises(ValueError, match="must be a JSON array"):
        parse_conditions({"date": "2026-06-15"})


def test_load_condition_json(tmp_path: Path) -> None:
    path = tmp_path / "condition.json"
    path.write_text(
        json.dumps(
            [
                {"date": "2026-06-15", "condition": "available", "last_modified_date": "2026-06-16"},
                {"date": "2026-06-14", "condition": "missing", "last_modified_date": None},
            ]
        ),
        encoding="utf-8",
    )
    rows = load_condition_json(path)
    assert [r.date for r in rows] == ["2026-06-14", "2026-06-15"]  # sorted
    assert rows[0].condition is Condition.MISSING
    assert rows[0].last_modified is None


def test_condition_round_trips_through_parquet(tmp_path: Path) -> None:
    pytest.importorskip("duckdb", reason="the lake extra is not installed")
    rows = [
        cond("2026-06-14", Condition.AVAILABLE),
        cond("2026-06-15", Condition.DEGRADED),
        cond("2026-06-16", Condition.MISSING, None),
    ]
    write_conditions(tmp_path, "GLBX.MDP3", rows)
    assert read_conditions(tmp_path, "GLBX.MDP3") == rows
    assert read_conditions(tmp_path, "NOPE.SUCH") == []

    # Whole-file replacement, not a merge: a re-fetch that disagrees IS the
    # staleness signal.
    write_conditions(tmp_path, "GLBX.MDP3", [cond("2026-06-14", Condition.DEGRADED, "2026-07-01")])
    assert read_conditions(tmp_path, "GLBX.MDP3") == [
        cond("2026-06-14", Condition.DEGRADED, "2026-07-01")
    ]
