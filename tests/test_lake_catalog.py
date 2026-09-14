"""bedivere.data.lake.catalog — the lineage check.

A resample killed halfway leaves every file valid and every read succeeding, so
nothing about the data can show it. The footers can.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.lake_support import SKIP_REASON

pytest.importorskip("duckdb", reason=SKIP_REASON)
pytest.importorskip("pandas", reason=SKIP_REASON)

from bedivere.core.types import Timeframe  # noqa: E402
from bedivere.data.lake.catalog import coverage_gaps, inventory  # noqa: E402
from bedivere.data.lake.layout import RollRule, local_continuous, session_path  # noqa: E402
from bedivere.data.lake.resample import resample_series  # noqa: E402
from bedivere.data.lake.writer import write_day  # noqa: E402
from tests.helpers import eth_session_days  # noqa: E402
from tests.lake_helpers import lake_table, ramp  # noqa: E402

SID = local_continuous("GLBX.MDP3", "NQ", RollRule.VOLUME, 0)
LABELS = ["2026-06-15", "2026-06-16", "2026-06-17"]
DAYS = eth_session_days(LABELS)


def _seed_raw(root: Path) -> None:
    for day in DAYS.days:
        write_day(
            root, SID, Timeframe.S1, day.label, lake_table(ramp(day.start_unix + 1, 60)), "nq.dbn.zst"
        )


def test_an_empty_lake_inventories_as_empty(tmp_path: Path) -> None:
    assert inventory(tmp_path) == []
    assert coverage_gaps(tmp_path) == []


def test_inventory_reports_identity_and_lineage(tmp_path: Path) -> None:
    _seed_raw(tmp_path)
    resample_series(tmp_path, SID, Timeframe.S1, Timeframe.S15, DAYS)

    got = {(e.timeframe, e.source, e.session_days, e.derived_from) for e in inventory(tmp_path)}
    assert got == {
        (Timeframe.S1, "nq.dbn.zst", 3, None),
        (Timeframe.S15, "resample:1s", 3, Timeframe.S1),
    }
    assert all(e.series == SID for e in inventory(tmp_path))


def test_a_complete_ladder_reports_no_gaps(tmp_path: Path) -> None:
    _seed_raw(tmp_path)
    resample_series(tmp_path, SID, Timeframe.S1, Timeframe.S15, DAYS)
    assert coverage_gaps(tmp_path) == []


def test_a_half_finished_resample_is_visible(tmp_path: Path) -> None:
    _seed_raw(tmp_path)
    resample_series(tmp_path, SID, Timeframe.S1, Timeframe.S15, DAYS, labels=LABELS[:2])

    (problem,) = coverage_gaps(tmp_path)
    assert "15s (from 1s)" in problem
    assert "2 of 3 session-days" in problem
    assert "1 missing (e.g. 2026-06-17)" in problem


def test_a_derived_series_whose_source_is_gone_is_named(tmp_path: Path) -> None:
    _seed_raw(tmp_path)
    resample_series(tmp_path, SID, Timeframe.S1, Timeframe.S15, DAYS)
    for label in LABELS:
        path = session_path(tmp_path, SID, Timeframe.S1, label)
        path.unlink()
        path.parent.rmdir()

    (problem,) = coverage_gaps(tmp_path)
    assert "derived from 1s, which is not in the lake" in problem


def test_timeframes_from_different_inputs_are_not_compared(tmp_path: Path) -> None:
    """The invariant is lineage, not "every timeframe spans the same range" — a
    directly-ingested 1h series legitimately covers years a 1s series does not."""
    _seed_raw(tmp_path)
    for label in ("2020-01-02", "2020-01-03"):
        day = eth_session_days([label]).days[0]
        write_day(
            tmp_path, SID, Timeframe.H1, label,
            lake_table(ramp(day.start_unix + 3600, 4, step=3600)), "nq-1h.dbn.zst",
        )
    assert coverage_gaps(tmp_path) == []
