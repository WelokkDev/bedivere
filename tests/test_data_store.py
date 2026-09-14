"""The zero-setup source, and coverage reporting.

A source that quietly returns a short series turns a data outage into a strategy
result; the defence is comparing what arrived against what session geometry
demands. That check is source-agnostic on purpose — a CSV, the lake and somebody
else's feed are all measured against the same grid.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bedivere.core.types import Timeframe
from bedivere.data.csv import CsvCandleSource
from bedivere.data.port import assess_coverage, expected_stamps_in_range
from tests.helpers import bar
from tests.strategy_fixture import ramp_bars, ramp_days

DAYS = ramp_days()
BARS = ramp_bars(DAYS)
TF = Timeframe.M5


# ---------- CSV as a source ----------


def _write_csv(path: Path, count: int = 5) -> Path:
    path.write_text(
        "timestamp,open,high,low,close,volume\n"
        + "".join(
            f"{b.timestamp},{b.open},{b.high},{b.low},{b.close},{b.volume}\n" for b in BARS[:count]
        ),
        encoding="utf-8",
    )
    return path


def test_a_csv_source_serves_the_half_open_range(tmp_path: Path) -> None:
    source = CsvCandleSource(_write_csv(tmp_path / "demo.csv", 10), symbol="DEMO", timeframe=TF)
    # (start, end] — the session convention, shared by every source.
    got = source.candles("DEMO", TF, BARS[0].timestamp, BARS[4].timestamp)
    assert [c.timestamp for c in got] == [b.timestamp for b in BARS[1:5]]


def test_a_csv_source_refuses_to_answer_for_another_series(tmp_path: Path) -> None:
    """Handing back the only bars on hand because the caller asked for a symbol
    this file never held is how a mismatch becomes a result."""
    source = CsvCandleSource(_write_csv(tmp_path / "demo.csv"), symbol="DEMO", timeframe=TF)
    assert len(source.candles("DEMO", TF, 0, 1 << 40)) == 5
    with pytest.raises(ValueError, match="holds DEMO 5m"):
        source.candles("OTHER", TF, 0, 1 << 40)
    with pytest.raises(ValueError, match="holds DEMO 5m"):
        source.candles("DEMO", Timeframe.M30, 0, 1 << 40)


def test_a_csv_source_names_itself(tmp_path: Path) -> None:
    source = CsvCandleSource(_write_csv(tmp_path / "demo.csv"), symbol="DEMO", timeframe=TF)
    assert source.describe() == f"csv:{tmp_path / 'demo.csv'} (DEMO 5m)"


# ---------- coverage ----------


def test_a_complete_range_reports_complete() -> None:
    day = DAYS.days[0]
    coverage = assess_coverage(
        [b for b in BARS if day.start_unix < b.timestamp <= day.end_unix],
        days=DAYS,
        symbol="DEMO",
        timeframe=TF,
        start_unix=day.start_unix,
        end_unix=day.end_unix,
    )
    assert coverage.complete
    assert coverage.expected == coverage.present > 0
    assert "100.0%" in coverage.describe()


def test_a_hole_is_visible_not_silently_short() -> None:
    day = DAYS.days[0]
    full = [b for b in BARS if day.start_unix < b.timestamp <= day.end_unix]
    punched = full[:20] + full[25:]  # five consecutive bars removed
    coverage = assess_coverage(
        punched,
        days=DAYS,
        symbol="DEMO",
        timeframe=TF,
        start_unix=day.start_unix,
        end_unix=day.end_unix,
    )
    assert not coverage.complete
    assert coverage.present == len(punched)
    assert [(g.from_unix, g.bars) for g in coverage.gaps] == [(full[20].timestamp, 5)]
    assert "1 gap(s)" in coverage.describe()


def test_weekend_time_is_not_a_gap() -> None:
    """Out-of-session time contributes nothing to `expected`. Reporting a weekend
    as missing data trains you to ignore the report."""
    first, second = DAYS.days[0], DAYS.days[1]
    spanning = [b for b in BARS if first.start_unix < b.timestamp <= second.end_unix]
    coverage = assess_coverage(
        spanning,
        days=DAYS,
        symbol="DEMO",
        timeframe=TF,
        start_unix=first.start_unix,
        end_unix=second.end_unix,
    )
    assert coverage.complete


def test_off_grid_bars_are_counted_separately() -> None:
    """A bar that is not on the session grid usually means a timezone or
    close-stamp convention mismatch — worth seeing, not tolerating."""
    day = DAYS.days[0]
    full = [b for b in BARS if day.start_unix < b.timestamp <= day.end_unix]
    coverage = assess_coverage(
        [*full, bar(day.start_unix + 137, 1, 1, 1, 1)],
        days=DAYS,
        symbol="DEMO",
        timeframe=TF,
        start_unix=day.start_unix,
        end_unix=day.end_unix,
    )
    assert coverage.unexpected == 1
    assert "off-grid" in coverage.describe()


def test_expected_stamps_match_the_session_grid() -> None:
    day = DAYS.days[0]
    stamps = expected_stamps_in_range(DAYS, TF, day.start_unix, day.end_unix)
    assert stamps[0] == day.start_unix + 300
    assert stamps[-1] == day.end_unix
    assert stamps == sorted(stamps)
