"""The candle store, and coverage reporting.

A source that quietly returns a short series turns a data outage into a
strategy result; the defence is comparing what arrived against what session
geometry demands.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bedivere.core.types import Timeframe
from bedivere.data.csv import CsvCandleSource
from bedivere.data.port import assess_coverage, expected_stamps_in_range
from bedivere.data.sqlite import CandleStoreError, SqliteCandleStore
from tests.helpers import bar
from tests.strategy_fixture import ramp_bars, ramp_days

DAYS = ramp_days()
BARS = ramp_bars(DAYS)
TF = Timeframe.M5


# ---------- the store ----------


def test_bars_round_trip_in_the_half_open_range(tmp_path: Path) -> None:
    with SqliteCandleStore(tmp_path / "c.db") as store:
        store.upsert("DEMO", TF, BARS)
        day = DAYS.days[0]
        got = store.candles("DEMO", TF, day.start_unix, day.end_unix)
    # (start, end] — the session convention, so one day's query is one day.
    assert got[0].timestamp == day.start_unix + 300
    assert got[-1].timestamp == day.end_unix
    assert all(a.timestamp < b.timestamp for a, b in zip(got, got[1:], strict=False))


def test_reimporting_corrects_rather_than_duplicates(tmp_path: Path) -> None:
    db = tmp_path / "c.db"
    with SqliteCandleStore(db) as store:
        first = store.upsert("DEMO", TF, BARS[:10])
        corrected = [bar(BARS[0].timestamp, 1.0, 2.0, 0.5, 1.5, v=99)]
        second = store.upsert("DEMO", TF, corrected)
        got = store.candles("DEMO", TF, 0, BARS[9].timestamp)
    assert (first.inserted, first.replaced) == (10, 0)
    assert (second.inserted, second.replaced) == (0, 1)
    assert len(got) == 10  # replaced, not appended
    assert got[0].volume == 99


def test_symbols_and_timeframes_do_not_bleed(tmp_path: Path) -> None:
    with SqliteCandleStore(tmp_path / "c.db") as store:
        store.upsert("DEMO", TF, BARS[:5])
        store.upsert("OTHER", TF, BARS[:5])
        store.upsert("DEMO", Timeframe.M30, BARS[:5])
        assert len(store.candles("DEMO", TF, 0, 1 << 40)) == 5
        assert {(s.symbol, s.timeframe) for s in store.inventory()} == {
            ("DEMO", "5m"),
            ("OTHER", "5m"),
            ("DEMO", "30m"),
        }


def test_a_read_only_store_refuses_writes(tmp_path: Path) -> None:
    db = tmp_path / "c.db"
    with SqliteCandleStore(db) as store:
        store.upsert("DEMO", TF, BARS[:2])
    with SqliteCandleStore(db, read_only=True) as store:
        with pytest.raises(CandleStoreError, match="read-only"):
            store.upsert("DEMO", TF, BARS[:2])


def test_a_missing_read_only_store_says_how_to_make_one(tmp_path: Path) -> None:
    with pytest.raises(CandleStoreError, match="import some bars first"):
        SqliteCandleStore(tmp_path / "nope.db", read_only=True)


# ---------- CSV as a source ----------


def test_a_csv_source_refuses_to_answer_for_another_series(tmp_path: Path) -> None:
    """Handing back the only bars on hand because the caller asked for a
    symbol this file never held is how a mismatch becomes a result."""
    path = tmp_path / "demo.csv"
    path.write_text(
        "timestamp,open,high,low,close,volume\n"
        + "".join(f"{b.timestamp},{b.open},{b.high},{b.low},{b.close},{b.volume}\n" for b in BARS[:5]),
        encoding="utf-8",
    )
    source = CsvCandleSource(path, symbol="DEMO", timeframe=TF)
    assert len(source.candles("DEMO", TF, 0, 1 << 40)) == 5
    with pytest.raises(ValueError, match="holds DEMO 5m"):
        source.candles("OTHER", TF, 0, 1 << 40)


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
    """Out-of-session time contributes nothing to `expected`. Reporting a
    weekend as missing data trains you to ignore the report."""
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
