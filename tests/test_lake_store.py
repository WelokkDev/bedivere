"""bedivere.data.lake — write, read, resample.

The resample assertions are the load-bearing ones. Coarsening is exact for
OHLCV, so a derived 5m bar must be the bar a native 5m feed would have printed —
and the lake must not invent bars for buckets the vendor never printed.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from tests.lake_support import SKIP_REASON

pytest.importorskip("duckdb", reason=SKIP_REASON)
pytest.importorskip("pandas", reason=SKIP_REASON)

from bedivere.core.session_days import SessionDay  # noqa: E402
from bedivere.core.types import Candle, Timeframe  # noqa: E402
from bedivere.data.lake.layout import RollRule, continuous, session_path  # noqa: E402
from bedivere.data.lake.read import (  # noqa: E402
    LakeMissingError,
    latest_stored_ts,
    partition_kv,
    partition_row_count,
    read_bars,
    read_table,
    roll_boundaries,
)
from bedivere.data.lake.resample import (  # noqa: E402
    ResampleError,
    check_ratio,
    resample_series,
    resample_table,
)
from bedivere.data.lake.schema import AS_TRADED, BarBatch, BarSchemaError  # noqa: E402
from bedivere.data.lake.writer import LakeWriteError, write_day  # noqa: E402
from tests.helpers import eth_session_days  # noqa: E402
from tests.lake_helpers import lake_table, ramp, rows_of  # noqa: E402

SID = continuous("GLBX.MDP3", "NQ", RollRule.VOLUME, 0)
DAYS = eth_session_days(["2026-06-15"])
DAY: SessionDay = DAYS.days[0]


# ---------- write / read round trip ----------


def test_round_trip_preserves_bars(tmp_path: Path) -> None:
    rows = ramp(DAY.start_unix + 1, 10)
    write_day(tmp_path, SID, Timeframe.S1, DAY.label, lake_table(rows), source="test")

    got = read_bars(tmp_path, SID, Timeframe.S1, DAY.start_unix, DAY.end_unix)
    assert len(got) == 10
    assert got[0] == Candle(
        timestamp=rows[0][0], open=100.0, high=102.0, low=99.0, close=100.5, volume=1.0
    )
    assert [c.timestamp for c in got] == sorted(c.timestamp for c in got)
    assert all(c.partial is None for c in got)


def test_read_window_is_half_open(tmp_path: Path) -> None:
    rows = ramp(DAY.start_unix + 1, 10)
    write_day(tmp_path, SID, Timeframe.S1, DAY.label, lake_table(rows), source="test")

    first, last = rows[0][0], rows[-1][0]
    inside = read_bars(tmp_path, SID, Timeframe.S1, first, last)
    assert [c.timestamp for c in inside] == [r[0] for r in rows[1:]]  # start out, end in


def test_read_without_partitions_is_loud(tmp_path: Path) -> None:
    with pytest.raises(LakeMissingError, match="no lake partitions"):
        read_bars(tmp_path, SID, Timeframe.S1, DAY.start_unix, DAY.end_unix)


def test_an_empty_window_inside_a_real_series_is_not_an_error(tmp_path: Path) -> None:
    # The other half of the rule above: a gap inside a real series is legitimate.
    write_day(tmp_path, SID, Timeframe.S1, DAY.label, lake_table(ramp(DAY.start_unix + 1, 3)), "t")
    assert read_bars(tmp_path, SID, Timeframe.S1, 1, 2) == []


def test_drop_synthetic_filters_marked_bars(tmp_path: Path) -> None:
    rows = ramp(DAY.start_unix + 1, 4)
    table = lake_table(rows)
    table.synthetic = [False, True, False, True]
    write_day(tmp_path, SID, Timeframe.S1, DAY.label, table, source="test")

    assert len(read_bars(tmp_path, SID, Timeframe.S1, DAY.start_unix, DAY.end_unix)) == 4
    kept = read_bars(tmp_path, SID, Timeframe.S1, DAY.start_unix, DAY.end_unix, drop_synthetic=True)
    assert len(kept) == 2


def test_roll_boundaries_are_detectable(tmp_path: Path) -> None:
    rows = ramp(DAY.start_unix + 1, 4, instrument=1) + ramp(DAY.start_unix + 5, 4, instrument=2)
    write_day(tmp_path, SID, Timeframe.S1, DAY.label, lake_table(rows), source="test")
    table = read_table(tmp_path, SID, Timeframe.S1, DAY.start_unix, DAY.end_unix)
    assert roll_boundaries(table) == [DAY.start_unix + 5]


def test_footer_answers_without_reading_bars(tmp_path: Path) -> None:
    write_day(tmp_path, SID, Timeframe.S1, DAY.label, lake_table(ramp(DAY.start_unix + 1, 6)), "t")
    path = session_path(tmp_path, SID, Timeframe.S1, DAY.label)
    assert partition_row_count(path) == 6
    assert latest_stored_ts(tmp_path, SID, Timeframe.S1) == DAY.start_unix + 6
    assert latest_stored_ts(tmp_path, SID, Timeframe.M5) is None


# ---------- the write seam ----------


def test_write_day_stamps_provenance_in_the_footer(tmp_path: Path) -> None:
    write_day(
        tmp_path, SID, Timeframe.S1, DAY.label, lake_table(ramp(DAY.start_unix + 1, 3)), "src.dbn.zst"
    )
    kv = partition_kv(session_path(tmp_path, SID, Timeframe.S1, DAY.label))
    assert kv["bedivere.source"] == "src.dbn.zst"
    assert kv["bedivere.dataset"] == SID.dataset
    assert kv["bedivere.symbol"] == SID.symbol
    assert kv["bedivere.series"] == SID.series
    assert kv["bedivere.timeframe"] == "1s"
    assert kv["bedivere.session"] == DAY.label
    assert kv["bedivere.price_basis"] == AS_TRADED


def test_extra_metadata_cannot_relabel_a_partition(tmp_path: Path) -> None:
    write_day(
        tmp_path, SID, Timeframe.S1, DAY.label, lake_table(ramp(DAY.start_unix + 1, 3)), "real.dbn",
        {"bedivere.series": "raw.LIES", "job": "abc123"},
    )
    kv = partition_kv(session_path(tmp_path, SID, Timeframe.S1, DAY.label))
    assert kv["bedivere.series"] == SID.series  # the base identity wins
    assert kv["job"] == "abc123"  # provenance the path cannot carry still lands


def test_write_day_is_temp_then_rename(tmp_path: Path) -> None:
    # A stray temp from an earlier crash must not survive a successful write.
    target = session_path(tmp_path, SID, Timeframe.S1, DAY.label)
    target.parent.mkdir(parents=True)
    stale_tmp = target.with_name(target.name + ".tmp")
    stale_tmp.write_bytes(b"torn write from a killed process")

    write_day(tmp_path, SID, Timeframe.S1, DAY.label, lake_table(ramp(DAY.start_unix + 1, 5)), "t")

    assert not stale_tmp.exists()
    assert len(read_bars(tmp_path, SID, Timeframe.S1, DAY.start_unix, DAY.end_unix)) == 5


def test_write_day_replaces_rather_than_appends(tmp_path: Path) -> None:
    write_day(tmp_path, SID, Timeframe.S1, DAY.label, lake_table(ramp(DAY.start_unix + 1, 9)), "a")
    write_day(tmp_path, SID, Timeframe.S1, DAY.label, lake_table(ramp(DAY.start_unix + 1, 4)), "b")
    assert len(read_bars(tmp_path, SID, Timeframe.S1, DAY.start_unix, DAY.end_unix)) == 4
    assert partition_kv(session_path(tmp_path, SID, Timeframe.S1, DAY.label))["bedivere.source"] == "b"


def test_write_day_refuses_an_empty_batch(tmp_path: Path) -> None:
    with pytest.raises(LakeWriteError, match="EMPTY"):
        write_day(tmp_path, SID, Timeframe.S1, DAY.label, BarBatch(), source="test")


def test_write_day_refuses_unsorted_stamps(tmp_path: Path) -> None:
    rows = ramp(DAY.start_unix + 1, 3) + ramp(DAY.start_unix + 2, 2)
    with pytest.raises(LakeWriteError, match="strictly ascending"):
        write_day(tmp_path, SID, Timeframe.S1, DAY.label, lake_table(rows), source="test")
    assert not session_path(tmp_path, SID, Timeframe.S1, DAY.label).is_file()


def test_write_day_refuses_bad_values(tmp_path: Path) -> None:
    def batch_with(edits: dict[str, tuple[int, float | int]]) -> BarBatch:
        table = lake_table(ramp(DAY.start_unix + 1, 3))
        for field, (index, value) in edits.items():
            cast("list[float | int]", getattr(table, field))[index] = value
        return table

    cases: tuple[tuple[dict[str, tuple[int, float | int]], str], ...] = (
        ({"close": (1, float("nan"))}, "non-finite close"),
        ({"high": (0, float("inf"))}, "non-finite high"),
        ({"volume": (2, -1.0)}, "negative volume"),
        ({"instrument_id": (0, 2**32)}, "uint32"),
        ({"ts": (0, -5)}, "non-positive"),
    )
    for edits, message in cases:
        with pytest.raises(LakeWriteError, match=message):
            write_day(tmp_path, SID, Timeframe.S1, DAY.label, batch_with(edits), source="test")


def test_write_day_refuses_a_ragged_batch(tmp_path: Path) -> None:
    table = lake_table(ramp(DAY.start_unix + 1, 3))
    table.volume.pop()
    with pytest.raises(BarSchemaError, match="ragged"):
        write_day(tmp_path, SID, Timeframe.S1, DAY.label, table, source="test")


def test_a_drifted_schema_refuses_rather_than_casting(tmp_path: Path) -> None:
    import duckdb

    path = session_path(tmp_path, SID, Timeframe.S1, DAY.label)
    path.parent.mkdir(parents=True)
    duckdb.connect(":memory:").execute(
        'COPY (SELECT 1::BIGINT ts, 1.0::DOUBLE "open", 1.0::DOUBLE high, 1.0::DOUBLE low, '
        # volume is INTEGER here, not DOUBLE
        '1.0::DOUBLE "close", 1::INTEGER volume, 1::UINTEGER instrument_id, '
        f"false::BOOLEAN synthetic) TO '{path.as_posix()}' (FORMAT PARQUET)"
    )
    with pytest.raises(BarSchemaError, match="does not match"):
        read_bars(tmp_path, SID, Timeframe.S1, 0, 1 << 40)


# ---------- resample ----------


def test_check_ratio_rejects_backwards_and_equal() -> None:
    assert check_ratio(Timeframe.S1, Timeframe.S15) == 15
    assert check_ratio(Timeframe.S15, Timeframe.M5) == 20
    assert check_ratio(Timeframe.M5, Timeframe.H4) == 48
    with pytest.raises(ResampleError, match="coarser"):
        check_ratio(Timeframe.S15, Timeframe.S1)
    with pytest.raises(ResampleError, match="coarser"):
        check_ratio(Timeframe.S15, Timeframe.S15)
    # No counterexample for the "divides evenly" guard exists in the current
    # Timeframe set; it is kept for the rung that eventually does not (10s, 3m).


def test_resample_is_exact_ohlcv() -> None:
    s = DAY.start_unix
    rows = [
        (s + 1, 10.0, 12.0, 9.0, 11.0, 1.0, 42),
        (s + 5, 11.0, 15.0, 10.5, 14.0, 2.0, 42),
        (s + 9, 14.0, 14.5, 8.0, 9.0, 3.0, 42),
        (s + 12, 9.0, 10.0, 8.5, 9.5, 4.0, 42),
        (s + 15, 9.5, 11.0, 9.25, 10.75, 5.0, 42),
    ]
    out = resample_table(lake_table(rows), DAY, Timeframe.S15)
    assert len(out) == 1
    row = rows_of(out)[0]
    assert row["ts"] == s + 15  # the bucket's last constituent bar
    assert (row["open"], row["high"], row["low"], row["close"]) == (10.0, 15.0, 8.0, 10.75)
    assert row["volume"] == 15.0
    assert row["instrument_id"] == 42


def test_resample_is_associative() -> None:
    # Holds only if bucket alignment is anchored on the session-day start at
    # every step rather than floored on the wall clock.
    table = lake_table(ramp(DAY.start_unix + 1, 900))
    direct = resample_table(table, DAY, Timeframe.M5)
    staged = resample_table(resample_table(table, DAY, Timeframe.S15), DAY, Timeframe.M5)
    assert rows_of(direct) == rows_of(staged)
    assert len(direct) == 3


def test_resample_anchors_buckets_on_the_session_open() -> None:
    # The load-bearing `- 1` in bucket_index_of: a close-stamp exactly on a
    # bucket boundary belongs to the bucket it ENDS, not the next one.
    s = DAY.start_unix
    out = resample_table(lake_table(ramp(s + 1, 30)), DAY, Timeframe.S15)
    assert out.ts == [s + 15, s + 30]


def test_resample_emits_nothing_for_untraded_buckets() -> None:
    s = DAY.start_unix
    rows = ramp(s + 1, 3) + ramp(s + 91, 3)  # an 87-second hole
    out = resample_table(lake_table(rows), DAY, Timeframe.S15)

    # Two buckets in, two bars out: the four empty ones produce nothing.
    assert len(out) == 2
    assert not any(out.synthetic)

    # Stamped at the LAST REAL BAR, not the bucket's grid end (s+15 / s+105) —
    # aggregate_bucket's convention, shared with MarketView.
    assert out.ts == [s + 3, s + 93]


def test_resample_refuses_a_bucket_that_spans_a_roll() -> None:
    s = DAY.start_unix
    rows = ramp(s + 1, 2, instrument=1) + ramp(s + 3, 2, instrument=2)
    with pytest.raises(ResampleError, match="roll landed mid-bucket"):
        resample_table(lake_table(rows), DAY, Timeframe.S15)


def test_resample_matches_the_engines_own_fold() -> None:
    """Not merely correct — IDENTICAL, stamp included, because `resample_table`
    imports `aggregate_bucket` rather than reimplementing it."""
    from bedivere.core.aggregator import aggregate_bucket

    rows = ramp(DAY.start_unix + 1, 15)
    out = resample_table(lake_table(rows), DAY, Timeframe.S15)
    folded = aggregate_bucket(
        [Candle(timestamp=r[0], open=r[1], high=r[2], low=r[3], close=r[4], volume=r[5]) for r in rows]
    )
    assert rows_of(out)[0] == {
        "ts": folded.timestamp,
        "open": folded.open,
        "high": folded.high,
        "low": folded.low,
        "close": folded.close,
        "volume": folded.volume,
        "instrument_id": 42,
        "synthetic": False,
    }


def test_resample_series_writes_partitions(tmp_path: Path) -> None:
    write_day(
        tmp_path, SID, Timeframe.S1, DAY.label, lake_table(ramp(DAY.start_unix + 1, 300)), "test"
    )
    report = resample_series(tmp_path, SID, Timeframe.S1, Timeframe.S15, DAYS)

    assert report.days_written == 1
    assert report.bars_in == 300
    assert report.bars_out == 20
    assert session_path(tmp_path, SID, Timeframe.S15, DAY.label).is_file()
    assert len(read_bars(tmp_path, SID, Timeframe.S15, DAY.start_unix, DAY.end_unix)) == 20
    assert "1s -> 15s" in report.describe()


def test_resample_series_refuses_a_day_the_calendar_does_not_cover(tmp_path: Path) -> None:
    write_day(tmp_path, SID, Timeframe.S1, DAY.label, lake_table(ramp(DAY.start_unix + 1, 30)), "t")
    with pytest.raises(ResampleError, match="session geometry must cover"):
        resample_series(tmp_path, SID, Timeframe.S1, Timeframe.S15, eth_session_days(["2026-06-16"]))
