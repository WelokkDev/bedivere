"""Research plumbing: availability, ties, stale joins, provenance, and restart."""

from __future__ import annotations

import time
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

from tests.lake_support import SKIP_REASON

pytest.importorskip("duckdb", reason=SKIP_REASON)
pytest.importorskip("pandas", reason=SKIP_REASON)
pytest.importorskip("databento_dbn", reason=SKIP_REASON)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402  # pyright: ignore[reportMissingTypeStubs]

from bedivere.core.session_days import SessionDay, SessionDays  # noqa: E402
from bedivere.data.lake.schema import BarSchemaError  # noqa: E402
from bedivere.data.lake.volume import NS, VolumeBar, VolumeSpec  # noqa: E402
from bedivere.data.lake.volume_build import build_volume_archive  # noqa: E402
from bedivere.data.lake.volume_research import (  # noqa: E402
    Provenance,
    align_available,
    load_volume_frame,
    volume_coverage,
)
from bedivere.data.lake.volume_store import COLUMNS, volume_path  # noqa: E402
from tests.test_volume import DAY, DAYS, TRADES  # noqa: E402
from tests.test_volume_lake import SID, archive, seed_time_bars  # noqa: E402

# pandas is optional and shipped without complete typing for dynamic columns.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false


def observed(days: SessionDays = DAYS) -> Provenance:
    """A Databento study's attestation about its own observation rows."""
    return Provenance(
        feed="databento",
        dataset=SID.dataset,
        symbol=SID.symbol,
        series=SID.series,
        instrument_namespace=f"databento:{SID.dataset}",
        price_basis="as_traded",
        sessions=days,
        time_unit="ns",
        epoch="unix",
        clock="ts_recv",
    )


OBSERVED = observed()


def seed(root: Path, spec: VolumeSpec | None = None) -> VolumeSpec:
    spec = spec or VolumeSpec(1000)
    build_volume_archive(archive(root / "trades.dbn"), root, SID, spec, DAYS)
    return spec


def test_loader_selects_definition_partial_policy_and_availability(tmp_path: Path) -> None:
    spec = seed(tmp_path)
    frame = load_volume_frame(tmp_path, SID, spec, DAYS, feed="databento")
    assert len(frame) == 2
    assert str(frame["available_ns"].dtype) == "int64"
    assert frame.attrs["bar_definition"]["threshold"] == 1000
    full = load_volume_frame(tmp_path, SID, spec, DAYS, feed="databento", include_partial=True)
    assert len(full) == 3 and int(full.iloc[-1]["available_ns"]) == DAY.end_unix * NS
    before_close = load_volume_frame(
        tmp_path, SID, spec, DAYS, feed="databento", include_partial=True, end_ns=DAY.end_unix * NS
    )
    assert len(before_close) == 2
    assert volume_coverage(tmp_path, SID, spec, DAYS)[0]["status"] == "valid"


def test_missing_and_corrupt_partitions_do_not_silently_enter_training(tmp_path: Path) -> None:
    spec = seed(tmp_path)
    missing = VolumeSpec(1234)
    with pytest.raises(FileNotFoundError, match="missing volume-bar session"):
        load_volume_frame(tmp_path, SID, missing, DAYS, feed="databento")
    empty = load_volume_frame(tmp_path, SID, missing, DAYS, feed="databento", allow_missing=True)
    assert empty.empty and str(empty["available_ns"].dtype) == "int64"
    assert empty.attrs["missing_sessions"] == [DAY.label]
    assert volume_coverage(tmp_path, SID, missing, DAYS)[0]["status"] == "missing"
    path = volume_path(tmp_path, SID, spec, DAYS, DAY.label)
    path.write_bytes(b"damaged")
    assert volume_coverage(tmp_path, SID, spec, DAYS)[0]["status"] == "invalid"


def test_alignment_preserves_order_respects_sessions_and_refuses_future_or_stale_values(
    tmp_path: Path,
) -> None:
    spec = seed(tmp_path)
    frame = load_volume_frame(tmp_path, SID, spec, DAYS, feed="databento")
    first = int(frame.iloc[0]["available_ns"])
    other = SessionDay("2026-06-16", DAY.end_unix, DAY.end_unix + 10)
    obs = pd.DataFrame(
        {
            "asof_ns": [first + 1, first - 1, first, other.start_unix * NS + 2, first + 1, first + 500],
            "session": [DAY.label, DAY.label, DAY.label, other.label, DAY.label, DAY.label],
            "instrument_id": [42, 42, 42, 42, 43, 42],
            "private_column": [5, 4, 3, 2, 1, 0],
        }
    )
    context = observed(replace(DAYS, days=(DAY, other)))
    joined = align_available(obs, frame, observation_provenance=context, max_age_ns=10)
    assert joined["private_column"].tolist() == [5, 4, 3, 2, 1, 0]
    assert int(joined.iloc[0]["volume_bar_id"]) == 0
    assert joined["volume_bar_id"].isna().tolist() == [False, True, True, True, True, True]
    inclusive = align_available(
        obs, frame, observation_provenance=context, max_age_ns=10, allow_exact_matches=True
    )
    assert not pd.isna(inclusive.iloc[2]["volume_bar_id"])


def test_alignment_returns_the_observations_themselves_with_bar_columns_attached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = seed(tmp_path)
    frame = load_volume_frame(tmp_path, SID, spec, DAYS, feed="databento")
    first = int(frame.iloc[0]["available_ns"])
    # Far from UTC, a database round trip would return local, microsecond timestamps.
    if hasattr(time, "tzset"):
        monkeypatch.setenv("TZ", "Asia/Tokyo")
        time.tzset()
    obs = pd.DataFrame(
        {
            "asof_ns": [first + 1, first - 1, first + 1],
            "session": pd.Categorical([DAY.label] * 3),
            "instrument_id": np.array([42, 42, 42], dtype=np.uint32),
            "decided_at": pd.to_datetime(
                ["2026-06-15 22:00:00.123456789"] * 2 + ["2026-06-15 22:00:01.000000001"], utc=True
            ),
            "stake": [Decimal("0.1000000000000000000001"), Decimal("7"), Decimal("7")],
            "note": [{"why": 1}, "text", 3.5],
            "month": pd.period_range("2026-01", periods=3, freq="M"),
            "small": np.array([1, 2, 3], dtype=np.int8),
        },
        index=pd.Index(["b", "a", "a"], name="decision"),  # named, unsorted, repeated
    )
    before = obs.copy(deep=True)
    try:
        joined = align_available(obs, frame, observation_provenance=OBSERVED, max_age_ns=10)
    finally:
        monkeypatch.undo()
        if hasattr(time, "tzset"):
            time.tzset()
    names = cast(list[str], list(obs.columns))
    # Index, order, dtypes and values: nothing of the caller's own was touched.
    pd.testing.assert_frame_equal(joined[names], before)
    pd.testing.assert_frame_equal(obs, before)
    assert list(joined.columns) == [*names, *(f"volume_{name}" for name in COLUMNS)]
    assert cast(Any, joined["decided_at"]).dt.hour.tolist() == [22, 22, 22]
    assert cast(Any, joined["decided_at"]).dt.nanosecond.tolist() == [789, 789, 1]
    assert joined["stake"].iloc[0] == Decimal("0.1000000000000000000001")
    assert joined["volume_bar_id"].isna().tolist() == [False, True, False]
    assert str(joined["volume_available_ns"].dtype) == "Int64"
    assert "provenance" in joined.attrs and "bar_definition" in joined.attrs


def test_a_trimmed_bar_frame_refuses_the_decisions_it_can_no_longer_answer(tmp_path: Path) -> None:
    spec = seed(tmp_path)
    whole = load_volume_frame(tmp_path, SID, spec, DAYS, feed="databento")
    first, second = (int(v) for v in cast(Any, whole["available_ns"]).tolist())
    assert whole.attrs["availability_window"] == (None, None)

    def seen(bars: pd.DataFrame, asof: int, max_age_ns: int = 10 * NS, **options: Any) -> int | None:
        obs = pd.DataFrame({"asof_ns": [asof], "session": [DAY.label], "instrument_id": [42]})
        joined = align_available(
            obs, bars, observation_provenance=OBSERVED, max_age_ns=max_age_ns, **options
        )
        value = cast(Any, joined["volume_bar_id"]).iloc[0]
        return None if pd.isna(value) else int(value)

    # Trimmed before the second bar: a later decision would be handed superseded bar 0.
    early = load_volume_frame(tmp_path, SID, spec, DAYS, feed="databento", end_ns=second)
    assert early.attrs["availability_window"] == (None, second)
    assert (seen(whole, second + 1), len(early)) == (1, 1)
    with pytest.raises(ValueError, match="look outside the availability window"):
        seen(early, second + 1)
    # Up to the bound the trimmed frame holds everything a decision can see.
    assert seen(early, second) == seen(whole, second) == 0
    with pytest.raises(ValueError, match="look outside the availability window"):
        seen(early, second, allow_exact_matches=True)  # the bar available at the bound is not held
    # Trimmed at the start, a lookback reaching before the bound would miss the cut bar.
    late = load_volume_frame(tmp_path, SID, spec, DAYS, feed="databento", start_ns=first + 1)
    assert seen(whole, first + 5) == 0
    with pytest.raises(ValueError, match="look outside the availability window"):
        seen(late, first + 5)
    assert seen(late, second + 5, max_age_ns=4) == seen(whole, second + 5, max_age_ns=4) is None
    assert seen(late, second + 5, max_age_ns=5) == seen(whole, second + 5, max_age_ns=5) == 1
    # A window that opens at or before the session does not trim that session.
    opened = load_volume_frame(
        tmp_path, SID, spec, DAYS, feed="databento", start_ns=DAY.start_unix * NS
    )
    assert seen(opened, second + 5, max_age_ns=3600 * NS) == 1
    # The record travels with the join's result.
    obs = pd.DataFrame({"asof_ns": [second], "session": [DAY.label], "instrument_id": [42]})
    joined = align_available(obs, early, observation_provenance=OBSERVED, max_age_ns=10 * NS)
    assert joined.attrs["availability_window"] == (None, second)


def test_equal_timestamp_split_bars_choose_last_id_and_observation_duplicates_survive(
    tmp_path: Path,
) -> None:
    spec = seed(tmp_path, VolumeSpec(100, boundary="split_trade"))
    frame = load_volume_frame(tmp_path, SID, spec, DAYS, feed="databento")
    first = int(frame.iloc[0]["available_ns"])
    obs = pd.DataFrame(
        {"asof_ns": [first + 1, first + 1], "session": [DAY.label] * 2, "instrument_id": [42] * 2}
    )
    joined = align_available(obs, frame, observation_provenance=OBSERVED, max_age_ns=100)
    assert joined["volume_bar_id"].tolist() == [3, 3]


def test_alignment_rejects_float_timestamps_and_mixed_definitions(tmp_path: Path) -> None:
    spec = seed(tmp_path)
    frame = load_volume_frame(tmp_path, SID, spec, DAYS, feed="databento")
    obs = pd.DataFrame(
        {"asof_ns": [float(101 * NS)], "session": [DAY.label], "instrument_id": [42]}
    )
    with pytest.raises(ValueError, match="integer dtypes"):
        align_available(obs, frame, observation_provenance=OBSERVED, max_age_ns=NS)
    obs["asof_ns"] = obs["asof_ns"].astype("int64")
    mixed = pd.concat([frame, frame.assign(definition="other")], ignore_index=True)
    with pytest.raises(ValueError, match="exactly one"):
        align_available(obs, mixed, observation_provenance=OBSERVED, max_age_ns=NS)


def test_resume_avoids_decoding_current_partitions_and_rebuilds_changed_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bedivere.data.lake import volume_build

    spec = seed(tmp_path)
    path = volume_path(tmp_path, SID, spec, DAYS, DAY.label)
    before = path.stat().st_mtime_ns
    with monkeypatch.context() as context:

        def no_decode(*_args: object) -> None:
            raise AssertionError("unchanged input should not be decoded")

        context.setattr(volume_build, "_selected_trades", no_decode)
        report = build_volume_archive(
            tmp_path / "trades.dbn", tmp_path, SID, spec, DAYS, resume=True
        )
        assert report[0]["status"] == "unchanged"
    assert path.stat().st_mtime_ns == before
    archive(tmp_path / "trades.dbn", [replace(t, volume=t.volume * 2) for t in TRADES])
    report = build_volume_archive(tmp_path / "trades.dbn", tmp_path, SID, spec, DAYS, resume=True)
    assert report[0]["status"] == "written" and report[0]["volume"] == 4900
    frame = load_volume_frame(tmp_path, SID, spec, DAYS, feed="databento", include_partial=True)
    assert int(cast(Any, frame["volume"]).sum()) == 4900


def test_incomplete_comparison_resume_builds_missing_representation(tmp_path: Path) -> None:
    spec = seed(tmp_path)
    reports = build_volume_archive(
        tmp_path / "trades.dbn", tmp_path, SID, spec, DAYS, compare_1s=True, resume=True
    )
    assert len(reports) == 2
    assert all(report["status"] == "written" for report in reports)
    again = build_volume_archive(
        tmp_path / "trades.dbn", tmp_path, SID, spec, DAYS, compare_1s=True, resume=True
    )
    assert all(report["status"] == "unchanged" for report in again)


def test_loader_preserves_real_world_integer_precision(tmp_path: Path) -> None:
    start = 1_800_000_000
    day = replace(DAY, start_unix=start, end_unix=start + 10)
    days = replace(DAYS, days=(day,))
    trades = [
        replace(t, start_ns=t.start_ns + (start - 100) * NS, end_ns=t.end_ns + (start - 100) * NS)
        for t in TRADES
    ]
    spec = VolumeSpec(1000)
    path = archive(tmp_path / "trades.dbn", trades, start=start * NS, end=(start + 10) * NS)
    build_volume_archive(path, tmp_path, SID, spec, days)
    frame = load_volume_frame(tmp_path, SID, spec, days, feed="databento")
    assert int(frame.iloc[0]["available_ns"]) == start * NS + 200
    obs = pd.DataFrame(
        {
            "asof_ns": [start * NS + 201, start * NS],
            "session": [day.label] * 2,
            "instrument_id": [42] * 2,
        }
    )
    joined = align_available(obs, frame, observation_provenance=observed(days), max_age_ns=10)
    # float64 spacing is 256 ns here, so any float round trip would move these.
    assert float(start * NS + 201) != start * NS + 201
    assert joined["asof_ns"].tolist() == [start * NS + 201, start * NS]
    assert [int(joined.iloc[0][f"volume_{name}"]) for name in ("start_ns", "end_ns")] == [
        start * NS + 100,
        start * NS + 200,
    ]
    assert int(joined.iloc[0]["volume_available_ns"]) == start * NS + 200
    assert str(joined["volume_available_ns"].dtype) == "Int64"
    assert pd.isna(joined.iloc[1]["volume_available_ns"])


def test_valid_looking_content_changes_fail_checksum_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bedivere.data.lake import volume_store

    spec = seed(tmp_path)
    path = volume_path(tmp_path, SID, spec, DAYS, DAY.label)
    bars = volume_store.read_volume_partition(path)
    # VWAP remains finite and schema-valid, but no longer matches what was written.
    changed = [replace(bar, vwap=10.01) for bar in bars]
    def changed_partition(_path: Path) -> list[VolumeBar]:
        return changed

    monkeypatch.setattr(volume_store, "read_volume_partition", changed_partition)
    coverage = volume_coverage(tmp_path, SID, spec, DAYS)
    assert coverage[0]["status"] == "invalid" and "checksum" in coverage[0]["error"]
    with pytest.raises(BarSchemaError, match="checksum"):
        build_volume_archive(tmp_path / "trades.dbn", tmp_path, SID, spec, DAYS, resume=True)


def test_resume_rechecks_continuous_contract_selection(tmp_path: Path) -> None:
    from bedivere.core.types import Timeframe
    from bedivere.data.lake.layout import session_path
    from bedivere.data.lake.read import read_partition
    from bedivere.data.lake.writer import write_day

    sid = replace(SID, series="local.v.0")
    spec = VolumeSpec(1000)
    seed_time_bars(tmp_path, sid)
    source = archive(tmp_path / "trades.dbn")
    build_volume_archive(source, tmp_path, sid, spec, DAYS)
    assert (
        build_volume_archive(source, tmp_path, sid, spec, DAYS, resume=True)[0]["status"]
        == "unchanged"
    )
    table = read_partition(session_path(tmp_path, sid, Timeframe.S1, DAY.label))
    table.instrument_id[:] = [43] * len(table.instrument_id)
    write_day(tmp_path, sid, Timeframe.S1, DAY.label, table, "revised selection")
    with pytest.raises(ValueError, match="no selected trades"):
        build_volume_archive(source, tmp_path, sid, spec, DAYS, resume=True)


def test_empty_bar_selection_aligns_to_null_without_float_timestamps(tmp_path: Path) -> None:
    spec = seed(tmp_path)
    frame = load_volume_frame(
        tmp_path, SID, spec, DAYS, feed="databento", end_ns=DAY.start_unix * NS
    )
    assert frame.empty
    # At the bound itself the empty frame holds all there is to see: nothing.
    obs = pd.DataFrame(
        {"asof_ns": [DAY.start_unix * NS], "session": [DAY.label], "instrument_id": [42]}
    )
    joined = align_available(obs, frame, observation_provenance=OBSERVED, max_age_ns=100)
    assert pd.isna(joined.iloc[0]["volume_bar_id"])
    assert str(joined["volume_available_ns"].dtype) == "Int64"
