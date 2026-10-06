"""Real DBN -> both volume representations -> Parquet -> research SQL."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from tests.lake_support import SKIP_REASON

pytest.importorskip("duckdb", reason=SKIP_REASON)
pytest.importorskip("pandas", reason=SKIP_REASON)
dbn = pytest.importorskip("databento_dbn", reason=SKIP_REASON)

from bedivere.cli import data as data_cli  # noqa: E402
from bedivere.core.session_days import SessionDay, SessionDays  # noqa: E402
from bedivere.core.types import Timeframe  # noqa: E402
from bedivere.data.lake.ingest import IngestError  # noqa: E402
from bedivere.data.lake.layout import SeriesId  # noqa: E402
from bedivere.data.lake.query import connect  # noqa: E402
from bedivere.data.lake.read import partition_kv  # noqa: E402
from bedivere.data.lake.schema import BarBatch, BarSchemaError  # noqa: E402
from bedivere.data.lake.volume import (  # noqa: E402
    NS,
    VolumeAccumulator,
    VolumeBar,
    VolumeInput,
    VolumeSpec,
)
from bedivere.data.lake.volume_build import (  # noqa: E402
    build_volume_archive,
    derive_volume,
    one_second_inputs,
)
from bedivere.data.lake.volume_research import load_volume_frame, volume_coverage  # noqa: E402
from bedivere.data.lake.volume_store import (  # noqa: E402
    read_volume_partition,
    volume_inventory,
    volume_path,
    write_volume_day,
)
from bedivere.data.lake.writer import source_range_metadata, write_day  # noqa: E402
from tests.test_volume import DAY, DAYS, TRADES  # noqa: E402

SID = SeriesId("GLBX.MDP3", "NQ", "raw.NQZ5")


def archive(
    path: Path,
    trades: list[VolumeInput] | None = None,
    *,
    record_edits: dict[str, Any] | None = None,
    **meta_edits: Any,
) -> Path:
    rows = TRADES if trades is None else trades
    metadata = {
        "dataset": SID.dataset,
        "schema": dbn.Schema.TRADES,
        "start": DAY.start_unix * NS,
        "end": DAY.end_unix * NS,
        "stype_in": dbn.SType.RAW_SYMBOL,
        "stype_out": dbn.SType.INSTRUMENT_ID,
        "symbols": ["NQZ5"],
        "mappings": [
            SimpleNamespace(
                raw_symbol="NQZ5",
                intervals=[
                    SimpleNamespace(
                        start_date=dt.date(1970, 1, 1),
                        end_date=dt.date(2030, 1, 1),
                        symbol="42",
                    )
                ],
            )
        ],
    }
    metadata.update(meta_edits)
    records = [
        dbn.TradeMsg(
            **{
                "publisher_id": 1,
                "instrument_id": t.instrument_id,
                "ts_event": t.start_ns - 1,
                "ts_recv": t.start_ns,
                "price": round(t.close * NS),
                "size": t.volume,
                "action": dbn.Action.TRADE,
                "side": dbn.Side.BID,
                "depth": 0,
                "sequence": i,
                **(record_edits or {}),
            }
        )
        for i, t in enumerate(rows)
    ]
    path.write_bytes(dbn.Metadata(**metadata).encode() + b"".join(bytes(r) for r in records))
    return path


def whole(day: SessionDay) -> dict[str, str]:
    """Footer record of a source that covered `day` from open to close."""
    return source_range_metadata(day.start_unix * NS, day.end_unix * NS)


def seed_time_bars(root: Path, sid: SeriesId = SID) -> None:
    batch = BarBatch()
    for row in one_second_inputs(TRADES):
        batch.append(
            row.end_ns // NS,
            row.open,
            row.high,
            row.low,
            row.close,
            float(row.volume),
            row.instrument_id,
        )
    write_day(root, sid, Timeframe.S1, DAY.label, batch, "fixture", whole(DAY))


def test_archive_comparison_conserves_volume_and_exposes_aggregation_loss(tmp_path: Path) -> None:
    spec = VolumeSpec(1000)
    reports = build_volume_archive(
        archive(tmp_path / "trades.dbn"), tmp_path, SID, spec, DAYS, compare_1s=True
    )
    reference, approximate = reports
    assert reference["volume"] == approximate["volume"] == 2450
    assert reference["full_bars"] == 2 and approximate["full_bars"] == 1
    assert reference["overshoot_max"] == 200 and approximate["overshoot_max"] == 600
    assert reference["partial_volume"] == 250 and approximate["partial_volume"] == 850
    reference_bars = read_volume_partition(Path(str(reference["path"])))
    assert reference_bars[0].end_ns == TRADES[1].end_ns
    assert reference_bars[-1].available_ns == DAY.end_unix * NS
    approx_bars = read_volume_partition(Path(str(approximate["path"])))
    assert approx_bars[0].close == 8 and approx_bars[0].low == 8
    assert all(b.vwap is None for b in approx_bars)
    footer = partition_kv(Path(str(reference["path"])))
    assert "sha256:" in footer["bedivere.source"]
    assert json.loads(footer["bedivere.bar_definition"])["clock"] == "ts_recv"
    assert len(volume_inventory(tmp_path)) == 2
    with connect(tmp_path) as con:
        assert con.execute("SELECT sum(volume) FROM volume_bars").fetchone() == (4900,)
        assert con.execute("SELECT count(*) FROM bars").fetchone() == (0,)
        stored = [row[:2] for row in con.execute("DESCRIBE volume_bars").fetchall()]
    # A lake without volume bars answers with the same columns and types, and no rows.
    with connect(tmp_path / "empty") as con:
        assert [row[:2] for row in con.execute("DESCRIBE volume_bars").fetchall()] == stored


def test_stored_1s_and_same_trade_1s_approximation_agree(tmp_path: Path) -> None:
    seed_time_bars(tmp_path)
    archive_reports = build_volume_archive(
        archive(tmp_path / "trades.dbn"), tmp_path, SID, VolumeSpec(1000), DAYS, compare_1s=True
    )
    expected = read_volume_partition(Path(str(archive_reports[1]["path"])))
    reports = derive_volume(tmp_path, SID, VolumeSpec(1000, "ohlcv-1s"), DAYS)
    assert read_volume_partition(Path(str(reports[0]["path"]))) == expected
    # Equal bars here, yet two datasets: each build has its own partition.
    assert reports[0]["path"] != archive_reports[1]["path"]
    assert read_volume_partition(Path(str(archive_reports[1]["path"]))) == expected


def stored_seconds(root: Path, days: SessionDays) -> None:
    """Stored 1s bars of 1,200 contracts a session: not what the archive's trades hold."""
    for day in days.days:
        batch = BarBatch()
        for i, volume in enumerate((700, 500)):
            batch.append(day.start_unix + 1 + i, 10.0, 11.0, 9.0, 10.5, float(volume), 42)
        write_day(root, SID, Timeframe.S1, day.label, batch, "fixture", whole(day))


def session_volumes(root: Path, spec: VolumeSpec, days: SessionDays) -> dict[str, int]:
    """Volume per session, as the research loader returns it."""
    frame = cast(
        Any, load_volume_frame(root, SID, spec, days, feed="databento", include_partial=True)
    )
    return {str(k): int(v) for k, v in frame.groupby("session")["volume"].sum().items()}


def test_compare_1s_and_from_1s_never_write_to_the_same_partition(tmp_path: Path) -> None:
    next_day = SessionDay("2026-06-16", 110, 120)
    days = replace(DAYS, days=(DAY, next_day))
    first = replace(DAYS, days=(DAY,))
    stored, rebuilt = VolumeSpec(1000, "ohlcv-1s"), VolumeSpec(1000, "trades-1s")
    stored_seconds(tmp_path, days)
    assert [r["volume"] for r in derive_volume(tmp_path, SID, stored, days)] == [1200, 1200]
    stored_paths = [volume_path(tmp_path, SID, stored, days, day.label) for day in days.days]
    before = [path.read_bytes() for path in stored_paths]

    # The archive holds 2,450 contracts for the first session only.
    trades = archive(tmp_path / "trades.dbn", end=120 * NS)
    reports = build_volume_archive(
        trades, tmp_path, SID, VolumeSpec(1000), first, compare_1s=True, resume=True
    )
    assert [(r["source"], r["volume"]) for r in reports] == [("trades", 2450), ("trades-1s", 2450)]
    rebuilt_path = volume_path(tmp_path, SID, rebuilt, days, DAY.label)
    assert Path(str(reports[1]["path"])) == rebuilt_path and rebuilt_path not in stored_paths
    assert [path.read_bytes() for path in stored_paths] == before
    assert session_volumes(tmp_path, stored, days) == {DAY.label: 1200, next_day.label: 1200}
    assert session_volumes(tmp_path, rebuilt, first) == {DAY.label: 2450}

    # Each frame and coverage row names what its partitions were built from.
    source = load_volume_frame(tmp_path, SID, stored, days, feed="databento").attrs["bar_source"]
    assert source.kind == "ohlcv-1s" and [label for label, _ in source.inputs] == [
        DAY.label,
        next_day.label,
    ]
    assert all(origin.startswith("bars.parquet:sha256:") for _, origin in source.inputs)
    approximation = load_volume_frame(tmp_path, SID, rebuilt, first, feed="databento")
    assert approximation.attrs["bar_source"].kind == "trades-1s"
    assert approximation.attrs["bar_source"].inputs[0][1].startswith(
        "same-trades-1s:trades.dbn:sha256:"
    )
    assert approximation.attrs["bar_definition"]["eligibility"] == "positive_trades_no_bad_recv"
    coverage = volume_coverage(tmp_path, SID, stored, days)
    assert [(row["status"], row["source"]) for row in coverage] == [
        ("valid", origin) for _, origin in source.inputs
    ]

    # Neither builder touches the other's dataset when it runs again.
    comparison = rebuilt_path.read_bytes()
    again = derive_volume(tmp_path, SID, stored, days, resume=True)
    assert [r["status"] for r in again] == ["unchanged", "unchanged"]
    derive_volume(tmp_path, SID, stored, days)
    assert rebuilt_path.read_bytes() == comparison
    resumed = build_volume_archive(
        trades, tmp_path, SID, VolumeSpec(1000), first, compare_1s=True, resume=True
    )
    assert [r["status"] for r in resumed] == ["unchanged", "unchanged"]
    assert session_volumes(tmp_path, stored, days) == {DAY.label: 1200, next_day.label: 1200}


def test_a_comparison_stored_before_trades_1s_is_refused_and_never_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bedivere.data.lake import volume_store

    stored = VolumeSpec(1000, "ohlcv-1s")
    stored_seconds(tmp_path, DAYS)
    builder = VolumeAccumulator(stored, DAY)
    bars = [b for row in one_second_inputs(TRADES) for b in builder.push(row)] + builder.finish()
    origin = "same-trades-1s:trades.dbn:sha256:" + "0" * 64

    def any_kind(_source: str) -> bool:
        return False

    def marked(_source: str) -> bool:
        return True

    with monkeypatch.context() as before:
        # As builds once wrote it: rebuilt seconds under the stored-seconds definition.
        before.setattr(volume_store, "_rebuilt_seconds", any_kind)
        path = write_volume_day(tmp_path, SID, stored, DAYS, DAY, bars, source=origin)
    kept = path.read_bytes()
    assert partition_kv(path)["bedivere.source"] == origin

    (row,) = volume_coverage(tmp_path, SID, stored, DAYS)
    assert row["status"] == "invalid" and "source=trades-1s" in row["error"]
    with pytest.raises(BarSchemaError, match="rebuilt from trades"):
        load_volume_frame(tmp_path, SID, stored, DAYS, feed="databento")
    for resume in (False, True):
        with pytest.raises(ValueError, match="does not replace it"):
            derive_volume(tmp_path, SID, stored, DAYS, resume=resume)
        assert path.read_bytes() == kept
    # A new write is held to its definition's kind of source, either way round.
    with pytest.raises(ValueError, match="and no other does"):
        write_volume_day(tmp_path, SID, stored, DAYS, DAY, bars, source=origin)
    rebuilt = replace(stored, source="trades-1s")
    with pytest.raises(ValueError, match="and no other does"):
        write_volume_day(tmp_path, SID, rebuilt, DAYS, DAY, bars, source="fixture")
    with monkeypatch.context() as before:
        # The reverse misfiling: a `trades-1s` partition that does not say so.
        before.setattr(volume_store, "_rebuilt_seconds", marked)
        write_volume_day(tmp_path, SID, rebuilt, DAYS, DAY, bars, source="fixture")
    (unmarked,) = volume_coverage(tmp_path, SID, rebuilt, DAYS)
    assert unmarked["status"] == "invalid" and "inconsistent bedivere.source" in unmarked["error"]
    path.unlink()
    assert derive_volume(tmp_path, SID, stored, DAYS)[0]["volume"] == 1200
    assert session_volumes(tmp_path, stored, DAYS) == {DAY.label: 1200}


def test_continuous_series_reuses_existing_lake_contract_selection(tmp_path: Path) -> None:
    sid = replace(SID, series="local.v.0")
    seed_time_bars(tmp_path, sid)
    reports = build_volume_archive(
        archive(tmp_path / "trades.dbn"), tmp_path, sid, VolumeSpec(1000), DAYS
    )
    assert reports[0]["volume"] == 2450


def test_split_bars_with_identical_nanosecond_closes_survive_storage(tmp_path: Path) -> None:
    spec = VolumeSpec(1000, boundary="split_trade")
    path = archive(tmp_path / "trades.dbn", [VolumeInput.trade(100 * NS, 12, 3200, 42)])
    build_volume_archive(path, tmp_path, SID, spec, DAYS)
    bars = read_volume_partition(volume_path(tmp_path, SID, spec, DAYS, DAY.label))
    assert [b.volume for b in bars] == [1000, 1000, 1000, 200]
    assert [b.bar_id for b in bars] == [0, 1, 2, 3]
    assert len({b.end_ns for b in bars}) == 1


def test_decoder_chunk_size_does_not_change_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bedivere.data.lake import trades

    path = archive(tmp_path / "trades.dbn")
    spec = VolumeSpec(1000)
    build_volume_archive(path, tmp_path, SID, spec, DAYS)
    target = volume_path(tmp_path, SID, spec, DAYS, DAY.label)
    expected = read_volume_partition(target)
    # The builder decodes through the shared trade reader.
    monkeypatch.setattr(trades, "CHUNK_BYTES", 7)
    build_volume_archive(path, tmp_path, SID, spec, DAYS)
    assert read_volume_partition(target) == expected


def test_bad_replacement_does_not_destroy_existing_partition(tmp_path: Path) -> None:
    spec = VolumeSpec(1000)
    builder = VolumeAccumulator(spec, DAY)
    bars = [b for t in TRADES for b in builder.push(t)] + builder.finish()
    path = write_volume_day(tmp_path, SID, spec, DAYS, DAY, bars, source="original")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="threshold"):
        write_volume_day(
            tmp_path, SID, spec, DAYS, DAY, [replace(bars[0], volume=1)], source="invalid"
        )
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="geometry"):
        write_volume_day(
            tmp_path,
            SID,
            spec,
            DAYS,
            replace(DAY, start_unix=99),
            bars,
            source="different calendar",
        )
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "edits,match",
    [
        ({"schema": dbn.Schema.OHLCV_1S}, "schema=trades"),
        ({"dataset": "OTHER"}, "dataset"),
        ({"end": 105 * NS}, "full session"),
        ({"start": 101 * NS}, "full session"),
        ({"limit": 5}, "record limit"),
    ],
)
def test_refuse_wrong_or_partial_archives(
    tmp_path: Path, edits: dict[str, Any], match: str
) -> None:
    path = archive(tmp_path / "trades.dbn", **edits)
    with pytest.raises(ValueError, match=match):
        build_volume_archive(path, tmp_path, SID, VolumeSpec(1000), DAYS)
    assert not (tmp_path / "event_bars").exists()


def test_refuse_truncated_record_and_out_of_order_trades(tmp_path: Path) -> None:
    path = archive(tmp_path / "trades.dbn")
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(IngestError, match="incomplete trailing"):
        build_volume_archive(path, tmp_path, SID, VolumeSpec(1000), DAYS)
    archive(path, list(reversed(TRADES)))
    with pytest.raises(ValueError, match="out of receive-time order"):
        build_volume_archive(path, tmp_path, SID, VolumeSpec(1000), DAYS)


def test_dry_run_computes_but_writes_nothing(tmp_path: Path) -> None:
    reports = build_volume_archive(
        archive(tmp_path / "trades.dbn"),
        tmp_path,
        SID,
        VolumeSpec(1000),
        DAYS,
        compare_1s=True,
        dry_run=True,
    )
    assert len(reports) == 2 and all(not r["written"] for r in reports)
    assert not (tmp_path / "event_bars").exists()


def test_volume_cli_and_inventory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = archive(tmp_path / "trades.dbn")
    session = tmp_path / "session.json"
    session.write_text(
        json.dumps(
            {
                "template": DAYS.template,
                "timezone": DAYS.timezone,
                "days": [
                    {"label": DAY.label, "startUnix": DAY.start_unix, "endUnix": DAY.end_unix}
                ],
            }
        )
    )
    args = [
        "volume",
        "--root",
        str(tmp_path),
        "--archive",
        str(path),
        "--session",
        str(session),
        "--symbol",
        "NQ",
        "--series",
        SID.series,
        "--threshold",
        "1000",
        "--compare-1s",
    ]
    assert data_cli.main(args) == 0
    reports = json.loads(capsys.readouterr().out)
    assert [r["volume"] for r in reports] == [2450, 2450]
    assert data_cli.main(["list", "--root", str(tmp_path)]) == 0
    assert "Volume-bar datasets" in capsys.readouterr().out
    assert data_cli.main([*args, "--threshold", "0"]) == 1
    assert "positive int64" in capsys.readouterr().err
    Path(reports[0]["path"]).write_bytes(b"corrupt")
    assert data_cli.main([*args, "--resume"]) == 1
    assert "Parquet" in capsys.readouterr().err


def test_stored_seconds_need_a_recorded_source_range_covering_the_whole_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from bedivere.data.lake import volume_build

    spec = VolumeSpec(1000, "ohlcv-1s")
    opened, closed = DAY.start_unix * NS, DAY.end_unix * NS
    batch = BarBatch()
    for row in one_second_inputs(TRADES):
        batch.append(row.end_ns // NS, row.open, row.high, row.low, row.close, float(row.volume), 42)

    def seed(extra: dict[str, str] | None) -> None:
        write_day(tmp_path, SID, Timeframe.S1, DAY.label, batch, "fixture", extra)

    # The same bars every time: only what their source covered differs.
    refusals = [
        (None, "records no source range"),
        (source_range_metadata(opened + 1, closed), "source 1970-01-01T00:01:40Z"),
        (source_range_metadata(opened, closed - 1), "session 1970-01-01T00:01:40Z"),
        ({"bedivere.source_start_ns": "soon", "bedivere.source_end_ns": str(closed)}, "no source"),
    ]
    for extra, match in refusals:
        seed(extra)
        for options in ({}, {"resume": True}, {"dry_run": True}):
            with pytest.raises(ValueError, match=match):
                derive_volume(tmp_path, SID, spec, DAYS, **options)
    assert not (tmp_path / "event_bars").exists()
    for extra in (whole(DAY), source_range_metadata(opened - 5, closed + 5)):
        seed(extra)
        assert derive_volume(tmp_path, SID, spec, DAYS)[0]["volume"] == 2450

    # Bars a build made before ranges were kept are not called current by a resume.
    def unchecked(*_args: object) -> None:
        return None

    seed(None)
    with monkeypatch.context() as before:
        before.setattr(volume_build, "_require_whole_sessions", unchecked)
        assert derive_volume(tmp_path, SID, spec, DAYS)[0]["status"] == "written"
    with pytest.raises(ValueError, match="records no source range"):
        derive_volume(tmp_path, SID, spec, DAYS, resume=True)
    session = tmp_path / "session.json"
    session.write_text(
        json.dumps(
            {
                "template": DAYS.template,
                "timezone": DAYS.timezone,
                "days": [
                    {"label": DAY.label, "startUnix": DAY.start_unix, "endUnix": DAY.end_unix}
                ],
            }
        )
    )
    args = ["volume", "--root", str(tmp_path), "--session", str(session), "--symbol", "NQ"]
    capsys.readouterr()
    assert data_cli.main([*args, "--series", SID.series, "--threshold", "1000", "--from", "1s"]) == 1
    assert "do not cover 1 requested session" in capsys.readouterr().err


def test_the_readme_commands_build_two_approximations_side_by_side(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stored_seconds(tmp_path, DAYS)
    session = tmp_path / "session.json"
    session.write_text(
        json.dumps(
            {
                "template": DAYS.template,
                "timezone": DAYS.timezone,
                "days": [
                    {"label": DAY.label, "startUnix": DAY.start_unix, "endUnix": DAY.end_unix}
                ],
            }
        )
    )
    common = [
        "volume",
        "--root",
        str(tmp_path),
        "--session",
        str(session),
        "--symbol",
        "NQ",
        "--series",
        SID.series,
        "--threshold",
        "1000",
    ]
    from_seconds = [*common, "--from", "1s"]
    compare = [*common, "--archive", str(archive(tmp_path / "trades.dbn")), "--compare-1s"]
    assert data_cli.main(from_seconds) == 0
    (stored,) = json.loads(capsys.readouterr().out)
    kept = Path(stored["path"]).read_bytes()
    assert data_cli.main(compare) == 0
    reference, approximation = json.loads(capsys.readouterr().out)
    assert [r["source"] for r in (stored, reference, approximation)] == [
        "ohlcv-1s",
        "trades",
        "trades-1s",
    ]
    assert len({r["path"] for r in (stored, reference, approximation)}) == 3
    assert (stored["volume"], approximation["volume"]) == (1200, 2450)
    assert Path(stored["path"]).read_bytes() == kept
    # In either order, and resumed: each command finds its own dataset current.
    for args in (from_seconds, compare):
        assert data_cli.main([*args, "--resume"]) == 0
        assert {r["status"] for r in json.loads(capsys.readouterr().out)} == {"unchanged"}
    assert len(volume_inventory(tmp_path)) == 3


def test_session_open_is_inclusive_and_resets_do_not_carry_volume(tmp_path: Path) -> None:
    next_day = SessionDay("2026-06-16", 110, 120)
    days = replace(DAYS, days=(DAY, next_day))
    trades = [VolumeInput.trade(100 * NS, 10, 700, 42), VolumeInput.trade(110 * NS, 12, 300, 42)]
    reports = build_volume_archive(
        archive(tmp_path / "trades.dbn", trades, end=120 * NS),
        tmp_path,
        SID,
        VolumeSpec(1000),
        days,
    )
    assert [r["volume"] for r in reports] == [700, 300]
    assert all(r["full_bars"] == 0 for r in reports)
    assert [read_volume_partition(Path(str(r["path"])))[0].available_ns for r in reports] == [
        110 * NS,
        120 * NS,
    ]


def test_nanosecond_precision_beyond_float_integer_range(tmp_path: Path) -> None:
    day = replace(DAY, start_unix=1_800_000_000, end_unix=1_800_000_010)
    days = replace(DAYS, days=(day,))
    spec = VolumeSpec(1)
    builder = VolumeAccumulator(spec, day)
    bars = builder.push(VolumeInput.trade(day.start_unix * NS + 123, 10, 1, 42))
    path = write_volume_day(tmp_path, SID, spec, days, day, bars, source="fixture")
    assert read_volume_partition(path)[0].end_ns == 1_800_000_000_000_000_123


@pytest.mark.parametrize(
    "edits,match",
    [
        ({"flags": dbn.F_BAD_TS_RECV}, "F_BAD_TS_RECV"),
        ({"price": 2**63 - 1}, "invalid trade"),
        ({"size": 0}, "invalid trade"),
        ({"action": dbn.Action.CANCEL}, "invalid trade"),
    ],
)
def test_invalid_vendor_trade_is_not_silently_dropped(
    tmp_path: Path,
    edits: dict[str, Any],
    match: str,
) -> None:
    path = archive(tmp_path / "trades.dbn", record_edits=edits)
    with pytest.raises(ValueError, match=match):
        build_volume_archive(path, tmp_path, SID, VolumeSpec(1000), DAYS)


def test_readback_failure_preserves_previous_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bedivere.data.lake import volume_store

    spec = VolumeSpec(1000)
    builder = VolumeAccumulator(spec, DAY)
    bars = [b for t in TRADES for b in builder.push(t)] + builder.finish()
    path = write_volume_day(tmp_path, SID, spec, DAYS, DAY, bars, source="original")
    before = path.read_bytes()

    def broken_read(_path: Path) -> list[VolumeBar]:
        return []

    monkeypatch.setattr(volume_store, "read_volume_partition", broken_read)
    with pytest.raises(ValueError, match="read-back"):
        write_volume_day(tmp_path, SID, spec, DAYS, DAY, bars, source="replacement")
    assert path.read_bytes() == before
    assert not list(tmp_path.rglob("*.tmp"))


def test_time_bar_cli_and_split_refusal(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    seed_time_bars(tmp_path)
    session = tmp_path / "session.json"
    session.write_text(
        json.dumps(
            {
                "template": DAYS.template,
                "timezone": DAYS.timezone,
                "days": [
                    {"label": DAY.label, "startUnix": DAY.start_unix, "endUnix": DAY.end_unix}
                ],
            }
        )
    )
    args = [
        "volume",
        "--root",
        str(tmp_path),
        "--from",
        "1s",
        "--session",
        str(session),
        "--symbol",
        "NQ",
        "--series",
        SID.series,
        "--threshold",
        "1000",
        "--dry-run",
    ]
    assert data_cli.main(args) == 0
    assert json.loads(capsys.readouterr().out)[0]["volume"] == 2450
    assert not (tmp_path / "event_bars").exists()
    assert data_cli.main([*args, "--boundary", "split_trade"]) == 1
    assert "cannot be split" in capsys.readouterr().err


def test_compressed_archive_matches_uncompressed(tmp_path: Path) -> None:
    import io

    path = archive(tmp_path / "trades.dbn")
    compressed = io.BytesIO()
    transcoder = dbn.Transcoder(compressed, dbn.Encoding.DBN, dbn.Compression.ZSTD)
    transcoder.write(path.read_bytes())
    transcoder.finish()
    zipped = tmp_path / "trades.dbn.zst"
    zipped.write_bytes(compressed.getvalue())
    spec = VolumeSpec(1000)
    build_volume_archive(path, tmp_path, SID, spec, DAYS)
    target = volume_path(tmp_path, SID, spec, DAYS, DAY.label)
    expected = read_volume_partition(target)
    build_volume_archive(zipped, tmp_path, SID, spec, DAYS)
    assert read_volume_partition(target) == expected
