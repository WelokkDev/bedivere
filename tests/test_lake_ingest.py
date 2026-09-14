"""bedivere.data.lake.ingest — DBN in, canonical Parquet out.

Every fixture here is a REAL DBN file, encoded with the vendor's own decoder and
read back through the production path: normalising a vendor's conventions is the
kind of work that looks right and is off by one period, one factor of 1e9, or one
contract, so the assertions are on the bytes rather than on a mock.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.lake_support import SKIP_REASON

pytest.importorskip("duckdb", reason=SKIP_REASON)
pytest.importorskip("pandas", reason=SKIP_REASON)
databento_dbn = pytest.importorskip("databento_dbn", reason=SKIP_REASON)

from bedivere.core.types import Candle, Timeframe  # noqa: E402
from bedivere.data.lake.ingest import (  # noqa: E402
    OUTRIGHT,
    IngestError,
    SymbolResolver,
    build_symbology,
    causal_front_months,
    check_complete,
    contract_pattern,
    ingest_dbn,
    observed_range,
    read_range,
)
from bedivere.data.lake.layout import RollRule, local_continuous, stored_labels  # noqa: E402
from bedivere.data.lake.read import read_bars, read_table, roll_boundaries  # noqa: E402
from tests.helpers import eth_session_days  # noqa: E402

SID = local_continuous("GLBX.MDP3", "NQ", RollRule.VOLUME, 0)
LABELS = ["2026-06-15", "2026-06-16", "2026-06-17"]
DAYS = eth_session_days(LABELS)

FRONT, BACK = 4_104_058, 4_104_177  # NQZ5 and NQH6 in the fixtures below
SPREAD_ID, MICRO_ID = 9_001, 9_002

NS = 1_000_000_000
SCALE = 1_000_000_000


def _mapping(raw_symbol: str, instrument_id: int) -> Any:
    """One symbology entry, in the shape `Metadata` accepts and reads back."""
    return SimpleNamespace(
        raw_symbol=raw_symbol,
        intervals=[
            SimpleNamespace(
                start_date=dt.date(2020, 1, 1), end_date=dt.date(2030, 1, 1), symbol=str(instrument_id)
            )
        ],
    )


DEFAULT_MAPPINGS = [
    _mapping("NQZ5", FRONT),
    _mapping("NQH6", BACK),
    _mapping("NQZ5-NQH6", SPREAD_ID),  # a calendar spread: a price DIFFERENCE
    _mapping("MNQZ5", MICRO_ID),  # a different product in the same batch job
]


def _record(instrument_id: int, close_ts: int, price: float, volume: int) -> Any:
    """One 1s OHLCV record. `ts_event` marks the interval START, so a bar that
    CLOSES at `close_ts` carries `close_ts - 1`."""
    return databento_dbn.OHLCVMsg(
        rtype=databento_dbn.RType.OHLCV_1S,
        publisher_id=1,
        instrument_id=instrument_id,
        ts_event=(close_ts - 1) * NS,
        open=int(price * SCALE),
        high=int((price + 2.0) * SCALE),
        low=int((price - 1.0) * SCALE),
        close=int((price + 0.5) * SCALE),
        volume=volume,
    )


def write_dbn(
    path: Path,
    records: list[Any],
    *,
    mappings: list[Any] | None = None,
    dataset: str = "GLBX.MDP3",
    declared_end_unix: int | None = None,
) -> Path:
    """A real, decodable DBN file: encoded header + fixed-size records.

    Uncompressed, because `zstandard` is not a dependency here;
    `test_compression_is_inferred_from_the_name` covers that mapping.
    """
    stamps = [r.ts_event // NS for r in records] or [0]
    meta = databento_dbn.Metadata(
        dataset=dataset,
        start=min(stamps) * NS,
        end=(declared_end_unix if declared_end_unix is not None else max(stamps)) * NS,
        stype_out=databento_dbn.SType.INSTRUMENT_ID,
        stype_in=databento_dbn.SType.PARENT,
        schema=databento_dbn.Schema.OHLCV_1S,
        symbols=["NQ.FUT"],
        mappings=DEFAULT_MAPPINGS if mappings is None else mappings,
    )
    path.write_bytes(meta.encode() + b"".join(bytes(r) for r in records))
    return path


def _one_day_records(day_index: int, instrument_id: int, count: int, price: float = 100.0) -> list[Any]:
    day = DAYS.days[day_index]
    return [
        _record(instrument_id, day.start_unix + 1 + i, price + i, volume=10)
        for i in range(count)
    ]


# ---------- the header, read without decoding the body ----------


def test_read_range_reports_what_the_file_claims(tmp_path: Path) -> None:
    path = write_dbn(tmp_path / "nq.dbn", _one_day_records(0, FRONT, 5))
    declared = read_range(path)
    assert declared.dataset == "GLBX.MDP3"
    assert declared.schema == "ohlcv-1s"
    assert declared.start_unix == DAYS.days[0].start_unix  # ts_event of the first bar


def test_compression_is_inferred_from_the_name(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="not a .dbn or .dbn.zst file"):
        read_range(tmp_path / "nq.parquet")


def test_check_complete_catches_a_truncated_download(tmp_path: Path) -> None:
    """An interrupted download keeps its truthful header while holding only the
    bytes that arrived, so everything downstream succeeds over the fraction that
    made it."""
    records = _one_day_records(0, FRONT, 5)
    honest = write_dbn(tmp_path / "full.dbn", records)
    assert check_complete(honest).records == 5

    truncated = write_dbn(
        tmp_path / "short.dbn", records, declared_end_unix=DAYS.days[0].start_unix + 30 * 86_400
    )
    with pytest.raises(IngestError, match="looks truncated"):
        check_complete(truncated)
    # A genuine trailing gap (a Friday end, a holiday) is still admitted.
    assert check_complete(truncated, max_trailing_gap_days=40).records == 5


def test_a_file_with_no_bars_is_loud(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="no OHLCV records"):
        observed_range(write_dbn(tmp_path / "empty.dbn", []))


# ---------- symbology ----------


def test_build_symbology_reads_the_headers_intervals(tmp_path: Path) -> None:
    from bedivere.data.lake.ingest import read_metadata

    resolver = build_symbology(read_metadata(write_dbn(tmp_path / "nq.dbn", _one_day_records(0, FRONT, 2))))
    assert len(resolver) == 4
    assert resolver.resolve(FRONT, DAYS.days[0].start_unix) == "NQZ5"
    assert resolver.resolve(999_999, DAYS.days[0].start_unix) == ""  # unknown, not a guess


def test_symbol_resolver_is_time_aware_for_reused_instrument_ids() -> None:
    """Ids are unique only within a day and get reused, so over a multi-year
    archive the same id can name an outright in one era and a spread in another.
    Flattened, the outright filter then drops real bars silently."""
    resolver = SymbolResolver(
        {
            "NQM0": [{"start_date": dt.date(2019, 3, 5), "end_date": dt.date(2020, 7, 5), "symbol": "16908"}],
            "NQU8-NQM9": [{"start_date": dt.date(2018, 3, 4), "end_date": dt.date(2018, 10, 7), "symbol": "16908"}],
            "NQZ5": [{"start_date": dt.date(2025, 3, 9), "end_date": dt.date(2026, 8, 8), "symbol": "999"}],
        }
    )
    assert resolver.ambiguous_ids == [16908]

    def at(iso: str) -> int:
        return int(dt.datetime.fromisoformat(iso + "T12:00:00+00:00").timestamp())

    assert resolver.resolve(16908, at("2020-01-15")) == "NQM0"
    assert resolver.resolve(16908, at("2018-06-15")) == "NQU8-NQM9"
    # Outside every interval for that id: unknown, not a guess.
    assert resolver.resolve(16908, at("2024-01-15")) == ""
    # Unambiguous ids take the fast path and ignore time entirely.
    assert resolver.resolve(999, at("1999-01-01")) == "NQZ5"


def test_symbology_intervals_are_half_open() -> None:
    resolver = SymbolResolver(
        {
            "A": [{"start_date": dt.date(2020, 1, 1), "end_date": dt.date(2020, 2, 1), "symbol": "7"}],
            "B": [{"start_date": dt.date(2020, 2, 1), "end_date": dt.date(2020, 3, 1), "symbol": "7"}],
        }
    )

    def at(iso: str) -> int:
        return int(dt.datetime.fromisoformat(iso + "T00:00:00+00:00").timestamp())

    assert resolver.resolve(7, at("2020-01-31")) == "A"
    assert resolver.resolve(7, at("2020-02-01")) == "B"  # end_date is exclusive


def test_contract_pattern_separates_a_micro_from_its_full_size_sibling() -> None:
    # One batch job can cover ES.FUT + NQ.FUT + MES.FUT + MNQ.FUT, so ranking
    # front months on the generic OUTRIGHT regex would compare ES volume with NQ.
    es, nq = contract_pattern("ES"), contract_pattern("NQ")
    mes, mnq = contract_pattern("MES"), contract_pattern("MNQ")
    assert es.match("ESZ5") and not es.match("MESZ5")
    assert nq.match("NQZ5") and not nq.match("MNQZ5")
    assert mes.match("MESZ5") and not mes.match("ESZ5")
    assert mnq.match("MNQZ5") and not mnq.match("NQZ5")
    for symbol in ("ESZ5", "NQZ5", "MESZ5", "MNQZ5"):
        assert OUTRIGHT.match(symbol)  # why OUTRIGHT alone is not enough

    assert not nq.match("NQM6-NQZ6")  # a calendar spread prices a DIFFERENCE
    assert not nq.match("NQ") and not nq.match("NQZ")
    assert nq.match("NQZ25")  # two-digit year


# ---------- the normalisation ----------


def test_close_stamps_and_price_scale_are_normalised_once(tmp_path: Path) -> None:
    """The vendor stamps an interval's START and prices in int64 at 1e-9;
    everything downstream of this module assumes close stamps and floats."""
    day = DAYS.days[0]
    path = write_dbn(tmp_path / "nq.dbn", [_record(FRONT, day.start_unix + 1, 21_000.25, 7)])
    report = ingest_dbn(path, tmp_path / "lake", SID, Timeframe.S1, DAYS)

    assert report.bars_written == 1
    (bar,) = read_bars(tmp_path / "lake", SID, Timeframe.S1, day.start_unix, day.end_unix)
    assert bar.timestamp == day.start_unix + 1  # ts_event + one period
    assert bar.open == 21_000.25
    assert bar.high == 21_002.25
    assert bar.low == 20_999.25
    assert bar.close == 21_000.75
    assert bar.volume == 7.0


def test_spreads_and_other_products_are_dropped_and_counted_apart(tmp_path: Path) -> None:
    day = DAYS.days[0]
    path = write_dbn(
        tmp_path / "nq.dbn",
        [
            _record(FRONT, day.start_unix + 1, 100.0, 1),
            _record(SPREAD_ID, day.start_unix + 2, 5.0, 1),  # a price DIFFERENCE, not an instrument
            _record(MICRO_ID, day.start_unix + 3, 100.0, 1),  # a different product entirely
            _record(FRONT, day.start_unix + 4, 101.0, 1),
        ],
    )
    report = ingest_dbn(path, tmp_path / "lake", SID, Timeframe.S1, DAYS)

    assert report.records_scanned == 4
    assert report.bars_written == 2
    assert report.records_spread == 1
    assert report.records_other_root == 1  # counted apart, so the report says WHICH
    assert "1 other-root" in report.describe() and "1 spread" in report.describe()


def test_off_session_records_are_dropped_not_misfiled(tmp_path: Path) -> None:
    day = DAYS.days[0]
    path = write_dbn(
        tmp_path / "nq.dbn",
        [
            _record(FRONT, day.start_unix - 3600, 100.0, 1),  # before the open
            _record(FRONT, day.start_unix + 1, 100.0, 1),
        ],
    )
    report = ingest_dbn(path, tmp_path / "lake", SID, Timeframe.S1, DAYS)
    assert report.records_off_session == 1
    assert report.bars_written == 1


def test_an_undefined_price_refuses(tmp_path: Path) -> None:
    # Scaled by 1e-9 the sentinel becomes a price around 9.2 billion, which
    # passes every finiteness check there is.
    day = DAYS.days[0]
    bad = databento_dbn.OHLCVMsg(
        rtype=databento_dbn.RType.OHLCV_1S, publisher_id=1, instrument_id=FRONT,
        ts_event=day.start_unix * NS, open=int(100.0 * SCALE), high=2**63 - 1,
        low=int(99.0 * SCALE), close=int(100.5 * SCALE), volume=1,
    )
    path = write_dbn(tmp_path / "nq.dbn", [bad])
    with pytest.raises(IngestError, match="UNDEF_PRICE in high"):
        ingest_dbn(path, tmp_path / "lake", SID, Timeframe.S1, DAYS)


def test_a_gap_stays_a_gap(tmp_path: Path) -> None:
    """No forward-fill: a second with no trades prints no record, and cloning the
    previous bar's high/low range would manufacture signals for anything that
    reads wicks."""
    day = DAYS.days[0]
    path = write_dbn(
        tmp_path / "nq.dbn",
        [
            _record(FRONT, day.start_unix + 1, 100.0, 1),
            _record(FRONT, day.start_unix + 60, 100.0, 1),  # 58 silent seconds
        ],
    )
    ingest_dbn(path, tmp_path / "lake", SID, Timeframe.S1, DAYS)
    got = read_bars(tmp_path / "lake", SID, Timeframe.S1, day.start_unix, day.end_unix)
    assert [c.timestamp for c in got] == [day.start_unix + 1, day.start_unix + 60]


def test_one_partition_per_session_day(tmp_path: Path) -> None:
    records: list[Any] = []
    for i in range(3):
        records.extend(_one_day_records(i, FRONT, 4))
    path = write_dbn(tmp_path / "nq.dbn", records)
    report = ingest_dbn(path, tmp_path / "lake", SID, Timeframe.S1, DAYS)

    assert report.days_written == 3
    assert report.bars_written == 12
    assert stored_labels(tmp_path / "lake", SID, Timeframe.S1) == LABELS


def test_a_dry_run_reports_without_writing(tmp_path: Path) -> None:
    path = write_dbn(tmp_path / "nq.dbn", _one_day_records(0, FRONT, 4))
    report = ingest_dbn(path, tmp_path / "lake", SID, Timeframe.S1, DAYS, dry_run=True)
    assert report.bars_written == 4
    assert stored_labels(tmp_path / "lake", SID, Timeframe.S1) == []


def test_a_roll_is_recorded_and_detectable_in_the_bars(tmp_path: Path) -> None:
    records = _one_day_records(0, FRONT, 3) + _one_day_records(1, BACK, 3)
    path = write_dbn(tmp_path / "nq.dbn", records)
    report = ingest_dbn(path, tmp_path / "lake", SID, Timeframe.S1, DAYS)

    assert report.rolls == ((LABELS[0], "NQZ5"), (LABELS[1], "NQH6"))
    table = read_table(tmp_path / "lake", SID, Timeframe.S1, 0, DAYS.days[-1].end_unix)
    assert roll_boundaries(table) == [DAYS.days[1].start_unix + 1]


# ---------- causal front-month selection ----------


def _two_contract_archive(tmp_path: Path) -> Path:
    """Volume leadership flips after day 0: NQZ5 leads day 0, NQH6 leads days 1-2."""
    records: list[Any] = []
    for i in range(3):
        day = DAYS.days[i]
        front_volume, back_volume = (100, 10) if i == 0 else (10, 100)
        records.append(_record(FRONT, day.start_unix + 1, 100.0, front_volume))
        records.append(_record(BACK, day.start_unix + 2, 500.0, back_volume))
    return write_dbn(tmp_path / "nq.dbn", records)


def test_front_months_rank_on_the_previous_day_not_the_current_one(tmp_path: Path) -> None:
    """Same-day volume is not knowable until the day is over, so ranking on it
    starts a roll day in the new contract from the open — lookahead landing
    exactly where price gaps."""
    from bedivere.data.lake.ingest import read_metadata

    path = _two_contract_archive(tmp_path)
    symbology = build_symbology(read_metadata(path))
    front = causal_front_months(path, symbology, DAYS, 1, "NQ")

    assert front[0] == FRONT  # no predecessor: ranked on itself, warm-up not tradeable
    assert front[1] == FRONT  # day 1 trades what led on day 0, though NQH6 led day 1
    assert front[2] == BACK  # and only now does the roll take effect


def test_a_day_without_records_does_not_leak_same_day_volume_into_the_next(
    tmp_path: Path,
) -> None:
    """A holiday the calendar lists, or a hole in the archive, right before a
    roll. Reaching back exactly one slot found nothing there and ranked the next
    day on its own volume — the leak D-1 exists to prevent, on the one day it
    costs a roll gap. The last winner actually observed is carried instead."""
    from bedivere.data.lake.ingest import read_metadata

    days = eth_session_days([*LABELS, "2026-06-18"])
    records: list[Any] = []
    for i, day in enumerate(days.days):
        if i == 1:
            continue  # nothing at all on 2026-06-16
        front_volume, back_volume = (100, 10) if i == 0 else (10, 100)
        records.append(_record(FRONT, day.start_unix + 1, 100.0, front_volume))
        records.append(_record(BACK, day.start_unix + 2, 500.0, back_volume))
    path = write_dbn(tmp_path / "nq.dbn", records)
    symbology = build_symbology(read_metadata(path))
    front = causal_front_months(path, symbology, days, 1, "NQ")

    assert 1 not in front  # nothing to keep on a day with no records
    assert front[2] == FRONT  # day 2 trades what led on day 0: day 1 said nothing
    assert front[3] == BACK  # the roll lands a day later than the leak put it


def test_ingesting_with_front_months_keeps_one_contract_per_day(tmp_path: Path) -> None:
    from bedivere.data.lake.ingest import read_metadata

    path = _two_contract_archive(tmp_path)
    symbology = build_symbology(read_metadata(path))
    front = causal_front_months(path, symbology, DAYS, 1, "NQ")
    ingest_dbn(path, tmp_path / "lake", SID, Timeframe.S1, DAYS, front_months=front)

    for i, expected in enumerate([FRONT, FRONT, BACK]):
        day = DAYS.days[i]
        table = read_table(tmp_path / "lake", SID, Timeframe.S1, day.start_unix, day.end_unix)
        assert table.instrument_id == [expected], LABELS[i]
        # And the price follows the contract, which is what a roll gap IS.
        assert table.open == [100.0 if expected == FRONT else 500.0]


def test_an_unsupported_roll_rule_says_so_rather_than_guessing(tmp_path: Path) -> None:
    from bedivere.data.lake.ingest import read_metadata

    path = _two_contract_archive(tmp_path)
    symbology = build_symbology(read_metadata(path))
    with pytest.raises(NotImplementedError, match="open-interest or expiry data"):
        causal_front_months(path, symbology, DAYS, 1, "NQ", roll=RollRule.OPEN_INTEREST)


# ---------- through the CLI ----------


def test_the_cli_ingests_derives_and_inventories(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The spec supplies the session calendar AND the series identity, so an
    ingest cannot land in a series the spec that drove it will never read."""
    import json

    from bedivere.cli import data as data_cli

    lake = tmp_path / "lake"
    # A full minute of 1s bars, so the 1s -> 15s rung has something to reduce.
    records = [
        _record(FRONT, DAYS.days[0].start_unix + 1 + i, 100.0 + (i % 7), volume=10) for i in range(60)
    ]
    archive = write_dbn(tmp_path / "nq.dbn", records)
    session = tmp_path / "session.json"
    session.write_text(
        json.dumps(
            {
                "template": DAYS.template,
                "timezone": DAYS.timezone,
                "days": [
                    {"label": d.label, "startUnix": d.start_unix, "endUnix": d.end_unix}
                    for d in DAYS.days
                ],
            }
        ),
        encoding="utf-8",
    )

    assert data_cli.main(
        [
            "ingest", "--archive", str(archive), "--session", str(session),
            "--symbol", "NQ", "--series", "local.v.0", "--root", str(lake),
        ]
    ) == 0
    out, err = capsys.readouterr()
    assert "60 bars across 1 session-days" in err
    # Chained, not fanned out from the raw series, so only the first rung pays
    # for reading the big one.
    for rung in ("1s -> 15s", "15s -> 5m", "5m -> 15m", "30m -> 1h", "2h -> 4h"):
        assert rung in err, rung
    assert "every derived timeframe covers exactly its source's days" in out

    # 60 one-second bars -> 4 fifteens.
    assert len(read_bars(lake, SID, Timeframe.S15, 0, DAYS.days[-1].end_unix)) == 4

    assert data_cli.main(["list", "--root", str(lake)]) == 0
    listed = capsys.readouterr().out
    assert "GLBX.MDP3:NQ:local.v.0" in listed
    assert "nq.dbn" in listed and "resample:1s" in listed  # lineage, read from the footers


def test_the_cli_refuses_an_archive_whose_dataset_disagrees(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The dataset is part of the series identity, so this would mislabel every
    # partition the run writes.
    import json

    from bedivere.cli import data as data_cli

    archive = write_dbn(tmp_path / "nq.dbn", _one_day_records(0, FRONT, 3), dataset="XNAS.ITCH")
    session = tmp_path / "session.json"
    session.write_text(
        json.dumps(
            {
                "template": DAYS.template,
                "timezone": DAYS.timezone,
                "days": [{"label": DAYS.days[0].label, "startUnix": DAYS.days[0].start_unix, "endUnix": DAYS.days[0].end_unix}],
            }
        ),
        encoding="utf-8",
    )
    assert data_cli.main(
        ["ingest", "--archive", str(archive), "--session", str(session), "--symbol", "NQ",
         "--root", str(tmp_path / "lake")]
    ) == 1
    assert "would mislabel every partition" in capsys.readouterr().err


def test_the_cli_needs_exactly_one_source_of_session_geometry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from bedivere.cli import data as data_cli

    archive = write_dbn(tmp_path / "nq.dbn", _one_day_records(0, FRONT, 2))
    assert data_cli.main(["ingest", "--archive", str(archive), "--root", str(tmp_path)]) == 1
    assert "exactly one of --spec" in capsys.readouterr().err


def test_the_cli_derives_a_named_rung_and_refuses_a_ladderless_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    from bedivere.cli import data as data_cli
    from bedivere.data.lake.schema import BarBatch
    from bedivere.data.lake.writer import write_day

    lake = tmp_path / "lake"
    day = DAYS.days[0]
    seeded = [
        Candle(
            timestamp=day.start_unix + 1 + i, open=100.0, high=101.0, low=99.0, close=100.5, volume=1.0
        )
        for i in range(30)
    ]
    write_day(
        lake, SID, Timeframe.S1, day.label,
        BarBatch.from_candles(seeded, instrument_id=FRONT),
        source="nq.dbn",
    )
    session = tmp_path / "session.json"
    session.write_text(
        json.dumps(
            {
                "template": DAYS.template,
                "timezone": DAYS.timezone,
                "days": [{"label": day.label, "startUnix": day.start_unix, "endUnix": day.end_unix}],
            }
        ),
        encoding="utf-8",
    )
    base = ["derive", "--session", str(session), "--symbol", "NQ", "--root", str(lake)]

    assert data_cli.main([*base, "--from", "1s", "--to", "15s"]) == 0
    assert "1s -> 15s" in capsys.readouterr().err
    assert len(read_bars(lake, SID, Timeframe.S15, 0, day.end_unix)) == 2

    # 4h is the top of every chain, so nothing derives FROM it.
    assert data_cli.main([*base, "--from", "4h"]) == 1
    assert "no derive ladder starts at 4h" in capsys.readouterr().err


def test_the_cli_refuses_a_series_token_it_cannot_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    from bedivere.cli import data as data_cli

    session = tmp_path / "session.json"
    day = DAYS.days[0]
    session.write_text(
        json.dumps(
            {
                "template": DAYS.template,
                "timezone": DAYS.timezone,
                "days": [{"label": day.label, "startUnix": day.start_unix, "endUnix": day.end_unix}],
            }
        ),
        encoding="utf-8",
    )
    assert data_cli.main(
        ["derive", "--session", str(session), "--symbol", "NQ", "--series", "local.v",
         "--root", str(tmp_path / "lake")]
    ) == 1
    assert "is not a series identity bedivere can name" in capsys.readouterr().err


def test_the_cli_names_the_missing_extra_rather_than_the_exception_class(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A machine missing a dependency is not a bedivere bug, and must not read
    like one: the seams already say what is missing, so the CLI renders them as
    a plain refusal rather than through the blanket `error: <ClassName>:` path."""
    import sys as _sys

    from bedivere.cli import data as data_cli

    monkeypatch.setitem(_sys.modules, "requests", None)
    assert data_cli.main(["jobs"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: the Databento HTTP edge needs `requests`")
    assert "bedivere[lake]" in err
    assert "VendorHttpUnavailable" not in err  # no exception class in the operator's face
