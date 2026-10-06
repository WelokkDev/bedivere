"""Trade-data API: retained fields, verification, selection, lineage, and totals.

Every archive here is synthetic and strategy-neutral.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import os
import random
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from tests.lake_support import SKIP_REASON

pytest.importorskip("duckdb", reason=SKIP_REASON)
pytest.importorskip("pandas", reason=SKIP_REASON)
dbn = pytest.importorskip("databento_dbn", reason=SKIP_REASON)

import duckdb  # noqa: E402
import pandas as pd  # noqa: E402  # pyright: ignore[reportMissingTypeStubs]

from bedivere.core.session_days import SessionDay, SessionDays  # noqa: E402
from bedivere.core.types import Timeframe  # noqa: E402
from bedivere.data.lake import provenance, trades, volume_research  # noqa: E402
from bedivere.data.lake.ingest import IngestError  # noqa: E402
from bedivere.data.lake.layout import SeriesId  # noqa: E402
from bedivere.data.lake.provenance import ProvenanceError  # noqa: E402
from bedivere.data.lake.schema import BarBatch  # noqa: E402
from bedivere.data.lake.trade_archive import (  # noqa: E402
    policy_key,
    prepare_trade_archive,
    receipt_path,
)
from bedivere.data.lake.trades import (  # noqa: E402
    INELIGIBLE,
    SIDE_CODES,
    Aggressor,
    ReadStatus,
    Trade,
    TradeIssue,
    TradeRead,
    TradeTotals,
    open_trade_archive,
    read_trades,
    summarize_trades,
)
from bedivere.data.lake.trades_frame import TRADE_COLUMNS, load_trade_frame  # noqa: E402
from bedivere.data.lake.writer import write_day  # noqa: E402

# pandas is optional and shipped without complete typing for dynamic columns.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false

NS = 1_000_000_000
DAY = SessionDay("2026-06-15", 100, 110)
NEXT = SessionDay("2026-06-16", 110, 120)
DAYS = SessionDays("test", "UTC", (DAY, NEXT))
SID = SeriesId("GLBX.MDP3", "NQ", "raw.NQZ5")
DEFERRED = SeriesId("GLBX.MDP3", "NQ", "raw.NQH6")
SYMBOLS = {"NQZ5": 42, "NQH6": 43, "ESZ5": 44}
BIG = 1_800_000_000  # Unix seconds whose nanoseconds exceed 2**53


def header(schema: Any = None, start: int = 100 * NS, end: int = 120 * NS, **edits: Any) -> bytes:
    fields: dict[str, Any] = {
        "dataset": "GLBX.MDP3",
        "schema": dbn.Schema.TRADES if schema is None else schema,
        "start": start,
        "end": end,
        "stype_in": dbn.SType.RAW_SYMBOL,
        "stype_out": dbn.SType.INSTRUMENT_ID,
        "symbols": sorted(SYMBOLS),
        "mappings": [
            SimpleNamespace(
                raw_symbol=symbol,
                intervals=[
                    SimpleNamespace(
                        start_date=dt.date(1970, 1, 1),
                        end_date=dt.date(2100, 1, 1),
                        symbol=str(iid),
                    )
                ],
            )
            for symbol, iid in SYMBOLS.items()
        ],
    }
    fields.update(edits)
    return dbn.Metadata(**fields).encode()


def trade(ts_recv: int, **fields: Any) -> Any:
    values: dict[str, Any] = {
        "publisher_id": 1,
        "instrument_id": 42,
        "ts_event": ts_recv - 7,
        "ts_recv": ts_recv,
        "price": 10 * NS,
        "size": 1,
        "action": dbn.Action.TRADE,
        "side": dbn.Side.BID,
        "depth": 0,
        "flags": 0,
        "ts_in_delta": 0,
        "sequence": 0,
    }
    values.update(fields)
    return dbn.TradeMsg(**values)


def encoded(record: Any, *, action: bytes | None = None, side: bytes | None = None) -> bytes:
    """Raw record bytes, optionally with codes no vendor enum can construct."""
    raw = bytearray(bytes(record))
    if action is not None:
        raw[28:29] = action
    if side is not None:
        raw[29:30] = side
    return bytes(raw)


def zstd(body: bytes) -> bytes:
    buffer = io.BytesIO()
    transcoder = dbn.Transcoder(buffer, dbn.Encoding.DBN, dbn.Compression.ZSTD)
    transcoder.write(body)
    transcoder.finish()
    return buffer.getvalue()


def bare_decode(compressed: bytes) -> tuple[int, bool]:
    """What the vendor decoder alone makes of `compressed`: how many records it
    yields, the header among them, and whether it is left holding part of one."""
    decoder = dbn.DBNDecoder(compression=dbn.Compression.ZSTD)
    decoder.write(compressed)
    return len(decoder.decode()), bool(decoder.buffer())


def write(path: Path, records: list[Any], **meta: Any) -> Path:
    body = header(**meta) + b"".join(r if isinstance(r, bytes) else bytes(r) for r in records)
    path.write_bytes(zstd(body) if path.name.endswith(".zst") else body)
    return path


def collect(read: TradeRead) -> list[Trade]:
    with read:
        return [t for batch in read for t in batch]


def read_all(path: Path, series: SeriesId = SID, /, **selection: Any) -> list[Trade]:
    return collect(read_trades(path, series, **selection))


# ---------- the record contract ----------


def test_retained_fields_are_exact_and_timestamps_never_touch_floats(tmp_path: Path) -> None:
    start = BIG * NS
    flags = 0xFF & ~dbn.F_BAD_TS_RECV  # every other bit, including reserved bit 0
    path = write(
        tmp_path / "t.dbn",
        [
            trade(
                start + 123,
                ts_event=start - 4_567,
                price=29_140_250_000_000,
                size=17,
                sequence=4_000_000_001,
                publisher_id=7,
                flags=flags,
                side=dbn.Side.ASK,
            ),
            trade(start + 124, price=-2_500_000_000, side=dbn.Side.NONE, publisher_id=7),
        ],
        start=start,
        end=start + 10 * NS,
    )
    first, second = read_all(path, start_ns=start, end_ns=start + NS)
    assert float(start + 123) != start + 123  # float64 cannot hold these
    assert (first.ts_recv, first.ts_event) == (start + 123, start - 4_567)
    assert (first.price_fixed, first.size, first.sequence) == (29_140_250_000_000, 17, 4_000_000_001)
    assert first.price == 29_140_250_000_000 / 1_000_000_000 == 29140.25
    assert (first.publisher_id, first.instrument_id, first.contract) == (7, 42, "NQZ5")
    assert (first.flags, first.issues, first.eligible) == (flags, TradeIssue(0), True)
    assert (first.side_code, first.aggressor) == ("A", Aggressor.SELL)
    assert (first.ordinal, second.ordinal) == (0, 1)
    assert (second.price_fixed, second.side_code, second.aggressor) == (-2_500_000_000, "N", "unknown")
    assert not hasattr(first, "depth")  # never exposed, observed or inserted


def test_receive_and_event_clocks_stay_distinct(tmp_path: Path) -> None:
    path = write(
        tmp_path / "t.dbn",
        [
            trade(101 * NS, ts_event=104 * NS),  # event clock inside the window, receive outside
            trade(104 * NS, ts_event=99 * NS),  # receive inside, event long before
            trade(104 * NS + 1, ts_event=dbn.UNDEF_TIMESTAMP),
            trade(104 * NS + 2, ts_event=2**63 + 5),
        ],
    )
    got = read_all(path, start_ns=102 * NS, end_ns=105 * NS)
    assert [t.ordinal for t in got] == [1, 2, 3]
    assert [t.ts_event for t in got] == [99 * NS, None, None]
    assert [t.issues for t in got] == [TradeIssue(0), *[TradeIssue.UNDEFINED_TS_EVENT] * 2]
    assert all(t.eligible for t in got)  # event time plays no part in receive-time bars


def test_side_codes_are_the_installed_vendor_definitions() -> None:
    vendor = {str(side) for side in dbn.Side.variants()}
    assert vendor == set(SIDE_CODES) == {"A", "B", "N"}
    assert (str(dbn.Side.ASK), str(dbn.Side.BID), str(dbn.Side.NONE)) == ("A", "B", "N")
    # Databento, action T: A = the aggressor was a seller, B = a buyer, N = none.
    assert dict(SIDE_CODES) == {"A": "sell", "B": "buy", "N": "unknown"}


def test_undefined_codes_are_kept_and_flagged_never_guessed(tmp_path: Path) -> None:
    path = write(
        tmp_path / "t.dbn",
        [
            encoded(trade(101 * NS), side=b"X"),
            encoded(trade(102 * NS), side=b"\xff"),
            encoded(trade(103 * NS), action=b"Z"),
            trade(104 * NS, side=dbn.Side.NONE),
        ],
    )
    x, high, action, none = read_all(path)
    assert (x.side_code, x.aggressor, x.issues) == ("X", "unknown", TradeIssue.UNDEFINED_SIDE)
    assert (high.side_code, high.aggressor) == ("\xff", "unknown")
    assert x.eligible and high.eligible  # the builder never used the side
    assert action.issues == TradeIssue.NON_TRADE_ACTION and not action.eligible
    assert (none.aggressor, none.issues) == ("unknown", TradeIssue(0))


def test_flags_are_kept_whole_and_only_documented_rules_decide_eligibility(tmp_path: Path) -> None:
    path = write(
        tmp_path / "t.dbn",
        [
            trade(101 * NS, flags=dbn.F_BAD_TS_RECV | 0x01),
            trade(102 * NS, price=dbn.UNDEF_PRICE),
            trade(103 * NS, size=0),
            trade(104 * NS, flags=dbn.F_MAYBE_BAD_BOOK | dbn.F_PUBLISHER_SPECIFIC | dbn.F_LAST),
        ],
    )
    bad_recv, undefined, empty, other = read_all(path)
    assert (bad_recv.flags, bad_recv.issues) == (dbn.F_BAD_TS_RECV | 0x01, TradeIssue.BAD_TS_RECV)
    assert (undefined.price_fixed, undefined.price) == (None, None)  # never 9.2e9
    assert undefined.issues == TradeIssue.UNDEFINED_PRICE
    assert empty.issues == TradeIssue.ZERO_SIZE
    assert other.flags == dbn.F_MAYBE_BAD_BOOK | dbn.F_PUBLISHER_SPECIFIC | dbn.F_LAST
    assert [t.eligible for t in (bad_recv, undefined, empty, other)] == [False, False, False, True]
    assert INELIGIBLE == (
        TradeIssue.UNDEFINED_PRICE
        | TradeIssue.ZERO_SIZE
        | TradeIssue.BAD_TS_RECV
        | TradeIssue.NON_TRADE_ACTION
    )


def test_ties_and_repeated_sequences_keep_archive_order_without_deduplication(
    tmp_path: Path,
) -> None:
    records = [trade(101 * NS, sequence=5, size=size) for size in (3, 1, 2)]
    records.append(trade(101 * NS, sequence=4, size=9))
    got = read_all(write(tmp_path / "t.dbn", records))
    assert [(t.ordinal, t.sequence, t.size) for t in got] == [(0, 5, 3), (1, 5, 1), (2, 5, 2), (3, 4, 9)]


# ---------- selection and ordering ----------


def interleaved() -> list[Any]:
    return [
        trade(101 * NS, instrument_id=43),
        trade(102 * NS),
        trade(100 * NS + 5, instrument_id=44),  # another root, earlier than its neighbours
        trade(103 * NS),
        trade(103 * NS, instrument_id=43),
        trade(104 * NS),
        trade(111 * NS),
        trade(112 * NS, instrument_id=43),
        trade(113 * NS),
    ]


@pytest.mark.parametrize("batch_records", [1, 2, 3, 1000])
def test_ordinals_and_contents_do_not_depend_on_batches_or_filters(
    tmp_path: Path, batch_records: int
) -> None:
    path = write(tmp_path / "t.dbn", interleaved())
    full = read_all(path, batch_records=batch_records)
    assert [t.ordinal for t in full] == [1, 3, 5, 6, 8]
    assert full == read_all(tmp_path / "t.dbn")
    window = read_all(path, start_ns=103 * NS, end_ns=112 * NS, batch_records=batch_records)
    assert window == [t for t in full if 103 * NS <= t.ts_recv < 112 * NS]
    with read_trades(path, SID, batch_records=batch_records) as read:
        sizes = [len(batch) for batch in read]
    assert sizes == [min(batch_records, 5 - i) for i in range(0, 5, batch_records)]
    assert [t.ordinal for t in read_all(path, DEFERRED)] == [0, 4, 7]


def test_unrelated_instruments_are_not_ordered_against_the_selection(tmp_path: Path) -> None:
    path = write(tmp_path / "t.dbn", interleaved())
    assert len(read_all(path)) == 5
    backwards = [*interleaved(), trade(110 * NS)]  # NQZ5 itself goes back in time
    with pytest.raises(ValueError, match="out of receive-time order"):
        read_all(write(tmp_path / "t.dbn", backwards))


def test_a_late_record_after_the_end_bound_is_found_or_refused(tmp_path: Path) -> None:
    window = {"start_ns": 100 * NS, "end_ns": 105 * NS}
    late = [trade(101 * NS), trade(106 * NS), trade(103 * NS)]
    got = read_all(write(tmp_path / "t.dbn", late), **window)
    # An early stop at 106 s would have lost this record without any error.
    assert [t.ts_recv for t in got] == [101 * NS, 103 * NS]
    disordered = [trade(101 * NS), trade(104 * NS), trade(106 * NS), trade(102 * NS)]
    with pytest.raises(ValueError, match="out of receive-time order"):
        read_all(write(tmp_path / "t.dbn", disordered), **window)
    other_root = [trade(101 * NS), trade(106 * NS), trade(102 * NS, instrument_id=44)]
    assert len(read_all(write(tmp_path / "t.dbn", other_root), **window)) == 1


def test_vendor_flagged_receive_times_are_kept_but_not_ordered(tmp_path: Path) -> None:
    records = [trade(103 * NS), trade(101 * NS, flags=dbn.F_BAD_TS_RECV), trade(104 * NS)]
    got = read_all(write(tmp_path / "t.dbn", records))
    assert [t.issues for t in got] == [TradeIssue(0), TradeIssue.BAD_TS_RECV, TradeIssue(0)]


def test_sessions_are_half_open_and_empty_selections_validate(tmp_path: Path) -> None:
    edges = [trade(100 * NS), trade(110 * NS - 1), trade(110 * NS), trade(120 * NS - 1)]
    path = write(tmp_path / "t.dbn", edges)
    got = read_all(path, sessions=DAYS)
    assert [t.session for t in got] == [DAY.label, DAY.label, NEXT.label, NEXT.label]
    quiet = replace(DAYS, days=(SessionDay("2026-06-17", 105, 106),))
    with read_trades(path, SID, sessions=quiet) as read:
        assert list(read) == []
        assert read.validated and read.source.trade_records == 4
    assert read_all(path, start_ns=105 * NS, end_ns=106 * NS) == []
    with pytest.raises(ValueError, match="requested receive-time range"):
        read_all(path, start_ns=90 * NS, end_ns=101 * NS)
    with pytest.raises(ValueError, match="requested full session"):
        read_all(path, sessions=replace(DAYS, days=(SessionDay("late", 115, 125),)))
    with pytest.raises(ValueError, match="record limit"):
        read_all(write(tmp_path / "t.dbn", edges, limit=4))


def test_one_publisher_per_session_or_per_calendar_free_selection(tmp_path: Path) -> None:
    records = [trade(101 * NS), trade(111 * NS, publisher_id=2)]
    path = write(tmp_path / "t.dbn", records)
    assert len(read_all(path, sessions=DAYS)) == 2
    with pytest.raises(ValueError, match="multiple publishers in one selection"):
        read_all(path)
    same_session = [trade(101 * NS), trade(102 * NS, publisher_id=2)]
    with pytest.raises(ValueError, match="multiple publishers in one session"):
        read_all(write(tmp_path / "t.dbn", same_session), sessions=DAYS)


def seed_reference(root: Path, sid: SeriesId, day: SessionDay, ids: list[int]) -> None:
    batch = BarBatch()
    for i, iid in enumerate(ids):
        batch.append(day.start_unix + 1 + i, 10.0, 10.0, 10.0, 10.0, 1.0, iid)
    write_day(root, sid, Timeframe.S1, day.label, batch, "fixture")


def test_continuous_series_reuses_reference_contracts_and_never_selects_nothing(
    tmp_path: Path,
) -> None:
    sid = replace(SID, series="local.v.0")
    path = write(tmp_path / "t.dbn", interleaved())
    with pytest.raises(ValueError, match="needs sessions and the lake_root"):
        read_all(path, sid, sessions=DAYS)
    with pytest.raises(FileNotFoundError, match="missing reference 1s partition"):
        read_all(path, sid, sessions=DAYS, lake_root=tmp_path)
    seed_reference(tmp_path, sid, DAY, [42])
    seed_reference(tmp_path, sid, NEXT, [42, 43])
    with pytest.raises(ValueError, match="exactly one contract"):
        read_all(path, sid, sessions=DAYS, lake_root=tmp_path)
    damaged = next(tmp_path.glob("bars/**/session=2026-06-16/bars.parquet"))
    damaged.write_bytes(b"not parquet")
    with pytest.raises(duckdb.Error, match="parquet"):
        read_all(path, sid, sessions=DAYS, lake_root=tmp_path)
    seed_reference(tmp_path, sid, NEXT, [43])  # the reference lake rolled overnight
    with read_trades(path, sid, sessions=DAYS, lake_root=tmp_path) as read:
        got = [t for batch in read for t in batch]
        dependencies = read.source.selection_dependencies
    assert [(t.contract, t.session) for t in got] == [
        ("NQZ5", DAY.label),
        ("NQZ5", DAY.label),
        ("NQZ5", DAY.label),
        ("NQH6", NEXT.label),
    ]
    reference = tmp_path.glob("bars/**/session=2026-06-16/bars.parquet")
    assert dependencies[1] == (NEXT.label, trades.fingerprint(next(reference)))
    assert dependencies[1][1].startswith("bars.parquet:sha256:")


# ---------- verification, completion, and failure ----------


def test_truncated_archives_fail_whichever_compression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [trade(100 * NS + i * 100_000_000, size=i + 1) for i in range(40)]
    plain = write(tmp_path / "t.dbn", records)
    body = plain.read_bytes()
    plain.write_bytes(body[:-1])
    with pytest.raises(IngestError, match="incomplete trailing DBN record"):
        read_all(plain)
    compressed = zstd(body)
    zipped = tmp_path / "t.dbn.zst"
    monkeypatch.setattr(trades, "CHUNK_BYTES", 7)
    zipped.write_bytes(compressed)
    assert len(read_all(zipped)) == 40
    # The decoder itself stays silent here: a cut can even decode every record.
    for cut in (1, 4, 20, len(compressed) // 2, len(compressed) - 10):
        zipped.write_bytes(compressed[:-cut])
        with pytest.raises(IngestError, match="incomplete trailing"):
            read_all(zipped)
        # Refused on opening, before a read exists: a resumed build never reads.
        with pytest.raises(IngestError, match="incomplete trailing zstd frame"):
            open_trade_archive(zipped)


def test_bytes_that_change_after_hashing_invalidate_the_read(tmp_path: Path) -> None:
    path = write(tmp_path / "t.dbn", [trade(101 * NS, size=5), trade(102 * NS, size=6)])
    body = path.read_bytes()
    before = path.stat()
    with open_trade_archive(path) as archive:
        # Same length, same mtime, different bytes: only the second digest can tell.
        with path.open("r+b") as handle:
            handle.seek(len(body) - 48 + 24)
            handle.write((7).to_bytes(4, "little"))
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        read = archive.read(SID)
        with pytest.raises(IngestError, match="differ from the bytes hashed before decoding"):
            list(read)
        assert read.status is ReadStatus.FAILED and isinstance(read.error, IngestError)
    path.write_bytes(body)
    with open_trade_archive(path) as archive:
        # Identical bytes under a new inode: content verifies, identity does not.
        replacement = tmp_path / "replacement.dbn"
        replacement.write_bytes(body)
        os.replace(replacement, path)
        with pytest.raises(IngestError, match="modified or replaced during the read"):
            list(archive.read(SID))


def test_batches_are_provisional_until_the_read_is_exhausted(tmp_path: Path) -> None:
    path = write(tmp_path / "t.dbn", [trade((100 + i) * NS) for i in range(5)])
    read = read_trades(path, SID, batch_records=2)
    assert read.status is ReadStatus.PENDING
    first = next(read)
    assert len(first) == 2 and read.status is ReadStatus.READING
    with pytest.raises(ValueError, match="reading, not validated"):
        read.source  # noqa: B018
    read.close()
    assert read.status is ReadStatus.INCOMPLETE and not read.validated
    with pytest.raises(ValueError, match="incomplete, not validated"):
        read.provenance("databento")
    # Every batch has arrived, but validation ends only at exhaustion.
    read = read_trades(path, SID, batch_records=5)
    with read:
        assert len(next(read)) == 5
    assert read.status is ReadStatus.INCOMPLETE
    with read_trades(path, SID, batch_records=5) as read:
        assert sum(len(batch) for batch in read) == 5
        assert read.validated and read.selected_records == 5


def test_a_failed_read_invalidates_everything_it_produced(tmp_path: Path) -> None:
    path = write(tmp_path / "t.dbn", [trade(101 * NS), trade(102 * NS), trade(101 * NS)])
    read = read_trades(path, SID, batch_records=1)
    produced: list[tuple[Trade, ...]] = []
    with pytest.raises(ValueError, match="out of receive-time order"):
        for batch in read:
            produced.append(batch)
    assert len(produced) == 2 and read.status is ReadStatus.FAILED
    with pytest.raises(ValueError, match="failed, not validated"):
        read.source  # noqa: B018
    truncated = write(tmp_path / "short.dbn", [trade(101 * NS)])
    truncated.write_bytes(truncated.read_bytes()[:-1])
    with pytest.raises(IngestError, match="incomplete trailing"):
        summarize_trades(truncated, SID)


def test_one_archive_serves_one_read_at_a_time(tmp_path: Path) -> None:
    path = write(tmp_path / "t.dbn", [trade((100 + i) * NS) for i in range(4)])
    with open_trade_archive(path) as archive:
        first = archive.read(SID, batch_records=1)
        next(first)
        second = archive.read(SID)
        with pytest.raises(ValueError, match="another read"):
            next(second)
        assert first.status is ReadStatus.READING  # untouched by the refused read
        assert sum(len(b) for b in first) == 3 and first.validated
        again = archive.read(SID)  # sequential reads re-verify from the same handle
        assert sum(len(b) for b in again) == 4 and again.validated
    with pytest.raises(ValueError, match="archive is closed"):
        next(archive.read(SID))


def test_non_trade_records_contradict_the_stored_schema(tmp_path: Path) -> None:
    ohlcv = dbn.OHLCVMsg(
        rtype=dbn.RType.OHLCV_1S,
        publisher_id=1,
        instrument_id=42,
        ts_event=101 * NS,
        open=1,
        high=1,
        low=1,
        close=1,
        volume=1,
    )
    path = write(tmp_path / "t.dbn", [trade(101 * NS), ohlcv])
    with pytest.raises(IngestError, match="OHLCVMsg record in a trades archive"):
        read_all(path)
    with pytest.raises(IngestError, match="outside the archive's declared range"):
        read_all(write(tmp_path / "t.dbn", [trade(130 * NS)]))


# ---------- prepared caches, receipts, and manifests ----------


def mbo_input(path: Path, start: int, end: int, rows: list[tuple[int, int, int, Any]]) -> Path:
    """Daily MBO: trades (T) plus a resting fill and a snapshot trade that must be skipped."""
    records = [
        dbn.MBOMsg(
            publisher_id=1,
            instrument_id=iid,
            ts_event=ts - 3,
            ts_recv=ts,
            order_id=i,
            price=(10 + i) * NS,
            size=size,
            action=dbn.Action.TRADE,
            side=side,
            sequence=1_000 + i,
        )
        for i, (ts, iid, size, side) in enumerate(rows)
    ]
    ts = rows[0][0]
    records.append(
        dbn.MBOMsg(
            publisher_id=1, instrument_id=42, ts_event=ts, ts_recv=ts, order_id=99,
            price=NS, size=50, action=dbn.Action.FILL, side=dbn.Side.BID,
        )
    )
    records.append(
        dbn.MBOMsg(
            publisher_id=1, instrument_id=42, ts_event=ts, ts_recv=ts, order_id=98,
            price=NS, size=60, action=dbn.Action.TRADE, side=dbn.Side.BID, flags=dbn.F_SNAPSHOT,
        )
    )
    path.write_bytes(
        header(dbn.Schema.MBO, start * NS, end * NS) + b"".join(bytes(r) for r in records)
    )
    return path


def prepared(tmp_path: Path) -> Path:
    """A managed cache from two contiguous MBO days, sources in a private directory."""
    source = tmp_path / "private-source-dir"
    source.mkdir(exist_ok=True)
    mbo_input(
        source / "day-1.mbo.dbn",
        100,
        110,
        [
            (101 * NS, 42, 4, dbn.Side.BID),
            (102 * NS, 43, 2, dbn.Side.ASK),
            (103 * NS, 42, 3, dbn.Side.ASK),
            (104 * NS, 42, 5, dbn.Side.NONE),
        ],
    )
    mbo_input(
        source / "day-2.mbo.dbn",
        110,
        120,
        [(111 * NS, 42, 7, dbn.Side.BID), (112 * NS, 43, 1, dbn.Side.BID)],
    )
    return prepare_trade_archive(source, tmp_path / "lake" / "_meta" / "trade_cache")


def edit_receipt(cache: Path, change: Callable[[dict[str, Any]], object]) -> None:
    receipt = receipt_path(cache)
    payload = json.loads(receipt.read_text())
    change(payload)
    receipt.write_text(json.dumps(payload, indent=2))


def test_prepared_cache_lineage_is_checked_without_following_receipt_paths(
    tmp_path: Path,
) -> None:
    cache = prepared(tmp_path)
    for source in (tmp_path / "private-source-dir").iterdir():
        source.unlink()  # a reader that opened recorded paths would now fail
    frame = load_trade_frame(cache, DEFERRED, start_ns=100 * NS, end_ns=120 * NS)
    source = frame.attrs["trade_source"]
    assert frame["size"].tolist() == [2, 1]
    # Full-archive counts cover every decoded record, not the two selected.
    assert (source.trade_records, source.trade_volume) == (6, 22)
    assert source.lineage is not None
    assert [(i.name, i.schema, i.trades) for i in source.lineage.inputs] == [
        ("day-1.mbo.dbn", "mbo", 4),
        ("day-2.mbo.dbn", "mbo", 2),
    ]
    assert source.lineage.policy_key == cache.name.split(".")[0]
    assert source.input_schemas == ("mbo",) and source.stored_schema == "trades"
    assert source.request_stype_in is None  # the cache header's stype is not the request's
    assert any("request symbology" in item for item in source.unavailable)
    assert any("depth was set to 0" in item for item in source.limitations)
    assert any("2 of 2 input file(s) had no vendor manifest" in item for item in source.limitations)
    assert "private-source-dir" not in repr(source) and str(tmp_path) not in repr(source)
    assert "depth" not in frame.columns


def test_managed_cache_names_follow_the_existing_policy_hash(tmp_path: Path) -> None:
    cache = prepared(tmp_path)
    receipt = json.loads(receipt_path(cache).read_text())
    import hashlib

    legacy = hashlib.sha256(json.dumps(receipt["policy"], sort_keys=True).encode()).hexdigest()
    assert cache.name == f"{legacy}.trades.dbn.zst" == f"{policy_key(receipt['policy'])}.trades.dbn.zst"
    impostor = cache.with_name("0" * 64 + ".trades.dbn.zst")
    cache.rename(impostor)
    receipt_path(cache).rename(receipt_path(impostor))
    with pytest.raises(IngestError, match="managed cache name does not match"):
        open_trade_archive(impostor)
    # Any other name is not held to the convention, but its receipt still is.
    renamed = impostor.with_name("renamed.trades.dbn.zst")
    impostor.rename(renamed)
    receipt_path(impostor).rename(receipt_path(renamed))
    assert len(read_all(renamed, start_ns=100 * NS, end_ns=120 * NS)) == 4


def shift_receipt_counts(payload: dict[str, Any]) -> None:
    payload["files"][0]["trades"] -= 1
    payload["files"][1]["trades"] += 1


def add_receipt_trade(payload: dict[str, Any]) -> None:
    payload["trades"] += 1
    payload["files"][0]["trades"] += 1
    payload["files"][0]["records"] += 1


Edit = Callable[[dict[str, Any]], object]
RECEIPT_CONTRADICTIONS: list[tuple[Edit, str, bool]] = [
    (lambda p: p.update(sha256="0" * 64), "SHA256 differs from its preparation receipt", False),
    (lambda p: p["policy"].update(version=2), "unsupported extraction-policy version", False),
    (lambda p: p.update(extra=1), "receipt must be an object", False),
    (lambda p: p["files"].reverse(), "extraction order", False),
    (lambda p: p.update(volume=p["volume"] + 1), "do not add up", False),
    (lambda p: p["policy"]["inputs"][1].update(start_ns=111 * NS), "overlap or leave a gap", True),
    (lambda p: p["policy"].update(dataset="XNAS.ITCH"), "receipt policy dataset", True),
    (lambda p: p["policy"]["inputs"][0].update(start_ns=99 * NS), "recorded input ranges", True),
    (add_receipt_trade, "its receipt records 7", False),
    (shift_receipt_counts, "per-input trade counts differ", False),
]


@pytest.mark.parametrize("change,match,rename", RECEIPT_CONTRADICTIONS)
def test_contradictory_receipts_fail_before_or_after_decoding(
    tmp_path: Path, change: Edit, match: str, rename: bool
) -> None:
    cache = prepared(tmp_path)
    if rename:  # a changed policy changes the key; test the check behind the name
        renamed = cache.with_name("copy.trades.dbn.zst")
        cache.rename(renamed)
        receipt_path(cache).rename(receipt_path(renamed))
        cache = renamed
    edit_receipt(cache, change)
    with pytest.raises(IngestError, match=match):
        read_all(cache, start_ns=100 * NS, end_ns=120 * NS)


def test_a_supplied_receipt_is_checked_and_a_missing_one_invents_nothing(tmp_path: Path) -> None:
    cache = prepared(tmp_path)
    moved = tmp_path / "receipt.json"
    receipt_path(cache).rename(moved)
    with read_trades(cache, SID, start_ns=100 * NS, end_ns=120 * NS) as read:
        list(read)
        bare = read.source
    assert bare.lineage is None and bare.input_schemas is None
    assert "original input schemas" in bare.unavailable
    with read_trades(cache, SID, start_ns=100 * NS, end_ns=120 * NS, receipt=moved) as read:
        list(read)
        assert read.source.input_schemas == ("mbo",)
    with pytest.raises(FileNotFoundError, match="receipt not found"):
        open_trade_archive(cache, receipt=tmp_path / "absent.json")


def manifest(path: Path, **edits: Any) -> None:
    import hashlib

    row = {
        "filename": path.name,
        "size": path.stat().st_size,
        "hash": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    row.update(edits)
    (path.parent / "manifest.json").write_text(json.dumps({"files": [row]}))


def test_a_cache_says_which_of_its_inputs_no_vendor_manifest_vouched_for(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    manifest(mbo_input(job / "day-1.mbo.dbn", 100, 120, [(101 * NS, 42, 4, dbn.Side.BID)]))
    cache = prepare_trade_archive(job, tmp_path / "lake" / "_meta" / "trade_cache")
    with read_trades(cache, SID, sessions=DAYS) as read:
        list(read)
        vouched = read.source
    assert vouched.lineage is not None and vouched.lineage.inputs[0].manifest_verified
    assert not any("no vendor manifest" in item for item in vouched.limitations)
    # The same download without its manifest: a different cache, and it says so.
    (job / "manifest.json").unlink()
    bare = prepare_trade_archive(job, tmp_path / "lake" / "_meta" / "trade_cache")
    with read_trades(bare, SID, sessions=DAYS) as read:
        list(read)
        assert any(
            "1 of 1 input file(s) had no vendor manifest" in item
            for item in read.source.limitations
        )
    assert bare != cache


def test_vendor_manifests_verify_size_and_hash_but_absence_proves_nothing(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    path = write(job / "glbx-mdp3.trades.dbn.zst", [trade(101 * NS)])
    with read_trades(path, SID) as read:
        list(read)
        assert read.source.input_schemas is None and read.source.request_stype_in is None
        assert "vendor manifest: size and sha256 match" not in read.source.checks
    manifest(path)
    with read_trades(path, SID) as read:
        list(read)
        verified = read.source
    assert verified.input_schemas == ("trades",) and verified.request_stype_in == "raw_symbol"
    assert "vendor manifest: size and sha256 match" in verified.checks
    # A managed name means preparation wrote the header: no lineage without a receipt.
    managed = path.rename(job / (policy_key({"any": "policy"}) + ".trades.dbn.zst"))
    manifest(managed)
    with read_trades(managed, SID) as read:
        list(read)
        assert read.source.input_schemas is None and read.source.request_stype_in is None
    path = managed.rename(path)
    manifest(path, hash="sha256:" + "0" * 64)
    with pytest.raises(IngestError, match="differs from vendor manifest"):
        open_trade_archive(path)
    manifest(path, filename="another.dbn.zst")
    with pytest.raises(IngestError, match="missing or duplicate entry"):
        open_trade_archive(path)
    (job / "manifest.json").write_text("{not json")
    with pytest.raises(IngestError, match="unreadable vendor manifest"):
        open_trade_archive(path)


# ---------- totals ----------


def random_records(seed: int) -> list[Any]:
    rng = random.Random(seed)
    records: list[Any] = []
    for i in range(60):
        record = trade(
            100 * NS + i * 10_000_000,
            instrument_id=rng.choice([42, 42, 42, 43]),
            size=rng.randrange(0, 30),
            price=rng.choice([10 * NS, dbn.UNDEF_PRICE]) if rng.random() < 0.1 else 11 * NS,
            flags=dbn.F_BAD_TS_RECV if rng.random() < 0.05 else 0,
            side=rng.choice([dbn.Side.ASK, dbn.Side.BID, dbn.Side.NONE]),
        )
        records.append(encoded(record, side=b"Q") if rng.random() < 0.05 else record)
    return records


@pytest.mark.parametrize("seed", range(5))
def test_side_totals_conserve_and_do_not_depend_on_batching(tmp_path: Path, seed: int) -> None:
    path = write(tmp_path / "t.dbn", random_records(seed))
    selected = read_all(path)
    eligible = [t for t in selected if t.eligible]
    expected = TradeTotals(
        records=len(eligible),
        volume=sum(t.size for t in eligible),
        buy_records=sum(t.aggressor == "buy" for t in eligible),
        buy_volume=sum(t.size for t in eligible if t.aggressor == "buy"),
        sell_records=sum(t.aggressor == "sell" for t in eligible),
        sell_volume=sum(t.size for t in eligible if t.aggressor == "sell"),
        unknown_records=sum(t.aggressor == "unknown" for t in eligible),
        unknown_volume=sum(t.size for t in eligible if t.aggressor == "unknown"),
        ineligible_records=len(selected) - len(eligible),
    )
    for batch_records in (1, 2, 7, 1000):
        totals = TradeTotals()
        with read_trades(path, SID, batch_records=batch_records) as read:
            for batch in read:
                totals += TradeTotals.of(batch)
        assert totals == expected
    summary = summarize_trades(path, SID)
    assert summary.totals == expected and summary.source.trade_records == 60
    assert TradeTotals.of([]) == TradeTotals()


def test_totals_refuse_counts_that_do_not_conserve() -> None:
    with pytest.raises(ValueError, match="add up"):
        TradeTotals(records=2, volume=3, buy_records=1, buy_volume=3)
    with pytest.raises(ValueError, match="non-negative integers"):
        TradeTotals(records=1, volume=1.0, buy_records=1, buy_volume=1)  # pyright: ignore[reportArgumentType]


def test_summaries_come_only_from_validated_reads(tmp_path: Path) -> None:
    path = write(tmp_path / "t.dbn", [trade(101 * NS, size=2), trade(102 * NS, size=3)])
    assert summarize_trades(path, SID, start_ns=103 * NS, end_ns=104 * NS).totals == TradeTotals()
    path.write_bytes(path.read_bytes()[:-3])
    with pytest.raises(IngestError, match="incomplete trailing"):
        summarize_trades(path, SID)


# ---------- the pandas view and join provenance ----------


def test_frame_keeps_integers_exact_and_undefined_values_null(tmp_path: Path) -> None:
    start = BIG * NS
    path = write(
        tmp_path / "t.dbn",
        [
            trade(start + 1, ts_event=dbn.UNDEF_TIMESTAMP, price=dbn.UNDEF_PRICE, size=3),
            trade(start + 2, price=29_140_250_000_001, side=dbn.Side.ASK, sequence=2**32 - 1),
        ],
        start=start,
        end=start + NS,
    )
    frame = load_trade_frame(path, SID, start_ns=start, end_ns=start + NS)
    dtypes = cast(dict[str, Any], frame.dtypes.to_dict())
    assert {name: str(dtype) for name, dtype in dtypes.items()} == TRADE_COLUMNS
    assert frame["ts_recv"].tolist() == [start + 1, start + 2]
    assert pd.isna(frame["ts_event"].iloc[0]) and pd.isna(frame["price_fixed"].iloc[0])
    assert int(frame["price_fixed"].iloc[1]) == 29_140_250_000_001
    assert frame["price"].iloc[1] == 29_140_250_000_001 / 1_000_000_000
    assert frame["aggressor"].tolist() == ["buy", "sell"]
    assert list(frame["aggressor"].cat.categories) == ["buy", "sell", "unknown"]
    assert frame["eligible"].tolist() == [False, True]
    assert frame["issues"].tolist() == [
        int(TradeIssue.UNDEFINED_PRICE | TradeIssue.UNDEFINED_TS_EVENT),
        0,
    ]
    assert frame.attrs["price_scale"] == 1_000_000_000
    assert "provenance" not in frame.attrs  # no calendar, so no join provenance
    empty = load_trade_frame(path, SID, start_ns=start + 5, end_ns=start + 6)
    assert empty.empty and empty.dtypes.equals(frame.dtypes)


def test_frame_loader_needs_a_bound_and_a_calendar_for_provenance(tmp_path: Path) -> None:
    path = write(tmp_path / "t.dbn", [trade(101 * NS)])
    with pytest.raises(ValueError, match="explicit bound"):
        load_trade_frame(path, SID)
    with pytest.raises(ValueError, match="explicit bound"):
        load_trade_frame(path, SID, start_ns=100 * NS)
    with pytest.raises(ProvenanceError, match="also needs sessions"):
        load_trade_frame(path, SID, start_ns=100 * NS, end_ns=101 * NS, feed="databento")
    with pytest.raises(ProvenanceError, match="one explicit value"):
        load_trade_frame(path, SID, sessions=DAYS, feed="unknown")
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(IngestError, match="incomplete trailing"):
        load_trade_frame(path, SID, sessions=DAYS)


def test_trade_rows_join_volume_bars_under_the_existing_contract(tmp_path: Path) -> None:
    from bedivere.data.lake.volume import VolumeSpec
    from bedivere.data.lake.volume_build import build_volume_archive
    from bedivere.data.lake.volume_research import align_available, load_volume_frame

    records = [trade((100 + i) * NS + 5, size=400) for i in range(9)]
    path = write(tmp_path / "t.dbn", records, end=110 * NS)
    days = replace(DAYS, days=(DAY,))
    spec = VolumeSpec(1000)
    build_volume_archive(path, tmp_path, SID, spec, days)
    bars = load_volume_frame(tmp_path, SID, spec, days, feed="databento")
    frame = load_trade_frame(path, SID, sessions=days, feed="databento")
    declared = frame.attrs["provenance"]
    assert declared.verified == {
        "dataset", "symbol", "series", "price_basis", "time_unit", "epoch", "clock",
    }
    assert (declared.feed, declared.instrument_namespace) == ("databento", "databento:GLBX.MDP3")
    observations = cast(
        pd.DataFrame,
        frame.rename(columns={"ts_recv": "asof_ns"})[["asof_ns", "session", "instrument_id", "size"]],
    )
    joined = align_available(observations, bars, observation_provenance=declared, max_age_ns=10 * NS)
    values = cast(list[Any], joined["volume_bar_id"].tolist())
    assert [None if pd.isna(v) else int(v) for v in values] == [None, None, None, 0, 0, 0, 1, 1, 1]
    assert "trade_source" not in joined.attrs["bar_provenance"].__slots__


def test_join_provenance_carries_no_artifact_or_receipt_facts() -> None:
    assert volume_research.Provenance is provenance.Provenance
    assert volume_research.ProvenanceError is provenance.ProvenanceError
    assert provenance.FIELDS == {
        "feed", "dataset", "symbol", "series", "instrument_namespace", "price_basis",
        "sessions", "time_unit", "epoch", "clock",
    }


def test_streaming_reader_imports_without_pandas() -> None:
    code = (
        "import sys; import bedivere.data.lake.trades; "
        "sys.exit(1 if 'pandas' in sys.modules else 0)"
    )
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    assert subprocess.run([sys.executable, "-c", code], env=env, check=False).returncode == 0
