"""Retained Databento trade records, streamed from a verified archive.

Strategy-independent access for offline research; nothing is reconstructed or
inferred. Verification is two passes over one open file handle:

  A. `open_trade_archive` hashes every byte before any decoding, checks the
     digest against a receipt or vendor manifest, and requires a `.zst` to end
     on a complete frame.
  B. `TradeArchive.read` decodes from the same handle through EOF, whatever
     the window, then requires the same digest, a complete last record, counts
     matching any receipt, and an unchanged file.

Batches are provisional until the read is exhausted and `status` is
`validated`. Ordinals count decoded trade records from zero before any
selection, and name a record in those exact bytes only.
"""

from __future__ import annotations

import hashlib
import os
import stat
from bisect import bisect_right
from collections.abc import Generator, Iterable, Mapping
from dataclasses import dataclass, fields
from enum import IntFlag, StrEnum
from pathlib import Path
from types import MappingProxyType, TracebackType
from typing import BinaryIO, Final, NamedTuple, cast

import databento_dbn

from bedivere.core.session_days import SessionDays
from bedivere.core.types import Timeframe
from bedivere.data.lake.ingest import (
    PRICE_SCALE,
    UNDEF_PRICE,
    CheckedDecoder,
    IngestError,
    SymbolResolver,
    ZstdFrames,
    build_symbology,
    contract_pattern,
    is_zstd,
)
from bedivere.data.lake.layout import SeriesId, session_path
from bedivere.data.lake.provenance import Provenance, ProvenanceError, require_explicit
from bedivere.data.lake.read import read_partition
from bedivere.data.lake.schema import AS_TRADED
from bedivere.data.lake.trade_archive import (
    MANAGED_NAME,
    TradeReceipt,
    read_receipt,
    receipt_path,
    verify_manifest,
)

NS: Final[int] = 1_000_000_000

CHUNK_BYTES: Final[int] = 1 << 20
"""Bytes handed to the decoder at a time. Results never depend on it."""

DEFAULT_BATCH_RECORDS: Final[int] = 65_536
"""Records per batch when a read does not say; read when the read is made."""


class Aggressor(StrEnum):
    """Which side initiated a trade, read from the vendor's side code alone."""

    BUY = "buy"
    SELL = "sell"
    UNKNOWN = "unknown"


# Databento trades schema: A = seller aggressor, B = buyer, N = none. An
# unknown side is never inferred from prices.
SIDE_CODES: Final[Mapping[str, Aggressor]] = MappingProxyType(
    {"A": Aggressor.SELL, "B": Aggressor.BUY, "N": Aggressor.UNKNOWN}
)


class TradeIssue(IntFlag):
    """Quality notes on a retained record. The raw fields are never altered."""

    UNDEFINED_PRICE = 1 << 0
    """`UNDEF_PRICE`: `price_fixed` is None."""

    ZERO_SIZE = 1 << 1

    BAD_TS_RECV = 1 << 2
    """`F_BAD_TS_RECV`: the vendor marks this `ts_recv` inaccurate."""

    NON_TRADE_ACTION = 1 << 3
    """An action code other than T, which the trades schema never uses."""

    UNDEFINED_SIDE = 1 << 4
    """A side code outside A/B/N, kept in `side_code`."""

    UNDEFINED_TS_EVENT = 1 << 5
    """`UNDEF_TIMESTAMP`, or a value beyond int64: `ts_event` is None."""


INELIGIBLE: Final[TradeIssue] = (
    TradeIssue.UNDEFINED_PRICE
    | TradeIssue.ZERO_SIZE
    | TradeIssue.BAD_TS_RECV
    | TradeIssue.NON_TRADE_ACTION
)
"""What keeps a record out of receive-time volume bars. Research reads retain
such records, flagged; the builder refuses a build that selects one."""

_ISSUES: Final[tuple[TradeIssue, ...]] = tuple(TradeIssue(bits) for bits in range(64))


class Trade(NamedTuple):
    """One decoded trade record as retained in its archive.

    `ordinal` counts decoded trade records from zero in archive order, before
    any selection. `sequence` is the venue's message sequence number: kept,
    never used to sort or deduplicate, and not comparable across channels.
    """

    ordinal: int
    ts_recv: int
    """Databento capture time, int Unix ns: the ordering and selection clock."""

    ts_event: int | None
    """Matching-engine time, int Unix ns; None if undefined."""

    session: str | None
    """The session label from the caller's calendar, if one was supplied."""

    instrument_id: int
    publisher_id: int
    contract: str
    """Raw symbol, resolved from the archive's own symbology at `ts_recv`."""

    price_fixed: int | None
    """Exact price in units of 1 / `PRICE_SCALE`; None for `UNDEF_PRICE`."""

    size: int
    sequence: int
    side_code: str
    """The vendor's one-character side code, exactly as stored."""

    aggressor: Aggressor
    flags: int
    """The vendor's flag byte, every bit kept."""

    issues: TradeIssue

    @property
    def price(self) -> float | None:
        return None if self.price_fixed is None else self.price_fixed / PRICE_SCALE

    @property
    def eligible(self) -> bool:
        return not self.issues & INELIGIBLE


class ReadStatus(StrEnum):
    PENDING = "pending"
    READING = "reading"
    """Batches produced so far are provisional."""

    VALIDATED = "validated"
    """Exhausted, with every end-of-read check passed."""

    FAILED = "failed"
    """An error ended the read; nothing it produced is valid."""

    INCOMPLETE = "incomplete"
    """Closed before exhaustion: validation never finished."""


@dataclass(frozen=True, slots=True)
class TradeSource:
    """Where a validated read's records came from, and how far that is known.

    Four kinds of statement, kept apart: CHECKED against the opened artifact
    (archive fields, counts, `checks`, `selection_dependencies`), RECORDED
    lineage from a receipt (`lineage`), ATTESTED by the caller (`attested`),
    and UNAVAILABLE (`unavailable`). Never part of what research joins compare.
    """

    archive_name: str
    archive_sha256: str
    archive_bytes: int
    stored_schema: str
    dataset: str
    declared_start_ns: int
    declared_end_ns: int
    trade_records: int
    """Every decoded trade record in the archive, before selection."""

    trade_volume: int
    checks: tuple[str, ...]
    selection_dependencies: tuple[tuple[str, str], ...]
    """(session label, `bars.parquet:sha256:...`) of each 1s reference partition."""

    lineage: TradeReceipt | None
    input_schemas: tuple[str, ...] | None
    """From a receipt, or the header of a file a vendor manifest verified; else None."""

    request_stype_in: str | None
    """The original request's input symbology: only from the vendor-written
    header of a manifest-verified download, never a prepared cache's."""

    attested: tuple[str, ...]
    unavailable: tuple[str, ...]
    limitations: tuple[str, ...]

    def __deepcopy__(self, memo: dict[int, object]) -> TradeSource:
        # Deeply immutable; pandas deep-copies attrs on nearly every operation.
        return self


@dataclass(frozen=True, slots=True)
class TradeTotals:
    """Trade-record counts and quantity by aggressor side, eligible records only.

    Records are as the vendor printed them, not orders or participants.
    `ineligible_records` counts the selected records left out.
    """

    records: int = 0
    volume: int = 0
    buy_records: int = 0
    buy_volume: int = 0
    sell_records: int = 0
    sell_volume: int = 0
    unknown_records: int = 0
    unknown_volume: int = 0
    ineligible_records: int = 0

    def __post_init__(self) -> None:
        if any(type(v) is not int or v < 0 for v in (getattr(self, f.name) for f in fields(self))):
            raise ValueError("trade totals are non-negative integers")
        sides = (
            self.buy_records + self.sell_records + self.unknown_records,
            self.buy_volume + self.sell_volume + self.unknown_volume,
        )
        if sides != (self.records, self.volume):
            raise ValueError("side counts and volumes must add up to the totals")

    @classmethod
    def of(cls, trades: Iterable[Trade]) -> TradeTotals:
        records = dict.fromkeys(Aggressor, 0)
        volume = dict.fromkeys(Aggressor, 0)
        ineligible = 0
        for trade in trades:
            if trade.issues & INELIGIBLE:
                ineligible += 1
                continue
            records[trade.aggressor] += 1
            volume[trade.aggressor] += trade.size
        return cls(
            records=sum(records.values()),
            volume=sum(volume.values()),
            buy_records=records[Aggressor.BUY],
            buy_volume=volume[Aggressor.BUY],
            sell_records=records[Aggressor.SELL],
            sell_volume=volume[Aggressor.SELL],
            unknown_records=records[Aggressor.UNKNOWN],
            unknown_volume=volume[Aggressor.UNKNOWN],
            ineligible_records=ineligible,
        )

    def __add__(self, other: TradeTotals) -> TradeTotals:
        return TradeTotals(*(getattr(self, f.name) + getattr(other, f.name) for f in fields(self)))


@dataclass(frozen=True, slots=True)
class TradeSummary:
    totals: TradeTotals
    source: TradeSource


def fingerprint(path: Path) -> str:
    """`<name>:sha256:<hex>`: the volume builder's source-fingerprint format."""
    with path.open("rb") as handle:
        return f"{path.name}:sha256:{hashlib.file_digest(handle, 'sha256').hexdigest()}"


def _identity(status: os.stat_result) -> tuple[int, int, int, int]:
    return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)


@dataclass(slots=True)
class _Opened:
    """State shared by an archive and its reads."""

    path: Path
    handle: BinaryIO
    status: os.stat_result
    sha256: str
    zstd: bool
    receipt: TradeReceipt | None
    manifest_verified: bool
    active: TradeRead | None = None
    closed: bool = False


@dataclass(frozen=True, slots=True)
class _Selection:
    series: SeriesId
    raw_symbol: str | None
    sessions: SessionDays | None
    start_ns: int | None
    end_ns: int | None
    lake_root: Path | None


class TradeArchive:
    """An archive whose bytes were hashed before any decoding (pass A).

    Holds its file open until closed. One read may run at a time.
    """

    def __init__(self, opened: _Opened) -> None:
        self._opened = opened

    @property
    def path(self) -> Path:
        return self._opened.path

    @property
    def name(self) -> str:
        return self._opened.path.name

    @property
    def sha256(self) -> str:
        return self._opened.sha256

    @property
    def size(self) -> int:
        return self._opened.status.st_size

    @property
    def fingerprint(self) -> str:
        return f"{self.name}:sha256:{self.sha256}"

    @property
    def receipt(self) -> TradeReceipt | None:
        """The preparation receipt, checked against this digest; None if absent."""
        return self._opened.receipt

    @property
    def manifest_verified(self) -> bool:
        """A vendor manifest beside the file matched its size and SHA256."""
        return self._opened.manifest_verified

    def read(
        self,
        series: SeriesId,
        *,
        sessions: SessionDays | None = None,
        start_ns: int | None = None,
        end_ns: int | None = None,
        lake_root: Path | None = None,
        batch_records: int | None = None,
        close_archive: bool = False,
    ) -> TradeRead:
        """Select one contract or continuous series; decode lazily (pass B).

        `raw.<CONTRACT>` selects by the archive's own symbology at `ts_recv`;
        any other series takes each session's contract from its 1s reference
        partition under `lake_root`. `sessions` selects `[open, close)` of each
        session and `start_ns`/`end_ns` bound `ts_recv` as `[start_ns, end_ns)`;
        both combine, and the archive must cover what they request.
        """
        if not isinstance(cast(object, series), SeriesId):
            raise ValueError("series must be a SeriesId")
        pattern = contract_pattern(series.symbol)
        raw_symbol = series.series.removeprefix("raw.") if series.series.startswith("raw.") else None
        if raw_symbol is not None and not pattern.fullmatch(raw_symbol):
            raise ValueError("raw series must name an outright futures contract of the selected root")
        if sessions is not None:
            if not isinstance(cast(object, sessions), SessionDays):
                raise ValueError("sessions must be resolved SessionDays")
            if not sessions.days:
                raise ValueError("at least one session-day is required")
        if raw_symbol is None and (sessions is None or lake_root is None):
            raise ValueError(
                f"continuous series {series.series!r} needs sessions and the lake_root "
                "holding its 1s reference partitions"
            )
        for bound in (start_ns, end_ns):
            if bound is not None and (type(bound) is not int or not 0 <= bound < 2**63):
                raise ValueError("receive-time bounds must be non-negative int64 nanoseconds")
        if start_ns is not None and end_ns is not None and end_ns <= start_ns:
            raise ValueError("end_ns must be after start_ns")
        if batch_records is None:
            batch_records = DEFAULT_BATCH_RECORDS
        if type(batch_records) is not int or batch_records <= 0:
            raise ValueError("batch_records must be a positive integer")
        selection = _Selection(series, raw_symbol, sessions, start_ns, end_ns, lake_root)
        return TradeRead(self._opened, selection, batch_records, self if close_archive else None)

    def close(self) -> None:
        opened = self._opened
        if opened.active is not None:
            opened.active.close()
        opened.closed = True
        opened.handle.close()

    def __enter__(self) -> TradeArchive:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def open_trade_archive(path: Path, *, receipt: Path | None = None) -> TradeArchive:
    """Pass A: hash the archive, then check its receipt and vendor manifest.

    `receipt` defaults to the one `prepare_trade_archive` writes beside the
    file. A present receipt or manifest must agree, or this raises; an absent
    one verifies nothing. A truncated `.zst` is refused here, before any read.
    """
    zstd = is_zstd(path)
    if receipt is not None and not receipt.is_file():
        raise FileNotFoundError(f"receipt not found: {receipt}")
    handle = path.open("rb")
    try:
        opened = os.fstat(handle.fileno())
        if not stat.S_ISREG(opened.st_mode):
            raise IngestError(f"{path.name}: not a regular file")
        digest = hashlib.sha256()
        frames = ZstdFrames(path.name) if zstd else None
        size = 0
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
            if frames is not None:
                frames.feed(chunk)
            size += len(chunk)
        if size != opened.st_size or _identity(os.fstat(handle.fileno())) != _identity(opened):
            raise IngestError(f"{path.name}: changed while being hashed; archives must not change")
        if frames is not None:
            # Here as well as at the end of a read: a resumed build decodes nothing.
            frames.require_complete()
        sha256 = digest.hexdigest()
        receipt_file = receipt if receipt is not None else receipt_path(path)
        parsed = read_receipt(receipt_file) if receipt_file.is_file() else None
        if parsed is not None:
            parsed.verify_artifact(path.name, sha256)
        manifest = verify_manifest(path, sha256, size)
        return TradeArchive(_Opened(path, handle, opened, sha256, zstd, parsed, manifest))
    except BaseException:
        handle.close()
        raise


class TradeRead:
    """One decoding pass (B): an iterator of provisional batches.

    Each batch is a tuple of `Trade`s in archive order. Nothing is validated
    until the iterator is exhausted and `status` is `validated`; closing early
    leaves the read `incomplete`.
    """

    def __init__(
        self,
        opened: _Opened,
        selection: _Selection,
        batch_records: int,
        owned: TradeArchive | None = None,
    ) -> None:
        self._opened = opened
        self._selection = selection
        self._batch_records = batch_records
        self._owned = owned
        self._status = ReadStatus.PENDING
        self._error: BaseException | None = None
        self._source: TradeSource | None = None
        self._selected = 0
        self._batches = self._run()

    @property
    def status(self) -> ReadStatus:
        return self._status

    @property
    def validated(self) -> bool:
        return self._status is ReadStatus.VALIDATED

    @property
    def error(self) -> BaseException | None:
        return self._error

    @property
    def selected_records(self) -> int:
        """Records yielded so far; final only once validated."""
        return self._selected

    @property
    def sessions(self) -> SessionDays | None:
        return self._selection.sessions

    @property
    def source(self) -> TradeSource:
        if self._source is None or not self.validated:
            raise ValueError(f"trade read is {self._status.value}, not validated")
        return self._source

    def provenance(self, feed: str) -> Provenance:
        """Research join provenance, for a validated read with a session calendar.

        `feed` and the calendar are the caller's attestation; the rest was
        applied to the archive by this read.
        """
        require_explicit("feed", feed)
        if not self.validated:
            raise ValueError(f"trade read is {self._status.value}, not validated")
        sessions = self._selection.sessions
        if sessions is None:
            raise ProvenanceError(
                "join provenance needs a session calendar; this read was made without one"
            )
        series = self._selection.series
        return Provenance(
            feed=feed,
            dataset=series.dataset,
            symbol=series.symbol,
            series=series.series,
            instrument_namespace=f"{feed}:{series.dataset}",
            price_basis=AS_TRADED,
            sessions=sessions,
            time_unit="ns",
            epoch="unix",
            clock="ts_recv",
            verified=frozenset(
                {"dataset", "symbol", "series", "price_basis", "time_unit", "epoch", "clock"}
            ),
        )

    def __iter__(self) -> TradeRead:
        return self

    def __next__(self) -> tuple[Trade, ...]:
        return next(self._batches)

    def close(self) -> None:
        self._batches.close()
        if self._status in (ReadStatus.PENDING, ReadStatus.READING):
            self._status = ReadStatus.INCOMPLETE
        if self._owned is not None:
            self._owned.close()

    def __enter__(self) -> TradeRead:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _run(self) -> Generator[tuple[Trade, ...], None, None]:
        opened = self._opened
        self._status = ReadStatus.READING
        try:
            if opened.closed:
                raise ValueError("trade archive is closed")
            if opened.active is not None:
                raise ValueError("another read of this trade archive is in progress")
            opened.active = self
            self._source = yield from self._scan()
            self._status = ReadStatus.VALIDATED
        except GeneratorExit:
            self._status = ReadStatus.INCOMPLETE
            raise
        except BaseException as exc:
            self._status = ReadStatus.FAILED
            self._error = exc
            raise
        finally:
            if opened.active is self:
                opened.active = None
            if self._owned is not None:
                self._owned.close()

    def _scan(self) -> Generator[tuple[Trade, ...], None, TradeSource]:
        opened, selection = self._opened, self._selection
        name = opened.path.name
        receipt = opened.receipt
        series = selection.series
        pattern = contract_pattern(series.symbol)
        raw_symbol = selection.raw_symbol
        days = selection.sessions.days if selection.sessions is not None else ()
        starts = [day.start_unix * NS for day in days]
        ends = [day.end_unix * NS for day in days]
        labels = [day.label for day in days]
        window_start, window_end = selection.start_ns, selection.end_ns
        side_codes = dict(SIDE_CODES)
        bad_ts_recv = databento_dbn.F_BAD_TS_RECV
        opened.handle.seek(0)
        digest = hashlib.sha256()
        decoder = CheckedDecoder(name, zstd=opened.zstd)
        meta: databento_dbn.Metadata | None = None
        symbology: SymbolResolver | None = None
        selected_ids: list[int] = []
        dependencies: list[tuple[str, str]] = []
        input_starts = [i.start_ns for i in receipt.inputs] if receipt else []
        input_ends = [i.end_ns for i in receipt.inputs] if receipt else []
        per_input = [[0, 0] for _ in input_starts]
        last_input = 0
        declared_start = declared_end = 0
        records = volume = 0
        previous = 0
        publishers: dict[str, int] = {}
        batch: list[Trade] = []
        hashed = 0
        while chunk := opened.handle.read(CHUNK_BYTES):
            digest.update(chunk)
            hashed += len(chunk)
            for rec in decoder.feed(chunk):
                if meta is None:
                    if not isinstance(rec, databento_dbn.Metadata):
                        raise IngestError(
                            f"{name}: first DBN record is {type(rec).__name__}, not Metadata"
                        )
                    meta = rec
                    declared_start, declared_end = self._check_header(meta)
                    symbology = build_symbology(meta)
                    if raw_symbol is None:
                        selected_ids, dependencies = self._continuous_contracts()
                    continue
                if not isinstance(rec, databento_dbn.TradeMsg):
                    if isinstance(rec, databento_dbn.ErrorMsg):
                        raise IngestError(f"error record in {name}: {rec}")
                    raise IngestError(
                        f"{name}: {type(rec).__name__} record in a trades archive; "
                        "only trade records are accepted"
                    )
                ordinal = records
                records += 1
                size = rec.size
                volume += size
                ts = rec.ts_recv
                if not 0 < ts < 2**63:
                    raise ValueError("invalid trade receive timestamp")
                if not declared_start <= ts < declared_end:
                    raise IngestError(f"{name}: a record lies outside the archive's declared range")
                if receipt is not None:
                    # Extraction wrote inputs in range order: the bucket never moves backwards.
                    if not input_starts[last_input] <= ts < input_ends[last_input]:
                        index = bisect_right(input_starts, ts) - 1
                        if index < last_input or ts >= input_ends[index]:
                            raise IngestError(
                                f"{name}: records do not follow the receipt's recorded input ranges"
                            )
                        last_input = index
                    per_input[last_input][0] += 1
                    per_input[last_input][1] += size
                if window_start is not None and ts < window_start:
                    continue
                if window_end is not None and ts >= window_end:
                    continue
                label = None
                index = -1
                if days:
                    index = bisect_right(starts, ts) - 1
                    if index < 0 or ts >= ends[index]:
                        continue
                    label = labels[index]
                assert symbology is not None
                if raw_symbol is not None:
                    symbol = symbology.resolve(rec.instrument_id, ts // NS)
                    if symbol != raw_symbol:
                        continue
                else:
                    if rec.instrument_id != selected_ids[index]:
                        continue
                    symbol = symbology.resolve(rec.instrument_id, ts // NS)
                if not pattern.fullmatch(symbol):
                    raise ValueError("selected instrument has missing or inconsistent outright symbology")
                flags = rec.flags
                issues = 0
                if flags & bad_ts_recv:
                    # An inaccurate ts_recv says nothing about the order around it.
                    issues |= TradeIssue.BAD_TS_RECV
                elif ts < previous:
                    raise ValueError("selected trades are out of receive-time order")
                else:
                    previous = ts
                scope = label if label is not None else ""
                if publishers.setdefault(scope, rec.publisher_id) != rec.publisher_id:
                    raise ValueError(
                        "multiple publishers in one session; select a single feed first"
                        if label is not None
                        else "multiple publishers in one selection; select a single feed first"
                    )
                price: int | None = rec.price
                if price == UNDEF_PRICE:
                    price = None
                    issues |= TradeIssue.UNDEFINED_PRICE
                if not size:
                    issues |= TradeIssue.ZERO_SIZE
                if str(rec.action) != "T":
                    issues |= TradeIssue.NON_TRADE_ACTION
                side_code = str(rec.side)
                aggressor = side_codes.get(side_code)
                if aggressor is None:
                    aggressor = Aggressor.UNKNOWN
                    issues |= TradeIssue.UNDEFINED_SIDE
                ts_event: int | None = rec.ts_event
                if ts_event >= 2**63:  # UNDEF_TIMESTAMP (2**64 - 1), or no int64 at all
                    ts_event = None
                    issues |= TradeIssue.UNDEFINED_TS_EVENT
                batch.append(
                    Trade(
                        ordinal,
                        ts,
                        ts_event,
                        label,
                        rec.instrument_id,
                        rec.publisher_id,
                        symbol,
                        price,
                        size,
                        rec.sequence,
                        side_code,
                        aggressor,
                        flags,
                        _ISSUES[issues],
                    )
                )
                if len(batch) == self._batch_records:
                    self._selected += len(batch)
                    yield tuple(batch)
                    batch = []
        checks = self._finish(decoder, meta, digest.hexdigest(), hashed)
        assert meta is not None
        if receipt is not None:
            receipt.verify_counts(records, volume, [(n, v) for n, v in per_input])
            checks.append("receipt: full-archive and per-input trade counts and volume match")
        source = self._describe(meta, records, volume, checks, dependencies)
        if batch:
            self._selected += len(batch)
            yield tuple(batch)
        return source

    def _check_header(self, meta: databento_dbn.Metadata) -> tuple[int, int]:
        selection, receipt = self._selection, self._opened.receipt
        if meta.dataset != selection.series.dataset or str(meta.schema) != "trades":
            raise ValueError("archive must have schema=trades and match the selected dataset")
        low = [bound for bound in (selection.start_ns,) if bound is not None]
        high = [bound for bound in (selection.end_ns,) if bound is not None]
        if selection.sessions is not None:
            low.append(selection.sessions.days[0].start_unix * NS)
            high.append(selection.sessions.days[-1].end_unix * NS)
        if low and high and max(low) >= min(high):
            low = high = []  # nothing requested, so no range to cover
        if (
            meta.limit
            or not meta.end
            or (low and meta.start > max(low))
            or (high and meta.end < min(high))
        ):
            raise ValueError(
                "archive must cover every requested full session, without a record limit"
                if selection.sessions is not None
                and selection.start_ns is None
                and selection.end_ns is None
                else "archive must cover the requested receive-time range, without a record limit"
            )
        if receipt is not None:
            receipt.verify_header(meta.dataset, meta.start, meta.end)
        return meta.start, meta.end

    def _continuous_contracts(self) -> tuple[list[int], list[tuple[str, str]]]:
        """Each session's contract from the 1s reference lake, fingerprinted as read."""
        selection = self._selection
        assert selection.sessions is not None and selection.lake_root is not None
        ids: list[int] = []
        dependencies: list[tuple[str, str]] = []
        for day in selection.sessions.days:
            path = session_path(selection.lake_root, selection.series, Timeframe.S1, day.label)
            if not path.is_file():
                raise FileNotFoundError(
                    f"{day.label}: missing reference 1s partition for {selection.series}: {path}"
                )
            before = _identity(path.stat())
            stamp = fingerprint(path)
            table = read_partition(path)
            if _identity(path.stat()) != before:
                raise ValueError(f"{day.label}: reference 1s partition changed while being read")
            found = set(table.instrument_id)
            if len(found) != 1 or 0 in found:
                raise ValueError(f"{day.label}: reference 1s lake must identify exactly one contract")
            ids.append(found.pop())
            dependencies.append((day.label, stamp))
        return ids, dependencies

    def _finish(
        self,
        decoder: CheckedDecoder,
        meta: databento_dbn.Metadata | None,
        sha256: str,
        hashed: int,
    ) -> list[str]:
        opened = self._opened
        name = opened.path.name
        decoder.finish()
        if meta is None:
            raise IngestError(f"{name}: no DBN metadata header")
        if hashed != opened.status.st_size or sha256 != opened.sha256:
            raise IngestError(
                f"{name}: bytes decoded differ from the bytes hashed before decoding; "
                "archives must not change while read"
            )
        now = (_identity(os.fstat(opened.handle.fileno())), _identity(opened.path.stat()))
        if now != (_identity(opened.status),) * 2:
            raise IngestError(f"{name}: file was modified or replaced during the read")
        checks = [
            "sha256 of every byte before decoding",
            "sha256 of the decoded bytes matches",
            "file identity (device, inode, size, mtime) unchanged",
            "no incomplete trailing DBN record",
            "header schema trades; every record a trade record",
            "every record inside the declared range",
        ]
        if decoder.zstd:
            checks.insert(4, "compressed input ends on a complete zstd frame")
        if opened.receipt is not None:
            checks.append("receipt: checksum, extraction policy and managed name")
            checks.append("receipt: dataset and input ranges match the header")
        if opened.manifest_verified:
            checks.append("vendor manifest: size and sha256 match")
        if self._selection.raw_symbol is None:
            checks.append("selection: 1s reference partitions unchanged while read")
        return checks

    def _describe(
        self,
        meta: databento_dbn.Metadata,
        records: int,
        volume: int,
        checks: list[str],
        dependencies: list[tuple[str, str]],
    ) -> TradeSource:
        opened, selection = self._opened, self._selection
        receipt = opened.receipt
        unavailable: list[str] = []
        limitations = [
            "ordinal and (archive_sha256, ordinal) identify records in these exact bytes only; "
            "recompression, encoder upgrades or cache regeneration can change them",
            "sequence is venue-assigned and scoped to its channel: not a global order or trade ID",
            "ts_recv is Databento's capture time, a feed-level proxy, not application "
            "processing time",
        ]
        input_schemas: tuple[str, ...] | None = None
        request_stype_in: str | None = None
        if receipt is not None:
            input_schemas = receipt.input_schemas
            checks.append("lineage: input schemas as recorded by the receipt")
            unavailable.append("original request symbology (not retained by the receipt)")
            limitations.append(
                "receipt consistency does not authenticate origin or prove market-data "
                "completeness; manifest_verified describes the preparation step"
            )
            unmanifested = sum(not i.manifest_verified for i in receipt.inputs)
            if unmanifested:
                limitations.append(
                    f"{unmanifested} of {len(receipt.inputs)} input file(s) had no vendor "
                    "manifest when prepared: their bytes were never checked against the "
                    "vendor's published size and hash"
                )
            if "mbo" in input_schemas:
                limitations.append(
                    "from MBO: depth was set to 0 and is not exposed; order_id and channel_id "
                    "were dropped, so the sequence's full scope cannot be recovered"
                )
        elif opened.manifest_verified and not MANAGED_NAME.fullmatch(opened.path.name):
            # The vendor wrote this header; a prepared cache's header it never wrote.
            input_schemas = (str(meta.schema),)
            if meta.stype_in is not None:
                request_stype_in = str(meta.stype_in)
            else:
                unavailable.append("original request symbology (not in the header)")
            checks.append("lineage: vendor-written header of a manifest-verified file")
        else:
            unavailable += [
                "original lineage (no receipt, and no vendor manifest for a vendor file)",
                "original input schemas",
                "original request symbology",
            ]
        attested: list[str] = []
        if selection.sessions is not None:
            attested.append(
                f"session calendar {selection.sessions.template!r} "
                f"({selection.sessions.timezone}), as supplied by the caller"
            )
        return TradeSource(
            archive_name=opened.path.name,
            archive_sha256=opened.sha256,
            archive_bytes=opened.status.st_size,
            stored_schema=str(meta.schema),
            dataset=meta.dataset,
            declared_start_ns=meta.start,
            declared_end_ns=meta.end,
            trade_records=records,
            trade_volume=volume,
            checks=tuple(checks),
            selection_dependencies=tuple(dependencies),
            lineage=receipt,
            input_schemas=input_schemas,
            request_stype_in=request_stype_in,
            attested=tuple(attested),
            unavailable=tuple(unavailable),
            limitations=tuple(limitations),
        )


def read_trades(
    path: Path,
    series: SeriesId,
    *,
    sessions: SessionDays | None = None,
    start_ns: int | None = None,
    end_ns: int | None = None,
    lake_root: Path | None = None,
    receipt: Path | None = None,
    batch_records: int | None = None,
) -> TradeRead:
    """Open (pass A, now) and read (pass B, as iterated) in one call. The read
    owns the archive and closes it however it ends."""
    archive = open_trade_archive(path, receipt=receipt)
    try:
        return archive.read(
            series,
            sessions=sessions,
            start_ns=start_ns,
            end_ns=end_ns,
            lake_root=lake_root,
            batch_records=batch_records,
            close_archive=True,
        )
    except BaseException:
        archive.close()
        raise


def summarize_trades(
    path: Path,
    series: SeriesId,
    *,
    sessions: SessionDays | None = None,
    start_ns: int | None = None,
    end_ns: int | None = None,
    lake_root: Path | None = None,
    receipt: Path | None = None,
) -> TradeSummary:
    """Side totals over one explicit selection, only from a validated read.

    Without `sessions` or bounds, that is every record of the contract.
    """
    totals = TradeTotals()
    with read_trades(
        path,
        series,
        sessions=sessions,
        start_ns=start_ns,
        end_ns=end_ns,
        lake_root=lake_root,
        receipt=receipt,
    ) as read:
        for batch in read:
            totals += TradeTotals.of(batch)
        return TradeSummary(totals, read.source)
