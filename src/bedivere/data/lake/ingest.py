"""DBN in, canonical Parquet out.

Vendor conventions are normalised once, here, so nothing downstream carries
them:

  * `ts_event` marks the interval START; bedivere close-stamps (`ts_event +
    period`), the convention every other module already assumes
  * prices are int64 fixed-point at 1e-9
  * `UNDEF_PRICE` / `UNDEF_TIMESTAMP` are sentinel maxima, not values
  * calendar spreads (`NQM6-NQZ6`) price a DIFFERENCE, not an instrument, and
    must never enter a series as if they were bars

Session geometry is handed over, never derived — this module decides nothing
about when a session starts, exactly like the rest of the engine.

No forward-fill: a second with no trades prints no record and we write no row,
because absence of a bar means absence of trades and a cloned bar would replay
its high/low range into a quiet stretch. Writes go through the one seam,
`lake.writer.write_day`.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final, cast

import databento_dbn

from bedivere.core.session_days import SessionDay, SessionDays
from bedivere.core.types import Timeframe
from bedivere.data.lake.layout import RollRule, SeriesId
from bedivere.data.lake.schema import BarBatch
from bedivere.data.lake.writer import write_day

PRICE_SCALE: Final[int] = 1_000_000_000
NS_PER_SEC: Final[int] = 1_000_000_000
UNDEF_PRICE: Final[int] = 2**63 - 1
UNDEF_TIMESTAMP: Final[int] = 2**64 - 1

_CHUNK_BYTES: Final[int] = 1 << 20

# Month codes, in the CME order. Shared by OUTRIGHT and contract_pattern.
_MONTH_CODES: Final[str] = "FGHJKMNQUVXZ"

# Any outright contract (NQZ5). Hyphenated calendar spreads are excluded.
OUTRIGHT: Final[re.Pattern[str]] = re.compile(rf"^[A-Z0-9]{{1,4}}[{_MONTH_CODES}]\d{{1,2}}$")


def contract_pattern(root: str) -> re.Pattern[str]:
    """Outright contracts of ONE root, e.g. `NQ` -> NQZ5 but not MNQZ5.

    One batch job can interleave ES.FUT + NQ.FUT + MES.FUT + MNQ.FUT in a single
    stream, and `OUTRIGHT` matches all four — ranking front months against that
    set would crown one winner across unrelated products.
    """
    return re.compile(rf"^{re.escape(root)}[{_MONTH_CODES}]\d{{1,2}}$")


class IngestError(Exception):
    """The file cannot be ingested as the series it was declared to be."""


def _compression_of(path: Path) -> databento_dbn.Compression:
    # The stubs type the enum members as `str`, but the runtime objects are
    # Compression variants — hence the casts.
    if path.name.endswith(".dbn.zst") or path.name.endswith(".zst"):
        return cast("databento_dbn.Compression", databento_dbn.Compression.ZSTD)
    if path.name.endswith(".dbn"):
        return cast("databento_dbn.Compression", databento_dbn.Compression.NONE)
    raise IngestError(f"{path.name}: not a .dbn or .dbn.zst file")


def _stream(path: Path) -> Iterator[object]:
    """Every DBN record in `path`, `Metadata` first, streamed chunk-wise so
    memory stays flat regardless of file size."""
    decoder = databento_dbn.DBNDecoder(compression=_compression_of(path))
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK_BYTES):
            decoder.write(chunk)
            yield from cast("list[object]", decoder.decode())


def read_metadata(path: Path) -> databento_dbn.Metadata:
    """The DBN header. Cheap: decoding stops at the first batch."""
    for rec in _stream(path):
        if isinstance(rec, databento_dbn.Metadata):
            return rec
        raise IngestError(f"{path.name}: first DBN record is {type(rec).__name__}, not Metadata")
    raise IngestError(f"{path.name}: no DBN metadata header")


@dataclass(frozen=True, slots=True)
class DbnRange:
    """What a DBN file says about itself, without decoding its body."""

    dataset: str
    schema: str
    start_unix: int
    end_unix: int


@dataclass(frozen=True, slots=True)
class IngestReport:
    series: SeriesId
    timeframe: Timeframe
    days_written: int
    bars_written: int
    records_scanned: int
    records_off_session: int
    records_spread: int
    records_other_root: int
    """Outrights belonging to a different product in the same file."""

    rolls: tuple[tuple[str, str], ...]
    """(session-day label, contract) at each change of instrument."""

    def describe(self) -> str:
        roll = " | ".join(f"{label} -> {name}" for label, name in self.rolls)
        return (
            f"{self.series} {self.timeframe.value}: "
            f"{self.bars_written:,} bars across {self.days_written} session-days "
            f"(scanned {self.records_scanned:,}, dropped "
            f"{self.records_other_root:,} other-root / "
            f"{self.records_spread:,} spread / "
            f"{self.records_off_session:,} off-session)\n  roll: {roll}"
        )


def read_range(path: Path) -> DbnRange:
    """The file's self-declared dataset, schema and range. Cheap: header only."""
    meta = read_metadata(path)
    return DbnRange(
        dataset=str(meta.dataset),
        schema=str(meta.schema),
        start_unix=meta.start // NS_PER_SEC,
        end_unix=meta.end // NS_PER_SEC,
    )


@dataclass(frozen=True, slots=True)
class ObservedRange:
    """What a DBN file actually CONTAINS, as opposed to what it claims."""

    first_unix: int
    last_unix: int
    records: int


def observed_range(path: Path) -> ObservedRange:
    first: int | None = None
    last = 0
    count = 0
    for rec in _ohlcv_records(path):
        count += 1
        if first is None:
            first = rec.ts_event // NS_PER_SEC
        last = rec.ts_event // NS_PER_SEC
    if first is None:
        raise IngestError(f"{path.name} contains no OHLCV records")
    return ObservedRange(first_unix=first, last_unix=last, records=count)


def check_complete(path: Path, *, max_trailing_gap_days: float = 4.0) -> ObservedRange:
    """Refuse a DBN file whose records stop well short of its declared range.

    An interrupted download keeps its truthful header while holding only the
    bytes that arrived, so everything downstream succeeds over the fraction that
    made it. The tolerance allows a real trailing gap (a range ending on a
    Friday, a holiday weekend) but not a truncation.
    """
    declared = read_range(path)
    got = observed_range(path)
    gap_days = (declared.end_unix - got.last_unix) / 86_400
    if gap_days > max_trailing_gap_days:
        covered = (got.last_unix - got.first_unix) / 86_400
        claimed = (declared.end_unix - declared.start_unix) / 86_400
        raise IngestError(
            f"{path.name} looks truncated: records end "
            f"{gap_days:.1f} days before the declared range end.\n"
            f"  declared: {_iso(declared.start_unix)} .. {_iso(declared.end_unix)}  ({claimed:.0f} days)\n"
            f"  actual:   {_iso(got.first_unix)} .. {_iso(got.last_unix)}  ({covered:.0f} days, {got.records:,} records)\n"
            "Re-download, or raise max_trailing_gap_days if the gap is genuine."
        )
    return got


def _iso(unix: int) -> str:
    return datetime.fromtimestamp(unix, tz=UTC).strftime("%Y-%m-%d")


class SymbolResolver:
    """`(instrument_id, timestamp) -> raw_symbol`, honouring the header's time
    intervals.

    Flattening those intervals into a `dict[int, str]` is WRONG over a long
    file: ids are unique only within a day and get reused, so the same id can
    name a front-month outright in one era and a calendar spread in another, and
    the outright filter then drops real bars silently. Raw symbols repeat across
    decades too (`NQM0` is June 2010 AND June 2020).
    """

    __slots__ = ("_multi", "_single")

    def __init__(self, mappings: dict[str, list[dict[str, object]]]) -> None:
        by_id: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
        for raw_symbol, intervals in mappings.items():
            for interval in intervals:
                iid = interval.get("symbol")
                if not iid:
                    continue
                start = interval.get("start_date")
                end = interval.get("end_date")
                by_id[int(cast("str", iid))].append(
                    (_date_to_unix(start), _date_to_unix(end), raw_symbol)
                )

        # Fast path: nearly every id means one symbol for all time.
        self._single: dict[int, str] = {}
        self._multi: dict[int, list[tuple[int, int, str]]] = {}
        for iid, spans in by_id.items():
            names = {name for _, _, name in spans}
            if len(names) == 1:
                self._single[iid] = names.pop()
            else:
                self._multi[iid] = sorted(spans)

    def resolve(self, instrument_id: int, unix_sec: int) -> str:
        """The raw_symbol this id meant at `unix_sec`, or "" if unknown."""
        name = self._single.get(instrument_id)
        if name is not None:
            return name
        spans = self._multi.get(instrument_id)
        if spans is None:
            return ""
        for start, end, candidate in spans:
            # Symbology intervals are half-open [start_date, end_date).
            if start <= unix_sec < end:
                return candidate
        return ""

    @property
    def ambiguous_ids(self) -> list[int]:
        """Ids that mean different symbols at different times."""
        return sorted(self._multi)

    def __len__(self) -> int:
        return len(self._single) + len(self._multi)


def _date_to_unix(value: object) -> int:
    if isinstance(value, datetime):
        return int(value.replace(tzinfo=UTC).timestamp())
    if isinstance(value, date):
        return int(datetime(value.year, value.month, value.day, tzinfo=UTC).timestamp())
    raise IngestError(f"symbology interval bound is not a date: {value!r}")


def build_symbology(meta: databento_dbn.Metadata) -> SymbolResolver:
    """Time-aware `instrument_id -> raw_symbol` from the DBN header.

    `Metadata.mappings` is `dict[raw_symbol, list[{start_date, end_date,
    symbol}]]`, dates as `datetime.date` and the id as a string.
    """
    return SymbolResolver(cast("dict[str, list[dict[str, object]]]", meta.mappings))


def _ohlcv_records(path: Path) -> Iterator[databento_dbn.OHLCVMsg]:
    """OHLCV records only. A DBN stream also carries symbol-mapping, system and
    error records, and treating one of those as a bar would read whatever sits at
    those struct offsets."""
    for rec in _stream(path):
        if isinstance(rec, databento_dbn.OHLCVMsg):
            yield rec


def _close_stamp(rec: databento_dbn.OHLCVMsg, period_seconds: int) -> int:
    """Interval-start nanoseconds -> close-stamped unix seconds."""
    return rec.ts_event // NS_PER_SEC + period_seconds


def _check_prices(rec: databento_dbn.OHLCVMsg, ts: int) -> None:
    for name, value in (
        ("open", rec.open),
        ("high", rec.high),
        ("low", rec.low),
        ("close", rec.close),
    ):
        if value == UNDEF_PRICE:
            raise IngestError(f"UNDEF_PRICE in {name} at ts={ts} instrument={rec.instrument_id}")
    if rec.ts_event == UNDEF_TIMESTAMP:
        raise IngestError(f"UNDEF_TIMESTAMP on record for instrument={rec.instrument_id}")


def causal_front_months(
    path: Path,
    symbology: SymbolResolver,
    days: SessionDays,
    period_seconds: int,
    root: str,
    roll: RollRule = RollRule.VOLUME,
) -> dict[int, int]:
    """Front month per session-day, ranked on the volume of the most recent
    PRIOR day that has records.

    Same-day volume is not knowable until the day is over, so ranking on it puts
    lookahead on the roll day itself — exactly where price gaps. D-1 removes it,
    and matches what a vendor's own `.v.` continuous series means.

    "Most recent prior day with records", not "the calendar slot before": a
    listed holiday or a hole in the archive has no volume to rank on, and
    falling back to the day's own volume there would reintroduce the leak on
    precisely the day a roll lands after a gap. The last winner actually
    observed is carried across instead, so the roll takes effect one day later
    than the leak would have put it — which is when it was knowable.

    The first session-day has no predecessor and is ranked on itself: warm-up,
    not tradeable.
    """
    if roll is not RollRule.VOLUME:
        raise NotImplementedError(
            f"local front-month selection supports {RollRule.VOLUME.value!r} only; "
            f"{roll.value!r} needs open-interest or expiry data this file does not carry"
        )

    resolve = days.make_resolver()
    index_of = {day.label: i for i, day in enumerate(days.days)}
    volume: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    pattern = contract_pattern(root)

    for rec in _ohlcv_records(path):
        ts = _close_stamp(rec, period_seconds)
        if not pattern.match(symbology.resolve(rec.instrument_id, rec.ts_event // NS_PER_SEC)):
            continue
        day = resolve(ts)
        if day is None:
            continue
        volume[index_of[day.label]][rec.instrument_id] += rec.volume

    # Ties break on the lower instrument id, so the ranking is deterministic
    # across re-runs of the same archive.
    winners: dict[int, int] = {}
    for di, per_instrument in volume.items():
        winners[di] = max(per_instrument.items(), key=lambda kv: (kv[1], -kv[0]))[0]

    # Shift forward: day D trades the contract that led on the last day with
    # records before it. A day with none has no entry — there is nothing to keep.
    front: dict[int, int] = {}
    carried: int | None = None
    for di in sorted(volume):
        front[di] = winners[di] if carried is None else carried
        carried = winners[di]
    return front


class _DayBuffer:
    """Column accumulator for one session-day: vendor units in, canonical
    `BarBatch` units out."""

    __slots__ = ("batch",)

    def __init__(self) -> None:
        self.batch = BarBatch()

    def __len__(self) -> int:
        return len(self.batch)

    def add(self, ts: int, rec: databento_dbn.OHLCVMsg) -> None:
        self.batch.append(
            ts,
            rec.open / PRICE_SCALE,
            rec.high / PRICE_SCALE,
            rec.low / PRICE_SCALE,
            rec.close / PRICE_SCALE,
            float(rec.volume),
            rec.instrument_id,
        )


def ingest_dbn(
    path: Path,
    root: Path,
    sid: SeriesId,
    timeframe: Timeframe,
    days: SessionDays,
    *,
    front_months: dict[int, int] | None = None,
    dry_run: bool = False,
) -> IngestReport:
    """Decode `path` into per-session-day Parquet partitions under `root`.

    `front_months` maps session-day INDEX (into `days.days`) to the
    instrument_id to keep — pass `causal_front_months` for a `parent`-symbology
    file that holds every contract, or None when the file is already a single
    series and every matching record should be kept.

    Records arrive in time order, so days are flushed as the walk crosses each
    boundary and memory stays at one session-day however long the file is.
    """
    symbology = build_symbology(read_metadata(path))
    period = timeframe.period_seconds
    resolve = days.make_resolver()
    index_of = {day.label: i for i, day in enumerate(days.days)}

    pattern = contract_pattern(sid.symbol)
    buffer = _DayBuffer()
    current: SessionDay | None = None
    scanned = off_session = spread = other_root = 0
    days_written = bars_written = 0
    rolls: list[tuple[str, str]] = []
    last_instrument: int | None = None

    def flush() -> None:
        nonlocal buffer, days_written, bars_written
        if current is None or len(buffer) == 0:
            buffer = _DayBuffer()
            return
        if not dry_run:
            write_day(root, sid, timeframe, current.label, buffer.batch, source=path.name)
        days_written += 1
        bars_written += len(buffer)
        buffer = _DayBuffer()

    for rec in _ohlcv_records(path):
        scanned += 1
        name = symbology.resolve(rec.instrument_id, rec.ts_event // NS_PER_SEC)
        if not pattern.match(name):
            # Split so the report distinguishes "also holds other products"
            # from "holds calendar spreads".
            if OUTRIGHT.match(name):
                other_root += 1
            else:
                spread += 1
            continue
        ts = _close_stamp(rec, period)
        day = resolve(ts)
        if day is None:
            off_session += 1
            continue
        if front_months is not None and front_months.get(index_of[day.label]) != rec.instrument_id:
            continue
        _check_prices(rec, ts)

        if current is None or day.label != current.label:
            flush()
            current = day
        if rec.instrument_id != last_instrument:
            rolls.append((day.label, name or str(rec.instrument_id)))
            last_instrument = rec.instrument_id
        buffer.add(ts, rec)

    flush()

    return IngestReport(
        series=sid,
        timeframe=timeframe,
        days_written=days_written,
        bars_written=bars_written,
        records_scanned=scanned,
        records_off_session=off_session,
        records_spread=spread,
        records_other_root=other_root,
        rolls=tuple(rolls),
    )
