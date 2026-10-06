"""The bar table's shape, as data — shared by the readers and the write seam.

A declarative spec rather than a library's schema object, checked against DuckDB
by both sides, so a stored file whose types have drifted REFUSES rather than
casting silently into price-sensitive code.

Column semantics:

  * `ts` is CLOSE-stamped unix seconds (the engine's convention everywhere),
    converted once at ingest from the vendor's interval-start nanoseconds.
  * `instrument_id` per bar is what makes a contract roll detectable: the switch
    is invisible in price alone, and an indicator whose window straddles one is
    averaging two different instruments.
  * `synthetic` marks a manufactured bar. Nothing writes True — the lake does
    not forward-fill — but consumers can refuse to trade on one without a
    schema migration if anything ever does.

`BarBatch` is the in-memory unit that crosses the seams. Its plain Python
columns are deliberately free of any columnar library, so the resample reduction
stays bit-identical to `aggregate_bucket`'s in-arrival-order folds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from bedivere.core.types import Candle

# Stamped into every partition's footer by `writer.partition_metadata`, and
# redundant with the path on purpose: a file copied out of the lake still says
# what it is, and `catalog` reads lineage back out of the data.
META_DATASET: Final[str] = "bedivere.dataset"
META_SYMBOL: Final[str] = "bedivere.symbol"
META_SERIES: Final[str] = "bedivere.series"
META_TIMEFRAME: Final[str] = "bedivere.timeframe"
META_SESSION: Final[str] = "bedivere.session"
META_SOURCE: Final[str] = "bedivere.source"
META_PRICE_BASIS: Final[str] = "bedivere.price_basis"

# The `[start, end)` Unix-nanosecond range a source declares it covers. The bars
# cannot say this: a partition starting hours into its session looks like a
# quiet open. A footer without these keys claims no coverage.
META_SOURCE_START_NS: Final[str] = "bedivere.source_start_ns"
META_SOURCE_END_NS: Final[str] = "bedivere.source_end_ns"

AS_TRADED: Final[str] = "as_traded"
"""The only basis the lake STORES. A persisted back-adjusted series changes
value at every subsequent roll, which is how two backtests six months apart come
to disagree about "the same data"."""

BACK_ADJUSTED: Final[str] = "back_adjusted"
"""What the read-time view reports itself as. Never appears in a footer."""


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """One column, its DuckDB type spelled as `DESCRIBE` spells it."""

    name: str
    duck_type: str
    nullable: bool


BAR_COLUMNS: Final[tuple[ColumnSpec, ...]] = (
    ColumnSpec("ts", "BIGINT", nullable=False),
    ColumnSpec("open", "DOUBLE", nullable=False),
    ColumnSpec("high", "DOUBLE", nullable=False),
    ColumnSpec("low", "DOUBLE", nullable=False),
    ColumnSpec("close", "DOUBLE", nullable=False),
    ColumnSpec("volume", "DOUBLE", nullable=False),
    ColumnSpec("instrument_id", "UINTEGER", nullable=False),
    ColumnSpec("synthetic", "BOOLEAN", nullable=False),
)

BAR_COLUMN_NAMES: Final[tuple[str, ...]] = tuple(c.name for c in BAR_COLUMNS)

# The explicit projection every lake query uses. Never `SELECT *`: DuckDB's
# hive-partition auto-detection appends dataset/symbol/series/tf/session as
# columns, and a `*` inside a rewrite would bake them into the file.
BAR_SELECT: Final[str] = ", ".join(BAR_COLUMN_NAMES)


class BarSchemaError(Exception):
    """Stored or in-memory bars do not match the lake's declared shape."""


def check_described_schema(described: list[tuple[str, str]], *, context: str) -> None:
    """Fail-closed dtype gate, run before any value is read.

    `described` is `[(name, type), ...]` from a DuckDB `DESCRIBE` of the
    projection. Anything but an exact match refuses the read, because a file with
    drifted types would otherwise flow, silently cast, into code that decides
    where a stop goes.
    """
    expected = [(c.name, c.duck_type) for c in BAR_COLUMNS]
    if described != expected:
        raise BarSchemaError(
            f"{context}: stored schema {described} does not match the lake's "
            f"declared shape {expected}"
        )


@dataclass(slots=True)
class BarBatch:
    """One session-day (or window) of bars as plain parallel columns.

    `validate_shape` is the cheap structural check; the write seam
    (`lake.writer`) owns the value-level invariants.
    """

    ts: list[int] = field(default_factory=lambda: list[int]())
    open: list[float] = field(default_factory=lambda: list[float]())
    high: list[float] = field(default_factory=lambda: list[float]())
    low: list[float] = field(default_factory=lambda: list[float]())
    close: list[float] = field(default_factory=lambda: list[float]())
    volume: list[float] = field(default_factory=lambda: list[float]())
    instrument_id: list[int] = field(default_factory=lambda: list[int]())
    synthetic: list[bool] = field(default_factory=lambda: list[bool]())

    def __len__(self) -> int:
        return len(self.ts)

    def validate_shape(self) -> None:
        lengths = {
            "ts": len(self.ts),
            "open": len(self.open),
            "high": len(self.high),
            "low": len(self.low),
            "close": len(self.close),
            "volume": len(self.volume),
            "instrument_id": len(self.instrument_id),
            "synthetic": len(self.synthetic),
        }
        if len(set(lengths.values())) != 1:
            raise BarSchemaError(f"ragged BarBatch: column lengths {lengths}")

    def append(
        self,
        ts: int,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        instrument_id: int,
        *,
        synthetic: bool = False,
    ) -> None:
        self.ts.append(ts)
        self.open.append(open_)
        self.high.append(high)
        self.low.append(low)
        self.close.append(close)
        self.volume.append(volume)
        self.instrument_id.append(instrument_id)
        self.synthetic.append(synthetic)

    def candles(self, *, drop_synthetic: bool = False) -> list[Candle]:
        """The engine's view of this batch. `partial` is left None: a stored bar
        is complete by construction, and partial-marking is the live
        aggregator's job."""
        return [
            Candle(
                timestamp=self.ts[i],
                open=self.open[i],
                high=self.high[i],
                low=self.low[i],
                close=self.close[i],
                volume=self.volume[i],
            )
            for i in range(len(self))
            if not (drop_synthetic and self.synthetic[i])
        ]

    @classmethod
    def from_candles(
        cls, bars: list[Candle], *, instrument_id: int, synthetic: bool = False
    ) -> BarBatch:
        return cls(
            ts=[c.timestamp for c in bars],
            open=[c.open for c in bars],
            high=[c.high for c in bars],
            low=[c.low for c in bars],
            close=[c.close for c in bars],
            volume=[c.volume for c in bars],
            instrument_id=[instrument_id] * len(bars),
            synthetic=[synthetic] * len(bars),
        )

    @classmethod
    def from_rows(
        cls, rows: list[tuple[int, float, float, float, float, float, int, bool]]
    ) -> BarBatch:
        """From fetched query rows in `BAR_COLUMN_NAMES` order."""
        batch = cls()
        for ts, open_, high, low, close, volume, instrument_id, synthetic in rows:
            batch.ts.append(int(ts))
            batch.open.append(float(open_))
            batch.high.append(float(high))
            batch.low.append(float(low))
            batch.close.append(float(close))
            batch.volume.append(float(volume))
            batch.instrument_id.append(int(instrument_id))
            batch.synthetic.append(bool(synthetic))
        return batch
