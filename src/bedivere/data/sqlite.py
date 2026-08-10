"""SqliteCandleStore — the zero-infrastructure candle cache.

One file, no server, no daemon: the storage a cloner can actually adopt on
day one. It implements `CandleSource`, so a run reading from it is wired
exactly like a run reading a CSV.

Schema — one table, and the PRIMARY KEY IS the index:

    CREATE TABLE candles (
      symbol TEXT, timeframe TEXT, timestamp INTEGER,
      open REAL, high REAL, low REAL, close REAL, volume REAL,
      PRIMARY KEY (symbol, timeframe, timestamp)
    ) WITHOUT ROWID;

`WITHOUT ROWID` makes that key the clustered B-tree, so rows for one
(symbol, timeframe) sit on disk in timestamp order and the only query the
engine ever issues — a range scan over exactly those three leading columns —
is a contiguous seek that never touches a second structure. A secondary
index on the same columns would be a duplicate of the table. The inventory
query (`GROUP BY symbol, timeframe`) rides the same key's leading prefix.

The key also makes the upsert path trivial and idempotent: re-importing an
overlapping CSV corrects the bars it overlaps instead of duplicating them,
which is what you want the second time a vendor re-issues a session.

WAL is enabled so a reader never blocks the writer; readers may additionally
open `read_only=True` (`mode=ro`), which is the honest default for anything
a live run touches.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from bedivere.core.types import Candle, Timeframe

SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    symbol    TEXT    NOT NULL,
    timeframe TEXT    NOT NULL,
    timestamp INTEGER NOT NULL,
    open      REAL    NOT NULL,
    high      REAL    NOT NULL,
    low       REAL    NOT NULL,
    close     REAL    NOT NULL,
    volume    REAL    NOT NULL,
    PRIMARY KEY (symbol, timeframe, timestamp)
) WITHOUT ROWID;
"""

_SELECT_RANGE = (
    "SELECT timestamp, open, high, low, close, volume FROM candles "
    "WHERE symbol = ? AND timeframe = ? AND timestamp > ? AND timestamp <= ? "
    "ORDER BY timestamp ASC"
)

_UPSERT = (
    "INSERT INTO candles (symbol, timeframe, timestamp, open, high, low, close, volume) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
    "ON CONFLICT (symbol, timeframe, timestamp) DO UPDATE SET "
    "open = excluded.open, high = excluded.high, low = excluded.low, "
    "close = excluded.close, volume = excluded.volume"
)

_INVENTORY = (
    "SELECT symbol, timeframe, COUNT(*), MIN(timestamp), MAX(timestamp) "
    "FROM candles GROUP BY symbol, timeframe ORDER BY symbol, timeframe"
)


class CandleStoreError(RuntimeError):
    """The store file is missing, unreadable, or not a bedivere cache."""


@dataclass(frozen=True, slots=True)
class Series:
    """One (symbol, timeframe) series present in a store."""

    symbol: str
    timeframe: str
    bars: int
    first_unix: int
    last_unix: int


@dataclass(frozen=True, slots=True)
class UpsertStats:
    """What an import actually did. `replaced` is counted BEFORE the write —
    "imported 500 bars" reads very differently once you know 480 of them
    overwrote bars that were already there."""

    written: int
    replaced: int

    @property
    def inserted(self) -> int:
        return self.written - self.replaced


class SqliteCandleStore:
    """A candle cache in one SQLite file. Implements `CandleSource`."""

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        file = Path(path)
        self._path = file
        self._read_only = read_only
        if read_only:
            if not file.exists():
                raise CandleStoreError(
                    f"candle store not found at {file} — import some bars first "
                    "(python -m bedivere.data import --db ... --symbol ... --timeframe ... FILE.csv)"
                )
            uri = f"{file.resolve().as_uri()}?mode=ro"
            self._conn = sqlite3.connect(uri, uri=True)
        else:
            file.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(file)
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.executescript(SCHEMA)
            self._conn.commit()
        self._conn.execute("PRAGMA busy_timeout = 5000")

    # ---------- CandleSource ----------

    def candles(
        self, symbol: str, timeframe: Timeframe, start_unix: int, end_unix: int
    ) -> list[Candle]:
        """Bars in `(start_unix, end_unix]`, ascending. The PK ordering is the
        row order, so no sort happens here or in SQLite."""
        rows: list[tuple[int, float, float, float, float, float]] = self._conn.execute(
            _SELECT_RANGE, (symbol, timeframe.value, start_unix, end_unix)
        ).fetchall()
        return [
            Candle(
                timestamp=int(ts),
                open=float(o),
                high=float(h),
                low=float(low),
                close=float(c),
                volume=float(v),
            )
            for ts, o, h, low, c, v in rows
        ]

    def describe(self) -> str:
        suffix = " (read-only)" if self._read_only else ""
        return f"sqlite:{self._path}{suffix}"

    # ---------- writing ----------

    def upsert(self, symbol: str, timeframe: Timeframe, bars: Sequence[Candle]) -> UpsertStats:
        """Insert-or-replace one series' bars in a single transaction. Safe to
        re-run over the same file: the key makes it idempotent."""
        if self._read_only:
            raise CandleStoreError(f"{self._path} was opened read-only — cannot upsert")
        if not bars:
            return UpsertStats(written=0, replaced=0)
        stamps = [b.timestamp for b in bars]
        existing: list[tuple[int]] = self._conn.execute(
            "SELECT timestamp FROM candles WHERE symbol = ? AND timeframe = ? "
            "AND timestamp >= ? AND timestamp <= ?",
            (symbol, timeframe.value, min(stamps), max(stamps)),
        ).fetchall()
        replaced = len({int(row[0]) for row in existing} & set(stamps))
        with self._conn:  # commit on success, rollback on any exception
            self._conn.executemany(
                _UPSERT,
                [
                    (symbol, timeframe.value, b.timestamp, b.open, b.high, b.low, b.close, b.volume)
                    for b in bars
                ],
            )
        return UpsertStats(written=len(bars), replaced=replaced)

    # ---------- inventory ----------

    def inventory(self) -> list[Series]:
        """Every series in the store — what `bedivere-data list` prints, and
        the first thing to look at when a run comes back empty."""
        rows: list[tuple[str, str, int, int, int]] = self._conn.execute(_INVENTORY).fetchall()
        return [
            Series(
                symbol=str(symbol),
                timeframe=str(timeframe),
                bars=int(count),
                first_unix=int(first),
                last_unix=int(last),
            )
            for symbol, timeframe, count, first, last in rows
        ]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SqliteCandleStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def import_candles(
    db_path: str | Path,
    *,
    symbol: str,
    timeframe: Timeframe,
    bars: Iterable[Candle],
) -> UpsertStats:
    """Convenience for the import command: open, upsert, close."""
    with SqliteCandleStore(db_path) as store:
        return store.upsert(symbol, timeframe, list(bars))
