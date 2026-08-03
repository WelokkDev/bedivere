"""CSV bar loading — the one place unruly external data gets disciplined.

`load_candles_csv` reads a header-carrying CSV into close-stamped Candles:
sorted ascending, duplicate timestamps refused. Timestamps may be unix
seconds or timezone-AWARE ISO-8601; naive ISO strings are refused — a
timestamp whose timezone must be guessed is not data, it's a bug waiting
for a DST transition.

The close-stamp convention matters: bedivere stamps a bar at its CLOSE
instant (a 5-minute bar covering 18:00:00–18:04:59 is stamped 18:05:00).
Many exports stamp the OPEN. If yours does, shift by the bar period as you
load — feeding open-stamped bars unshifted silently misaligns every
session boundary and HTF bucket downstream.

Loaders for other backends (a database, a broker export) belong beside
this one in bedivere/data/ — the contract is simply "return clean
Candles"; the streams and the engine never fetch data themselves.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from bedivere.core.types import Candle

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


def load_candles_csv(path: str | Path) -> list[Candle]:
    """Read Candles from a CSV with (at least) the REQUIRED_COLUMNS header.
    Extra columns are ignored. Rows come back sorted ascending; duplicate
    timestamps raise."""
    file = Path(path)
    with file.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            raise ValueError(
                f"{file.name}: missing column(s) {missing} — need {list(REQUIRED_COLUMNS)}, got {header}"
            )
        out: list[Candle] = []
        for line_no, row in enumerate(reader, start=2):
            try:
                out.append(
                    Candle(
                        timestamp=parse_timestamp(row["timestamp"]),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                    )
                )
            except (ValueError, TypeError) as e:
                raise ValueError(f"{file.name}:{line_no}: {e}") from e

    out.sort(key=lambda c: c.timestamp)
    for prev, cur in zip(out, out[1:], strict=False):
        if cur.timestamp == prev.timestamp:
            raise ValueError(
                f"{file.name}: duplicate timestamp {cur.timestamp} — de-duplicate upstream; "
                "two bars at one instant cannot both be true"
            )
    return out


def parse_timestamp(text: str) -> int:
    """Unix seconds from a CSV cell: an integer, or timezone-aware ISO-8601.
    Naive ISO (no offset, no Z) is refused loudly."""
    cell = text.strip()
    try:
        return int(cell)
    except ValueError:
        pass
    dt = datetime.fromisoformat(cell)  # raises ValueError on garbage
    if dt.tzinfo is None:
        raise ValueError(
            f"naive ISO timestamp {cell!r} — add an offset (e.g. 2026-07-15T10:00:00-04:00 "
            "or ...Z) or use unix seconds; guessing timezones corrupts session alignment"
        )
    return int(dt.timestamp())
