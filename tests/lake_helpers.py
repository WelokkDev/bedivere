"""Shared seeding for the lake tests. Skips when the `lake` extra is absent."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("duckdb", reason="the lake extra is not installed")
pytest.importorskip("pandas", reason="the lake extra is not installed")

from bedivere.core.session_days import SessionDay  # noqa: E402
from bedivere.core.types import Candle, Timeframe  # noqa: E402
from bedivere.data.lake.schema import BarBatch  # noqa: E402
from bedivere.data.lake.writer import write_day  # noqa: E402

Row = tuple[int, float, float, float, float, float, int]


def lake_table(rows: list[Row]) -> BarBatch:
    """(ts, o, h, l, c, v, instrument_id) -> a bar batch."""
    return BarBatch(
        ts=[r[0] for r in rows],
        open=[r[1] for r in rows],
        high=[r[2] for r in rows],
        low=[r[3] for r in rows],
        close=[r[4] for r in rows],
        volume=[r[5] for r in rows],
        instrument_id=[r[6] for r in rows],
        synthetic=[False] * len(rows),
    )


def rows_of(batch: BarBatch) -> list[dict[str, object]]:
    """Row-dict view, so two batches compare field by field in one assertion."""
    return [
        {
            "ts": batch.ts[i],
            "open": batch.open[i],
            "high": batch.high[i],
            "low": batch.low[i],
            "close": batch.close[i],
            "volume": batch.volume[i],
            "instrument_id": batch.instrument_id[i],
            "synthetic": batch.synthetic[i],
        }
        for i in range(len(batch))
    ]


def ramp(start_ts: int, count: int, instrument: int = 42, step: int = 1) -> list[Row]:
    """`count` consecutive bars with distinct, NON-MONOTONE extremes.

    The non-monotonicity is the point: on a rising ramp "max of maxes" and "the
    last bar's high" are the same number, so a wrong reduction would pass.
    """
    rows: list[Row] = []
    for i in range(count):
        base = 100.0 + (i % 7)
        rows.append(
            (start_ts + i * step, base, base + 2.0, base - 1.0, base + 0.5, float(i + 1), instrument)
        )
    return rows


def seed_day(
    root: Path, sid: object, timeframe: Timeframe, day: SessionDay, rows: list[Row]
) -> None:
    write_day(root, sid, timeframe, day.label, lake_table(rows), source="test")  # pyright: ignore[reportArgumentType]


def flat_bars(stamps: list[int], level: float, volume: float = 7.0) -> list[Candle]:
    return [
        Candle(timestamp=ts, open=level, high=level + 2, low=level - 2, close=level, volume=volume)
        for ts in stamps
    ]
