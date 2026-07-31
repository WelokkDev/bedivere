"""ReplayStream ordering contract + the CSV loader's discipline."""

from __future__ import annotations

from pathlib import Path

import pytest

from bedivere.core.types import Timeframe
from bedivere.data import load_candles_csv, parse_timestamp
from bedivere.streams import ReplayStream
from tests.helpers import bar


def test_stream_yields_barevents_in_order() -> None:
    bars = [bar(1000 + i * 300, 1, 2, 0.5, 1.5, v=i) for i in range(5)]
    stream = ReplayStream(symbol="DEMO", timeframe=Timeframe.M5, bars=bars)
    events = list(stream)
    assert len(stream) == 5
    assert [e.ts for e in events] == [b.timestamp for b in bars]
    for e, b in zip(events, bars, strict=True):
        assert e.candle == b
        assert e.symbol == "DEMO"
        assert e.timeframe == Timeframe.M5
        assert e.backfill is False


def test_stream_refuses_disorder_and_duplicates() -> None:
    a, b2 = bar(1000, 1, 2, 0.5, 1.5), bar(1300, 1, 2, 0.5, 1.5)
    with pytest.raises(ValueError, match="not after"):
        ReplayStream(symbol="DEMO", timeframe=Timeframe.M5, bars=[b2, a])
    with pytest.raises(ValueError, match="not after"):
        ReplayStream(symbol="DEMO", timeframe=Timeframe.M5, bars=[a, a])


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "bars.csv"
    p.write_text(text, encoding="utf-8")
    return p


def test_csv_loads_sorts_and_types(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "timestamp,open,high,low,close,volume,extra\n"
        "1300,2,3,1,2.5,20,ignored\n"
        "1000,1,2,0.5,1.5,10,ignored\n",
    )
    candles = load_candles_csv(p)
    assert [c.timestamp for c in candles] == [1000, 1300]  # sorted ascending
    assert candles[0].open == 1.0 and candles[1].volume == 20.0


def test_csv_refuses_duplicates_and_missing_columns(tmp_path: Path) -> None:
    dup = _write(tmp_path, "timestamp,open,high,low,close,volume\n1,1,1,1,1,1\n1,2,2,2,2,2\n")
    with pytest.raises(ValueError, match="duplicate timestamp"):
        load_candles_csv(dup)
    missing = _write(tmp_path, "timestamp,open,high,low,close\n1,1,1,1,1\n")
    with pytest.raises(ValueError, match="missing column"):
        load_candles_csv(missing)


def test_timestamp_forms() -> None:
    assert parse_timestamp("1784930400") == 1784930400
    # tz-aware ISO — offset and Z both fine.
    assert parse_timestamp("2026-07-15T10:00:00-04:00") == parse_timestamp(
        "2026-07-15T14:00:00Z"
    )
    with pytest.raises(ValueError, match="naive ISO"):
        parse_timestamp("2026-07-15T10:00:00")
