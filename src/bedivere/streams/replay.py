"""ReplayStream — the backtest BarStream: bring-your-own-bars, replayed.

bedivere deliberately has no data-fetching layer: how bars reach you (CSV,
a database, a broker export) is your business. This stream takes an
in-memory list of close-stamped Candles and replays them as BarEvents;
`backfill` is always False in replay.

The constructor VALIDATES ordering rather than mending it: bars must be
strictly ascending by timestamp. Sorting is data preparation and belongs at
the loader (bedivere.data sorts and rejects duplicates); a disordered
series reaching the engine boundary is a defect worth stopping on.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from bedivere.core.types import Candle, Timeframe
from bedivere.engine.events import BarEvent


class ReplayStream:
    """Bars for one (symbol, timeframe), strictly ascending, replayed as
    BarEvents."""

    __slots__ = ("_bars", "_symbol", "_timeframe")

    def __init__(
        self,
        *,
        symbol: str,
        timeframe: Timeframe,
        bars: Sequence[Candle],
    ) -> None:
        for prev, cur in zip(bars, bars[1:], strict=False):
            if cur.timestamp <= prev.timestamp:
                raise ValueError(
                    f"ReplayStream: bar {cur.timestamp} is not after {prev.timestamp} — "
                    "sort and de-duplicate at the loader (bedivere.data), not here"
                )
        self._symbol = symbol
        self._timeframe = timeframe
        self._bars = list(bars)

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def timeframe(self) -> Timeframe:
        return self._timeframe

    @property
    def bars(self) -> list[Candle]:
        """The series (ascending). Callers must not mutate."""
        return self._bars

    def __len__(self) -> int:
        return len(self._bars)

    def __iter__(self) -> Iterator[BarEvent]:
        symbol = self._symbol
        timeframe = self._timeframe
        for candle in self._bars:
            yield BarEvent(
                ts=candle.timestamp, symbol=symbol, timeframe=timeframe, candle=candle
            )
