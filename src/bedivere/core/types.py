"""Core data model: the ONE Candle and the ONE Timeframe.

Every module in bedivere imports Candle and Timeframe from here — there is
deliberately no second bar type anywhere. The Timeframe enum owns the single
copy of the coarseness ordering and the intraday period table.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True, slots=True)
class Candle:
    """Close-stamped OHLCV bar (unix seconds).

    `timestamp` is the CLOSE of the bar, not the open: a 15-minute bar
    covering 18:00:00-18:14:59 ET is stamped 18:15:00.

    `partial` is True on the most-recent aggregated bar when its
    session-day's expected end is in the future — more underlying bars may
    still arrive. None means the field is absent (the common case).
    """

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    partial: bool | None = None


class Timeframe(StrEnum):
    """Timeframe labels with the canonical coarseness ordering.

    `order` is the single source of the "higher TF is coarser" ranking.
    `period_seconds` covers the intraday TFs; "1d" has no fixed intraday
    period and raises — daily bars are session rollups, out of scope for
    the intraday chain.
    """

    S1 = "1s"
    S15 = "15s"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H2 = "2h"
    H4 = "4h"
    D1 = "1d"

    @property
    def order(self) -> int:
        return _TF_ORDER[self]

    @property
    def period_seconds(self) -> int:
        sec = _PERIOD_SECONDS.get(self)
        if sec is None:
            raise ValueError(
                'Timeframe "1d" has no intraday period — daily bars are session rollups, out of scope for the intraday chain.'
            )
        return sec


_TF_ORDER: dict[Timeframe, int] = {
    Timeframe.S1: 0,
    Timeframe.S15: 1,
    Timeframe.M5: 2,
    Timeframe.M15: 3,
    Timeframe.M30: 4,
    Timeframe.H1: 5,
    Timeframe.H2: 6,
    Timeframe.H4: 7,
    Timeframe.D1: 8,
}

_PERIOD_SECONDS: dict[Timeframe, int] = {
    Timeframe.S1: 1,
    Timeframe.S15: 15,
    Timeframe.M5: 5 * 60,
    Timeframe.M15: 15 * 60,
    Timeframe.M30: 30 * 60,
    Timeframe.H1: 60 * 60,
    Timeframe.H2: 120 * 60,
    Timeframe.H4: 240 * 60,
}

SUPPORTED_TIMEFRAMES: tuple[Timeframe, ...] = (
    Timeframe.S1,
    Timeframe.S15,
    Timeframe.M5,
    Timeframe.M15,
    Timeframe.M30,
    Timeframe.H1,
    Timeframe.H2,
    Timeframe.H4,
)
