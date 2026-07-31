"""Body-direction classification — a tiny OHLC helper for strategy code."""

from __future__ import annotations

from typing import Literal

from bedivere.core.types import Candle

CandleDirection = Literal["bullish", "bearish", "doji"]


def get_candle_direction(candle: Candle) -> CandleDirection:
    """Classifies a candle by body direction. Pure OHLC math."""
    if candle.close > candle.open:
        return "bullish"
    if candle.close < candle.open:
        return "bearish"
    return "doji"
