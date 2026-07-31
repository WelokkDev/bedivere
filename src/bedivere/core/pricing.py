"""Integer price/money core.

Inside the engine, prices are int TICKS on the instrument grid and money is
int CENTS — float arithmetic never touches a decision or a PnL number.
Floats exist only at the boundary (bar files, config JSON, display), and
every float→tick conversion happens here, exactly:

  - Boundary floats are read as the decimal they display as
    (`Fraction(Decimal(str(x)))`) — a price of 20000.25 means the decimal
    20000.25, not the nearest binary double.
  - `price_to_ticks` REFUSES off-grid input (loud beats silently rounded:
    an off-grid bar price means a mis-specified instrument).
  - Computed levels (stop/target math) go through `conservative_ticks`,
    the ONE away-from-profit rounding rule: prices we BUY at round UP,
    prices we SELL at round DOWN — strictly worse-or-equal for our side,
    for entry, stop, and target legs alike.

The instrument numbers themselves (tickSize, pointValue) are caller-supplied
data — nothing here hardcodes an instrument.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Literal

Side = Literal["buy", "sell"]


class PricingError(ValueError):
    """Invalid instrument spec or money amount."""


class OffGridPriceError(PricingError):
    """A price that must sit on the tick grid does not."""


def _decimal_fraction(value: float | int | str, what: str) -> Fraction:
    """The decimal `value` displays as, exactly. `str` of a float is its
    shortest round-trip repr — the decimal the producer wrote."""
    if isinstance(value, bool):
        raise PricingError(f"{what} must be a number, not bool")
    try:
        # NaN survives Decimal() and fails in Fraction() (ValueError);
        # Infinity fails there with OverflowError — catch both.
        return Fraction(Decimal(str(value)))
    except (InvalidOperation, ValueError, OverflowError) as e:
        raise PricingError(f"{what} is not a finite decimal: {value!r}") from e


@dataclass(frozen=True, slots=True)
class InstrumentSpec:
    """One instrument's pricing grid: exact tick size plus the exact cent
    value of one tick per contract. Built from handoff numbers via
    `spec_from_handoff` (which validates integrality)."""

    symbol: str
    tick_size: Fraction  # exact, > 0 (e.g. 1/4 for NQ)
    tick_value_cents: int  # exact cents per tick per contract (> 0)

    def __post_init__(self) -> None:
        if self.tick_size <= 0:
            raise PricingError(f"{self.symbol}: tick_size must be > 0, got {self.tick_size}")
        if self.tick_value_cents <= 0:
            raise PricingError(
                f"{self.symbol}: tick_value_cents must be a positive integer, got {self.tick_value_cents}"
            )


def spec_from_handoff(
    symbol: str, tick_size: float | str, point_value: float | str
) -> InstrumentSpec:
    """Build a spec from an instruments handoff `{tickSize, pointValue}`.

    tick_value = tick_size × point_value must be whole cents — a fractional-
    cent tick value cannot be represented in the integer core and means the
    handoff numbers are wrong for this engine.
    """
    tick = _decimal_fraction(tick_size, f"{symbol}.tickSize")
    point = _decimal_fraction(point_value, f"{symbol}.pointValue")
    if tick <= 0:
        raise PricingError(f"{symbol}.tickSize must be > 0, got {tick_size!r}")
    if point <= 0:
        raise PricingError(f"{symbol}.pointValue must be > 0, got {point_value!r}")
    tick_value = tick * point * 100
    if tick_value.denominator != 1:
        raise PricingError(
            f"{symbol}: tickSize {tick_size!r} × pointValue {point_value!r} = ${float(tick * point)} per tick is not whole cents"
        )
    return InstrumentSpec(symbol=symbol, tick_size=tick, tick_value_cents=int(tick_value))


# ---------- price <-> ticks ----------


def price_to_ticks(price: float, spec: InstrumentSpec) -> int:
    """Exact conversion of an on-grid price to ticks; OffGridPriceError if
    the price does not sit on the instrument grid. Use for prices that are
    grid-truth by contract (bar OHLC, broker fills)."""
    q = _decimal_fraction(price, f"{spec.symbol} price") / spec.tick_size
    if q.denominator != 1:
        raise OffGridPriceError(
            f"{spec.symbol}: price {price!r} is not on the {float(spec.tick_size)} tick grid"
        )
    return int(q)


def ticks_to_price(ticks: int, spec: InstrumentSpec) -> float:
    """Display/boundary conversion back to a float price."""
    return float(spec.tick_size * ticks)


def floor_to_ticks(price: float, spec: InstrumentSpec) -> int:
    """Largest tick <= price (exact decimal reading of `price`)."""
    return math.floor(_decimal_fraction(price, f"{spec.symbol} price") / spec.tick_size)


def ceil_to_ticks(price: float, spec: InstrumentSpec) -> int:
    """Smallest tick >= price (exact decimal reading of `price`)."""
    return math.ceil(_decimal_fraction(price, f"{spec.symbol} price") / spec.tick_size)


def conservative_ticks(price: float, spec: InstrumentSpec, side: Side) -> int:
    """THE away-from-profit rounding rule, applied once at the port: a price
    we BUY at rounds up (pay more), a price we SELL at rounds down (receive
    less). On-grid prices pass through unchanged; the result is never better
    for our side than the input and never a full tick away."""
    if side == "buy":
        return ceil_to_ticks(price, spec)
    return floor_to_ticks(price, spec)


# ---------- money (int cents) ----------


def cents_from_dollars(dollars: float | str) -> int:
    """Exact dollars→cents for config/boundary amounts; PricingError on
    sub-cent amounts (a $0.001 commission is a config mistake, not a value
    to round)."""
    cents = _decimal_fraction(dollars, "dollar amount") * 100
    if cents.denominator != 1:
        raise PricingError(f"dollar amount {dollars!r} is not whole cents")
    return int(cents)


def ticks_pnl_cents(ticks_delta: int, qty: int, spec: InstrumentSpec) -> int:
    """PnL in cents for a signed tick move on a signed/unsigned quantity —
    pure integer arithmetic."""
    return ticks_delta * qty * spec.tick_value_cents
