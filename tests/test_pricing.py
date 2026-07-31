"""bedivere.core.pricing — the integer tick/cent core."""

from __future__ import annotations

import random
from fractions import Fraction

import pytest

from bedivere.core.pricing import (
    InstrumentSpec,
    OffGridPriceError,
    PricingError,
    ceil_to_ticks,
    cents_from_dollars,
    conservative_ticks,
    floor_to_ticks,
    price_to_ticks,
    spec_from_handoff,
    ticks_pnl_cents,
    ticks_to_price,
)

NQ = spec_from_handoff("NQ", 0.25, 20)
MNQ = spec_from_handoff("MNQ", 0.25, 2)


# ---------- spec construction ----------


def test_spec_from_handoff_exact_tick_values() -> None:
    assert NQ.tick_size == Fraction(1, 4)
    assert NQ.tick_value_cents == 500  # 0.25 × $20 × 100
    assert MNQ.tick_value_cents == 50
    es = spec_from_handoff("ES", "0.25", "50")
    assert es.tick_value_cents == 1250
    # A 0.1-tick instrument reads as the exact decimal 1/10, not the double.
    zn_ish = spec_from_handoff("X", 0.1, 1000)
    assert zn_ish.tick_size == Fraction(1, 10)
    assert zn_ish.tick_value_cents == 10_000


def test_spec_rejects_fractional_cent_tick_value() -> None:
    with pytest.raises(PricingError, match="not whole cents"):
        spec_from_handoff("BAD", 0.25, 0.001)  # $0.00025/tick


def test_spec_rejects_nonpositive_numbers() -> None:
    with pytest.raises(PricingError):
        spec_from_handoff("BAD", 0, 20)
    with pytest.raises(PricingError):
        spec_from_handoff("BAD", 0.25, -1)
    with pytest.raises(PricingError):
        InstrumentSpec(symbol="BAD", tick_size=Fraction(1, 4), tick_value_cents=0)


# ---------- price <-> ticks ----------


def test_price_to_ticks_exact_on_grid() -> None:
    assert price_to_ticks(20000.25, NQ) == 80001
    assert price_to_ticks(0.0, NQ) == 0
    assert price_to_ticks(-1.25, NQ) == -5  # spreads can be negative
    assert ticks_to_price(80001, NQ) == 20000.25


def test_price_to_ticks_rejects_off_grid() -> None:
    with pytest.raises(OffGridPriceError, match="tick grid"):
        price_to_ticks(20000.30, NQ)
    with pytest.raises(OffGridPriceError):
        price_to_ticks(20000.125, NQ)


def test_floor_ceil_to_ticks() -> None:
    # 20000.30 sits between ticks 80001 (20000.25) and 80002 (20000.50).
    assert floor_to_ticks(20000.30, NQ) == 80001
    assert ceil_to_ticks(20000.30, NQ) == 80002
    # On-grid input: floor == ceil == exact.
    assert floor_to_ticks(20000.25, NQ) == ceil_to_ticks(20000.25, NQ) == 80001
    # Negative off-grid: floor moves down, ceil moves up.
    assert floor_to_ticks(-0.30, NQ) == -2
    assert ceil_to_ticks(-0.30, NQ) == -1


def test_conservative_rounding_directions() -> None:
    # Buys round UP (pay more), sells round DOWN (receive less).
    assert conservative_ticks(20000.30, NQ, "buy") == 80002
    assert conservative_ticks(20000.30, NQ, "sell") == 80001
    # On-grid prices pass through unchanged for both sides.
    assert conservative_ticks(20000.25, NQ, "buy") == 80001
    assert conservative_ticks(20000.25, NQ, "sell") == 80001


def test_conservative_rounding_property() -> None:
    """Random prices: result is on-grid, never better for our side, and
    strictly less than one tick away."""
    rng = random.Random(20260725)
    grids = [NQ, spec_from_handoff("X", 0.1, 1000), spec_from_handoff("Y", "0.5", "10")]
    for _ in range(500):
        spec = grids[rng.randrange(len(grids))]
        # Random decimal with up to 4 places — mostly off-grid.
        price = round(rng.uniform(-100, 25000), rng.randrange(0, 5))
        buy = conservative_ticks(price, spec, "buy")
        sell = conservative_ticks(price, spec, "sell")
        tick = spec.tick_size
        assert buy - 1 < Fraction(str(price)) / tick <= buy
        assert sell <= Fraction(str(price)) / tick < sell + 1
        assert buy - sell in (0, 1)


# ---------- money ----------


def test_cents_from_dollars_exact() -> None:
    assert cents_from_dollars(2.25) == 225
    assert cents_from_dollars("0.62") == 62
    assert cents_from_dollars(0) == 0
    assert cents_from_dollars(-1.5) == -150


def test_cents_from_dollars_rejects_sub_cent() -> None:
    with pytest.raises(PricingError, match="whole cents"):
        cents_from_dollars(0.001)


def test_ticks_pnl_cents_integer_arithmetic() -> None:
    # Long 2 NQ, +10 ticks: 10 × 2 × $5.00 = $100.00.
    assert ticks_pnl_cents(10, 2, NQ) == 10_000
    # Short expressed as negative delta.
    assert ticks_pnl_cents(-3, 1, NQ) == -1_500
    assert ticks_pnl_cents(7, 3, MNQ) == 1_050


def test_boolean_is_not_a_number() -> None:
    with pytest.raises(PricingError, match="bool"):
        cents_from_dollars(True)  # bool is an int subclass; must be rejected
