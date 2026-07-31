"""BracketIntent — port-side rounding, fill-anchored targets, stamps."""

from __future__ import annotations

import random

import pytest

from bedivere.core.pricing import spec_from_handoff
from bedivere.engine.intents import BracketIntent, TimingStamps

NQ = spec_from_handoff("NQ", 0.25, 20)


def test_from_prices_rounds_away_from_profit_per_leg() -> None:
    # Long stop is a SELL leg: off-grid rounds DOWN (bigger modelled risk).
    long_intent = BracketIntent.from_prices(
        NQ, direction="long", qty=1, stop_price=19_990.60, target_rr=3.0, signal_price=20_000.10
    )
    assert long_intent.stop_ticks == round(19_990.50 / 0.25)
    assert long_intent.signal_price_ticks == round(20_000.00 / 0.25)
    # Short stop is a BUY leg: off-grid rounds UP.
    short_intent = BracketIntent.from_prices(
        NQ, direction="short", qty=1, stop_price=20_010.10, target_rr=3.0, signal_price=20_000.20
    )
    assert short_intent.stop_ticks == round(20_010.25 / 0.25)
    assert short_intent.signal_price_ticks == round(20_000.25 / 0.25)


def test_max_risk_caps_the_stop_but_never_past_the_structure() -> None:
    """Fixed-risk mode: stop = fill − cap, EXCEPT where the structural level
    is closer to the fill, which still wins. MNQ-style numbers: a 120-tick
    cap is 30 points."""
    cap = 120

    # 1) Cap BINDS — the structural stop (19_900, i.e. 404 ticks away) is
    #    further than the cap, so risk is exactly 120 ticks.
    wide = BracketIntent.from_prices(
        NQ,
        direction="long",
        qty=1,
        stop_price=19_900.0,
        target_rr=2.0,
        signal_price=20_000.0,
        max_risk_ticks=cap,
    )
    fill = round(20_001.0 / 0.25)
    assert wide.stop_ticks_for_fill(fill) == fill - cap
    # ...and the target follows the RESOLVED stop, so 1:2 stays exact.
    assert wide.target_ticks_for_fill(fill) == fill + 2 * cap

    # 2) Structure BINDS — the structural level sits 8 ticks away, well
    #    inside the cap, so the stop tightens to it and risk is 8, not 120.
    tight = BracketIntent.from_prices(
        NQ,
        direction="long",
        qty=1,
        stop_price=19_999.0,
        target_rr=2.0,
        signal_price=20_000.0,
        max_risk_ticks=cap,
    )
    assert tight.stop_ticks_for_fill(fill) == tight.stop_ticks
    assert fill - tight.stop_ticks == 8
    assert tight.target_ticks_for_fill(fill) == fill + 2 * 8

    # 3) Short mirrors both directions.
    short_wide = BracketIntent.from_prices(
        NQ,
        direction="short",
        qty=1,
        stop_price=20_100.0,
        target_rr=2.0,
        signal_price=20_000.0,
        max_risk_ticks=cap,
    )
    assert short_wide.stop_ticks_for_fill(fill) == fill + cap
    assert short_wide.target_ticks_for_fill(fill) == fill - 2 * cap
    short_tight = BracketIntent.from_prices(
        NQ,
        direction="short",
        qty=1,
        stop_price=20_001.0,
        target_rr=2.0,
        signal_price=20_000.0,
        max_risk_ticks=cap,
    )
    assert short_tight.stop_ticks_for_fill(fill) == short_tight.stop_ticks

    # 4) Uncapped is untouched — the structural stop stands whatever it costs.
    assert wide.stop_ticks == round(19_900.0 / 0.25)
    uncapped = BracketIntent.from_prices(
        NQ, direction="long", qty=1, stop_price=19_900.0, target_rr=2.0, signal_price=20_000.0
    )
    assert uncapped.stop_ticks_for_fill(fill) == uncapped.stop_ticks

    # 5) A nonsense cap is refused at construction.
    with pytest.raises(ValueError, match="max_risk_ticks"):
        BracketIntent.from_prices(
            NQ,
            direction="long",
            qty=1,
            stop_price=19_900.0,
            target_rr=2.0,
            signal_price=20_000.0,
            max_risk_ticks=0,
        )


def test_target_anchors_on_fill_and_rounds_conservatively() -> None:
    intent = BracketIntent.from_prices(
        NQ, direction="long", qty=1, stop_price=19_990.0, target_rr=3.0, signal_price=20_000.0
    )
    fill = round(20_001.0 / 0.25)
    risk = fill - intent.stop_ticks  # 44 ticks
    assert intent.target_ticks_for_fill(fill) == fill + 3 * risk

    # Fractional rr: long target (a SELL) floors.
    frac = BracketIntent.from_prices(
        NQ, direction="long", qty=1, stop_price=19_999.25, target_rr=2.5, signal_price=20_000.0
    )
    fill2 = round(20_000.0 / 0.25)
    # risk = 3 ticks → raw target = fill + 7.5 → floor to +7.
    assert frac.target_ticks_for_fill(fill2) == fill2 + 7

    # Short mirror: target (a BUY) ceils.
    frac_s = BracketIntent.from_prices(
        NQ, direction="short", qty=1, stop_price=20_000.75, target_rr=2.5, signal_price=20_000.0
    )
    fill3 = round(20_000.0 / 0.25)
    # risk = 3 ticks → raw target = fill − 7.5 → ceil to −7.
    assert frac_s.target_ticks_for_fill(fill3) == fill3 - 7


def test_non_positive_risk_raises() -> None:
    intent = BracketIntent.from_prices(
        NQ, direction="long", qty=1, stop_price=20_000.0, target_rr=3.0, signal_price=20_000.0
    )
    with pytest.raises(ValueError, match="no positive risk"):
        intent.target_ticks_for_fill(intent.stop_ticks)
    with pytest.raises(ValueError, match="no positive risk"):
        intent.target_ticks_for_fill(intent.stop_ticks - 4)


def test_validation_and_stamps() -> None:
    with pytest.raises(ValueError, match="qty"):
        BracketIntent.from_prices(
            NQ, direction="long", qty=0, stop_price=1.0, target_rr=3.0, signal_price=2.0
        )
    with pytest.raises(ValueError, match="target_rr"):
        BracketIntent.from_prices(
            NQ, direction="long", qty=1, stop_price=1.0, target_rr=0, signal_price=2.0
        )
    stamps = TimingStamps()
    assert stamps.to_jsonable() == {
        "ts_bar_close": None,
        "ts_event_received": None,
        "ts_decided": None,
        "ts_submitted": None,
        "ts_acked": None,
        "ts_filled": None,
    }


def test_property_everything_crossing_the_port_is_on_grid() -> None:
    rng = random.Random(20260725)
    for _ in range(300):
        direction = "long" if rng.random() < 0.5 else "short"
        base = round(rng.uniform(1000, 25_000), rng.randrange(0, 4))
        offset = round(rng.uniform(1, 50), rng.randrange(0, 4))
        stop = base - offset if direction == "long" else base + offset
        intent = BracketIntent.from_prices(
            NQ,
            direction=direction,
            qty=1,
            stop_price=stop,
            target_rr=rng.choice([1.0, 1.5, 2.0, 3.0, 3.3]),
            signal_price=base,
        )
        # Ticks are ints by construction — the grid property is structural.
        fill = intent.signal_price_ticks + (4 if direction == "long" else -4)
        target = intent.target_ticks_for_fill(fill)
        assert isinstance(intent.stop_ticks, int)
        assert isinstance(target, int)
        if direction == "long":
            assert target > fill > intent.stop_ticks
        else:
            assert target < fill < intent.stop_ticks
