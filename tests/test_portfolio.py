"""Portfolio — engine-owned state from venue events, integer money."""

from __future__ import annotations

import pytest

from bedivere.core.pricing import spec_from_handoff
from bedivere.engine.intents import BracketIntent, OrderEvent
from bedivere.engine.portfolio import Portfolio

NQ = spec_from_handoff("NQ", 0.25, 20)


def _intent(direction: str = "long") -> BracketIntent:
    assert direction in ("long", "short")
    return BracketIntent.from_prices(
        NQ,
        direction=direction,  # type: ignore[arg-type]  # narrowed by the assert
        qty=2,
        stop_price=19_990.0 if direction == "long" else 20_010.0,
        target_rr=3.0,
        signal_price=20_000.0,
        tag="NQ-30m-setup-123",
    )


def _fill(kind: str, bid: int, price_ticks: int, *, direction: str = "long", ts: int = 100,
          fee: int = 0, ambiguous: bool = False) -> OrderEvent:
    assert kind in ("entry_acked", "entry_fill", "protection_placed", "stop_fill",
                    "target_fill", "flatten_fill", "cancelled")
    assert direction in ("long", "short")
    return OrderEvent(
        ts=ts,
        kind=kind,  # type: ignore[arg-type]  # narrowed by the assert
        bracket_id=bid,
        symbol="NQ",
        direction=direction,  # type: ignore[arg-type]
        qty=2,
        price_ticks=price_ticks,
        fee_cents=fee,
        ambiguous=ambiguous,
    )


def test_round_trip_long_win() -> None:
    p = Portfolio(spec=NQ)
    p.register_intent(1, _intent("long"))
    p.apply([_fill("entry_fill", 1, 80_005, fee=124)])
    assert p.position_count() == 1
    assert p.position_qty("NQ") == 2
    assert p.position_qty("ES") == 0

    p.apply([_fill("target_fill", 1, 80_137, ts=200, fee=124)])
    assert p.position_count() == 0
    assert len(p.trades) == 1
    t = p.trades[0]
    assert (t.entry_ticks, t.exit_ticks) == (80_005, 80_137)
    assert t.pnl_cents == (80_137 - 80_005) * 2 * 500  # +132 ticks × 2 × $5
    assert t.fees_cents == 248
    assert t.tag == "NQ-30m-setup-123"
    assert t.stamps["ts_bar_close"] is None  # stamps mirror the intent
    j = t.to_jsonable(NQ)
    assert j["entryPrice"] == 20_001.25
    assert j["netCents"] == t.pnl_cents - 248
    assert j["signalTicks"] == 80_000


def test_short_loss_and_ambiguous_counter() -> None:
    p = Portfolio(spec=NQ)
    p.register_intent(7, _intent("short"))
    p.apply([_fill("entry_fill", 7, 80_000, direction="short")])
    p.apply([_fill("stop_fill", 7, 80_040, direction="short", ts=300, ambiguous=True)])
    t = p.trades[0]
    assert t.pnl_cents == (80_000 - 80_040) * 2 * 500  # −40 ticks short
    assert t.ambiguous is True
    assert p.ambiguous_fills == 1
    assert p.summary_jsonable()["losses"] == 1


def test_informational_events_are_no_ops() -> None:
    p = Portfolio(spec=NQ)
    p.apply([_fill("entry_acked", 1, 0), _fill("protection_placed", 1, 0), _fill("cancelled", 1, 0)])
    assert p.position_count() == 0
    assert p.trades == []


def test_duplicate_and_unknown_events_raise() -> None:
    p = Portfolio(spec=NQ)
    p.register_intent(1, _intent())
    p.apply([_fill("entry_fill", 1, 80_005)])
    with pytest.raises(ValueError, match="duplicate entry_fill"):
        p.apply([_fill("entry_fill", 1, 80_006)])
    with pytest.raises(ValueError, match="unknown bracket"):
        p.apply([_fill("stop_fill", 99, 80_000)])


def test_summary_shape() -> None:
    p = Portfolio(spec=NQ)
    s = p.summary_jsonable()
    assert s == {
        "trades": 0,
        "wins": 0,
        "losses": 0,
        "flat": 0,
        "pnlCents": 0,
        "feesCents": 0,
        "netCents": 0,
        "ambiguousFills": 0,
        "openPositions": 0,
    }
