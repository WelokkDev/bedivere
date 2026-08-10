"""SimBroker — the matching rules, one by one."""

from __future__ import annotations

import pytest

from bedivere.brokers.sim import SimBroker, SimBrokerConfig
from bedivere.core.pricing import spec_from_handoff
from bedivere.core.types import Timeframe
from bedivere.engine.events import BarEvent
from bedivere.engine.intents import BracketIntent, OrderEvent
from tests.helpers import bar

NQ = spec_from_handoff("NQ", 0.25, 20)
PERIOD = 300
T0 = 1_780_524_000  # aligned reference instant


def _cfg(
    latency_ms: int = 250, spread: int = 1, fee: int = 0, *, defer: bool = False
) -> SimBrokerConfig:
    return SimBrokerConfig(
        bar_period_seconds=PERIOD,
        latency_ms=latency_ms,
        half_spread_ticks=spread,
        commission_cents_per_side_per_contract=fee,
        seed=0,
        defer_protection_one_bar=defer,
    )


def _intent(direction: str = "long", stop: float = 19_990.0, rr: float = 3.0) -> BracketIntent:
    assert direction in ("long", "short")
    intent = BracketIntent.from_prices(
        NQ,
        direction=direction,  # type: ignore[arg-type]  # narrowed by the assert
        qty=1,
        stop_price=stop,
        target_rr=rr,
        signal_price=20_000.0,
    )
    intent.stamps.ts_bar_close = T0 * 1000
    intent.stamps.ts_event_received = T0 * 1000
    intent.stamps.ts_decided = T0 * 1000
    return intent


def _bar(i: int, o: float, h: float, low: float, c: float) -> BarEvent:
    ts = T0 + i * PERIOD
    return BarEvent(ts=ts, symbol="NQ", timeframe=Timeframe.M5, candle=bar(ts, o, h, low, c, v=1))


def _t(price: float) -> int:
    return round(price / 0.25)


def test_market_entry_fills_next_bar_open_plus_spread_and_stamps() -> None:
    broker = SimBroker(NQ, _cfg())
    intent = _intent("long")
    bid = broker.submit_bracket(intent)
    assert intent.stamps.ts_submitted == T0 * 1000
    assert intent.stamps.ts_acked == T0 * 1000 + 250

    events = broker.drain(_bar(1, 20_001.0, 20_005.0, 20_000.0, 20_004.0))
    kinds = [e.kind for e in events]
    assert kinds == ["entry_fill", "protection_placed"]
    fill = events[0]
    assert fill.bracket_id == bid
    assert fill.price_ticks == _t(20_001.25)  # open + 1 tick half-spread
    # Contiguous bars: the bar opens the instant the decision was made, so
    # the fill lands when the order becomes effective (ack), at the open.
    assert intent.stamps.ts_filled == T0 * 1000 + 250

    short = _intent("short", stop=20_010.0)
    broker.submit_bracket(short)
    ev2 = broker.drain(_bar(2, 20_004.0, 20_006.0, 20_002.0, 20_003.0))
    assert ev2[0].kind == "entry_fill"
    assert ev2[0].price_ticks == _t(20_003.75)  # open − spread for the short


def test_latency_longer_than_a_bar_skips_bars() -> None:
    broker = SimBroker(NQ, _cfg(latency_ms=600_000))  # 10 minutes
    broker.submit_bracket(_intent("long"))
    # Bars whose close_ms is NOT strictly after acked don't fill.
    assert broker.drain(_bar(1, 20_001, 20_002, 20_000, 20_001)) == []
    assert broker.drain(_bar(2, 20_002, 20_003, 20_001, 20_002)) == []  # close == acked
    events = broker.drain(_bar(3, 20_003, 20_004, 20_002, 20_003))
    assert [e.kind for e in events] == ["entry_fill", "protection_placed"]
    assert events[0].price_ticks == _t(20_003.25)


def _open_long(broker: SimBroker, stop: float = 19_990.0, rr: float = 3.0) -> tuple[int, int]:
    """Submit + fill a long; returns (bracket_id, fill_ticks)."""
    intent = _intent("long", stop=stop, rr=rr)
    bid = broker.submit_bracket(intent)
    events = broker.drain(_bar(1, 20_001.0, 20_001.5, 20_000.75, 20_001.0))
    assert events[0].kind == "entry_fill"
    fill = events[0].price_ticks
    assert fill is not None
    return bid, fill


def test_stop_triggers_at_touch_fills_at_trigger() -> None:
    broker = SimBroker(NQ, _cfg())
    _open_long(broker)  # stop 19990.0 → 79960 ticks
    events = broker.drain(_bar(2, 19_995.0, 19_996.0, 19_990.0, 19_991.0))  # low == stop
    assert [e.kind for e in events] == ["stop_fill"]
    assert events[0].price_ticks == _t(19_990.0)
    assert events[0].ambiguous is False


def test_stop_gap_through_fills_at_open() -> None:
    broker = SimBroker(NQ, _cfg())
    _open_long(broker)
    events = broker.drain(_bar(2, 19_980.0, 19_985.0, 19_978.0, 19_982.0))  # opens through
    assert [e.kind for e in events] == ["stop_fill"]
    assert events[0].price_ticks == _t(19_980.0)


def test_limit_target_fills_strictly_through_not_at_touch() -> None:
    broker = SimBroker(NQ, _cfg())
    _, fill = _open_long(broker)
    target_ticks = fill + 3 * (fill - _t(19_990.0))
    target_price = target_ticks * 0.25

    # At-touch exactly: NO fill.
    assert broker.drain(_bar(2, 20_002.0, target_price, 20_001.0, 20_002.0)) == []
    # Strictly through: fills AT the limit.
    events = broker.drain(_bar(3, 20_002.0, target_price + 0.25, 20_001.0, 20_003.0))
    assert [e.kind for e in events] == ["target_fill"]
    assert events[0].price_ticks == target_ticks


def test_limit_gap_through_fills_at_the_better_open() -> None:
    broker = SimBroker(NQ, _cfg())
    _, fill = _open_long(broker)
    target_ticks = fill + 3 * (fill - _t(19_990.0))
    open_beyond = (target_ticks + 8) * 0.25
    events = broker.drain(_bar(2, open_beyond, open_beyond + 1, open_beyond - 1, open_beyond))
    assert [e.kind for e in events] == ["target_fill"]
    assert events[0].price_ticks == target_ticks + 8


def test_both_touched_resolves_stop_first_and_flags_ambiguous() -> None:
    broker = SimBroker(NQ, _cfg())
    _, fill = _open_long(broker)
    target_price = (fill + 3 * (fill - _t(19_990.0))) * 0.25
    huge = _bar(2, 20_000.0, target_price + 5, 19_985.0, 20_000.0)  # spans both
    events = broker.drain(huge)
    assert [e.kind for e in events] == ["stop_fill"]
    assert events[0].ambiguous is True
    assert events[0].price_ticks == _t(19_990.0)


def test_same_bar_entry_and_stop() -> None:
    broker = SimBroker(NQ, _cfg())
    broker.submit_bracket(_intent("long", stop=19_998.0))
    # Entry fills at open 20001+spread, then the SAME bar trades down to the stop.
    events = broker.drain(_bar(1, 20_001.0, 20_002.0, 19_997.0, 19_999.0))
    assert [e.kind for e in events] == ["entry_fill", "protection_placed", "stop_fill"]
    assert events[2].price_ticks == _t(19_998.0)


def test_gap_through_stop_entry_scratches_immediately() -> None:
    broker = SimBroker(NQ, _cfg())
    broker.submit_bracket(_intent("long", stop=19_998.0))
    # Next bar opens BELOW the stop: honest scratch at the entry fill.
    events = broker.drain(_bar(1, 19_990.0, 19_992.0, 19_988.0, 19_991.0))
    assert [e.kind for e in events] == ["entry_fill", "stop_fill"]
    assert events[0].price_ticks == events[1].price_ticks == _t(19_990.25)


def test_flatten_cancels_pending_and_exits_open_at_next_open() -> None:
    broker = SimBroker(NQ, _cfg())
    bid, _ = _open_long(broker)
    pending = broker.submit_bracket(_intent("long"))
    broker.flatten("NQ")
    events = broker.drain(_bar(2, 20_005.0, 20_006.0, 20_004.0, 20_005.0))
    assert [e.kind for e in events] == ["flatten_fill", "cancelled"]
    assert events[0].bracket_id == bid
    assert events[0].price_ticks == _t(20_004.75)  # open − spread on the exit sell
    # The swept pending entry reports itself, with the cause attached.
    assert (events[1].bracket_id, events[1].reason) == (pending, "flatten")
    with pytest.raises(ValueError, match="pending"):
        broker.cancel(pending)  # already cancelled by flatten


def test_change_amends_and_misuse_raises() -> None:
    broker = SimBroker(NQ, _cfg())
    bid, fill = _open_long(broker)
    broker.change(bid, stop_ticks=fill - 4)
    events = broker.drain(_bar(2, 20_000.5, 20_000.75, fill * 0.25 - 1.0, 20_000.0))
    assert [e.kind for e in events] == ["stop_fill"]
    assert events[0].price_ticks == fill - 4

    # A cancel is confirmed by an event, not by silently flipping a phase.
    fresh = SimBroker(NQ, _cfg())
    pending = fresh.submit_bracket(_intent("long"))
    fresh.cancel(pending)
    cancelled = fresh.drain(_bar(1, 20_001, 20_002, 20_000, 20_001))
    assert [e.kind for e in cancelled] == ["cancelled"]
    assert cancelled[0].reason == "strategy_cancel"
    # A retired bracket has no legs to amend — working OR future.
    with pytest.raises(ValueError, match="not open or pending"):
        fresh.change(pending, stop_ticks=1)
    with pytest.raises(KeyError):
        fresh.change(999, stop_ticks=1)


def test_fees_charged_per_side_per_contract() -> None:
    broker = SimBroker(NQ, _cfg(fee=62))
    intent = BracketIntent.from_prices(
        NQ, direction="long", qty=3, stop_price=19_998.0, target_rr=3.0, signal_price=20_000.0
    )
    intent.stamps.ts_decided = T0 * 1000
    broker.submit_bracket(intent)
    events = broker.drain(_bar(1, 20_001.0, 20_002.0, 20_000.0, 20_001.0))
    assert events[0].kind == "entry_fill"
    assert events[0].fee_cents == 62 * 3


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        SimBrokerConfig(bar_period_seconds=0, latency_ms=0, half_spread_ticks=0,
                        commission_cents_per_side_per_contract=0, seed=0,
                        defer_protection_one_bar=False)
    with pytest.raises(ValueError):
        SimBrokerConfig(bar_period_seconds=300, latency_ms=-1, half_spread_ticks=0,
                        commission_cents_per_side_per_contract=0, seed=0,
                        defer_protection_one_bar=False)


def test_deterministic_event_order_across_brackets() -> None:
    broker = SimBroker(NQ, _cfg())
    a = broker.submit_bracket(_intent("long", stop=19_990.0))
    b = broker.submit_bracket(_intent("long", stop=19_991.0))
    events: list[OrderEvent] = broker.drain(_bar(1, 20_001.0, 20_001.5, 20_000.75, 20_001.0))
    assert [(e.bracket_id, e.kind) for e in events] == [
        (a, "entry_fill"),
        (a, "protection_placed"),
        (b, "entry_fill"),
        (b, "protection_placed"),
    ]


# ---------- the naked window: defer_protection_one_bar ----------


def test_by_default_protection_is_armed_on_the_entry_bar() -> None:
    """At a coarse base the naked window is sub-bar, so a bar that fills the
    entry AND reaches the stop exits on that same bar."""
    broker = SimBroker(NQ, _cfg())
    broker.submit_bracket(_intent("long", stop=19_990.0))
    broker.drain(_bar(1, 20_000.0, 20_000.0, 19_000.0, 19_500.0))
    kinds = [e.kind for e in broker.drain(_bar(2, 19_500.0, 19_500.0, 19_400.0, 19_450.0))]
    assert "stop_fill" not in kinds, "it should already have exited on the entry bar"
    assert broker.deferred_protection_bars == 0


def test_deferring_moves_that_exit_to_the_next_bar_and_counts_it() -> None:
    """At a fine base the naked window is a real slice of the bar, and a
    stop a few ticks away cannot absorb it. Deferring is a CHOICE, and both
    halves of it are reported: how often it applied, and how much it hid."""
    broker = SimBroker(NQ, _cfg(defer=True))
    broker.submit_bracket(_intent("long", stop=19_990.0))
    entry_bar = [e.kind for e in broker.drain(_bar(1, 20_000.0, 20_000.0, 19_000.0, 19_500.0))]

    assert "entry_fill" in entry_bar
    assert "stop_fill" not in entry_bar, "protection must not be armed on the fill bar"
    assert broker.deferred_protection_bars == 1
    # The bar WOULD have stopped us out, and the counter says so — a silent
    # deferral is an unmeasured assumption.
    assert broker.deferred_protection_suppressed == 1

    later = [e.kind for e in broker.drain(_bar(2, 19_500.0, 19_500.0, 19_400.0, 19_450.0))]
    assert "stop_fill" in later, "protection must be live from the very next bar"


def test_a_deferral_that_hid_nothing_is_counted_separately() -> None:
    """`deferred_protection_bars` counts frequency; `..._suppressed` counts
    consequence. A run where the two differ wildly is telling you the knob
    barely matters for it — which is a finding, not noise."""
    broker = SimBroker(NQ, _cfg(defer=True))
    broker.submit_bracket(_intent("long", stop=19_000.0, rr=50.0))
    broker.drain(_bar(1, 20_000.0, 20_000.5, 19_999.5, 20_000.0))  # neither leg reachable

    assert broker.deferred_protection_bars == 1
    assert broker.deferred_protection_suppressed == 0
