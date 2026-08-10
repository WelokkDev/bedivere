"""Resting entries: the two rounding rules, the marketable valve, and the
sim's matching of limit / stop / stop-limit entries.

The rounding pair is the load-bearing part. `conservative_ticks` answers "at
what price" and rounds away from profit; the resting rules answer "does it
trigger at all" and round away from the market. They point OPPOSITE ways for
limits, and a test that only pinned one of them would let the other silently
become the first, which is exactly how a resting-entry backtest starts
inventing fills.
"""

from __future__ import annotations

import random

import pytest

from bedivere.brokers.sim import SimBroker, SimBrokerConfig
from bedivere.core.pricing import conservative_ticks, spec_from_handoff
from bedivere.core.types import Timeframe
from bedivere.engine.events import BarEvent
from bedivere.engine.intents import (
    BracketIntent,
    EntryOrder,
    marketable_entry_reason,
    resting_limit_ticks,
    resting_stop_ticks,
    rr_target_ticks,
)
from tests.helpers import bar

NQ = spec_from_handoff("NQ", 0.25, 20)
PERIOD = 300
T0 = 1_780_524_000


def _t(price: float) -> int:
    return round(price / 0.25)


def _cfg(latency_ms: int = 250, spread: int = 1, fee: int = 0) -> SimBrokerConfig:
    return SimBrokerConfig(
        bar_period_seconds=PERIOD,
        latency_ms=latency_ms,
        half_spread_ticks=spread,
        commission_cents_per_side_per_contract=fee,
        seed=0,
        defer_protection_one_bar=False,
    )


def _bar(i: int, o: float, h: float, low: float, c: float) -> BarEvent:
    ts = T0 + i * PERIOD
    return BarEvent(ts=ts, symbol="NQ", timeframe=Timeframe.M5, candle=bar(ts, o, h, low, c, v=1))


def _stamped(intent: BracketIntent) -> BracketIntent:
    intent.stamps.ts_bar_close = T0 * 1000
    intent.stamps.ts_event_received = T0 * 1000
    intent.stamps.ts_decided = T0 * 1000
    return intent


# ---------- the two rounding rules, and their opposition ----------


def test_resting_limit_rounds_away_from_the_market() -> None:
    # A long entry limit is a BUY resting BELOW: away-from-the-market floors.
    assert resting_limit_ticks(20_000.10, NQ, "long") == _t(20_000.00)
    # A short entry limit is a SELL resting ABOVE: it ceils.
    assert resting_limit_ticks(20_000.10, NQ, "short") == _t(20_000.25)
    # On-grid input is untouched either way.
    assert resting_limit_ticks(20_000.25, NQ, "long") == _t(20_000.25)
    assert resting_limit_ticks(20_000.25, NQ, "short") == _t(20_000.25)


def test_resting_stop_mirrors_the_limit_rule() -> None:
    # A long entry stop rests ABOVE the market, so away-from-the-market ceils.
    assert resting_stop_ticks(20_000.10, NQ, "long") == _t(20_000.25)
    # A short entry stop rests BELOW: it floors.
    assert resting_stop_ticks(20_000.10, NQ, "short") == _t(20_000.00)


def test_the_two_rules_point_opposite_ways_for_limits() -> None:
    """Assert the opposition directly rather than
    inferring it from two separate tables."""
    rng = random.Random(20260804)
    for _ in range(400):
        # Deliberately off-grid: on-grid prices agree trivially and prove
        # nothing about which direction each rule rounds.
        price = round(rng.uniform(1_000, 25_000), 2)
        if price * 4 == int(price * 4):
            continue

        # LIMIT: the resting rule is strictly the other side of the grid cell
        # from the away-from-profit rule — exactly one tick apart.
        assert resting_limit_ticks(price, NQ, "long") == conservative_ticks(price, NQ, "buy") - 1
        assert resting_limit_ticks(price, NQ, "short") == conservative_ticks(price, NQ, "sell") + 1

        # STOP: numerically the same answer as away-from-profit for the ENTRY
        # side — a consequence of the geometry, not the reason for the rule.
        assert resting_stop_ticks(price, NQ, "long") == conservative_ticks(price, NQ, "buy")
        assert resting_stop_ticks(price, NQ, "short") == conservative_ticks(price, NQ, "sell")

        # The guarantee both rules exist for: a resting level is never nearer
        # the market than the price asked for, so rounding can only ever make
        # a fill LESS likely.
        assert resting_limit_ticks(price, NQ, "long") * 0.25 < price  # buy limit below
        assert resting_limit_ticks(price, NQ, "short") * 0.25 > price  # sell limit above
        assert resting_stop_ticks(price, NQ, "long") * 0.25 > price  # buy stop above
        assert resting_stop_ticks(price, NQ, "short") * 0.25 < price  # sell stop below


def test_rr_target_ticks_is_the_one_shared_computation() -> None:
    # Long target is a SELL → floors; short target is a BUY → ceils.
    assert rr_target_ticks("long", 1000, 3, 2.5) == 1007  # 1007.5 floored
    assert rr_target_ticks("short", 1000, 3, 2.5) == 993  # 992.5 ceiled
    with pytest.raises(ValueError, match="risk must be positive"):
        rr_target_ticks("long", 1000, 0, 2.0)


# ---------- EntryOrder: the pair cannot be built wrong ----------


def test_entry_order_validates_its_pair() -> None:
    assert EntryOrder().type == "market"
    with pytest.raises(ValueError, match="requires limit_ticks"):
        EntryOrder(type="limit")
    with pytest.raises(ValueError, match="requires stop_ticks"):
        EntryOrder(type="stop")
    with pytest.raises(ValueError, match="requires limit_ticks"):
        EntryOrder(type="stop_limit", stop_ticks=100)  # missing half the pair
    with pytest.raises(ValueError, match="requires stop_ticks"):
        EntryOrder(type="stop_limit", limit_ticks=100)
    with pytest.raises(ValueError, match="must not carry limit_ticks"):
        EntryOrder(type="market", limit_ticks=100)  # a stray level
    with pytest.raises(ValueError, match="must not carry stop_ticks"):
        EntryOrder(type="limit", limit_ticks=100, stop_ticks=100)

    # `resting_ticks` is the side-of-market level: the TRIGGER for a
    # stop-limit, never its cap.
    assert EntryOrder().resting_ticks is None
    assert EntryOrder(type="limit", limit_ticks=7).resting_ticks == 7
    assert EntryOrder(type="stop_limit", stop_ticks=7, limit_ticks=9).resting_ticks == 7


def test_from_prices_refuses_contradictory_entry_kwargs() -> None:
    common = {"direction": "long", "qty": 1, "stop_price": 19_990.0, "target_rr": 2.0,
              "signal_price": 20_000.0}
    with pytest.raises(ValueError, match="mutually exclusive"):
        BracketIntent.from_prices(
            NQ, **common, entry_limit_price=19_995.0, entry_stop_price=20_005.0  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="needs entry_stop_price"):
        BracketIntent.from_prices(NQ, **common, entry_limit_slippage_ticks=2)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be >= 0"):
        BracketIntent.from_prices(
            NQ, **common, entry_stop_price=20_005.0, entry_limit_slippage_ticks=-1  # type: ignore[arg-type]
        )


def test_market_entry_stays_byte_identical_without_entry_kwargs() -> None:
    """Backward compatibility: an intent built the way every existing caller
    builds one is a market entry and behaves exactly as before."""
    intent = BracketIntent.from_prices(
        NQ, direction="long", qty=1, stop_price=19_990.0, target_rr=3.0, signal_price=20_000.0
    )
    assert intent.entry == EntryOrder()
    assert intent.entry.type == "market"
    assert intent.fixed_target_ticks is None
    fill = _t(20_001.0)
    assert intent.target_ticks_for_fill(fill) == fill + 3 * (fill - intent.stop_ticks)


# ---------- the marketable-entry valve: 4 types x 2 directions ----------


def _entry_intent(direction: str, **kwargs: float | int) -> BracketIntent:
    assert direction in ("long", "short")
    return BracketIntent.from_prices(
        NQ,
        direction=direction,  # type: ignore[arg-type]  # narrowed by the assert
        qty=1,
        stop_price=19_000.0 if direction == "long" else 21_000.0,
        target_rr=2.0,
        signal_price=20_000.0,
        **kwargs,  # type: ignore[arg-type]
    )


def test_marketable_valve_covers_every_entry_type_and_direction() -> None:
    market_ticks = _t(20_000.0)

    # MARKET: marketable by design — never a complaint.
    assert marketable_entry_reason(_entry_intent("long"), market_ticks) is None
    assert marketable_entry_reason(_entry_intent("short"), market_ticks) is None

    # LIMIT. A buy limit must rest strictly BELOW; a sell limit strictly ABOVE.
    ok_long = _entry_intent("long", entry_limit_price=19_999.0)
    bad_long = _entry_intent("long", entry_limit_price=20_001.0)
    assert marketable_entry_reason(ok_long, market_ticks) is None
    assert "at/above the market" in (marketable_entry_reason(bad_long, market_ticks) or "")
    ok_short = _entry_intent("short", entry_limit_price=20_001.0)
    bad_short = _entry_intent("short", entry_limit_price=19_999.0)
    assert marketable_entry_reason(ok_short, market_ticks) is None
    assert "at/below the market" in (marketable_entry_reason(bad_short, market_ticks) or "")

    # STOP. Exactly the mirror: a buy stop rests ABOVE, a sell stop BELOW.
    assert marketable_entry_reason(_entry_intent("long", entry_stop_price=20_001.0), market_ticks) is None
    assert "at/below the market" in (
        marketable_entry_reason(_entry_intent("long", entry_stop_price=19_999.0), market_ticks) or ""
    )
    assert marketable_entry_reason(_entry_intent("short", entry_stop_price=19_999.0), market_ticks) is None
    assert "at/above the market" in (
        marketable_entry_reason(_entry_intent("short", entry_stop_price=20_001.0), market_ticks) or ""
    )

    # STOP-LIMIT is judged on its TRIGGER, never its cap — the cap is
    # expected to sit past the trigger and says nothing about which side of
    # the market the order rests on.
    good = _entry_intent("long", entry_stop_price=20_001.0, entry_limit_slippage_ticks=4)
    assert good.entry.limit_ticks == _t(20_001.0) + 4  # cap past the trigger
    assert marketable_entry_reason(good, market_ticks) is None
    bad = _entry_intent("long", entry_stop_price=19_999.0, entry_limit_slippage_ticks=4)
    assert "at/below the market" in (marketable_entry_reason(bad, market_ticks) or "")


def test_at_the_market_counts_as_marketable() -> None:
    """A level the market is already at is not resting — for every resting
    type, both ways."""
    at = _t(20_000.0)
    for kwargs in (
        {"entry_limit_price": 20_000.0},
        {"entry_stop_price": 20_000.0},
        {"entry_stop_price": 20_000.0, "entry_limit_slippage_ticks": 2},
    ):
        assert marketable_entry_reason(_entry_intent("long", **kwargs), at) is not None
        assert marketable_entry_reason(_entry_intent("short", **kwargs), at) is not None


def test_sim_refuses_a_marketable_resting_entry_instead_of_filling_it() -> None:
    broker = SimBroker(NQ, _cfg())
    broker.drain(_bar(1, 20_000.0, 20_000.5, 19_999.5, 20_000.0))  # the venue now has a price

    # A buy limit ABOVE the market: the silent-disaster case. It must be
    # refused, not filled at the worst price in the next bar.
    bid = broker.submit_bracket(_stamped(_entry_intent("long", entry_limit_price=20_010.0)))
    events = broker.drain(_bar(2, 20_001.0, 20_002.0, 19_995.0, 20_000.0))
    assert [e.kind for e in events] == ["cancelled"]
    assert events[0].bracket_id == bid
    assert events[0].reason == "marketable_entry"
    assert "would fill immediately instead of resting" in events[0].detail
    assert broker.marketable_entries_refused == 1

    # A correctly-placed one is untouched by the valve.
    good = broker.submit_bracket(_stamped(_entry_intent("long", entry_limit_price=19_990.0)))
    ok = broker.drain(_bar(3, 20_000.0, 20_001.0, 19_999.0, 20_000.0))
    assert ok == []  # rests, unfilled — no refusal, no fill
    assert broker.marketable_entries_refused == 1
    broker.cancel(good)


def test_valve_is_silent_before_the_venue_has_seen_a_price() -> None:
    """With no reference price, refusing or admitting would both be
    arbitrary — so the first order through is never judged."""
    broker = SimBroker(NQ, _cfg())
    broker.submit_bracket(_stamped(_entry_intent("long", entry_limit_price=25_000.0)))
    assert broker.marketable_entries_refused == 0


# ---------- sim matching: limit / stop / stop-limit ----------


def _rest_long_limit(broker: SimBroker, level: float) -> int:
    broker.drain(_bar(0, 20_000.0, 20_000.0, 20_000.0, 20_000.0))
    return broker.submit_bracket(
        _stamped(
            BracketIntent.from_prices(
                NQ,
                direction="long",
                qty=1,
                entry_limit_price=level,
                stop_price=level - 10.0,
                target_rr=2.0,
                signal_price=20_000.0,
            )
        )
    )


def test_limit_entry_fills_strictly_through_never_at_touch() -> None:
    broker = SimBroker(NQ, _cfg())
    _rest_long_limit(broker, 19_990.0)

    # Touched exactly: NOT a fill. Whether a merely-touched limit would have
    # filled is a queue question OHLC cannot answer.
    assert broker.drain(_bar(1, 19_995.0, 19_996.0, 19_990.0, 19_994.0)) == []
    # Strictly through: fills AT the level, and pays no half-spread — the
    # maker never crosses.
    events = broker.drain(_bar(2, 19_995.0, 19_996.0, 19_989.75, 19_994.0))
    assert [e.kind for e in events] == ["entry_fill", "protection_placed"]
    assert events[0].price_ticks == _t(19_990.0)


def test_limit_entry_that_gaps_through_fills_at_the_better_open() -> None:
    broker = SimBroker(NQ, _cfg())
    _rest_long_limit(broker, 19_990.0)
    events = broker.drain(_bar(1, 19_985.0, 19_986.0, 19_984.0, 19_985.5))
    assert events[0].kind == "entry_fill"
    assert events[0].price_ticks == _t(19_985.0)  # the favorable gap is real


def _rest_long_stop(broker: SimBroker, trigger: float, *, slippage: int | None = None) -> int:
    broker.drain(_bar(0, 20_000.0, 20_000.0, 20_000.0, 20_000.0))
    return broker.submit_bracket(
        _stamped(
            BracketIntent.from_prices(
                NQ,
                direction="long",
                qty=1,
                entry_stop_price=trigger,
                entry_limit_slippage_ticks=slippage,
                stop_price=trigger - 10.0,
                target_rr=2.0,
                signal_price=20_000.0,
                target_price=trigger + 20.0,
            )
        )
    )


def test_stop_entry_triggers_at_touch_and_pays_the_spread() -> None:
    broker = SimBroker(NQ, _cfg(spread=1))
    _rest_long_stop(broker, 20_010.0)

    assert broker.drain(_bar(1, 20_001.0, 20_009.75, 20_000.0, 20_005.0)) == []  # short of it
    events = broker.drain(_bar(2, 20_005.0, 20_010.0, 20_004.0, 20_008.0))  # high == trigger
    assert events[0].kind == "entry_fill"
    # A triggered stop is a TAKER: it fills WORSE than its own trigger.
    assert events[0].price_ticks == _t(20_010.0) + 1
    assert events[0].fill_open_ticks == _t(20_005.0)


def test_stop_entry_that_gaps_through_fills_at_the_open() -> None:
    broker = SimBroker(NQ, _cfg(spread=1))
    _rest_long_stop(broker, 20_010.0)
    events = broker.drain(_bar(1, 20_020.0, 20_021.0, 20_019.0, 20_020.0))
    assert events[0].kind == "entry_fill"
    assert events[0].price_ticks == _t(20_020.0) + 1  # honest about the gap
    assert events[0].fill_open_ticks == _t(20_020.0)


def test_stop_limit_can_trigger_and_still_not_fill() -> None:
    """The whole trade a stop-limit makes: you keep the price, and in
    exchange you can trigger into nothing. A venue that always filled it
    would be modelling a plain stop."""
    broker = SimBroker(NQ, _cfg(spread=1))
    bid = _rest_long_stop(broker, 20_010.0, slippage=2)  # cap at trigger + 2

    # The bar OPENS through the trigger and well past the cap: it triggers,
    # but the taker price it would pay (20_030.00 + spread) is through
    # 20_010.50, so there is no fill. This is the case a plain stop fills and
    # a stop-limit does not.
    assert broker.drain(_bar(1, 20_030.0, 20_031.0, 20_029.0, 20_030.0)) == []

    # It is now a working LIMIT at the cap, matched strictly-through from the
    # next bar. At-touch on the cap is still not a fill.
    assert broker.drain(_bar(2, 20_020.0, 20_021.0, 20_010.5, 20_015.0)) == []
    events = broker.drain(_bar(3, 20_020.0, 20_021.0, 20_010.25, 20_015.0))
    assert [e.kind for e in events] == ["entry_fill", "protection_placed"]
    assert events[0].bracket_id == bid
    # Filled as a MAKER at its cap: no half-spread, which is what the cap
    # bought.
    assert events[0].price_ticks == _t(20_010.5)


def test_stop_limit_fills_at_the_trigger_when_the_market_stays_inside_the_cap() -> None:
    broker = SimBroker(NQ, _cfg(spread=1))
    _rest_long_stop(broker, 20_010.0, slippage=2)
    events = broker.drain(_bar(1, 20_005.0, 20_010.25, 20_004.0, 20_010.0))
    assert events[0].kind == "entry_fill"
    assert events[0].price_ticks == _t(20_010.0) + 1  # inside the cap → taker fill


def test_an_unfilled_resting_entry_rests_until_it_is_cancelled() -> None:
    broker = SimBroker(NQ, _cfg())
    bid = _rest_long_limit(broker, 19_000.0)  # nowhere near
    for i in range(1, 6):
        assert broker.drain(_bar(i, 20_000.0, 20_001.0, 19_999.0, 20_000.0)) == []
    broker.cancel(bid)
    events = broker.drain(_bar(6, 20_000.0, 20_001.0, 19_999.0, 20_000.0))
    assert [(e.kind, e.reason) for e in events] == [("cancelled", "strategy_cancel")]


# ---------- fixed vs fill-derived target ----------


def test_fixed_target_keeps_entry_slippage_in_realised_r() -> None:
    """A stop entry fills past its trigger. With a fill-derived target the
    whole bracket slides up with it and the slippage vanishes; with a fixed
    one the target stays put and the loss shows up where it belongs."""
    trigger, stop = _t(20_010.0), _t(20_000.0)
    slipped_fill = trigger + 4  # filled 4 ticks through the trigger

    sliding = BracketIntent.from_prices(
        NQ, direction="long", qty=1, entry_stop_price=20_010.0, stop_price=20_000.0,
        target_rr=2.0, signal_price=20_000.0,
    )
    fixed = BracketIntent.from_prices(
        NQ, direction="long", qty=1, entry_stop_price=20_010.0, stop_price=20_000.0,
        target_rr=2.0, signal_price=20_000.0, target_price=20_030.0,
    )
    assert fixed.fixed_target_ticks == rr_target_ticks("long", trigger, trigger - stop, 2.0)

    # The sliding target moves with the fill, so the DISTANCE to it is
    # unchanged and the trade still looks like a clean 2R...
    sliding_target = sliding.target_ticks_for_fill(slipped_fill)
    assert sliding_target == slipped_fill + 2 * (slipped_fill - stop)
    assert sliding_target > _t(20_030.0)
    # ...while the fixed target does not move, so the 4 ticks of entry
    # slippage come straight out of realised R.
    assert fixed.target_ticks_for_fill(slipped_fill) == _t(20_030.0)

    reward = _t(20_030.0) - slipped_fill
    risk = fixed.risk_ticks_for_fill(slipped_fill)
    assert reward / risk < 2.0  # the honest number


def test_risk_ticks_for_fill_raises_rather_than_inventing_a_bracket() -> None:
    intent = BracketIntent.from_prices(
        NQ, direction="long", qty=1, stop_price=20_000.0, target_rr=2.0, signal_price=20_000.0
    )
    with pytest.raises(ValueError, match="no positive risk"):
        intent.risk_ticks_for_fill(intent.stop_ticks)
    # A fixed target does not make a zero-risk bracket sane.
    fixed = BracketIntent.from_prices(
        NQ, direction="long", qty=1, stop_price=20_000.0, target_rr=2.0,
        signal_price=20_000.0, target_price=20_050.0,
    )
    with pytest.raises(ValueError, match="no positive risk"):
        fixed.target_ticks_for_fill(fixed.stop_ticks - 4)


def test_entry_through_a_fixed_target_scratches_instead_of_booking_a_win() -> None:
    """A fixed target the entry already gapped past is a limit through the
    market: marketable the instant it is placed. Booking it as a win would
    pay us for a level we never had to wait for."""
    broker = SimBroker(NQ, _cfg(spread=1))
    broker.drain(_bar(0, 20_000.0, 20_000.0, 20_000.0, 20_000.0))
    broker.submit_bracket(
        _stamped(
            BracketIntent.from_prices(
                NQ, direction="long", qty=1, entry_stop_price=20_010.0, stop_price=20_000.0,
                target_rr=2.0, signal_price=20_000.0, target_price=20_015.0,
            )
        )
    )
    events = broker.drain(_bar(1, 20_020.0, 20_021.0, 20_019.0, 20_020.0))
    assert [e.kind for e in events] == ["entry_fill", "target_fill"]
    assert events[0].price_ticks == events[1].price_ticks  # an honest scratch


# ---------- change() on a pending bracket ----------


def test_change_on_a_pending_bracket_sets_what_it_will_arm_on_fill() -> None:
    """The protective legs do not exist yet, so there is nothing to amend —
    the call sets the levels the bracket arms with. Cancel-and-resubmit would
    churn the entry order for a level the venue never saw."""
    broker = SimBroker(NQ, _cfg(spread=1))
    bid = _rest_long_stop(broker, 20_010.0)

    # Re-anchor twice while the entry is still working, exactly as a stop
    # tracking a still-forming extreme would.
    broker.change(bid, stop_ticks=_t(19_998.0), target_ticks=_t(20_034.0))
    broker.change(bid, stop_ticks=_t(19_996.0), target_ticks=_t(20_038.0))
    assert broker.drain(_bar(1, 20_001.0, 20_005.0, 20_000.0, 20_002.0)) == []  # still resting

    events = broker.drain(_bar(2, 20_005.0, 20_011.0, 20_004.0, 20_010.0))
    assert [e.kind for e in events] == ["entry_fill", "protection_placed"]
    # The venue armed the LAST levels it was told, not the submitted ones.
    assert events[1].stop_ticks == _t(19_996.0)
    assert events[1].target_ticks == _t(20_038.0)


def test_change_still_amends_working_legs_on_an_open_bracket() -> None:
    broker = SimBroker(NQ, _cfg(spread=1))
    bid = _rest_long_stop(broker, 20_010.0)
    broker.drain(_bar(1, 20_005.0, 20_011.0, 20_004.0, 20_010.0))  # fills
    broker.change(bid, stop_ticks=_t(20_009.0))
    events = broker.drain(_bar(2, 20_010.0, 20_010.5, 20_009.0, 20_009.5))
    assert [e.kind for e in events] == ["stop_fill"]
    assert events[0].price_ticks == _t(20_009.0)


# ---------- the settlement poll ----------


def test_drain_none_settles_queued_events_without_advancing() -> None:
    """A shutdown flatten happens after the stream is exhausted. Without a
    poll its events have no way into the portfolio and the run's own last
    action goes unrecorded."""
    broker = SimBroker(NQ, _cfg())
    bid = _rest_long_limit(broker, 19_000.0)
    last = broker.drain(_bar(1, 20_000.0, 20_001.0, 19_999.0, 20_000.0))
    assert last == []

    broker.flatten("NQ")  # the stream is over; there is no next bar
    events = broker.drain(None)
    assert [(e.kind, e.reason, e.bracket_id) for e in events] == [("cancelled", "flatten", bid)]
    # Stamped with the last instant the venue actually saw — never a clock read.
    assert events[0].ts == T0 + PERIOD

    # Delivered exactly once, and a poll with nothing queued is empty.
    assert broker.drain(None) == []


def test_drain_none_advances_nothing() -> None:
    """A poll delivers; it must not MATCH. Polling repeatedly may not fill a
    resting entry, move the valve's reference price, or stamp anything."""
    broker = SimBroker(NQ, _cfg())
    bid = _rest_long_limit(broker, 19_990.0)  # validly below the 20_000 market

    for _ in range(3):
        assert broker.drain(None) == []

    # Still pending, and it fills on the next REAL bar exactly as it would
    # have without the polls — no state was advanced by them.
    events = broker.drain(_bar(1, 19_995.0, 19_996.0, 19_989.0, 19_994.0))
    assert [e.kind for e in events] == ["entry_fill", "protection_placed"]
    assert events[0].bracket_id == bid
    assert events[0].price_ticks == _t(19_990.0)
