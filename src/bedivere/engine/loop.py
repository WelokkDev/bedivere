"""The ONE engine loop — environment-agnostic by construction.

    for event in stream:
        clock.advance(event.ts)      # ReplayClock only; LiveClock IS time
        venue = broker.drain(event)  # venue first: fills settle BEFORE the
        portfolio.apply(venue)       # strategy sees the bar that caused them
        view.update(event.candle)    # buckets + observers; closes → sink
        strategy.on_order_event(...) # venue events, then
        strategy.on_bar(ctx)         # the bar — may submit brackets

Warm-up rule: strategy callbacks are suppressed until the tradeable start,
and the WarmupGate HARD-FAILS if that instant arrives with any component
unready (identical rule in backtest and live). Backfill-flagged bars
advance state but never reach the strategy — moot in replay, structural for
a live feed.

The close sink and the journal are the two observability seams. The sink
receives every HTF close (including during warm-up — derived state must
build while the strategy is still suppressed). The journal receives every
venue event automatically and is exposed to the strategy on the context, so
custom fields — scores, feature vectors, decision narratives — land in the
same timeline as the fills they explain.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from bedivere.core.clock import Clock
from bedivere.core.session_days import SessionDays
from bedivere.core.types import Candle
from bedivere.engine.events import BarEvent
from bedivere.engine.intents import BracketIntent, OrderEvent
from bedivere.engine.journal import DecisionJournal
from bedivere.engine.portfolio import Portfolio, TradeRecord
from bedivere.engine.warmup import WarmupGate
from bedivere.view.market_view import HtfClose, MarketView


class CloseSink(Protocol):
    """Cross-TF derived state fed by HTF closes (an indicator registry, a
    setup tracker, ...). Called for every close of every derived TF, in the
    exact order MarketView.update reported them — during warm-up too, so
    derived state is fully built by the first tradeable instant. The return
    value is ignored by the loop."""

    def on_close(self, close: HtfClose) -> object: ...


class PortfolioView(Protocol):
    """The strategy's READ surface of the Portfolio: queries, plus
    `register_intent` — the one sanctioned write.

    Mutating methods are deliberately absent. `apply`, `trades` and the PnL
    fields all exist on the real Portfolio, and a strategy that reached for
    them would be keeping its own books against engine-owned state — the
    exact desynchronisation `exposure_count()` exists to prevent. Typing the
    context with this protocol makes that a typecheck failure instead of a
    review comment.

    It is structural, so `Portfolio` satisfies it without inheriting
    anything, and a test can hand a strategy a stub with no engine at all.
    """

    def register_intent(self, bracket_id: int, intent: BracketIntent) -> None: ...

    def position_count(self) -> int: ...

    def position_qty(self, symbol: str) -> int: ...

    def exposure_count(self) -> int: ...

    def bracket_live(self, bracket_id: int) -> bool: ...

    def intent_for(self, bracket_id: int) -> BracketIntent | None: ...

    def entry_ticks_for(self, bracket_id: int) -> int | None: ...

    def last_trade_for(self, bracket_id: int) -> TradeRecord | None: ...


@dataclass(slots=True)
class RunContext:
    """What a strategy may see. Everything here is engine-owned state the
    strategy QUERIES — it holds none of it."""

    bar: Candle
    event_received_ms: int  # backtest: bar close ms; live: feed receive time
    view: MarketView
    close_sink: CloseSink | None
    portfolio: PortfolioView
    broker: BrokerLike
    clock: Clock
    days: SessionDays
    base_period_seconds: int
    journal: DecisionJournal
    # The HTF bucket completions AT THIS INSTANT, in coarseness-ascending
    # order. The loop already computes these to feed the close sink; PUSHING
    # them is what lets a strategy on a fine base act on a coarse close
    # without polling `len(view.completed(tf))` every single bar, the way
    # examples/sma_cross.py does with its `seen_closes` counter. Empty on the
    # overwhelming majority of base bars.
    closes: tuple[HtfClose, ...] = ()


class BrokerLike(Protocol):
    """Structural slice of bedivere.brokers.port.Broker the loop/strategy
    needs (kept local to avoid a hard import edge engine→brokers).

    `cancel` and `change` are here because a strategy with a WORKING entry
    needs both: it retires an unfilled entry when the setup that placed it
    dies, and re-anchors the stop that entry will arm on fill while the level
    is still forming. Both have always been on the port; the loop's slice was
    simply narrower than the port until a strategy needed them, which meant a
    strategy could not reach them at all."""

    def submit_bracket(self, intent: BracketIntent) -> int: ...

    def cancel(self, bracket_id: int) -> None: ...

    def change(
        self, bracket_id: int, *, stop_ticks: int | None = None, target_ticks: int | None = None
    ) -> None: ...

    def flatten(self, symbol: str) -> None: ...

    def drain(self, event: BarEvent | None) -> list[OrderEvent]: ...


class Strategy(Protocol):
    """The strategy handlers. A strategy module must import nothing
    environment-specific — composition supplies everything via ctx."""

    def on_start(self, ctx: RunContext) -> None: ...

    def on_bar(self, ctx: RunContext) -> None: ...

    def on_order_event(self, ev: OrderEvent) -> None: ...

    def on_stop(self) -> None: ...


@dataclass(frozen=True, slots=True)
class LoopStats:
    bars: int
    suppressed_bars: int  # warm-up/pre-window bars the strategy never saw
    htf_closes: int
    venue_events: int
    first_traded_bar_ts: int | None


def run_loop(
    *,
    stream: Iterable[BarEvent],
    clock: Clock,
    view: MarketView,
    broker: BrokerLike,
    portfolio: Portfolio,
    strategy: Strategy,
    tradeable_start_unix: int,
    close_sink: CloseSink | None = None,
    warmup_gate: WarmupGate | None = None,
    journal: DecisionJournal | None = None,
) -> LoopStats:
    """Drive one run to stream exhaustion. `warmup_gate=None` means no
    declared requirements (nothing to enforce); a passed gate hard-fails at
    the first tradeable instant if unready. Venue events are journaled
    automatically (kind = the event kind, e.g. "entry_fill")."""
    if journal is None:
        journal = DecisionJournal()
    advance = getattr(clock, "advance", None)  # ReplayClock only
    bars = 0
    suppressed = 0
    closes_total = 0
    venue_total = 0
    first_traded: int | None = None
    started = False

    for event in stream:
        bars += 1
        if callable(advance):
            advance(event.ts)

        venue = broker.drain(event)
        venue_total += len(venue)
        portfolio.apply(venue)
        for ev in venue:
            _journal_venue_event(journal, ev)

        closes = view.update(event.candle)
        closes_total += len(closes)
        if close_sink is not None:
            for close in closes:
                close_sink.on_close(close)

        if event.ts < tradeable_start_unix or event.backfill:
            suppressed += 1
            continue

        ctx = RunContext(
            bar=event.candle,
            event_received_ms=(
                event.received_at_ms if event.received_at_ms is not None else event.ts * 1000
            ),
            view=view,
            close_sink=close_sink,
            portfolio=portfolio,
            broker=broker,
            clock=clock,
            days=view.days,
            base_period_seconds=view.base_tf.period_seconds,
            journal=journal,
            closes=tuple(closes),
        )
        if not started:
            if warmup_gate is not None:
                warmup_gate.assert_ready_for_trading(event.ts)
            strategy.on_start(ctx)
            started = True
            first_traded = event.ts
        for ev in venue:
            strategy.on_order_event(ev)
        strategy.on_bar(ctx)

    if started:
        strategy.on_stop()
    return LoopStats(
        bars=bars,
        suppressed_bars=suppressed,
        htf_closes=closes_total,
        venue_events=venue_total,
        first_traded_bar_ts=first_traded,
    )


def _journal_venue_event(journal: DecisionJournal, ev: OrderEvent) -> None:
    """One journal line per venue event — the baseline timeline every run
    gets without strategy effort. Optional fields are omitted when absent,
    keeping lines lean and the JSONL deterministic."""
    fields: dict[str, object] = {
        "bracketId": ev.bracket_id,
        "symbol": ev.symbol,
        "direction": ev.direction,
        "qty": ev.qty,
    }
    if ev.price_ticks is not None:
        fields["priceTicks"] = ev.price_ticks
    if ev.fee_cents:
        fields["feeCents"] = ev.fee_cents
    if ev.ambiguous:
        fields["ambiguous"] = True
    if ev.stop_ticks is not None:
        fields["stopTicks"] = ev.stop_ticks
    if ev.target_ticks is not None:
        fields["targetTicks"] = ev.target_ticks
    journal.emit(ev.kind, ev.ts, **fields)
