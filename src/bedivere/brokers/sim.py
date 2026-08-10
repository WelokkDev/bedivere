"""SimBroker — deterministic backtest venue, integer core.

Matching per base-TF bar, LEAN-convention, all prices int ticks. Each rule
below applies to BOTH the entry leg that uses it and the protective leg of
the same kind — that is why there are four rules and not eight:

- MARKET entries fill at the open of the first bar whose close-stamp is
  strictly after the order's effective time (`ts_acked` from the latency
  model), at open ± half-spread against the taker.
- LIMITS (a resting entry, and the target leg) fill STRICTLY-THROUGH only —
  at-touch is not a fill — at the limit price; a bar OPENING through fills at
  the open, because that favourable gap is real. Whether a merely-touched
  limit would have filled is a queue-position question OHLC cannot answer,
  and strictly-through is the answer that never invents a trade. A limit is
  the MAKER: no half-spread.
- STOPS (a resting entry trigger, and the protective leg) trigger AT TOUCH
  and fill at the level, unless the bar opens through it → fill at the open
  (gap rule). A stop is a TAKER: it pays the half-spread and can fill WORSE
  than its level, which is why a strategy entering on one must set
  `fixed_target_ticks` rather than derive its target from the fill.
- STOP-LIMITS use the stop rule for WHEN and the limit rule for HOW MUCH. If
  the market ran past the cap the order does NOT fill on the trigger bar; it
  becomes a working limit there. Triggered-and-unfilled is the whole trade a
  stop-limit makes.

Around those: a resting entry on the WRONG side of the market is REFUSED at
submit (`marketable_entry_reason` → `cancelled`, reason "marketable_entry").
The stop resolves at the entry fill and rides back out on
`protection_placed`. A bar where both legs were hittable resolves STOP-FIRST
and stamps the trade `ambiguous`. The naked window between fill and
protection is the latency span — sub-bar at 5m, so protection is live from
the entry bar; `_entry_bar_legs` still suppresses the leg an intrabar fill
cannot prove, and `defer_protection_one_bar` moves both legs to the next bar
for runs where a quarter-bar window is too large to model.

Costs are config, never defaults: latency (submit→ack ms), a flat half-spread
on marketable fills, per-side commission. `seed` is recorded for provenance
but inert: every model here is fully deterministic.

Stamps (unix ms): ts_submitted = ts_decided (modelled zero compute);
ts_acked = ts_submitted + latency; entry ts_filled = the fill bar's open, or
its close when the fill was intrabar (bar resolution cannot place it more
precisely); exit fills are recorded on the trade at bar close.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from bedivere.core.pricing import InstrumentSpec, price_to_ticks
from bedivere.engine.events import BarEvent
from bedivere.engine.intents import (
    BracketIntent,
    CancelReason,
    OrderEvent,
    marketable_entry_reason,
)


@dataclass(frozen=True, slots=True)
class SimBrokerConfig:
    """All fields required on purpose — omitting a cost must be impossible;
    zeros are visible, explicit choices."""

    bar_period_seconds: int  # the base stream's bar period (window math)
    latency_ms: int  # fixed submit→ack milliseconds
    half_spread_ticks: int  # charged against the taker on marketable fills
    commission_cents_per_side_per_contract: int
    seed: int  # recorded in the result for provenance; inert until a probabilistic mode exists
    # Arm protection from the bar AFTER the entry fill, instead of the fill
    # bar itself. REQUIRED, never defaulted, because it moves results in both
    # directions and the right answer depends entirely on bar size.
    #
    # At a 5m base the naked window is the latency span — sub-bar, so
    # protecting from the fill bar is measured honesty. At a 1s base with
    # 250 ms latency it is a QUARTER of the bar, and a strategy whose stop
    # sits a few ticks from its entry cannot absorb that: protecting from the
    # fill bar credits it a stop-out it may well not have gotten to, and
    # deferring credits it a bar of free room it may not have deserved.
    # Neither is free. Deferring is the treatment backtesting.py adopts for
    # exactly this case, and what it suppresses is COUNTED
    # (`deferred_protection_suppressed`) rather than absorbed, so you can see
    # the size of the assumption instead of inheriting it.
    defer_protection_one_bar: bool

    def __post_init__(self) -> None:
        if self.bar_period_seconds < 1:
            raise ValueError("bar_period_seconds must be >= 1")
        if self.latency_ms < 0 or self.half_spread_ticks < 0:
            raise ValueError("latency_ms and half_spread_ticks must be >= 0")
        if self.commission_cents_per_side_per_contract < 0:
            raise ValueError("commission must be >= 0")


_Phase = Literal["pending_entry", "open", "closed", "cancelled"]

# (fill_ticks, filled_at_the_bar's_open, we_crossed_the_spread)
_EntryFill = tuple[int, bool, bool]


@dataclass(slots=True)
class _Bracket:
    bracket_id: int
    intent: BracketIntent
    phase: _Phase
    effective_ms: int  # entry may match bars with close_ms > this
    entry_ticks: int | None = None
    # True when the entry filled at its bar's OPEN (every market entry, and a
    # resting entry the bar gapped through). False means it filled intrabar,
    # where the bar's pre-fill range is not ours to trade against.
    entry_filled_at_open: bool = True
    # True when WE crossed the spread to reach the level (market/stop), False
    # when the market came to US (limit). Decides which protective leg is
    # provable on the fill bar — see `_entry_bar_legs`.
    entry_filled_as_taker: bool = True
    # stop-limit only: the trigger has fired, so the order is now a working
    # limit at its cap. Survives across bars.
    entry_triggered: bool = False
    stop_ticks: int = 0
    target_ticks: int | None = None
    # Set during a strategy callback; the exit matches from the NEXT drained
    # bar (venue-first ordering makes that structural, not a timer).
    flatten_requested: bool = False
    # A cancel reports back exactly once, on the next drain. Silently flipping
    # the phase would leave anything that tracks exposure off the events —
    # the portfolio's pending set among them — permanently wedged.
    cancelled_emitted: bool = False
    cancel_reason: CancelReason = ""
    cancel_detail: str = ""


class SimBroker:
    def __init__(self, spec: InstrumentSpec, config: SimBrokerConfig) -> None:
        self._spec = spec
        self._cfg = config
        self._brackets: dict[int, _Bracket] = {}
        # Submission order — deterministic matching. Finished brackets are
        # pruned after each drain (they can never produce another event), so
        # the per-bar cost tracks LIVE brackets rather than the run's total;
        # at a 1-second base that difference is the whole run time. Lookups
        # still go through `_brackets`, which keeps everything forever.
        self._order: list[int] = []
        self._next_id = 1
        # Freshest price the venue has seen — the marketable-entry valve's
        # reference. None until the first bar: there is nothing to compare
        # against, and inventing a reference would admit or refuse orders
        # arbitrarily.
        self._market_ticks: int | None = None
        # Last bar instant the venue advanced to, so a settlement poll can
        # stamp queued events without reading a clock.
        self._last_ts: int | None = None
        self.marketable_entries_refused = 0
        # Deferred-protection accounting. The first counts entry bars that
        # went unprotected by config; the second counts how many of those the
        # protection WOULD have exited on — the size of the assumption, not
        # merely the frequency of it. Both stay 0 when the knob is off.
        self.deferred_protection_bars = 0
        self.deferred_protection_suppressed = 0

    # ---------- port: preflight ----------

    def preflight(self) -> str | None:
        """Nothing to reconcile: a freshly constructed SimBroker holds no
        position and no working order, and there is no venue that could
        disagree. A REAL adapter has real work to do here — see
        `bedivere.brokers.port.Broker.preflight`."""
        return None

    # ---------- port: submit / change / cancel / flatten ----------

    def submit_bracket(self, intent: BracketIntent) -> int:
        st = intent.stamps
        if st.ts_decided is None:
            raise ValueError("submit_bracket: intent.stamps.ts_decided must be set")
        st.ts_submitted = st.ts_decided  # modelled zero compute time
        st.ts_acked = st.ts_submitted + self._cfg.latency_ms
        bracket_id = self._next_id
        self._next_id += 1
        b = _Bracket(
            bracket_id=bracket_id,
            intent=intent,
            phase="pending_entry",
            effective_ms=st.ts_acked,
            stop_ticks=intent.stop_ticks,
        )
        # The valve. A refusal is a COMMAND outcome like any other: the
        # bracket goes straight to cancelled and reports itself on the next
        # drain, so anything tracking exposure clears normally.
        detail = (
            None
            if self._market_ticks is None
            else marketable_entry_reason(intent, self._market_ticks)
        )
        if detail is not None:
            b.phase = "cancelled"
            b.cancel_reason = "marketable_entry"
            b.cancel_detail = detail
            self.marketable_entries_refused += 1
        self._brackets[bracket_id] = b
        self._order.append(bracket_id)
        return bracket_id

    def change(
        self, bracket_id: int, *, stop_ticks: int | None = None, target_ticks: int | None = None
    ) -> None:
        """Amend the protective legs.

        Open bracket: the legs are working orders, so this is a real amend.

        PENDING bracket: the legs do not exist yet — there is nothing working
        to amend — so this sets the levels the bracket WILL arm on fill. A
        real venue behaves the same way (there is nothing to send), and a
        strategy whose stop tracks a still-forming extreme needs exactly that.
        Cancel-and-resubmit would churn the entry order for a level the venue
        has never seen.
        """
        b = self._require(bracket_id)
        if b.phase == "pending_entry":
            if stop_ticks is not None:
                b.intent.stop_ticks = stop_ticks
                b.stop_ticks = stop_ticks
            if target_ticks is not None:
                b.intent.fixed_target_ticks = target_ticks
            return
        if b.phase != "open":
            raise ValueError(f"change: bracket {bracket_id} is {b.phase}, not open or pending")
        if stop_ticks is not None:
            b.stop_ticks = stop_ticks
        if target_ticks is not None:
            b.target_ticks = target_ticks

    def cancel(self, bracket_id: int) -> None:
        b = self._require(bracket_id)
        if b.phase != "pending_entry":
            raise ValueError(
                f"cancel: bracket {bracket_id} is {b.phase} — only pending entries cancel; use flatten for open positions"
            )
        b.phase = "cancelled"
        b.cancel_reason = "strategy_cancel"

    def flatten(self, symbol: str) -> None:
        """Cancel pending entries and market-out open positions for the
        symbol; the exit matches from the next drained bar onward."""
        for bracket_id in self._order:
            b = self._brackets[bracket_id]
            if b.intent.symbol != symbol:
                continue
            if b.phase == "pending_entry":
                b.phase = "cancelled"
                b.cancel_reason = "flatten"
            elif b.phase == "open" and not b.flatten_requested:
                b.flatten_requested = True

    # ---------- port: drain ----------

    def drain(self, event: BarEvent | None) -> list[OrderEvent]:
        """Advance the venue with one bar, or settle without one.

        `event=None` is the SETTLEMENT POLL: deliver what is already queued,
        advance nothing, stamp nothing. For this broker "queued" means the
        `cancelled` confirmations — everything else it knows how to settle
        needs a price, and it has no new one. Queued events carry the last
        bar instant the venue saw, never a clock read.
        """
        if event is None:
            return self._settle_queued(None)

        out: list[OrderEvent] = []
        bar = event.candle
        close_ms = bar.timestamp * 1000
        open_ms = (bar.timestamp - self._cfg.bar_period_seconds) * 1000

        for bracket_id in self._order:
            b = self._brackets[bracket_id]
            if b.phase == "pending_entry":
                if close_ms > b.effective_ms:
                    opened = self._fill_entry(b, event, open_ms, out)
                    # A just-opened bracket is immediately exposed to this
                    # bar's range (post-open naked window is sub-bar), with
                    # the caveat `_entry_bar_legs` describes — unless the run
                    # declared that window too large to model, in which case
                    # protection arms from the NEXT bar and what that
                    # suppressed is counted rather than absorbed.
                    if opened and self._cfg.defer_protection_one_bar:
                        self.deferred_protection_bars += 1
                        allow_stop, allow_target = self._entry_bar_legs(b)
                        if self._protection_would_fire(
                            b, event, allow_stop=allow_stop, allow_target=allow_target
                        ):
                            self.deferred_protection_suppressed += 1
                    elif opened:
                        allow_stop, allow_target = self._entry_bar_legs(b)
                        self._match_protection(
                            b, event, out, allow_stop=allow_stop, allow_target=allow_target
                        )
            elif b.phase == "open":
                if b.flatten_requested:
                    self._fill_flatten(b, event, open_ms, out)
                else:
                    self._match_protection(b, event, out)
            elif b.phase == "cancelled" and not b.cancelled_emitted:
                out.append(self._cancelled_event(b, event.ts))

        # The valve's reference for orders submitted from THIS bar's
        # callbacks: the last price the venue has actually seen. Set AFTER
        # matching, so a bracket submitted during the previous bar was judged
        # against the previous close — exactly the information the strategy
        # had when it decided.
        self._market_ticks = self._to_ticks(bar.close)
        self._last_ts = bar.timestamp
        self._prune()
        return out

    def _settle_queued(self, ts: int | None) -> list[OrderEvent]:
        """Deliver queued `cancelled` confirmations without advancing. `ts`
        is the bar instant when there is one; without a bar each event falls
        back to the venue's last seen instant, then to the order's own ack —
        both real instants the venue already knew."""
        out: list[OrderEvent] = []
        for bracket_id in self._order:
            b = self._brackets[bracket_id]
            if b.phase == "cancelled" and not b.cancelled_emitted:
                stamp = ts if ts is not None else self._settlement_ts(b)
                out.append(self._cancelled_event(b, stamp))
        self._prune()
        return out

    def _settlement_ts(self, b: _Bracket) -> int:
        return self._last_ts if self._last_ts is not None else b.effective_ms // 1000

    def _cancelled_event(self, b: _Bracket, ts: int) -> OrderEvent:
        b.cancelled_emitted = True
        return OrderEvent(
            ts=ts,
            kind="cancelled",
            bracket_id=b.bracket_id,
            symbol=b.intent.symbol,
            direction=b.intent.direction,
            qty=b.intent.qty,
            reason=b.cancel_reason,
            detail=b.cancel_detail,
        )

    def _prune(self) -> None:
        """Drop brackets that can never produce another event. Pure
        bookkeeping: the survivors keep their relative submission order, so
        matching stays deterministic."""
        self._order = [
            bid
            for bid in self._order
            if not (
                (b := self._brackets[bid]).phase == "closed"
                or (b.phase == "cancelled" and b.cancelled_emitted)
            )
        ]

    # ---------- internals ----------

    def _require(self, bracket_id: int) -> _Bracket:
        b = self._brackets.get(bracket_id)
        if b is None:
            raise KeyError(f"unknown bracket id {bracket_id}")
        return b

    def _fee(self, qty: int) -> int:
        return self._cfg.commission_cents_per_side_per_contract * qty

    def _entry_match(self, b: _Bracket, event: BarEvent) -> _EntryFill | None:
        """`(fill_ticks, at_open, taker)` for this bar, or None if the entry
        does not fill on it. The four matching rules are in the module
        docstring; what is specific to this function:

        It MUTATES `b.entry_triggered` — a stop-limit's trigger is part of the
        test and survives the bar whether or not a fill followed.

        A stop-limit's trigger bar cannot ALSO be a limit fill. OHLC does not
        order a bar's own extremes, so "price ran past my cap and then came
        back inside it" is the both-touched ambiguity, and waiting is the
        answer that never invents a trade.

        `at_open` is reported rather than inferred from the price: a bar can
        open EXACTLY at a resting level and only cross later, and that fill
        lands at the level intrabar, not at the open instant.

        `taker` says whether WE crossed the spread or the market came to US.
        It decides the half-spread and — via `_entry_bar_legs` — which
        protective leg is provable on the fill bar.
        """
        intent = b.intent
        entry = intent.entry
        long = intent.direction == "long"
        spread = self._cfg.half_spread_ticks
        open_ticks = self._to_ticks(event.candle.open)

        if entry.is_market:
            return ((open_ticks + spread if long else open_ticks - spread), True, True)

        if entry.is_limit:
            assert entry.limit_ticks is not None  # EntryOrder validates the pair
            return self._limit_match(entry.limit_ticks, long, open_ticks, event)

        assert entry.stop_ticks is not None  # EntryOrder validates the pair
        trigger = entry.stop_ticks
        cap = entry.limit_ticks  # None for a plain stop

        if b.entry_triggered:
            # A stop-limit that triggered on an earlier bar is now nothing but
            # a resting limit at its cap.
            assert cap is not None
            return self._limit_match(cap, long, open_ticks, event)

        if long:
            # A buy stop rests ABOVE: it triggers the moment price touches it.
            if open_ticks >= trigger:
                b.entry_triggered = True
                taker_price = open_ticks + spread  # opened through — at the open
                at_open = True
            elif self._to_ticks(event.candle.high) >= trigger:
                b.entry_triggered = True
                taker_price = trigger + spread
                at_open = False
            else:
                return None
            if cap is None or taker_price <= cap:
                return (taker_price, at_open, True)
            return None  # triggered, but through the cap — rests at the cap now
        # A sell stop rests BELOW.
        if open_ticks <= trigger:
            b.entry_triggered = True
            taker_price = open_ticks - spread
            at_open = True
        elif self._to_ticks(event.candle.low) <= trigger:
            b.entry_triggered = True
            taker_price = trigger - spread
            at_open = False
        else:
            return None
        if cap is None or taker_price >= cap:
            return (taker_price, at_open, True)
        return None

    def _limit_match(
        self, limit: int, long: bool, open_ticks: int, event: BarEvent
    ) -> _EntryFill | None:
        """THE resting-limit rule, shared by a plain limit entry and by a
        stop-limit that has already triggered: strictly-through only, at the
        level, or at the open when the bar gapped through it. The maker never
        crosses, so no half-spread either way."""
        if long:
            # A buy limit rests BELOW: price must trade strictly under it.
            if open_ticks < limit:
                return (open_ticks, True, False)  # gapped through — better than asked
            return (limit, False, False) if self._to_ticks(event.candle.low) < limit else None
        # A sell limit rests ABOVE: price must trade strictly over it.
        if open_ticks > limit:
            return (open_ticks, True, False)
        return (limit, False, False) if self._to_ticks(event.candle.high) > limit else None

    def _entry_bar_legs(self, b: _Bracket) -> tuple[bool, bool]:
        """`(allow_stop, allow_target)` for the bar an entry filled on.

        A fill AT THE OPEN (every market entry, and a resting entry the bar
        gapped through) happens at the bar's first instant, so the whole range
        played out while we were in it: both legs count.

        An INTRABAR fill splits the bar, and which leg is provable turns on
        WHO CROSSED — which is why the flag is `taker`, not the entry type:

        - MAKER fill, price came DOWN to us (a limit, or a triggered
          stop-limit resting at its cap): the bar opened at/above our level,
          so price must cross it — our fill — before it can reach a stop
          BELOW. Any stop touch is necessarily after entry, so the stop
          counts. The high may well have printed before the dip that filled
          us, so the target does not.
        - TAKER fill, we crossed UP to the level (a stop, or a stop-limit
          filling at its trigger): exactly the mirror. The bar opened below
          the trigger, so price must cross it before reaching a target ABOVE —
          the target counts. The low may have printed before price came back
          up through the trigger, so the stop does not.

        Longs described; shorts mirror throughout, and the rule holds because
        "taker" already means "we moved toward the market", direction-free.
        """
        if b.entry_filled_at_open:
            return True, True
        if b.entry_filled_as_taker:
            return False, True
        return True, False

    def _fill_entry(self, b: _Bracket, event: BarEvent, open_ms: int, out: list[OrderEvent]) -> bool:
        """Fill the entry if this bar fills it. Returns True when the bracket
        is OPEN afterwards (False = no fill yet, or a degenerate instant
        scratch); an unfilled resting entry simply stays `pending_entry` and
        is re-offered the next bar."""
        intent = b.intent
        open_ticks = self._to_ticks(event.candle.open)
        matched = self._entry_match(b, event)
        if matched is None:
            return False  # resting entry untriggered — nothing happened
        fill, at_open, taker = matched
        b.entry_ticks = fill
        b.entry_filled_at_open = at_open
        b.entry_filled_as_taker = taker
        # Fill instant: the bar's open, but never before the order became
        # effective — contiguous bars make that ack+0 (the market order
        # executes the moment it arrives, at ~the open price).
        #
        # A resting entry that filled INTRABAR is the exception: all we know
        # is that price crossed the level somewhere inside the bar, so it is
        # stamped at the close — the same "intrabar timing is unknowable at
        # bar resolution" rule the exit legs already follow. One that gapped
        # through at the open keeps the open instant, which is exact.
        instant = open_ms if at_open else event.candle.timestamp * 1000
        acked = intent.stamps.ts_acked
        intent.stamps.ts_filled = instant if acked is None else max(instant, acked)
        out.append(
            OrderEvent(
                ts=event.ts,
                kind="entry_fill",
                bracket_id=b.bracket_id,
                symbol=intent.symbol,
                direction=intent.direction,
                qty=intent.qty,
                price_ticks=fill,
                fee_cents=self._fee(intent.qty),
                # The slippage baseline: a market entry fills at THIS open ±
                # spread, so the sim's measured slip is the model itself.
                fill_open_ticks=open_ticks,
            )
        )

        # The stop is fill-derived whenever the intent carries a risk cap, so
        # it resolves HERE — before the gap check reads it and before the
        # target is computed from it.
        b.stop_ticks = intent.stop_ticks_for_fill(fill)

        # Gap-through-stop entry: the fill landed at/through the stop level,
        # so protection triggers the instant it exists — model the honest
        # scratch (exit at the same fill) instead of inventing a target.
        gapped = fill <= b.stop_ticks if intent.direction == "long" else fill >= b.stop_ticks
        if gapped:
            b.phase = "closed"
            out.append(
                OrderEvent(
                    ts=event.ts,
                    kind="stop_fill",
                    bracket_id=b.bracket_id,
                    symbol=intent.symbol,
                    direction=intent.direction,
                    qty=intent.qty,
                    price_ticks=fill,
                    fee_cents=self._fee(intent.qty),
                )
            )
            return False

        b.target_ticks = intent.target_ticks_for_fill(fill)

        # The mirror case, reachable only with a FIXED target: the entry
        # filled at or beyond a target set at submit time. That target leg is
        # a limit already through the market, marketable the instant it is
        # placed — so the honest model is the same instant scratch, not a win
        # booked at a level we never had to wait for.
        overshot = (
            fill >= b.target_ticks if intent.direction == "long" else fill <= b.target_ticks
        )
        if overshot:
            b.phase = "closed"
            out.append(
                OrderEvent(
                    ts=event.ts,
                    kind="target_fill",
                    bracket_id=b.bracket_id,
                    symbol=intent.symbol,
                    direction=intent.direction,
                    qty=intent.qty,
                    price_ticks=fill,
                    fee_cents=self._fee(intent.qty),
                )
            )
            return False

        b.phase = "open"
        out.append(
            OrderEvent(
                ts=event.ts,
                kind="protection_placed",
                bracket_id=b.bracket_id,
                symbol=intent.symbol,
                direction=intent.direction,
                qty=intent.qty,
                stop_ticks=b.stop_ticks,
                target_ticks=b.target_ticks,
            )
        )
        return True

    def _fill_flatten(self, b: _Bracket, event: BarEvent, open_ms: int, out: list[OrderEvent]) -> None:
        intent = b.intent
        open_ticks = self._to_ticks(event.candle.open)
        spread = self._cfg.half_spread_ticks
        # Exiting: long sells (worse = lower), short buys (worse = higher).
        fill = open_ticks - spread if intent.direction == "long" else open_ticks + spread
        b.phase = "closed"
        out.append(
            OrderEvent(
                ts=event.ts,
                kind="flatten_fill",
                bracket_id=b.bracket_id,
                symbol=intent.symbol,
                direction=intent.direction,
                qty=intent.qty,
                price_ticks=fill,
                fee_cents=self._fee(intent.qty),
            )
        )

    def _match_protection(
        self,
        b: _Bracket,
        event: BarEvent,
        out: list[OrderEvent],
        *,
        allow_stop: bool = True,
        allow_target: bool = True,
    ) -> None:
        """Match the protective legs against this bar and emit the exit.

        `allow_stop` / `allow_target` suppress a leg for ONE bar — the bar an
        intrabar entry filled on, where part of the range happened before the
        fill and the two legs are not symmetric about it. `_entry_bar_legs`
        derives which one survives.
        """
        intent = b.intent
        stop_hit, target_hit, stop_fill, target_fill = self._protection_hits(
            b, event, allow_stop=allow_stop, allow_target=allow_target
        )
        if not stop_hit and not target_hit:
            return
        ambiguous = stop_hit and target_hit
        kind: Literal["stop_fill", "target_fill"] = (
            "stop_fill" if stop_hit else "target_fill"  # stop-first on ambiguity
        )
        fill = stop_fill if stop_hit else target_fill
        b.phase = "closed"
        out.append(
            OrderEvent(
                ts=event.ts,
                kind=kind,
                bracket_id=b.bracket_id,
                symbol=intent.symbol,
                direction=intent.direction,
                qty=intent.qty,
                price_ticks=fill,
                fee_cents=self._fee(intent.qty),
                ambiguous=ambiguous,
            )
        )

    def _protection_hits(
        self,
        b: _Bracket,
        event: BarEvent,
        *,
        allow_stop: bool,
        allow_target: bool,
    ) -> tuple[bool, bool, int, int]:
        """`(stop_hit, target_hit, stop_fill, target_fill)` for one bar.

        Extracted so the matcher and the deferred-protection probe cannot
        drift: "would this have filled?" and "fill it" must be the same
        question asked twice, or the counter that measures the deferral
        measures something else.
        """
        intent = b.intent
        assert b.target_ticks is not None
        bar = event.candle
        open_t = self._to_ticks(bar.open)
        high_t = self._to_ticks(bar.high)
        low_t = self._to_ticks(bar.low)
        stop = b.stop_ticks
        target = b.target_ticks

        if intent.direction == "long":
            stop_hit = low_t <= stop
            stop_fill = open_t if open_t <= stop else stop  # gap rule
            target_hit = open_t > target or high_t > target  # strictly-through
            target_fill = open_t if open_t > target else target
        else:
            stop_hit = high_t >= stop
            stop_fill = open_t if open_t >= stop else stop
            target_hit = open_t < target or low_t < target
            target_fill = open_t if open_t < target else target

        return stop_hit and allow_stop, target_hit and allow_target, stop_fill, target_fill

    def _protection_would_fire(
        self, b: _Bracket, event: BarEvent, *, allow_stop: bool, allow_target: bool
    ) -> bool:
        """Would protection have exited on this bar, had it been armed? The
        measurement behind `deferred_protection_suppressed` — the size of the
        assumption `defer_protection_one_bar` is making, per run."""
        stop_hit, target_hit, _, _ = self._protection_hits(
            b, event, allow_stop=allow_stop, allow_target=allow_target
        )
        return stop_hit or target_hit

    def _to_ticks(self, price: float) -> int:
        """Bar prices are grid-truth by contract — off-grid means the
        instrument spec is wrong for this data, and that must be loud."""
        return price_to_ticks(price, self._spec)
