"""SimBroker — deterministic backtest venue, integer core.

Matching per base-TF bar, LEAN-convention, all prices int ticks:

- MARKET entries fill at the open of the first bar whose close-stamp is
  strictly after the order's effective time (`ts_acked` from the latency
  model — "orders only match data with ts > ts_submitted_effective"), at
  open ± half-spread against the taker.
- The STOP resolves at that fill (`stop_ticks_for_fill`) — with a risk cap
  on the intent it is fill-derived, exactly like the target — and both
  resolved levels ride back out on the `protection_placed` event.
- STOPS trigger at touch and fill at the trigger, UNLESS the bar opens
  through the level → fill at open (gap rule; honest about overnight gaps).
- LIMITS (the target leg) fill strictly-through only — at-touch is not a
  fill — at the limit price; a bar OPENING through the limit fills at the
  open (the favorable gap is real).
- BOTH-TOUCHED ambiguity: stop and target both hittable in one bar →
  resolve STOP-FIRST (the pessimistic reading), increment
  `ambiguous_fills`, stamp the trade ambiguous.
- The naked window between entry fill and protection is the latency span;
  at 5m bars it is sub-bar, so protection is active from the entry bar
  itself — measured honesty over sub-bar fiction.

LatencyModel: fixed submit→ack milliseconds. SpreadModel: flat half-spread
ticks charged on marketable fills (entries + flattens; stops fill at their
level per the rule above). FeeModel: per-side cents per contract. None of
the three is omittable — a zero must be explicit config (`SimBrokerConfig`
has no defaults). `seed` is recorded but inert: v1 models are fully
deterministic.

Stamps (unix ms): ts_submitted = ts_decided (modelled zero compute);
ts_acked = ts_submitted + latency; entry ts_filled = the fill bar's open
instant; exit fills are recorded on the trade at bar close (intrabar
timing is unknowable at bar resolution).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from bedivere.core.pricing import InstrumentSpec, price_to_ticks
from bedivere.engine.events import BarEvent
from bedivere.engine.intents import BracketIntent, OrderEvent


@dataclass(frozen=True, slots=True)
class SimBrokerConfig:
    """All fields required on purpose — omitting a cost must be impossible;
    zeros are visible, explicit choices."""

    bar_period_seconds: int  # the base stream's bar period (window math)
    latency_ms: int  # fixed submit→ack milliseconds
    half_spread_ticks: int  # charged against the taker on marketable fills
    commission_cents_per_side_per_contract: int
    seed: int  # recorded in the result for provenance; inert until a probabilistic mode exists

    def __post_init__(self) -> None:
        if self.bar_period_seconds < 1:
            raise ValueError("bar_period_seconds must be >= 1")
        if self.latency_ms < 0 or self.half_spread_ticks < 0:
            raise ValueError("latency_ms and half_spread_ticks must be >= 0")
        if self.commission_cents_per_side_per_contract < 0:
            raise ValueError("commission must be >= 0")


_Phase = Literal["pending_entry", "open", "closed", "cancelled"]


@dataclass(slots=True)
class _Bracket:
    bracket_id: int
    intent: BracketIntent
    phase: _Phase
    effective_ms: int  # entry may match bars with close_ms > this
    entry_ticks: int | None = None
    stop_ticks: int = 0
    target_ticks: int | None = None
    # Set during a strategy callback; the exit matches from the NEXT drained
    # bar (venue-first ordering makes that structural, not a timer).
    flatten_requested: bool = False


class SimBroker:
    def __init__(self, spec: InstrumentSpec, config: SimBrokerConfig) -> None:
        self._spec = spec
        self._cfg = config
        self._brackets: dict[int, _Bracket] = {}
        self._order: list[int] = []  # submission order — deterministic matching
        self._next_id = 1

    # ---------- port: submit / change / cancel / flatten ----------

    def submit_bracket(self, intent: BracketIntent) -> int:
        st = intent.stamps
        if st.ts_decided is None:
            raise ValueError("submit_bracket: intent.stamps.ts_decided must be set")
        st.ts_submitted = st.ts_decided  # modelled zero compute time
        st.ts_acked = st.ts_submitted + self._cfg.latency_ms
        bracket_id = self._next_id
        self._next_id += 1
        self._brackets[bracket_id] = _Bracket(
            bracket_id=bracket_id,
            intent=intent,
            phase="pending_entry",
            effective_ms=st.ts_acked,
            stop_ticks=intent.stop_ticks,
        )
        self._order.append(bracket_id)
        return bracket_id

    def change(
        self, bracket_id: int, *, stop_ticks: int | None = None, target_ticks: int | None = None
    ) -> None:
        b = self._require(bracket_id)
        if b.phase != "open":
            raise ValueError(f"change: bracket {bracket_id} is {b.phase}, not open")
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

    def flatten(self, symbol: str) -> None:
        """Cancel pending entries and market-out open positions for the
        symbol; the exit matches from the next drained bar onward."""
        for bracket_id in self._order:
            b = self._brackets[bracket_id]
            if b.intent.symbol != symbol:
                continue
            if b.phase == "pending_entry":
                b.phase = "cancelled"
            elif b.phase == "open" and not b.flatten_requested:
                b.flatten_requested = True

    # ---------- port: drain ----------

    def drain(self, event: BarEvent) -> list[OrderEvent]:
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
                    # bar's range (post-open naked window is sub-bar).
                    if opened:
                        self._match_protection(b, event, out)
            elif b.phase == "open":
                if b.flatten_requested:
                    self._fill_flatten(b, event, open_ms, out)
                else:
                    self._match_protection(b, event, out)
        return out

    # ---------- internals ----------

    def _require(self, bracket_id: int) -> _Bracket:
        b = self._brackets.get(bracket_id)
        if b is None:
            raise KeyError(f"unknown bracket id {bracket_id}")
        return b

    def _fee(self, qty: int) -> int:
        return self._cfg.commission_cents_per_side_per_contract * qty

    def _fill_entry(self, b: _Bracket, event: BarEvent, open_ms: int, out: list[OrderEvent]) -> bool:
        """Market entry at this bar's open ± spread. Returns True when the
        bracket is OPEN afterwards (False = degenerate instant scratch)."""
        intent = b.intent
        open_ticks = self._to_ticks(event.candle.open)
        spread = self._cfg.half_spread_ticks
        fill = open_ticks + spread if intent.direction == "long" else open_ticks - spread
        b.entry_ticks = fill
        # Fill instant: the bar's open, but never before the order became
        # effective — contiguous bars make that ack+0 (the market order
        # executes the moment it arrives, at ~the open price).
        acked = intent.stamps.ts_acked
        intent.stamps.ts_filled = open_ms if acked is None else max(open_ms, acked)
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
            )
        )

        # The stop is fill-derived whenever the intent carries a risk cap, so
        # it resolves HERE — before the gap check reads it and before the
        # target is computed from it.
        b.stop_ticks = intent.stop_ticks_for_fill(fill)

        # Gap-through-stop entry: the bar opened at/through the stop level,
        # so protection triggers the instant it exists — model the honest
        # scratch (exit at the same fill) instead of inventing a target.
        gapped = (
            fill <= b.stop_ticks if intent.direction == "long" else fill >= b.stop_ticks
        )
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

    def _match_protection(self, b: _Bracket, event: BarEvent, out: list[OrderEvent]) -> None:
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

    def _to_ticks(self, price: float) -> int:
        """Bar prices are grid-truth by contract — off-grid means the
        instrument spec is wrong for this data, and that must be loud."""
        return price_to_ticks(price, self._spec)
