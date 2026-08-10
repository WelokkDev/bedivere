"""Portfolio — engine-owned trading state.

Positions, fills, realized PnL, completed trades. Strategies QUERY it and
never mirror it in loop-locals: "one trade at a time" is
`portfolio.exposure_count() == 0` for a strategy that rests entries, and
`position_count() == 0` for one that only ever enters at market. It builds
its state solely from the broker's OrderEvents — the same events in backtest
and live — plus the intent registry the engine shares with it.

Money is int cents throughout (bedivere.core.pricing); float dollars appear
only in the jsonable output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bedivere.core.pricing import InstrumentSpec, ticks_pnl_cents, ticks_to_price
from bedivere.engine.intents import BracketIntent, Direction, OrderEvent


@dataclass(slots=True)
class TradeRecord:
    """One completed round trip (entry fill → exit fill)."""

    bracket_id: int
    symbol: str
    direction: Direction
    qty: int
    entry_ticks: int
    exit_ticks: int
    signal_ticks: int  # the confirmation close the entry was decided on
    exit_kind: str  # "stop_fill" | "target_fill" | "flatten_fill"
    entry_ts: int  # unix seconds (bar-level)
    exit_ts: int
    pnl_cents: int  # price PnL net of nothing — fees split out
    fees_cents: int
    ambiguous: bool
    tag: str
    stamps: dict[str, int | None]  # the six ts_* (ms) from the intent
    # |entry − stop| for the stop the venue ACTUALLY armed (fill-derived when
    # the intent carries a risk cap). None only if protection never landed.
    risk_ticks: int | None = None
    # The open of the bar the entry filled on — the sim's fill baseline, so
    # `entry − open` measures what half_spread_ticks models. None when the
    # venue could not match the fill to a bar (live adapters may not).
    entry_open_ticks: int | None = None
    # How the entry reached the venue ("market" | "limit" | "stop" |
    # "stop_limit"). Carried because the two slip measures below mean
    # DIFFERENT things per entry type, and a slippage aggregate that mixes
    # them is not a number anyone can act on.
    entry_type: str = "market"

    def entry_slip_ticks(self) -> int | None:
        """Fill against the OPEN of the bar it filled on, signed so positive
        is adverse (a long paying more, a short receiving less).

        Read this by entry type. A MARKET entry crosses the spread at the
        open, so the sim's answer is exactly `+half_spread_ticks` and the live
        answer is what the book actually charged — that comparison is the
        point. A RESTING entry did not fill at the open at all: for a limit,
        `fill − open` is how far from the open the level sat (price
        improvement, not slippage), and for a stop it is how far the market
        travelled to reach the trigger. Neither is an execution cost.
        """
        if self.entry_open_ticks is None:
            return None
        if self.direction == "long":
            return self.entry_ticks - self.entry_open_ticks
        return self.entry_open_ticks - self.entry_ticks

    def signal_slip_ticks(self) -> int | None:
        """Fill against the price the DECISION was made at, signed so positive
        is adverse.

        This is the measure that stays meaningful for a resting entry: a stop
        entry's signal price is its trigger, so this is the trigger-to-fill
        cost a taker pays — and it is precisely the cost a fill-derived target
        would have absorbed silently. Returns None when no signal price was
        recorded.
        """
        if not self.signal_ticks:
            return None
        if self.direction == "long":
            return self.entry_ticks - self.signal_ticks
        return self.signal_ticks - self.entry_ticks

    def r_multiple(self, spec: InstrumentSpec) -> float | None:
        """Result in units of the risk actually taken — the only unit that
        compares trades (or runs) with different stop distances."""
        if not self.risk_ticks:
            return None
        risk_cents = ticks_pnl_cents(self.risk_ticks, self.qty, spec)
        if risk_cents == 0:
            return None
        return (self.pnl_cents - self.fees_cents) / risk_cents

    def to_jsonable(self, spec: InstrumentSpec) -> dict[str, object]:
        r = self.r_multiple(spec)
        return {
            "riskTicks": self.risk_ticks,
            "rMultiple": None if r is None else round(r, 4),
            "durationSeconds": self.exit_ts - self.entry_ts,
            "entryType": self.entry_type,
            "entryOpenTicks": self.entry_open_ticks,
            "entrySlipTicks": self.entry_slip_ticks(),
            "signalSlipTicks": self.signal_slip_ticks(),
            "bracketId": self.bracket_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "qty": self.qty,
            "entryPrice": ticks_to_price(self.entry_ticks, spec),
            "exitPrice": ticks_to_price(self.exit_ticks, spec),
            "signalPrice": ticks_to_price(self.signal_ticks, spec),
            "entryTicks": self.entry_ticks,
            "exitTicks": self.exit_ticks,
            "signalTicks": self.signal_ticks,
            "exitKind": self.exit_kind,
            "entryTs": self.entry_ts,
            "exitTs": self.exit_ts,
            "pnlCents": self.pnl_cents,
            "feesCents": self.fees_cents,
            "netCents": self.pnl_cents - self.fees_cents,
            "ambiguous": self.ambiguous,
            "tag": self.tag,
            "stamps": self.stamps,
        }


@dataclass(slots=True)
class _OpenPosition:
    bracket_id: int
    symbol: str
    direction: Direction
    qty: int
    entry_ticks: int
    entry_ts: int
    fees_cents: int
    tag: str
    stop_ticks: int | None = None  # from protection_placed (fill-derived)
    entry_open_ticks: int | None = None  # open of the bar the entry filled on
    entry_type: str = "market"


@dataclass(slots=True)
class Portfolio:
    spec: InstrumentSpec
    _open: dict[int, _OpenPosition] = field(default_factory=dict[int, _OpenPosition])
    _intents: dict[int, BracketIntent] = field(default_factory=dict[int, BracketIntent])
    # Registered but not yet filled or retired — the other half of
    # `exposure_count()`. A set, not a counter: an event for a bracket the
    # portfolio never saw must be a no-op rather than a negative number.
    _pending: set[int] = field(default_factory=set[int])
    trades: list[TradeRecord] = field(default_factory=list[TradeRecord])
    realized_pnl_cents: int = 0
    fees_cents: int = 0
    ambiguous_fills: int = 0

    def register_intent(self, bracket_id: int, intent: BracketIntent) -> None:
        """Hand the portfolio the intent behind a bracket id. This is also
        what puts the bracket into the pending set — a strategy that submits
        without registering gets no `exposure_count()` protection, which is
        why every example registers on the line after it submits."""
        self._intents[bracket_id] = intent
        self._pending.add(bracket_id)

    # ---------- queries (the strategy's read surface) ----------

    def position_count(self) -> int:
        return len(self._open)

    def exposure_count(self) -> int:
        """Open positions PLUS working entries that have not filled yet.

        This is the query a strategy with a resting entry must ask, and
        `position_count()` is not it: a submitted-but-unfilled entry is an
        idea already committed to the venue, and a strategy that only counts
        filled positions will happily commit a second one. The alternative —
        the strategy keeping a `pending` flag of its own — is exactly the
        private bookkeeping the engine exists to make unnecessary, and it
        desynchronises the first time a bracket is refused or cancelled by
        something the strategy did not initiate.

        Pending brackets enter on `register_intent` and leave on their
        `entry_fill` or `cancelled` event, which is why a cancel that emits no
        event would wedge this number permanently.
        """
        return len(self._open) + len(self._pending)

    def bracket_live(self, bracket_id: int) -> bool:
        """Is this bracket still the engine's business — pending entry or
        open position? A strategy tracking a bracket it submitted uses this
        to SELF-HEAL: a tracker stuck on a bracket the portfolio no longer
        holds means its terminal event never arrived (a suppressed bar, a
        reconcile), and the portfolio is truth."""
        return bracket_id in self._open or bracket_id in self._pending

    def intent_for(self, bracket_id: int) -> BracketIntent | None:
        """The intent behind a bracket, for strategies that want to report
        what they ASKED for next to what they got — signal price against
        fill, planned risk against realised."""
        return self._intents.get(bracket_id)

    def entry_ticks_for(self, bracket_id: int) -> int | None:
        """Fill price of an OPEN bracket's entry. None once it has closed —
        ask `last_trade_for` then; the two together cover a `protection_placed`
        that arrives in the same drain batch as the exit."""
        pos = self._open.get(bracket_id)
        return None if pos is None else pos.entry_ticks

    def last_trade_for(self, bracket_id: int) -> TradeRecord | None:
        """The most recent completed round trip for a bracket, so
        `on_order_event` — which has no context — can report the move a fill
        actually produced without the strategy keeping its own books."""
        for trade in reversed(self.trades):
            if trade.bracket_id == bracket_id:
                return trade
        return None

    def position_qty(self, symbol: str) -> int:
        """Signed net quantity for a symbol (+long / −short)."""
        total = 0
        for pos in self._open.values():
            if pos.symbol != symbol:
                continue
            if pos.direction == "long":
                total += pos.qty
            else:
                total -= pos.qty
        return total

    # ---------- event application ----------

    def apply(self, events: list[OrderEvent]) -> None:
        for ev in events:
            self._apply_one(ev)

    def _apply_one(self, ev: OrderEvent) -> None:
        if ev.kind == "entry_fill":
            assert ev.price_ticks is not None
            if ev.bracket_id in self._open:
                raise ValueError(f"duplicate entry_fill for bracket {ev.bracket_id}")
            self._pending.discard(ev.bracket_id)
            intent = self._intents.get(ev.bracket_id)
            self._open[ev.bracket_id] = _OpenPosition(
                bracket_id=ev.bracket_id,
                symbol=ev.symbol,
                direction=ev.direction,
                qty=ev.qty,
                entry_ticks=ev.price_ticks,
                entry_ts=ev.ts,
                fees_cents=ev.fee_cents,
                tag=intent.tag if intent is not None else "",
                entry_open_ticks=ev.fill_open_ticks,
                entry_type=intent.entry.type if intent is not None else "market",
            )
            self.fees_cents += ev.fee_cents
            return

        if ev.kind == "protection_placed":
            # Both legs are fill-derived, so the venue is the only thing that
            # knows where the stop actually landed — record it for the trade's
            # risk (and so R-multiples need no join against the journal).
            pos = self._open.get(ev.bracket_id)
            if pos is not None and ev.stop_ticks is not None:
                pos.stop_ticks = ev.stop_ticks
            return

        if ev.kind in ("stop_fill", "target_fill", "flatten_fill"):
            assert ev.price_ticks is not None
            pos = self._open.pop(ev.bracket_id, None)
            if pos is None:
                raise ValueError(f"{ev.kind} for unknown bracket {ev.bracket_id}")
            delta = (
                ev.price_ticks - pos.entry_ticks
                if pos.direction == "long"
                else pos.entry_ticks - ev.price_ticks
            )
            pnl = ticks_pnl_cents(delta, pos.qty, self.spec)
            fees = pos.fees_cents + ev.fee_cents
            self.realized_pnl_cents += pnl
            self.fees_cents += ev.fee_cents
            if ev.ambiguous:
                self.ambiguous_fills += 1
            intent = self._intents.get(ev.bracket_id)
            self.trades.append(
                TradeRecord(
                    bracket_id=ev.bracket_id,
                    symbol=pos.symbol,
                    direction=pos.direction,
                    qty=pos.qty,
                    entry_ticks=pos.entry_ticks,
                    exit_ticks=ev.price_ticks,
                    signal_ticks=intent.signal_price_ticks if intent is not None else 0,
                    exit_kind=ev.kind,
                    entry_ts=pos.entry_ts,
                    exit_ts=ev.ts,
                    pnl_cents=pnl,
                    fees_cents=fees,
                    ambiguous=ev.ambiguous,
                    tag=pos.tag,
                    stamps=intent.stamps.to_jsonable() if intent is not None else {},
                    risk_ticks=(
                        None
                        if pos.stop_ticks is None
                        else (
                            pos.entry_ticks - pos.stop_ticks
                            if pos.direction == "long"
                            else pos.stop_ticks - pos.entry_ticks
                        )
                    ),
                    entry_open_ticks=pos.entry_open_ticks,
                    entry_type=pos.entry_type,
                )
            )
            return

        if ev.kind == "cancelled":
            # The bracket was retired without a round trip (its setup died,
            # a flatten swept it, the venue refused it). Nothing to book —
            # but the exposure it was holding has to be released, and this
            # event is the ONLY thing that says so.
            self._pending.discard(ev.bracket_id)
            return

        # entry_acked / protection_placed: informational.

    def summary_jsonable(self) -> dict[str, object]:
        wins = sum(1 for t in self.trades if t.pnl_cents - t.fees_cents > 0)
        losses = sum(1 for t in self.trades if t.pnl_cents - t.fees_cents < 0)
        return {
            "trades": len(self.trades),
            "wins": wins,
            "losses": losses,
            "flat": len(self.trades) - wins - losses,
            "pnlCents": self.realized_pnl_cents,
            "feesCents": self.fees_cents,
            "netCents": self.realized_pnl_cents - self.fees_cents,
            "ambiguousFills": self.ambiguous_fills,
            "openPositions": len(self._open),
        }
