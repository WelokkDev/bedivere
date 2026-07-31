"""Portfolio — engine-owned trading state.

Positions, fills, realized PnL, completed trades. Strategies QUERY it
("one trade at a time" is `portfolio.position_count() == 0`) and never
mirror it in loop-locals. It builds its state solely from the broker's
OrderEvents — the same events in backtest and live — plus the intent
registry the engine shares with it.

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


@dataclass(slots=True)
class Portfolio:
    spec: InstrumentSpec
    _open: dict[int, _OpenPosition] = field(default_factory=dict[int, _OpenPosition])
    _intents: dict[int, BracketIntent] = field(default_factory=dict[int, BracketIntent])
    trades: list[TradeRecord] = field(default_factory=list[TradeRecord])
    realized_pnl_cents: int = 0
    fees_cents: int = 0
    ambiguous_fills: int = 0

    def register_intent(self, bracket_id: int, intent: BracketIntent) -> None:
        self._intents[bracket_id] = intent

    # ---------- queries (the strategy's read surface) ----------

    def position_count(self) -> int:
        return len(self._open)

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
                )
            )
            return

        # entry_acked / protection_placed / cancelled: informational.

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
