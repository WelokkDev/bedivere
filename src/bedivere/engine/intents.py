"""Order intents + venue events, with the six latency stamps.

A `BracketIntent` is the ONLY thing a strategy can ask a broker to do:
market entry + fixed stop + rr-derived target. Prices on the intent
are INT TICKS — the away-from-profit rounding happened exactly once, in
`BracketIntent.from_prices` (the port's doorstep, shared by every broker),
so everything downstream is integer arithmetic and a property test can
assert nothing off-grid ever crosses the port.

The target is a RULE, not a price: bracket geometry anchors on the FILL
(risk = fill − stop; target = fill + rr × risk), so the broker computes the
target after the entry fills — `target_ticks_for_fill` is that one shared
computation, rounded away-from-profit for the target leg's side.

Stamps are unix MILLISECONDS, same six names in backtest and live:
modelled there, measured here — never renamed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal

from bedivere.core.pricing import InstrumentSpec, conservative_ticks
from bedivere.engine.events import PriorityClass

Direction = Literal["long", "short"]


@dataclass(slots=True)
class TimingStamps:
    """The six per-intent stamps (unix ms). None = not reached yet."""

    ts_bar_close: int | None = None
    ts_event_received: int | None = None
    ts_decided: int | None = None
    ts_submitted: int | None = None
    ts_acked: int | None = None
    ts_filled: int | None = None

    def to_jsonable(self) -> dict[str, int | None]:
        return {
            "ts_bar_close": self.ts_bar_close,
            "ts_event_received": self.ts_event_received,
            "ts_decided": self.ts_decided,
            "ts_submitted": self.ts_submitted,
            "ts_acked": self.ts_acked,
            "ts_filled": self.ts_filled,
        }


@dataclass(slots=True)
class BracketIntent:
    """Market entry + stop + rr-target, integer-clean. Build via
    `from_prices` — direct construction is for tests that already have
    ticks."""

    symbol: str
    direction: Direction
    qty: int
    # The structural stop level, on-grid. With max_risk_ticks unset this IS
    # the stop; with it set this is the BOUND the risk cap may not cross —
    # the resolved stop is whichever of the two sits closer to the fill.
    stop_ticks: int
    target_rr: float  # target = fill + rr × (fill − stop), signed by direction
    signal_price_ticks: int  # the confirmation close that triggered the entry
    # Hard cap on entry→stop distance. None = the structural stop stands
    # whatever it costs.
    max_risk_ticks: int | None = None
    tag: str = ""  # opaque strategy context (zone id etc.), audited through
    stamps: TimingStamps = field(default_factory=TimingStamps)

    def __post_init__(self) -> None:
        if self.qty < 1:
            raise ValueError("BracketIntent.qty must be >= 1")
        if self.target_rr <= 0:
            raise ValueError("BracketIntent.target_rr must be > 0")
        if self.max_risk_ticks is not None and self.max_risk_ticks < 1:
            raise ValueError("BracketIntent.max_risk_ticks must be >= 1 or None")

    @classmethod
    def from_prices(
        cls,
        spec: InstrumentSpec,
        *,
        direction: Direction,
        qty: int,
        stop_price: float,
        target_rr: float,
        signal_price: float,
        max_risk_ticks: int | None = None,
        tag: str = "",
    ) -> BracketIntent:
        """THE port-side rounding: the stop is a price we SELL at when long
        (round down → bigger modelled risk) and BUY at when short (round up
        — same rule); the signal price is informational and rounds to
        nearest via the same conservative rule for its exit side."""
        stop_side: Literal["buy", "sell"] = "sell" if direction == "long" else "buy"
        return cls(
            symbol=spec.symbol,
            direction=direction,
            qty=qty,
            stop_ticks=conservative_ticks(stop_price, spec, stop_side),
            target_rr=target_rr,
            signal_price_ticks=conservative_ticks(signal_price, spec, stop_side),
            max_risk_ticks=max_risk_ticks,
            tag=tag,
        )

    def stop_ticks_for_fill(self, fill_ticks: int) -> int:
        """The stop THIS fill gets. Without a risk cap that is the structural
        level, unchanged. With one, risk is capped at max_risk_ticks — but the
        structural level still wins whenever it is CLOSER to the fill, so the
        cap only ever loosens a stop up to the structure, never past it:

            long:  max(fill − cap, structural)   short: min(fill + cap, structural)

        Both inputs are on-grid ints, so the result is too — the single
        away-from-profit rounding still happens once, in `from_prices`.
        """
        if self.max_risk_ticks is None:
            return self.stop_ticks
        if self.direction == "long":
            return max(fill_ticks - self.max_risk_ticks, self.stop_ticks)
        return min(fill_ticks + self.max_risk_ticks, self.stop_ticks)

    def target_ticks_for_fill(self, fill_ticks: int) -> int:
        """Target from the actual fill: fill + rr × risk, rounded
        away-from-profit for the target leg (long target is a SELL → floor;
        short target is a BUY → ceil). Risk is measured against the RESOLVED
        stop, so a risk cap pulls the target in with it and the realized R:R
        stays exact. Raises if the fill is on the wrong side of the stop
        (risk must be positive)."""
        stop = self.stop_ticks_for_fill(fill_ticks)
        if self.direction == "long":
            risk = fill_ticks - stop
            if risk <= 0:
                raise ValueError(
                    f"long bracket fill {fill_ticks} at/under stop {stop} — no positive risk"
                )
            raw = fill_ticks + self.target_rr * risk
            return int(raw // 1)  # floor: sell target rounds down
        risk = stop - fill_ticks
        if risk <= 0:
            raise ValueError(
                f"short bracket fill {fill_ticks} at/over stop {stop} — no positive risk"
            )
        raw = fill_ticks - self.target_rr * risk
        return -int((-raw) // 1)  # ceil: buy target rounds up


OrderEventKind = Literal[
    "entry_acked",
    "entry_fill",
    "protection_placed",
    "stop_fill",
    "target_fill",
    "flatten_fill",
    "cancelled",
]


@dataclass(frozen=True, slots=True)
class OrderEvent:
    """One venue-side happening for a bracket. `ts` is unix SECONDS (event
    ordering); precise timing lives in the intent's ms stamps."""

    priority_class: ClassVar[PriorityClass] = PriorityClass.VENUE

    ts: int
    kind: OrderEventKind
    bracket_id: int
    symbol: str
    direction: Direction
    qty: int
    price_ticks: int | None = None  # fills only
    fee_cents: int = 0
    ambiguous: bool = False  # stop/target both touched in the fill bar
    # protection_placed only: the levels the venue actually armed, resolved
    # against the entry fill. The strategy cannot know these at submit time
    # (both are fill-derived), so the venue reports them back.
    stop_ticks: int | None = None
    target_ticks: int | None = None
