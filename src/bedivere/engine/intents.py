"""Order intents + venue events, with the six latency stamps.

A `BracketIntent` is the ONLY thing a strategy can ask a broker to do: an
entry (market, resting limit, resting stop, or stop-limit) + fixed stop +
target. Prices on the intent are INT TICKS — the rounding happened exactly
once, in `BracketIntent.from_prices` (the port's doorstep, shared by every
broker), so everything downstream is integer arithmetic and a property test
can assert nothing off-grid ever crosses the port.

TWO rounding rules live at that doorstep, and they point OPPOSITE ways:

- Stop and target round AWAY FROM PROFIT (`conservative_ticks`): a price we
  buy at rounds up, a price we sell at rounds down. Those levels are going
  to fill; the only question is at what price, so assume the worse one.
- A resting ENTRY rounds AWAY FROM THE MARKET (`resting_limit_ticks` /
  `resting_stop_ticks`): a BUY limit rests below and rounds DOWN, a BUY stop
  rests above and rounds UP, and the sells mirror both. The question there
  is not "at what price" but "does it trigger AT ALL", and the conservative
  answer is the one that triggers LESS often. Rounding an entry
  away-from-profit would nudge every resting order toward an easier fill and
  invent trades the book never offered — the classic way a resting-entry
  backtest lies.

WHICH resting type to use is not a preference, it is arithmetic: a resting
order on the WRONG side of the market is immediately marketable and fills at
once, silently, at the worst price in the bar. "Buy when price RISES back to
X" is a buy STOP; "buy when price FALLS to X" is a buy LIMIT.
`marketable_entry_reason` is the shared valve that makes a mis-set level
LOUD instead of plausible, and it lives here rather than in a broker so a
sim and a live adapter can never disagree about what "marketable" means.

The target is normally a RULE, not a price: bracket geometry anchors on the
FILL (risk = fill − stop; target = fill + rr × risk), so the broker computes
the target after the entry fills. A strategy whose whole bracket must be
known at submit time sets `fixed_target_ticks` instead. That is REQUIRED for
any entry that can fill away from its own level — a stop entry always can —
because a target that slides with the fill silently absorbs entry slippage
and makes realised R come out looking clean when it isn't.

Stamps are unix MILLISECONDS, same six names in backtest and live: modelled
there, measured here — never renamed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal

from bedivere.core.pricing import (
    InstrumentSpec,
    ceil_to_ticks,
    conservative_ticks,
    floor_to_ticks,
)
from bedivere.engine.events import PriorityClass

Direction = Literal["long", "short"]
EntryType = Literal["market", "limit", "stop", "stop_limit"]

# Why a bracket was retired without a round trip. "" is the unlabelled
# cancel; the rest name a cause, so a `cancelled` line in the journal answers
# "why" without a join against broker logs.
CancelReason = Literal[
    "",
    "strategy_cancel",
    "flatten",
    "marketable_entry",
    "venue_refused",
    "venue_rejected",
]


def resting_limit_ticks(price: float, spec: InstrumentSpec, direction: Direction) -> int:
    """Away-from-the-market rounding for a resting entry limit: a BUY limit
    rests below and rounds DOWN, a SELL limit rests above and rounds UP. The
    resolved level is never CLOSER to the market than asked, so rounding can
    only make the fill less likely, never manufacture one."""
    return floor_to_ticks(price, spec) if direction == "long" else ceil_to_ticks(price, spec)


def resting_stop_ticks(price: float, spec: InstrumentSpec, direction: Direction) -> int:
    """The mirror of `resting_limit_ticks`: a resting stop sits on the other
    side of the market, so a BUY stop rounds UP and a SELL stop rounds DOWN.

    This coincides numerically with `conservative_ticks` for the entry side —
    a consequence, not the reason. These answer "does it trigger at all",
    which is a different question from "at what price".
    """
    return ceil_to_ticks(price, spec) if direction == "long" else floor_to_ticks(price, spec)


def rr_target_ticks(direction: Direction, anchor_ticks: int, risk_ticks: int, rr: float) -> int:
    """`anchor ± rr × risk`, rounded away from profit (a long target is a
    SELL, so it floors). Published once so the two callers cannot drift: a
    broker passes the FILL as anchor, a strategy fixing its bracket up front
    passes its entry TRIGGER.
    """
    if risk_ticks <= 0:
        raise ValueError(f"rr_target_ticks: risk must be positive, got {risk_ticks}")
    if direction == "long":
        return int((anchor_ticks + rr * risk_ticks) // 1)
    return -int((-(anchor_ticks - rr * risk_ticks)) // 1)


@dataclass(frozen=True, slots=True)
class EntryOrder:
    """How the entry reaches the venue. The type/level pair is validated
    HERE, so a "limit with no price" or a market carrying a stray level
    cannot be constructed at all rather than failing inside a broker:

    | type         | carries       | fills                                    |
    |--------------|---------------|------------------------------------------|
    | `market`     | nothing       | always, crossing the spread              |
    | `limit`      | `limit_ticks` | when price comes DOWN to it (long)       |
    | `stop`       | `stop_ticks`  | always, once price reaches the trigger   |
    | `stop_limit` | BOTH          | once triggered, never worse than the cap |

    `stop_ticks` here is the ENTRY TRIGGER and has nothing to do with
    `BracketIntent.stop_ticks`, which is the protective stop. The two are
    always on opposite sides of the entry.

    A stop-limit's trade is real: you keep the price, and in exchange you can
    trigger and not get filled at all, which a plain stop never does.

    There is deliberately NO expiry field: a working entry rests until the
    strategy cancels it or the run flattens at session end.
    """

    type: EntryType = "market"
    limit_ticks: int | None = None
    stop_ticks: int | None = None

    def __post_init__(self) -> None:
        wants_limit = self.type in ("limit", "stop_limit")
        wants_stop = self.type in ("stop", "stop_limit")
        if wants_limit and self.limit_ticks is None:
            raise ValueError(f"EntryOrder(type={self.type!r}) requires limit_ticks")
        if wants_stop and self.stop_ticks is None:
            raise ValueError(f"EntryOrder(type={self.type!r}) requires stop_ticks")
        if not wants_limit and self.limit_ticks is not None:
            raise ValueError(f"EntryOrder(type={self.type!r}) must not carry limit_ticks")
        if not wants_stop and self.stop_ticks is not None:
            raise ValueError(f"EntryOrder(type={self.type!r}) must not carry stop_ticks")

    @property
    def is_market(self) -> bool:
        return self.type == "market"

    @property
    def is_stop(self) -> bool:
        return self.type == "stop"

    @property
    def is_stop_limit(self) -> bool:
        return self.type == "stop_limit"

    @property
    def is_limit(self) -> bool:
        return self.type == "limit"

    @property
    def resting_ticks(self) -> int | None:
        """The level that decides WHICH SIDE of the market this entry rests
        on (None for a market entry) — the marketable-entry valve's input.

        For a stop-limit that is the TRIGGER, not the limit: the trigger is
        what must sit above the market for a buy. The limit is a cap on the
        fill price, and it is expected to sit PAST the trigger.
        """
        return self.limit_ticks if self.type == "limit" else self.stop_ticks

    def to_jsonable(self) -> dict[str, object]:
        return {"type": self.type, "limitTicks": self.limit_ticks, "stopTicks": self.stop_ticks}


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
    """Entry + stop + rr-target, integer-clean. Build via `from_prices` —
    direct construction is for tests that already have ticks."""

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
    # An ABSOLUTE target level, overriding the rr-from-fill rule. None (the
    # default) keeps the fill-derived target, which is correct for a market
    # entry. Set, the whole bracket is known at submit time and a fill worse
    # than intended shows up as a smaller realised R instead of being hidden
    # by a target that slid along with it. Required by any entry that can fill
    # away from its own level — a stop entry always can.
    fixed_target_ticks: int | None = None
    # How the entry reaches the venue. Defaults to market, so an intent built
    # the way every existing caller builds one behaves exactly as it did.
    entry: EntryOrder = field(default_factory=EntryOrder)
    tag: str = ""  # opaque strategy context (setup id etc.), audited through
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
        entry_limit_price: float | None = None,
        entry_stop_price: float | None = None,
        entry_limit_slippage_ticks: int | None = None,
        target_price: float | None = None,
        tag: str = "",
    ) -> BracketIntent:
        """THE port-side rounding: the stop is a price we SELL at when long
        (round down → bigger modelled risk) and BUY at when short (round up
        — same rule); the signal price is informational and rounds via the
        same conservative rule for its exit side.

        `entry_limit_price` / `entry_stop_price` opt into a resting entry and
        round by the OPPOSITE rule (away from the market — see the module
        docstring). They are mutually exclusive; with neither the entry is a
        market order.

        `entry_limit_slippage_ticks` makes a stop entry a STOP-LIMIT, capping
        the fill that many ticks past the trigger. A tick count rather than a
        second price on purpose: the trigger is already on-grid, so the cap
        needs no second rounding rule. Zero means "my trigger or better".

        `target_price` fixes the target absolutely; omitted, the broker
        derives it from the fill.
        """
        stop_side: Literal["buy", "sell"] = "sell" if direction == "long" else "buy"
        if entry_limit_price is not None and entry_stop_price is not None:
            raise ValueError(
                "BracketIntent.from_prices: entry_limit_price and entry_stop_price are "
                "mutually exclusive — an entry rests on ONE side of the market"
            )
        if entry_limit_slippage_ticks is not None:
            if entry_stop_price is None:
                raise ValueError(
                    "BracketIntent.from_prices: entry_limit_slippage_ticks needs "
                    "entry_stop_price — it caps the fill of a STOP entry"
                )
            if entry_limit_slippage_ticks < 0:
                raise ValueError(
                    "BracketIntent.from_prices: entry_limit_slippage_ticks must be >= 0 — "
                    "a cap inside the trigger would refuse the fill it just triggered on"
                )
        if entry_limit_price is not None:
            entry = EntryOrder(
                type="limit",
                limit_ticks=resting_limit_ticks(entry_limit_price, spec, direction),
            )
        elif entry_stop_price is not None:
            trigger = resting_stop_ticks(entry_stop_price, spec, direction)
            if entry_limit_slippage_ticks is None:
                entry = EntryOrder(type="stop", stop_ticks=trigger)
            else:
                slip = entry_limit_slippage_ticks
                entry = EntryOrder(
                    type="stop_limit",
                    stop_ticks=trigger,
                    # PAST the trigger, in the direction the fill can drift:
                    # a buy may pay more, a sell may receive less.
                    limit_ticks=trigger + slip if direction == "long" else trigger - slip,
                )
        else:
            entry = EntryOrder()
        return cls(
            symbol=spec.symbol,
            direction=direction,
            qty=qty,
            stop_ticks=conservative_ticks(stop_price, spec, stop_side),
            target_rr=target_rr,
            signal_price_ticks=conservative_ticks(signal_price, spec, stop_side),
            max_risk_ticks=max_risk_ticks,
            fixed_target_ticks=(
                None if target_price is None else conservative_ticks(target_price, spec, stop_side)
            ),
            entry=entry,
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

    def risk_ticks_for_fill(self, fill_ticks: int) -> int:
        """|fill − resolved stop|, positive by construction. Raises when the
        fill is at or past the stop, which is not a bracket at all — every
        broker relies on that being LOUD rather than producing a nonsense
        target."""
        stop = self.stop_ticks_for_fill(fill_ticks)
        risk = fill_ticks - stop if self.direction == "long" else stop - fill_ticks
        if risk <= 0:
            side = "at/under" if self.direction == "long" else "at/over"
            raise ValueError(
                f"{self.direction} bracket fill {fill_ticks} {side} stop {stop} — no positive risk"
            )
        return risk

    def target_ticks_for_fill(self, fill_ticks: int) -> int:
        """The target THIS fill gets.

        With `fixed_target_ticks` set the answer is that level, unchanged —
        the bracket was fully specified at submit time and does not move with
        the fill. Otherwise: fill + rr × risk, rounded away-from-profit for
        the target leg (long target is a SELL → floor; short target is a BUY
        → ceil). Risk is measured against the RESOLVED stop, so a risk cap
        pulls the target in with it and the realized R:R stays exact.

        Raises if the fill is on the wrong side of the stop (risk must be
        positive) — checked on BOTH paths, since a fixed target does not make
        a zero-risk bracket sane.
        """
        risk = self.risk_ticks_for_fill(fill_ticks)
        if self.fixed_target_ticks is not None:
            return self.fixed_target_ticks
        return rr_target_ticks(self.direction, fill_ticks, risk, self.target_rr)


def marketable_entry_reason(intent: BracketIntent, market_ticks: int) -> str | None:
    """THE resting-entry valve: a one-line reason when this entry would be
    immediately marketable against `market_ticks`, else None.

    A resting entry must sit STRICTLY on its own side: a buy limit below the
    market, a buy stop above it, sells mirroring both. At-the-market counts
    as marketable — a level the market is already at is not resting.

    Without this the failure is silent and plausible: a buy limit placed
    ABOVE the market fills instantly at the wrong price and produces a trade
    that reads like every other one. `market_ticks` is each broker's freshest
    price (the sim uses the last bar close it drained).
    """
    level = intent.entry.resting_ticks
    if level is None:
        return None  # a market entry is marketable BY DESIGN
    if intent.entry.is_limit:
        wrong_side = level >= market_ticks if intent.direction == "long" else level <= market_ticks
        side_word = "at/above" if intent.direction == "long" else "at/below"
    else:  # stop and stop-limit are both judged on the TRIGGER
        wrong_side = level <= market_ticks if intent.direction == "long" else level >= market_ticks
        side_word = "at/below" if intent.direction == "long" else "at/above"
    if not wrong_side:
        return None
    return (
        f"{intent.direction} {intent.entry.type} entry at {level} ticks is {side_word} the "
        f"market ({market_ticks}) — it would fill immediately instead of resting"
    )


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
    # `cancelled` only: WHY the bracket was retired without a round trip.
    # "" keeps every pre-existing emitter's bytes unchanged.
    reason: CancelReason = ""
    # `cancelled` only: the human-readable detail behind `reason` (the
    # marketable-entry valve's message, a venue error string). Never parsed.
    detail: str = ""
    # entry fills only: the open of the fill's bar. This is the baseline that
    # makes MEASURED slippage comparable to MODELLED slippage — a live fill
    # minus this open is exactly the quantity `half_spread_ticks` claims to
    # predict. None = the fill could not be matched to its bar.
    #
    # It stays the bar's OPEN for limit entries too (the field means what it
    # is named), but read it differently there: a resting limit pays no
    # spread, so fill − open measures how far from the open the level sat —
    # real price improvement, not slippage. Only for a MARKET entry is the
    # sim's answer exactly ±half_spread_ticks by construction.
    fill_open_ticks: int | None = None
    # protection_placed only: the levels the venue actually armed, resolved
    # against the entry fill. The strategy cannot know these at submit time
    # (both are fill-derived), so the venue reports them back.
    stop_ticks: int | None = None
    target_ticks: int | None = None
