"""A strategy plugin that lives OUTSIDE the library — the clone-don't-fork
claim, exercised.

Nothing in `bedivere` imports this module. The CLI tests reach it exactly the
way a cloner's own strategy is reached: a spec names
`"tests.strategy_fixture:PLUGIN"` and the resolver imports it. If registering
a strategy ever came to require a library edit, these tests would be the
first thing to break.

Everything here is deliberately boring — a ramp source and a strategy that
enters once — because the tests using it are about the WIRING, and a fixture
with opinions makes a wiring failure look like a strategy failure.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

from bedivere.brokers.port import BrokerContext
from bedivere.brokers.sim import SimBroker, SimBrokerConfig
from bedivere.config.base import StrategyConfig
from bedivere.config.resolve import StrategyContext, StrategyPlugin
from bedivere.core.pricing import InstrumentSpec
from bedivere.core.session_days import SessionDays
from bedivere.core.types import Candle, Timeframe
from bedivere.engine.events import BarEvent
from bedivere.engine.intents import BracketIntent, OrderEvent
from bedivere.engine.loop import RunContext
from bedivere.engine.warmup import WarmupRequirement
from bedivere.run.supervise import request_stop
from bedivere.sessions import cme_futures_sessions
from bedivere.streams.feed import FeedContext
from bedivere.streams.live import LiveBarStream

SYMBOL = "DEMO"
BASE_TF = Timeframe.M5
FIRST_CLOSE_DATE = "2026-07-14"
LAST_CLOSE_DATE = "2026-07-16"


class RampConfig(StrategyConfig):
    """Every knob, so `--set` has something at each nesting level to reach."""

    kind: Literal["ramp"] = "ramp"
    enter_after_bars: int = 2
    stop_distance: float = 2.0
    rr: float = 1.0
    qty: int = 1
    warmup_bars: int = 0  # completed 30m bars demanded before trading
    tag: str = "ramp"


class RampStrategy:
    """Submit exactly one long bracket, `enter_after_bars` tradeable bars in.

    One entry keeps the assertions about trades, journal lines and index
    numbers exact — a strategy that sometimes traded twice would make every
    test that counts anything probabilistic.
    """

    def __init__(self, config: RampConfig, instrument: InstrumentSpec) -> None:
        self.cfg = config
        self.spec = instrument
        self.bars = 0
        self.submitted = 0
        self.fills = 0

    def on_start(self, ctx: RunContext) -> None:
        ctx.journal.emit("run_start", ctx.bar.timestamp, tag=self.cfg.tag)

    def on_bar(self, ctx: RunContext) -> None:
        self.bars += 1
        if self.submitted or self.bars < self.cfg.enter_after_bars:
            return
        intent = BracketIntent.from_prices(
            self.spec,
            direction="long",
            qty=self.cfg.qty,
            stop_price=ctx.bar.close - self.cfg.stop_distance,
            target_rr=self.cfg.rr,
            signal_price=ctx.bar.close,
            tag=self.cfg.tag,
        )
        intent.stamps.ts_bar_close = ctx.bar.timestamp * 1000
        intent.stamps.ts_event_received = ctx.event_received_ms
        intent.stamps.ts_decided = ctx.clock.now_ms()
        bracket_id = ctx.broker.submit_bracket(intent)
        ctx.portfolio.register_intent(bracket_id, intent)
        self.submitted += 1
        ctx.journal.emit("entry_submitted", ctx.bar.timestamp, bracketId=bracket_id)

    def on_order_event(self, ev: OrderEvent) -> None:
        if ev.kind == "entry_fill":
            self.fills += 1

    def on_stop(self) -> None:
        pass

    def summary_jsonable(self) -> dict[str, object]:
        return {"bars": self.bars, "submitted": self.submitted, "fills": self.fills}


def _build(cfg: RampConfig, ctx: StrategyContext) -> RampStrategy:
    return RampStrategy(cfg, ctx.instrument)


def _warmup(cfg: RampConfig) -> list[WarmupRequirement]:
    return [WarmupRequirement(Timeframe.M30, cfg.warmup_bars)] if cfg.warmup_bars else []


PLUGIN = StrategyPlugin(
    kind="ramp",
    config_model=RampConfig,
    build=_build,
    warmup=_warmup,
)

# A plugin whose kind disagrees with nothing in particular — used to prove the
# spec/strategy mismatch check fires.
OTHER_PLUGIN = StrategyPlugin(kind="not_ramp", config_model=RampConfig, build=_build)

NOT_A_PLUGIN = "just a string"


# ---------- the data source ----------


def ramp_days() -> SessionDays:
    return cme_futures_sessions(FIRST_CLOSE_DATE, LAST_CLOSE_DATE)


def ramp_bars(days: SessionDays) -> list[Candle]:
    """+1.00 per 5m bar, on-grid, monotone. A long bracket placed anywhere
    fills and then reaches its target, so trade counts are hand-checkable."""
    out: list[Candle] = []
    price = 100.0
    for day in days.days:
        for ts in range(day.start_unix + 300, day.end_unix + 1, 300):
            o = price
            c = o + 1.0
            out.append(Candle(timestamp=ts, open=o, high=c + 0.25, low=o - 0.25, close=c, volume=10))
            price = c
    return out


class RampSource:
    """`CandleSource` over the ramp, generated once."""

    def __init__(self) -> None:
        self._days = ramp_days()
        self._bars = ramp_bars(self._days)

    @property
    def days(self) -> SessionDays:
        return self._days

    def candles(
        self, symbol: str, timeframe: Timeframe, start_unix: int, end_unix: int
    ) -> list[Candle]:
        if symbol != SYMBOL or timeframe is not BASE_TF:
            raise ValueError(f"fixture source holds {SYMBOL} {BASE_TF.value}")
        return [b for b in self._bars if start_unix < b.timestamp <= end_unix]

    def describe(self) -> str:
        return "fixture:ramp"


def build_ramp_source() -> RampSource:
    return RampSource()


class GappySource(RampSource):
    """The ramp with an hour punched out of the second session-day — a feed
    outage, which must be visible rather than silently short."""

    def candles(
        self, symbol: str, timeframe: Timeframe, start_unix: int, end_unix: int
    ) -> list[Candle]:
        day = self.days.days[1]
        hole = range(day.start_unix + 3600, day.start_unix + 7200)
        return [
            b
            for b in super().candles(symbol, timeframe, start_unix, end_unix)
            if b.timestamp not in hole
        ]

    def describe(self) -> str:
        return "fixture:ramp-with-a-hole"


def build_gappy_source() -> GappySource:
    return GappySource()


# ---------- feed adapters, for the live CLI tests ----------


class SynchronousReplayFeed:
    """Push the whole window into the stream at `start()`, on the CALLER's
    thread, then close it.

    Deliberately not `ReplayFeed`: a background thread makes "how many bars
    did the run see" a race. Everything downstream — the LiveBarStream, its
    receive stamps, the sim venue, the archive — is the real live path; only
    the threading is taken out of the test's way.
    """

    def __init__(self, ctx: FeedContext, *, stop_after: int | None = None) -> None:
        self.ctx = ctx
        self.stop_after = stop_after
        self.on_bar: Callable[[int], None] | None = None
        self.stopped = False

    def start(self, stream: LiveBarStream) -> None:
        source = self.ctx.source
        assert source is not None
        bars = source.candles(
            self.ctx.symbol,
            self.ctx.timeframe,
            self.ctx.window_start_unix,
            self.ctx.window_end_unix,
        )
        for pushed, candle in enumerate(bars, start=1):
            if not stream.push(candle):
                break
            if self.on_bar is not None:
                self.on_bar(pushed)
            if self.stop_after is not None and pushed >= self.stop_after:
                return  # leave the stream OPEN — something else must end the run
        stream.close()

    def stop(self) -> None:
        self.stopped = True


def build_synchronous_feed(ctx: FeedContext) -> SynchronousReplayFeed:
    return SynchronousReplayFeed(ctx)


def build_stopping_feed(
    ctx: FeedContext, *, runs_dir: str, stop_after: int = 3
) -> SynchronousReplayFeed:
    """Pushes `stop_after` bars, then creates the STOP sentinel and leaves the
    stream OPEN — so the only thing that can end the run is supervision
    noticing the sentinel.

    `runs_dir` arrives through the spec's `options`, which is the same way a
    real adapter receives its endpoint: bedivere hands the factory a context
    and gets out of the way."""
    feed = SynchronousReplayFeed(ctx, stop_after=stop_after)

    def touch_sentinel(pushed: int) -> None:
        if pushed == stop_after:
            request_stop(Path(runs_dir))

    feed.on_bar = touch_sentinel
    return feed


class ExplodingFeed:
    """A feed adapter that fails on start — the run must still release its
    lock."""

    def __init__(self, ctx: FeedContext) -> None:
        self.ctx = ctx

    def start(self, stream: LiveBarStream) -> None:
        raise RuntimeError("exploding feed adapter")

    def stop(self) -> None:
        return


def build_exploding_feed(ctx: FeedContext) -> ExplodingFeed:
    return ExplodingFeed(ctx)


# ---------- a broker adapter with a preflight opinion ----------

BROKER_CONTEXTS: list[BrokerContext] = []


class BlockableBroker:
    """A stand-in venue adapter: a SimBroker underneath, plus a preflight that
    can be told the venue is dirty. It is what a real adapter's SHAPE looks
    like — everything except talking to anything."""

    def __init__(self, ctx: BrokerContext, *, reason: str | None = None) -> None:
        self._reason = reason
        self._sim = SimBroker(
            ctx.instrument,
            SimBrokerConfig(
                bar_period_seconds=BASE_TF.period_seconds,
                latency_ms=250,
                half_spread_ticks=1,
                commission_cents_per_side_per_contract=105,
                seed=7,
                defer_protection_one_bar=False,
            ),
        )

    def preflight(self) -> str | None:
        return self._reason

    def submit_bracket(self, intent: BracketIntent) -> int:
        return self._sim.submit_bracket(intent)

    def change(
        self, bracket_id: int, *, stop_ticks: int | None = None, target_ticks: int | None = None
    ) -> None:
        self._sim.change(bracket_id, stop_ticks=stop_ticks, target_ticks=target_ticks)

    def cancel(self, bracket_id: int) -> None:
        self._sim.cancel(bracket_id)

    def flatten(self, symbol: str) -> None:
        self._sim.flatten(symbol)

    def drain(self, event: BarEvent | None) -> list[OrderEvent]:
        return self._sim.drain(event)


def build_blocked_broker(ctx: BrokerContext, *, reason: str | None = None) -> BlockableBroker:
    BROKER_CONTEXTS.append(ctx)
    return BlockableBroker(ctx, reason=reason)
