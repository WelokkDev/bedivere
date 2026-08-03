"""run_live / run_shadow — the shadow composition end to end.

Bars are pre-pushed into the LiveBarStream with a settable clock, so the
whole "live" session is deterministic: same crafted ramp as the backtest
e2e, one hand-computable trade, receive stamps proven to ride through to
the intent's latency stamps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from bedivere.core.pricing import spec_from_handoff
from bedivere.core.types import Candle, Timeframe
from bedivere.engine.intents import BracketIntent, OrderEvent
from bedivere.engine.loop import RunContext
from bedivere.engine.warmup import WarmupRequirement
from bedivere.run.live import run_shadow
from bedivere.sessions import cme_futures_sessions
from bedivere.streams.live import LiveBarStream
from tests.helpers import bar
from tests.test_live_stream import FakeClock
from tests.test_notify import FakeNotifier

SPEC = spec_from_handoff("DEMO", 0.25, 20)
DAYS = cme_futures_sessions("2026-07-14", "2026-07-15")


def _history() -> list[Candle]:
    day1 = DAYS.days[0]
    return [
        bar(ts, 100.0, 100.25, 99.75, 100.0, v=10)
        for ts in range(day1.start_unix + 300, day1.end_unix + 1, 300)
    ]


def _day2_bars() -> list[Candle]:
    out: list[Candle] = []
    price = 100.0
    day = DAYS.days[1]
    for ts in range(day.start_unix + 300, day.end_unix + 1, 300):
        o = price
        c = o + 1.0
        out.append(bar(ts, o, c + 0.25, o - 0.25, c, v=10))
        price = c
    return out


class OneShotLong:
    def __init__(self, *, explode_after_fill: bool = False) -> None:
        self.entered = False
        self.explode = explode_after_fill
        self.filled = False
        self.decision_received_ms: int | None = None

    def on_start(self, ctx: RunContext) -> None:
        ctx.journal.emit("run_start", ctx.bar.timestamp)

    def on_bar(self, ctx: RunContext) -> None:
        if self.filled and self.explode:
            raise RuntimeError("strategy exploded mid-position")
        if self.entered:
            return
        self.decision_received_ms = ctx.event_received_ms
        intent = BracketIntent.from_prices(
            SPEC,
            direction="long",
            qty=1,
            stop_price=ctx.bar.close - 2.0,
            target_rr=1.0,
            signal_price=ctx.bar.close,
        )
        intent.stamps.ts_bar_close = ctx.bar.timestamp * 1000
        intent.stamps.ts_event_received = ctx.event_received_ms
        intent.stamps.ts_decided = ctx.clock.now_ms()
        bid = ctx.broker.submit_bracket(intent)
        ctx.portfolio.register_intent(bid, intent)
        self.entered = True

    def on_order_event(self, ev: OrderEvent) -> None:
        if ev.kind == "entry_fill":
            self.filled = True

    def on_stop(self) -> None:
        pass


def _shadow_run(
    tmp_path: Path,
    *,
    explode: bool = False,
) -> tuple[Any, FakeNotifier, FakeClock, list[Candle]]:
    clock = FakeClock(DAYS.days[0].start_unix)
    day2 = _day2_bars()
    window = (DAYS.days[1].start_unix, DAYS.days[1].end_unix)
    stream = LiveBarStream(
        symbol="DEMO",
        timeframe=Timeframe.M5,
        clock=clock,
        window_end_unix=window[1],
        history=_history(),
        idle_poll_s=0.01,
    )
    # Pre-push the whole session with per-bar receive instants: bar N is
    # "received" one second after its close.
    for candle in day2:
        clock.unix = candle.timestamp + 1
        assert stream.push(candle)
    notifier = FakeNotifier()
    run = run_shadow(
        stream=stream,
        symbol="DEMO",
        base_timeframe=Timeframe.M5,
        derived_timeframes=[Timeframe.M30],
        days=DAYS,
        instrument=SPEC,
        strategy=OneShotLong(explode_after_fill=explode),
        window=window,
        latency_ms=250,
        half_spread_ticks=0,
        commission_cents_per_side_per_contract=100,
        seed=1,
        clock=clock,
        warmup=[WarmupRequirement(Timeframe.M30, 4)],
        journal_context={"experiment": "live-e2e"},
        journal_path=tmp_path / "journal.jsonl",
        notifier=notifier,
    )
    return run, notifier, clock, day2


def test_shadow_session_end_to_end(tmp_path: Path) -> None:
    run, notifier, _clock, day2 = _shadow_run(tmp_path)
    result = run.result

    assert result["mode"] == "shadow"
    trades: Any = result["trades"]
    assert len(trades) == 1
    t = trades[0]
    assert t["entryPrice"] == 101.0 and t["exitPrice"] == 103.0
    assert t["exitKind"] == "target_fill"

    # The receive stamp is the PUSH instant (close + 1s), not the close —
    # and it rode into the intent's latency stamps untouched.
    decision_ts = day2[0].timestamp
    assert t["stamps"]["ts_event_received"] == (decision_ts + 1) * 1000

    # Feed counters surfaced; every session bar arrived live.
    feed: Any = result["feed"]
    assert feed["historyBars"] == len(_history())
    assert feed["liveBars"] == len(day2)
    assert feed["duplicates"] == 0

    # Notifier lifecycle: start banner, routed fills, finish banner.
    assert any(text.startswith("▶ shadow started") for text in notifier.sent)
    assert any("ENTRY" in text for text in notifier.sent)
    assert any("TARGET" in text for text in notifier.sent)
    assert any(text.startswith("⏹ shadow finished — 1 trade(s)") for text in notifier.sent)
    assert notifier.closed
    notify: Any = result["notify"]
    assert notify["routed"] >= 2

    # The journal streamed to disk per event, context stamped on every line.
    lines = (tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == result["journal"]["events"]
    first = json.loads(lines[0])
    assert first["experiment"] == "live-e2e" and first["mode"] == "shadow"


def test_dying_run_flattens_and_screams(tmp_path: Path) -> None:
    clock = FakeClock(DAYS.days[0].start_unix)
    window = (DAYS.days[1].start_unix, DAYS.days[1].end_unix)
    stream = LiveBarStream(
        symbol="DEMO",
        timeframe=Timeframe.M5,
        clock=clock,
        window_end_unix=window[1],
        history=_history(),
        idle_poll_s=0.01,
    )
    for candle in _day2_bars():
        clock.unix = candle.timestamp + 1
        stream.push(candle)
    notifier = FakeNotifier()

    with pytest.raises(RuntimeError, match="strategy exploded"):
        run_shadow(
            stream=stream,
            symbol="DEMO",
            base_timeframe=Timeframe.M5,
            derived_timeframes=[Timeframe.M30],
            days=DAYS,
            instrument=SPEC,
            strategy=OneShotLong(explode_after_fill=True),
            window=window,
            latency_ms=250,
            half_spread_ticks=0,
            commission_cents_per_side_per_contract=100,
            seed=1,
            clock=clock,
            journal_path=tmp_path / "journal.jsonl",
            notifier=notifier,
        )

    # The fail-closed rail fired: flatten warning + crash notice, notifier
    # closed, and the streamed journal shows the fill with no exit.
    assert any("🚨" in text and "flatten sent" in text for text in notifier.sent)
    assert any(text.startswith("💥 shadow run failed") for text in notifier.sent)
    assert notifier.closed
    kinds = [
        json.loads(line)["kind"]
        for line in (tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert "entry_fill" in kinds
    assert "target_fill" not in kinds and "stop_fill" not in kinds
