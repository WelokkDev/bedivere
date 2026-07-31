"""run_backtest end-to-end: one crafted trade, byte-identical reruns.

The bars are constructed, not random: a flat warm-up day, then a steady
+1.00/bar ramp. The strategy enters once at the first tradeable bar with a
2.00 stop and rr=1, so the fill, the resolved protection levels, and the
target exit are all hand-computable — and asserted exactly.
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
from bedivere.engine.warmup import WarmupError, WarmupRequirement
from bedivere.run import BacktestRun, run_backtest
from bedivere.sessions import cme_futures_sessions
from tests.helpers import bar

SPEC = spec_from_handoff("DEMO", 0.25, 20)
DAYS = cme_futures_sessions("2026-07-14", "2026-07-16")


def _bars() -> list[Candle]:
    """Day 1 flat at 100.00; days 2-3 ramp +1.00 per 5m bar (all on-grid)."""
    out: list[Candle] = []
    day1 = DAYS.days[0]
    for ts in range(day1.start_unix + 300, day1.end_unix + 1, 300):
        out.append(bar(ts, 100.0, 100.25, 99.75, 100.0, v=10))
    price = 100.0
    for day in DAYS.days[1:]:
        for ts in range(day.start_unix + 300, day.end_unix + 1, 300):
            o = price
            c = o + 1.0
            out.append(bar(ts, o, c + 0.25, o - 0.25, c, v=10))
            price = c
    return out


class OneShotLong:
    """Enter one long bracket at the first tradeable bar; journal the
    decision with a custom feature vector; count what got blocked."""

    def __init__(self) -> None:
        self.entered = False
        self.blocked = 0
        self.fills: list[str] = []

    def on_start(self, ctx: RunContext) -> None:
        ctx.journal.emit("run_start", ctx.bar.timestamp, firstBar=ctx.bar.timestamp)

    def on_bar(self, ctx: RunContext) -> None:
        if self.entered:
            self.blocked += ctx.portfolio.position_count() > 0
            return
        intent = BracketIntent.from_prices(
            SPEC,
            direction="long",
            qty=1,
            stop_price=ctx.bar.close - 2.0,
            target_rr=1.0,
            signal_price=ctx.bar.close,
            tag="one-shot",
        )
        intent.stamps.ts_bar_close = ctx.bar.timestamp * 1000
        intent.stamps.ts_event_received = ctx.event_received_ms
        intent.stamps.ts_decided = ctx.clock.now_ms()
        bracket_id = ctx.broker.submit_bracket(intent)
        ctx.portfolio.register_intent(bracket_id, intent)
        ctx.journal.emit(
            "entry_decision",
            ctx.bar.timestamp,
            bracketId=bracket_id,
            vector=[ctx.bar.open, ctx.bar.close],  # any custom fields you like
            score=0.75,
        )
        self.entered = True

    def on_order_event(self, ev: OrderEvent) -> None:
        self.fills.append(ev.kind)

    def on_stop(self) -> None:
        pass

    def summary_jsonable(self) -> dict[str, object]:
        return {"entered": self.entered, "blockedWhileOpen": self.blocked}


def _run(tmp_path: Path | None = None) -> BacktestRun:
    if tmp_path is not None:
        tmp_path.mkdir(parents=True, exist_ok=True)
    return run_backtest(
        bars=_bars(),
        symbol="DEMO",
        base_timeframe=Timeframe.M5,
        derived_timeframes=[Timeframe.M30, Timeframe.H1],
        days=DAYS,
        instrument=SPEC,
        strategy=OneShotLong(),
        window=(DAYS.days[1].start_unix, DAYS.days[-1].end_unix),
        latency_ms=250,
        half_spread_ticks=0,
        commission_cents_per_side_per_contract=100,
        seed=1,
        warmup=[WarmupRequirement(Timeframe.M30, 4)],
        params={"strategy": "one-shot-long", "rr": 1.0},
        journal_context={"experiment": "e2e"},
        journal_path=None if tmp_path is None else tmp_path / "journal.jsonl",
    )


def test_the_one_trade_is_exactly_as_constructed() -> None:
    run = _run()
    trades: Any = run.result["trades"]
    assert len(trades) == 1
    t = trades[0]
    # Decision bar closes 101; fill bar opens 101 → entry 101.00 (404 ticks).
    # Stop = decision close − 2.00 = 99.00; risk 8 ticks; rr 1 → target
    # 103.00, first strictly-through on the bar whose high is 103.25.
    assert t["entryPrice"] == 101.0
    assert t["exitPrice"] == 103.0
    assert t["exitKind"] == "target_fill"
    assert t["riskTicks"] == 8
    assert t["pnlCents"] == 8 * 500  # 8 ticks × $5/tick
    assert t["feesCents"] == 200
    assert t["ambiguous"] is False
    # The six stamps rode through; acked = decided + latency.
    assert t["stamps"]["ts_acked"] == t["stamps"]["ts_decided"] + 250

    summary: Any = run.result["summary"]
    assert summary["trades"] == 1 and summary["wins"] == 1
    stats: Any = run.result["stats"]
    assert stats["rMultiples"]["known"] == 1
    strategy_summary: Any = run.result["strategy"]
    assert strategy_summary["entered"] is True

    loop: Any = run.result["loop"]
    assert loop["suppressedBars"] > 0  # the warm-up day never reached the strategy
    # Venue events all journaled + the strategy's own kinds present.
    kinds = [e["kind"] for e in run.journal.events]
    assert kinds.count("entry_fill") == 1 and kinds.count("target_fill") == 1
    assert "entry_decision" in kinds and "run_start" in kinds
    assert len([k for k in kinds if k in ("entry_fill", "protection_placed", "target_fill")]) == loop["venueEvents"]


def test_run_twice_is_byte_identical(tmp_path: Path) -> None:
    a = _run(tmp_path / "a")
    b = _run(tmp_path / "b")
    assert a.result["resultHash"] == b.result["resultHash"]
    assert a.result == b.result
    ja = (tmp_path / "a" / "journal.jsonl").read_bytes()
    jb = (tmp_path / "b" / "journal.jsonl").read_bytes()
    assert ja == jb and len(ja) > 0
    # Context is stamped into every written line.
    first = json.loads(ja.splitlines()[0])
    assert first["experiment"] == "e2e"


def test_write_archives_result_and_journal(tmp_path: Path) -> None:
    run = _run()
    out = run.write(tmp_path / "archive")
    result_disk = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result_disk["resultHash"] == run.result["resultHash"]
    assert (out / "journal.jsonl").exists()


def test_unready_warmup_hard_fails() -> None:
    with pytest.raises(WarmupError, match="unready"):
        run_backtest(
            bars=_bars(),
            symbol="DEMO",
            base_timeframe=Timeframe.M5,
            derived_timeframes=[Timeframe.M30],
            days=DAYS,
            instrument=SPEC,
            strategy=OneShotLong(),
            window=(DAYS.days[1].start_unix, DAYS.days[-1].end_unix),
            latency_ms=250,
            half_spread_ticks=0,
            commission_cents_per_side_per_contract=100,
            seed=1,
            warmup=[WarmupRequirement(Timeframe.M30, 10_000)],
        )


def test_symbol_spec_mismatch_refused() -> None:
    with pytest.raises(ValueError, match="one spec per run"):
        run_backtest(
            bars=_bars(),
            symbol="OTHER",
            base_timeframe=Timeframe.M5,
            derived_timeframes=[Timeframe.M30],
            days=DAYS,
            instrument=SPEC,
            strategy=OneShotLong(),
            window=(DAYS.days[1].start_unix, DAYS.days[-1].end_unix),
            latency_ms=0,
            half_spread_ticks=0,
            commission_cents_per_side_per_contract=0,
            seed=1,
        )
