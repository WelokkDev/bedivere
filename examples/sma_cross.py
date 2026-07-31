"""SMA crossover on synthetic data — the whole bedivere loop in one file.

Run it:

    python examples/sma_cross.py

It builds three weeks of synthetic 5-minute bars (seeded random walk — no
data files, no network), derives 30m bars in the MarketView, trades a fast/
slow SMA cross with a bracket, and writes runs/sma-demo/{result.json,
journal.jsonl}. Run it twice: the resultHash is identical.

Honesty note: the synthetic walk drifts upward by construction, so this
long-only demo prints green. The edge is manufactured; the machinery is
the point.

The strategy shows the three seams you'd use for real work:
  - ctx.view.completed(tf): higher-timeframe series maintained for you
  - ctx.journal.emit(...): your own kinds and fields (vectors, scores)
    landing in the same timeline as the fills
  - summary_jsonable(): your own counters in the result envelope
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # runnable pre-install

from bedivere import (  # noqa: E402
    BracketIntent,
    Candle,
    OrderEvent,
    RunContext,
    SessionDays,
    Timeframe,
    WarmupRequirement,
    cme_futures_sessions,
    run_backtest,
    spec_from_handoff,
)

SYMBOL = "DEMO"
SPEC = spec_from_handoff(SYMBOL, 0.25, 20)  # NQ-like grid: $5.00 per tick
FAST, SLOW = 5, 12
RR = 1.5
SIGNAL_TF = Timeframe.M30


def synthetic_bars(days_labels_first: str, days_labels_last: str, seed: int) -> tuple[list[Candle], SessionDays]:
    """A seeded random-walk 5m series over real CME ETH session-days."""
    days = cme_futures_sessions(days_labels_first, days_labels_last)
    rng = random.Random(seed)
    out: list[Candle] = []
    price = 20_000.0
    for day in days.days:
        for ts in range(day.start_unix + 300, day.end_unix + 1, 300):
            o = price
            c = round((o + rng.uniform(-8, 8.4)) * 4) / 4  # keep the walk on-grid
            h = max(o, c) + round(rng.uniform(0, 3) * 4) / 4
            low = min(o, c) - round(rng.uniform(0, 3) * 4) / 4
            out.append(Candle(timestamp=ts, open=o, high=h, low=low, close=c, volume=rng.uniform(50, 500)))
            price = c
    return out, days


def sma(series: list[Candle], n: int) -> float:
    return sum(c.close for c in series[-n:]) / n


class SmaCross:
    """Long when the fast SMA of 30m closes crosses above the slow; bracket
    with the stop under the recent 30m swing low and an rr-derived target."""

    def __init__(self) -> None:
        self.seen_closes = 0
        self.was_above: bool | None = None
        self.pending = False  # submitted, not yet filled
        self.signals = 0
        self.entries = 0
        self.blocked = 0

    # ---- strategy protocol ----

    def on_start(self, ctx: RunContext) -> None:
        ctx.journal.emit("run_start", ctx.bar.timestamp, tf=SIGNAL_TF.value, fast=FAST, slow=SLOW)

    def on_bar(self, ctx: RunContext) -> None:
        completed = ctx.view.completed(SIGNAL_TF)
        if len(completed) == self.seen_closes or len(completed) < SLOW:
            return  # act once per NEW 30m close, never mid-bucket
        self.seen_closes = len(completed)

        fast, slow = sma(completed, FAST), sma(completed, SLOW)
        above = fast > slow
        crossed_up = self.was_above is False and above
        self.was_above = above
        if not crossed_up:
            return

        self.signals += 1
        # The journal takes whatever fields you want — here a tiny feature
        # vector alongside the readable numbers.
        ctx.journal.emit(
            "signal",
            ctx.bar.timestamp,
            fast=round(fast, 2),
            slow=round(slow, 2),
            vector=[round(fast - slow, 4), round(completed[-1].close - fast, 4)],
        )
        if self.pending or ctx.portfolio.position_count() > 0:
            self.blocked += 1
            ctx.journal.emit("blocked_in_position", ctx.bar.timestamp)
            return

        stop_price = min(c.low for c in completed[-6:]) - 0.5
        intent = BracketIntent.from_prices(
            SPEC,
            direction="long",
            qty=1,
            stop_price=stop_price,
            target_rr=RR,
            signal_price=completed[-1].close,
            tag=f"cross@{ctx.bar.timestamp}",
        )
        intent.stamps.ts_bar_close = ctx.bar.timestamp * 1000
        intent.stamps.ts_event_received = ctx.event_received_ms
        intent.stamps.ts_decided = ctx.clock.now_ms()
        bracket_id = ctx.broker.submit_bracket(intent)
        ctx.portfolio.register_intent(bracket_id, intent)
        self.pending = True
        self.entries += 1
        ctx.journal.emit("entry_submitted", ctx.bar.timestamp, bracketId=bracket_id, stop=stop_price)

    def on_order_event(self, ev: OrderEvent) -> None:
        if ev.kind == "entry_fill":
            self.pending = False
        if ev.kind == "cancelled":
            self.pending = False

    def on_stop(self) -> None:
        pass

    # ---- your numbers, in the result envelope ----

    def summary_jsonable(self) -> dict[str, object]:
        return {"signals": self.signals, "entries": self.entries, "blockedInPosition": self.blocked}


def main() -> None:
    bars, days = synthetic_bars("2026-06-01", "2026-06-19", seed=7)
    run = run_backtest(
        bars=bars,
        symbol=SYMBOL,
        base_timeframe=Timeframe.M5,
        derived_timeframes=[SIGNAL_TF, Timeframe.H1],
        days=days,
        instrument=SPEC,
        strategy=SmaCross(),
        # Day 1 is warm-up history; trade from day 2's open to the last close.
        window=(days.days[1].start_unix, days.days[-1].end_unix),
        latency_ms=250,
        half_spread_ticks=1,
        commission_cents_per_side_per_contract=105,
        seed=7,
        warmup=[WarmupRequirement(SIGNAL_TF, SLOW)],
        params={"fast": FAST, "slow": SLOW, "rr": RR, "signalTf": SIGNAL_TF.value},
        journal_context={"example": "sma_cross"},
    )
    out = run.write("runs/sma-demo")
    print(json.dumps({k: run.result[k] for k in ("summary", "stats", "strategy", "paramsHash", "resultHash")}, indent=2))
    print(f"\narchived to {out}/ — run me again and compare resultHash.")


if __name__ == "__main__":
    main()
