"""A shadow session in one file: live-pushed bars, shadow venue, alerts.

Run it:

    python examples/shadow_session.py

A feeder thread stands in for your real data feed, replaying synthetic
5-minute bars at high speed (~40 bars/second) through the SAME machinery a
real session uses: `LiveBarStream.push()` on arrival, receive-time stamps,
`SimBroker` filling brackets against the live bars, and the journal routing
fills to a console notifier as they happen. Swap the feeder for your
websocket callback, swap AcceleratedClock for LiveClock, and nothing else
changes. Going armed later is `run_live` with your own Broker adapter.

Ctrl+C once ends the session gracefully (archives the run); twice aborts.
Set BEDIVERE_DISCORD_WEBHOOK to route the same alerts to Discord instead.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))  # examples.* importable pre-install
sys.path.insert(0, str(_ROOT / "src"))  # bedivere importable pre-install

from bedivere import (  # noqa: E402
    ConsoleNotifier,
    DiscordNotifier,
    LiveBarStream,
    Notifier,
    Timeframe,
    WarmupRequirement,
    install_sigint_stop,
    run_shadow,
)
from examples.sma_cross import SPEC, SYMBOL, SmaCross, synthetic_bars  # noqa: E402


class AcceleratedClock:
    """Demo-only: the session replays weeks of synthetic bars in seconds, so
    wall time would sit months after every bar and the sim venue would
    (correctly) refuse to fill orders decided 'in the future'. This clock
    rides the bar stream instead — a REAL session passes LiveClock and
    deletes this class."""

    def __init__(self, start_unix: int) -> None:
        self._unix = start_unix

    def advance(self, ts_unix: int) -> None:  # the loop's duck-typed hook
        self._unix = ts_unix

    def now_unix(self) -> int:
        return self._unix

    def now_ms(self) -> int:
        return self._unix * 1000


def build_notifier() -> Notifier:
    webhook = os.environ.get("BEDIVERE_DISCORD_WEBHOOK", "")
    if webhook:
        print("routing alerts to Discord")
        return DiscordNotifier(webhook)
    return ConsoleNotifier(prefix="[alert]")


def main() -> None:
    bars, days = synthetic_bars("2026-06-01", "2026-06-12", seed=11)
    clock = AcceleratedClock(bars[0].timestamp)

    # Day 1 is pre-loaded history (warm-up); the rest arrives "live".
    day1_end = days.days[0].end_unix
    history = [b for b in bars if b.timestamp <= day1_end]
    session = [b for b in bars if b.timestamp > day1_end]

    stream = LiveBarStream(
        symbol=SYMBOL,
        timeframe=Timeframe.M5,
        clock=clock,
        window_end_unix=session[-1].timestamp,
        history=history,
    )

    def feeder() -> None:
        """Your feed adapter goes here. This one replays the synthetic
        session fast; a real one pushes from a websocket callback."""
        for candle in session:
            if not stream.push(candle):
                return  # stream closed (Ctrl+C) — stop feeding
            time.sleep(0.025)
        stream.close()

    restore_sigint = install_sigint_stop(stream)
    threading.Thread(target=feeder, name="demo-feeder", daemon=True).start()

    try:
        run = run_shadow(
            stream=stream,
            symbol=SYMBOL,
            base_timeframe=Timeframe.M5,
            derived_timeframes=[Timeframe.M30, Timeframe.H1],
            days=days,
            instrument=SPEC,
            strategy=SmaCross(),
            window=(days.days[1].start_unix, session[-1].timestamp),
            latency_ms=250,
            half_spread_ticks=1,
            commission_cents_per_side_per_contract=105,
            seed=11,
            clock=clock,
            warmup=[WarmupRequirement(Timeframe.M30, 12)],
            journal_context={"example": "shadow_session"},
            notifier=build_notifier(),
            notify_on=["entry_fill", "stop_fill", "target_fill", "flatten_fill", "signal"],
        )
    finally:
        restore_sigint()

    out = run.write("runs/shadow-demo")
    summary = run.result["summary"]
    print(
        f"\nshadow session over — {summary['trades']} shadow trade(s), "
        f"net {summary['netCents'] / 100:.2f} USD · archived to {out}/"
    )


if __name__ == "__main__":
    main()
