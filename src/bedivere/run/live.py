"""run_live / run_shadow — the live composition roots.

One engine, three environments: a backtest replays bars, a shadow session
runs live bars against the in-process SimBroker, a live session runs the
same bars against YOUR venue adapter. The only moving parts are the triad
{BarStream, Clock, Broker}; everything downstream of the stream is the
run_loop both runners share with run_backtest.

Live-specific machinery owned here:
  - `journal_path` streams JSONL flushed per event (crash-safe) — a
    backtest writes its journal at the end; a live session must never owe
    the disk anything.
  - `journal_sinks` fan out each context-merged line; a `notifier` +
    `notify_on` kinds install a NotificationRouter as one more sink.
  - PREFLIGHT: `broker.preflight()` runs before the first bar and REFUSES
    the run on any reason it returns — a position or working order the run
    did not create is not something to trade around.
  - FAIL-CLOSED FLATTEN: if the run ends or DIES with an open position,
    `broker.flatten(symbol)` is sent and the notifier warns you to confirm
    at the venue.
  - `install_sigint_stop`: first Ctrl+C ends the session gracefully,
    second aborts.
"""

from __future__ import annotations

import json
import signal
import sys
import threading
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import Any, Protocol

from bedivere.brokers.sim import SimBroker, SimBrokerConfig
from bedivere.core.clock import Clock, LiveClock
from bedivere.core.pricing import InstrumentSpec
from bedivere.core.session_days import SessionDays
from bedivere.core.types import Timeframe
from bedivere.engine.events import BarEvent
from bedivere.engine.journal import DecisionJournal
from bedivere.engine.loop import BrokerLike, CloseSink, Strategy, run_loop
from bedivere.engine.portfolio import Portfolio
from bedivere.engine.warmup import WarmupGate, WarmupRequirement
from bedivere.notify.port import Notifier, NullNotifier
from bedivere.notify.router import DEFAULT_KINDS, NotificationRouter
from bedivere.run.record import RunRecord, build_result
from bedivere.streams.live import LiveBarStream
from bedivere.view.market_view import MarketView, ObserverFactory


class LiveBroker(BrokerLike, Protocol):
    """What a LIVE run needs beyond what the loop needs: the arming gate.

    The engine loop is deliberately narrower — it never asks a broker whether
    the run should have started, because by the time the loop is running that
    question is already answered. Live composition is where it belongs, and
    typing it here means an adapter without a `preflight` fails at your
    composition root rather than at the venue.
    """

    def preflight(self) -> str | None: ...


class PreflightError(RuntimeError):
    """The venue is not in a state this run may start from."""


class _StreamingTap:
    """Journal sink for live runs: append each line to a JSONL file, flushed
    per event — the crash-safe record."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8", newline="\n")
        self.lines = 0

    def __call__(self, line: dict[str, object]) -> None:
        self._file.write(json.dumps(line, sort_keys=True, separators=(",", ":")) + "\n")
        self._file.flush()
        self.lines += 1

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:  # noqa: BLE001, S110 — teardown must not raise
            pass


def _hhmm_utc(unix_sec: int) -> str:
    """Render a GIVEN instant (not a wall-clock read)."""
    return datetime.fromtimestamp(unix_sec, UTC).strftime("%Y-%m-%d %H:%M") + "Z"


def run_live(
    *,
    stream: Iterable[BarEvent],
    symbol: str,
    base_timeframe: Timeframe,
    derived_timeframes: Sequence[Timeframe],
    days: SessionDays,
    instrument: InstrumentSpec,
    broker: LiveBroker,
    clock: Clock,
    strategy: Strategy,
    window: tuple[int, int],
    warmup: Sequence[WarmupRequirement] = (),
    observer_factory: ObserverFactory | None = None,
    close_sink: CloseSink | None = None,
    portfolio: Portfolio | None = None,
    params: dict[str, Any] | None = None,
    journal_context: dict[str, Any] | None = None,
    journal_path: str | Path | None = None,
    journal_sinks: Sequence[Callable[[dict[str, object]], None]] = (),
    notifier: Notifier | None = None,
    notify_on: Iterable[str] | None = None,
    notify_known_kinds: Iterable[str] | None = None,
    mode: str = "live",
) -> RunRecord:
    """Drive one session: an injected stream (usually a LiveBarStream your
    feed adapter pushes into), an injected clock (LiveClock in production),
    and an injected broker — SimBroker for an in-process shadow, your own
    venue adapter for real orders. Nothing else differs from a backtest.

    This is the MAIN trader. The engine only knows which BROKER you gave
    it, never which ACCOUNT that broker points at — whether a session is
    "paper" is a property of your account, so YOU declare it: pass
    `mode="paper"` / `mode="funded"` / whatever, and that label rides
    into the banners, the journal context, and the result. The engine's
    only intrinsic distinction is shadow (simulated fills in-process,
    see run_shadow) versus live (a real order path).

    The result envelope matches run_backtest's shape (plus `mode`, feed
    and notify counters). Its `resultHash` fingerprints THIS run — live
    hashes are not reproducible (receive stamps are real)."""
    window_start, window_end = window
    if window_start >= window_end:
        raise ValueError(f"window start {window_start} must be before end {window_end}")
    if instrument.symbol != symbol:
        raise ValueError(
            f'instrument is for "{instrument.symbol}" but the run is for "{symbol}" — one spec per run'
        )

    view = MarketView(
        base_tf=base_timeframe,
        derived_tfs=derived_timeframes,
        days=days,
        observer_factory=observer_factory,
    )
    # A live composition may OWN the portfolio — a supervisor reporting open
    # positions and trade counts every few seconds needs a handle on it, and
    # reaching into a run mid-flight for one is worse than being handed it.
    book = portfolio if portfolio is not None else Portfolio(spec=instrument)
    gate = WarmupGate(view, list(warmup)) if warmup else None
    alerts: Notifier = notifier if notifier is not None else NullNotifier()

    context: dict[str, Any] = {"mode": mode}
    if journal_context:
        context.update(journal_context)
    journal = DecisionJournal(context=context)

    router: NotificationRouter | None = None
    if notifier is not None:
        kinds = frozenset(notify_on) if notify_on is not None else DEFAULT_KINDS
        router = NotificationRouter(kinds, notifier, known_kinds=notify_known_kinds)

    # Reconcile BEFORE anything is opened for writing: a refused run leaves no
    # journal file and no archive behind. The notifier IS used — a refusal is
    # precisely the thing an operator who is not at the terminal needs told.
    blocked = broker.preflight()
    if blocked is not None:
        message = (
            f"⛔ {mode} refused to start — {symbol}: {blocked} — "
            "reconcile at the venue, then start again"
        )
        alerts.send(message)
        alerts.close()
        raise PreflightError(message)

    tap = _StreamingTap(Path(journal_path)) if journal_path is not None else None
    sinks: list[Callable[[dict[str, object]], None]] = []
    if tap is not None:
        sinks.append(tap)
    if router is not None:
        sinks.append(router)
    sinks.extend(journal_sinks)
    if sinks:

        def fan_out(line: dict[str, object]) -> None:
            for sink in sinks:
                try:
                    sink(line)
                except Exception as e:  # noqa: BLE001 — observability may not sink the loop
                    sys.stderr.write(f"[journal] sink failed: {e}\n")

        journal.sink = fan_out

    alerts.send(
        f"▶ {mode} started — {symbol} {base_timeframe.value} · until {_hhmm_utc(window_end)}"
    )

    try:
        loop_stats = run_loop(
            stream=stream,
            clock=clock,
            view=view,
            broker=broker,
            portfolio=book,
            strategy=strategy,
            tradeable_start_unix=window_start,
            close_sink=close_sink,
            warmup_gate=gate,
            journal=journal,
        )
    except BaseException as e:  # incl. KeyboardInterrupt — a dying run must say so
        if book.position_count() > 0:
            broker.flatten(symbol)
            alerts.send(
                f"🚨 {mode} run failed with an open position — flatten sent; CONFIRM at your venue"
            )
        alerts.send(f"💥 {mode} run failed: {e}")
        alerts.close()
        if tap is not None:
            tap.close()
        raise

    if book.position_count() > 0:
        broker.flatten(symbol)
        alerts.send(
            f"⚠ {mode} window ended with an open position — flatten sent; CONFIRM at your venue"
        )

    result = build_result(
        symbol=symbol,
        base_timeframe=base_timeframe,
        view=view,
        window=window,
        warmup=warmup,
        loop_stats=loop_stats,
        portfolio=book,
        instrument=instrument,
        journal=journal,
        strategy=strategy,
        params=params,
        extra={
            "mode": mode,
            "feed": {
                # Envelope keys are camelCase like the rest of the result;
                # the counters live on the stream as snake_case attributes.
                camel: value
                for attr, camel in (
                    ("history_bars", "historyBars"),
                    ("live_bars", "liveBars"),
                    ("backfill_bars", "backfillBars"),
                    ("duplicates", "duplicates"),
                    ("dropped", "dropped"),
                )
                if isinstance(value := getattr(stream, attr, None), int)
            },
            "notify": {
                "kinds": sorted(router.kinds) if router is not None else [],
                "routed": router.routed if router is not None else 0,
                "delivered": getattr(alerts, "delivered", None),
                "dropped": getattr(alerts, "dropped", None),
            },
        },
    )

    summary = result["summary"]
    alerts.send(
        f"⏹ {mode} finished — {summary['trades']} trade(s), net {summary['netCents'] / 100:.2f} USD"
        + (" · ⚠ position was open at shutdown" if summary["openPositions"] else "")
    )
    alerts.close()
    if tap is not None:
        tap.close()
    return RunRecord(result=result, journal=journal, portfolio=book, view=view)


def run_shadow(
    *,
    stream: Iterable[BarEvent],
    symbol: str,
    base_timeframe: Timeframe,
    derived_timeframes: Sequence[Timeframe],
    days: SessionDays,
    instrument: InstrumentSpec,
    strategy: Strategy,
    window: tuple[int, int],
    latency_ms: int,
    half_spread_ticks: int,
    commission_cents_per_side_per_contract: int,
    seed: int,
    defer_protection_one_bar: bool,
    clock: Clock | None = None,
    warmup: Sequence[WarmupRequirement] = (),
    observer_factory: ObserverFactory | None = None,
    close_sink: CloseSink | None = None,
    portfolio: Portfolio | None = None,
    params: dict[str, Any] | None = None,
    journal_context: dict[str, Any] | None = None,
    journal_path: str | Path | None = None,
    journal_sinks: Sequence[Callable[[dict[str, object]], None]] = (),
    notifier: Notifier | None = None,
    notify_on: Iterable[str] | None = None,
    notify_known_kinds: Iterable[str] | None = None,
) -> RunRecord:
    """The shadow session: live bars, simulated venue, zero real orders —
    "shadow" because fills are modeled IN-PROCESS by the SimBroker, which
    is the one paper-ness the engine can truthfully claim (an order path
    to a simulated ACCOUNT is your broker adapter's business, and belongs
    on run_live with a mode label of your choosing). The strategy runs on
    the live stream exactly as backtested; brackets go to the SAME
    SimBroker, filled against the live bars — decisions play out for real
    while nothing reaches a venue. Costs are the same four explicit,
    non-defaulted parameters as run_backtest. Going real later is
    `run_live` with your own Broker adapter — the one-line swap is the
    whole design."""
    return run_live(
        stream=stream,
        symbol=symbol,
        base_timeframe=base_timeframe,
        derived_timeframes=derived_timeframes,
        days=days,
        instrument=instrument,
        broker=SimBroker(
            instrument,
            SimBrokerConfig(
                bar_period_seconds=base_timeframe.period_seconds,
                latency_ms=latency_ms,
                half_spread_ticks=half_spread_ticks,
                commission_cents_per_side_per_contract=commission_cents_per_side_per_contract,
                seed=seed,
                defer_protection_one_bar=defer_protection_one_bar,
            ),
        ),
        clock=clock if clock is not None else LiveClock(),
        strategy=strategy,
        window=window,
        warmup=warmup,
        observer_factory=observer_factory,
        close_sink=close_sink,
        portfolio=portfolio,
        params=params,
        journal_context=journal_context,
        journal_path=journal_path,
        journal_sinks=journal_sinks,
        notifier=notifier,
        notify_on=notify_on,
        notify_known_kinds=notify_known_kinds,
        mode="shadow",
    )


def install_sigint_stop(
    stream: LiveBarStream, *, on_first: Callable[[], None] | None = None
) -> Callable[[], None]:
    """First Ctrl+C: graceful stop (close the stream, let the run archive).
    Second Ctrl+C: hard abort (KeyboardInterrupt). Returns a restore
    callable; a no-op when not on the main thread (tests, embedders)."""
    if threading.current_thread() is not threading.main_thread():
        return lambda: None
    state = {"fired": False}
    previous = signal.getsignal(signal.SIGINT)

    def handler(_signum: int, _frame: FrameType | None) -> None:
        if not state["fired"]:
            state["fired"] = True
            sys.stderr.write("\n[live] stop requested — finishing up (Ctrl+C again to abort)\n")
            if on_first is not None:
                on_first()
            stream.close()
            return
        signal.signal(signal.SIGINT, previous)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handler)

    def restore() -> None:
        if signal.getsignal(signal.SIGINT) is handler:
            signal.signal(signal.SIGINT, previous)

    return restore
