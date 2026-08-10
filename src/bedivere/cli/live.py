"""`python -m bedivere.live` — run a spec against live bars.

    python -m bedivere.live --spec specs/x.json --mode shadow --until 15:55
    python -m bedivere.live --spec specs/x.json --mode paper  --notify auto

Same spec file as the backtest. That is the design goal made operational: the
window is the one thing a live run cannot inherit (a backtest's window is
history), so the runner derives its own from *now* and the session-day close,
and everything else in the file means exactly what it meant when you
backtested it.

`--mode` names the ORDER PATH:

    shadow          the in-process SimBroker. No venue is contacted; fills
                    are modelled against the live bars. This is the only
                    "no real orders" claim bedivere can make on its own.
    anything else   the spec's `broker` adapter — a real order path. The
                    label is YOURS ("paper", "funded", "sim101"), because the
                    engine knows which BROKER it was given and never which
                    ACCOUNT that broker points at. It rides into the banner,
                    the journal context, the result and the index line.

Around the run sits the operational layer a session needs and a backtest does
not: a one-run lock at the runs root, a phase-and-snapshot `ACTIVE.json` any
launcher can poll, a `STOP` sentinel for a graceful stop without a signal,
`broker.preflight()` refusing to arm into a venue this run cannot explain,
and a `finalize` on every exit path — including the failing ones — so a lock
never outlives the run that took it.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from bedivere.brokers.port import BrokerContext
from bedivere.brokers.sim import SimBroker, SimBrokerConfig
from bedivere.cli.common import (
    CliError,
    ResolvedRun,
    add_spec_arguments,
    build_notifier,
    build_source,
    emit_result,
    history_start,
    note,
    resolve_run,
    run_command,
)
from bedivere.config.resolve import ResolutionError, resolve_factory
from bedivere.config.spec import SpecError, archived_spec, resolve_live_window
from bedivere.core.clock import Clock, LiveClock
from bedivere.core.types import Candle
from bedivere.data.port import CandleSource, assess_coverage
from bedivere.engine.intents import OrderEvent
from bedivere.engine.loop import RunContext
from bedivere.engine.portfolio import Portfolio
from bedivere.notify.port import Notifier
from bedivere.run.archive import JOURNAL_NAME, run_id_for, utc_stamp, write_run
from bedivere.run.live import LiveBroker, PreflightError, install_sigint_stop, run_live
from bedivere.run.supervise import ActiveRunError, RunSupervisor
from bedivere.streams.feed import FeedAdapter, FeedContext
from bedivere.streams.live import LiveBarStream

SHADOW = "shadow"


def _utc(unix_sec: int) -> str:
    """Render a GIVEN instant (not a wall-clock read)."""
    return datetime.fromtimestamp(unix_sec, UTC).strftime("%Y-%m-%d %H:%MZ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bedivere.live",
        description="Run a spec against live bars (shadow venue by default).",
    )
    add_spec_arguments(parser)
    parser.add_argument(
        "--mode",
        default=SHADOW,
        metavar="LABEL",
        help=f'"{SHADOW}" (in-process SimBroker, no venue) or your own label for a real '
        "order path, which requires the spec's `broker` adapter",
    )
    parser.add_argument(
        "--until",
        default=None,
        metavar="HH:MM",
        help="operator stop time in the session timezone (default: the session-day close)",
    )
    parser.add_argument(
        "--notify",
        choices=["auto", "discord", "console", "off"],
        default=None,
        help="alert transport (default: the spec's notify.transport)",
    )
    parser.add_argument(
        "--notify-on",
        dest="notify_on",
        action="append",
        default=None,
        metavar="KIND",
        help="journal kind to alert on, repeatable (overrides the spec's notify.on)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_command(lambda: run(args, clock=LiveClock()))


def run(args: argparse.Namespace, *, clock: Clock) -> int:
    """The live runner proper. `clock` is injected so a test can drive a whole
    session at a chosen instant — the ONLY thing separating a test session
    from a real one."""
    overrides: list[str] = list(args.overrides)
    now = clock.now_unix()
    run_cfg = resolve_run(args.spec, overrides, now_unix=now)

    try:
        window = resolve_live_window(run_cfg.days, now_unix=now, until=args.until)
    except SpecError as e:
        raise CliError(str(e)) from e

    runs_dir: Path = args.runs_dir
    ran_at = utc_stamp(now)
    run_id = run_id_for(ran_at, run_cfg.params_hash)
    supervisor = RunSupervisor(
        runs_dir=runs_dir, run_id=run_id, run_dir=runs_dir / run_id, clock=clock
    )
    # The lock comes FIRST, before a feed is opened or a broker is built:
    # a run that may not start should contact nothing.
    try:
        supervisor.acquire(
            mode=args.mode,
            kind=run_cfg.kind,
            symbol=run_cfg.symbol,
            paramsHash=run_cfg.params_hash,
            windowStartUnix=window[0],
            windowEndUnix=window[1],
        )
    except ActiveRunError as e:
        raise CliError(str(e)) from e

    try:
        return _run_locked(args, run_cfg, supervisor, clock=clock, window=window, ran_at=ran_at)
    except BaseException as e:  # incl. KeyboardInterrupt — a dying run must release
        supervisor.finalize("failed", error=str(e) or type(e).__name__)
        raise


def _run_locked(
    args: argparse.Namespace,
    run_cfg: ResolvedRun,
    supervisor: RunSupervisor,
    *,
    clock: Clock,
    window: tuple[int, int],
    ran_at: str,
) -> int:
    runs_dir: Path = args.runs_dir
    run_dir = supervisor.run_dir
    spec = run_cfg.spec
    overrides: list[str] = list(args.overrides)

    supervisor.set_phase("connecting")
    broker = _build_broker(run_cfg, mode=args.mode, clock=clock)

    supervisor.set_phase("backfilling")
    source = build_source(run_cfg) if spec.data is not None else None
    history = _load_history(run_cfg, source, window_start=window[0])

    stream = LiveBarStream(
        symbol=run_cfg.symbol,
        timeframe=run_cfg.base_tf,
        clock=clock,
        window_end_unix=window[1],
        history=history,
        on_status=note,
    )
    feed = _build_feed(run_cfg, source, clock=clock, window=window)

    portfolio = Portfolio(spec=run_cfg.instrument)
    supervisor.on_stop_requested = stream.close
    supervisor.snapshot = lambda: {
        "trades": len(portfolio.trades),
        "openPositions": portfolio.position_count(),
        "exposure": portfolio.exposure_count(),
        "feed": {
            "historyBars": stream.history_bars,
            "liveBars": stream.live_bars,
            "backfillBars": stream.backfill_bars,
            "duplicates": stream.duplicates,
            "dropped": stream.dropped,
        },
    }

    notifier = _build_notifier(args, run_cfg)
    restore_sigint = install_sigint_stop(
        stream, on_first=lambda: supervisor.set_phase("stopping", stopRequestedBy="sigint")
    )
    supervisor.start()

    note(
        f"{args.mode} {run_cfg.kind} {run_cfg.symbol} {run_cfg.base_tf.value} · "
        f"{len(history)} history bar(s) · until {_utc(window[1])} · run {supervisor.run_id}"
    )
    try:
        supervisor.set_phase("warming")
        feed.start(stream)
        record = run_live(
            stream=stream,
            symbol=run_cfg.symbol,
            base_timeframe=run_cfg.base_tf,
            derived_timeframes=list(spec.derived_timeframes),
            days=run_cfg.days,
            instrument=run_cfg.instrument,
            broker=broker,
            clock=clock,
            strategy=_PhaseReporting(run_cfg.strategy(), supervisor),
            window=window,
            warmup=run_cfg.warmup,
            observer_factory=run_cfg.plugin.observers(run_cfg.config),
            close_sink=run_cfg.plugin.close_sink(run_cfg.config),
            portfolio=portfolio,
            params=run_cfg.config_json,
            journal_context={"kind": run_cfg.kind},
            journal_path=run_dir / JOURNAL_NAME,
            notifier=notifier,
            notify_on=_notify_kinds(args, run_cfg),
            # The strategy's own vocabulary, so `--notify-on typoed_kind`
            # fails at startup instead of alerting on nothing all session.
            notify_known_kinds=run_cfg.plugin.notify_kinds(run_cfg.config) or None,
            mode=args.mode,
        )
    except PreflightError as e:
        # The venue refused the run, which is an operator fact, not a bug —
        # one stderr line and exit 1, not a traceback.
        raise CliError(str(e)) from e
    finally:
        restore_sigint()
        feed.stop()

    result: dict[str, Any] = record.result
    write_run(
        runs_dir,
        supervisor.run_id,
        result=result,
        spec=archived_spec(spec, resolved_config=run_cfg.config_json, window=window),
        # The journal already streamed into the run directory, flushed per
        # event. Rewriting it from memory here would replace the crash-safe
        # copy with one that only exists because we survived.
        journal=None,
        ran_at_utc=ran_at,
        kind=run_cfg.kind,
        mode=args.mode,
        note=args.note or f"{args.mode} session",
        overrides=overrides,
        index_extra=run_cfg.plugin.index_columns(run_cfg.config, result),
    )
    note(f"archived -> {run_dir}")
    supervisor.finalize(
        "stopped",
        summary=result["summary"],
        stoppedBy="sentinel" if supervisor.stop_requested else "window",
    )

    summary = result["summary"]
    note(
        f"done — {summary['trades']} trade(s), net {summary['netCents'] / 100:.2f} USD"
        + (" · ⚠ a position was open at shutdown" if summary["openPositions"] else "")
    )
    emit_result(result)
    return 0


# ---------- composition pieces ----------


class _PhaseReporting:
    """Delegating strategy wrapper: identical decisions, plus the one phase
    transition the engine knows about and the supervisor does not.

    `on_start` is called by the loop at the FIRST TRADEABLE INSTANT — after
    warm-up is satisfied and the gate has passed — which is precisely the
    moment "warming" becomes "running". Deriving it any other way would be
    guessing at something the loop already knows exactly.
    """

    def __init__(self, inner: Any, supervisor: RunSupervisor) -> None:
        self._inner = inner
        self._supervisor = supervisor

    def on_start(self, ctx: RunContext) -> None:
        self._supervisor.set_phase("running", firstTradeableBarTs=ctx.bar.timestamp)
        self._inner.on_start(ctx)

    def on_bar(self, ctx: RunContext) -> None:
        self._inner.on_bar(ctx)

    def on_order_event(self, ev: OrderEvent) -> None:
        self._inner.on_order_event(ev)

    def on_stop(self) -> None:
        self._inner.on_stop()

    def __getattr__(self, name: str) -> Any:
        """Anything the engine looks up by name — `summary_jsonable` today —
        belongs to the wrapped strategy.

        DELEGATING rather than declaring is what keeps the wrapper invisible.
        A `summary_jsonable` defined here would give a strategy that has none
        an empty `"strategy"` block in its result, so the same strategy would
        produce a different envelope live than in a backtest purely because
        the live runner wraps it.
        """
        if name.startswith("_"):  # never recurse through our own attributes
            raise AttributeError(name)
        return getattr(self._inner, name)


def _build_broker(run_cfg: ResolvedRun, *, mode: str, clock: Clock) -> LiveBroker:
    """`shadow` is the in-process SimBroker; anything else is a real order
    path and therefore needs an adapter this repo does not ship."""
    if mode == SHADOW:
        sim = run_cfg.spec.sim
        return SimBroker(
            run_cfg.instrument,
            SimBrokerConfig(
                bar_period_seconds=run_cfg.base_tf.period_seconds,
                latency_ms=sim.latency_ms,
                half_spread_ticks=sim.half_spread_ticks,
                commission_cents_per_side_per_contract=sim.commission_cents_per_side_per_contract,
                seed=sim.seed,
                defer_protection_one_bar=sim.defer_protection_one_bar,
            ),
        )
    factory_spec = run_cfg.spec.broker
    if factory_spec is None:
        raise CliError(
            f'--mode "{mode}" is a REAL order path and this spec has no `broker` block. '
            f'bedivere ships no venue adapter: add {{"factory": "my_pkg:build", ...}}, '
            f'or use --mode {SHADOW} for the in-process sim venue.'
        )
    try:
        factory = resolve_factory(factory_spec.factory, what="broker.factory")
        broker: Any = factory(
            BrokerContext(
                instrument=run_cfg.instrument,
                symbol=run_cfg.symbol,
                clock=clock,
                mode=mode,
            ),
            **factory_spec.options,
        )
    except (ResolutionError, TypeError, ValueError) as e:
        raise CliError(f"broker.factory {factory_spec.factory}: {e}") from e
    for method in ("preflight", "submit_bracket", "cancel", "change", "flatten", "drain"):
        if not callable(getattr(broker, method, None)):
            raise CliError(
                f'broker.factory "{factory_spec.factory}" returned {type(broker).__name__}, '
                f"which has no .{method}() — see bedivere.brokers.port.Broker"
            )
    return cast(LiveBroker, broker)


def _build_feed(
    run_cfg: ResolvedRun,
    source: CandleSource | None,
    *,
    clock: Clock,
    window: tuple[int, int],
) -> FeedAdapter:
    factory_spec = run_cfg.spec.feed
    if factory_spec is None:
        raise CliError(
            "this spec has no `feed` block, so no bars would ever arrive. Name your feed "
            'adapter, or rehearse against history with {"factory": '
            '"bedivere.streams.feed:build_replay_feed"}.'
        )
    ctx = FeedContext(
        symbol=run_cfg.symbol,
        timeframe=run_cfg.base_tf,
        window_start_unix=window[0],
        window_end_unix=window[1],
        clock=clock,
        source=source,
    )
    try:
        factory = resolve_factory(factory_spec.factory, what="feed.factory")
        adapter: Any = factory(ctx, **factory_spec.options)
    except (ResolutionError, TypeError, ValueError) as e:
        raise CliError(f"feed.factory {factory_spec.factory}: {e}") from e
    for method in ("start", "stop"):
        if not callable(getattr(adapter, method, None)):
            raise CliError(
                f'feed.factory "{factory_spec.factory}" returned {type(adapter).__name__}, '
                f"which has no .{method}() — see bedivere.streams.feed.FeedAdapter"
            )
    return cast(FeedAdapter, adapter)


def _load_history(
    run_cfg: ResolvedRun, source: CandleSource | None, *, window_start: int
) -> list[Candle]:
    """Warm-up history: completed bars from the lookback start up to now.

    A declared lookback with nothing to satisfy it is refused HERE rather
    than by the warm-up gate mid-session. Both refusals are correct; this one
    happens before a venue has been contacted.
    """
    if not run_cfg.warmup:
        return []
    load_start = history_start(run_cfg, window_start)
    if source is None:
        raise CliError(
            f"{run_cfg.kind} declares warm-up "
            f"({', '.join(f'{r.tf.value}x{r.bars}' for r in run_cfg.warmup)}) "
            "but the spec has no `data` block to backfill it from"
        )
    bars = source.candles(run_cfg.symbol, run_cfg.base_tf, load_start, window_start)
    coverage = assess_coverage(
        bars,
        days=run_cfg.days,
        symbol=run_cfg.symbol,
        timeframe=run_cfg.base_tf,
        start_unix=load_start,
        end_unix=window_start,
    )
    note(f"backfill: {coverage.describe()} ({source.describe()})")
    if not bars:
        raise CliError(
            f"no warm-up history for {run_cfg.symbol} {run_cfg.base_tf.value} in "
            f"({load_start}, {window_start}] — the run would start unready"
        )
    return bars


def _build_notifier(args: argparse.Namespace, run_cfg: ResolvedRun) -> Notifier | None:
    transport = args.notify if args.notify is not None else run_cfg.spec.notify.transport
    return build_notifier(transport)


def _notify_kinds(args: argparse.Namespace, run_cfg: ResolvedRun) -> list[str] | None:
    if args.notify_on:
        return list(args.notify_on)
    configured = run_cfg.spec.notify.on
    return list(configured) if configured else None


if __name__ == "__main__":
    sys.exit(main())
