"""run_backtest — the backtest composition root: bars + sessions +
strategy in, deterministic result out.

Everything environment-specific is wired HERE; strategy modules stay
composition-free. The backtest triad is {ReplayStream, ReplayClock,
SimBroker} — the live runner swaps exactly those three and nothing else
(see bedivere.run.live).

Costs are NOT defaulted: `latency_ms`, `half_spread_ticks`,
`commission_cents_per_side_per_contract`, `seed` and
`defer_protection_one_bar` are required keyword arguments with no
fallback. A frictionless backtest must be a choice you can see in your own
call site (`half_spread_ticks=0`), never something the library assumed for
you — and the naked-window treatment is the same kind of decision: it moves
results in BOTH directions and the right answer depends on your bar size.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bedivere.brokers.sim import SimBroker, SimBrokerConfig
from bedivere.core.clock import ReplayClock
from bedivere.core.pricing import InstrumentSpec
from bedivere.core.session_days import SessionDays
from bedivere.core.types import Candle, Timeframe
from bedivere.engine.journal import DecisionJournal
from bedivere.engine.loop import CloseSink, LoopStats, Strategy, run_loop
from bedivere.engine.portfolio import Portfolio
from bedivere.engine.warmup import WarmupGate, WarmupRequirement
from bedivere.run.record import RunRecord, build_result
from bedivere.streams.replay import ReplayStream
from bedivere.streams.sparse import (
    SparseReplayStream,
    assert_trades_covered,
    trade_coverage,
)
from bedivere.view.market_view import MarketView, ObserverFactory


def run_backtest(
    *,
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
    bars: Sequence[Candle] | None = None,
    stream: SparseReplayStream | None = None,
    warmup: Sequence[WarmupRequirement] = (),
    observer_factory: ObserverFactory | None = None,
    close_sink: CloseSink | None = None,
    params: dict[str, Any] | None = None,
    journal_context: dict[str, Any] | None = None,
    journal_path: str | Path | None = None,
    strict_fidelity: bool = True,
) -> RunRecord:
    """Wire the replay triad {ReplayStream, ReplayClock, SimBroker} around
    one strategy and drive it to completion.

    `bars` is the full base-TF series INCLUDING warm-up history; `window`
    is (tradeable_start_unix, end_unix) — bars before the start feed the
    view and the brokers' clock but never reach the strategy, and bars
    after the end must simply not be in `bars`. `params` is any dict you
    want hashed into `paramsHash` for provenance (strategy settings,
    experiment labels); it is echoed under `"params"` in the result.

    Pass `stream` INSTEAD of `bars` to replay at mixed fidelity (see
    `bedivere.streams.sparse`). The result then carries a `replay` block
    describing what was replayed finely, and with `strict_fidelity` both
    sparse guards run automatically after the loop: the derived coarse
    series must reproduce the pre-pass's trigger selection, and every trade
    must have played out inside a fine window. Turn it off only to inspect
    a run you already know is failing one of them.
    """
    if (bars is None) == (stream is None):
        raise ValueError("run_backtest: pass exactly one of `bars` or `stream`")
    window_start, window_end = window
    if window_start >= window_end:
        raise ValueError(f"window start {window_start} must be before end {window_end}")
    if instrument.symbol != symbol:
        raise ValueError(
            f'instrument is for "{instrument.symbol}" but the run is for "{symbol}" — one spec per run'
        )
    if stream is not None and stream.coarse_timeframe not in derived_timeframes:
        raise ValueError(
            f"run_backtest: a sparse stream's coarse TF {stream.coarse_timeframe.value} must be "
            "in derived_timeframes — the agreement guard reads that series from the view"
        )
    sim = SimBrokerConfig(
        bar_period_seconds=base_timeframe.period_seconds,
        latency_ms=latency_ms,
        half_spread_ticks=half_spread_ticks,
        commission_cents_per_side_per_contract=commission_cents_per_side_per_contract,
        seed=seed,
        defer_protection_one_bar=defer_protection_one_bar,
    )

    bar_source: SparseReplayStream | ReplayStream = (
        stream
        if stream is not None
        else ReplayStream(symbol=symbol, timeframe=base_timeframe, bars=bars or ())
    )
    view = MarketView(
        base_tf=base_timeframe,
        derived_tfs=derived_timeframes,
        days=days,
        observer_factory=observer_factory,
    )
    broker = SimBroker(instrument, sim)
    portfolio = Portfolio(spec=instrument)
    journal = DecisionJournal(context=dict(journal_context) if journal_context else {})
    first_ts = bar_source.first_ts
    clock = ReplayClock(first_ts if first_ts is not None else window_start)
    gate = WarmupGate(view, list(warmup)) if warmup else None

    loop_stats: LoopStats = run_loop(
        stream=bar_source,
        clock=clock,
        view=view,
        broker=broker,
        portfolio=portfolio,
        strategy=strategy,
        tradeable_start_unix=window_start,
        close_sink=close_sink,
        warmup_gate=gate,
        journal=journal,
    )

    extra: dict[str, Any] = {
        "sim": {
            "latencyMs": sim.latency_ms,
            "halfSpreadTicks": sim.half_spread_ticks,
            "commissionCentsPerSidePerContract": sim.commission_cents_per_side_per_contract,
            "seed": sim.seed,
            "deferProtectionOneBar": sim.defer_protection_one_bar,
            "deferredProtectionBars": broker.deferred_protection_bars,
            "deferredProtectionSuppressed": broker.deferred_protection_suppressed,
        }
    }
    if stream is not None:
        # Both guards run BEFORE the result is built: a run that fails one of
        # them has no numbers worth reporting, and returning them anyway is
        # how a mixed-fidelity backtest quietly becomes a claim.
        if strict_fidelity:
            stream.verify(view.completed(stream.coarse_timeframe))
            assert_trades_covered(portfolio.trades, stream.windows, stream.days)
        report = stream.report
        extra["replay"] = {
            **report.to_jsonable(),
            "strict": strict_fidelity,
            "tradesCovered": trade_coverage(
                portfolio.trades, stream.windows, stream.days
            ),
        }
    else:
        extra["replay"] = {"mode": "full", "fineTimeframe": base_timeframe.value}

    result = build_result(
        symbol=symbol,
        base_timeframe=base_timeframe,
        view=view,
        window=window,
        warmup=warmup,
        loop_stats=loop_stats,
        portfolio=portfolio,
        instrument=instrument,
        journal=journal,
        strategy=strategy,
        params=params,
        extra=extra,
    )

    if journal_path is not None:
        journal.write_jsonl(Path(journal_path))
    return RunRecord(result=result, journal=journal, portfolio=portfolio, view=view)
