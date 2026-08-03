"""run_backtest — the backtest composition root: bars + sessions +
strategy in, deterministic result out.

Everything environment-specific is wired HERE; strategy modules stay
composition-free. The backtest triad is {ReplayStream, ReplayClock,
SimBroker} — the live runner swaps exactly those three and nothing else
(see bedivere.run.live).

Costs are NOT defaulted: `latency_ms`, `half_spread_ticks`,
`commission_cents_per_side_per_contract`, and `seed` are required keyword
arguments with no fallback. A frictionless backtest must be a choice you
can see in your own call site (`half_spread_ticks=0`), never something the
library assumed for you.
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
from bedivere.view.market_view import MarketView, ObserverFactory


def run_backtest(
    *,
    bars: Sequence[Candle],
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
    warmup: Sequence[WarmupRequirement] = (),
    observer_factory: ObserverFactory | None = None,
    registry: CloseSink | None = None,
    params: dict[str, Any] | None = None,
    journal_context: dict[str, Any] | None = None,
    journal_path: str | Path | None = None,
) -> RunRecord:
    """Wire the replay triad {ReplayStream, ReplayClock, SimBroker} around
    one strategy and drive it to completion.

    `bars` is the full base-TF series INCLUDING warm-up history; `window`
    is (tradeable_start_unix, end_unix) — bars before the start feed the
    view and the brokers' clock but never reach the strategy, and bars
    after the end must simply not be in `bars`. `params` is any dict you
    want hashed into `paramsHash` for provenance (strategy settings,
    experiment labels); it is echoed under `"params"` in the result.
    """
    window_start, window_end = window
    if window_start >= window_end:
        raise ValueError(f"window start {window_start} must be before end {window_end}")
    if instrument.symbol != symbol:
        raise ValueError(
            f'instrument is for "{instrument.symbol}" but the run is for "{symbol}" — one spec per run'
        )
    sim = SimBrokerConfig(
        bar_period_seconds=base_timeframe.period_seconds,
        latency_ms=latency_ms,
        half_spread_ticks=half_spread_ticks,
        commission_cents_per_side_per_contract=commission_cents_per_side_per_contract,
        seed=seed,
    )

    stream = ReplayStream(symbol=symbol, timeframe=base_timeframe, bars=bars)
    view = MarketView(
        base_tf=base_timeframe,
        derived_tfs=derived_timeframes,
        days=days,
        observer_factory=observer_factory,
    )
    broker = SimBroker(instrument, sim)
    portfolio = Portfolio(spec=instrument)
    journal = DecisionJournal(context=dict(journal_context) if journal_context else {})
    clock = ReplayClock(bars[0].timestamp if bars else window_start)
    gate = WarmupGate(view, list(warmup)) if warmup else None

    loop_stats: LoopStats = run_loop(
        stream=stream,
        clock=clock,
        view=view,
        broker=broker,
        portfolio=portfolio,
        strategy=strategy,
        tradeable_start_unix=window_start,
        registry=registry,
        warmup_gate=gate,
        journal=journal,
    )

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
        extra={
            "sim": {
                "latencyMs": sim.latency_ms,
                "halfSpreadTicks": sim.half_spread_ticks,
                "commissionCentsPerSidePerContract": sim.commission_cents_per_side_per_contract,
                "seed": sim.seed,
            }
        },
    )

    if journal_path is not None:
        journal.write_jsonl(Path(journal_path))
    return RunRecord(result=result, journal=journal, portfolio=portfolio, view=view)
