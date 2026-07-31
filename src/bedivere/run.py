"""run_backtest — the composition root: bars + sessions + strategy in,
deterministic result out.

Everything environment-specific is wired HERE; strategy modules stay
composition-free. The result dict is the run-twice determinism gate: no
wall-clock field exists anywhere in it, so the same bars + the same config
produce a byte-identical `resultHash` — pin one in a test and refactor
fearlessly.

Costs are NOT defaulted: `latency_ms`, `half_spread_ticks`,
`commission_cents_per_side_per_contract`, and `seed` are required keyword
arguments with no fallback. A frictionless backtest must be a choice you
can see in your own call site (`half_spread_ticks=0`), never something the
library assumed for you.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
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
from bedivere.stats import compute_stats
from bedivere.streams import ReplayStream
from bedivere.view.market_view import MarketView, ObserverFactory


@dataclass(frozen=True, slots=True)
class BacktestRun:
    """One finished run: the deterministic result envelope plus live access
    to the journal, portfolio, and view for programmatic digging."""

    result: dict[str, Any]
    journal: DecisionJournal
    portfolio: Portfolio
    view: MarketView

    def write(self, directory: str | Path) -> Path:
        """Archive the run: result.json + journal.jsonl under `directory`
        (created if needed). Both files are deterministic — diff two runs
        directly. Returns the directory."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text(
            json.dumps(self.result, sort_keys=True, indent=2), encoding="utf-8"
        )
        self.journal.write_jsonl(out / "journal.jsonl")
        return out


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
) -> BacktestRun:
    """Wire the replay triad {ReplayStream, ReplayClock, SimBroker} around
    one strategy and drive it to completion.

    `bars` is the full base-TF series INCLUDING warm-up history; `window`
    is (tradeable_start_unix, end_unix) — bars before the start feed the
    view and the brokers' clock but never reach the strategy, and bars
    after the end must simply not be in `bars`. `params` is any dict you
    want hashed into `paramsHash` for provenance (strategy settings,
    experiment labels); it is echoed under `"params"` in the result.

    If the strategy object has a `summary_jsonable()` method, its output is
    included under `"strategy"` — put your own counters there.
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

    result: dict[str, Any] = {
        "symbol": symbol,
        "baseTimeframe": base_timeframe.value,
        "derivedTimeframes": [tf.value for tf in view.derived_tfs],
        "window": {"startUnix": window_start, "endUnix": window_end},
        "sim": {
            "latencyMs": sim.latency_ms,
            "halfSpreadTicks": sim.half_spread_ticks,
            "commissionCentsPerSidePerContract": sim.commission_cents_per_side_per_contract,
            "seed": sim.seed,
        },
        "warmup": {r.tf.value: r.bars for r in warmup},
        "loop": {
            "bars": loop_stats.bars,
            "suppressedBars": loop_stats.suppressed_bars,
            "htfCloses": loop_stats.htf_closes,
            "venueEvents": loop_stats.venue_events,
            "firstTradedBarTs": loop_stats.first_traded_bar_ts,
        },
        "trades": [t.to_jsonable(instrument) for t in portfolio.trades],
        "summary": portfolio.summary_jsonable(),
        "stats": compute_stats(portfolio.trades, instrument),
        "journal": {"events": len(journal.events)},
    }
    if params is not None:
        result["params"] = params
        result["paramsHash"] = _sha256_of(params)
    strategy_summary = getattr(strategy, "summary_jsonable", None)
    if callable(strategy_summary):
        result["strategy"] = strategy_summary()

    result["resultHash"] = _sha256_of(result)

    if journal_path is not None:
        journal.write_jsonl(Path(journal_path))
    return BacktestRun(result=result, journal=journal, portfolio=portfolio, view=view)


def _sha256_of(payload: dict[str, Any]) -> str:
    """Canonical-JSON sha256 — sorted keys, tight separators, utf-8. Raises
    on non-serializable values: a result that cannot be canonicalized cannot
    be compared, and that must be loud."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
