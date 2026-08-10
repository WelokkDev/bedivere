"""bedivere — a bar-replay trading engine that reports what actually
happened.

The quickstart surface is re-exported here; everything else lives in its
submodule and is meant to be imported from there.

    from bedivere import (
        Timeframe, spec_from_handoff, cme_futures_sessions,
        run_backtest, WarmupRequirement,
    )
"""

from bedivere.brokers.port import BrokerContext
from bedivere.brokers.sim import SimBroker, SimBrokerConfig
from bedivere.config.base import (
    ConfigError,
    StrategyConfig,
    canonical_json,
    config_jsonable,
    load_config,
    params_hash,
    parse_override,
)
from bedivere.config.resolve import (
    ResolutionError,
    StrategyContext,
    StrategyPlugin,
    resolve_strategy,
)
from bedivere.config.spec import RunSpec, SpecError, load_spec, parse_spec
from bedivere.core.clock import Clock, LiveClock, ReplayClock
from bedivere.core.pricing import (
    InstrumentSpec,
    OffGridPriceError,
    PricingError,
    price_to_ticks,
    spec_from_handoff,
    ticks_to_price,
)
from bedivere.core.session_days import SessionDay, SessionDays, parse_session_days
from bedivere.core.types import SUPPORTED_TIMEFRAMES, Candle, Timeframe
from bedivere.data.csv import CsvCandleSource, load_candles_csv
from bedivere.data.port import CandleSource, Coverage, assess_coverage
from bedivere.data.sqlite import SqliteCandleStore
from bedivere.engine.intents import (
    BracketIntent,
    CancelReason,
    Direction,
    EntryOrder,
    EntryType,
    OrderEvent,
    marketable_entry_reason,
    resting_limit_ticks,
    resting_stop_ticks,
    rr_target_ticks,
)
from bedivere.engine.journal import DecisionJournal
from bedivere.engine.loop import CloseSink, RunContext, Strategy, run_loop
from bedivere.engine.metrics import compute_metrics
from bedivere.engine.portfolio import Portfolio, TradeRecord
from bedivere.engine.warmup import WarmupError, WarmupGate, WarmupRequirement
from bedivere.notify.discord import DiscordNotifier
from bedivere.notify.format import format_event
from bedivere.notify.port import ConsoleNotifier, Notifier, NullNotifier
from bedivere.notify.router import DEFAULT_KINDS, VENUE_KINDS, NotificationRouter
from bedivere.run.archive import read_index, write_run
from bedivere.run.backtest import run_backtest
from bedivere.run.live import (
    LiveBroker,
    PreflightError,
    install_sigint_stop,
    run_live,
    run_shadow,
)
from bedivere.run.record import RunRecord, decisions_digest
from bedivere.run.supervise import ActiveRun, ActiveRunError, RunSupervisor
from bedivere.sessions import DailySessionSpec, build_session_days, cme_futures_sessions
from bedivere.streams.feed import FeedAdapter, FeedContext, ReplayFeed
from bedivere.streams.live import LiveBarStream
from bedivere.streams.replay import ReplayStream
from bedivere.streams.sparse import (
    AgreementError,
    CoverageError,
    FineLoader,
    FineWindow,
    ReplayFidelity,
    SparseReplayError,
    SparseReplayStream,
    TriggerRule,
    assert_trades_covered,
    build_fine_windows,
    select_triggers,
    trade_coverage,
    verify_agreement,
)
from bedivere.view.frozen import FrozenSource, build_frozen_source, build_frozen_view
from bedivere.view.market_view import CloseObserver, HtfClose, MarketView

__version__ = "0.1.0"

__all__ = [
    "ActiveRun",
    "ActiveRunError",
    "BracketIntent",
    "BrokerContext",
    "CancelReason",
    "Candle",
    "CandleSource",
    "Clock",
    "CloseObserver",
    "CloseSink",
    "ConfigError",
    "ConsoleNotifier",
    "Coverage",
    "CsvCandleSource",
    "DEFAULT_KINDS",
    "DailySessionSpec",
    "DecisionJournal",
    "Direction",
    "DiscordNotifier",
    "EntryOrder",
    "EntryType",
    "FeedAdapter",
    "FeedContext",
    "FrozenSource",
    "HtfClose",
    "InstrumentSpec",
    "LiveBarStream",
    "LiveBroker",
    "LiveClock",
    "MarketView",
    "NotificationRouter",
    "Notifier",
    "NullNotifier",
    "OffGridPriceError",
    "OrderEvent",
    "Portfolio",
    "PreflightError",
    "PricingError",
    "ReplayClock",
    "ReplayFeed",
    "ReplayStream",
    "ResolutionError",
    "RunContext",
    "RunRecord",
    "RunSpec",
    "RunSupervisor",
    "SUPPORTED_TIMEFRAMES",
    "SessionDay",
    "SessionDays",
    "SimBroker",
    "SimBrokerConfig",
    "SpecError",
    "SqliteCandleStore",
    "Strategy",
    "StrategyConfig",
    "StrategyContext",
    "StrategyPlugin",
    "Timeframe",
    "TradeRecord",
    "VENUE_KINDS",
    "WarmupError",
    "WarmupGate",
    "WarmupRequirement",
    "assess_coverage",
    "build_frozen_source",
    "build_frozen_view",
    "build_session_days",
    "canonical_json",
    "cme_futures_sessions",
    "AgreementError",
    "CoverageError",
    "FineLoader",
    "FineWindow",
    "ReplayFidelity",
    "SparseReplayError",
    "SparseReplayStream",
    "TriggerRule",
    "assert_trades_covered",
    "build_fine_windows",
    "compute_metrics",
    "decisions_digest",
    "select_triggers",
    "trade_coverage",
    "verify_agreement",
    "config_jsonable",
    "format_event",
    "install_sigint_stop",
    "load_candles_csv",
    "load_config",
    "load_spec",
    "marketable_entry_reason",
    "params_hash",
    "parse_override",
    "parse_session_days",
    "parse_spec",
    "price_to_ticks",
    "read_index",
    "resolve_strategy",
    "resting_limit_ticks",
    "resting_stop_ticks",
    "rr_target_ticks",
    "run_backtest",
    "run_live",
    "run_loop",
    "run_shadow",
    "spec_from_handoff",
    "ticks_to_price",
    "write_run",
]
