"""bedivere — a bar-replay trading engine that reports what actually
happened.

The quickstart surface is re-exported here; everything else lives in its
submodule and is meant to be imported from there.

    from bedivere import (
        Timeframe, spec_from_handoff, cme_futures_sessions,
        run_backtest, WarmupRequirement,
    )
"""

from bedivere.brokers.sim import SimBroker, SimBrokerConfig
from bedivere.core.clock import Clock, LiveClock, ReplayClock
from bedivere.core.pricing import (
    InstrumentSpec,
    OffGridPriceError,
    PricingError,
    spec_from_handoff,
)
from bedivere.core.session_days import SessionDay, SessionDays, parse_session_days
from bedivere.core.types import SUPPORTED_TIMEFRAMES, Candle, Timeframe
from bedivere.data import load_candles_csv
from bedivere.engine.intents import BracketIntent, OrderEvent
from bedivere.engine.journal import DecisionJournal
from bedivere.engine.loop import CloseSink, RunContext, Strategy, run_loop
from bedivere.engine.portfolio import Portfolio, TradeRecord
from bedivere.engine.warmup import WarmupError, WarmupGate, WarmupRequirement
from bedivere.run import BacktestRun, run_backtest
from bedivere.sessions import DailySessionSpec, build_session_days, cme_futures_sessions
from bedivere.stats import compute_stats
from bedivere.streams import ReplayStream
from bedivere.view.frozen import FrozenSource, build_frozen_source, build_frozen_view
from bedivere.view.market_view import CloseObserver, HtfClose, MarketView

__version__ = "0.1.0"

__all__ = [
    "SUPPORTED_TIMEFRAMES",
    "BacktestRun",
    "BracketIntent",
    "Candle",
    "Clock",
    "CloseObserver",
    "CloseSink",
    "DailySessionSpec",
    "DecisionJournal",
    "FrozenSource",
    "HtfClose",
    "InstrumentSpec",
    "LiveClock",
    "MarketView",
    "OffGridPriceError",
    "OrderEvent",
    "Portfolio",
    "PricingError",
    "ReplayClock",
    "ReplayStream",
    "RunContext",
    "SessionDay",
    "SessionDays",
    "SimBroker",
    "SimBrokerConfig",
    "Strategy",
    "Timeframe",
    "TradeRecord",
    "WarmupError",
    "WarmupGate",
    "WarmupRequirement",
    "build_frozen_source",
    "build_frozen_view",
    "build_session_days",
    "cme_futures_sessions",
    "compute_stats",
    "load_candles_csv",
    "parse_session_days",
    "run_backtest",
    "run_loop",
    "spec_from_handoff",
]
