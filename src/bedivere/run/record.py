"""RunRecord and the result envelope — the uniform product of every run.

Backtest, shadow and live sessions all return the same record: a JSON
result envelope for machines, plus live access to the journal, portfolio
and view for programmatic digging. The envelope core is built HERE, once,
so a live run is comparable to the backtest that promoted it field by
field — runners add only their environment block (sim costs, or
mode/feed/notify counters) on top.

The result dict is the run-twice determinism gate: no wall-clock field
exists anywhere in it, so the same bars + the same config produce a
byte-identical `resultHash` — pin one in a test and refactor fearlessly.
(Live results carry real receive stamps, so THEIR hash fingerprints the
one session rather than reproducing.)
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bedivere.core.pricing import InstrumentSpec
from bedivere.core.types import Timeframe
from bedivere.engine.journal import DecisionJournal
from bedivere.engine.loop import LoopStats
from bedivere.engine.metrics import compute_metrics
from bedivere.engine.portfolio import Portfolio
from bedivere.engine.warmup import WarmupRequirement
from bedivere.view.market_view import MarketView


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One finished run: the result envelope plus live access to the
    journal, portfolio, and view."""

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


def build_result(
    *,
    symbol: str,
    base_timeframe: Timeframe,
    view: MarketView,
    window: tuple[int, int],
    warmup: Sequence[WarmupRequirement],
    loop_stats: LoopStats,
    portfolio: Portfolio,
    instrument: InstrumentSpec,
    journal: DecisionJournal,
    strategy: object,
    params: dict[str, Any] | None,
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Assemble and seal one result envelope. `extra` is the runner's
    environment block (backtest: sim costs; live: mode + feed/notify
    counters) — it is hashed like everything else. If the strategy object
    has a `summary_jsonable()` method, its output is included under
    `"strategy"` — put your own counters there."""
    result: dict[str, Any] = {
        "symbol": symbol,
        "baseTimeframe": base_timeframe.value,
        "derivedTimeframes": [tf.value for tf in view.derived_tfs],
        "window": {"startUnix": window[0], "endUnix": window[1]},
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
        "metrics": compute_metrics(portfolio.trades, instrument),
        "journal": {"events": len(journal.events)},
    }
    result.update(extra)
    if params is not None:
        result["params"] = params
        result["paramsHash"] = canonical_sha256(params)
    strategy_summary = getattr(strategy, "summary_jsonable", None)
    if callable(strategy_summary):
        result["strategy"] = strategy_summary()
    result["resultHash"] = canonical_sha256(result)
    return result


def canonical_sha256(payload: dict[str, Any]) -> str:
    """Canonical-JSON sha256 — sorted keys, tight separators, utf-8. Raises
    on non-serializable values: a result that cannot be canonicalized cannot
    be compared, and that must be loud."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
