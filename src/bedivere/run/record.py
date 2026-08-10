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
from bedivere.run.archive import write_run_files
from bedivere.view.market_view import MarketView


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One finished run: the result envelope plus live access to the
    journal, portfolio, and view."""

    result: dict[str, Any]
    journal: DecisionJournal
    portfolio: Portfolio
    view: MarketView

    def write(self, directory: str | Path, *, spec: dict[str, Any] | None = None) -> Path:
        """Archive the run: result.json + journal.jsonl under `directory`
        (created if needed), plus spec.json when a resolved spec is supplied.
        Both files are deterministic — diff two runs directly. Returns the
        directory.

        A thin wrapper over `bedivere.run.archive.write_run_files`, so a
        hand-composed script and the CLI produce byte-identical directories
        and the crash-safe write lives in one place. The CLI additionally
        appends to the runs index; a library caller who names their own
        directory has opted out of that, which is why this is the narrower
        of the two entry points.
        """
        return write_run_files(
            Path(directory), result=self.result, journal=self.journal, spec=spec
        )


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
        "metrics": compute_metrics(portfolio.trades, instrument, days=view.days),
        # sha256 folds the journal into resultHash: one hash, both artifacts.
        "journal": {"events": len(journal.events), "sha256": journal.sha256_hex()},
    }
    result.update(extra)
    if params is not None:
        result["params"] = params
        result["paramsHash"] = canonical_sha256(params)
    strategy_summary = getattr(strategy, "summary_jsonable", None)
    if callable(strategy_summary):
        result["strategy"] = strategy_summary()
    result["decisionsHash"] = decisions_digest(result)
    result["resultHash"] = canonical_sha256(result)
    return result


# What a run DECIDED and what came of it, as opposed to how it was executed.
# Deliberately excludes every environment block (sim costs, feed counters,
# the replay fidelity report) and every count of the machinery (`loop.bars`
# differs between a coarse and a fine replay of the same session by three
# orders of magnitude, and means nothing about the trading).
#
# `journal` is in here via its sha256, which makes this a far stronger claim
# than a trade-list comparison: two runs matching on this hash agree line for
# line about every signal seen, every rejection and its named reason, and
# every venue event — not merely about what they ended up trading.
_DECISION_KEYS = ("trades", "summary", "strategy", "journal")


def decisions_digest(result: dict[str, Any]) -> str:
    """Fingerprint of the decisions and outcomes only.

    `resultHash` answers "is this the same run?" and must change when the
    environment changes — that is what makes it a determinism gate. But two
    runs can legitimately differ in environment while being required to make
    IDENTICAL decisions: a sparse replay against a full one, a refactor that
    changes only a counter's name. `decisionsHash` is the comparison for
    those, and keeping it a separate field rather than narrowing `resultHash`
    means a fidelity label still shows up as a changed run.
    """
    return canonical_sha256({k: result[k] for k in _DECISION_KEYS if k in result})


def canonical_sha256(payload: dict[str, Any]) -> str:
    """Canonical-JSON sha256 — sorted keys, tight separators, utf-8. Raises
    on non-serializable values: a result that cannot be canonicalized cannot
    be compared, and that must be loud."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
