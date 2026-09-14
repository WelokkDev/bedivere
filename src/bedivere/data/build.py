"""DataSpec → CandleSource: the one place a spec's `data` block becomes an
object the engine can ask for bars.

Three shapes, one port:

    {"source": "csv",  "path": "data/DEMO-5m.csv"}

    {"source": "lake", "dataset": "GLBX.MDP3", "series": "local.v.0",
                       "root": "data/lake"}

    {"source": "lake", "dataset": "GLBX.MDP3", "series": "local.v.0",
                       "adjustment": {"method": "back_adjusted",
                                      "asOf": "2026-06-16"}}

    {"source": "python", "factory": "my_pkg.feed:build_source",
                         "options": {"account": "..."}}

The `python` form takes any source at all — a vendor SDK, an internal service,
someone else's warehouse — as a factory returning a `CandleSource`, with no file
format in between and no library edit. `options` is splatted into it as keyword
arguments, so the factory declares its own parameters and rejects the ones it
does not know in its own words.

Every source that CAN report what it holds is coverage-checked by default, so a
window reaching past the end of the store raises instead of coming back short —
by session-day when the caller hands over the run's calendar, which the CLI
always does. `"allowStale": true` opts out, deliberately and visibly, in the
spec file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from bedivere.config.resolve import ResolutionError, resolve_factory
from bedivere.config.spec import DataSpec
from bedivere.core.session_days import SessionDays
from bedivere.core.types import Timeframe
from bedivere.data.csv import CsvCandleSource
from bedivere.data.port import CandleSource
from bedivere.data.source import (
    BackAdjustedLakeSource,
    CoverageCheckedSource,
    LakeCandleSource,
)


def build_candle_source(
    spec: DataSpec, *, symbol: str, timeframe: Timeframe, days: SessionDays | None = None
) -> CandleSource:
    """Resolve a spec's `data` block. `symbol`/`timeframe` are the run's, and are
    handed to the CSV source because a file cannot state its own. `days` is the
    run's session calendar; with it the coverage gate refuses by session-day,
    without it by a coarse time tolerance (see `bedivere.data.source`)."""
    if spec.source == "csv":
        return CsvCandleSource(cast(str, spec.path), symbol=symbol, timeframe=timeframe)

    if spec.source == "lake":
        lake = LakeCandleSource(
            cast(str, spec.dataset),
            cast(str, spec.series),
            Path(spec.root) if spec.root else None,
        )
        built: CandleSource = lake
        if spec.adjustment is not None:
            # AdjustmentSpec's own validator guarantees `as_of` is present.
            built = BackAdjustedLakeSource(lake, cast(str, spec.adjustment.as_of))
        return _maybe_checked(built, spec, days)

    factory = resolve_factory(cast(str, spec.factory), what="data.factory")
    source: Any = factory(**spec.options)
    if not hasattr(source, "candles"):
        raise ResolutionError(
            f'data.factory "{spec.factory}" returned {type(source).__name__}, which is not a '
            "CandleSource — it needs .candles(symbol, timeframe, start_unix, end_unix)"
        )
    return _maybe_checked(cast(CandleSource, source), spec, days)


def _maybe_checked(source: CandleSource, spec: DataSpec, days: SessionDays | None) -> CandleSource:
    """Wrap unless the spec opted out — or unless the source cannot answer.

    A source with no `latest()` has nothing to check against, and wrapping it
    would turn "cannot say how fresh it is" into a refusal on the first read.
    Right for a store that should be able to say; wrong for a `python` factory
    whose author was never asked for more than `candles()`.
    """
    if spec.allow_stale or not callable(getattr(source, "latest", None)):
        return source
    return CoverageCheckedSource(source, days)
