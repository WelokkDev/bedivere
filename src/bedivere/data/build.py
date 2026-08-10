"""DataSpec → CandleSource: the one place a spec's `data` block becomes an
object the engine can ask for bars.

Three shapes, one port:

    {"source": "csv",    "path": "data/DEMO-5m.csv"}
    {"source": "sqlite", "path": "data/candles.db"}
    {"source": "python", "factory": "my_pkg.feed:build_source",
                         "options": {"account": "..."}}

The `python` form takes any source at all — a vendor SDK, an internal
service, a parquet lake — as a factory returning a `CandleSource`, with no
file format in between and no library edit. `options` is splatted into it as
keyword arguments, so the factory
declares its own parameters and rejects the ones it does not know in its own
words.

SQLite is opened READ-ONLY here. A run reads bars; the thing that writes them
is the import command, deliberately a separate act.
"""

from __future__ import annotations

from typing import Any, cast

from bedivere.config.resolve import ResolutionError, resolve_factory
from bedivere.config.spec import DataSpec
from bedivere.core.types import Timeframe
from bedivere.data.csv import CsvCandleSource
from bedivere.data.port import CandleSource
from bedivere.data.sqlite import SqliteCandleStore


def build_candle_source(
    spec: DataSpec, *, symbol: str, timeframe: Timeframe
) -> CandleSource:
    """Resolve a spec's `data` block. `symbol`/`timeframe` are the run's, and
    are handed to the CSV source because a file cannot state its own."""
    if spec.source == "csv":
        return CsvCandleSource(cast(str, spec.path), symbol=symbol, timeframe=timeframe)
    if spec.source == "sqlite":
        return SqliteCandleStore(cast(str, spec.path), read_only=True)
    factory = resolve_factory(cast(str, spec.factory), what="data.factory")
    source: Any = factory(**spec.options)
    if not hasattr(source, "candles"):
        raise ResolutionError(
            f'data.factory "{spec.factory}" returned {type(source).__name__}, which is not a '
            "CandleSource — it needs .candles(symbol, timeframe, start_unix, end_unix)"
        )
    return cast(CandleSource, source)
