"""Data: the CandleSource port and the sources that implement it.

    port.py    the protocol + honest coverage reporting
    csv.py     load_candles_csv (the zero-setup path) + CsvCandleSource
    sqlite.py  SqliteCandleStore — one file, upsertable, read-only for runs
    build.py   a spec's `data` block -> a CandleSource

Nothing downstream depends on any of these concretely; the engine only ever
sees the protocol. `build` is deliberately NOT re-exported here — it reaches
up into bedivere.config, and importing it from the package root would make
every `from bedivere.data import ...` drag the config layer in with it.
"""

from bedivere.data.csv import CsvCandleSource, load_candles_csv, parse_timestamp
from bedivere.data.port import (
    CandleSource,
    Coverage,
    Gap,
    assess_coverage,
    expected_stamps_in_range,
)
from bedivere.data.sqlite import (
    CandleStoreError,
    Series,
    SqliteCandleStore,
    UpsertStats,
    import_candles,
)

__all__ = [
    "CandleSource",
    "CandleStoreError",
    "Coverage",
    "CsvCandleSource",
    "Gap",
    "Series",
    "SqliteCandleStore",
    "UpsertStats",
    "assess_coverage",
    "expected_stamps_in_range",
    "import_candles",
    "load_candles_csv",
    "parse_timestamp",
]
