"""Data: the CandleSource port, and the stores that implement it.

    port.py     the protocol, the optional inspection surface, and honest
                coverage reporting against session geometry
    csv.py      load_candles_csv (the zero-setup path) + CsvCandleSource
    source.py   the lake as a CandleSource, plus the gates a run passes first
    quality.py  a vendor's per-day quality feed, reconciled onto session-days
    lake/       the Parquet bar lake itself — layout, schema, the write seam,
                reads, resampling, DBN ingest, back-adjustment, SQL
    vendor/     one package per data provider; the lake speaks none of their
                vocabulary
    build.py    a spec's `data` block -> a CandleSource

Nothing downstream depends on any of these concretely; the engine only ever
sees `CandleSource`. `build` and `lake` are deliberately NOT re-exported here:
`build` reaches up into bedivere.config, and `lake` imports DuckDB at module
scope, so a clone that only replays a CSV must not need a native wheel to
`import bedivere`. Reach the lake as `bedivere.data.lake.<module>`, or through
`source.py`, which imports it lazily.
"""

from bedivere.data.csv import CsvCandleSource, load_candles_csv, parse_timestamp
from bedivere.data.port import (
    CandleSource,
    Coverage,
    Gap,
    InspectableSource,
    assess_coverage,
    expected_stamps_in_range,
)
from bedivere.data.quality import (
    DataQualityError,
    SessionDayQuality,
    assess,
    clean_labels,
    require_clean,
    utc_dates_of,
)
from bedivere.data.source import (
    BackAdjustedLakeSource,
    BarSourceError,
    CoverageCheckedSource,
    LakeCandleSource,
    require_coverage,
    require_timeframes,
)

__all__ = [
    "BackAdjustedLakeSource",
    "BarSourceError",
    "CandleSource",
    "Coverage",
    "CoverageCheckedSource",
    "CsvCandleSource",
    "DataQualityError",
    "Gap",
    "InspectableSource",
    "LakeCandleSource",
    "SessionDayQuality",
    "assess",
    "assess_coverage",
    "clean_labels",
    "expected_stamps_in_range",
    "load_candles_csv",
    "parse_timestamp",
    "require_clean",
    "require_coverage",
    "require_timeframes",
    "utc_dates_of",
]
