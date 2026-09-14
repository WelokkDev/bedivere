"""The bar lake — a vendor-native, research-grade bar store.

One Parquet file per (series, timeframe, session-day), laid out Hive-style so
DuckDB discovers the partitions unaided and reads them in place. Three
properties fall out of that shape, and they are the whole reason the lake
exists:

  * Provenance is the PATH. The feed, the contract-selection rule and the
    timeframe are directory levels, not columns, so two different price series
    cannot land in the same place and be concatenated by accident.
  * Partitions are immutable and re-derivable from an archived `.dbn.zst` —
    directly, or by resampling one that was. Nothing appends.
  * Storage is raw. The back-adjusted continuous series everyone actually
    backtests on is a read-time derivation (`adjust.py`), pinned to an as-of
    date, never a second store that can drift from the first.

`read.read_bars` hands back the same `list[Candle]` under the same half-open
`(start, end]` convention as every other `CandleSource`, so moving a backtest
onto the lake changes the spec's `data` block and nothing else. Every module
here imports the `lake` extra at module scope; `bedivere.data.source` is the
seam that turns a missing extra into a sentence instead of a traceback.
"""
