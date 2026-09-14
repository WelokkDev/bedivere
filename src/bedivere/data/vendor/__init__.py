"""Vendor edges — everything that knows one data provider's vocabulary.

The lake proper (`bedivere.data.lake`) speaks only `SeriesId`, `Timeframe` and
`Candle`. A second vendor is a second package beside this one, and the lake
does not change to accommodate it.
"""
