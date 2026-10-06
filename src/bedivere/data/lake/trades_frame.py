"""A pandas view of explicitly bounded trade selections.

A frame holds every selected record in memory, so the loader needs a bound;
stream anything larger with `trades.read_trades`. Timestamps and fixed-point
prices stay integers, and undefined values are nulls, never sentinels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd  # pyright: ignore[reportMissingTypeStubs]

from bedivere.core.session_days import SessionDays
from bedivere.data.lake.ingest import PRICE_SCALE
from bedivere.data.lake.layout import SeriesId
from bedivere.data.lake.provenance import ProvenanceError, require_explicit
from bedivere.data.lake.trades import Aggressor, read_trades

# pandas is optional and shipped without complete typing for dynamic columns.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false

TRADE_COLUMNS: Final[dict[str, str]] = {
    "ordinal": "int64",
    "ts_recv": "int64",
    "ts_event": "Int64",
    "session": "object",
    "instrument_id": "uint32",
    "publisher_id": "uint16",
    "contract": "object",
    "price_fixed": "Int64",
    "price": "Float64",
    "size": "int64",
    "sequence": "uint32",
    "side_code": "object",
    "aggressor": "category",
    "flags": "uint8",
    "issues": "uint8",
    "eligible": "bool",
}


def load_trade_frame(
    archive: Path,
    series: SeriesId,
    *,
    sessions: SessionDays | None = None,
    start_ns: int | None = None,
    end_ns: int | None = None,
    lake_root: Path | None = None,
    feed: str | None = None,
    receipt: Path | None = None,
) -> pd.DataFrame:
    """Every selected record, flagged ones included, in archive order.

    Selection is `TradeArchive.read`'s. `attrs` carry `trade_source`,
    `price_scale` and `receive_window`; with a calendar and `feed=` (the
    caller's attestation), also the `provenance` research joins require.
    """
    if sessions is None and (start_ns is None or end_ns is None):
        raise ValueError(
            "load_trade_frame needs an explicit bound: sessions=, or both start_ns= and "
            "end_ns=; stream larger selections with read_trades"
        )
    if feed is not None:
        require_explicit("feed", feed)
        if sessions is None:
            raise ProvenanceError(
                "feed= attests join provenance, which also needs sessions=; omit feed for a "
                "calendar-free read"
            )
    columns: dict[str, list[object]] = {name: [] for name in TRADE_COLUMNS}
    with read_trades(
        archive,
        series,
        sessions=sessions,
        start_ns=start_ns,
        end_ns=end_ns,
        lake_root=lake_root,
        receipt=receipt,
    ) as read:
        for batch in read:
            for trade in batch:
                columns["ordinal"].append(trade.ordinal)
                columns["ts_recv"].append(trade.ts_recv)
                columns["ts_event"].append(trade.ts_event)
                columns["session"].append(trade.session)
                columns["instrument_id"].append(trade.instrument_id)
                columns["publisher_id"].append(trade.publisher_id)
                columns["contract"].append(trade.contract)
                columns["price_fixed"].append(trade.price_fixed)
                columns["price"].append(trade.price)
                columns["size"].append(trade.size)
                columns["sequence"].append(trade.sequence)
                columns["side_code"].append(trade.side_code)
                columns["aggressor"].append(trade.aggressor.value)
                columns["flags"].append(trade.flags)
                columns["issues"].append(int(trade.issues))
                columns["eligible"].append(trade.eligible)
        source = read.source
        provenance = read.provenance(feed) if feed is not None else None
    data: dict[str, object] = {}
    for name, dtype in TRADE_COLUMNS.items():
        values = columns[name]
        if dtype == "object":
            data[name] = pd.Series(values, dtype="object")
        elif dtype == "category":
            data[name] = pd.Categorical(values, categories=[side.value for side in Aggressor])
        elif dtype[0].isupper():
            # Nullable extension arrays: built from Python ints, never via float.
            data[name] = pd.array(values, dtype=dtype)
        else:
            data[name] = np.asarray(values, dtype=dtype)
    frame = pd.DataFrame(data)
    frame.attrs.update(
        {
            "trade_source": source,
            "price_scale": PRICE_SCALE,
            "receive_window": (start_ns, end_ns),
        }
    )
    if provenance is not None:
        frame.attrs["provenance"] = provenance
    return frame
