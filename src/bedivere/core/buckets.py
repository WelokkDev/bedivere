"""Session-aligned bucket arithmetic — the ONE home.

Three consumers (the aggregator, the frozen view, the incremental
MarketView) all import this math from here; the frozen-view/market-view
equivalence tests pin its behavior.

Both functions are pure arithmetic on ALREADY-RESOLVED session-day bounds
(bedivere.core.session_days) — no session policy lives here.
"""

from __future__ import annotations


def bucket_index_of(timestamp: int, session_day_start_unix: int, period_seconds: int) -> int:
    """Bucket index for a close-stamped bar within its session-day. The `- 1`
    is load-bearing: it pulls a boundary close-stamp into the bucket whose
    data window it ends, not the next one. DO NOT REMOVE."""
    return (timestamp - session_day_start_unix - 1) // period_seconds


def bucket_period_end(
    session_day_start_unix: int,
    session_day_end_unix: int,
    bucket_index: int,
    period_seconds: int,
) -> int:
    """A bucket's period-end: its natural grid close clamped to the
    session-day end (the trailing stub bucket ends at the session close)."""
    return min(
        session_day_start_unix + (bucket_index + 1) * period_seconds,
        session_day_end_unix,
    )
