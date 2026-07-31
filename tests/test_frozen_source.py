"""FrozenSource vs per-instant builds — the build-once-slice optimization
must be a pure perf refactor: source.view_at(asOf) byte-equal to
build_frozen_view for a diverse asOf sweep across a session boundary and a
maintenance break."""

from __future__ import annotations

import math

from bedivere.core.types import Candle, Timeframe
from bedivere.view.frozen import build_frozen_source, build_frozen_view
from tests.helpers import candles_equal, eth_session_days

STEP = 300
BASE = 1_700_000_000
TFS = [Timeframe.M15, Timeframe.M30, Timeframe.H1, Timeframe.H2, Timeframe.H4]


def series(n: int) -> list[Candle]:
    out: list[Candle] = []
    prev = 18000.0
    for i in range(n):
        ts = BASE + i * STEP
        close = 18000 + i * 0.1 + 8 * math.sin(i / 15)
        open_ = prev
        out.append(
            Candle(
                timestamp=ts,
                open=open_,
                high=max(open_, close) + 2,
                low=min(open_, close) - 2,
                close=close,
                volume=1000 + (i % 10) * 50,
            )
        )
        prev = close
    return out


def test_source_equals_view_across_sweep() -> None:
    # BASE is Tue 2023-11-14 17:13 ET (inside the maintenance break); the
    # 300-bar sweep crosses into the Wed and Thu ETH session-days.
    days = eth_session_days(["2023-11-15", "2023-11-16"])
    full = series(300)  # ~25h: session boundary + maintenance break
    source = build_frozen_source(full, TFS, days)

    def check(as_of: int) -> None:
        a = source.view_at(as_of)
        b = build_frozen_view(full, as_of, TFS, days)
        assert candles_equal(a.primary, b.primary), f"primary @ {as_of}"
        for tf in TFS:
            assert candles_equal(a.completed[tf], b.completed[tf]), f"completed {tf} @ {as_of}"
            assert candles_equal(a.as_of_view[tf], b.as_of_view[tf]), f"asOfView {tf} @ {as_of}"

    check(BASE - 100)  # before the first bar
    for i in range(0, len(full), 11):
        check(full[i].timestamp)  # bar closes
    for i in range(7, len(full), 31):
        ts = full[i].timestamp
        check(ts - 1)
        check(ts + 1)
        check(ts + 150)
    check(full[-1].timestamp + 100_000)  # after the last bar
