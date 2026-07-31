"""bedivere.engine.warmup — lookback computation + the runtime readiness gate."""

from __future__ import annotations

import pytest

from bedivere.core.types import Timeframe
from bedivere.engine.warmup import (
    WarmupError,
    WarmupGate,
    WarmupRequirement,
    lookback_load_start,
)
from bedivere.view.market_view import MarketView
from tests.helpers import bar, et, eth_session_days

# Three consecutive ETH days (23h each: 46 30m bars, 5 full 4h + 1 stub).
DAYS = eth_session_days(["2026-07-14", "2026-07-15", "2026-07-16"])


def test_requirement_validation() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        WarmupRequirement(tf=Timeframe.M30, bars=-1)
    with pytest.raises(ValueError, match="1d"):
        WarmupRequirement(tf=Timeframe.D1, bars=5)


def test_no_requirements_loads_nothing_extra() -> None:
    start = et("2026-07-16T09:30:00")
    assert lookback_load_start(DAYS, start, []) == start
    assert lookback_load_start(DAYS, start, [WarmupRequirement(Timeframe.M30, 0)]) == start


def test_partial_day_counts_only_stamps_at_or_before_start() -> None:
    # Tradeable start 09:30 ET on day 3: day 3 contributes 30m stamps
    # 18:30 ... 09:30 = 31 bars. Needing 31 → day 3's open suffices.
    start = et("2026-07-16T09:30:00")
    day3_open = et("2026-07-15T18:00:00")
    assert (
        lookback_load_start(DAYS, start, [WarmupRequirement(Timeframe.M30, 31)]) == day3_open
    )
    # One more bar forces the previous day.
    day2_open = et("2026-07-14T18:00:00")
    assert (
        lookback_load_start(DAYS, start, [WarmupRequirement(Timeframe.M30, 32)]) == day2_open
    )


def test_multi_tf_requirements_take_the_deepest() -> None:
    start = et("2026-07-16T09:30:00")
    day2_open = et("2026-07-14T18:00:00")
    got = lookback_load_start(
        DAYS,
        start,
        [WarmupRequirement(Timeframe.M30, 5), WarmupRequirement(Timeframe.H4, 8)],
    )
    # 4h: day 3 contributes stamps 22:00, 02:00, 06:00 (3 ≤ 09:30 — the
    # 10:00 grid close is after) → need 5 more → day 2 (6 bars) covers.
    assert got == day2_open


def test_exceeding_handed_days_hard_fails() -> None:
    start = et("2026-07-16T09:30:00")
    with pytest.raises(WarmupError, match="30m"):
        lookback_load_start(DAYS, start, [WarmupRequirement(Timeframe.M30, 1000)])


def test_gate_readiness_and_hard_fail() -> None:
    view = MarketView(base_tf=Timeframe.M5, derived_tfs=[Timeframe.M30], days=DAYS)
    gate = WarmupGate(
        view,
        [WarmupRequirement(Timeframe.M5, 6), WarmupRequirement(Timeframe.M30, 1)],
    )
    assert not gate.is_ready()
    assert gate.missing() == [
        "5m: have 0/6 completed bars",
        "30m: have 0/1 completed bars",
    ]
    with pytest.raises(WarmupError, match="unready"):
        gate.assert_ready_for_trading(et("2026-07-14T09:30:00"))

    t0 = et("2026-07-13T18:00:00")
    for i in range(6):  # six 5m bars complete the first 30m bucket
        view.update(bar(t0 + (i + 1) * 300, 1, 2, 0.5, 1.5))
    assert gate.is_ready()
    assert gate.missing() == []
    gate.assert_ready_for_trading(et("2026-07-14T09:30:00"))  # no raise


def test_gate_rejects_unmaintained_tf() -> None:
    view = MarketView(base_tf=Timeframe.M5, derived_tfs=[Timeframe.M30], days=DAYS)
    with pytest.raises(WarmupError, match="does not maintain"):
        WarmupGate(view, [WarmupRequirement(Timeframe.H4, 1)])
