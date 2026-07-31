"""bedivere.core.clock — the injected deterministic clock."""

from __future__ import annotations

import time

import pytest

from bedivere.core.clock import Clock, LiveClock, ReplayClock


def test_replay_clock_starts_at_start() -> None:
    c = ReplayClock(1_784_930_400)
    assert c.now_unix() == 1_784_930_400
    assert c.now_ms() == 1_784_930_400_000


def test_replay_clock_advances_monotonically() -> None:
    c = ReplayClock(100)
    c.advance(150)
    assert c.now_unix() == 150
    c.advance(150)  # equal instants are legal (several events per instant)
    assert c.now_unix() == 150
    c.advance(151)
    assert c.now_unix() == 151


def test_replay_clock_rejects_regression() -> None:
    c = ReplayClock(100)
    c.advance(200)
    with pytest.raises(ValueError, match="regression"):
        c.advance(199)
    # State survives the failed advance.
    assert c.now_unix() == 200


def test_live_clock_reads_wall_clock() -> None:
    c = LiveClock()
    before = time.time()
    unix = c.now_unix()
    ms = c.now_ms()
    after = time.time()
    assert int(before) - 1 <= unix <= int(after) + 1
    assert abs(ms / 1000 - unix) < 2


def test_both_satisfy_the_protocol() -> None:
    # Static check: assignment to the protocol type must typecheck.
    replay: Clock = ReplayClock(0)
    live: Clock = LiveClock()
    assert replay.now_unix() == 0
    assert live.now_unix() > 0
