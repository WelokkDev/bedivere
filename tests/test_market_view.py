"""MarketView vs the frozen view — the push/pull equivalence oracle.

The incremental view must agree with FrozenSource.view_at (the pinned
pull-shaped construction) at EVERY bar instant: completed cut, forming
bucket, and primary — including gaps, missing boundary bars, out-of-session
bars, and session-day changes.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import pytest

from bedivere.core.types import Candle, Timeframe
from bedivere.view.frozen import build_frozen_source
from bedivere.view.market_view import MarketView
from tests.helpers import bar, et, eth_session_days

DAYS2 = eth_session_days(["2026-07-15", "2026-07-16"])
HTFS = [Timeframe.M30, Timeframe.H1, Timeframe.H2, Timeframe.H4]


def _random_series(
    rng: random.Random,
    *,
    period: int,
    drop_rate: float = 0.0,
    out_of_session: bool = False,
) -> list[Candle]:
    """Random-walk close-stamped bars over the two fixture days, optionally
    with dropped bars and a couple of out-of-session stragglers."""
    out: list[Candle] = []
    price = 20_000.0
    for close_date in ("2026-07-15", "2026-07-16"):
        day = next(d for d in DAYS2.days if d.label == close_date)
        for ts in range(day.start_unix + period, day.end_unix + 1, period):
            if drop_rate and rng.random() < drop_rate:
                continue
            o = price
            c = o + rng.uniform(-20, 20)
            h = max(o, c) + rng.uniform(0, 8)
            low = min(o, c) - rng.uniform(0, 8)
            out.append(bar(ts, o, h, low, c, v=rng.uniform(100, 5000)))
            price = c
    if out_of_session:
        # Two bars inside the 17:00→18:00 maintenance break (no session-day).
        for ts_iso in ("2026-07-15T17:30:00", "2026-07-15T17:45:00"):
            ts = et(ts_iso)
            out.append(bar(ts, price, price + 5, price - 5, price, v=42))
    out.sort(key=lambda c: c.timestamp)
    return out


def _assert_view_matches_frozen(bars: list[Candle], base: Timeframe, tfs: list[Timeframe], every: int = 1) -> None:
    view = MarketView(base_tf=base, derived_tfs=tfs, days=DAYS2)
    source = build_frozen_source(bars, tfs, DAYS2)
    for i, b in enumerate(bars):
        view.update(b)
        if i % every and i != len(bars) - 1:
            continue
        fv = source.view_at(b.timestamp)
        assert view.primary == fv.primary, f"primary diverged at {b.timestamp}"
        for tf in tfs:
            assert view.completed(tf) == fv.completed[tf], (
                f"{tf.value} completed diverged at bar {i} ts {b.timestamp}"
            )
            frozen_forming = (
                fv.as_of_view[tf][-1]
                if len(fv.as_of_view[tf]) > len(fv.completed[tf])
                else None
            )
            assert view.forming(tf) == frozen_forming, (
                f"{tf.value} forming diverged at bar {i} ts {b.timestamp}"
            )


def test_full_5m_series_matches_frozen_at_every_instant() -> None:
    rng = random.Random(1)
    _assert_view_matches_frozen(_random_series(rng, period=300), Timeframe.M5, HTFS)


def test_gappy_series_with_out_of_session_bars_matches_frozen() -> None:
    rng = random.Random(2)
    bars = _random_series(rng, period=300, drop_rate=0.25, out_of_session=True)
    _assert_view_matches_frozen(bars, Timeframe.M5, HTFS)


def test_missing_session_close_stub_stays_forming_until_next_day() -> None:
    rng = random.Random(3)
    bars = [
        b
        for b in _random_series(rng, period=300)
        # Drop the 17:00 session-close bar of day 1: its 4h stub bucket must
        # stay FORMING until day 2's first bar proves the period elapsed.
        if b.timestamp != et("2026-07-15T17:00:00")
    ]
    _assert_view_matches_frozen(bars, Timeframe.M5, HTFS)


def test_base_tf_is_a_parameter_15m_and_15s_work() -> None:
    rng = random.Random(4)
    _assert_view_matches_frozen(_random_series(rng, period=900), Timeframe.M15, HTFS)
    # 15s base deriving 5m — sampled compare to bound test time.
    bars_15s = _random_series(rng, period=15, drop_rate=0.4)
    _assert_view_matches_frozen(bars_15s, Timeframe.S15, [Timeframe.M5], every=97)


def test_derived_must_be_coarser_than_base() -> None:
    with pytest.raises(ValueError, match="not coarser"):
        MarketView(base_tf=Timeframe.M15, derived_tfs=[Timeframe.M5], days=DAYS2)
    with pytest.raises(ValueError, match="not coarser"):
        MarketView(base_tf=Timeframe.M5, derived_tfs=[Timeframe.M5], days=DAYS2)


def test_out_of_order_bar_raises() -> None:
    view = MarketView(base_tf=Timeframe.M5, derived_tfs=HTFS, days=DAYS2)
    t0 = et("2026-07-14T18:05:00")
    view.update(bar(t0, 1, 2, 0.5, 1.5))
    with pytest.raises(ValueError, match="out of order"):
        view.update(bar(t0, 1, 2, 0.5, 1.5))  # duplicate
    with pytest.raises(ValueError, match="out of order"):
        view.update(bar(t0 - 300, 1, 2, 0.5, 1.5))  # regression


def test_untracked_tf_accessor_raises() -> None:
    view = MarketView(base_tf=Timeframe.M5, derived_tfs=[Timeframe.M30], days=DAYS2)
    with pytest.raises(KeyError, match="not a tracked"):
        view.completed(Timeframe.H4)


def test_closes_report_tf_candle_index_in_coarseness_order() -> None:
    view = MarketView(
        base_tf=Timeframe.M5,
        derived_tfs=[Timeframe.H1, Timeframe.M30],
        days=DAYS2,
    )
    start = et("2026-07-14T18:00:00")
    closes_seen: list[tuple[str, int]] = []
    for i in range(24):  # two hours of 5m bars
        ts = start + (i + 1) * 300
        for close in view.update(bar(ts, 1, 2, 0.5, 1.5, v=1)):
            assert close.candle == view.completed(close.tf)[close.index]
            closes_seen.append((close.tf.value, close.index))
    # Two hours: four 30m closes, one 1h close after 12 bars + one after 24.
    assert closes_seen == [
        ("30m", 0),
        ("30m", 1),
        ("1h", 0),
        ("30m", 2),
        ("30m", 3),
        ("1h", 1),
    ]


# ---------- the observer seam ----------


class _RecordingObserver:
    """Counts calls and returns a payload every second close of its TF."""

    def __init__(self, tf: Timeframe) -> None:
        self.tf = tf
        self.calls: list[int] = []  # completed-list length at each call

    def on_completed(self, completed: Sequence[Candle]) -> object | None:
        self.calls.append(len(completed))
        n = len(completed)
        return (self.tf.value, n) if n % 2 == 0 else None


def test_observer_called_once_per_close_and_payload_forwarded() -> None:
    observers: dict[Timeframe, _RecordingObserver] = {}

    def factory(tf: Timeframe) -> _RecordingObserver:
        observers[tf] = _RecordingObserver(tf)
        return observers[tf]

    view = MarketView(
        base_tf=Timeframe.M5,
        derived_tfs=[Timeframe.M30, Timeframe.H1],
        days=DAYS2,
        observer_factory=factory,
    )
    assert set(observers) == {Timeframe.M30, Timeframe.H1}  # one per derived TF

    start = et("2026-07-14T18:00:00")
    payloads: list[tuple[str, object | None]] = []
    for i in range(24):  # two hours of 5m bars
        ts = start + (i + 1) * 300
        for close in view.update(bar(ts, 1, 2, 0.5, 1.5, v=1)):
            payloads.append((close.tf.value, close.payload))

    # One call per completed bar, with the post-append length each time.
    assert observers[Timeframe.M30].calls == [1, 2, 3, 4]
    assert observers[Timeframe.H1].calls == [1, 2]
    # Payloads forwarded verbatim; None where the observer returned None.
    assert payloads == [
        ("30m", None),
        ("30m", ("30m", 2)),
        ("1h", None),
        ("30m", None),
        ("30m", ("30m", 4)),
        ("1h", ("1h", 2)),
    ]


def test_no_factory_means_none_payloads() -> None:
    view = MarketView(base_tf=Timeframe.M5, derived_tfs=[Timeframe.M30], days=DAYS2)
    start = et("2026-07-14T18:00:00")
    for i in range(6):
        ts = start + (i + 1) * 300
        for close in view.update(bar(ts, 1, 2, 0.5, 1.5)):
            assert close.payload is None
