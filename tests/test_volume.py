"""Numerical and causal invariants of volume-clock aggregation (no lake extra)."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from bedivere.core.session_days import SessionDay, SessionDays
from bedivere.data.lake.volume import (
    NS,
    FormingBar,
    VolumeAccumulator,
    VolumeBar,
    VolumeInput,
    VolumeSpec,
    summarize,
)

DAY = SessionDay("2026-06-15", 100, 110)
DAYS = SessionDays("test", "UTC", (DAY,))
TRADES = [
    VolumeInput.trade(100 * NS + 100, 10.0, 400, 42),
    VolumeInput.trade(100 * NS + 200, 12.0, 800, 42),
    VolumeInput.trade(100 * NS + 300, 8.0, 400, 42),
    VolumeInput.trade(101 * NS + 100, 11.0, 600, 42),
    VolumeInput.trade(101 * NS + 300, 9.0, 250, 42),
]


def test_whole_trade_boundary_and_remainder_availability() -> None:
    builder = VolumeAccumulator(VolumeSpec(1000), DAY)
    bars = [bar for trade in TRADES for bar in builder.push(trade)]
    assert [bar.volume for bar in bars] == [1200, 1000]
    assert (bars[0].open, bars[0].high, bars[0].low, bars[0].close) == (10, 12, 10, 12)
    assert bars[0].vwap == pytest.approx((400 * 10 + 800 * 12) / 1200)
    assert bars[1].start_ns == TRADES[2].start_ns
    (remainder,) = builder.finish()
    assert remainder.volume == 250 and remainder.is_partial
    assert remainder.end_ns == TRADES[-1].end_ns
    assert remainder.available_ns == DAY.end_unix * NS
    assert [bar.bar_id for bar in [*bars, remainder]] == [0, 1, 2]
    assert sum(bar.volume for bar in [*bars, remainder]) == sum(t.volume for t in TRADES)
    assert builder.finish() == []
    with pytest.raises(ValueError, match="after session finish"):
        builder.push(TRADES[-1])


def test_split_trade_preserves_price_and_quantity_at_same_timestamp() -> None:
    builder = VolumeAccumulator(VolumeSpec(1000, boundary="split_trade"), DAY)
    bars = builder.push(VolumeInput.trade(100 * NS, -2.5, 3250, 42))
    assert [bar.volume for bar in bars] == [1000, 1000, 1000]
    assert {bar.end_ns for bar in bars} == {100 * NS}
    assert {bar.vwap for bar in bars} == {-2.5}
    assert [bar.bar_id for bar in bars] == [0, 1, 2]
    (tail,) = builder.finish()
    assert tail.volume == 250
    assert all(bar.input_count == 1 for bar in [*bars, tail])


@pytest.mark.parametrize("split", [False, True])
def test_every_batch_boundary_gives_identical_bars(split: bool) -> None:
    spec = VolumeSpec(1000, boundary="split_trade" if split else "whole_trade")
    expected: list[VolumeBar] | None = None
    for cut in range(len(TRADES) + 1):
        builder = VolumeAccumulator(spec, DAY)
        bars: list[VolumeBar] = []
        for chunk in (TRADES[:cut], TRADES[cut:]):
            for trade in chunk:
                bars.extend(builder.push(trade))
        bars.extend(builder.finish())
        if expected is None:
            expected = bars
        assert bars == expected
        assert sum(b.volume for b in bars) == 2450
        assert sum(b.vwap * b.volume for b in bars if b.vwap is not None) == pytest.approx(
            sum(t.close * t.volume for t in TRADES)
        )


def test_approximation_preserves_wicks_but_does_not_invent_vwap() -> None:
    builder = VolumeAccumulator(VolumeSpec(1000, "ohlcv-1s"), DAY)
    (bar,) = builder.push(VolumeInput(100 * NS, 101 * NS, 10, 20, 5, 12, 1400, 42))
    assert (bar.open, bar.high, bar.low, bar.close, bar.volume) == (10, 20, 5, 12, 1400)
    assert bar.vwap is None
    assert bar.available_ns == 101 * NS


def test_no_empty_bar_or_extra_bar_on_exact_threshold() -> None:
    builder = VolumeAccumulator(VolumeSpec(400), DAY)
    assert len(builder.push(TRADES[0])) == 1
    assert builder.finish() == []
    assert VolumeAccumulator(VolumeSpec(400), DAY).finish() == []


@pytest.mark.parametrize("threshold", [0, -1, True, 1.5, 2**63])
def test_invalid_threshold(threshold: int) -> None:
    with pytest.raises(ValueError, match="positive int64"):
        VolumeSpec(threshold)


@pytest.mark.parametrize("source", ["ohlcv-1s", "trades-1s"])
def test_approximate_sources_cannot_split(source: str) -> None:
    with pytest.raises(ValueError, match="cannot be split"):
        VolumeSpec(1000, source, "split_trade")
    with pytest.raises(ValueError, match="supports sources trades, ohlcv-1s and trades-1s"):
        VolumeSpec(1000, source.replace("1s", "1m"))


@pytest.mark.parametrize(
    "row,match",
    [
        (replace(TRADES[0], volume=-1), "volume"),
        (replace(TRADES[0], volume=1.2), "volume"),
        (replace(TRADES[0], instrument_id=0), "instrument"),
        (replace(TRADES[0], open=float("nan")), "one timestamp and one price"),
        (VolumeInput.trade(110 * NS, 10, 100, 42), "session close"),
        (VolumeInput.trade(99 * NS, 10, 100, 42), "outside"),
    ],
)
def test_bad_inputs_fail_before_accumulating(row: VolumeInput, match: str) -> None:
    builder = VolumeAccumulator(VolumeSpec(1000), DAY)
    with pytest.raises(ValueError, match=match):
        builder.push(row)
    assert builder.finish() == []


def test_duplicate_times_are_valid_but_backwards_order_and_rolls_are_not() -> None:
    builder = VolumeAccumulator(VolumeSpec(1000), DAY)
    builder.push(TRADES[1])
    assert builder.push(TRADES[1])[0].volume == 1600
    with pytest.raises(ValueError, match="out of order"):
        builder.push(TRADES[0])
    with pytest.raises(ValueError, match="mix instruments"):
        builder.push(replace(TRADES[2], instrument_id=43))


def test_definition_identity_includes_policy_source_and_calendar() -> None:
    spec = VolumeSpec(1000)
    keys = {
        spec.key(DAYS),
        replace(spec, threshold=1001).key(DAYS),
        replace(spec, boundary="split_trade").key(DAYS),
        replace(spec, source="ohlcv-1s").key(DAYS),
        replace(spec, source="trades-1s").key(DAYS),
        spec.key(replace(DAYS, template="other")),
    }
    assert len(keys) == 6
    # Seconds rebuilt from trades are not stored seconds, down to what they admit.
    eligibility = {
        source: json.loads(replace(spec, source=source).definition(DAYS))["eligibility"]
        for source in ("trades", "ohlcv-1s", "trades-1s")
    }
    assert eligibility == {
        "trades": "positive_trades_no_bad_recv",
        "ohlcv-1s": "positive_nonsynthetic_ohlcv",
        "trades-1s": "positive_trades_no_bad_recv",
    }
    # Appending another session does not change an existing dataset's identity.
    assert spec.key(DAYS) == spec.key(
        replace(DAYS, days=(*DAYS.days, SessionDay("next", 120, 130)))
    )


def test_summary_excludes_remainders_from_overshoot_statistics() -> None:
    builder = VolumeAccumulator(VolumeSpec(1000), DAY)
    bars = [b for row in TRADES for b in builder.push(row)] + builder.finish()
    result = summarize(bars, 1000)
    assert result["volume"] == 2450
    assert result["overshoot_max"] == 200
    assert result["full_bars"] == 2
    assert result["partial_volume"] == 250


def test_split_bars_match_independent_unit_volume_expansion() -> None:
    import random

    rng = random.Random(42)
    trades = [
        VolumeInput.trade(100 * NS + i, float(rng.randrange(-10, 20)), rng.randrange(1, 25), 42)
        for i in range(80)
    ]
    builder = VolumeAccumulator(VolumeSpec(17, boundary="split_trade"), DAY)
    bars = [b for row in trades for b in builder.push(row)] + builder.finish()
    units = [(row.close, row.end_ns) for row in trades for _ in range(row.volume)]
    for i, bar in enumerate(bars):
        expected = units[i * 17 : (i + 1) * 17]
        prices = [price for price, _ in expected]
        assert (bar.open, bar.high, bar.low, bar.close) == (
            prices[0],
            max(prices),
            min(prices),
            prices[-1],
        )
        assert (bar.start_ns, bar.end_ns) == (expected[0][1], expected[-1][1])
        assert bar.volume == len(expected)
        assert bar.vwap == pytest.approx(sum(prices) / len(prices))


def test_forming_view_emits_nothing_and_anticipates_no_boundary() -> None:
    watched = VolumeAccumulator(VolumeSpec(1000), DAY)
    plain = VolumeAccumulator(VolumeSpec(1000), DAY)
    assert watched.forming() is None
    emitted: list[VolumeBar] = []
    reference: list[VolumeBar] = []
    for trade in TRADES:
        emitted.extend(watched.push(trade))
        watched.forming()
        reference.extend(plain.push(trade))
    assert watched.forming() == FormingBar(2, TRADES[-1].start_ns, TRADES[-1].end_ns, 9, 9, 9, 9, 250, 42, 1, 9.0)
    assert emitted == reference
    assert watched.finish() == plain.finish()
    assert watched.forming() is None
    # Under nearest_second the view is the raw accumulation, and VWAP stays None.
    nearest = VolumeAccumulator(VolumeSpec(1000, "ohlcv-1s", "nearest_second"), DAY)
    assert nearest.push(VolumeInput(100 * NS, 101 * NS, 10, 20, 5, 12, 900, 42)) == []
    assert nearest.forming() == FormingBar(0, 100 * NS, 101 * NS, 10, 20, 5, 12, 900, 42, 1, None)
    (bar,) = nearest.push(VolumeInput(101 * NS, 102 * NS, 12, 13, 11, 12, 300, 42))
    assert (bar.volume, bar.end_ns, bar.available_ns) == (900, 101 * NS, 102 * NS)
    assert nearest.forming() == FormingBar(1, 101 * NS, 102 * NS, 12, 13, 11, 12, 300, 42, 1, None)
