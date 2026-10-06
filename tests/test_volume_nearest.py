"""Whole-second nearest boundaries, independently checked in volume space."""

from __future__ import annotations

import json
import random
from bisect import bisect_left
from dataclasses import replace
from itertools import accumulate

import pytest

from bedivere.data.lake.volume import (
    NS,
    VolumeAccumulator,
    VolumeBar,
    VolumeInput,
    VolumeSpec,
    summarize,
)
from tests.test_volume import DAY, DAYS


def second(index: int, volume: int) -> VolumeInput:
    price = float(index + 10)
    return VolumeInput(
        (100 + index) * NS, (101 + index) * NS, price, price + 2, price - 2, price + 1, volume, 42
    )


def oracle(rows: list[VolumeInput], threshold: int) -> list[VolumeBar]:
    """Batch search for each target, distinct from the O(1) streaming algorithm."""
    positive = [row for row in rows if row.volume]
    cumulative = list(accumulate((row.volume for row in positive), initial=0))
    cuts: list[tuple[int, int, bool]] = []
    previous = 0
    for target in range(threshold, cumulative[-1] + 1, threshold):
        crossing = bisect_left(cumulative, target)
        boundary = min((crossing - 1, crossing), key=lambda i: abs(cumulative[i] - target))
        if boundary > previous:
            cuts.append((boundary, positive[crossing - 1].end_ns, False))
            previous = boundary
    if previous < len(positive):
        cuts.append((len(positive), DAY.end_unix * NS, True))
    bars: list[VolumeBar] = []
    previous = 0
    for boundary, available, partial in cuts:
        group = positive[previous:boundary]
        bars.append(
            VolumeBar(
                len(bars),
                group[0].start_ns,
                group[-1].end_ns,
                available,
                group[0].open,
                max(row.high for row in group),
                min(row.low for row in group),
                group[-1].close,
                sum(row.volume for row in group),
                42,
                len(group),
                None,
                partial,
            )
        )
        previous = boundary
    return bars


def test_previous_boundary_requires_crossing_second_before_publication() -> None:
    builder = VolumeAccumulator(VolumeSpec(5000, "ohlcv-1s", "nearest_second"), DAY)
    assert builder.push(second(0, 4500)) == []
    # Equidistant boundaries choose the earlier one, without the crossing second's prices.
    (bar,) = builder.push(second(4, 1000))
    assert bar.volume == 4500 and not bar.is_partial
    assert bar.end_ns == 101 * NS and bar.available_ns == 105 * NS
    assert (bar.open, bar.high, bar.low, bar.close, bar.input_count) == (10, 12, 8, 11, 1)
    (tail,) = builder.finish()
    assert tail.volume == 1000 and tail.open == 14 and tail.available_ns == 110 * NS
    assert tail.is_partial and bar.vwap is tail.vwap is None


def test_next_boundary_and_exact_target_need_no_extra_delay() -> None:
    for volumes in ([4900, 110], [4900, 100]):
        builder = VolumeAccumulator(VolumeSpec(5000, "ohlcv-1s", "nearest_second"), DAY)
        assert builder.push(second(0, volumes[0])) == []
        (bar,) = builder.push(second(1, volumes[1]))
        assert bar.volume == sum(volumes)
        assert bar.available_ns == bar.end_ns == 102 * NS
        assert builder.finish() == []


def test_burst_coalesces_targets_without_empty_bars_or_splitting_candles() -> None:
    builder = VolumeAccumulator(VolumeSpec(1000, "ohlcv-1s", "nearest_second"), DAY)
    builder.push(second(0, 900))
    bars = builder.push(second(1, 3000))
    assert [bar.volume for bar in bars] == [900, 3000]
    assert [bar.available_ns for bar in bars] == [102 * NS, 102 * NS]
    assert [bar.input_count for bar in bars] == [1, 1]
    assert builder.finish() == []
    # This would be intractable if implementation iterated every crossed target.
    huge = VolumeAccumulator(VolumeSpec(1, "ohlcv-1s", "nearest_second"), DAY)
    assert huge.push(second(0, 2**60))[0].volume == 2**60
    assert huge.finish() == []


def test_remainder_is_unclosed_volume_not_necessarily_undersized() -> None:
    builder = VolumeAccumulator(VolumeSpec(1000, "ohlcv-1s", "nearest_second"), DAY)
    assert builder.push(second(0, 1750))[0].volume == 1750
    assert builder.push(second(1, 1050)) == []  # nearest target boundary was already emitted
    (tail,) = builder.finish()
    assert tail.is_partial and tail.volume == 1050 and tail.available_ns == 110 * NS


@pytest.mark.parametrize("seed", range(30))
def test_stream_matches_independent_batch_oracle_at_every_prefix(seed: int) -> None:
    rng = random.Random(seed)
    threshold = rng.randrange(3, 40)
    rows = [second(i, rng.randrange(0, 100)) for i in range(8)]
    builder = VolumeAccumulator(VolumeSpec(threshold, "ohlcv-1s", "nearest_second"), DAY)
    emitted: list[VolumeBar] = []
    for i, row in enumerate(rows):
        emitted.extend(builder.push(row))
        assert emitted == [bar for bar in oracle(rows[: i + 1], threshold) if not bar.is_partial]
    emitted.extend(builder.finish())
    assert emitted == oracle(rows, threshold)
    assert sum(bar.volume for bar in emitted) == sum(row.volume for row in rows)
    assert sum(bar.input_count for bar in emitted) == sum(row.volume > 0 for row in rows)
    assert builder.finish() == []


def test_empty_and_zero_seconds_never_manufacture_bars() -> None:
    builder = VolumeAccumulator(VolumeSpec(1000, "ohlcv-1s", "nearest_second"), DAY)
    assert builder.push(second(0, 0)) == []
    assert builder.finish() == []


def test_mode_identity_and_symmetric_volume_error_reporting() -> None:
    spec = VolumeSpec(5000, "ohlcv-1s", "nearest_second")
    assert spec.key(DAYS) != replace(spec, boundary="whole_trade").key(DAYS)
    assert json.loads(spec.definition(DAYS))["availability"] == "crossing_second_end"
    # Protect hashes of pre-existing datasets from accidental policy migration.
    assert (
        VolumeSpec(1000).key(DAYS)
        == "aff6136ba7b60f4fcb5e573d2208c6825f297f2a0f77b0beef29221753e983ed"
    )
    with pytest.raises(ValueError, match="requires source=ohlcv-1s"):
        VolumeSpec(5000, boundary="nearest_second")
    builder = VolumeAccumulator(spec, DAY)
    bars = [bar for row in [second(0, 4500), second(1, 1000)] for bar in builder.push(row)]
    report = summarize(bars + builder.finish(), 5000)
    assert report["volume_deviation_p95"] == report["undershoot_max"] == 500
    assert report["overshoot_max"] == 0
