"""bedivere.core.buckets — the single-home bucket arithmetic."""

from __future__ import annotations

from bedivere.core.buckets import bucket_index_of, bucket_period_end


def test_boundary_close_stamp_lands_in_its_own_bucket() -> None:
    # A close-stamp exactly at a grid boundary belongs to the bucket whose
    # data window it ENDS — the load-bearing `- 1`.
    start = 1_000_000
    period = 1800
    assert bucket_index_of(start + 1, start, period) == 0
    assert bucket_index_of(start + period, start, period) == 0
    assert bucket_index_of(start + period + 1, start, period) == 1
    assert bucket_index_of(start + 2 * period, start, period) == 1


def test_period_end_grid_and_stub_clamp() -> None:
    start, end = 1_000_000, 1_000_000 + 82_800  # 23h session
    assert bucket_period_end(start, end, 0, 1800) == start + 1800
    assert bucket_period_end(start, end, 45, 1800) == end  # last full 30m ends at close
    # 4h buckets: the 6th (index 5) would grid-end past the session → clamped.
    assert bucket_period_end(start, end, 5, 14_400) == end
