"""bedivere.streams.sparse — window selection, mixed-fidelity emission, guards.

The end-to-end gate lives in test_sparse_parity.py. This file pins the
pieces: that the pre-pass is causal, that windows clamp and merge and carry
the way the docstrings claim, that the stream emits one contiguous ascending
series across a resolution change, and that both guards fire.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from bedivere.core.types import Candle, Timeframe
from bedivere.streams.sparse import (
    AgreementError,
    CoverageError,
    FineLoader,
    FineWindow,
    SparseReplayStream,
    TriggerRule,
    assert_trades_covered,
    build_fine_windows,
    select_triggers,
    trade_coverage,
    verify_agreement,
)
from tests.helpers import bar, make_session_days

# Two 1-hour session days, an hour apart. Small enough to reason about by
# hand: 5-minute coarse bars, 15-second fine bars.
DAY1_OPEN, DAY1_CLOSE = 1_000_000, 1_003_600
DAY2_OPEN, DAY2_CLOSE = 1_007_200, 1_010_800
DAYS = make_session_days(
    "test",
    "UTC",
    [("d1", DAY1_OPEN, DAY1_CLOSE), ("d2", DAY2_OPEN, DAY2_CLOSE)],
)
COARSE_TF, FINE_TF = Timeframe.M5, Timeframe.S15
_COARSE_STEP, _FINE_STEP = 300, 15


def _coarse(day_open: int, day_close: int) -> list[Candle]:
    return [
        bar(ts, 10, 10, 10, 10, 1)
        for ts in range(day_open + _COARSE_STEP, day_close + 1, _COARSE_STEP)
    ]


def _fine(day_open: int, day_close: int) -> list[Candle]:
    return [
        bar(ts, 10, 10, 10, 10, 1)
        for ts in range(day_open + _FINE_STEP, day_close + 1, _FINE_STEP)
    ]


ALL_COARSE: list[Candle] = _coarse(DAY1_OPEN, DAY1_CLOSE) + _coarse(DAY2_OPEN, DAY2_CLOSE)
ALL_FINE: list[Candle] = _fine(DAY1_OPEN, DAY1_CLOSE) + _fine(DAY2_OPEN, DAY2_CLOSE)
_FINE_BY_TS: dict[int, Candle] = {c.timestamp: c for c in ALL_FINE}


def _loader(start: int, end: int) -> list[Candle]:
    return [_FINE_BY_TS[t] for t in range(start + 1, end + 1) if t in _FINE_BY_TS]


def _fires_at(stamp: int) -> TriggerRule:
    """A rule that triggers on exactly one coarse close."""

    def rule(prefix: Sequence[Candle]) -> bool:
        return prefix[-1].timestamp == stamp

    return rule


def _never(_prefix: Sequence[Candle]) -> bool:
    return False


class _Trade:
    """The structural slice the coverage guard needs."""

    def __init__(self, entry_ts: int, exit_ts: int) -> None:
        self.entry_ts = entry_ts
        self.exit_ts = exit_ts


# ---------- the pre-pass ----------


def test_select_triggers_sees_only_the_prefix() -> None:
    """Causality is structural, not a convention: the rule is handed
    `coarse[: i + 1]` and can therefore never read its own future."""
    seen: list[int] = []

    def rule(prefix: Sequence[Candle]) -> bool:
        seen.append(len(prefix))
        return prefix[-1].timestamp == ALL_COARSE[3].timestamp

    triggers = select_triggers(ALL_COARSE, rule)
    assert triggers == [ALL_COARSE[3].timestamp]
    assert seen == list(range(1, len(ALL_COARSE) + 1))


# ---------- windows ----------


def test_window_runs_to_the_session_close_not_the_candle_close() -> None:
    t = DAY1_OPEN + 600
    [w] = [x for x in build_fine_windows([t], DAYS, carry_across_sessions=False)]
    assert (w.start_unix, w.end_unix) == (t, DAY1_CLOSE)
    assert w.triggers == (t,)


def test_overlapping_windows_merge_and_keep_every_trigger() -> None:
    a, b = DAY1_OPEN + 600, DAY1_OPEN + 900
    merged = build_fine_windows([a, b], DAYS, carry_across_sessions=False)
    assert len(merged) == 1
    assert merged[0].triggers == (a, b)
    assert (merged[0].start_unix, merged[0].end_unix) == (a, DAY1_CLOSE)


def test_trigger_at_the_session_close_opens_nothing() -> None:
    """There is no fine time left in the day to descend into."""
    assert build_fine_windows([DAY1_CLOSE], DAYS, carry_across_sessions=False) == []


def test_window_seconds_shortens_the_horizon_and_clamps_to_the_close() -> None:
    t = DAY1_OPEN + 600
    [short] = build_fine_windows(
        [t], DAYS, window_seconds=300, carry_across_sessions=False
    )
    assert (short.start_unix, short.end_unix) == (t, t + 300)

    late = DAY1_CLOSE - 100
    [clamped] = build_fine_windows(
        [late], DAYS, window_seconds=9_999, carry_across_sessions=False
    )
    assert clamped.end_unix == DAY1_CLOSE, "a horizon may not run past the session"


def test_window_seconds_must_be_positive() -> None:
    with pytest.raises(ValueError, match="window_seconds"):
        build_fine_windows([DAY1_OPEN + 600], DAYS, window_seconds=0)


def test_a_window_reaching_the_close_carries_into_the_next_session() -> None:
    """A position may have survived the bell, so the next day opens fine."""
    windows = build_fine_windows([DAY1_OPEN + 600], DAYS, carry_across_sessions=True)
    spans = {(w.start_unix, w.end_unix) for w in windows}
    assert (DAY2_OPEN, DAY2_CLOSE) in spans


def test_a_shortened_window_does_not_carry() -> None:
    """It never reached the close, so no position can be riding past it."""
    windows = build_fine_windows(
        [DAY1_OPEN + 600], DAYS, window_seconds=300, carry_across_sessions=True
    )
    assert all(w.start_unix != DAY2_OPEN for w in windows)


def test_carry_is_one_day_deep() -> None:
    """Cascading would degenerate to full fidelity for the rest of the run
    after a single overnight hold. Two nights is the guard's job."""
    days3 = make_session_days(
        "test",
        "UTC",
        [
            ("d1", DAY1_OPEN, DAY1_CLOSE),
            ("d2", DAY2_OPEN, DAY2_CLOSE),
            ("d3", DAY2_OPEN + 7_200, DAY2_CLOSE + 7_200),
        ],
    )
    windows = build_fine_windows([DAY1_OPEN + 600], days3, carry_across_sessions=True)
    assert all(w.start_unix != DAY2_OPEN + 7_200 for w in windows)


# ---------- the stream ----------


def _stream(
    rule: TriggerRule, *, fine_loader: FineLoader = _loader
) -> SparseReplayStream:
    return SparseReplayStream(
        symbol="DEMO",
        fine_timeframe=FINE_TF,
        coarse_timeframe=COARSE_TF,
        coarse_bars=ALL_COARSE,
        rule=rule,
        fine_loader=fine_loader,
        days=DAYS,
    )


def test_no_trigger_means_a_pure_coarse_replay() -> None:
    stream = _stream(_never)
    events = list(stream)
    assert [e.ts for e in events] == [c.timestamp for c in ALL_COARSE]
    assert stream.report.fine_bars == 0
    assert all(e.timeframe is COARSE_TF for e in events)


def test_emission_is_strictly_ascending_across_the_resolution_change() -> None:
    """The one property the engine loop depends on. A duplicated or
    out-of-order bar at a region boundary would raise in MarketView."""
    target = ALL_COARSE[2].timestamp
    events = list(_stream(_fires_at(target)))
    stamps = [e.ts for e in events]
    assert stamps == sorted(set(stamps)), "bars must be ascending and unique"


def test_coarse_bars_inside_a_window_are_replaced_not_added() -> None:
    """The fine bars rebuild those buckets; emitting both would double-count
    the volume and push the view out of order."""
    target = ALL_COARSE[2].timestamp
    events = list(_stream(_fires_at(target)))
    covered = [c.timestamp for c in ALL_COARSE if target < c.timestamp <= DAY1_CLOSE]
    coarse_emitted = {e.ts for e in events if e.timeframe is COARSE_TF}
    assert not (coarse_emitted & set(covered))
    # ...and the fine bars for that span are there instead.
    fine_emitted = {e.ts for e in events if e.timeframe is FINE_TF}
    assert target + _FINE_STEP in fine_emitted


def test_each_event_is_stamped_with_the_resolution_it_carries() -> None:
    """Mixed fidelity is visible on the stream itself, not only in the
    report. The loop ignores the field: it cannot tell, and need not."""
    target = ALL_COARSE[2].timestamp
    events = list(_stream(_fires_at(target)))
    assert {e.timeframe for e in events} == {COARSE_TF, FINE_TF}


def test_report_counts_and_fine_fraction() -> None:
    target = ALL_COARSE[2].timestamp
    stream = _stream(_fires_at(target))
    events = list(stream)
    report = stream.report
    assert report.coarse_bars + report.fine_bars == len(events)
    assert report.mode == "sparse"
    payload = report.to_jsonable()
    assert 0.0 < payload["fineFraction"] < 1.0  # type: ignore[operator]


def test_a_loader_returning_bars_outside_its_window_is_rejected() -> None:
    stream = SparseReplayStream(
        symbol="DEMO",
        fine_timeframe=FINE_TF,
        coarse_timeframe=COARSE_TF,
        coarse_bars=ALL_COARSE,
        rule=_fires_at(ALL_COARSE[2].timestamp),
        fine_loader=lambda _s, _e: [bar(DAY1_OPEN + 1, 1.0, 1.0, 1.0, 1.0)],
        days=DAYS,
    )
    with pytest.raises(ValueError, match="outside the requested"):
        list(stream)


def test_a_loader_returning_disordered_bars_is_rejected() -> None:
    target = ALL_COARSE[2].timestamp
    stream = SparseReplayStream(
        symbol="DEMO",
        fine_timeframe=FINE_TF,
        coarse_timeframe=COARSE_TF,
        coarse_bars=ALL_COARSE,
        rule=_fires_at(target),
        fine_loader=lambda _s, _e: [
            bar(target + 2 * _FINE_STEP, 1, 1, 1, 1),
            bar(target + _FINE_STEP, 1, 1, 1, 1),
        ],
        days=DAYS,
    )
    with pytest.raises(ValueError, match="strictly ascending"):
        list(stream)


def test_coarse_must_be_coarser_than_fine() -> None:
    with pytest.raises(ValueError, match="must be coarser"):
        SparseReplayStream(
            symbol="DEMO",
            fine_timeframe=COARSE_TF,
            coarse_timeframe=FINE_TF,
            coarse_bars=ALL_COARSE,
            rule=_never,
            fine_loader=_loader,
            days=DAYS,
        )


# ---------- guard 1: agreement ----------


def test_verify_agreement_passes_when_the_series_reproduce() -> None:
    rule = _fires_at(ALL_COARSE[2].timestamp)

    assert verify_agreement(
        run_coarse=ALL_COARSE, rule=rule, expected=[ALL_COARSE[2].timestamp]
    ) == [ALL_COARSE[2].timestamp]


def test_verify_agreement_raises_when_the_run_finds_a_trigger_the_prepass_missed() -> None:
    """The failure this exists for: a cached coarse bar that is NOT the
    aggregate of its fine bars, so the two passes see different markets."""
    with pytest.raises(AgreementError, match="disagrees with the coarse pre-pass"):
        verify_agreement(
            run_coarse=ALL_COARSE,
            rule=_fires_at(ALL_COARSE[2].timestamp),
            expected=[],
        )


def test_verify_agreement_is_scoped_to_the_runs_own_span() -> None:
    """The pre-pass may be handed a wider series than the run replayed;
    triggers outside the span were never the run's to reproduce."""
    short = ALL_COARSE[:4]
    verify_agreement(
        run_coarse=short,
        rule=_never,
        expected=[ALL_COARSE[-1].timestamp],  # beyond the run — ignored
    )


# ---------- guard 2: coverage ----------


def test_a_trade_inside_one_window_is_covered() -> None:
    t = DAY1_OPEN + 600
    windows = build_fine_windows([t], DAYS, carry_across_sessions=False)
    assert trade_coverage([_Trade(t + 120, t + 300)], windows, DAYS) == [True]


def test_a_trade_starting_on_the_coarse_trigger_bar_is_not_covered() -> None:
    """The bar stamped at `start_unix` IS the coarse trigger bar — an entry
    filled on it was matched at coarse resolution."""
    t = DAY1_OPEN + 600
    windows = build_fine_windows([t], DAYS, carry_across_sessions=False)
    assert trade_coverage([_Trade(t, t + 300)], windows, DAYS) == [False]


def test_an_overnight_trade_is_covered_when_the_carry_applies() -> None:
    """The overnight GAP is not a coverage hole — no bars exist in it. A
    check against wall time would fail this trade for a reason that does
    not exist; the check is against SESSION time."""
    t = DAY1_OPEN + 600
    windows = build_fine_windows([t], DAYS, carry_across_sessions=True)
    assert trade_coverage([_Trade(t + 120, DAY2_OPEN + 300)], windows, DAYS) == [True]


def test_an_overnight_trade_is_uncovered_without_the_carry() -> None:
    t = DAY1_OPEN + 600
    windows = build_fine_windows([t], DAYS, carry_across_sessions=False)
    assert trade_coverage([_Trade(t + 120, DAY2_OPEN + 300)], windows, DAYS) == [False]


def test_assert_trades_covered_names_the_escapees() -> None:
    windows = [FineWindow(start_unix=DAY1_OPEN, end_unix=DAY1_OPEN + 300, triggers=())]
    with pytest.raises(CoverageError) as excinfo:
        assert_trades_covered(
            [_Trade(DAY1_OPEN + 60, DAY1_OPEN + 120), _Trade(DAY1_OPEN + 60, DAY1_CLOSE)],
            windows,
            DAYS,
        )
    message = str(excinfo.value)
    assert "1 of 2 trade(s)" in message
    assert "#1" in message


def test_no_trades_is_trivially_covered() -> None:
    assert_trades_covered([], [], DAYS)
