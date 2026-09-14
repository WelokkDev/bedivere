"""bedivere.data.source — the seam between a store and everything that reads bars.

The coverage check is the part worth pinning: it lives INSIDE the source so every
read site gets it without remembering to ask.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest

from bedivere.config.spec import DataSpec
from bedivere.core.session_days import SessionDay, SessionDays
from bedivere.core.types import Candle, Timeframe
from bedivere.data.build import build_candle_source
from bedivere.data.port import CandleSource
from bedivere.data.source import (
    AS_TRADED,
    BarSourceError,
    CoverageCheckedSource,
    require_coverage,
    require_timeframes,
)
from tests.lake_support import requires_lake

LATEST = 1_000_000


class FakeSource:
    """A store that records how often it is asked what it holds."""

    def __init__(self, latest: int | None = LATEST, stored: Iterable[str] = ()) -> None:
        self._latest = latest
        self._stored = frozenset(stored)
        self.latest_calls = 0
        self.covered_calls = 0
        self.candles_calls = 0

    def describe(self) -> str:
        return "fake:store"

    @property
    def price_basis(self) -> str:
        return AS_TRADED

    def has(self, symbol: str, timeframe: Timeframe) -> bool:
        return True

    def latest(self, symbol: str, timeframe: Timeframe) -> int | None:
        self.latest_calls += 1
        return self._latest

    def covered_days(
        self, symbol: str, timeframe: Timeframe, days: SessionDays
    ) -> frozenset[str]:
        self.covered_calls += 1
        return frozenset(d.label for d in days.days if d.label in self._stored)

    def candles(
        self, symbol: str, timeframe: Timeframe, start_unix: int, end_unix: int
    ) -> list[Candle]:
        self.candles_calls += 1
        return [Candle(LATEST, 1.0, 1.0, 1.0, 1.0, 1.0)]


class BareSource:
    """The whole `CandleSource` contract and nothing more — what a `python`
    factory in somebody else's repo is entitled to be."""

    def describe(self) -> str:
        return "bare:store"

    def candles(
        self, symbol: str, timeframe: Timeframe, start_unix: int, end_unix: int
    ) -> list[Candle]:
        return [Candle(LATEST, 1.0, 1.0, 1.0, 1.0, 1.0)]


def build_bare_source() -> BareSource:
    return BareSource()


# ---------- the coverage check ----------


def test_a_window_inside_the_data_is_served() -> None:
    checked = CoverageCheckedSource(FakeSource())
    assert len(checked.candles("NQ", Timeframe.M5, LATEST - 3600, LATEST)) == 1


def test_a_window_days_past_the_data_is_refused() -> None:
    checked = CoverageCheckedSource(FakeSource())
    with pytest.raises(BarSourceError, match="only through"):
        checked.candles("NQ", Timeframe.M5, LATEST - 3600, LATEST + 3 * 86_400)


def test_the_refusal_says_how_to_fix_it() -> None:
    with pytest.raises(BarSourceError) as e:
        require_coverage(FakeSource(), "NQ", Timeframe.M5, LATEST + 5 * 86_400)
    message = str(e.value)
    assert "silently short" in message
    assert "allowStale" in message


def test_latest_is_memoized_per_symbol_and_timeframe() -> None:
    inner = FakeSource()
    checked = CoverageCheckedSource(inner)
    for _ in range(5):
        checked.candles("NQ", Timeframe.S1, LATEST - 10, LATEST)
    assert inner.candles_calls == 5
    assert inner.latest_calls == 1

    checked.candles("NQ", Timeframe.M5, LATEST - 10, LATEST)
    checked.candles("MNQ", Timeframe.S1, LATEST - 10, LATEST)
    assert inner.latest_calls == 3  # one per (symbol, timeframe)


def test_an_empty_store_does_not_gate() -> None:
    # `require_timeframes` owns the nothing-stored case, with a better message.
    checked = CoverageCheckedSource(FakeSource(latest=None))
    assert len(checked.candles("NQ", Timeframe.M5, 0, LATEST + 10_000)) == 1


def test_a_complete_but_sparse_session_is_not_flagged() -> None:
    # A session's last trade lands seconds to minutes before its close, so a
    # window ending at the close sits past the newest bar on every complete day.
    checked = CoverageCheckedSource(FakeSource())
    checked.candles("NQ", Timeframe.S1, 0, LATEST + 3)
    checked.candles("NQ", Timeframe.M5, 0, LATEST + 3600)


def test_the_tolerance_boundary() -> None:
    checked = CoverageCheckedSource(FakeSource())
    checked.candles("NQ", Timeframe.M5, 0, LATEST + 86_400)  # exactly one session short: served
    with pytest.raises(BarSourceError):
        checked.candles("NQ", Timeframe.M5, 0, LATEST + 86_401)


def test_the_identity_survives_wrapping() -> None:
    wrapped = CoverageCheckedSource(FakeSource())
    assert wrapped.describe() == "fake:store"
    assert wrapped.price_basis == AS_TRADED
    assert wrapped.pinned_as_of is None


def test_the_two_gates_answer_different_questions() -> None:
    inner = FakeSource()
    require_timeframes(inner, "NQ", [Timeframe.M5])  # does the pair exist at all
    require_coverage(inner, "NQ", Timeframe.M5, LATEST)  # does it reach far enough
    with pytest.raises(BarSourceError, match="only through"):
        require_coverage(inner, "NQ", Timeframe.M5, LATEST + 5 * 86_400)


def test_require_timeframes_names_what_was_missing_and_refuses_to_fall_back() -> None:
    class Partial(FakeSource):
        def has(self, symbol: str, timeframe: Timeframe) -> bool:
            return timeframe is Timeframe.M5

    with pytest.raises(BarSourceError) as e:
        require_timeframes(Partial(), "NQ", [Timeframe.M5, Timeframe.S1, Timeframe.H1])
    message = str(e.value)
    assert "1s, 1h" in message and "needed: 5m, 1s, 1h" in message
    assert "will NOT fall back" in message


def test_a_source_that_cannot_answer_is_simply_not_gated() -> None:
    # A CSV is the entire series it holds, so there is nothing to compare against.
    require_timeframes(BareSource(), "NQ", [Timeframe.M5])
    require_coverage(BareSource(), "NQ", Timeframe.M5, LATEST + 10 * 86_400)


def test_the_wrapper_refuses_rather_than_silently_answering_for_the_inner(tmp_path: Path) -> None:
    days = SessionDays(
        template="T", timezone="UTC", days=(SessionDay(label="2026-03-02", start_unix=1, end_unix=2),)
    )
    wrapped = CoverageCheckedSource(BareSource())
    for call in (
        lambda: wrapped.has("NQ", Timeframe.M5),
        lambda: wrapped.covered_days("NQ", Timeframe.M5, days),
        lambda: wrapped.price_basis,
    ):
        with pytest.raises(BarSourceError, match="bare CandleSource"):
            call()


# ---------- the session-aware gate ----------

DAY = 86_400
# Three closes exactly one CME close-to-close apart, then a weekend gap.
E0, E1, E2 = LATEST - 2 * DAY, LATEST - DAY, LATEST
CALENDAR = SessionDays(
    template="T",
    timezone="UTC",
    days=(
        SessionDay(label="D0", start_unix=E0 - 82_800, end_unix=E0),
        SessionDay(label="D1", start_unix=E1 - 82_800, end_unix=E1),
        SessionDay(label="D2", start_unix=E2 - 82_800, end_unix=E2),
        SessionDay(label="D3", start_unix=E2 + 49 * 3600, end_unix=E2 + 72 * 3600),
    ),
)


def test_with_a_calendar_a_store_one_session_behind_is_refused() -> None:
    """86 400 s of slack is exactly one CME session, so a lake whose newest close
    is the one before the window's end passed the time check and ran — with a
    warning — on the bars that were there. The calendar is what can tell."""
    inner = FakeSource(latest=E1, stored=["D0", "D1"])
    checked = CoverageCheckedSource(inner, CALENDAR)
    checked.candles("NQ", Timeframe.M5, E0, E1)  # reaches D1, which is held
    with pytest.raises(BarSourceError, match="session-day D2") as e:
        checked.candles("NQ", Timeframe.M5, E0, E2)  # one session past: exactly 86 400 s
    assert "silently short" in str(e.value) and "allowStale" in str(e.value)

    # Without a calendar the same read is waved through — the limitation
    # DEFAULT_MAX_SHORTFALL documents, and why the CLI always passes one.
    CoverageCheckedSource(FakeSource(latest=E1, stored=["D0", "D1"])).candles(
        "NQ", Timeframe.M5, E0, E2
    )


def test_a_window_ending_inside_a_session_needs_only_the_close_before_it() -> None:
    """A live warm-up runs to `now`, mid-session, and a store holds closed days.
    The in-progress tail is a hole for assess_coverage to report, not a reach
    failure — but the last close IS required."""
    now = E2 - 3600  # inside D2
    current = FakeSource(latest=E1, stored=["D0", "D1"])
    CoverageCheckedSource(current, CALENDAR).candles("NQ", Timeframe.M5, E0, now)

    behind = FakeSource(latest=E0, stored=["D0"])
    with pytest.raises(BarSourceError, match="session-day D1"):
        CoverageCheckedSource(behind, CALENDAR).candles("NQ", Timeframe.M5, E0 - DAY, now)


def test_a_window_ending_in_a_gap_needs_the_session_before_it() -> None:
    """An explicit endUnix on a Saturday, over a store complete through Friday's
    close. Time alone cannot tell this from a store two days behind, and the
    calendar-less check refuses it; the calendar serves it."""
    saturday = E2 + 43 * 3600
    inner = FakeSource(latest=E2, stored=["D0", "D1", "D2"])
    CoverageCheckedSource(inner, CALENDAR).candles("NQ", Timeframe.M5, E0, saturday)
    with pytest.raises(BarSourceError, match="only through"):
        CoverageCheckedSource(inner).candles("NQ", Timeframe.M5, E0, saturday)


def test_a_window_before_the_first_close_falls_back_to_the_time_check() -> None:
    """Nothing has closed by the window's end, so there is no session to reach."""
    inner = FakeSource(latest=E0 - 7200, stored=["D0"])
    checked = CoverageCheckedSource(inner, CALENDAR)
    checked.candles("NQ", Timeframe.M5, E0 - 80_000, E0 - 3600)
    assert inner.covered_calls == 0


def test_covered_days_is_memoized_like_latest() -> None:
    inner = FakeSource(latest=E2, stored=["D0", "D1", "D2"])
    checked = CoverageCheckedSource(inner, CALENDAR)
    for _ in range(5):
        checked.candles("NQ", Timeframe.S1, E1, E2)
    assert inner.candles_calls == 5
    assert inner.covered_calls == 1


def test_a_pinned_view_refuses_by_session_too() -> None:
    class Pinned(FakeSource):
        pinned_as_of = "D1"

        def covered_days(
            self, symbol: str, timeframe: Timeframe, days: SessionDays
        ) -> frozenset[str]:
            held = super().covered_days(symbol, timeframe, days)
            return frozenset(label for label in held if label <= self.pinned_as_of)

    # The store HAS D2; this view will not serve it, and must not say "ingest".
    inner = Pinned(latest=E1, stored=["D0", "D1", "D2"])
    checked = CoverageCheckedSource(inner, CALENDAR)
    checked.candles("NQ", Timeframe.M5, E0, E1)
    with pytest.raises(BarSourceError, match="pinned at its as-of date") as e:
        checked.candles("NQ", Timeframe.M5, E0, E2)
    assert "session-day D2" in str(e.value)
    assert "Ingest" not in str(e.value)


def test_a_latest_only_source_keeps_the_time_check_under_a_calendar() -> None:
    """A `python` factory that answers `latest` but not `covered_days` was never
    asked for more; a calendar must not turn its first read into a refusal."""

    class LatestOnly(BareSource):
        def latest(self, symbol: str, timeframe: Timeframe) -> int | None:
            return LATEST

    checked = CoverageCheckedSource(LatestOnly(), CALENDAR)
    checked.candles("NQ", Timeframe.M5, 0, LATEST + 3600)
    with pytest.raises(BarSourceError, match="only through"):
        checked.candles("NQ", Timeframe.M5, 0, LATEST + 3 * DAY)


# ---------- how a spec applies it ----------


def _build(**payload: object) -> CandleSource:
    return build_candle_source(
        DataSpec.model_validate(payload), symbol="DEMO", timeframe=Timeframe.M5
    )


@requires_lake
def test_a_spec_built_source_is_coverage_checked_by_default(tmp_path: Path) -> None:
    built = _build(source="lake", dataset="GLBX.MDP3", series="local.v.0", root=str(tmp_path))
    assert isinstance(built, CoverageCheckedSource)


@requires_lake
def test_allow_stale_opts_out_visibly_in_the_spec(tmp_path: Path) -> None:
    built = _build(
        source="lake", dataset="GLBX.MDP3", series="local.v.0", root=str(tmp_path), allowStale=True
    )
    assert not isinstance(built, CoverageCheckedSource)
    assert built.describe() == "lake:GLBX.MDP3/local.v.0"


def test_allow_stale_must_be_a_boolean() -> None:
    with pytest.raises(ValueError, match="allowStale"):
        DataSpec.model_validate(
            {"source": "lake", "dataset": "D", "series": "s", "allowStale": "yes"}
        )


def test_a_lake_spec_needs_both_halves_of_the_identity() -> None:
    for payload in (
        {"source": "lake"},
        {"source": "lake", "dataset": "GLBX.MDP3"},
        {"source": "lake", "series": "local.v.0"},
    ):
        with pytest.raises(ValueError, match="no safe default"):
            DataSpec.model_validate(payload)


def test_a_lake_spec_cannot_be_named_by_path_alone() -> None:
    with pytest.raises(ValueError, match="cannot say which"):
        DataSpec.model_validate({"source": "lake", "path": "data/lake"})


def test_a_bare_python_source_is_left_unwrapped() -> None:
    built = _build(source="python", factory="tests.test_bar_source:build_bare_source")
    assert isinstance(built, BareSource)


def test_a_capable_python_source_is_wrapped_like_any_other() -> None:
    built = _build(source="python", factory="tests.test_bar_source:build_fake_source")
    assert isinstance(built, CoverageCheckedSource)
    with pytest.raises(BarSourceError, match="only through"):
        built.candles("DEMO", Timeframe.M5, 0, LATEST + 5 * 86_400)


def build_fake_source() -> FakeSource:
    return FakeSource()


def test_a_csv_spec_stays_dependency_free(tmp_path: Path) -> None:
    path = tmp_path / "demo.csv"
    path.write_text("timestamp,open,high,low,close,volume\n100,1,2,0.5,1.5,3\n", encoding="utf-8")
    built = _build(source="csv", path=str(path))
    assert built.describe().startswith("csv:")
    assert len(built.candles("DEMO", Timeframe.M5, 0, 1 << 40)) == 1
