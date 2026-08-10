"""The CandleSource port, and honest coverage reporting.

Everything downstream of this file depends on the PROTOCOL, never on the
storage: a composition asks for `(symbol, timeframe, start, end)` and gets
bars. A CSV file, a SQLite cache and a vendor SDK are then the same thing to
a run, which is what lets a strategy backtested from a file run live from a
feed without a line changing.

    candles(symbol, tf, start_unix, end_unix) -> list[Candle]

The range is HALF-OPEN — `(start_unix, end_unix]` — matching the session-day
convention used everywhere in bedivere (start exclusive, close-stamp
inclusive), so `(day.start_unix, day.end_unix]` is exactly one session.

The second half of this module exists because "I got 4 800 bars" is not the
same claim as "the range is covered". A source that silently returns a short
series turns a data outage into a strategy result: the run looks fine, the
funnel just quietly has fewer chances in it. `assess_coverage` compares what
came back against the close-stamps session geometry DEMANDS and names the
gaps — the same `expected_close_stamps` arithmetic the warm-up loader uses,
so both answers come from one definition of "the bars that should exist".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from bedivere.core.session_days import SessionDays, expected_close_stamps
from bedivere.core.types import Candle, Timeframe


class CandleSource(Protocol):
    """Bars for one (symbol, timeframe) over a half-open `(start, end]` range,
    ascending and duplicate-free. Implementations own their own validation —
    the engine never re-sorts or de-duplicates what a source handed it."""

    def candles(
        self, symbol: str, timeframe: Timeframe, start_unix: int, end_unix: int
    ) -> list[Candle]:
        ...

    def describe(self) -> str:
        """One line naming where these bars came from, for the run log and the
        archived provenance. Not parsed by anything."""
        ...


@dataclass(frozen=True, slots=True)
class Gap:
    """A contiguous run of close-stamps the source did not supply."""

    from_unix: int  # first missing close-stamp
    to_unix: int  # last missing close-stamp (inclusive)
    bars: int

    def to_jsonable(self) -> dict[str, int]:
        return {"fromUnix": self.from_unix, "toUnix": self.to_unix, "bars": self.bars}


@dataclass(frozen=True, slots=True)
class Coverage:
    """What the source actually delivered versus what session geometry says
    should exist. `unexpected` counts bars that arrived OUTSIDE the grid —
    usually a timezone or close-stamp convention mismatch, and worth seeing
    rather than tolerating."""

    symbol: str
    timeframe: Timeframe
    start_unix: int
    end_unix: int
    expected: int
    present: int
    unexpected: int
    gaps: tuple[Gap, ...]

    @property
    def complete(self) -> bool:
        return not self.gaps and self.expected == self.present

    def describe(self) -> str:
        if self.expected == 0:
            return f"{self.symbol} {self.timeframe.value}: no session time in the requested range"
        pct = 100.0 * self.present / self.expected
        head = (
            f"{self.symbol} {self.timeframe.value}: {self.present}/{self.expected} bars ({pct:.1f}%)"
        )
        if self.unexpected:
            head += f", {self.unexpected} off-grid"
        if not self.gaps:
            return head
        shown = ", ".join(f"{g.from_unix}..{g.to_unix} ({g.bars})" for g in self.gaps[:3])
        more = f" +{len(self.gaps) - 3} more" if len(self.gaps) > 3 else ""
        return f"{head} — {len(self.gaps)} gap(s): {shown}{more}"

    def to_jsonable(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe.value,
            "startUnix": self.start_unix,
            "endUnix": self.end_unix,
            "expected": self.expected,
            "present": self.present,
            "unexpected": self.unexpected,
            "gaps": [g.to_jsonable() for g in self.gaps],
        }


def expected_stamps_in_range(
    days: SessionDays, timeframe: Timeframe, start_unix: int, end_unix: int
) -> list[int]:
    """Every close-stamp session geometry demands in `(start_unix, end_unix]`,
    ascending. Out-of-session time contributes nothing — a weekend is not a
    gap, and reporting it as one would train you to ignore the report."""
    out: list[int] = []
    for day in days.overlapping(start_unix, end_unix):
        out.extend(
            ts for ts in expected_close_stamps(day, timeframe) if start_unix < ts <= end_unix
        )
    out.sort()
    return out


def assess_coverage(
    bars: list[Candle],
    *,
    days: SessionDays,
    symbol: str,
    timeframe: Timeframe,
    start_unix: int,
    end_unix: int,
) -> Coverage:
    """Compare delivered bars against the session grid and group what is
    missing into contiguous gaps."""
    expected = expected_stamps_in_range(days, timeframe, start_unix, end_unix)
    expected_set = set(expected)
    present_set = {b.timestamp for b in bars if start_unix < b.timestamp <= end_unix}

    gaps: list[Gap] = []
    run_start: int | None = None
    run_end = 0
    run_bars = 0
    for ts in expected:
        if ts in present_set:
            if run_start is not None:
                gaps.append(Gap(from_unix=run_start, to_unix=run_end, bars=run_bars))
                run_start, run_bars = None, 0
            continue
        if run_start is None:
            run_start, run_bars = ts, 0
        run_end = ts
        run_bars += 1
    if run_start is not None:
        gaps.append(Gap(from_unix=run_start, to_unix=run_end, bars=run_bars))

    return Coverage(
        symbol=symbol,
        timeframe=timeframe,
        start_unix=start_unix,
        end_unix=end_unix,
        expected=len(expected),
        present=len(expected_set & present_set),
        unexpected=len(present_set - expected_set),
        gaps=tuple(gaps),
    )
