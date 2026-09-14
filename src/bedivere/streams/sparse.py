"""SparseReplayStream — coarse everywhere, fine where it matters.

A strategy that decides on 5-minute closes but *fills* inside the candle has
to be replayed at the fine resolution, or its stop and target match against
bars far larger than the moves they were built for. A month of one-second
bars over a nearly round-the-clock futures session is ~1.7M bars against
~5.8k at five minutes.

The shortcut: if the rule that ARMS a setup reads nothing but completed
coarse bars, the instants worth descending into are knowable from the coarse
series ALONE. A pre-pass finds them; the replay stays coarse everywhere else.
The difference stays inside BarStream — one of the three things that may
differ between environments — and the rule is injected, so this module never
learns why its windows matter.

THE PRECONDITION, and the only one: the trigger rule must read no intrabar
data. The coarse pass chooses the regions the fine pass replays, so anything
that would have been found in a rejected region is invisible. Add one
intrabar term and the shortcut dies — not noisily, just wrongly.
`verify_agreement` checks the CONSEQUENCE (the two coarse series agree),
never the premise; keeping the premise true is the caller's job, which is why
`TriggerRule` receives completed coarse bars and nothing else.

WINDOWS OUTLIVE THEIR CANDLE. A position can survive the bar that opened it,
and bars matched coarsely while it is open would fill its protection at the
wrong resolution. So a window runs to its SESSION-DAY end, and one that
reaches the close carries one day forward — this engine has no time exit, so
a bracket survives the bell. Anything beyond that (a two-night hold, a
shortened `window_seconds` horizon) is caught by `assert_trades_covered`
rather than hoped about.

THE PARITY GATE: a sparse run must reproduce a full run's `decisionsHash`
exactly, or the shortcut is not usable for that strategy.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Literal, Protocol

from bedivere.core.session_days import SessionDays
from bedivere.core.types import Candle, Timeframe
from bedivere.engine.events import BarEvent

# Given the completed coarse bars up to AND INCLUDING one candidate, does that
# candidate open a window worth replaying finely? Called once per coarse bar
# with the prefix ending at it, so the rule is causal by construction — it
# cannot see its own future even by accident.
#
# The rule must read the coarse series ONLY. See THE PRECONDITION above.
TriggerRule = Callable[[Sequence[Candle]], bool]

# Fine bars for a half-open `(start_unix, end_unix]` range — the same
# convention as `bedivere.data.CandleSource.candles`, so a source binds
# straight to it: `lambda a, b: source.candles(sym, tf, a, b)`.
FineLoader = Callable[[int, int], list[Candle]]


class TradeLike(Protocol):
    """The slice of a TradeRecord the coverage check needs. Structural, so
    coverage can be checked against anything with an entry and an exit —
    including a plain tuple in a test."""

    entry_ts: int
    exit_ts: int


class SparseReplayError(AssertionError):
    """Base for the two guards. AssertionError, because both mean a run's
    numbers are not what they claim to be — this is never a recoverable
    condition to be caught and logged."""


class AgreementError(SparseReplayError):
    """The coarse series the sparse run derived does not select the same
    triggers the pre-pass did — so a cached coarse bar is not the aggregate
    of its fine bars, and the whole shortcut is unsound for this data."""


class CoverageError(SparseReplayError):
    """A trade played out partly or wholly at coarse resolution."""


@dataclass(frozen=True, slots=True)
class FineWindow:
    """A half-open `(start_unix, end_unix]` span replayed at fine
    resolution. `triggers` are the coarse close-stamps that caused it —
    plural because overlapping windows merge, and dropping the causes would
    make the report unreadable."""

    start_unix: int
    end_unix: int
    triggers: tuple[int, ...]

    def contains(self, unix_sec: int) -> bool:
        return self.start_unix < unix_sec <= self.end_unix

    def covers(self, from_unix: int, to_unix: int) -> bool:
        """True when every instant in `(from_unix, to_unix]` is inside this
        window. `from_unix` is EXCLUSIVE to match the window's own half-open
        shape: a bar stamped at `start_unix` is the coarse trigger bar, not a
        fine one."""
        return self.start_unix <= from_unix and to_unix <= self.end_unix

    @property
    def seconds(self) -> int:
        return self.end_unix - self.start_unix

    def to_jsonable(self) -> dict[str, object]:
        return {
            "startUnix": self.start_unix,
            "endUnix": self.end_unix,
            "triggers": list(self.triggers),
        }


@dataclass(frozen=True, slots=True)
class ReplayFidelity:
    """What a run actually replayed, and at what resolution. Rides in the
    result envelope so "was this trade matched finely?" is a lookup rather
    than an argument."""

    mode: Literal["sparse", "full"]
    fine_timeframe: str
    coarse_timeframe: str
    windows: tuple[FineWindow, ...]
    triggers: int
    coarse_bars: int
    fine_bars: int
    span_seconds: int

    @property
    def fine_seconds(self) -> int:
        return sum(w.seconds for w in self.windows)

    def to_jsonable(self) -> dict[str, object]:
        span = self.span_seconds
        return {
            "mode": self.mode,
            "fineTimeframe": self.fine_timeframe,
            "coarseTimeframe": self.coarse_timeframe,
            "triggers": self.triggers,
            "windows": [w.to_jsonable() for w in self.windows],
            "windowCount": len(self.windows),
            "coarseBars": self.coarse_bars,
            "fineBars": self.fine_bars,
            "barsEmitted": self.coarse_bars + self.fine_bars,
            "fineSeconds": self.fine_seconds,
            "spanSeconds": span,
            # The headline: what fraction of wall-time was replayed finely.
            # None rather than 0 for an empty span — undefined is not zero.
            "fineFraction": (
                None if span <= 0 else round(self.fine_seconds / span, 6)
            ),
        }


def select_triggers(coarse: Sequence[Candle], rule: TriggerRule) -> list[int]:
    """Close-stamps of the coarse bars the rule fires on.

    The rule sees `coarse[: i + 1]` — everything up to and including the
    candidate, and nothing after it. That prefix slicing is what makes the
    pre-pass causal: a rule cannot accidentally read its own future, so the
    windows this returns are the windows a live run would have opened.
    """
    return [c.timestamp for i, c in enumerate(coarse) if rule(coarse[: i + 1])]


def _merge_windows(windows: Sequence[FineWindow]) -> list[FineWindow]:
    merged: list[FineWindow] = []
    for w in sorted(windows, key=lambda x: x.start_unix):
        if merged and w.start_unix <= merged[-1].end_unix:
            last = merged[-1]
            merged[-1] = replace(
                last,
                end_unix=max(last.end_unix, w.end_unix),
                triggers=last.triggers + w.triggers,
            )
        else:
            merged.append(w)
    return merged


def _next_session_day(days: SessionDays, after_label: str) -> int | None:
    """Index of the session day following `after_label`, or None."""
    for i, day in enumerate(days.days):
        if day.label == after_label:
            return i + 1 if i + 1 < len(days.days) else None
    return None


def _carry_across_sessions(
    windows: Sequence[FineWindow], days: SessionDays
) -> list[FineWindow]:
    """Extend fine coverage into the next session day for every window that
    reached its own session close.

    A window ending exactly at a session close is the signature of a position
    that MIGHT still be open — this engine has no time exit, so a bracket
    survives the bell and exits on stop or target whenever they come. Its
    exit would then be matched against the next morning's coarse bars, which
    is a different trade. The pre-pass cannot know how long a position rides,
    but it can see that one might be riding, and covering the next session is
    a bounded, pre-computable answer.

    Carry is ONE day deep, deliberately. Cascading it — letting a carried day
    that also reaches its close carry again — degenerates to full fidelity
    for the rest of the run after a single overnight hold, which throws away
    the entire saving to cover a case that is rare and already detected. A
    position spanning two nights escapes this and is caught by
    `assert_trades_covered` instead: a failed run, not a wrong number.
    """
    carried: list[FineWindow] = []
    for w in windows:
        day = days.day_containing(w.end_unix)
        if day is None or w.end_unix < day.end_unix:
            continue
        nxt = _next_session_day(days, day.label)
        if nxt is None:
            continue
        following = days.days[nxt]
        carried.append(
            FineWindow(
                start_unix=following.start_unix,
                end_unix=following.end_unix,
                triggers=(),  # no trigger caused this one; a possible position did
            )
        )
    return _merge_windows(list(windows) + carried)


def build_fine_windows(
    triggers: Sequence[int],
    days: SessionDays,
    *,
    window_seconds: int | None = None,
    carry_across_sessions: bool = True,
) -> list[FineWindow]:
    """Triggers → merged, session-clamped fine windows, ascending.

    By default each trigger opens `(trigger, session_day_end]` — see the
    module docstring for why the window must outlive the trigger's own
    candle. Overlapping windows merge (a run that arms four times in a
    morning descends once), and a trigger AT its session close opens
    nothing: there is no remaining fine time in that day to descend into.

    `window_seconds` shortens that horizon to `trigger + n`, still clamped to
    the session close. It is a REAL trade, not a tuning knob: the default is
    the only horizon that cannot cut a position short, and anything shorter
    bets that no trade outlives it. The bet is checked rather than trusted —
    `assert_trades_covered` fails the run if a trade escapes — so the honest
    way to use this is to shorten it, let the guard tell you when you went
    too far, and keep the number that survives a full-fidelity parity run.

    Without it, savings are bounded by the first trigger of each day: after
    that the rest of the session is fine, which for a strategy that signals
    early most mornings is barely a saving at all.

    `carry_across_sessions` extends coverage one session day past any window
    that reached its own close — see `_carry_across_sessions` for why a
    position that survives the bell needs it. Turn it off only when the
    strategy provably cannot hold overnight; the coverage guard is what tells
    you if you were wrong.
    """
    if window_seconds is not None and window_seconds < 1:
        raise ValueError("build_fine_windows: window_seconds must be >= 1 or None")
    raw: list[FineWindow] = []
    for t in triggers:
        day = days.day_containing(t)
        if day is None or t >= day.end_unix:
            continue
        end = day.end_unix if window_seconds is None else min(t + window_seconds, day.end_unix)
        raw.append(FineWindow(start_unix=t, end_unix=end, triggers=(t,)))

    merged = _merge_windows(raw)
    if carry_across_sessions:
        merged = _carry_across_sessions(merged, days)
    return merged


class SparseReplayStream:
    """A BarStream that emits coarse bars everywhere except inside its fine
    windows, where it emits fine bars instead.

    Each `BarEvent` is stamped with the timeframe of the bar it actually
    carries, so a consumer inspecting the stream can see the resolution
    change. The engine loop does not read that field — it treats every event
    as one closed base bar, which is exactly the point: the loop cannot tell,
    and does not need to.

    Counters are only complete once iteration is exhausted; read `report`
    after the run, not during it.
    """

    __slots__ = (
        "_coarse",
        "_coarse_tf",
        "_days",
        "_fine_loader",
        "_fine_tf",
        "_fine_count",
        "_coarse_count",
        "_rule",
        "_symbol",
        "_triggers",
        "_windows",
    )

    def __init__(
        self,
        *,
        symbol: str,
        fine_timeframe: Timeframe,
        coarse_timeframe: Timeframe,
        coarse_bars: Sequence[Candle],
        rule: TriggerRule,
        fine_loader: FineLoader,
        days: SessionDays,
        window_seconds: int | None = None,
        carry_across_sessions: bool = True,
    ) -> None:
        if coarse_timeframe.period_seconds <= fine_timeframe.period_seconds:
            raise ValueError(
                f"SparseReplayStream: coarse TF {coarse_timeframe.value} must be coarser "
                f"than the fine TF {fine_timeframe.value}"
            )
        for prev, cur in zip(coarse_bars, coarse_bars[1:], strict=False):
            if cur.timestamp <= prev.timestamp:
                raise ValueError(
                    f"SparseReplayStream: coarse bar {cur.timestamp} is not after "
                    f"{prev.timestamp} — sort and de-duplicate at the loader"
                )
        self._symbol = symbol
        self._fine_tf = fine_timeframe
        self._coarse_tf = coarse_timeframe
        self._days = days
        self._coarse = list(coarse_bars)
        self._fine_loader = fine_loader
        # Kept so the stream can verify ITSELF against the run's derived
        # coarse series. A guard the caller has to remember to call is a
        # guard that eventually goes uncalled.
        self._rule = rule
        self._triggers = select_triggers(self._coarse, rule)
        self._windows = build_fine_windows(
            self._triggers,
            days,
            window_seconds=window_seconds,
            carry_across_sessions=carry_across_sessions,
        )
        self._coarse_count = 0
        self._fine_count = 0

    # ---------- inspection ----------

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def timeframe(self) -> Timeframe:
        """The FINE timeframe — the resolution the run is nominally at, and
        the one a composition must use as its `base_timeframe`."""
        return self._fine_tf

    @property
    def first_ts(self) -> int | None:
        return self._coarse[0].timestamp if self._coarse else None

    @property
    def coarse_timeframe(self) -> Timeframe:
        return self._coarse_tf

    @property
    def days(self) -> SessionDays:
        """The session geometry the windows were clamped to — the coverage
        guard needs it to tell an overnight gap from a coarse region."""
        return self._days

    @property
    def triggers(self) -> list[int]:
        """The pre-pass's selection — the input to `verify_agreement`."""
        return list(self._triggers)

    def verify(self, run_coarse: Sequence[Candle]) -> list[int]:
        """Check this run's derived coarse series against the pre-pass, using
        the rule the stream was built with. Raises `AgreementError` on any
        difference. `run_backtest` calls this automatically."""
        return verify_agreement(
            run_coarse=run_coarse, rule=self._rule, expected=self._triggers
        )

    @property
    def windows(self) -> list[FineWindow]:
        return list(self._windows)

    @property
    def report(self) -> ReplayFidelity:
        """Valid once iteration is exhausted."""
        first = self._coarse[0].timestamp if self._coarse else 0
        last = self._coarse[-1].timestamp if self._coarse else 0
        return ReplayFidelity(
            mode="sparse",
            fine_timeframe=self._fine_tf.value,
            coarse_timeframe=self._coarse_tf.value,
            windows=tuple(self._windows),
            triggers=len(self._triggers),
            coarse_bars=self._coarse_count,
            fine_bars=self._fine_count,
            span_seconds=max(0, last - first),
        )

    # ---------- the stream ----------

    def __iter__(self) -> Iterator[BarEvent]:
        self._coarse_count = 0
        self._fine_count = 0
        coarse = self._coarse
        idx = 0
        last_ts: int | None = None

        for window in self._windows:
            # Coarse bars up to and INCLUDING the window's opening stamp. The
            # bar at `start_unix` is the trigger's own candle: the rule read
            # its close, so it belongs to the coarse region, and the fine bars
            # that follow build the bucket after it.
            while idx < len(coarse) and coarse[idx].timestamp <= window.start_unix:
                yield self._event(coarse[idx], self._coarse_tf)
                last_ts = coarse[idx].timestamp
                self._coarse_count += 1
                idx += 1

            for candle in self._fine_bars(window):
                if last_ts is not None and candle.timestamp <= last_ts:
                    raise ValueError(
                        f"SparseReplayStream: fine bar {candle.timestamp} in window "
                        f"({window.start_unix}, {window.end_unix}] is not after the "
                        f"previously emitted bar {last_ts} — the fine loader returned "
                        "bars that overlap the coarse region"
                    )
                yield self._event(candle, self._fine_tf)
                last_ts = candle.timestamp
                self._fine_count += 1

            # The window replaced these; emitting them too would double-count
            # the bucket and push the view's bars out of order.
            while idx < len(coarse) and coarse[idx].timestamp <= window.end_unix:
                idx += 1

        while idx < len(coarse):
            yield self._event(coarse[idx], self._coarse_tf)
            self._coarse_count += 1
            idx += 1

    def _fine_bars(self, window: FineWindow) -> list[Candle]:
        bars = self._fine_loader(window.start_unix, window.end_unix)
        for prev, cur in zip(bars, bars[1:], strict=False):
            if cur.timestamp <= prev.timestamp:
                raise ValueError(
                    f"SparseReplayStream: fine loader returned {cur.timestamp} after "
                    f"{prev.timestamp} for window ({window.start_unix}, "
                    f"{window.end_unix}] — bars must be strictly ascending"
                )
        if bars and not (
            window.contains(bars[0].timestamp) and window.contains(bars[-1].timestamp)
        ):
            raise ValueError(
                f"SparseReplayStream: fine loader returned bars outside the requested "
                f"window ({window.start_unix}, {window.end_unix}] — got "
                f"{bars[0].timestamp}..{bars[-1].timestamp}"
            )
        return bars

    def _event(self, candle: Candle, timeframe: Timeframe) -> BarEvent:
        return BarEvent(
            ts=candle.timestamp,
            symbol=self._symbol,
            timeframe=timeframe,
            candle=candle,
        )


# ---------- the two guards ----------


def verify_agreement(
    *,
    run_coarse: Sequence[Candle],
    rule: TriggerRule,
    expected: Sequence[int],
) -> list[int]:
    """Re-derive the triggers from the coarse series the RUN produced and
    check them against the pre-pass's selection. Returns the re-derived
    list; raises `AgreementError` on any difference.

    This is the check that turns the module's one data assumption — a cached
    coarse bar equals the aggregate of its fine bars — into something the run
    verifies rather than trusts. The pre-pass read cached coarse bars; the
    run derived its own coarse series, partly from those same bars and partly
    by aggregating fine ones. If the two disagree about a single trigger, the
    cache and the fine data describe different markets and every number the
    run produced is suspect.

    Comparison is restricted to the run's own coarse span: the pre-pass may
    have been handed a wider series than the run replayed, and triggers
    outside the span were never the run's to reproduce.
    """
    actual = select_triggers(run_coarse, rule)
    if not run_coarse:
        return actual
    first, last = run_coarse[0].timestamp, run_coarse[-1].timestamp
    scoped = [t for t in expected if first <= t <= last]
    if actual != scoped:
        missing = sorted(set(scoped) - set(actual))
        extra = sorted(set(actual) - set(scoped))
        raise AgreementError(
            "sparse replay disagrees with the coarse pre-pass: "
            f"{len(missing)} trigger(s) the pre-pass found are absent from the run's "
            f"coarse series {missing[:5]}, {len(extra)} the run found were not "
            f"pre-selected {extra[:5]}. A cached coarse bar is not the aggregate of "
            "its fine bars, or the trigger rule reads intrabar data (see the "
            "bedivere.streams.sparse module docstring)."
        )
    return actual


def _is_covered(
    entry_ts: int, exit_ts: int, windows: Sequence[FineWindow], days: SessionDays
) -> bool:
    """Was every BAR this trade was open for replayed finely?

    The check is against SESSION TIME, not wall time, and that distinction is
    the whole subtlety. A position held overnight spans two windows with a
    gap between them — but no bars exist in that gap, so it is not a coverage
    hole. Requiring one window to contain the whole span would fail every
    overnight trade for a reason that does not exist.

    So the span is cut into per-session-day pieces and each piece must sit
    inside a single window. That is sound because windows are merged and
    session-clamped: within one day they are disjoint and non-adjacent, so a
    piece spanning two of them genuinely straddles a coarse region.

    The lower bound is `entry_ts - 1` because the entry BAR itself, stamped
    at `entry_ts`, must be fine — and a window's `start_unix` is exclusive.
    """
    for day in days.overlapping(entry_ts, exit_ts):
        lo = max(entry_ts - 1, day.start_unix)
        hi = min(exit_ts, day.end_unix)
        if lo >= hi:
            continue
        if not any(w.covers(lo, hi) for w in windows):
            return False
    return True


def trade_coverage(
    trades: Sequence[TradeLike], windows: Sequence[FineWindow], days: SessionDays
) -> list[bool]:
    """Per-trade: did this round trip play out entirely at fine resolution?
    Same order as `trades`. This is what labels a result's trades rather
    than labelling only the run — the distinction a backtester that silently
    mixes resolutions cannot make."""
    return [_is_covered(t.entry_ts, t.exit_ts, windows, days) for t in trades]


def assert_trades_covered(
    trades: Sequence[TradeLike], windows: Sequence[FineWindow], days: SessionDays
) -> None:
    """Fail the run if any trade escaped a fine window.

    The hole this closes is a position riding past the coverage it was given:
    a trade whose exit is matched against coarse bars is not the trade the
    strategy would have taken, and a run containing one cannot be compared
    with a full-fidelity run — so it stops here rather than being averaged
    into a metric.

    One overnight hold is handled by the carry-forward in
    `build_fine_windows`; a position spanning TWO nights, or one held past a
    shortened `window_seconds` horizon, lands here.
    """
    covered = trade_coverage(trades, windows, days)
    escaped = [i for i, ok in enumerate(covered) if not ok]
    if not escaped:
        return
    detail = ", ".join(
        f"#{i} ({trades[i].entry_ts}..{trades[i].exit_ts})" for i in escaped[:5]
    )
    raise CoverageError(
        f"{len(escaped)} of {len(trades)} trade(s) played out partly at COARSE "
        f"resolution: {detail}. Their stops and targets were matched against bars "
        "larger than the moves they were built for. Usual causes: a "
        "`window_seconds` horizon shorter than the strategy holds, or a position "
        "spanning more than one night. Widen the windows or replay in full."
    )
