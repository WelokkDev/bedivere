"""MarketView — the incremental multi-timeframe view (push-shaped).

`update(bar)` appends one closed base-TF bar and advances every derived
HTF bucket in place; when a bucket completes, the closed candle is appended
to that TF's completed list, that TF's observer (if any) is notified, and
the close is reported to the caller. There is no `view_at(as_of)` anywhere —
state IS the view.

Contract parity with the frozen view (the pull-shaped test oracle,
bedivere.view.frozen): after update(bar), for every derived TF,

    completed(tf) == FrozenSource.view_at(bar.timestamp).completed[tf]
    forming(tf)   == the straddling bucket's partial aggregate (or None)

using the same bucket arithmetic (bedivere.core.buckets — the load-bearing
`-1`, session-day-clamped period ends) and the same aggregate reduction
(open=first, high=max, low=min, close/timestamp=last, volume=sum, folded in
arrival order so floats are bit-identical to aggregate_bucket). Signal code
should read completed bars ONLY; the forming aggregate is exposed for
inspection but is never reported as a close.

Observers are the extension seam: pass `observer_factory` to attach one
stateful observer per derived TF (an incremental detector, an indicator
fold, ...). Each observer is called exactly once per completed bar of its
TF, with the full completed series after the append; whatever it returns
rides out on `HtfClose.payload` for that close.

Base TF is a constructor parameter: 15s/5m/15m all work; the derived set
must be strictly coarser than the base.

Session-day boundaries arrive as resolved SessionDays data (build them with
bedivere.sessions or hand over your own). Bars outside every session-day
stay in `primary` but never enter HTF aggregation — mirroring the frozen
view and aggregator.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from bedivere.core.buckets import bucket_index_of, bucket_period_end
from bedivere.core.session_days import SessionDays
from bedivere.core.types import Candle, Timeframe


class CloseObserver(Protocol):
    """A per-TF incremental observer. `on_completed` is called once per
    completed bar of its TF, AFTER the bar was appended, with the TF's full
    completed series (live list — do not mutate). The return value (None for
    "nothing this close") is forwarded on `HtfClose.payload`."""

    def on_completed(self, completed: Sequence[Candle]) -> object | None: ...


ObserverFactory = Callable[[Timeframe], "CloseObserver | None"]


@dataclass(frozen=True, slots=True)
class HtfClose:
    """One HTF bucket completion, reported by update(). `index` is the
    candle's position in that TF's completed list; `payload` is whatever the
    TF's observer returned for this close (None without an observer)."""

    tf: Timeframe
    candle: Candle
    index: int
    payload: object | None


@dataclass(slots=True)
class _OpenBucket:
    day_label: str
    index: int
    period_end: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    last_ts: int


@dataclass(slots=True)
class _TfState:
    tf: Timeframe
    period_seconds: int
    completed: list[Candle]
    observer: CloseObserver | None
    open_bucket: _OpenBucket | None


class MarketView:
    def __init__(
        self,
        *,
        base_tf: Timeframe,
        derived_tfs: Sequence[Timeframe],
        days: SessionDays,
        observer_factory: ObserverFactory | None = None,
    ) -> None:
        base_period = base_tf.period_seconds  # raises on "1d"
        ordered = sorted(set(derived_tfs), key=lambda tf: tf.order)
        for tf in ordered:
            if tf.period_seconds <= base_period:
                raise ValueError(
                    f'MarketView: derived TF "{tf.value}" is not coarser than the base TF "{base_tf.value}"'
                )
        self._base_tf = base_tf
        self._days = days
        self._resolve_day = days.make_resolver()
        self._primary: list[Candle] = []
        self._states: dict[Timeframe, _TfState] = {
            tf: _TfState(
                tf=tf,
                period_seconds=tf.period_seconds,
                completed=[],
                observer=observer_factory(tf) if observer_factory is not None else None,
                open_bucket=None,
            )
            for tf in ordered
        }

    # ---------- push ----------

    def update(self, bar: Candle) -> list[HtfClose]:
        """Append one closed base-TF bar; returns this instant's HTF closes
        in coarseness-ascending TF order. Bars must arrive strictly
        ascending — a regression or duplicate raises (the stream owns
        ordering; a violation here is an upstream bug)."""
        if self._primary and bar.timestamp <= self._primary[-1].timestamp:
            raise ValueError(
                f"MarketView.update: bar {bar.timestamp} is not after {self._primary[-1].timestamp} — stream out of order"
            )
        self._primary.append(bar)
        ts = bar.timestamp
        sd = self._resolve_day(ts)

        closes: list[HtfClose] = []
        for state in self._states.values():
            ob = state.open_bucket

            # Out-of-session bar: stays in primary only, but its timestamp
            # still proves any open bucket's period-end is behind us.
            if sd is None:
                if ob is not None:
                    assert ob.period_end <= ts, (
                        f"open {state.tf.value} bucket period_end {ob.period_end} > out-of-session bar {ts}"
                    )
                    self._finalize(state, closes)
                continue

            key_index = bucket_index_of(ts, sd.start_unix, state.period_seconds)

            # 1) A bar that doesn't extend the open bucket proves the bucket
            #    closed (its period-end is behind us — asserted, not assumed).
            if ob is not None and (ob.day_label != sd.label or ob.index != key_index):
                assert ob.period_end <= ts, (
                    f"open {state.tf.value} bucket period_end {ob.period_end} > bar ts {ts}"
                )
                self._finalize(state, closes)
                ob = None

            # 2) Open or extend the bucket this bar belongs to.
            if ob is None:
                ob = _OpenBucket(
                    day_label=sd.label,
                    index=key_index,
                    period_end=bucket_period_end(
                        sd.start_unix, sd.end_unix, key_index, state.period_seconds
                    ),
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                    last_ts=ts,
                )
                state.open_bucket = ob
            else:
                # Same fold order as aggregate_bucket → bit-identical floats.
                if bar.high > ob.high:
                    ob.high = bar.high
                if bar.low < ob.low:
                    ob.low = bar.low
                ob.close = bar.close
                ob.volume += bar.volume
                ob.last_ts = ts

            # 3) A bar AT the period end (grid close or session-end stub)
            #    completes its own bucket immediately.
            if ob.period_end <= ts:
                self._finalize(state, closes)

        return closes

    def _finalize(self, state: _TfState, closes: list[HtfClose]) -> None:
        ob = state.open_bucket
        assert ob is not None
        candle = Candle(
            timestamp=ob.last_ts,
            open=ob.open,
            high=ob.high,
            low=ob.low,
            close=ob.close,
            volume=ob.volume,
        )
        state.completed.append(candle)
        state.open_bucket = None
        payload = (
            state.observer.on_completed(state.completed)
            if state.observer is not None
            else None
        )
        closes.append(
            HtfClose(tf=state.tf, candle=candle, index=len(state.completed) - 1, payload=payload)
        )

    # ---------- accessors (live lists — callers must not mutate) ----------

    @property
    def base_tf(self) -> Timeframe:
        return self._base_tf

    @property
    def derived_tfs(self) -> tuple[Timeframe, ...]:
        return tuple(self._states.keys())

    @property
    def days(self) -> SessionDays:
        return self._days

    def as_of(self) -> int | None:
        """The last appended base-bar close-stamp (None before any bar)."""
        return self._primary[-1].timestamp if self._primary else None

    @property
    def primary(self) -> list[Candle]:
        return self._primary

    def completed(self, tf: Timeframe) -> list[Candle]:
        """The append-only completed series for a derived TF — the ONLY bars
        safe for signal detection. Never a copy: strategies index into it."""
        return self._state_of(tf).completed

    def forming(self, tf: Timeframe) -> Candle | None:
        """The straddling bucket's aggregate-so-far (NOT flagged partial),
        or None. Inspection only — never reported as a close."""
        ob = self._state_of(tf).open_bucket
        if ob is None:
            return None
        return Candle(
            timestamp=ob.last_ts,
            open=ob.open,
            high=ob.high,
            low=ob.low,
            close=ob.close,
            volume=ob.volume,
        )

    def _state_of(self, tf: Timeframe) -> _TfState:
        state = self._states.get(tf)
        if state is None:
            raise KeyError(
                f'MarketView: "{tf.value}" is not a tracked derived TF (have: {[t.value for t in self._states]})'
            )
        return state
