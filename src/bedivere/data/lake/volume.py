"""Volume-clock bars for research, independent of the fixed-timeframe engine.

An accumulator belongs to one instrument and one session. Trades use receive
time, with input order breaking ties; time-bar inputs are an approximation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from statistics import median
from typing import Literal

from bedivere.core.session_days import SessionDay, SessionDays

NS = 1_000_000_000
Boundary = Literal["whole_trade", "split_trade", "nearest_second"]
ApproxBoundary = Literal["whole_trade", "nearest_second"]


@dataclass(frozen=True, slots=True)
class VolumeSpec:
    threshold: int
    source: str = "trades"
    """Part of the dataset's identity: `trades`, `ohlcv-1s` (stored one-second
    bars), or `trades-1s` (one-second bars rebuilt from the same trades)."""

    boundary: Boundary = "whole_trade"

    def __post_init__(self) -> None:
        if type(self.threshold) is not int or not 0 < self.threshold < 2**63:
            raise ValueError("volume threshold must be a positive int64 integer")
        if self.boundary not in ("whole_trade", "split_trade", "nearest_second"):
            raise ValueError("boundary must be whole_trade, split_trade, or nearest_second")
        if self.source == "trades" and self.boundary == "nearest_second":
            raise ValueError("nearest_second requires source=ohlcv-1s or trades-1s")
        if self.source != "trades":
            if self.source not in ("ohlcv-1s", "trades-1s"):
                raise ValueError("this builder supports sources trades, ohlcv-1s and trades-1s")
            if self.boundary == "split_trade":
                raise ValueError("split_trade requires trades; OHLCV cannot be split faithfully")

    def definition(self, days: SessionDays) -> str:
        return json.dumps(
            {
                **asdict(self),
                "kind": "volume",
                "version": 1,
                "clock": "ts_recv",
                "tie_break": "source_order",
                "reset": "session",
                "remainder": "emit_flagged",
                "calendar": days.template,
                "timezone": days.timezone,
                "eligibility": "positive_nonsynthetic_ohlcv"
                if self.source == "ohlcv-1s"
                else "positive_trades_no_bad_recv",
                # Only nearest_second adds keys, so existing definition hashes hold.
                **(
                    {
                        "targets": "session_cumulative_multiples",
                        "boundary_ties": "earlier_second",
                        "shared_boundaries": "coalesce_no_empty_bars",
                        "availability": "crossing_second_end",
                    }
                    if self.boundary == "nearest_second"
                    else {}
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def key(self, days: SessionDays) -> str:
        return hashlib.sha256(self.definition(days).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class VolumeInput:
    start_ns: int
    end_ns: int
    open: float
    high: float
    low: float
    close: float
    volume: int
    instrument_id: int

    @classmethod
    def trade(cls, ts_ns: int, price: float, size: int, instrument_id: int) -> VolumeInput:
        return cls(ts_ns, ts_ns, price, price, price, price, size, instrument_id)


@dataclass(frozen=True, slots=True)
class VolumeBar:
    bar_id: int
    start_ns: int
    end_ns: int
    available_ns: int
    open: float
    high: float
    low: float
    close: float
    volume: int
    instrument_id: int
    input_count: int
    vwap: float | None
    is_partial: bool


@dataclass(frozen=True, slots=True)
class FormingBar:
    """The bar in progress: inputs accepted since the last boundary.

    Not a `VolumeBar`: it is not yet available, and whether it closes on its
    threshold or as a remainder is unknown. `bar_id` is the id it will take.
    """

    bar_id: int
    start_ns: int
    end_ns: int
    open: float
    high: float
    low: float
    close: float
    volume: int
    instrument_id: int
    input_count: int
    vwap: float | None


class VolumeAccumulator:
    """Stateful threshold crossing; no source row, volume, or wick is invented.

    `input_count` is not additive in split mode, where one trade can feed
    several bars. Remainders become available at session close. nearest_second
    keeps whole candles and can close below the target.
    """

    def __init__(self, spec: VolumeSpec, day: SessionDay) -> None:
        self.spec = spec
        self.day = day
        self._last_end = 0
        self._instrument: int | None = None
        self._finished = False
        self._bar_id = 0
        self._cumulative = 0
        self._volume = self._count = 0
        self._start = self._end = 0
        self._open = self._high = self._low = self._close = self._pv = 0.0

    def push(self, row: VolumeInput) -> list[VolumeBar]:
        if self._finished:
            raise ValueError("cannot push after session finish")
        if (
            type(row.start_ns) is not int
            or type(row.end_ns) is not int
            or not 0 < row.start_ns <= row.end_ns < 2**63
        ):
            raise ValueError("input timestamps must be positive int64 nanoseconds")
        if not self.day.start_unix * NS <= row.start_ns <= row.end_ns <= self.day.end_unix * NS:
            raise ValueError("input lies outside the session")
        if self.spec.source == "trades":
            if row.end_ns >= self.day.end_unix * NS:
                raise ValueError("trade timestamps use [session open, session close)")
            if row.start_ns != row.end_ns or not row.open == row.high == row.low == row.close:
                raise ValueError("trades must carry one timestamp and one price")
        elif row.end_ns - row.start_ns != NS or row.start_ns % NS:
            raise ValueError("approximation inputs must be complete aligned one-second intervals")
        if row.start_ns < self._last_end:
            raise ValueError("input is out of order or its intervals overlap")
        if type(row.volume) is not int or not 0 <= row.volume < 2**63:
            raise ValueError("input volume must be a non-negative int64 integer")
        if type(row.instrument_id) is not int or not 0 < row.instrument_id < 2**32:
            raise ValueError("instrument_id must be a known positive uint32")
        prices = (row.open, row.high, row.low, row.close)
        if not all(math.isfinite(p) for p in prices):
            raise ValueError("non-finite price")
        if not row.low <= min(row.open, row.close) <= max(row.open, row.close) <= row.high:
            raise ValueError("inconsistent OHLC prices")
        if self._instrument is not None and self._instrument != row.instrument_id:
            raise ValueError("a session cannot mix instruments; split the contract series first")
        self._last_end = row.end_ns
        self._instrument = row.instrument_id
        if self.spec.boundary == "nearest_second":
            return self._push_nearest(row)
        remaining = row.volume
        out: list[VolumeBar] = []
        while remaining:
            size = (
                remaining
                if self.spec.boundary == "whole_trade"
                else min(remaining, self.spec.threshold - self._volume)
            )
            self._add(row, size)
            remaining -= size
            if self._volume >= self.spec.threshold:
                out.append(self._emit(partial=False))
        return out

    def _add(self, row: VolumeInput, size: int) -> None:
        if not size:
            return
        if self._volume == 0:
            self._start, self._open = row.start_ns, row.open
            self._high, self._low = row.high, row.low
        self._high, self._low = max(self._high, row.high), min(self._low, row.low)
        self._end, self._close = row.end_ns, row.close
        self._volume += size
        self._count += 1
        self._pv += row.close * size

    def _push_nearest(self, row: VolumeInput) -> list[VolumeBar]:
        before = self._cumulative
        after = before + row.volume
        threshold = self.spec.threshold
        first_target = (before // threshold + 1) * threshold
        out: list[VolumeBar] = []
        if after >= first_target:
            # Every crossed target picks one of these two boundaries, so testing
            # the first and last is enough: O(1) however large the second.
            if 2 * (first_target - before) <= row.volume and self._volume:
                out.append(self._emit(partial=False, available_ns=row.end_ns))
            self._add(row, row.volume)
            last_target = after // threshold * threshold
            if 2 * (last_target - before) > row.volume:
                out.append(self._emit(partial=False))
        else:
            self._add(row, row.volume)
        self._cumulative = after
        return out

    def finish(self) -> list[VolumeBar]:
        self._finished = True
        return [self._emit(partial=True)] if self._volume else []

    def forming(self) -> FormingBar | None:
        """The bar in progress, or None when nothing has accumulated or after
        `finish`. Under `nearest_second`, a boundary chosen later is not anticipated."""
        if self._finished or not self._volume:
            return None
        assert self._instrument is not None
        return FormingBar(
            self._bar_id,
            self._start,
            self._end,
            self._open,
            self._high,
            self._low,
            self._close,
            self._volume,
            self._instrument,
            self._count,
            self._pv / self._volume if self.spec.source == "trades" else None,
        )

    def _emit(self, *, partial: bool, available_ns: int | None = None) -> VolumeBar:
        assert self._instrument is not None
        if partial:
            available_ns = self.day.end_unix * NS
        elif available_ns is None:
            available_ns = self._end
        bar = VolumeBar(
            self._bar_id,
            self._start,
            self._end,
            available_ns,
            self._open,
            self._high,
            self._low,
            self._close,
            self._volume,
            self._instrument,
            self._count,
            self._pv / self._volume if self.spec.source == "trades" else None,
            partial,
        )
        self._bar_id += 1
        self._volume = self._count = 0
        self._pv = 0.0
        return bar


def summarize(bars: list[VolumeBar], threshold: int) -> dict[str, int | float | None]:
    """Descriptive comparison, not a claim about model quality or matched bars."""
    full = [b for b in bars if not b.is_partial]
    deviations = [abs(b.volume - threshold) for b in full]

    def p95(values: list[float]) -> float | None:
        return sorted(values)[math.ceil(len(values) * 0.95) - 1] if values else None

    return {
        "bars": len(bars),
        "full_bars": len(full),
        "partial_bars": len(bars) - len(full),
        "volume": sum(b.volume for b in bars),
        "input_contributions": sum(b.input_count for b in bars),
        "partial_volume": sum(b.volume for b in bars if b.is_partial),
        "overshoot_p95": p95([float(max(0, b.volume - threshold)) for b in full]),
        "overshoot_max": max((max(0, b.volume - threshold) for b in full), default=0),
        "undershoot_max": max((max(0, threshold - b.volume) for b in full), default=0),
        "volume_deviation_median": median(deviations) if deviations else None,
        "volume_deviation_p95": p95([float(value) for value in deviations]),
        "volume_deviation_max": max(deviations, default=0),
        "duration_seconds_p95": p95([(b.end_ns - b.start_ns) / NS for b in full]),
    }
