"""The lake as a `CandleSource`, and the gates a run passes before it reads.

Fail-closed, and no fallback. Two series that select contracts by different
rules hold different instruments on the same dates, so they differ by a roll gap
on any close-stamp near a roll; a "lake first, something else on miss" resolver
would run a warm-up on one series and the tradeable window on the other with
nothing downstream able to notice. A run names one source, or fails at the start.

Three gates, answering three different questions:

    require_timeframes   does this (symbol, timeframe) exist here at all?
    require_coverage     does it reach the session-day the window runs to?
    the as-of horizon    (adjusted views only) is this bar inside the pin?

`CoverageCheckedSource` makes the second hold by construction rather than by
every read site remembering to ask — the backtest runner, the live warm-up, the
sparse fine-window loader and the coverage CLI all read. It is handed the run's
session calendar because "reaches far enough" is a question about session-days,
not seconds: a store whose newest close is the one before the window's end is a
whole session short, and no time tolerance tight enough to catch that also
admits a warm-up that runs to `now` on a store current through the last close.

The `bedivere.data.lake` modules are imported lazily here, so a clone that only
replays a CSV never needs DuckDB, and a missing `lake` extra reports itself
instead of raising `ModuleNotFoundError` three layers down.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, cast

from bedivere.core.session_days import SessionDay, SessionDays
from bedivere.core.types import Candle, Timeframe
from bedivere.data.port import CandleSource

if TYPE_CHECKING:  # pragma: no cover - typing only
    from bedivere.data.lake.layout import SeriesId

LAKE_EXTRA_HINT = (
    "install the lake extra (`pip install 'bedivere[lake]'`, or "
    "`uv sync --extra lake`) — DuckDB and the DBN decoder are not part of the "
    "base install, so a clone that only replays a CSV stays dependency-free"
)


class BarSourceError(Exception):
    """The requested source cannot serve what a run needs."""


# The calendar-less backstop, and coarse by necessity. Measured in seconds, a
# store that is exactly one session behind the window looks the same as a
# complete store read to `now` late in the next session — both sit up to one
# close-to-close interval past the newest bar — so no tolerance tight enough to
# refuse the first admits the second. This catches a store that is DAYS behind;
# the session-aware check in `require_coverage`, which every spec-built source
# gets, is what catches one session behind.
DEFAULT_MAX_SHORTFALL = 86_400

AS_TRADED = "as_traded"
BACK_ADJUSTED = "back_adjusted"


def lake_module(module: str) -> Any:
    """`bedivere.data.lake.<module>`, or a `BarSourceError` that says what to do.

    Public because the CLI needs the same courtesy: `bedivere-data list` on a
    base install should name the missing extra rather than raise
    `ModuleNotFoundError` from inside an import statement.
    """
    try:
        import importlib

        return importlib.import_module(f"bedivere.data.lake.{module}")
    except ImportError as e:
        raise BarSourceError(f"the lake {module} stack failed to import: {e}\n{LAKE_EXTRA_HINT}") from e


# ---------- the lake ----------


class LakeCandleSource:
    """The Parquet bar lake, bound to one dataset and series.

    `series` is the contract-selection rule (`v.0`, `local.v.0`, `raw.NQZ5`),
    part of the identity because two selection rules are two different price
    series — see `lake.layout`.
    """

    __slots__ = ("_dataset", "_root", "_series")

    def __init__(self, dataset: str, series: str, root: Path | None = None) -> None:
        self._dataset = dataset
        self._series = series
        self._root = lake_module("query").lake_root(root)

    # ---------- identity ----------

    def describe(self) -> str:
        return f"lake:{self._dataset}/{self._series}"

    @property
    def root(self) -> Path:
        return self._root

    @property
    def price_basis(self) -> str:
        """As-traded by construction: the lake stores the vendor's raw
        per-contract prices and never bakes a roll offset into them."""
        return AS_TRADED

    @property
    def pinned_as_of(self) -> str | None:
        return None

    def series_id(self, symbol: str) -> SeriesId:
        return lake_module("layout").SeriesId(
            dataset=self._dataset, symbol=symbol, series=self._series
        )

    # ---------- CandleSource ----------

    def candles(
        self, symbol: str, timeframe: Timeframe, start_unix: int, end_unix: int
    ) -> list[Candle]:
        return lake_module("read").read_bars(
            self._root, self.series_id(symbol), timeframe, start_unix, end_unix
        )

    # ---------- inspection ----------

    def has(self, symbol: str, timeframe: Timeframe) -> bool:
        return bool(lake_module("layout").stored_labels(self._root, self.series_id(symbol), timeframe))

    def latest(self, symbol: str, timeframe: Timeframe) -> int | None:
        return lake_module("read").latest_stored_ts(self._root, self.series_id(symbol), timeframe)

    def covered_days(
        self, symbol: str, timeframe: Timeframe, days: SessionDays
    ) -> frozenset[str]:
        """Which of `days` this series holds — a directory listing, not a scan.

        The lake partitions by session label, so the coverage map is exact and
        free: no Parquet footer is opened, and a gap in the MIDDLE of a series is
        visible rather than inferred away by a first/last comparison.
        """
        stored = frozenset(
            lake_module("layout").stored_labels(self._root, self.series_id(symbol), timeframe)
        )
        return frozenset(d.label for d in days.days if d.label in stored)


class BackAdjustedLakeSource:
    """A lake series expressed in its back-adjusted view, pinned to an as-of date.

    `lake.adjust` is applied at read time — the store stays raw and nothing
    adjusted is ever persisted. The as-of date is part of the IDENTITY: it pins
    the roll set the offsets were computed from, so the same label always means
    the same numbers, and two pins are two price series readable side by side.
    Reads past the as-of horizon refuse, which is the stale-view trap the pin
    exists to prevent.

    Use this for system testing and equity curves. Anything keyed on absolute
    price levels — a zone, a round number, a level from a chart — wants the raw
    series underneath.
    """

    __slots__ = ("_adjustments", "_as_of", "_inner")

    def __init__(self, inner: LakeCandleSource, as_of: str) -> None:
        adjust = lake_module("adjust")
        try:
            adjust.require_iso_date(as_of, 'back_adjusted "asOf"')
        except adjust.AdjustmentError as e:
            raise BarSourceError(str(e)) from e
        self._inner = inner
        self._as_of = as_of
        # One derivation per (symbol, timeframe) per instance, so a run sees one
        # consistent view even if the lake is topped up underneath it.
        self._adjustments: dict[tuple[str, Timeframe], Any] = {}

    # ---------- identity ----------

    def describe(self) -> str:
        return f"{self._inner.describe()}@backadj:{self._as_of}"

    @property
    def price_basis(self) -> str:
        return BACK_ADJUSTED

    @property
    def pinned_as_of(self) -> str:
        return self._as_of

    # ---------- CandleSource ----------

    def candles(
        self, symbol: str, timeframe: Timeframe, start_unix: int, end_unix: int
    ) -> list[Candle]:
        adj = self._adjustment(symbol, timeframe)
        # The request check carries `require_coverage`'s slack, so a
        # close-stamped `end` at the pin itself is not refused over the seconds
        # between the last trade and the close. The serve check below is what
        # actually holds the pin: no bar past the horizon is returned.
        if end_unix > adj.horizon_ts + DEFAULT_MAX_SHORTFALL:
            raise BarSourceError(
                f"{self.describe()}: the requested window ends at {end_unix}, past the as-of "
                f"horizon {adj.horizon_ts} (last stored bar of {self._as_of}). Reading an "
                "adjusted series beyond its pinned date reintroduces the stale-view problem "
                "the pin exists to prevent — re-derive with a later asOf instead."
            )
        raw = self._inner.candles(symbol, timeframe, start_unix, end_unix)
        if raw and raw[-1].timestamp > adj.horizon_ts:
            raise BarSourceError(
                f"{self.describe()}: the window reaches bars past the as-of horizon "
                f"{adj.horizon_ts} (first offender at {raw[-1].timestamp}) — the pin "
                "guarantees nothing beyond its date is served. Re-derive with a later asOf."
            )
        return list(adj.apply(raw))

    # ---------- inspection ----------

    def has(self, symbol: str, timeframe: Timeframe) -> bool:
        return self._inner.has(symbol, timeframe)

    def latest(self, symbol: str, timeframe: Timeframe) -> int | None:
        """The as-of horizon, not the store's newest bar — bars past the pin are
        not servable through this view, and a freshness gate must see that."""
        if self._inner.latest(symbol, timeframe) is None:
            return None
        return int(self._adjustment(symbol, timeframe).horizon_ts)

    def covered_days(
        self, symbol: str, timeframe: Timeframe, days: SessionDays
    ) -> frozenset[str]:
        """The inner series' days, truncated at the pin.

        A day past the horizon is stored but not servable (`candles` refuses
        it), so reporting it as covered would move the failure from plan time to
        run time. ISO labels compare lexicographically, which is why the as-of
        spelling must be canonical.
        """
        inner = self._inner.covered_days(symbol, timeframe, days)
        return frozenset(label for label in inner if label <= self._as_of)

    def _adjustment(self, symbol: str, timeframe: Timeframe) -> Any:
        """Derive once per (symbol, timeframe), caching refusals too so a failed
        derivation costs one full-series scan rather than one per read."""
        key = (symbol, timeframe)
        cached = self._adjustments.get(key)
        if cached is not None:
            if isinstance(cached, BarSourceError):
                raise cached
            return cached
        adjust = lake_module("adjust")
        schema = lake_module("schema")
        try:
            adj = adjust.back_adjustment(
                self._inner.root, self._inner.series_id(symbol), timeframe, self._as_of
            )
        except (adjust.AdjustmentError, schema.BarSchemaError) as e:
            # BarSchemaError included so schema drift surfaces as this seam's
            # error type rather than a raw traceback.
            failure = BarSourceError(str(e))
            self._adjustments[key] = failure
            raise failure from e
        self._adjustments[key] = adj
        return adj


# ---------- the coverage wrapper ----------


class CoverageCheckedSource:
    """Wraps a source so a read that overruns the store raises instead of quietly
    returning a short series.

    `days` is the run's session calendar. With it the gate asks the exact
    question — is the last session-day that closes at or before the window's
    end stored? — and a store one session behind is refused. Without it, or over
    an inner source that cannot say which days it holds, the gate falls back to
    the coarse time tolerance `DEFAULT_MAX_SHORTFALL` documents.

    `latest()` and `covered_days()` are memoized per (symbol, timeframe) because
    a sparse replay calls `candles()` once per fine window, and re-reading a
    Parquet footer or re-listing a directory per window would make the shortcut
    cost what it saves. A store topped up mid-run is therefore not noticed,
    which is correct: a run sees one view of its data.
    """

    __slots__ = ("_covered", "_days", "_inner", "_latest")

    def __init__(self, inner: CandleSource, days: SessionDays | None = None) -> None:
        self._inner = inner
        self._days = days
        self._latest: dict[tuple[str, Timeframe], int | None] = {}
        self._covered: dict[tuple[str, Timeframe, SessionDays], frozenset[str]] = {}

    def describe(self) -> str:
        return self._inner.describe()

    @property
    def inner(self) -> CandleSource:
        return self._inner

    @property
    def price_basis(self) -> str:
        return _require_capability(self._inner, "price_basis", self.describe())

    @property
    def pinned_as_of(self) -> str | None:
        return getattr(self._inner, "pinned_as_of", None)

    def candles(
        self, symbol: str, timeframe: Timeframe, start_unix: int, end_unix: int
    ) -> list[Candle]:
        # The calendar only helps a source that can say which days it holds. A
        # `latest`-only source (a `python` factory that was never asked for
        # more) keeps the time check rather than refusing on its first read.
        session_aware = callable(getattr(self._inner, "covered_days", None))
        require_coverage(
            self, symbol, timeframe, end_unix, days=self._days if session_aware else None
        )
        return self._inner.candles(symbol, timeframe, start_unix, end_unix)

    def has(self, symbol: str, timeframe: Timeframe) -> bool:
        probe = _require_capability(self._inner, "has", self.describe())
        return bool(probe(symbol, timeframe))

    def latest(self, symbol: str, timeframe: Timeframe) -> int | None:
        key = (symbol, timeframe)
        if key not in self._latest:
            probe = _require_capability(self._inner, "latest", self.describe())
            self._latest[key] = probe(symbol, timeframe)
        return self._latest[key]

    def covered_days(
        self, symbol: str, timeframe: Timeframe, days: SessionDays
    ) -> frozenset[str]:
        """Deliberately not coverage-checked: asking which days are held is the
        question `require_coverage` answers, so gating it would be circular."""
        key = (symbol, timeframe, days)
        if key not in self._covered:
            probe = _require_capability(self._inner, "covered_days", self.describe())
            self._covered[key] = frozenset(probe(symbol, timeframe, days))
        return self._covered[key]


def _require_capability(source: object, name: str, label: str) -> Any:
    """A capability the wrapper must not swallow — `build_candle_source` wraps by
    default, so returning None here would disarm a gate for every built source."""
    value = getattr(source, name, None)
    if value is None:
        raise BarSourceError(
            f"{label} cannot answer `{name}` — it is a bare CandleSource. The gates "
            "in bedivere.data.source need a source that can report what it holds."
        )
    return value


# ---------- the gates ----------


def filler_hint(source: CandleSource) -> str:
    """How to extend this store. Shared so an empty-window message and a
    shortfall message cannot recommend two different repairs for one store."""
    if source.describe().startswith("lake"):
        return "Ingest the range from the vendor (bedivere-data ingest)"
    return "Extend the file this source reads"


def require_timeframes(
    source: CandleSource, symbol: str, timeframes: Iterable[Timeframe]
) -> None:
    """Fail before a run starts if its source cannot serve every timeframe.

    Without this the failure surfaces an hour later as a backtest that took no
    trades — or took them at the wrong contract's prices. A source with no `has`
    is not gated: a CSV is the whole series it holds and says so on first read.
    """
    probe = getattr(source, "has", None)
    if not callable(probe):
        return
    wanted = list(dict.fromkeys(timeframes))
    missing = [tf for tf in wanted if not bool(probe(symbol, tf))]
    if missing:
        raise BarSourceError(
            f"{source.describe()} has no {symbol} bars for "
            f"{', '.join(tf.value for tf in missing)} "
            f"(needed: {', '.join(tf.value for tf in wanted)}).\n"
            "Ingest them, or name a different source — this will NOT fall back to another "
            "store, because two stores that roll on different dates are not the same price "
            "series and mixing them is not detectable downstream."
        )


def require_coverage(
    source: CandleSource,
    symbol: str,
    timeframe: Timeframe,
    end_unix: int,
    *,
    days: SessionDays | None = None,
    max_shortfall: int = DEFAULT_MAX_SHORTFALL,
) -> None:
    """Raise when the requested window reaches past what the source holds.

    With `days`, and a source that can say which days it holds, the question is
    exact: the last session-day that closes at or before `end_unix` must be
    stored. A window ending at a close — every backtest window — needs that very
    day. One ending inside a session, a live warm-up running to `now`, needs the
    close before it; the in-progress tail is a hole for `assess_coverage` to
    report, not a reach failure. Either way this only fires when the shortfall
    matters: a 2023 window read off a store that is a week behind is complete.

    Without a calendar the check degrades to a time tolerance, which cannot tell
    one session behind from bar-level slack — see `DEFAULT_MAX_SHORTFALL`.
    """
    probe = getattr(source, "latest", None)
    if not callable(probe):
        return
    latest = cast("int | None", probe(symbol, timeframe))
    if latest is None:
        return  # nothing stored at all: `require_timeframes` owns that, with a better message

    reach = _last_close_at_or_before(days, end_unix) if days is not None else None
    if reach is not None:
        covered_probe = getattr(source, "covered_days", None)
        if callable(covered_probe):
            if reach.label in cast("frozenset[str]", covered_probe(symbol, timeframe, days)):
                return
            _refuse_reach(source, symbol, timeframe, end_unix, latest, reach)

    if end_unix - latest <= max_shortfall:
        return
    _refuse_shortfall(source, symbol, timeframe, end_unix, latest)


def _last_close_at_or_before(days: SessionDays, end_unix: int) -> SessionDay | None:
    """The session-day a window REACHES: the last one whose close-stamp is at or
    before `end_unix`. Closes ascend because days are ordered and disjoint."""
    i = bisect_right([d.end_unix for d in days.days], end_unix) - 1
    return days.days[i] if i >= 0 else None


def _refuse_reach(
    source: CandleSource,
    symbol: str,
    timeframe: Timeframe,
    end_unix: int,
    latest: int,
    reach: SessionDay,
) -> NoReturn:
    # A pinned source truncates `covered_days` at its pin, so "top the store up"
    # would be wrong: the store may well hold the day; this view will not serve it.
    if getattr(source, "pinned_as_of", None):
        raise BarSourceError(
            f"{source.describe()} is pinned at its as-of date and serves nothing past "
            f"{_iso(latest)}, but the request runs to {_iso(end_unix)}, into session-day "
            f"{reach.label} (closes {_iso(reach.end_unix)}). "
            "Re-derive with a later asOf, or read the raw series."
        )
    raise BarSourceError(
        f"{source.describe()} holds no {symbol} {timeframe.value} bars for session-day "
        f"{reach.label} (closes {_iso(reach.end_unix)}), which the request runs to "
        f"({_iso(end_unix)}); its newest bar is {_iso(latest)}. The answer would be "
        "silently short.\n"
        f"{filler_hint(source)}, name a different source, or set "
        '`"allowStale": true` in the spec\'s data block to accept what is there.'
    )


def _refuse_shortfall(
    source: CandleSource, symbol: str, timeframe: Timeframe, end_unix: int, latest: int
) -> NoReturn:
    # A pinned source reports its pin as `latest`, so "top the store up" would be
    # wrong twice over: the store is current, and skipping this check would only
    # move the refusal to the pin's own horizon guard.
    if getattr(source, "pinned_as_of", None):
        raise BarSourceError(
            f"{source.describe()} is pinned at its as-of date and serves nothing past "
            f"{_iso(latest)}, but the request runs to {_iso(end_unix)} "
            f"({(end_unix - latest) / 86_400:.1f} days past the pin). "
            "Re-derive with a later asOf, or read the raw series."
        )
    raise BarSourceError(
        f"{source.describe()} holds {symbol} {timeframe.value} only through "
        f"{_iso(latest)}, but the request runs to {_iso(end_unix)} "
        f"({(end_unix - latest) / 86_400:.1f} days past). The answer would be silently short.\n"
        f"{filler_hint(source)}, name a different source, or set "
        '`"allowStale": true` in the spec\'s data block to accept what is there.'
    )


def _iso(unix_sec: int) -> str:
    return datetime.fromtimestamp(unix_sec, UTC).strftime("%Y-%m-%d %H:%M")
