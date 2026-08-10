"""The spec envelope — one JSON file that fully describes a run.

    {
      "strategy": "examples.sma_cross:PLUGIN",
      "symbol": "DEMO",
      "config":  {"kind": "sma_cross", "fast": 5, "slow": 12, "rr": 1.5},
      "baseTimeframe": "5m",
      "derivedTimeframes": ["30m", "1h"],
      "session":  {"timezone": "America/New_York", "openTime": "18:00", ...},
      "instruments": {"DEMO": {"tickSize": 0.25, "pointValue": 20}},
      "window":  {"firstTradeDate": "2026-06-02", "lastTradeDate": "2026-06-19"},
      "sim":     {"latencyMs": 250, "halfSpreadTicks": 1, ...},
      "notify":  {"on": ["entry_fill"], "transport": "auto"},
      "data":    {"source": "sqlite", "path": "data/candles.db"}
    }

The design goal, and the reason `window` is optional: **a backtested spec IS
the run config.** Promoting a backtest to a live session is a `--mode` flag,
not an edit — the live runner derives its own window from *now* and the
session-day close, and everything else in the file means exactly what it
meant in the backtest.

Everything here is pydantic with `extra="forbid"`, so a misspelled envelope
key fails with the path that was wrong instead of falling back to a default
you did not choose. Costs (`sim`) have no defaults at all, matching
`run_backtest`: a frictionless run must be visible in the file.

Two blocks exist because bedivere ships no venue: `feed` names the adapter
that pushes live bars, `broker` names the adapter that places real orders.
A shadow session needs neither (`bedivere.streams.feed:build_replay_feed`
replays a candle source through the same LiveBarStream a real feed uses).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Self, cast
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator
from pydantic.alias_generators import to_camel

from bedivere.config.base import ConfigError, format_validation_error
from bedivere.core.pricing import InstrumentSpec, spec_from_handoff
from bedivere.core.session_days import SessionDay, SessionDays, parse_session_days
from bedivere.core.types import Timeframe
from bedivere.sessions import DailySessionSpec, build_session_days


class SpecError(ValueError):
    """A malformed spec envelope, or one that cannot be resolved."""


class SpecModel(BaseModel):
    """Every envelope block: frozen, camelCase on the wire, snake_case in
    Python, and no unknown keys anywhere."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        alias_generator=to_camel,
        populate_by_name=True,
    )


class InstrumentEntry(SpecModel):
    """One instrument's handoff numbers. Validated into an `InstrumentSpec`
    (which is where "tickSize x pointValue must be whole cents" is enforced)."""

    tick_size: float | str
    point_value: float | str


class SessionDayRow(SpecModel):
    label: str
    start_unix: int
    end_unix: int


class SessionSpec(SpecModel):
    """Session policy: either already-resolved rows, or the rules to expand
    into them.

    The rules form is what a spec normally carries — a calendar that a live
    run can re-expand around today, which resolved rows cannot do. Omitting
    `firstCloseDate`/`lastCloseDate` is legal there and means "derive them
    from now": `lookbackDays` back for warm-up history, one day forward so an
    overnight session that CLOSES tomorrow is present.
    """

    # Resolved form.
    days: tuple[SessionDayRow, ...] | None = None
    # Rules form.
    timezone: str | None = None
    open_time: str | None = None
    close_time: str | None = None
    close_weekdays: tuple[int, ...] = (0, 1, 2, 3, 4)
    holidays: tuple[str, ...] = ()
    first_close_date: str | None = None
    last_close_date: str | None = None
    lookback_days: int = 10
    # Both forms.
    template: str = "custom_daily"

    @model_validator(mode="after")
    def _one_form_only(self) -> Self:
        if self.days is not None:
            rules = [
                name
                for name in ("timezone", "open_time", "close_time")
                if getattr(self, name) is not None
            ]
            if rules:
                raise ValueError(
                    f"session carries resolved `days` AND rule field(s) {rules} — "
                    "pick one form; rules that disagree with the rows they sit beside "
                    "cannot both be the calendar"
                )
            if not self.days:
                raise ValueError("session.days is empty — a run needs at least one session-day")
            return self
        missing = [
            to_camel(name)
            for name in ("timezone", "open_time", "close_time")
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(
                f"session needs {missing} (the rules form) or `days` (already-resolved rows)"
            )
        if self.lookback_days < 0:
            raise ValueError("session.lookbackDays must be >= 0")
        return self


class WindowSpec(SpecModel):
    """The tradeable window, as unix stamps or as session-day labels.

    Labels are the form worth writing by hand: `firstTradeDate` is the day
    trading starts (its session OPEN is the tradeable start, so everything
    before it is warm-up history), `lastTradeDate` the day it ends (its
    CLOSE). Stamps are what the archived spec carries, so a re-run is exact.
    """

    start_unix: int | None = None
    end_unix: int | None = None
    first_trade_date: str | None = None
    last_trade_date: str | None = None

    @model_validator(mode="after")
    def _one_form_only(self) -> Self:
        stamps = self.start_unix is not None and self.end_unix is not None
        labels = self.first_trade_date is not None and self.last_trade_date is not None
        if stamps and labels:
            raise ValueError(
                "window carries both stamps and dates — pick one; two windows are not a window"
            )
        if not stamps and not labels:
            raise ValueError(
                "window needs startUnix+endUnix or firstTradeDate+lastTradeDate (both halves of one pair)"
            )
        if stamps and cast(int, self.start_unix) >= cast(int, self.end_unix):
            raise ValueError(
                f"window startUnix {self.start_unix} is not before endUnix {self.end_unix}"
            )
        return self


class SimSpec(SpecModel):
    """Sim venue costs and modelling choices. No defaults — same rule as
    `run_backtest`: a frictionless run has to be a zero you can see in the
    file, and a naked-window treatment has to be a decision you can see you
    made."""

    latency_ms: int
    half_spread_ticks: int
    commission_cents_per_side_per_contract: int
    seed: int
    # Arm protection from the bar AFTER the entry fill. At a 5m base the
    # naked window is sub-bar and this is false; at a 1s base it is a
    # quarter of the bar and false is a fiction. See SimBrokerConfig.
    defer_protection_one_bar: bool


class ReplaySpec(SpecModel):
    """How finely to replay — see docs/backtest-fidelity.md.

    `full` (the default) replays every base bar. `sparse` runs the strategy's
    trigger rule over the coarser series named here, and descends to the base
    timeframe only inside the windows that rule opens. It requires the
    strategy's plugin to publish a `trigger_rule`; without one the spec is
    asking for a shortcut whose safety nobody has declared, and that fails
    loudly rather than silently falling back to a full replay.

    `strict` runs both fidelity guards and fails the run on either. Turning it
    off keeps the run but leaves the escape visible in
    `replay.tradesCovered` — it is for inspecting a failure, not for living
    with one.
    """

    mode: Literal["full", "sparse"] = "full"
    coarse_timeframe: Timeframe | None = None
    window_seconds: int | None = None
    carry_across_sessions: bool = True
    strict: bool = True

    @model_validator(mode="after")
    def _sparse_needs_a_coarse_timeframe(self) -> Self:
        if self.mode == "sparse" and self.coarse_timeframe is None:
            raise ValueError(
                'replay.mode "sparse" needs a coarseTimeframe — the pre-pass has to '
                "know which series to run the trigger rule over"
            )
        return self


class NotifySpec(SpecModel):
    """`on` is the journal kinds that alert; None means the engine's default
    fill kinds. `transport` auto = Discord when BEDIVERE_DISCORD_WEBHOOK is
    set, console otherwise."""

    on: tuple[str, ...] | None = None
    transport: Literal["auto", "console", "discord", "off"] = "auto"


class DataSpec(SpecModel):
    """Where bars come from. `csv` and `sqlite` need a `path`; `python` names
    a factory returning a `CandleSource`, which is how a proprietary feed
    plugs in without a file format in between."""

    source: Literal["csv", "sqlite", "python"]
    path: str | None = None
    factory: str | None = None
    options: dict[str, Any] = {}

    @model_validator(mode="after")
    def _shape(self) -> Self:
        if self.source == "python":
            if self.factory is None:
                raise ValueError('data.source "python" needs `factory` ("module:attribute")')
            if self.path is not None:
                raise ValueError('data.source "python" takes `factory`, not `path`')
        else:
            if self.path is None:
                raise ValueError(f'data.source "{self.source}" needs `path`')
            if self.factory is not None:
                raise ValueError(f'data.source "{self.source}" takes `path`, not `factory`')
        return self


class FactorySpec(SpecModel):
    """An adapter bedivere does not ship: `factory` is an import path, and
    `options` are handed to it as keyword arguments after its context.

    Option KEYS are passed through verbatim — they are your factory's
    parameter names, not envelope fields, so they keep whatever casing your
    Python signature uses (`delay_s`, not `delayS`)."""

    factory: str
    options: dict[str, Any] = {}


class RunSpec(SpecModel):
    """The whole envelope. One document, one run, either environment."""

    strategy: str
    symbol: str
    config: dict[str, Any]
    base_timeframe: Timeframe
    derived_timeframes: tuple[Timeframe, ...] = ()
    session: SessionSpec
    instruments: dict[str, InstrumentEntry]
    sim: SimSpec
    window: WindowSpec | None = None
    replay: ReplaySpec = ReplaySpec()
    notify: NotifySpec = NotifySpec()
    data: DataSpec | None = None
    feed: FactorySpec | None = None
    broker: FactorySpec | None = None
    note: str = ""

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if self.symbol not in self.instruments:
            raise ValueError(
                f'instruments has no entry for symbol "{self.symbol}" '
                f"(has: {sorted(self.instruments)})"
            )
        if not isinstance(self.config.get("kind"), str):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError('config.kind must be a string naming the strategy config model')
        base = self.base_timeframe
        shallow = [tf.value for tf in self.derived_timeframes if tf.order <= base.order]
        if shallow:
            raise ValueError(
                f"derivedTimeframes {shallow} are not coarser than baseTimeframe "
                f'"{base.value}" — derived bars are aggregated UP from the base'
            )
        return self

    def jsonable(self) -> dict[str, Any]:
        """The envelope as plain camelCase JSON data, defaults materialized."""
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)


# ---------- loading ----------


def parse_spec(data: object) -> RunSpec:
    """Validate one spec document. Errors name the offending path."""
    if not isinstance(data, dict):
        raise SpecError("spec must be a JSON object")
    try:
        return RunSpec.model_validate(cast(dict[str, Any], data))
    except ValidationError as e:
        raise SpecError(f"spec: {format_validation_error(e)}") from e


def load_spec(path: str | Path) -> RunSpec:
    """Read and validate a spec file."""
    file = Path(path)
    try:
        raw: object = json.loads(file.read_text(encoding="utf-8"))
    except OSError as e:
        raise SpecError(f"{file}: cannot read spec ({e})") from e
    except json.JSONDecodeError as e:
        raise SpecError(f"{file}: not valid JSON — {e}") from e
    try:
        return parse_spec(raw)
    except SpecError as e:
        raise SpecError(f"{file}: {e}") from e


# ---------- resolution ----------


def resolve_instrument(spec: RunSpec) -> InstrumentSpec:
    """The run's instrument, built through `spec_from_handoff` so a tick
    value that is not whole cents fails here rather than mid-run."""
    entry = spec.instruments[spec.symbol]
    try:
        return spec_from_handoff(spec.symbol, entry.tick_size, entry.point_value)
    except ValueError as e:
        raise SpecError(f'instruments["{spec.symbol}"]: {e}') from e


def resolve_sessions(spec: RunSpec, *, now_unix: int | None = None) -> SessionDays:
    """Expand the session block into resolved days.

    `now_unix` is the run's notion of "now", handed over as data (this module
    never reads the wall). It is only consulted when the rules form omits its
    date range — the live case, where the calendar has to be re-expanded
    around today rather than around whenever the spec was written.
    """
    session = spec.session
    if session.days is not None:
        return parse_session_days(
            {
                "template": session.template,
                "timezone": session.timezone or "UTC",
                "days": [
                    {"label": d.label, "startUnix": d.start_unix, "endUnix": d.end_unix}
                    for d in session.days
                ],
            }
        )

    timezone = cast(str, session.timezone)
    first, last = session.first_close_date, session.last_close_date
    if first is None or last is None:
        if now_unix is None:
            raise SpecError(
                "session.firstCloseDate/lastCloseDate are required for a backtest — "
                "only a live run may derive its calendar from now"
            )
        today = datetime.fromtimestamp(now_unix, ZoneInfo(timezone)).date()
        # One day FORWARD as well: an overnight session opens the evening
        # before the date it closes on, so today's evening session carries
        # tomorrow's label and would otherwise be missing entirely.
        first = first or (today - timedelta(days=session.lookback_days)).isoformat()
        last = last or (today + timedelta(days=1)).isoformat()

    rules = DailySessionSpec(
        timezone=timezone,
        open_time=cast(str, session.open_time),
        close_time=cast(str, session.close_time),
        close_weekdays=tuple(session.close_weekdays),
        holidays=frozenset(session.holidays),
        template=session.template,
    )
    try:
        return build_session_days(rules, first, last)
    except ValueError as e:
        raise SpecError(f"session: {e}") from e


def resolve_backtest_window(spec: RunSpec, days: SessionDays) -> tuple[int, int]:
    """(tradeable_start, end) for a replay run. Required: a backtest with an
    implicit window silently changes meaning when the calendar grows."""
    window = spec.window
    if window is None:
        raise SpecError(
            "a backtest spec needs a `window` — "
            '{"firstTradeDate": "YYYY-MM-DD", "lastTradeDate": "YYYY-MM-DD"} '
            "or explicit startUnix/endUnix"
        )
    if window.start_unix is not None and window.end_unix is not None:
        return window.start_unix, window.end_unix
    first = _day_by_label(days, cast(str, window.first_trade_date), "firstTradeDate")
    last = _day_by_label(days, cast(str, window.last_trade_date), "lastTradeDate")
    if first.start_unix >= last.end_unix:
        raise SpecError(
            f"window.firstTradeDate {first.label} opens at or after "
            f"window.lastTradeDate {last.label} closes"
        )
    return first.start_unix, last.end_unix


def resolve_live_window(
    days: SessionDays, *, now_unix: int, until: str | None = None
) -> tuple[int, int]:
    """(now, end) for a live run. The end is `--until HH:MM` in the session
    timezone, or the close of the session-day containing now.

    The spec's own window is deliberately IGNORED here — it is the backtest's
    history, and a live run that inherited it would either be over before it
    started or be trading a window nobody chose. That asymmetry is what lets
    one file serve both environments.
    """
    day = days.day_containing(now_unix)
    if day is None:
        raise SpecError(
            f"no session-day contains {_utc(now_unix)} — the market is closed, or the "
            "spec's calendar does not reach today (check session.closeWeekdays/holidays)"
        )
    if until is None:
        return now_unix, day.end_unix
    hour, minute = _parse_hhmm(until)
    label = datetime.fromisoformat(day.label).date()
    end = int(
        datetime(label.year, label.month, label.day, hour, minute, tzinfo=ZoneInfo(days.timezone)).timestamp()
    )
    if end <= now_unix:
        raise SpecError(
            f"--until {until} ({_utc(end)}) is not in the future — it is already past"
        )
    if end > day.end_unix:
        raise SpecError(
            f"--until {until} ({_utc(end)}) is after the session-day close "
            f"({_utc(day.end_unix)}) — the venue closes first"
        )
    return now_unix, end


def archived_spec(
    spec: RunSpec, *, resolved_config: dict[str, Any], window: tuple[int, int]
) -> dict[str, Any]:
    """The spec.json written into a run directory: the same envelope with the
    resolved config baked in and the window materialized to stamps.

    It re-runs identically with NO `--set` flags and no dependence on what
    "today" meant. A spec plus a list of overrides you have to remember to
    re-apply is not a reproducible record.
    """
    out = spec.jsonable()
    out["config"] = resolved_config
    out["window"] = {"startUnix": window[0], "endUnix": window[1]}
    return out


# ---------- helpers ----------


def _day_by_label(days: SessionDays, label: str, what: str) -> SessionDay:
    for day in days.days:
        if day.label == label:
            return day
    available = [d.label for d in days.days]
    hint = f"{available[0]}..{available[-1]}" if available else "none"
    raise SpecError(f'window.{what} "{label}" is not a session-day in this calendar ({hint})')


def _parse_hhmm(text: str) -> tuple[int, int]:
    parts = text.split(":")
    if len(parts) != 2:
        raise SpecError(f'--until must be "HH:MM" 24-hour, got {text!r}')
    try:
        hour, minute = int(parts[0]), int(parts[1])
    except ValueError as e:
        raise SpecError(f'--until must be "HH:MM" 24-hour, got {text!r}') from e
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise SpecError(f"--until out of range: {text!r}")
    return hour, minute


def _utc(unix_sec: int) -> str:
    """Render a GIVEN instant (not a wall-clock read)."""
    return datetime.fromtimestamp(unix_sec, UTC).strftime("%Y-%m-%d %H:%MZ")


__all__ = [
    "ConfigError",
    "DataSpec",
    "FactorySpec",
    "InstrumentEntry",
    "NotifySpec",
    "RunSpec",
    "SessionDayRow",
    "SessionSpec",
    "SimSpec",
    "SpecError",
    "SpecModel",
    "WindowSpec",
    "archived_spec",
    "load_spec",
    "parse_spec",
    "resolve_backtest_window",
    "resolve_instrument",
    "resolve_live_window",
    "resolve_sessions",
]
