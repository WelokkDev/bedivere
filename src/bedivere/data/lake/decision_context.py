"""Decision-time context: bar history, trade windows, and the bar in progress.

All three share the research-join contract of `align_available`: compatible
provenance, no crossing of a session or instrument, bars in scope by
`available_ns` and trades by `ts_recv`, and exact timestamp matches opt-in.
The forming bar is not estimated: the stored definition is replayed over the
trades the bars were built from, and checked against the stored bars.

History and window results are long frames keyed by `observation`, the 0-based
input position. An observation with nothing in scope contributes no rows (or
NULL forming columns); a session a frame was not loaded for is refused.
"""

from __future__ import annotations

import json
from bisect import bisect_left
from collections.abc import Iterable
from typing import Any, Final, cast

import duckdb
import numpy as np
import pandas as pd  # pyright: ignore[reportMissingTypeStubs]

from bedivere.core.session_days import SessionDays
from bedivere.data.lake.ingest import PRICE_SCALE
from bedivere.data.lake.provenance import Provenance
from bedivere.data.lake.research_join import (
    BAR_TIMESTAMPS,
    TRADE_TIMESTAMPS,
    bar_provenance,
    bar_window,
    check_bar_coverage,
    check_bar_rows,
    check_bar_sequence,
    check_observation_rows,
    check_relation,
    check_single_definition,
    check_trade_rows,
    column_names,
    conflicts,
    declared_provenance,
    definition_key,
    observation_keys,
    quoted,
    refuse,
    require_attrs,
    require_bar_columns,
    require_observation_columns,
    trade_provenance,
)
from bedivere.data.lake.trades import TradeIssue, TradeSource
from bedivere.data.lake.trades_frame import TRADE_COLUMNS
from bedivere.data.lake.volume import (
    NS,
    FormingBar,
    VolumeAccumulator,
    VolumeBar,
    VolumeInput,
    VolumeSpec,
)
from bedivere.data.lake.volume_research import BarSource
from bedivere.data.lake.volume_store import COLUMNS

# pandas is optional and shipped without complete typing for dynamic columns.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false

POSITION: Final[str] = "observation"
"""Result column: the 0-based position of the observation row in the input."""

LAG: Final[str] = "lag"
"""History result column: 0 for the most recent completed bar in scope."""

FORMING_COLUMNS: Final[tuple[str, ...]] = (
    "bar_id",
    "start_ns",
    "end_ns",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "input_count",
    "vwap",
)
"""Attached by `forming_bars`: integers as nullable `Int64`, prices as `float64`."""

_ORDER: Final[str] = "__observation"

_BARS_REMEDY: Final[str] = (
    "Nothing is reconciled or mapped automatically: join bars built from the observations' "
    "own source, or convert the observations upstream and declare what they then are."
)
_TRADES_REMEDY: Final[str] = (
    "Nothing is reconciled or mapped automatically: read the trades the observations were "
    "made from, under the calendar that labelled them, and declare what they are."
)


def _age(max_age_ns: object) -> int:
    if type(max_age_ns) is not int or not 0 <= max_age_ns < 2**63:
        raise ValueError("max_age_ns must be an explicit non-negative int64 duration")
    return max_age_ns


def _require_trade_columns(names: list[str]) -> None:
    if not set(TRADE_COLUMNS) <= set(names):
        raise ValueError("trades must include every column of the trade loader (TRADE_COLUMNS)")


def _attached(alias: str, names: Iterable[str], prefix: str) -> str:
    return ", ".join(f"{alias}.{quoted(name)} AS {quoted(prefix + name)}" for name in names)


def _require_trade_sessions(observed: Iterable[str], trades: Provenance) -> None:
    """Every observed session is one the trade frame was loaded for: an empty
    result could not tell an unloaded session from one without trades."""
    loaded = [day.label for day in trades.sessions.days]
    absent = sorted(set(observed) - set(loaded))
    if absent:
        held = f"{loaded[0]} .. {loaded[-1]}" if len(loaded) > 1 else ", ".join(loaded)
        raise ValueError(
            f"observations in session(s) {absent} cannot be answered from this trade frame: it "
            f"was loaded for {len(loaded)} session(s) ({held}) and holds no trades of any "
            "other, which an empty result would not tell from a session without trades. "
            "Reload the trades with a calendar that includes every observed session"
        )


def _require_same_archive(
    sessions: Iterable[str], bars: pd.DataFrame, trades: pd.DataFrame
) -> None:
    """The trades come from the archive, and contract selection, the bars
    record being built from, session by session.

    The bar still forming has no stored bar to check a replay against, so only
    the recorded digests can tell another archive of the same contract apart.
    """
    held, read = bars.attrs.get("bar_source"), trades.attrs.get("trade_source")
    if not isinstance(held, BarSource) or not isinstance(read, TradeSource):
        raise ValueError(
            "forming bars need what each frame was loaded from: attrs['bar_source'] from "
            "load_volume_frame and attrs['trade_source'] from load_trade_frame. Reload them"
        )
    built_from = dict(held.inputs)
    selection = dict(read.selection_dependencies)
    differing: list[str] = []
    for label in sorted(set(sessions)):
        # Identity is the archive's bytes, not its name, plus any 1s selection.
        wanted = f":sha256:{read.archive_sha256}" + (
            f":selection:{selection[label]}" if label in selection else ""
        )
        if not built_from[label].endswith(wanted):
            differing.append(f"{label}: bars built from {built_from[label]}")
    if differing:
        shown = differing[:8] + ([f"... and {len(differing) - 8} more"] if len(differing) > 8 else [])
        raise ValueError(
            f"the trades were read from {read.archive_name} (sha256 {read.archive_sha256}), "
            "which is not what these bars were built from:\n  " + "\n  ".join(shown) + "\n"
            "A forming bar is replayed only from the archive, and contract selection, its bars "
            "came from: load the trades from that archive, or rebuild the bars from this one."
        )


def _receive_window(trades: pd.DataFrame) -> tuple[int | None, int | None]:
    """The `[start_ns, end_ns)` receive-time bounds `load_trade_frame` applied.
    A frame without the attribute counts as whole sessions."""
    window = cast(object, trades.attrs.get("receive_window", (None, None)))
    if (
        not isinstance(window, tuple)
        or len(cast(tuple[object, ...], window)) != 2
        or not all(bound is None or type(bound) is int for bound in cast(tuple[object, ...], window))
    ):
        raise ValueError("trades.attrs['receive_window'] must be a (start_ns, end_ns) pair of int or None")
    return cast(tuple[int | None, int | None], window)


def bar_history(
    observations: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    observation_provenance: Provenance,
    depth: int,
    max_age_ns: int,
    allow_exact_matches: bool = False,
    prefix: str = "volume_",
) -> pd.DataFrame:
    """The `depth` most recent completed bars in scope at each decision time.

    In scope: same session and instrument, available before `asof_ns` (or at
    it, with `allow_exact_matches`) and at most `max_age_ns` earlier. Lag 0 is
    the bar `align_available` would attach; lag k is k `bar_id`s before it. One
    row per (observation, lag); an observation with no bar in scope has none.
    Bars must be an unbroken run of `bar_id` per session, as loaded frames are.
    """
    if type(depth) is not int or not 0 < depth < 2**63:
        raise ValueError("depth must be a positive int64 number of bars")
    age = _age(max_age_ns)
    obs_names = column_names(observations)
    bar_names = column_names(bars)
    require_observation_columns(
        obs_names, prefix=prefix, attached=COLUMNS, reserved=(POSITION, LAG, _ORDER)
    )
    require_bar_columns(bar_names)
    declared = declared_provenance(observation_provenance)
    loaded, definition = bar_provenance(bars)
    refuse(
        "observations and volume bars have incompatible provenance",
        conflicts(declared, loaded, ("observations", "bars")),
        _BARS_REMEDY,
    )
    require_attrs(observations.attrs, declared, "observations")
    require_attrs(bars.attrs, loaded, "bars")
    window = bar_window(bars)
    key = definition_key(definition)
    cmp = ">=" if allow_exact_matches else ">"
    with duckdb.connect(":memory:") as con:
        con.register("observations", observation_keys(observations, obs_names, declared, _ORDER))
        con.register("research_bars", bars)
        check_relation(con, "observations", "observations", ["asof_ns"], obs_names, declared)
        check_relation(con, "research_bars", "bars", BAR_TIMESTAMPS, bar_names, loaded)
        check_single_definition(con, key)
        check_observation_rows(con, declared)
        check_bar_rows(con, loaded)
        check_bar_sequence(con)
        check_bar_coverage(
            con, window, reach=f"o.asof_ns - {age}", allow_exact_matches=allow_exact_matches
        )
        result = con.execute(
            f"WITH o AS (SELECT {_ORDER}, asof_ns, session, instrument_id FROM observations), "
            "b AS (SELECT session, instrument_id, available_ns, bar_id FROM research_bars "
            "QUALIFY row_number() OVER "
            "(PARTITION BY session, instrument_id, available_ns ORDER BY bar_id DESC) = 1), "
            "latest AS (SELECT o.*, b.bar_id AS __latest FROM o ASOF LEFT JOIN b "
            f"ON o.session = b.session AND o.instrument_id = b.instrument_id "
            f"AND o.asof_ns {cmp} b.available_ns) "
            f"SELECT l.{_ORDER} AS {POSITION}, l.asof_ns, l.session, l.instrument_id, "
            f"l.__latest - h.bar_id AS {LAG}, {_attached('h', COLUMNS, prefix)} "
            "FROM latest l JOIN research_bars h "
            "ON h.session = l.session AND h.instrument_id = l.instrument_id "
            f"AND h.bar_id BETWEEN l.__latest - {depth - 1} AND l.__latest "
            f"AND l.asof_ns - h.available_ns <= {age} "
            f"ORDER BY {POSITION}, {LAG}"
        ).df()
    result.attrs.update({k: v for k, v in bars.attrs.items() if k != "provenance"})
    result.attrs.update({"provenance": declared, "bar_provenance": loaded})
    return result


def trade_windows(
    observations: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    observation_provenance: Provenance,
    lookback_ns: int | None = None,
    start_column: str | None = None,
    allow_exact_matches: bool = False,
    prefix: str = "trade_",
) -> pd.DataFrame:
    """The trades of a causal window ending at each decision time.

    The window is `[start, asof_ns)` on `ts_recv` (closed with
    `allow_exact_matches`), within the observation's session and instrument.
    Its start is `asof_ns - lookback_ns` or the int64 observation column
    `start_column`: give exactly one. One row per (observation, trade), in
    archive order. `trades` must come from `load_trade_frame` with a calendar
    and `feed=`, and must hold every observed session and window in full.
    """
    if (lookback_ns is None) == (start_column is None):
        raise ValueError(
            "give exactly one of lookback_ns (a fixed duration before each decision) or "
            "start_column (an int64 observation column holding each window's start)"
        )
    if lookback_ns is not None and (type(lookback_ns) is not int or not 0 <= lookback_ns < 2**63):
        raise ValueError("lookback_ns must be a non-negative int64 duration in nanoseconds")
    obs_names = column_names(observations)
    trade_names = column_names(trades)
    require_observation_columns(
        obs_names, prefix=prefix, attached=TRADE_COLUMNS, reserved=(POSITION, _ORDER)
    )
    if start_column is not None and start_column not in obs_names:
        raise ValueError(f"observations have no column {start_column!r} to start windows from")
    _require_trade_columns(trade_names)
    declared = declared_provenance(observation_provenance)
    loaded = trade_provenance(trades)
    refuse(
        "observations and trades have incompatible provenance",
        conflicts(declared, loaded, ("observations", "trades")),
        _TRADES_REMEDY,
    )
    require_attrs(observations.attrs, declared, "observations")
    require_attrs(trades.attrs, loaded, "trades")
    bounded_from, bounded_to = _receive_window(trades)
    cmp = "<=" if allow_exact_matches else "<"
    with duckdb.connect(":memory:") as con:
        con.register(
            "observations",
            observation_keys(
                observations, obs_names, declared, _ORDER, *([start_column] if start_column else [])
            ),
        )
        con.register("research_trades", trades)
        check_relation(con, "observations", "observations", ["asof_ns"], obs_names, declared)
        check_relation(con, "research_trades", "trades", TRADE_TIMESTAMPS, trade_names, loaded)
        check_observation_rows(con, declared)
        check_trade_rows(con, loaded)
        _require_trade_sessions(
            (
                str(row[0])
                for row in con.execute(
                    "SELECT DISTINCT CAST(session AS VARCHAR) FROM observations"
                ).fetchall()
            ),
            loaded,
        )
        if start_column is None:
            start = f"asof_ns - {lookback_ns}"
        else:
            start = quoted(start_column)
            kind = next(
                str(row[1])
                for row in con.execute("DESCRIBE observations").fetchall()
                if str(row[0]) == start_column
            )
            if kind != "BIGINT":
                raise ValueError(
                    f"observations.{start_column} has DuckDB type {kind}; window starts must "
                    "use an integer dtype holding int64 Unix nanoseconds, never floats or "
                    "datetimes: convert upstream"
                )
            late = con.execute(
                f"SELECT count(*) FROM observations WHERE {start} > asof_ns OR {start} < 0"
            ).fetchone()
            if late and late[0]:
                raise ValueError(
                    f"observations.{start_column} holds a window start after its decision "
                    "time, or a negative one; a causal window ends at asof_ns"
                )
        outside = [
            f"{start} < {bounded_from}" if bounded_from is not None else None,
            # The frame holds ts_recv < end_ns, so only an exact match at end_ns is uncovered.
            (f"asof_ns {'>=' if allow_exact_matches else '>'} {bounded_to}")
            if bounded_to is not None
            else None,
        ]
        if any(outside):
            reaching = con.execute(
                "SELECT count(*) FROM observations WHERE "
                + " OR ".join(clause for clause in outside if clause is not None)
            ).fetchone()
            if reaching and reaching[0]:
                raise ValueError(
                    f"{reaching[0]} observation window(s) reach outside the trade frame's "
                    f"receive-time bounds [{bounded_from}, {bounded_to}), so their trades are "
                    "not all present; reload the trades over whole sessions or narrow the windows"
                )
        result = con.execute(
            f"WITH o AS (SELECT {_ORDER}, asof_ns, session, instrument_id, {start} AS __start "
            "FROM observations) "
            f"SELECT o.{_ORDER} AS {POSITION}, o.asof_ns, o.session, o.instrument_id, "
            f"{_attached('t', TRADE_COLUMNS, prefix)} FROM o JOIN research_trades t "
            "ON t.session = o.session AND t.instrument_id = o.instrument_id "
            f"AND t.ts_recv >= o.__start AND t.ts_recv {cmp} o.asof_ns "
            f"ORDER BY {POSITION}, t.ordinal"
        ).df()
    # The loader's dtypes, not DuckDB's data-dependent ones: nullable stays nullable.
    result = result.astype({f"{prefix}{name}": dtype for name, dtype in TRADE_COLUMNS.items()})
    result.attrs.update({k: v for k, v in trades.attrs.items() if k != "provenance"})
    result.attrs.update({"provenance": declared, "trade_provenance": loaded})
    return result


def _replayable(definition: dict[str, object], sessions: SessionDays) -> VolumeSpec:
    """The `VolumeSpec` that constructs `definition`, or a refusal."""
    source = definition.get("source")
    if source != "trades":
        raise ValueError(
            f"forming bars are replayed from trades, but this definition's source is "
            f"{source!r}; its bar in progress would need its one-second bars, not trades"
        )
    try:
        spec = VolumeSpec(
            cast(int, definition.get("threshold")), "trades", cast(Any, definition.get("boundary"))
        )
    except ValueError as exc:
        raise ValueError(f"stored definition cannot be replayed: {exc}") from exc
    if json.loads(spec.definition(sessions)) != definition:
        raise ValueError(
            "the stored definition was constructed by a builder version or policy this one "
            "does not reproduce; forming bars need a definition this builder constructs "
            "identically"
        )
    return spec


def _dense(column: Any, name: str, dtype: type[Any]) -> np.ndarray[Any, Any]:
    """A NumPy column with no masked (NULL) entry."""
    if np.ma.is_masked(column):
        raise ValueError(f"trades.{name} holds NULL values; reload the trades")
    return np.asarray(np.ma.getdata(column), dtype=dtype)


def forming_bars(
    observations: pd.DataFrame,
    bars: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    observation_provenance: Provenance,
    allow_exact_matches: bool = False,
    prefix: str = "forming_",
) -> pd.DataFrame:
    """The bar in progress at each decision time, under the bars' definition.

    The stored definition's own policy (`VolumeAccumulator`) is replayed over
    each session's trades before `asof_ns` (or at it, with
    `allow_exact_matches`); its unclosed state is attached as `FormingBar`
    fields under `prefix`, NULL when nothing is forming.

    The trades must be whole sessions from the archive and contract selection
    the bars record being built from. Every bar the replay completes must
    equal the stored one, and every stored bar available by a decision must
    already be completed; otherwise the call fails.
    """
    obs_names = column_names(observations)
    bar_names = column_names(bars)
    trade_names = column_names(trades)
    require_observation_columns(
        obs_names, prefix=prefix, attached=FORMING_COLUMNS, reserved=(_ORDER,)
    )
    require_bar_columns(bar_names)
    _require_trade_columns(trade_names)
    declared = declared_provenance(observation_provenance)
    loaded, definition = bar_provenance(bars)
    source = trade_provenance(trades)
    refuse(
        "observations and volume bars have incompatible provenance",
        conflicts(declared, loaded, ("observations", "bars")),
        _BARS_REMEDY,
    )
    refuse(
        "observations and trades have incompatible provenance",
        conflicts(declared, source, ("observations", "trades")),
        _TRADES_REMEDY,
    )
    require_attrs(observations.attrs, declared, "observations")
    require_attrs(bars.attrs, loaded, "bars")
    require_attrs(trades.attrs, source, "trades")
    window = _receive_window(trades)
    if window != (None, None):
        raise ValueError(
            f"forming bars need every trade of each observed session from its open, but this "
            f"trade frame was loaded with receive-time bounds {window}; a missing tail cannot "
            "be detected, so reload it with sessions= alone"
        )
    held = bar_window(bars)
    key = definition_key(definition)
    with duckdb.connect(":memory:") as con:
        con.register("observations", observation_keys(observations, obs_names, declared, _ORDER))
        con.register("research_bars", bars)
        con.register("research_trades", trades)
        check_relation(con, "observations", "observations", ["asof_ns"], obs_names, declared)
        check_relation(con, "research_bars", "bars", BAR_TIMESTAMPS, bar_names, loaded)
        check_relation(con, "research_trades", "trades", TRADE_TIMESTAMPS, trade_names, source)
        check_single_definition(con, key)
        check_observation_rows(con, declared)
        check_bar_rows(con, loaded)
        check_trade_rows(con, source)
        spec = _replayable(definition, loaded.sessions)
        # The replay runs from each session's open, so the frame must hold bars from there.
        check_bar_coverage(con, held, reach="s.start_ns", allow_exact_matches=allow_exact_matches)
        decisions = con.execute(
            f"SELECT CAST(session AS VARCHAR), instrument_id::BIGINT, asof_ns, {_ORDER} "
            f"FROM observations ORDER BY session, instrument_id, asof_ns, {_ORDER}"
        ).fetchall()
        stored = con.execute(
            f"SELECT CAST(session AS VARCHAR), {', '.join(COLUMNS)} FROM research_bars"
        ).fetchall()
        # Group sizes in the feed's order: each (session, instrument) is a slice below.
        groups = con.execute(
            "SELECT CAST(session AS VARCHAR), instrument_id::BIGINT, count(*) FROM research_trades "
            "GROUP BY session, instrument_id ORDER BY session, instrument_id"
        ).fetchall()
        feed = cast(
            dict[str, Any],
            con.execute(
                "SELECT ordinal, ts_recv, price_fixed, size::BIGINT AS size, eligible, "
                "issues::BIGINT AS issues FROM research_trades "
                "ORDER BY session, instrument_id, ordinal"
            ).fetchnumpy(),
        )
    covered = {day.label for day in loaded.sessions.days}
    uncovered = sorted({str(row[0]) for row in decisions} - covered)
    if uncovered:
        raise ValueError(
            f"forming bars for session(s) {uncovered} cannot be checked: the bars frame holds "
            f"no partition for them (missing_sessions={bars.attrs.get('missing_sessions')}); "
            "build or load those sessions first"
        )
    _require_trade_sessions((str(row[0]) for row in decisions), source)
    _require_same_archive((str(row[0]) for row in decisions), bars, trades)
    expected: dict[tuple[str, int, int], VolumeBar] = {}
    # Stored full bars per (session, instrument), in bar_id (and availability) order.
    published: dict[tuple[str, int], list[tuple[int, int]]] = {}
    for row in stored:
        bar = VolumeBar(*row[1:])
        expected[(str(row[0]), bar.instrument_id, bar.bar_id)] = bar
        if not bar.is_partial:
            published.setdefault((str(row[0]), bar.instrument_id), []).append(
                (bar.available_ns, bar.bar_id)
            )
    for sequence in published.values():
        sequence.sort(key=lambda pair: pair[1])
    eligible_all = _dense(feed["eligible"], "eligible", np.bool_)
    ts_all = _dense(feed["ts_recv"], "ts_recv", np.int64)
    size_all = _dense(feed["size"], "size", np.int64)
    ordinal_all = _dense(feed["ordinal"], "ordinal", np.int64)
    issues_all = _dense(feed["issues"], "issues", np.int64)
    if bool((np.ma.getmaskarray(feed["price_fixed"]) & eligible_all).any()):
        raise ValueError("trades marked eligible have no price; reload the trades")
    price_all = np.asarray(np.ma.getdata(feed["price_fixed"]), dtype=np.int64)
    slices: dict[tuple[str, int], tuple[int, int]] = {}
    offset = 0
    for label, instrument, count in groups:
        slices[(str(label), int(instrument))] = (offset, offset + int(count))
        offset += int(count)
    days = {day.label: day for day in declared.sessions.days}
    forming: list[FormingBar | None] = [None] * len(observations)
    index = 0
    while index < len(decisions):
        label, instrument = str(decisions[index][0]), int(decisions[index][1])
        day = days[label]
        close_ns = day.end_unix * NS
        low, high = slices.get((label, instrument), (0, 0))
        ts = cast(list[int], ts_all[low:high].tolist())
        prices = cast(list[int], price_all[low:high].tolist())
        sizes = cast(list[int], size_all[low:high].tolist())
        eligible = cast(list[bool], eligible_all[low:high].tolist())
        ordinals = cast(list[int], ordinal_all[low:high].tolist())
        issues = cast(list[int], issues_all[low:high].tolist())
        availability = [pair[0] for pair in published.get((label, instrument), [])]
        bar_ids = [pair[1] for pair in published.get((label, instrument), [])]
        accumulator = VolumeAccumulator(spec, day)
        cursor = 0
        completed = 0
        while index < len(decisions) and (str(decisions[index][0]), int(decisions[index][1])) == (
            label,
            instrument,
        ):
            asof, position = int(decisions[index][2]), int(decisions[index][3])
            limit = asof + 1 if allow_exact_matches else asof
            while cursor < len(ts) and ts[cursor] < limit:
                if not eligible[cursor]:
                    raise ValueError(
                        f"session {label!r}: trade {ordinals[cursor]} is ineligible for "
                        f"receive-time bars ({TradeIssue(issues[cursor])!r}); the builder refuses "
                        "such a record, so these bars were not built from these trades"
                    )
                try:
                    emitted = accumulator.push(
                        VolumeInput.trade(
                            ts[cursor], prices[cursor] / PRICE_SCALE, sizes[cursor], instrument
                        )
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"session {label!r}: the trades cannot be replayed under the bars' "
                        f"definition: {exc}"
                    ) from exc
                for bar in emitted:
                    known = expected.get((label, instrument, bar.bar_id))
                    if known is None:
                        # Its window holds this stretch, so the frame would hold the bar.
                        raise ValueError(
                            f"session {label!r}: replaying the trades completes bar "
                            f"{bar.bar_id}, which the stored bars do not hold; these trades "
                            f"are not the records the bars were built from (replayed {bar})"
                        )
                    if known != bar:
                        raise ValueError(
                            f"session {label!r}: replaying the trades does not reproduce "
                            f"stored bar {bar.bar_id}; these trades are not the records the "
                            f"bars were built from (replayed {bar}, stored {known})"
                        )
                completed += len(emitted)
                cursor += 1
            # A stored bar available by this decision must already have been replayed.
            due = bisect_left(availability, limit)
            if due and bar_ids[due - 1] >= completed:
                raise ValueError(
                    f"session {label!r}: stored bar {bar_ids[due - 1]} became available at "
                    f"{availability[due - 1]}, before the decision at {asof}, but replaying the "
                    "supplied trades has not completed it; the trades do not cover this session "
                    "from its open, or are not the records the bars were built from"
                )
            forming[position] = (
                None if allow_exact_matches and asof >= close_ns else accumulator.forming()
            )
            index += 1
    # The observations themselves, index included; columns attach by position.
    result = observations.copy()
    for name in FORMING_COLUMNS:
        values = [None if bar is None else getattr(bar, name) for bar in forming]
        if name in ("open", "high", "low", "close", "vwap"):
            result[prefix + name] = np.array(
                [np.nan if value is None else value for value in values], dtype=np.float64
            )
        else:
            result[prefix + name] = pd.array(values, dtype="Int64")
    result.attrs = {
        **{k: v for k, v in bars.attrs.items() if k != "provenance"},
        "provenance": declared,
        "bar_provenance": loaded,
        "trade_provenance": source,
        "trade_source": trades.attrs.get("trade_source"),
    }
    return result
