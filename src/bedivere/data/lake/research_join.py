"""The frame contract every research join shares.

`align_available` and the decision-context functions accept the same frames
and refuse them for the same reasons, so those checks live here once. Nothing
here reads storage or interprets a feature column, and a declaration never
overrides what a frame already says about itself.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Hashable, Iterable
from typing import Any, Final, cast

import duckdb
import numpy as np
import pandas as pd  # pyright: ignore[reportMissingTypeStubs]

from bedivere.core.session_days import SessionDays
from bedivere.data.lake.provenance import Provenance, ProvenanceError
from bedivere.data.lake.volume import NS
from bedivere.data.lake.volume_store import COLUMNS

OBSERVATION_KEYS: Final[tuple[str, ...]] = ("asof_ns", "session", "instrument_id")

BAR_IDENTITY: Final[frozenset[str]] = frozenset(
    {"dataset", "symbol", "series", "definition", "session"}
)
"""The research loader's identity columns, alongside `volume_store.COLUMNS`."""

BAR_TIMESTAMPS: Final[tuple[str, ...]] = ("start_ns", "end_ns", "available_ns", "bar_id")
"""Must be BIGINT and non-negative; `bar_id` is a join tiebreaker."""

TRADE_TIMESTAMPS: Final[tuple[str, ...]] = ("ts_recv", "ordinal")

INTEGER_TYPES: Final[tuple[str, ...]] = tuple(
    sign + name for sign in ("", "U") for name in ("TINYINT", "SMALLINT", "INTEGER", "BIGINT")
)

NEVER_OVERRIDE: Final[str] = (
    "A declaration never overrides what a frame already says about itself."
)


def facts(p: Provenance) -> dict[str, str]:
    """The scalar facts two frames must share; session bounds compare per label."""
    return {
        "feed": p.feed,
        "dataset": p.dataset,
        "symbol": p.symbol,
        "series": p.series,
        "instrument_namespace": p.instrument_namespace,
        "price_basis": p.price_basis,
        "calendar": p.sessions.template,
        "timezone": p.sessions.timezone,
        "time_unit": p.time_unit,
        "epoch": p.epoch,
        "clock": p.clock,
    }


def _basis(p: Provenance, fact: str) -> str:
    field = "sessions" if fact in ("calendar", "timezone") else fact
    return "verified" if field in p.verified else "attested"


def conflicts(left: Provenance, right: Provenance, names: tuple[str, str]) -> list[str]:
    """Every fact the two records disagree on, with both values and their basis."""
    lines = [
        f"{fact}: {names[0]} {a!r} ({_basis(left, fact)}) vs {names[1]} {b!r} ({_basis(right, fact)})"
        for (fact, a), b in zip(facts(left).items(), facts(right).values(), strict=True)
        if a != b
    ]
    bounds = {day.label: (day.start_unix, day.end_unix) for day in right.sessions.days}
    lines += [
        f"session {day.label!r}: {names[0]} ({day.start_unix}, {day.end_unix}] vs "
        f"{names[1]} ({bounds[day.label][0]}, {bounds[day.label][1]}] (Unix seconds)"
        for day in left.sessions.days
        if day.label in bounds and (day.start_unix, day.end_unix) != bounds[day.label]
    ]
    return lines


def refuse(headline: str, lines: list[str], remedy: str) -> None:
    if lines:
        shown = lines[:8] + ([f"... and {len(lines) - 8} more"] if len(lines) > 8 else [])
        raise ProvenanceError(f"{headline}:\n  " + "\n  ".join(shown) + f"\n{remedy}")


def quoted(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def column_names(frame: pd.DataFrame) -> list[str]:
    names = list(cast(Iterable[object], frame.columns))
    if not all(isinstance(name, str) for name in names):
        raise ValueError("column names must be strings")
    if len(set(names)) != len(names):
        raise ValueError("duplicate column names are not supported")
    return cast(list[str], names)


def require_observation_columns(
    names: list[str], *, prefix: str, attached: Iterable[str], reserved: Iterable[str]
) -> None:
    """The join keys are present, and no attached or reserved name is taken."""
    if any(name in names for name in reserved) or not prefix:
        raise ValueError("reserved column name or empty prefix")
    if {f"{prefix}{name}" for name in attached} & set(names):
        raise ValueError("attached columns would overwrite observation columns")
    if not set(OBSERVATION_KEYS) <= set(names):
        raise ValueError("observations require asof_ns, session, instrument_id")


def observation_keys(
    observations: pd.DataFrame,
    names: list[str],
    declared: Provenance,
    position: str,
    *also: str,
) -> pd.DataFrame:
    """The only observation columns a join shows the database.

    Join keys, restated provenance facts, `also`, and each row's input position
    under `position`. Private columns never enter the database, so they cannot
    come back altered; results are matched to the caller's rows by position.
    """
    read = [*OBSERVATION_KEYS, *(fact for fact in facts(declared) if fact in names), *also]
    keys = cast(pd.DataFrame, observations[list(dict.fromkeys(read))])
    return keys.assign(**{position: np.arange(len(observations), dtype=np.int64)})  # pyright: ignore[reportUnknownMemberType]


def declared_provenance(value: object) -> Provenance:
    if not isinstance(value, Provenance):
        raise ProvenanceError(
            "observation_provenance must be a Provenance declaring what the observations are"
        )
    return value


def bar_provenance(bars: pd.DataFrame) -> tuple[Provenance, dict[str, object]]:
    """The record `load_volume_frame` attached, or a refusal."""
    loaded = bars.attrs.get("provenance")
    definition = bars.attrs.get("bar_definition")
    if not isinstance(loaded, Provenance) or not isinstance(definition, dict):
        raise ProvenanceError(
            "bars carry no loader provenance (attrs 'provenance' and 'bar_definition'); load "
            "them with load_volume_frame(..., feed=...). Hand-built frames, and concatenations "
            "of frames with different attrs, do not have it"
        )
    return loaded, cast(dict[str, object], definition)


def bar_window(bars: pd.DataFrame) -> tuple[int | None, int | None]:
    """The `[start_ns, end_ns)` availability window `load_volume_frame` applied.

    Required: nothing else tells a trimmed frame from a whole one.
    """
    window = cast(object, bars.attrs.get("availability_window"))
    if (
        not isinstance(window, tuple)
        or len(cast(tuple[object, ...], window)) != 2
        or not all(bound is None or type(bound) is int for bound in cast(tuple[object, ...], window))
    ):
        raise ValueError(
            "bars.attrs['availability_window'] must be the (start_ns, end_ns) pair "
            "load_volume_frame recorded; reload the bars"
        )
    return cast(tuple[int | None, int | None], window)


def trade_provenance(trades: pd.DataFrame) -> Provenance:
    """The record `load_trade_frame` attached under a calendar and `feed=`."""
    loaded = trades.attrs.get("provenance")
    if not isinstance(loaded, Provenance):
        raise ProvenanceError(
            "trades carry no loader provenance (attrs 'provenance'); load them with "
            "load_trade_frame(..., sessions=..., feed=...). Hand-built frames, calendar-free "
            "reads, and concatenations of frames with different attrs do not have it"
        )
    return loaded


def definition_key(definition: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(definition, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def require_attrs(attrs: dict[Hashable, Any], declared: Provenance, role: str) -> None:
    """Frame attributes may restate the declared provenance, never contradict it."""
    stated = attrs.get("provenance")
    if stated is not None:
        if not isinstance(stated, Provenance):
            raise ProvenanceError(
                f"{role}.attrs['provenance'] is a {type(stated).__name__}, not a Provenance; "
                "remove or convert it"
            )
        refuse(
            f"{role}.attrs['provenance'] contradicts the declared provenance",
            conflicts(stated, declared, (f"{role}.attrs", "declared")),
            NEVER_OVERRIDE,
        )
    declared_facts = facts(declared)
    restated = [(f"attrs[{name!r}]", name, attrs[name]) for name in declared_facts if name in attrs]
    definition = attrs.get("bar_definition")
    if isinstance(definition, dict):
        stored = cast(dict[str, object], definition)
        restated += [
            (f"attrs['bar_definition'][{name!r}]", name, stored[name])
            for name in declared_facts
            if name in stored
        ]
    refuse(
        f"{role} attributes contradict the declared provenance",
        [
            f"{where} is {found!r}, declared {name} {declared_facts[name]!r}"
            for where, name, found in restated
            if found != declared_facts[name]
        ],
        NEVER_OVERRIDE,
    )


def require_columns(
    con: duckdb.DuckDBPyConnection,
    relation: str,
    names: list[str],
    declared: Provenance,
    role: str,
) -> None:
    """Identity columns may restate the declared provenance, never contradict it."""
    for fact, value in facts(declared).items():
        if fact not in names:
            continue
        found = [
            row[0]
            for row in con.execute(
                f"SELECT DISTINCT CAST({quoted(fact)} AS VARCHAR) FROM {relation} "
                "ORDER BY 1 NULLS FIRST LIMIT 6"
            ).fetchall()
        ]
        if None in found:
            problem = "has NULL (unknown) values"
        elif len(found) > 1:
            problem = f"mixes {found}"
        elif found and found[0] != value:
            problem = f"holds {found[0]!r}"
        else:
            continue
        raise ProvenanceError(
            f"{role} column {fact!r} {problem}, but the declared {fact} is {value!r}; "
            "a declaration never overrides row data"
        )


def check_relation(
    con: duckdb.DuckDBPyConnection,
    relation: str,
    role: str,
    timestamps: Iterable[str],
    names: list[str],
    provenance: Provenance,
) -> None:
    """Pin the join-key dtypes of a registered relation, refuse NULL or
    non-positive keys, and check its identity columns against `provenance`."""
    timestamps = list(timestamps)
    types = dict((str(r[0]), str(r[1])) for r in con.execute(f"DESCRIBE {relation}").fetchall())
    for field in timestamps:
        if types[field] != "BIGINT":
            raise ValueError(
                f"{role}.{field} has DuckDB type {types[field]}; join timestamps must "
                "use integer dtypes holding int64 Unix nanoseconds, never floats or "
                "datetimes: convert upstream"
            )
    if types["instrument_id"] not in INTEGER_TYPES:
        raise ValueError(
            f"{role}.instrument_id has DuckDB type {types['instrument_id']}; "
            "identifiers must use integer dtypes"
        )
    if types["session"] != "VARCHAR" and not types["session"].startswith("ENUM("):
        # DuckDB types an empty or all-NULL object column INTEGER.
        labelled = con.execute(f"SELECT count(session) FROM {relation}").fetchone()
        if labelled and labelled[0]:
            raise ValueError(
                f"{role}.session has DuckDB type {types['session']}; session labels "
                "must be strings"
            )
    nulls = " OR ".join(f"{field} IS NULL OR {field} < 0" for field in timestamps)
    invalid = con.execute(
        f"SELECT count(*) FROM {relation} WHERE {nulls} OR session IS NULL "
        "OR instrument_id IS NULL OR instrument_id <= 0"
    ).fetchone()
    if invalid and invalid[0]:
        raise ValueError(
            f"{role} have null or negative join keys, or an instrument_id that is not "
            "positive (0 is the lake's unknown-contract sentinel)"
        )
    require_columns(con, relation, names, provenance, role)


def session_table(con: duckdb.DuckDBPyConnection, table: str, sessions: SessionDays) -> None:
    con.execute(f"CREATE TEMP TABLE {table} (label VARCHAR, start_ns BIGINT, end_ns BIGINT)")
    if not sessions.days:
        return
    # One vectorized insert; per-row parameters cost ~0.3 s for a long calendar.
    rows = pd.DataFrame(
        {
            "label": [day.label for day in sessions.days],
            "start_ns": pd.Series([day.start_unix * NS for day in sessions.days], dtype="int64"),
            "end_ns": pd.Series([day.end_unix * NS for day in sessions.days], dtype="int64"),
        }
    )
    con.register(f"{table}_rows", rows)
    con.execute(f"INSERT INTO {table} SELECT label, start_ns, end_ns FROM {table}_rows")


def check_single_definition(con: duckdb.DuckDBPyConnection, key: str) -> None:
    """`research_bars` holds one dataset and definition, and its `definition`
    column is the hash of the definition its attributes carry."""
    n_definitions = con.execute(
        "SELECT count(*) FROM (SELECT DISTINCT dataset, symbol, series, definition FROM research_bars)"
    ).fetchone()
    if n_definitions and n_definitions[0] > 1:
        raise ValueError("select exactly one bar dataset and definition before alignment")
    stored = con.execute("SELECT DISTINCT definition FROM research_bars").fetchall()
    if stored and stored[0][0] != key:
        raise ProvenanceError(
            f"bars' definition column {stored[0][0]!r} is not the sha256 of "
            f"attrs['bar_definition'] ({key}); reload the bars"
        )


def check_observation_rows(con: duckdb.DuckDBPyConnection, declared: Provenance) -> None:
    """Every observed session is declared, and each decision time lies inside
    its session's `[start, end]`."""
    session_table(con, "observation_sessions", declared.sessions)
    outside = con.execute(
        "SELECT o.session, o.asof_ns, s.start_ns, s.end_ns FROM observations o "
        "LEFT JOIN observation_sessions s ON CAST(o.session AS VARCHAR) = s.label "
        "WHERE s.label IS NULL OR o.asof_ns NOT BETWEEN s.start_ns AND s.end_ns "
        "ORDER BY s.label IS NOT NULL, o.session, o.asof_ns LIMIT 1"
    ).fetchone()
    if outside is not None:
        label, asof, start, end = outside
        if start is None:
            raise ProvenanceError(
                f"observation session {label!r} is not in the declared calendar "
                f"{declared.sessions.template!r}; declare every observed session's bounds"
            )
        raise ProvenanceError(
            f"observation asof_ns={asof} lies outside its declared session {label!r} "
            f"[{start}, {end}]; asof_ns must be int64 Unix nanoseconds on the declared "
            "clock, labelled by the declared calendar"
        )


def check_bar_coverage(
    con: duckdb.DuckDBPyConnection,
    window: tuple[int | None, int | None],
    *,
    reach: str,
    allow_exact_matches: bool,
) -> None:
    """Every decision's lookback lies inside the bars' availability window.

    `reach` is SQL for where a lookback starts, over an observation `o` and its
    session row `s`. Run after `check_observation_rows`, which supplies `s`.
    """
    start, end = window
    outside = [
        f"greatest({reach}, s.start_ns) < {start}" if start is not None else None,
        # The frame holds available_ns < end_ns, so only an exact match at end_ns is uncovered.
        f"o.asof_ns {'>=' if allow_exact_matches else '>'} {end}" if end is not None else None,
    ]
    if not any(outside):
        return
    reaching = con.execute(
        "SELECT count(*) FROM observations o "
        "JOIN observation_sessions s ON CAST(o.session AS VARCHAR) = s.label WHERE "
        + " OR ".join(clause for clause in outside if clause is not None)
    ).fetchone()
    if reaching and reaching[0]:
        raise ValueError(
            f"{reaching[0]} observation(s) look outside the availability window "
            f"[{start}, {end}) this bar frame was loaded with, so a bar they should see may "
            "have been trimmed away; reload the bars without start_ns/end_ns, or with a window "
            "that holds every decision and all it looks back over"
        )


def check_bar_rows(con: duckdb.DuckDBPyConnection, loaded: Provenance) -> None:
    """Every bar lies inside a loaded session, becomes available no earlier
    than it ends, and `(session, bar_id)` is unique."""
    session_table(con, "bar_sessions", loaded.sessions)
    misplaced = con.execute(
        "SELECT b.session, b.bar_id, [b.start_ns, b.end_ns, b.available_ns], "
        "[s.start_ns, s.end_ns] FROM research_bars b "
        "LEFT JOIN bar_sessions s ON CAST(b.session AS VARCHAR) = s.label "
        "WHERE s.label IS NULL OR NOT (s.start_ns <= b.start_ns AND b.start_ns <= b.end_ns "
        "AND b.end_ns <= b.available_ns AND b.available_ns <= s.end_ns) "
        "ORDER BY b.session, b.bar_id LIMIT 1"
    ).fetchone()
    if misplaced is not None:
        raise ProvenanceError(
            f"bar {misplaced[1]} of session {misplaced[0]!r} has start/end/available_ns "
            f"{misplaced[2]}, but its provenance session bounds are {misplaced[3]}: a bar "
            "must lie inside a loaded session and cannot become available before it ends. "
            "Reload the bars rather than editing them"
        )
    duplicates = con.execute(
        "SELECT count(*) FROM (SELECT session, bar_id FROM research_bars "
        "GROUP BY session, bar_id HAVING count(*) > 1)"
    ).fetchone()
    if duplicates and duplicates[0]:
        raise ValueError("duplicate bar identities")


def check_bar_sequence(con: duckdb.DuckDBPyConnection) -> None:
    """Within each session, `bar_id` runs without gaps and `available_ns` never
    decreases along it, which lets a history count back by `bar_id`."""
    gap = con.execute(
        "SELECT session, min(bar_id), max(bar_id), count(*) FROM research_bars "
        "GROUP BY session HAVING max(bar_id) - min(bar_id) + 1 <> count(*) ORDER BY session LIMIT 1"
    ).fetchone()
    if gap is not None:
        raise ValueError(
            f"bars of session {gap[0]!r} are not a contiguous run of bar_id ({gap[3]} bars span "
            f"{gap[1]}..{gap[2]}); a history counts back by bar_id, so load the bars with the "
            "research loader and filter only by its availability window"
        )
    backwards = con.execute(
        "SELECT session, bar_id FROM (SELECT session, bar_id, available_ns, "
        "lag(available_ns) OVER (PARTITION BY session ORDER BY bar_id) AS previous "
        "FROM research_bars) WHERE available_ns < previous ORDER BY session, bar_id LIMIT 1"
    ).fetchone()
    if backwards is not None:
        raise ProvenanceError(
            f"bar {backwards[1]} of session {backwards[0]!r} becomes available before the bar "
            "with the previous bar_id; stored partitions never do, so reload the bars"
        )


def check_trade_rows(con: duckdb.DuckDBPyConnection, loaded: Provenance) -> None:
    """Every trade lies inside its declared session's `[open, close)` on
    `ts_recv`, and ordinals are unique."""
    session_table(con, "trade_sessions", loaded.sessions)
    outside = con.execute(
        "SELECT t.session, t.ordinal, t.ts_recv, s.start_ns, s.end_ns FROM research_trades t "
        "LEFT JOIN trade_sessions s ON CAST(t.session AS VARCHAR) = s.label "
        "WHERE s.label IS NULL OR t.ts_recv < s.start_ns OR t.ts_recv >= s.end_ns "
        "ORDER BY s.label IS NOT NULL, t.session, t.ordinal LIMIT 1"
    ).fetchone()
    if outside is not None:
        label, ordinal, ts, start, end = outside
        if start is None:
            raise ProvenanceError(
                f"trade session {label!r} is not in the trades' declared calendar; reload the "
                "trades under the calendar that labelled them"
            )
        raise ProvenanceError(
            f"trade {ordinal} of session {label!r} has ts_recv={ts}, outside that session's "
            f"[{start}, {end}); trades are selected by [open, close), so reload them rather "
            "than editing them"
        )
    duplicates = con.execute(
        "SELECT count(*) FROM (SELECT ordinal FROM research_trades "
        "GROUP BY ordinal HAVING count(*) > 1)"
    ).fetchone()
    if duplicates and duplicates[0]:
        raise ValueError("duplicate trade ordinals; a frame holds each archive record once")


def require_bar_columns(names: list[str]) -> None:
    if not (set(COLUMNS) | BAR_IDENTITY) <= set(names):
        raise ValueError("bars must include the research loader's columns and identity")
