"""Back-adjustment as a READ-TIME derivation over the raw lake — never a store.

Raw as-traded prices are the facts; an adjusted continuous series is a view
whose every bar changes value at the next roll, and persisting one is how two
backtests six months apart come to disagree about "the same data". So the view
is derived on demand, pinned to an explicit as-of date: same as-of, same roll
set, same series, forever.

Per Schwager's split, an adjusted series reflects price SWINGS as an account
would have experienced them across rolls — use it for system testing and equity
curves — but not the price LEVELS anything traded at, so a zone, a stop level or
any absolute-price rule wants the raw series underneath.

Gap method: prior session close -> next session open. The textbook alternative
measures both contracts at the same instant, but a front-month series stores
only one leg — the other contract simply is not in the lake. The cost, stated
plainly: one overnight move per roll is folded into the offset alongside the
true calendar spread, so these offsets differ from a settlement-based spread by
that move. Raw per-contract partitions would make a same-timestamp method
possible; until then the as-of label is what pins which method produced a series.

Fail-closed at the edges: a session holding more than one `instrument_id`, or a
roll boundary with a session missing beside it, refuses rather than fold an
unrelated move into the offset. Reads past the as-of horizon refuse too, in
`data.source`.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import duckdb

from bedivere.core.types import Candle, Timeframe
from bedivere.data.lake.layout import SeriesId, session_path, stored_labels
from bedivere.data.lake.schema import BAR_SELECT, check_described_schema


class AdjustmentError(Exception):
    """The adjusted view cannot be derived (or served) fail-closed."""


# `date.fromisoformat` alone is NOT enough: it also accepts '20260612' and
# '2026-W24-5', and the session filter below compares lexicographically, so a
# basic-format pin would sort before every dashed label and silently include
# sessions and rolls past the intended date.
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def require_iso_date(value: str, what: str) -> None:
    """Canonical `YYYY-MM-DD`, or a refusal naming what was wrong."""
    if not _ISO_DATE_RE.match(value):
        raise AdjustmentError(f"{what} must be a canonical ISO date (YYYY-MM-DD), got {value!r}")
    try:
        date.fromisoformat(value)
    except ValueError as e:
        raise AdjustmentError(f"{what} is not a real calendar date: {value!r}") from e


@dataclass(frozen=True, slots=True)
class EraBoundary:
    """The first session of a NEW contract era, and the gap stepped across.

    `gap` = (new era's first open) - (old era's last close): the number added to
    every bar STRICTLY BEFORE `first_ts` to express it in the next era's terms.
    Additive, not ratio — this is the variant whose R-multiples and point
    distances survive.
    """

    first_session: str
    first_ts: int
    gap: float


@dataclass(frozen=True, slots=True)
class Adjustment:
    """A derived, as-of-pinned back-adjustment for one (series, timeframe).

    A value object: derive once, apply to any bars of the same series and
    timeframe. `horizon_ts` is the last stored bar at or before the as-of date
    — no bar past it is ever served through this view.
    """

    as_of: str
    boundaries: tuple[EraBoundary, ...]
    horizon_ts: int
    # Parallel precomputes so `offset_at` is one bisect and one index; `apply`
    # runs per bar over a whole backtest range. suffix[i] = sum of gaps of
    # boundaries[i:].
    _firsts: tuple[int, ...]
    _suffix: tuple[float, ...]

    def offset_at(self, ts: int) -> float:
        """Cumulative gap for a bar close-stamped `ts` (0.0 in the newest era)."""
        i = bisect_right(self._firsts, ts)
        return self._suffix[i] if i < len(self._suffix) else 0.0

    def apply(self, bars: list[Candle]) -> list[Candle]:
        """Shift every price field; leave stamps, volume and `partial` alone."""
        out: list[Candle] = []
        for c in bars:
            off = self.offset_at(c.timestamp)
            if off == 0.0:
                out.append(c)
                continue
            out.append(
                replace(
                    c,
                    open=c.open + off,
                    high=c.high + off,
                    low=c.low + off,
                    close=c.close + off,
                )
            )
        return out


# Calendar-day distances two ADJACENT stored sessions may legitimately sit apart
# at a roll boundary: 1 (consecutive weekdays), 3 (Fri->Mon), 4 (a long-weekend
# holiday). 2 is always a missing mid-week session and >4 is a hole, either of
# which would fold the absent sessions' move into the offset. The one blind spot:
# a 4-day Thu->Mon step cannot tell "Friday holiday" from "Friday missing"
# without the session calendar, which this module does not own.
_ALLOWED_BOUNDARY_GAPS_DAYS = (1, 3, 4)

# Per-session aggregates over an explicit file list, grouped by filename.
# `instrument_id` is checked with count(DISTINCT)/arg_min, which sees every bar
# rather than just the day's first and last.
_SESSIONS_SQL = """
SELECT filename,
       min(ts)                          AS first_ts,
       max(ts)                          AS last_ts,
       arg_min("open", ts)              AS first_open,
       arg_max("close", ts)             AS last_close,
       arg_min(instrument_id, ts)       AS iid,
       count(DISTINCT instrument_id)    AS id_count
  FROM {rel}
 GROUP BY filename
"""


def _sql_paths(paths: list[Path]) -> str:
    quoted = ("'" + p.as_posix().replace("'", "''") + "'" for p in paths)
    return "[" + ", ".join(quoted) + "]"


def back_adjustment(root: Path, sid: SeriesId, timeframe: Timeframe, as_of: str) -> Adjustment:
    """Derive the as-of-pinned back-adjustment from the stored raw series.

    The session set is `stored_labels(...) <= as_of` — labels resolved in
    Python, files then named explicitly, so later rolls are excluded by
    construction and the same call reproduces the same series after the lake has
    grown past the pin.
    """
    require_iso_date(as_of, "asOf")

    labels = [lbl for lbl in stored_labels(root, sid, timeframe) if lbl <= as_of]
    if not labels:
        raise AdjustmentError(
            f"{sid} {timeframe.value} holds nothing at or before {as_of} — "
            "an adjustment needs at least one stored session to anchor on"
        )
    paths = [session_path(root, sid, timeframe, lbl) for lbl in labels]
    path_list = _sql_paths(paths)

    conn = duckdb.connect(":memory:")
    try:
        # Pin the stored dtypes before any value is read, so a drifted close or
        # instrument_id refuses rather than casting into the offsets.
        described_raw = conn.execute(
            f"DESCRIBE SELECT {BAR_SELECT} FROM "
            f"read_parquet({path_list}, hive_partitioning=false)"
        ).fetchall()
        check_described_schema(
            [(str(r[0]), str(r[1])) for r in described_raw],
            context=f"{sid} {timeframe.value} adjustment derivation",
        )
        rel = f"read_parquet({path_list}, hive_partitioning=false, filename=true)"
        rows = conn.execute(_SESSIONS_SQL.format(rel=rel)).fetchall()
    except duckdb.Error as e:
        raise AdjustmentError(
            f"cannot read {sid} {timeframe.value} for adjustment derivation: {e}"
        ) from e
    finally:
        conn.close()

    by_label: dict[str, tuple[int, int, float, float, int, int]] = {}
    for filename, first_ts, last_ts, first_open, last_close, iid, id_count in rows:
        by_label[_session_of(str(filename))] = (
            int(first_ts),
            int(last_ts),
            float(first_open),
            float(last_close),
            int(iid),
            int(id_count),
        )

    boundaries: list[EraBoundary] = []
    prev_label: str | None = None
    prev_id: int | None = None
    prev_close: float | None = None
    horizon_ts = 0
    for label in labels:
        if label not in by_label:
            raise AdjustmentError(
                f"session {label} of {sid} {timeframe.value} is stored but returned no "
                "rows — the partition is unreadable or empty"
            )
        first_ts, last_ts, first_open, last_close, iid, id_count = by_label[label]
        if id_count != 1:
            raise AdjustmentError(
                f"session {label} of {sid} {timeframe.value} holds {id_count} instrument "
                "ids — the stored series is malformed"
            )
        if prev_id is not None and iid != prev_id and prev_close is not None:
            assert prev_label is not None
            hole = (date.fromisoformat(label) - date.fromisoformat(prev_label)).days
            if hole not in _ALLOWED_BOUNDARY_GAPS_DAYS:
                raise AdjustmentError(
                    f"the roll between {prev_label} and {label} of {sid} {timeframe.value} "
                    f"spans {hole} calendar days — not a legitimate weekday/weekend step, so "
                    "a session is missing at the boundary. The gap would fold the absent "
                    "sessions' move into the offset; re-ingest the missing days first."
                )
            boundaries.append(
                EraBoundary(first_session=label, first_ts=first_ts, gap=first_open - prev_close)
            )
        prev_label = label
        prev_id = iid
        prev_close = last_close
        horizon_ts = last_ts

    suffix: list[float] = [0.0] * (len(boundaries) + 1)
    for i in range(len(boundaries) - 1, -1, -1):
        suffix[i] = boundaries[i].gap + suffix[i + 1]

    return Adjustment(
        as_of=as_of,
        boundaries=tuple(boundaries),
        horizon_ts=horizon_ts,
        _firsts=tuple(b.first_ts for b in boundaries),
        _suffix=tuple(suffix[:-1] or ()),
    )


def _session_of(filename: str) -> str:
    """The session label inside a partition path (`.../session=LABEL/bars.parquet`)."""
    for part in Path(filename).parts:
        if part.startswith("session="):
            return part.removeprefix("session=")
    raise AdjustmentError(f"no session= component in partition path {filename!r}")
