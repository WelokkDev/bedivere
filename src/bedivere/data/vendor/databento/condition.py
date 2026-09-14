"""Databento's per-day data-quality feed — the only way to learn that a day you
already hold is suspect. Conditions, verbatim from the API docs:

    available: the data is available with no known issues
    degraded:  the data is available, but there may be missing data or other
               correctness issues
    pending:   the data is not yet available, but may be available soon
    missing:   the data is not available

`degraded` is the one that matters, because the bars ARE there and they look
like bars. Bar count cannot diagnose it: a degraded weekday can sit at 80-90% of
a normal one while a healthy Sunday session sits near 10%. Only this feed says
so, and `last_modified_date` is the only signal that a day ingested months ago
has since been reissued.

Every batch job ships a `condition.json` beside the data, so history for a range
already downloaded loads with no API key and no further cost. DuckDB is imported
lazily, inside the two functions that touch the Parquet store, so parsing and
the severity fold stay usable on a base install.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, cast


class Condition(StrEnum):
    """A day's data-quality condition. Values are Databento's own strings."""

    AVAILABLE = "available"
    DEGRADED = "degraded"
    PENDING = "pending"
    MISSING = "missing"

    @property
    def severity(self) -> int:
        """Ordering for "worst condition wins" folds."""
        return _SEVERITY[self]

    @property
    def usable(self) -> bool:
        """True only for `available`. Everything else is a reason to stop."""
        return self is Condition.AVAILABLE


_SEVERITY: Final[dict[Condition, int]] = {
    Condition.AVAILABLE: 0,
    Condition.DEGRADED: 1,
    Condition.PENDING: 2,
    Condition.MISSING: 3,
}


@dataclass(frozen=True, slots=True)
class DayCondition:
    """One UTC date's condition, as published."""

    date: str
    """ISO `yyyy-mm-dd`, a UTC calendar date — NOT a session-day label."""

    condition: Condition
    last_modified: str | None
    """ISO date the day was last generated, or None when `missing`."""


# The store's declared shape, as DuckDB's `DESCRIBE` spells it.
CONDITION_COLUMNS: Final[tuple[tuple[str, str], ...]] = (
    ("date", "VARCHAR"),
    ("condition", "VARCHAR"),
    ("last_modified", "VARCHAR"),
)


def parse_conditions(payload: object) -> list[DayCondition]:
    """Parse a `condition.json` body (a list of `{date, condition, ...}`).

    Loud on anything unexpected: a feed that silently drops rows it does not
    understand gives a clean bill of health for days it never checked.
    """
    if not isinstance(payload, list):
        raise ValueError("condition payload must be a JSON array")
    entries = cast("list[object]", payload)
    rows: list[DayCondition] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"condition[{i}] must be an object")
        item = cast("dict[str, object]", entry)
        day = item.get("date")
        raw = item.get("condition")
        modified = item.get("last_modified_date")
        if not isinstance(day, str):
            raise ValueError(f"condition[{i}].date must be a string")
        if not isinstance(raw, str):
            raise ValueError(f"condition[{i}].condition must be a string")
        try:
            condition = Condition(raw)
        except ValueError as exc:
            raise ValueError(
                f"condition[{i}].condition {raw!r} is not a known Databento condition "
                f"({', '.join(c.value for c in Condition)})"
            ) from exc
        if modified is not None and not isinstance(modified, str):
            raise ValueError(f"condition[{i}].last_modified_date must be a string or null")
        rows.append(DayCondition(date=day, condition=condition, last_modified=modified))
    rows.sort(key=lambda r: r.date)
    return rows


def load_condition_json(path: Path) -> list[DayCondition]:
    """Load the `condition.json` that ships beside every batch download."""
    with path.open(encoding="utf-8") as fh:
        return parse_conditions(json.load(fh))


API_KEY_ENV: Final[str] = "DATABENTO_API_KEY"

HTTP_EXTRA_HINT: Final[str] = (
    "install the lake extra (`pip install 'bedivere[lake]'`, or `uv sync --extra lake`) "
    "— `requests` ships with it, because the vendor edge and the lake are bought "
    "together: the only reason to call this API is to fill a lake"
)


class VendorHttpUnavailable(ImportError):
    """`requests` is not installed, so no vendor call can be made."""


class DatabentoApiError(Exception):
    """The API refused, or answered something this package cannot read.

    `status` is the HTTP status when there was one, so a caller can tell a 404
    (no such job) from a refusal it should quote verbatim.
    """

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class PartialSymbolsError(DatabentoApiError):
    """HTTP 206, documented as "successful request, with partially resolved
    symbols". Refused here: pricing or buying a subset of what was asked is a
    silent substitution, and a typo in one symbol of several would otherwise
    go through as the others."""


def resolve_api_key(api_key: str | None = None) -> str:
    """An explicit key, else `$DATABENTO_API_KEY`, else a refusal."""
    key = api_key if api_key else os.environ.get(API_KEY_ENV)
    if not key:
        raise ValueError(
            f"no Databento API key — pass api_key or set {API_KEY_ENV} "
            "(the bedivere CLIs also read it from ./.env)"
        )
    return key


def http_client() -> Any:
    """The `requests` module, or a sentence naming the extra that carries it.

    The same courtesy `bedivere.data.source.lake_module` gives the lake stack,
    and for the same reason: `bedivere-data cost` on a base install should say
    which extra is missing, not raise `ModuleNotFoundError` from inside an
    import statement three frames down.

    Imported lazily and through one seam, so the conditions vocabulary — the
    enum, the parser, the severity fold `bedivere.data.quality` runs — stays
    usable on a machine with no network stack at all.
    """
    try:
        import requests
    except ImportError as e:
        raise VendorHttpUnavailable(
            f"the Databento HTTP edge needs `requests`: {e}\n{HTTP_EXTRA_HINT}"
        ) from e
    return requests


def raise_for_status(response: Any, path: str) -> None:
    """Refuse anything but a plain success, quoting the API's own message.

    `requests.raise_for_status()` says "400 Client Error" and throws the body
    away — and the body's `detail` is where the API names the offending
    parameter. It also passes 206, which the API documents as "partially
    resolved symbols"; on a billed edge that is a refusal, not a success.
    """
    status = response.status_code
    if status == 206:
        raise PartialSymbolsError(
            f"{path}: HTTP 206 — the API resolved only some of the requested symbols; "
            f"check each one against the dataset.\n  {_detail(response)}",
            status=status,
        )
    if 200 <= status < 300:
        return
    raise DatabentoApiError(f"{path}: HTTP {status} — {_detail(response)}", status=status)


def _detail(response: Any) -> str:
    return response.text.strip()[:500]


def fetch_conditions(
    dataset: str, start_date: str, end_date: str, *, api_key: str | None = None
) -> list[DayCondition]:
    """Pull conditions from the API. Free — this endpoint is not billed.

    Both dates are INCLUSIVE UTC calendar dates, `yyyy-mm-dd` — unlike the
    `start`/`end` of a history request, whose `end` is exclusive.

    A direct HTTPS call rather than the vendor client package: same endpoint,
    same auth convention (the key as the basic-auth username).
    """
    path = "metadata.get_dataset_condition"
    response = http_client().get(
        f"https://hist.databento.com/v0/{path}",
        params={"dataset": dataset, "start_date": start_date, "end_date": end_date},
        auth=(resolve_api_key(api_key), ""),
        timeout=30,
    )
    raise_for_status(response, path)
    return parse_conditions(response.json())


def condition_path(root: Path, dataset: str) -> Path:
    return root / "_meta" / "condition" / f"dataset={dataset}" / "condition.parquet"


def write_conditions(root: Path, dataset: str, rows: list[DayCondition]) -> Path:
    """Persist a dataset's condition history, replacing any previous copy.

    Whole-file rather than a merge: a merge would have to decide what to do when
    a re-fetch disagrees with what is stored, and that disagreement IS the
    staleness signal. Row-at-a-time insertion is fine here — a condition history
    is hundreds of rows, and the seam's vectorized-write rule is about bars.
    """
    import duckdb

    path = condition_path(root, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE cond (date VARCHAR NOT NULL, condition VARCHAR NOT NULL, last_modified VARCHAR)"
    )
    con.executemany(
        "INSERT INTO cond VALUES (?, ?, ?)",
        [(r.date, r.condition.value, r.last_modified) for r in rows],
    )
    tmp = path.with_name(path.name + ".tmp")
    try:
        target = tmp.as_posix().replace("'", "''")
        con.execute(f"COPY cond TO '{target}' (FORMAT PARQUET, COMPRESSION 'zstd')")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
        con.close()
    return path


def read_conditions(root: Path, dataset: str) -> list[DayCondition]:
    """Load a dataset's stored condition history, or [] when none is stored."""
    import duckdb

    path = condition_path(root, dataset)
    if not path.is_file():
        return []
    con = duckdb.connect(":memory:")
    try:
        source = path.as_posix().replace("'", "''")
        described = [
            (str(row[0]), str(row[1]))
            for row in con.execute(
                f"DESCRIBE SELECT date, condition, last_modified FROM read_parquet('{source}')"
            ).fetchall()
        ]
        if described != list(CONDITION_COLUMNS):
            raise ValueError(
                f"{path}: stored condition schema {described} does not match "
                f"{list(CONDITION_COLUMNS)}"
            )
        fetched = con.execute(
            f"SELECT date, condition, last_modified FROM read_parquet('{source}') ORDER BY date"
        ).fetchall()
    finally:
        con.close()
    return [
        DayCondition(date=str(d), condition=Condition(str(c)), last_modified=_optional_str(m))
        for d, c, m in fetched
    ]


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise ValueError(f"condition last_modified is not a string: {value!r}")
