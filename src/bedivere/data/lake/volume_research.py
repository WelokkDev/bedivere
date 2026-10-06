"""Strategy-independent research access and causal alignment for volume bars.

No feature recipes, labels, or strategy imports belong here. A join is refused
unless both frames carry explicit, identical provenance: `load_volume_frame`
verifies what partition footers record, and the caller attests the rest.
Bar-construction policy is deliberately not compared.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import duckdb
import pandas as pd  # pyright: ignore[reportMissingTypeStubs]

from bedivere.core.session_days import SessionDay, SessionDays
from bedivere.data.lake.layout import SeriesId

# Re-exported: the contract lives in a pandas-free module the trade reader shares.
from bedivere.data.lake.provenance import FIELDS as _FIELDS
from bedivere.data.lake.provenance import Provenance as Provenance
from bedivere.data.lake.provenance import ProvenanceError as ProvenanceError
from bedivere.data.lake.provenance import require_explicit as _explicit
from bedivere.data.lake.research_join import (
    BAR_TIMESTAMPS,
    bar_provenance,
    bar_window,
    check_bar_coverage,
    check_bar_rows,
    check_observation_rows,
    check_relation,
    check_single_definition,
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
)
from bedivere.data.lake.schema import AS_TRADED, BarSchemaError
from bedivere.data.lake.volume import VolumeSpec
from bedivere.data.lake.volume_store import (
    COLUMNS,
    checked_volume_day,
    volume_path,
)

# Footers never name a vendor, so the feed and its instrument-ID namespace can
# only be attested by the caller.
_LOADER_VERIFIED = _FIELDS - {"feed", "instrument_namespace"}

_ORDER = "__observation_order"


@dataclass(frozen=True, slots=True)
class BarSource:
    """What a loaded frame's bars were built from, as its partitions record it.

    Recorded at build time, not re-checked here: a resumed build is what
    compares the inputs with their current fingerprints.
    """

    kind: str
    """The definition's `source`: `trades`, `ohlcv-1s`, or `trades-1s`."""

    inputs: tuple[tuple[str, str], ...]
    """(session label, recorded source) per partition read, in calendar order.
    A source is `<name>:sha256:<hex>`, prefixed `same-trades-1s:` for rebuilt
    seconds and suffixed `:selection:<fingerprint>` for a continuous series."""

    def __deepcopy__(self, memo: dict[int, object]) -> BarSource:
        # Deeply immutable; pandas deep-copies attrs on nearly every operation.
        return self


def volume_coverage(
    root: Path,
    sid: SeriesId,
    spec: VolumeSpec,
    days: SessionDays,
) -> list[dict[str, str]]:
    """Every expected session as missing, invalid, or valid.

    Checks stored integrity only, not whether the external source has changed.
    """
    results: list[dict[str, str]] = []
    for day in days.days:
        path = volume_path(root, sid, spec, days, day.label)
        if not path.is_file():
            results.append({"session": day.label, "status": "missing", "path": str(path)})
            continue
        try:
            _, source = checked_volume_day(root, sid, spec, days, day)
        except (ValueError, OSError, duckdb.Error, BarSchemaError) as exc:
            results.append(
                {"session": day.label, "status": "invalid", "error": str(exc), "path": str(path)}
            )
        else:
            results.append(
                {"session": day.label, "status": "valid", "path": str(path), "source": source}
            )
    return results


def load_volume_frame(
    root: Path,
    sid: SeriesId,
    spec: VolumeSpec,
    days: SessionDays,
    *,
    feed: str,
    include_partial: bool = False,
    start_ns: int | None = None,
    end_ns: int | None = None,
    allow_missing: bool = False,
) -> pd.DataFrame:
    """Load one definition over explicit sessions.

    `feed` is the caller's attestation: footers never name a vendor.
    `[start_ns, end_ns)` filters on availability, and joins refuse a decision
    that looks outside it. Missing sessions fail unless `allow_missing`;
    corrupt ones always do. Remainders are excluded by default.
    """
    _explicit("feed", feed)
    for bound in (start_ns, end_ns):
        if bound is not None and (type(bound) is not int or not 0 <= bound < 2**63):
            raise ValueError("availability bounds must be non-negative int64 nanoseconds")
    if start_ns is not None and end_ns is not None and end_ns <= start_ns:
        raise ValueError("end_ns must be after start_ns")
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    loaded: list[SessionDay] = []
    inputs: list[tuple[str, str]] = []
    for day in days.days:
        path = volume_path(root, sid, spec, days, day.label)
        if not path.is_file():
            if not allow_missing:
                raise FileNotFoundError(f"missing volume-bar session {day.label}: {path}")
            missing.append(day.label)
            continue
        stored, source = checked_volume_day(root, sid, spec, days, day)
        loaded.append(day)
        inputs.append((day.label, source))
        for bar in stored:
            if (
                bar.is_partial
                and not include_partial
                or start_ns is not None
                and bar.available_ns < start_ns
                or end_ns is not None
                and bar.available_ns >= end_ns
            ):
                continue
            rows.append(
                {
                    **{name: getattr(bar, name) for name in COLUMNS},
                    "session": day.label,
                    "dataset": sid.dataset,
                    "symbol": sid.symbol,
                    "series": sid.series,
                    "definition": spec.key(days),
                }
            )
    identity = ["dataset", "symbol", "series", "definition", "session"]
    frame = pd.DataFrame(rows, columns=[*COLUMNS, *identity])
    dtypes = {
        name: {"BIGINT": "int64", "UINTEGER": "uint32", "DOUBLE": "float64", "BOOLEAN": "bool"}[
            kind
        ]
        for name, kind in COLUMNS.items()
    }
    frame = frame.astype({**dtypes, **dict.fromkeys(identity, "object")})  # pyright: ignore[reportUnknownMemberType]
    definition = json.loads(spec.definition(days))
    frame.attrs.update(
        {
            "availability_window": (start_ns, end_ns),
            "bar_definition": definition,
            "bar_source": BarSource(spec.source, tuple(inputs)),
            "missing_sessions": missing,
            "price_basis": AS_TRADED,
            "provenance": Provenance(
                feed=feed,
                dataset=sid.dataset,
                symbol=sid.symbol,
                series=sid.series,
                # Partitions store the feed's own instrument_id for their dataset.
                instrument_namespace=f"{feed}:{sid.dataset}",
                price_basis=AS_TRADED,
                sessions=replace(days, days=tuple(loaded)),
                time_unit="ns",
                epoch="unix",
                clock=definition["clock"],
                # Reading no partition verifies nothing.
                verified=_LOADER_VERIFIED if loaded else frozenset(),
            ),
        }
    )
    return frame


def align_available(
    observations: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    observation_provenance: Provenance,
    max_age_ns: int,
    allow_exact_matches: bool = False,
    prefix: str = "volume_",
) -> pd.DataFrame:
    """Attach the latest available bar within the same session and instrument.

    Observations need `asof_ns` (int64 Unix ns), `session` and `instrument_id`;
    the result is the same rows with bar columns added. Bars qualify by
    `available_ns`, never `end_ns`. Exact matches default off: the order of a
    decision and a bar at the same timestamp is unknown. Stale or absent
    matches are NULL; incompatible provenance raises ProvenanceError.
    """
    if type(max_age_ns) is not int or not 0 <= max_age_ns < 2**63:
        raise ValueError("max_age_ns must be an explicit non-negative int64 duration")
    obs_names = column_names(observations)
    bar_names = column_names(bars)
    require_observation_columns(obs_names, prefix=prefix, attached=COLUMNS, reserved=(_ORDER,))
    require_bar_columns(bar_names)
    declared = declared_provenance(observation_provenance)
    loaded, definition = bar_provenance(bars)
    refuse(
        "observations and volume bars have incompatible provenance",
        conflicts(declared, loaded, ("observations", "bars")),
        "Nothing is reconciled or mapped automatically: join bars built from the observations' "
        "own source, or convert the observations upstream and declare what they then are.",
    )
    require_attrs(observations.attrs, declared, "observations")
    require_attrs(bars.attrs, loaded, "bars")
    window = bar_window(bars)
    key = definition_key(definition)
    with duckdb.connect(":memory:") as con:
        con.register("observations", observation_keys(observations, obs_names, declared, _ORDER))
        con.register("research_bars", bars)
        check_relation(con, "observations", "observations", ["asof_ns"], obs_names, declared)
        check_relation(con, "research_bars", "bars", BAR_TIMESTAMPS, bar_names, loaded)
        check_single_definition(con, key)
        check_observation_rows(con, declared)
        check_bar_rows(con, loaded)
        check_bar_coverage(
            con,
            window,
            reach=f"o.asof_ns - {max_age_ns}",
            allow_exact_matches=allow_exact_matches,
        )
        cmp = ">=" if allow_exact_matches else ">"
        projection = ", ".join(
            f"CASE WHEN o.asof_ns - b.available_ns <= {max_age_ns} "
            f"THEN b.{quoted(name)} END AS {quoted(prefix + name)}"
            for name in COLUMNS
        )
        attached = con.execute(
            "WITH b AS (SELECT * FROM research_bars QUALIFY row_number() OVER "
            "(PARTITION BY session, instrument_id, available_ns ORDER BY bar_id DESC) = 1) "
            f"SELECT {projection} FROM observations o ASOF LEFT JOIN b "
            f"ON o.session = b.session AND o.instrument_id = b.instrument_id AND o.asof_ns {cmp} b.available_ns "
            f"ORDER BY o.{_ORDER}"
        ).df()
    # Bar columns attach by position, whatever the index.
    result = observations.copy()
    for name in COLUMNS:
        result[prefix + name] = attached[prefix + name].array  # pyright: ignore[reportUnknownMemberType]
    result.attrs = {
        **{key: value for key, value in bars.attrs.items() if key != "provenance"},
        "provenance": declared,
        "bar_provenance": loaded,
    }
    return result
