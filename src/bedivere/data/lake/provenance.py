"""The research join contract: what a frame's rows are, declared field by field.

Importable without pandas or DuckDB. Carries no artifact hashes or receipts:
those describe one read's bytes and live in `trades.TradeSource`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, cast

from bedivere.core.session_days import SessionDays

_NS: Final[int] = 1_000_000_000


class ProvenanceError(ValueError):
    """Provenance is missing, unknown, mixed, contradictory, or incompatible."""


# Stand-ins for "we do not know", refused wherever a declared value belongs.
PLACEHOLDERS: Final[frozenset[str]] = frozenset({"", "unknown", "mixed", "none", "null", "nan"})

FIELDS: Final[frozenset[str]] = frozenset(
    {
        "feed",
        "dataset",
        "symbol",
        "series",
        "instrument_namespace",
        "price_basis",
        "sessions",
        "time_unit",
        "epoch",
        "clock",
    }
)


def require_explicit(name: str, value: object) -> None:
    if not isinstance(value, str) or value != value.strip() or value.lower() in PLACEHOLDERS:
        raise ProvenanceError(
            f"provenance {name} must be one explicit value, not {value!r}; "
            "unknown or mixed provenance cannot be joined"
        )


@dataclass(frozen=True, slots=True)
class Provenance:
    """What a research frame's rows are: no defaults, no placeholders.

    `verified` names the fields a loader checked against stored data; the rest
    are the builder's attestation. Joins compare both kinds identically.
    """

    feed: str
    """Vendor or pipeline, e.g. `databento` or `nt8`."""

    dataset: str
    symbol: str
    series: str
    """Contract selection, e.g. `raw.NQU6` or `v.0`."""

    instrument_namespace: str
    """Who assigned the `instrument_id` values, e.g. `databento:GLBX.MDP3`."""

    price_basis: str
    """`as_traded`, or the adjustment applied, e.g. `back_adjusted`."""

    sessions: SessionDays
    """The calendar the `session` labels came from."""

    time_unit: str
    """Always `ns`."""

    epoch: str
    """Always `unix`."""

    clock: str
    """E.g. Databento's `ts_recv`."""

    verified: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        for name in sorted(FIELDS - {"sessions"}):
            require_explicit(name, getattr(self, name))
        if (self.time_unit, self.epoch) != ("ns", "unix"):
            raise ProvenanceError(
                f"timestamps must be integer Unix nanoseconds, not time_unit={self.time_unit!r} "
                f"epoch={self.epoch!r}; convert them upstream without floats, then declare "
                "time_unit='ns', epoch='unix'"
            )
        sessions = cast(object, self.sessions)
        if not isinstance(sessions, SessionDays):
            raise ProvenanceError("provenance sessions must be resolved SessionDays")
        require_explicit("calendar", sessions.template)
        require_explicit("timezone", sessions.timezone)
        labels = [day.label for day in sessions.days]
        if len(set(labels)) != len(labels):
            raise ProvenanceError("each session label must resolve to exactly one boundary pair")
        for day in sessions.days:
            require_explicit("session label", day.label)
            start, end = day.start_unix, day.end_unix
            if type(start) is not int or type(end) is not int or not 0 <= start < end < 2**63 // _NS:
                raise ProvenanceError(
                    f"session {day.label!r} needs integer Unix-second bounds with start < end"
                )
        verified = cast(object, self.verified)
        if not isinstance(verified, frozenset) or not cast(frozenset[object], verified) <= FIELDS:
            raise ProvenanceError(f"verified must be a frozenset drawn from {sorted(FIELDS)}")

    def __deepcopy__(self, memo: dict[int, object]) -> Provenance:
        # Deeply immutable; pandas deep-copies attrs on nearly every operation.
        return self
