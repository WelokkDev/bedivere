"""Joins refuse unless observations and volume bars share explicit provenance."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from tests.lake_support import SKIP_REASON

pytest.importorskip("duckdb", reason=SKIP_REASON)
pytest.importorskip("pandas", reason=SKIP_REASON)
pytest.importorskip("databento_dbn", reason=SKIP_REASON)

import pandas as pd  # noqa: E402  # pyright: ignore[reportMissingTypeStubs]

from bedivere.core.session_days import SessionDay  # noqa: E402
from bedivere.data.lake.read import partition_kv  # noqa: E402
from bedivere.data.lake.volume import NS, VolumeSpec  # noqa: E402
from bedivere.data.lake.volume_research import (  # noqa: E402
    Provenance,
    ProvenanceError,
    align_available,
    load_volume_frame,
)
from bedivere.data.lake.volume_store import volume_path  # noqa: E402
from tests.test_volume import DAY, DAYS  # noqa: E402
from tests.test_volume_lake import SID  # noqa: E402
from tests.test_volume_research import OBSERVED, observed, seed  # noqa: E402

# pandas is optional and shipped without complete typing for dynamic columns.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false

# When the fixture's two full 1,000-contract bars become available.
FIRST = 100 * NS + 200
SECOND = 101 * NS + 100
NEXT = SessionDay("2026-06-16", DAY.end_unix, DAY.end_unix + 10)


def loaded(root: Path, spec: VolumeSpec | None = None) -> pd.DataFrame:
    return load_volume_frame(root, SID, seed(root, spec), DAYS, feed="databento")


def observation(asof_ns: int = FIRST + 1, **columns: object) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asof_ns": [asof_ns],
            "session": [DAY.label],
            "instrument_id": [42],
            **{name: [value] for name, value in columns.items()},
        }
    )


def join(
    observations: pd.DataFrame,
    bars: pd.DataFrame,
    context: Provenance = OBSERVED,
    *,
    max_age_ns: int = 10 * NS,
    allow_exact_matches: bool = False,
) -> pd.DataFrame:
    return align_available(
        observations,
        bars,
        observation_provenance=context,
        max_age_ns=max_age_ns,
        allow_exact_matches=allow_exact_matches,
    )


def bar_ids(joined: pd.DataFrame) -> list[int | None]:
    values = cast(list[Any], joined["volume_bar_id"].tolist())
    return [None if pd.isna(value) else int(value) for value in values]


def test_compatible_provenance_joins_any_bar_construction_policy(tmp_path: Path) -> None:
    # No threshold or boundary policy is part of the contract, only what the data is.
    for spec in (VolumeSpec(1000), VolumeSpec(100, boundary="split_trade")):
        joined = join(observation(), loaded(tmp_path, spec))
        assert bar_ids(joined) != [None]
        assert joined.attrs["provenance"] == OBSERVED
        assert joined.attrs["bar_definition"]["threshold"] == spec.threshold
        # What the bars were built from travels with them, outside the join contract.
        assert joined.attrs["bar_source"].kind == "trades"
        assert [label for label, _ in joined.attrs["bar_source"].inputs] == [DAY.label]
    stored = joined.attrs["bar_provenance"]
    assert stored.verified == {
        "dataset",
        "symbol",
        "series",
        "price_basis",
        "sessions",
        "time_unit",
        "epoch",
        "clock",
    }
    # Footers never name a vendor: the feed, and the ID namespace it implies, are attested.
    assert (stored.feed, stored.instrument_namespace) == ("databento", "databento:GLBX.MDP3")
    assert (stored.price_basis, stored.clock, stored.sessions) == ("as_traded", "ts_recv", DAYS)
    assert OBSERVED.verified == frozenset()


def test_frames_share_the_immutable_record_instead_of_copying_calendars(tmp_path: Path) -> None:
    bars = loaded(tmp_path)
    # pandas deep-copies attrs on nearly every operation, even column access.
    assert bars[bars["volume"] > 0].attrs["provenance"] is bars.attrs["provenance"]
    assert bars["close"].attrs["provenance"] is bars.attrs["provenance"]
    assert bars["close"].attrs["bar_source"] is bars.attrs["bar_source"]


def test_nt8_like_observations_never_receive_databento_bars(tmp_path: Path) -> None:
    bars = loaded(tmp_path)
    with pytest.raises(
        ProvenanceError,
        match=r"feed: observations 'nt8' \(attested\) vs bars 'databento' \(attested\)",
    ):
        join(observation(), bars, replace(OBSERVED, feed="nt8"))
    # A realistic NT8 export differs in several ways at once; each is named.
    nt8 = replace(
        OBSERVED,
        feed="nt8",
        instrument_namespace="nt8:instrument",
        price_basis="back_adjusted",
        sessions=replace(DAYS, template="CME US Index Futures ETH", timezone="America/Chicago"),
        clock="nt8_bar_close",
    )
    with pytest.raises(ProvenanceError) as refused:
        join(observation(), bars, nt8)
    for fact in ("feed", "instrument_namespace", "price_basis", "calendar", "timezone", "clock"):
        assert f"\n  {fact}: observations" in str(refused.value)
    assert "Nothing is reconciled or mapped automatically" in str(refused.value)


@pytest.mark.parametrize(
    "fact,value,stored",
    [
        ("dataset", "IFUS.IMPACT", "'GLBX.MDP3' (verified)"),
        ("symbol", "MNQ", "'NQ' (verified)"),
        ("series", "v.0", "'raw.NQZ5' (verified)"),
        ("price_basis", "back_adjusted", "'as_traded' (verified)"),
        ("clock", "ts_event", "'ts_recv' (verified)"),
    ],
)
def test_each_incompatible_identity_is_refused_by_name(
    tmp_path: Path, fact: str, value: str, stored: str
) -> None:
    changes: dict[str, Any] = {fact: value}
    with pytest.raises(
        ProvenanceError,
        match=rf"{fact}: observations '{re.escape(value)}' \(attested\) vs bars {re.escape(stored)}",
    ):
        join(observation(), loaded(tmp_path), replace(OBSERVED, **changes))


def test_equal_instrument_ids_from_another_namespace_are_not_the_same_contract(
    tmp_path: Path,
) -> None:
    bars = loaded(tmp_path)
    assert set(bars["instrument_id"]) == {42}
    # Same feed, dataset and number, but the observations carry a private ID map.
    remapped = replace(OBSERVED, instrument_namespace="private:contract_registry")
    with pytest.raises(
        ProvenanceError,
        match=r"instrument_namespace: observations 'private:contract_registry' \(attested\) "
        r"vs bars 'databento:GLBX.MDP3' \(attested\)",
    ):
        join(observation(), bars, remapped)
    assert bar_ids(join(observation(), bars)) == [0]


def test_missing_provenance_is_refused(tmp_path: Path) -> None:
    bars = loaded(tmp_path)
    stripped = bars.copy()
    stripped.attrs.clear()
    hand_built = pd.DataFrame(bars.to_dict(orient="list"))
    # Frames loaded under different attestations lose attrs when concatenated.
    replica = load_volume_frame(tmp_path, SID, VolumeSpec(1000), DAYS, feed="databento-replica")
    for frame in (stripped, hand_built, pd.concat([bars, replica], ignore_index=True)):
        with pytest.raises(ProvenanceError, match="bars carry no loader provenance"):
            join(observation(), frame)
    with pytest.raises(ProvenanceError, match="observation_provenance must be a Provenance"):
        join(observation(), bars, cast(Any, None))


@pytest.mark.parametrize(
    "fact,value",
    [
        ("feed", "unknown"),
        ("price_basis", "mixed"),
        ("dataset", ""),
        ("series", " raw.NQZ5"),
        ("clock", "None"),
        ("instrument_namespace", "null"),
    ],
)
def test_unknown_mixed_or_blank_declarations_are_refused(fact: str, value: str) -> None:
    changes: dict[str, Any] = {fact: value}
    with pytest.raises(ProvenanceError, match=f"provenance {fact} must be one explicit value"):
        replace(OBSERVED, **changes)


def test_incomplete_calendars_and_attestations_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ProvenanceError, match="provenance timezone must be one explicit value"):
        replace(OBSERVED, sessions=replace(DAYS, timezone="unknown"))
    with pytest.raises(ProvenanceError, match="exactly one boundary pair"):
        replace(OBSERVED, sessions=replace(DAYS, days=(DAY, replace(DAY, start_unix=200, end_unix=210))))
    with pytest.raises(ProvenanceError, match="verified must be a frozenset"):
        replace(OBSERVED, verified=frozenset({"everything"}))
    # The attestation is checked before any partition is read.
    with pytest.raises(ProvenanceError, match="provenance feed must be one explicit value"):
        load_volume_frame(tmp_path, SID, VolumeSpec(1000), DAYS, feed="unknown")


def test_row_data_and_attributes_cannot_be_overridden_by_a_declaration(tmp_path: Path) -> None:
    bars = loaded(tmp_path)
    mixed = pd.concat([observation(feed="databento"), observation(feed="nt8")])
    claimed = observation()
    claimed.attrs["price_basis"] = "back_adjusted"
    restated = observation()
    restated.attrs["provenance"] = replace(OBSERVED, feed="nt8")
    for observations, message in (
        (
            observation(price_basis="back_adjusted"),
            "observations column 'price_basis' holds 'back_adjusted', "
            "but the declared price_basis is 'as_traded'",
        ),
        (observation(dataset=None), "observations column 'dataset' has NULL (unknown) values"),
        (mixed, "observations column 'feed' mixes ['databento', 'nt8']"),
        (claimed, "attrs['price_basis'] is 'back_adjusted', declared price_basis 'as_traded'"),
        (restated, "feed: observations.attrs 'nt8' (attested) vs declared 'databento'"),
    ):
        with pytest.raises(ProvenanceError, match=re.escape(message)):
            join(observations, bars)
    moved = bars.copy()
    moved.attrs["bar_definition"] = {**bars.attrs["bar_definition"], "timezone": "America/Chicago"}
    backdated = bars.copy()
    backdated.loc[0, "available_ns"] = backdated.loc[0, "end_ns"] - 1
    for frame, message in (
        (
            bars.assign(series="raw.NQH7"),
            "bars column 'series' holds 'raw.NQH7', but the declared series is 'raw.NQZ5'",
        ),
        (moved, "attrs['bar_definition']['timezone'] is 'America/Chicago', declared timezone 'UTC'"),
        (backdated, "cannot become available before it ends"),
    ):
        with pytest.raises(ProvenanceError, match=re.escape(message)):
            join(observation(), frame)


@pytest.mark.parametrize(
    "fact,change",
    [("calendar", {"template": "cme_us_index_futures_rth"}), ("timezone", {"timezone": "Etc/GMT+5"})],
)
def test_session_template_and_timezone_must_match(
    tmp_path: Path, fact: str, change: dict[str, Any]
) -> None:
    context = replace(OBSERVED, sessions=replace(DAYS, **change))
    stored = {"calendar": DAYS.template, "timezone": DAYS.timezone}[fact]
    with pytest.raises(
        ProvenanceError,
        match=rf"{fact}: observations '{re.escape(next(iter(change.values())))}' \(attested\) "
        rf"vs bars '{stored}' \(verified\)",
    ):
        join(observation(), loaded(tmp_path), context)


def test_a_shared_session_label_must_resolve_to_the_same_boundaries(tmp_path: Path) -> None:
    longer = replace(DAYS, days=(replace(DAY, end_unix=DAY.end_unix + 3600),))
    with pytest.raises(
        ProvenanceError,
        match=re.escape("session '2026-06-15': observations (100, 3710] vs bars (100, 110]"),
    ):
        join(observation(), loaded(tmp_path), replace(OBSERVED, sessions=longer))


def test_observation_rows_must_fit_their_declared_sessions(tmp_path: Path) -> None:
    bars = loaded(tmp_path)
    with pytest.raises(
        ProvenanceError, match="observation session '2026-06-16' is not in the declared calendar"
    ):
        join(observation().assign(session=NEXT.label), bars)
    with pytest.raises(
        ProvenanceError, match=r"asof_ns=110000000001 lies outside its declared session '2026-06-15'"
    ):
        join(observation(asof_ns=DAY.end_unix * NS + 1), bars)
    # Both session edges belong to the session; neither is refused.
    edges = pd.concat([observation(asof_ns=DAY.start_unix * NS), observation(asof_ns=DAY.end_unix * NS)])
    assert bar_ids(join(edges, bars)) == [None, 1]


def test_timestamp_units_are_declared_and_never_rescaled(tmp_path: Path) -> None:
    for change in ({"time_unit": "us"}, {"time_unit": "ms"}, {"epoch": "dotnet_ticks"}):
        changes: dict[str, Any] = change
        with pytest.raises(ProvenanceError, match="timestamps must be integer Unix nanoseconds"):
            replace(OBSERVED, **changes)
    bars = loaded(tmp_path)
    # Microseconds mislabelled as nanoseconds are not guessed back; they miss the session.
    with pytest.raises(ProvenanceError, match="lies outside its declared session"):
        join(observation(asof_ns=(FIRST + 1) // 1000), bars)
    for values in (
        [float(FIRST + 1)],
        pd.to_datetime([FIRST + 1]),
        pd.array([FIRST + 1], dtype="UInt64"),
    ):
        with pytest.raises(ValueError, match="integer dtypes holding int64 Unix nanoseconds"):
            join(observation().assign(asof_ns=values), bars)


def test_bars_are_invisible_until_available(tmp_path: Path) -> None:
    bars = loaded(tmp_path)
    times = [FIRST - 1, FIRST, FIRST + 1, SECOND - 1, SECOND, SECOND + 1]
    observations = pd.concat([observation(asof_ns=t) for t in times], ignore_index=True)
    assert bar_ids(join(observations, bars)) == [None, None, 0, 0, 0, 1]
    assert bar_ids(join(observations, bars, allow_exact_matches=True)) == [None, 0, 0, 0, 1, 1]


def test_compatible_but_unmatched_observations_get_null_not_an_error(tmp_path: Path) -> None:
    bars = loaded(tmp_path)
    context = observed(replace(DAYS, days=(DAY, NEXT)))
    observations = pd.DataFrame(
        {
            "asof_ns": [FIRST - 1, FIRST + 1, NEXT.start_unix * NS + 1, SECOND + NS + 1],
            "session": [DAY.label, DAY.label, NEXT.label, DAY.label],
            "instrument_id": [42, 43, 42, 42],
        }
    )
    assert bar_ids(join(observations, bars, context, max_age_ns=NS)) == [None] * 4
    # Even a generous age limit never crosses a session or an instrument.
    assert bar_ids(join(observations, bars, context, max_age_ns=3600 * NS)) == [None, None, None, 1]
    # Missing sessions verify nothing, yet a compatible join simply finds no bars.
    empty = load_volume_frame(
        tmp_path, SID, VolumeSpec(1234), DAYS, feed="databento", allow_missing=True
    )
    assert empty.attrs["provenance"].verified == frozenset()
    assert empty.attrs["provenance"].sessions.days == ()
    assert bar_ids(join(observation(), empty)) == [None]


def test_existing_partitions_load_unchanged_under_the_same_definition_hash(tmp_path: Path) -> None:
    spec = seed(tmp_path)
    nearest = VolumeSpec(1000, "ohlcv-1s", "nearest_second")
    # Protect the hashes of datasets written before provenance checks existed.
    assert spec.key(DAYS) == "aff6136ba7b60f4fcb5e573d2208c6825f297f2a0f77b0beef29221753e983ed"
    assert nearest.key(DAYS) == "f7683cf81fc8bc2d58fc2d539f9dd909f018528e6cabea420d808191e271b0b0"
    # The loader needs no footer key that existing partitions lack.
    assert set(partition_kv(volume_path(tmp_path, SID, spec, DAYS, DAY.label))) == {
        "bedivere.dataset",
        "bedivere.symbol",
        "bedivere.series",
        "bedivere.session",
        "bedivere.price_basis",
        "bedivere.source",
        "bedivere.bar_definition",
        "bedivere.session_start_ns",
        "bedivere.session_end_ns",
        "bedivere.content_sha256",
    }
    files = sorted(path for path in tmp_path.rglob("*") if path.is_file())
    before = [(path, path.read_bytes(), path.stat().st_mtime_ns) for path in files]
    bars = load_volume_frame(tmp_path, SID, spec, DAYS, feed="databento")
    join(observation(), bars)
    assert sorted(path for path in tmp_path.rglob("*") if path.is_file()) == files
    assert [(path, path.read_bytes(), path.stat().st_mtime_ns) for path in files] == before
    assert set(bars["definition"]) == {spec.key(DAYS)}
