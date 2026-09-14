"""bedivere.data.lake.layout — provenance is the path."""

from __future__ import annotations

from pathlib import Path

import pytest

from bedivere.core.types import Timeframe
from bedivere.data.lake.layout import (
    RollRule,
    SeriesId,
    continuous,
    label_of,
    local_continuous,
    raw_contract,
    series_dir,
    session_path,
    stored_labels,
    to_symbol,
)


def test_series_tokens_by_kind() -> None:
    assert continuous("GLBX.MDP3", "NQ", RollRule.VOLUME, 0).series == "v.0"
    assert continuous("GLBX.MDP3", "ES", RollRule.OPEN_INTEREST, 1).series == "n.1"
    assert continuous("GLBX.MDP3", "NG", RollRule.CALENDAR, 2).series == "c.2"
    assert raw_contract("GLBX.MDP3", "NQ", "NQZ5").series == "raw.NQZ5"
    assert local_continuous("GLBX.MDP3", "NQ", RollRule.VOLUME, 0).series == "local.v.0"


def test_different_series_cannot_share_a_path() -> None:
    """The whole anti-mixing guarantee: separate directories mean no writer can
    overwrite another series and no reader can concatenate two by accident."""
    root = Path("/lake")
    paths = {
        series_dir(root, sid, Timeframe.S1)
        for sid in (
            continuous("GLBX.MDP3", "NQ", RollRule.VOLUME, 0),
            continuous("GLBX.MDP3", "NQ", RollRule.OPEN_INTEREST, 0),
            continuous("GLBX.MDP3", "NQ", RollRule.VOLUME, 1),
            local_continuous("GLBX.MDP3", "NQ", RollRule.VOLUME, 0),
            raw_contract("GLBX.MDP3", "NQ", "NQZ5"),
            continuous("GLBX.MDP3", "MNQ", RollRule.VOLUME, 0),
        )
    }
    assert len(paths) == 6


def test_a_locally_derived_series_is_distinct_from_the_vendors() -> None:
    # Ours ranks over the file we bought, the vendor's over the venue.
    vendor = continuous("GLBX.MDP3", "NQ", RollRule.VOLUME, 0)
    ours = local_continuous("GLBX.MDP3", "NQ", RollRule.VOLUME, 0)
    assert vendor != ours
    assert series_dir(Path("/lake"), vendor, Timeframe.S1) != series_dir(
        Path("/lake"), ours, Timeframe.S1
    )


def test_to_symbol_maps_only_vendor_addressable_series() -> None:
    assert to_symbol(continuous("GLBX.MDP3", "NQ", RollRule.VOLUME, 0)) == "NQ.v.0"
    assert to_symbol(raw_contract("GLBX.MDP3", "NQ", "NQZ5")) == "NQZ5"
    with pytest.raises(ValueError, match="assembled locally"):
        to_symbol(local_continuous("GLBX.MDP3", "NQ", RollRule.VOLUME, 0))


def test_partition_tokens_are_validated() -> None:
    # Tokens become directory names, joined onto the lake root.
    for bad in ("../escape", "a/b", "a\\b", "", " lead"):
        with pytest.raises(ValueError, match="partition tokens"):
            SeriesId(dataset="GLBX.MDP3", symbol=bad, series="v.0")


def test_a_negative_rank_is_refused() -> None:
    with pytest.raises(ValueError, match="rank must be"):
        continuous("GLBX.MDP3", "NQ", RollRule.VOLUME, -1)
    with pytest.raises(ValueError, match="rank must be"):
        local_continuous("GLBX.MDP3", "NQ", RollRule.VOLUME, -1)


def test_session_path_requires_an_iso_label() -> None:
    sid = continuous("GLBX.MDP3", "NQ", RollRule.VOLUME, 0)
    path = session_path(Path("/lake"), sid, Timeframe.S1, "2026-06-15")
    assert path.parts[-2:] == ("session=2026-06-15", "bars.parquet")
    with pytest.raises(ValueError, match="ISO"):
        session_path(Path("/lake"), sid, Timeframe.S1, "15-06-2026")


def test_label_of_round_trips(tmp_path: Path) -> None:
    sid = continuous("GLBX.MDP3", "NQ", RollRule.VOLUME, 0)
    path = session_path(tmp_path, sid, Timeframe.S1, "2026-06-15")
    assert label_of(path.parent) == "2026-06-15"
    with pytest.raises(ValueError, match="not a session partition"):
        label_of(tmp_path)


def test_stored_labels_lists_only_written_days(tmp_path: Path) -> None:
    sid = continuous("GLBX.MDP3", "NQ", RollRule.VOLUME, 0)
    assert stored_labels(tmp_path, sid, Timeframe.S1) == []

    for label in ("2026-06-16", "2026-06-15"):
        path = session_path(tmp_path, sid, Timeframe.S1, label)
        path.parent.mkdir(parents=True)
        path.write_bytes(b"")
    # A partition directory with no bars.parquet in it is not a stored day.
    (series_dir(tmp_path, sid, Timeframe.S1) / "session=2026-06-17").mkdir()

    assert stored_labels(tmp_path, sid, Timeframe.S1) == ["2026-06-15", "2026-06-16"]
