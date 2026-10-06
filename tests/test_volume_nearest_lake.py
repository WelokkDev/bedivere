"""Nearest-second policy integration, persisted causality, and CLI selection."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from tests.lake_support import SKIP_REASON

pytest.importorskip("duckdb", reason=SKIP_REASON)
pytest.importorskip("pandas", reason=SKIP_REASON)
pytest.importorskip("databento_dbn", reason=SKIP_REASON)

import pandas as pd  # noqa: E402  # pyright: ignore[reportMissingTypeStubs]

from bedivere.cli import data as data_cli  # noqa: E402
from bedivere.core.types import Timeframe  # noqa: E402
from bedivere.data.lake.schema import BarBatch  # noqa: E402
from bedivere.data.lake.volume import NS, VolumeInput, VolumeSpec  # noqa: E402
from bedivere.data.lake.volume_build import (  # noqa: E402
    build_volume_archive,
    derive_volume,
    one_second_inputs,
)
from bedivere.data.lake.volume_research import (  # noqa: E402
    align_available,
    load_volume_frame,
    volume_coverage,
)
from bedivere.data.lake.volume_store import (  # noqa: E402
    checked_volume_partition,
    validate_bars,
    volume_path,
)
from bedivere.data.lake.writer import write_day  # noqa: E402
from tests.test_volume import DAY, DAYS  # noqa: E402
from tests.test_volume_lake import SID, archive, seed_time_bars, whole  # noqa: E402
from tests.test_volume_research import OBSERVED  # noqa: E402


def test_policy_roundtrip_and_causal_alignment(tmp_path: Path) -> None:
    trades = [
        VolumeInput.trade(100 * NS + 1, 10, 4500, 42),
        VolumeInput.trade(104 * NS + 1, 30, 1000, 42),
    ]
    path = archive(tmp_path / "trades.dbn", trades)
    spec = VolumeSpec(5000)
    reports = build_volume_archive(
        path, tmp_path, SID, spec, DAYS, compare_1s=True, approx_boundary="nearest_second"
    )
    nearest = replace(spec, source="trades-1s", boundary="nearest_second")
    assert (reports[1]["source"], reports[1]["boundary"]) == ("trades-1s", "nearest_second")
    bars = checked_volume_partition(tmp_path, SID, nearest, DAYS, DAY)
    assert bars[0].volume == 4500 and bars[0].available_ns == 105 * NS
    assert bars[0].end_ns == 101 * NS and not bars[0].is_partial
    assert volume_coverage(tmp_path, SID, nearest, DAYS)[0]["status"] == "valid"
    frame = load_volume_frame(tmp_path, SID, nearest, DAYS, feed="databento")
    obs = pd.DataFrame(
        {
            "asof_ns": [101 * NS + 1, 105 * NS - 1, 105 * NS, 105 * NS + 1],
            "session": [DAY.label] * 4,
            "instrument_id": [42] * 4,
        }
    )
    # The boundary at 101 s was only selected at 105 s: decisions in between must not see it.
    joined = cast(
        Any, align_available(obs, frame, observation_provenance=OBSERVED, max_age_ns=10 * NS)
    )
    assert joined["volume_bar_id"].isna().tolist() == [True, True, True, False]
    assert joined.iloc[-1]["volume_close"] == 10
    assert joined.iloc[-1]["volume_end_ns"] == 101 * NS
    assert joined.iloc[-1]["volume_available_ns"] == 105 * NS
    inclusive = cast(
        Any,
        align_available(
            obs,
            frame,
            observation_provenance=OBSERVED,
            max_age_ns=10 * NS,
            allow_exact_matches=True,
        ),
    )
    assert inclusive["volume_bar_id"].isna().tolist() == [True, True, False, False]
    # Distinct policy paths, not a migration or replacement of the old dataset.
    old_spec = replace(nearest, boundary="whole_trade")
    build_volume_archive(path, tmp_path, SID, spec, DAYS, compare_1s=True)
    assert volume_path(tmp_path, SID, nearest, DAYS, DAY.label) != volume_path(
        tmp_path, SID, old_spec, DAYS, DAY.label
    )
    assert checked_volume_partition(tmp_path, SID, old_spec, DAYS, DAY)[0].volume == 5500
    again = build_volume_archive(
        path,
        tmp_path,
        SID,
        spec,
        DAYS,
        compare_1s=True,
        approx_boundary="nearest_second",
        resume=True,
    )
    assert all(report["status"] == "unchanged" for report in again)


@pytest.mark.parametrize(
    "volumes,threshold", [([4500, 1000], 5000), ([1750, 1050], 1000), ([900, 3000], 1000)]
)
def test_same_trade_seconds_and_stored_seconds_agree(
    tmp_path: Path,
    volumes: list[int],
    threshold: int,
) -> None:
    trades = [
        VolumeInput.trade((100 + i) * NS + 1, float(i + 10), volume, 42)
        for i, volume in enumerate(volumes)
    ]
    path = archive(tmp_path / "trades.dbn", trades)
    spec = VolumeSpec(threshold, "ohlcv-1s", "nearest_second")
    rebuilt = replace(spec, source="trades-1s")
    build_volume_archive(
        path,
        tmp_path,
        SID,
        VolumeSpec(threshold),
        DAYS,
        compare_1s=True,
        approx_boundary="nearest_second",
    )
    expected = checked_volume_partition(tmp_path, SID, rebuilt, DAYS, DAY)
    assert not volume_path(tmp_path, SID, spec, DAYS, DAY.label).exists()
    batch = BarBatch()
    for row in one_second_inputs(trades):
        batch.append(
            row.end_ns // NS,
            row.open,
            row.high,
            row.low,
            row.close,
            float(row.volume),
            row.instrument_id,
        )
    write_day(tmp_path, SID, Timeframe.S1, DAY.label, batch, "fixture", whole(DAY))
    derive_volume(tmp_path, SID, spec, DAYS)
    # Equal bars, stored as two datasets: neither build wrote over the other.
    assert checked_volume_partition(tmp_path, SID, spec, DAYS, DAY) == expected
    assert checked_volume_partition(tmp_path, SID, rebuilt, DAYS, DAY) == expected
    assert derive_volume(tmp_path, SID, spec, DAYS, resume=True)[0]["status"] == "unchanged"
    validate_bars(expected, spec, DAY)
    # Earlier boundaries are allowed, but fractional-second publication is not.
    broken = [replace(expected[0], available_ns=expected[0].available_ns + 1), *expected[1:]]
    with pytest.raises(ValueError, match="whole seconds"):
        validate_bars(broken, spec, DAY)


def test_cli_exposes_policy_without_changing_defaults(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = archive(tmp_path / "trades.dbn")
    session = tmp_path / "sessions.json"
    session.write_text(
        json.dumps(
            {
                "template": DAYS.template,
                "timezone": DAYS.timezone,
                "days": [
                    {"label": DAY.label, "startUnix": DAY.start_unix, "endUnix": DAY.end_unix}
                ],
            }
        )
    )
    args = [
        "volume",
        "--root",
        str(tmp_path),
        "--session",
        str(session),
        "--symbol",
        "NQ",
        "--series",
        SID.series,
        "--threshold",
        "1000",
        "--archive",
        str(path),
    ]
    assert data_cli.main([*args, "--compare-1s", "--approx-boundary", "nearest_second"]) == 0
    reports = json.loads(capsys.readouterr().out)
    assert reports[0]["boundary"] == "whole_trade" and reports[1]["boundary"] == "nearest_second"
    assert data_cli.main([*args, "--approx-boundary", "nearest_second"]) == 1
    assert "requires --compare-1s" in capsys.readouterr().err
    assert data_cli.main([*args, "--boundary", "nearest_second"]) == 1
    assert "requires source=ohlcv-1s" in capsys.readouterr().err
    seed_time_bars(tmp_path)
    assert data_cli.main([*args[:-2], "--from", "1s", "--boundary", "nearest_second"]) == 0
    derived = json.loads(capsys.readouterr().out)
    assert len(derived) == 1 and derived[0]["boundary"] == "nearest_second"
