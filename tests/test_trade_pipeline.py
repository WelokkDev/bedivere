"""One synthetic pass through the whole public trade path.

Daily MBO downloads -> prepared trade cache -> trade reader -> volume builder
-> stored-bar loader -> research join -> decision-time context. Failure paths
are tested where they live.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from tests.lake_support import SKIP_REASON

pytest.importorskip("duckdb", reason=SKIP_REASON)
pytest.importorskip("pandas", reason=SKIP_REASON)
dbn = pytest.importorskip("databento_dbn", reason=SKIP_REASON)

import pandas as pd  # noqa: E402  # pyright: ignore[reportMissingTypeStubs]

from bedivere.data.lake.decision_context import (  # noqa: E402
    bar_history,
    forming_bars,
    trade_windows,
)
from bedivere.data.lake.trade_archive import prepare_trade_archive  # noqa: E402
from bedivere.data.lake.trades import TradeTotals, read_trades, summarize_trades  # noqa: E402
from bedivere.data.lake.trades_frame import load_trade_frame  # noqa: E402
from bedivere.data.lake.volume import VolumeSpec  # noqa: E402
from bedivere.data.lake.volume_build import build_volume_archive  # noqa: E402
from bedivere.data.lake.volume_research import align_available, load_volume_frame  # noqa: E402
from tests.test_trades import DAY, DAYS, NEXT, NS, SID, mbo_input  # noqa: E402

# pandas is optional and shipped without complete typing for dynamic columns.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false

SPEC = VolumeSpec(10)
NEAREST = VolumeSpec(10, source="trades-1s", boundary="nearest_second")


def at(second: int) -> int:
    """Mid-second receive time, so each trade's one-second bucket is unambiguous."""
    return second * NS + NS // 2


def downloads(tmp_path: Path) -> Path:
    """Two contiguous UTC days. NQZ5 is instrument 42; the NQH6 (43) trade is not
    selected. `mbo_input` also adds a resting fill and a snapshot trade per day."""
    source = tmp_path / "downloads"
    source.mkdir()
    mbo_input(
        source / "day-1.mbo.dbn",
        100,
        110,
        [
            (at(101), 42, 4, dbn.Side.BID),
            (at(102) - 1, 43, 8, dbn.Side.ASK),
            (at(102), 42, 3, dbn.Side.ASK),
            (at(103), 42, 2, dbn.Side.NONE),
            (at(104), 42, 9, dbn.Side.BID),
            (at(106), 42, 5, dbn.Side.ASK),
        ],
    )
    mbo_input(
        source / "day-2.mbo.dbn",
        110,
        120,
        [(at(111), 42, 12, dbn.Side.BID), (at(113), 42, 1, dbn.Side.NONE)],
    )
    return source


def test_mbo_downloads_reach_an_availability_aware_join(tmp_path: Path) -> None:
    root = tmp_path / "lake"
    cache = prepare_trade_archive(downloads(tmp_path), root / "_meta" / "trade_cache")

    # Reader: no fills or snapshot trades; ordinals count the unselected NQH6 record.
    with read_trades(cache, SID, sessions=DAYS) as read:
        selected = [t for batch in read for t in batch]
    assert read.validated
    assert (read.source.trade_records, read.source.trade_volume) == (8, 44)
    assert "receipt: full-archive and per-input trade counts and volume match" in read.source.checks
    assert [t.ordinal for t in selected] == [0, 2, 3, 4, 5, 6, 7]
    assert [t.session for t in selected] == [DAY.label] * 5 + [NEXT.label] * 2
    assert [t.sequence for t in selected] == [1000, 1002, 1003, 1004, 1005, 1000, 1001]
    assert summarize_trades(cache, SID, sessions=DAYS).totals == TradeTotals(
        records=7, volume=36, buy_records=3, buy_volume=25,
        sell_records=2, sell_volume=8, unknown_records=2, unknown_volume=3,
    )

    # Builder: trade bars plus the nearest-second approximation, then a no-op resume.
    reports = build_volume_archive(
        cache, root, SID, SPEC, DAYS, compare_1s=True, approx_boundary="nearest_second"
    )
    assert {(r["status"], r["source_check"]) for r in reports} == {("written", "full_read")}
    stored = {p: p.stat().st_mtime_ns for p in (root / "event_bars").rglob("bars.parquet")}
    assert len(stored) == 4
    resumed = build_volume_archive(
        cache, root, SID, SPEC, DAYS, compare_1s=True, approx_boundary="nearest_second",
        resume=True,
    )
    assert {(r["status"], r["source_check"]) for r in resumed} == {("unchanged", "fingerprint")}
    assert {p: p.stat().st_mtime_ns for p in (root / "event_bars").rglob("bars.parquet")} == stored

    # Loader: volume conserved per session; nearest-second bars publish after they end.
    trade_bars = load_volume_frame(root, SID, SPEC, DAYS, feed="databento", include_partial=True)
    nearest = load_volume_frame(root, SID, NEAREST, DAYS, feed="databento", include_partial=True)
    volumes: dict[str, int] = {}
    for session, volume in cast(
        list[tuple[str, int]],
        list(trade_bars[["session", "volume"]].itertuples(index=False, name=None)),
    ):
        volumes[session] = volumes.get(session, 0) + volume
    assert volumes == {DAY.label: 23, NEXT.label: 13}
    shape = ["session", "bar_id", "end_ns", "available_ns", "volume", "is_partial"]
    assert list(nearest[shape].itertuples(index=False, name=None)) == [
        (DAY.label, 0, 104 * NS, 105 * NS, 9, False),
        (DAY.label, 1, 105 * NS, 107 * NS, 9, False),
        (DAY.label, 2, 107 * NS, 110 * NS, 5, True),
        (NEXT.label, 0, 112 * NS, 112 * NS, 12, False),
        (NEXT.label, 1, 114 * NS, 120 * NS, 1, True),
    ]

    # Research join. Observation times: 101.5 102.5 103.5 104.5 106.5 | 111.5 113.5.
    frame = load_trade_frame(cache, SID, sessions=DAYS, feed="databento")
    observations = cast(
        pd.DataFrame,
        frame.rename(columns={"ts_recv": "asof_ns"})[["asof_ns", "session", "instrument_id"]],
    )

    def seen(bars: pd.DataFrame, *, exact: bool = False) -> list[int | None]:
        joined = align_available(
            observations,
            bars,
            observation_provenance=frame.attrs["provenance"],
            max_age_ns=60 * NS,
            allow_exact_matches=exact,
        )
        values = cast(list[Any], joined["volume_bar_id"].tolist())
        return [None if pd.isna(v) else int(v) for v in values]

    # A bar's closing trade sees it only with exact matches. 111.5 never sees day one.
    assert seen(trade_bars) == [None, None, None, None, 0, None, 0]
    assert seen(trade_bars, exact=True) == [None, None, None, 0, 0, 0, 0]
    # By end_ns, 104.5 would see nearest bar 0 and 106.5 bar 1; both publish later.
    assert seen(nearest) == [None, None, None, None, 0, None, 0]
    assert seen(nearest, exact=True) == [None, None, None, None, 0, None, 0]

    # Decision-time context at each trade's own receive time.
    declared = frame.attrs["provenance"]
    history = bar_history(
        observations, trade_bars, observation_provenance=declared, depth=2, max_age_ns=60 * NS
    )
    assert history[["observation", "lag", "volume_bar_id", "volume_volume"]].values.tolist() == [
        [4, 0, 0, 18],
        [6, 0, 0, 12],
    ]
    windows = trade_windows(observations, frame, observation_provenance=declared, lookback_ns=2 * NS)
    assert windows[["observation", "trade_ordinal"]].values.tolist() == [
        [1, 0], [2, 0], [2, 2], [3, 2], [3, 3], [4, 4], [6, 6],
    ]
    forming = forming_bars(observations, trade_bars, frame, observation_provenance=declared)
    volumes_forming = cast(list[Any], forming["forming_volume"].tolist())
    # Day one's bar closes on the 9-lot at 104.5 s; day two's first trade closes its own.
    assert [None if pd.isna(v) else int(v) for v in volumes_forming] == [
        None, 4, 7, 9, None, None, None,
    ]
    forming_ids = cast(list[Any], forming["forming_bar_id"].tolist())
    assert [None if pd.isna(v) else int(v) for v in forming_ids] == [
        None, 0, 0, 0, None, None, None,
    ]
