"""Decision-time context: bar history, causal trade windows, and the forming bar."""

from __future__ import annotations

import re
from collections.abc import Callable
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
from bedivere.core.types import Timeframe  # noqa: E402
from bedivere.data.lake.decision_context import (  # noqa: E402
    FORMING_COLUMNS,
    bar_history,
    forming_bars,
    trade_windows,
)
from bedivere.data.lake.layout import session_path  # noqa: E402
from bedivere.data.lake.provenance import ProvenanceError  # noqa: E402
from bedivere.data.lake.read import read_partition  # noqa: E402
from bedivere.data.lake.research_join import definition_key  # noqa: E402
from bedivere.data.lake.trades import TradeIssue  # noqa: E402
from bedivere.data.lake.trades_frame import TRADE_COLUMNS, load_trade_frame  # noqa: E402
from bedivere.data.lake.volume import (  # noqa: E402
    NS,
    FormingBar,
    VolumeAccumulator,
    VolumeInput,
    VolumeSpec,
)
from bedivere.data.lake.volume_build import build_volume_archive, derive_volume  # noqa: E402
from bedivere.data.lake.volume_research import align_available, load_volume_frame  # noqa: E402
from bedivere.data.lake.writer import write_day  # noqa: E402
from tests.test_volume import DAY, DAYS, TRADES  # noqa: E402
from tests.test_volume_lake import SID, archive, seed_time_bars  # noqa: E402
from tests.test_volume_research import OBSERVED, observed  # noqa: E402

# pandas is optional and shipped without complete typing for dynamic columns.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false

# The fixture's trades: 400@10 at 100.1, 800@12 at 100.2, 400@8 at 100.3,
# 600@11 at 101.1, 250@9 at 101.3 (all in microseconds past the second).
T = [t.start_ns for t in TRADES]
FIRST = T[1]  # whole-trade Q=1000: bar 0 (1200) closes on the second trade
SECOND = T[3]  # bar 1 (1000) closes on the fourth; 250 remain at the close
NEXT = SessionDay("2026-06-16", DAY.end_unix, DAY.end_unix + 10)
HOUR = 3600 * NS


def seeded(root: Path, spec: VolumeSpec | None = None) -> tuple[VolumeSpec, pd.DataFrame, pd.DataFrame]:
    spec = spec or VolumeSpec(1000)
    path = archive(root / "trades.dbn")
    build_volume_archive(path, root, SID, spec, DAYS)
    bars = load_volume_frame(root, SID, spec, DAYS, feed="databento")
    trades = load_trade_frame(path, SID, sessions=DAYS, feed="databento")
    return spec, bars, trades


def decisions(*asof: int, **columns: object) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asof_ns": list(asof),
            "session": [DAY.label] * len(asof),
            "instrument_id": [42] * len(asof),
            **{name: list(cast(Any, values)) for name, values in columns.items()},
        }
    )


def ids(frame: pd.DataFrame, column: str) -> list[int | None]:
    values = cast(list[Any], frame[column].tolist())
    return [None if pd.isna(value) else int(value) for value in values]


# ---------- bar history ----------


def test_history_counts_back_from_the_bar_align_available_attaches(tmp_path: Path) -> None:
    _, bars, _ = seeded(tmp_path)
    obs = decisions(FIRST - 1, FIRST, FIRST + 1, SECOND + 1, DAY.end_unix * NS, private=[5, 4, 3, 2, 1])
    history = bar_history(obs, bars, observation_provenance=OBSERVED, depth=5, max_age_ns=HOUR)
    assert list(history.columns[:5]) == ["observation", "asof_ns", "session", "instrument_id", "lag"]
    assert "private" not in history.columns
    assert history[["observation", "lag", "volume_bar_id"]].values.tolist() == [
        [2, 0, 0],
        [3, 0, 1],
        [3, 1, 0],
        [4, 0, 1],
        [4, 1, 0],
    ]
    # Lag 0 is exactly what the single-bar join attaches, exact matches included.
    for exact in (False, True):
        single = align_available(
            obs, bars, observation_provenance=OBSERVED, max_age_ns=HOUR, allow_exact_matches=exact
        )
        deep = bar_history(
            obs,
            bars,
            observation_provenance=OBSERVED,
            depth=5,
            max_age_ns=HOUR,
            allow_exact_matches=exact,
        )
        latest = cast(Any, deep[deep["lag"] == 0].set_index("observation"))["volume_bar_id"]
        assert [latest.get(i) for i in range(len(obs))] == ids(single, "volume_bar_id")
    assert str(history["volume_available_ns"].dtype) == "int64"
    assert history.attrs["provenance"] == OBSERVED
    assert history.attrs["bar_provenance"] is bars.attrs["provenance"]
    assert history.attrs["bar_definition"]["threshold"] == 1000


def test_history_depth_and_age_bound_every_bar_and_never_cross_sessions(tmp_path: Path) -> None:
    _, bars, _ = seeded(tmp_path)
    obs = decisions(SECOND + 1)
    shallow = bar_history(obs, bars, observation_provenance=OBSERVED, depth=1, max_age_ns=HOUR)
    assert ids(shallow, "volume_bar_id") == [1]
    # Bar 0 became available almost a second before bar 1: a 10 ns limit keeps bar 1 only.
    recent = bar_history(obs, bars, observation_provenance=OBSERVED, depth=5, max_age_ns=10)
    assert ids(recent, "volume_bar_id") == [1]
    stale = bar_history(decisions(SECOND + 11), bars, observation_provenance=OBSERVED, depth=5, max_age_ns=10)
    assert stale.empty and str(stale["volume_available_ns"].dtype) == "int64"
    other = pd.DataFrame(
        {"asof_ns": [NEXT.start_unix * NS + 1], "session": [NEXT.label], "instrument_id": [42]}
    )
    context = observed(replace(DAYS, days=(DAY, NEXT)))
    assert bar_history(other, bars, observation_provenance=context, depth=5, max_age_ns=HOUR).empty
    wrong_contract = decisions(SECOND + 1).assign(instrument_id=43)
    assert bar_history(wrong_contract, bars, observation_provenance=OBSERVED, depth=5, max_age_ns=HOUR).empty


def test_history_orders_same_timestamp_split_bars_by_id(tmp_path: Path) -> None:
    _, bars, _ = seeded(tmp_path, VolumeSpec(100, boundary="split_trade"))
    # 400 contracts at the first timestamp are bars 0-3; 800 at the second are 4-11.
    history = bar_history(decisions(T[1] + 1), bars, observation_provenance=OBSERVED, depth=3, max_age_ns=HOUR)
    assert history[["lag", "volume_bar_id"]].values.tolist() == [[0, 11], [1, 10], [2, 9]]
    assert set(history["volume_available_ns"]) == {T[1]}


def test_history_over_an_availability_window_is_refused_where_it_would_reach_outside(
    tmp_path: Path,
) -> None:
    spec, whole, _ = seeded(tmp_path, VolumeSpec(100, boundary="split_trade"))

    def history(bars: pd.DataFrame, asof: int, age: int, **options: Any) -> list[int | None]:
        found = bar_history(
            decisions(asof), bars, observation_provenance=OBSERVED, depth=10, max_age_ns=age, **options
        )
        return ids(found, "volume_bar_id")

    # The loader's window trims the oldest bars: the frame begins at bar 4.
    trimmed = load_volume_frame(tmp_path, SID, spec, DAYS, feed="databento", start_ns=T[1])
    assert int(cast(Any, trimmed["bar_id"]).min()) == 4
    assert (whole.attrs["availability_window"], trimmed.attrs["availability_window"]) == (
        (None, None),
        (T[1], None),
    )
    # An hour back would reach bars 0-3, which it no longer holds: refused, not cut short.
    assert len(history(whole, T[1] + 1, HOUR)) == 10
    with pytest.raises(ValueError, match="look outside the availability window"):
        history(trimmed, T[1] + 1, HOUR)
    # A lookback inside the window is answered as the whole frame answers it.
    assert history(trimmed, T[1] + 1, 1) == history(whole, T[1] + 1, 1) == list(range(11, 3, -1))
    # Trimmed at the other end: bar 21 exists by T[3] + 1, and the frame stops at 15.
    early = load_volume_frame(tmp_path, SID, spec, DAYS, feed="databento", end_ns=T[3])
    assert history(early, T[3], 1 * NS)[0] == history(whole, T[3], 1 * NS)[0] == 15
    assert history(whole, T[3] + 1, 1 * NS)[0] == 21
    with pytest.raises(ValueError, match="look outside the availability window"):
        history(early, T[3] + 1, 1 * NS)
    # A bar available exactly at the bound is one the frame does not hold.
    with pytest.raises(ValueError, match="look outside the availability window"):
        history(early, T[3], 1 * NS, allow_exact_matches=True)
    # The forming bar is replayed from the open, so it needs the session from there.
    trades = load_trade_frame(tmp_path / "trades.dbn", SID, sessions=DAYS, feed="databento")
    with pytest.raises(ValueError, match="look outside the availability window"):
        forming_bars(decisions(T[1] + 1), trimmed, trades, observation_provenance=OBSERVED)
    assert ids(forming_bars(decisions(T[3]), early, trades, observation_provenance=OBSERVED), "forming_bar_id") == [None]
    # Without the loader's record of its window nothing tells trimmed from whole.
    stripped = early.copy()
    del stripped.attrs["availability_window"]
    with pytest.raises(ValueError, match="availability_window.* must be the"):
        history(stripped, T[3], 1 * NS)


def test_history_refuses_holes_reserved_names_and_bad_bounds(tmp_path: Path) -> None:
    _, bars, _ = seeded(tmp_path, VolumeSpec(100, boundary="split_trade"))
    with pytest.raises(ValueError, match="contiguous run of bar_id"):
        bar_history(decisions(SECOND + 1), bars.drop(index=5), observation_provenance=OBSERVED, depth=2, max_age_ns=HOUR)
    with pytest.raises(ValueError, match="reserved column name"):
        bar_history(decisions(SECOND + 1, lag=[0]), bars, observation_provenance=OBSERVED, depth=2, max_age_ns=HOUR)
    with pytest.raises(ValueError, match="reserved column name"):
        bar_history(decisions(SECOND + 1), bars, observation_provenance=OBSERVED, depth=2, max_age_ns=HOUR, prefix="")
    for depth in (0, 2**63, cast(Any, 1.0)):
        with pytest.raises(ValueError, match="depth must be a positive int64"):
            bar_history(decisions(SECOND + 1), bars, observation_provenance=OBSERVED, depth=depth, max_age_ns=HOUR)
    with pytest.raises(ValueError, match="max_age_ns must be an explicit"):
        bar_history(decisions(SECOND + 1), bars, observation_provenance=OBSERVED, depth=1, max_age_ns=cast(Any, 1.5))
    with pytest.raises(ProvenanceError, match=r"feed: observations 'nt8' \(attested\)"):
        bar_history(decisions(SECOND + 1), bars, observation_provenance=replace(OBSERVED, feed="nt8"), depth=1, max_age_ns=HOUR)


def test_positions_follow_input_order_whatever_the_index_sorting_or_duplicates(tmp_path: Path) -> None:
    _, bars, trades = seeded(tmp_path)
    close = DAY.end_unix * NS
    obs = decisions(SECOND + 1, FIRST + 1, close, FIRST + 1, T[2] + 50, T[0] + 50, tag=list("abcdef"))
    # A private column the database could not even read: it never gets to.
    obs["month"] = pd.period_range("2026-01", periods=6, freq="M")
    obs = obs.set_index(pd.Index([10, 20, 30, 40, 50, 60]))
    columns = cast(list[str], list(obs.columns))
    history = bar_history(obs, bars, observation_provenance=OBSERVED, depth=5, max_age_ns=HOUR)
    assert history["observation"].tolist() == [0, 0, 1, 2, 2, 3, 4]
    assert history["volume_bar_id"].tolist() == [1, 0, 0, 1, 0, 0, 0]
    windows = trade_windows(obs, trades, observation_provenance=OBSERVED, lookback_ns=150)
    assert windows[["observation", "trade_ordinal"]].values.tolist() == [
        [0, 3], [1, 0], [1, 1], [3, 0], [3, 1], [4, 1], [4, 2], [5, 0],
    ]
    forming = forming_bars(obs, bars, trades, observation_provenance=OBSERVED)
    assert ids(forming, "forming_bar_id") == [None, None, 2, None, 1, 0]
    # The observations themselves come back: their columns, dtypes and index.
    pd.testing.assert_frame_equal(forming[columns], obs)
    # The input frame is untouched: no position column, same index.
    assert list(obs.columns) == columns and obs.index.tolist() == [10, 20, 30, 40, 50, 60]


def test_second_session_in_the_same_call_and_custom_prefixes(tmp_path: Path) -> None:
    _, bars, trades = seeded(tmp_path)
    context = observed(replace(DAYS, days=(DAY, NEXT)))
    obs = pd.DataFrame(
        {
            "asof_ns": [SECOND + 1, NEXT.start_unix * NS + 1, T[2] + 50],
            "session": [DAY.label, NEXT.label, DAY.label],
            "instrument_id": [42, 42, 42],
        }
    )
    history = bar_history(obs, bars, observation_provenance=context, depth=5, max_age_ns=HOUR, prefix="bar_")
    assert history[["observation", "bar_bar_id"]].values.tolist() == [[0, 1], [0, 0], [2, 0]]
    # The trades were loaded for the first session only.
    with pytest.raises(ValueError, match=re.escape("session(s) ['2026-06-16'] cannot be answered")):
        trade_windows(obs, trades, observation_provenance=context, lookback_ns=NS)
    loaded = cast(pd.DataFrame, obs.iloc[[0, 2]])
    windows = trade_windows(loaded, trades, observation_provenance=context, lookback_ns=NS, prefix="t_")
    assert windows[["observation", "t_ordinal"]].values.tolist() == [
        [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2],
    ]
    # A session the bars frame holds no partition for cannot be checked.
    with pytest.raises(ValueError, match=re.escape("session(s) ['2026-06-16'] cannot be checked")):
        forming_bars(obs, bars, trades, observation_provenance=context)
    forming = forming_bars(loaded, bars, trades, observation_provenance=context, prefix="live_")
    assert ids(forming, "live_bar_id") == [None, 1]


def test_a_session_the_trades_were_not_loaded_for_is_refused_but_a_quiet_one_is_an_answer(
    tmp_path: Path,
) -> None:
    quiet = SessionDay("2026-06-17", NEXT.end_unix, NEXT.end_unix + 10)
    two = replace(DAYS, days=(DAY, NEXT))
    opened = NEXT.start_unix * NS
    later = [VolumeInput.trade(opened + 100, 10.0, 5, 42), VolumeInput.trade(opened + 200, 11.0, 3, 42)]
    path = archive(tmp_path / "trades.dbn", [*TRADES, *later], end=quiet.end_unix * NS)
    spec = VolumeSpec(1000)
    build_volume_archive(path, tmp_path, SID, spec, two)
    bars = load_volume_frame(tmp_path, SID, spec, two, feed="databento")
    context = observed(replace(DAYS, days=(DAY, NEXT, quiet)))
    decision = pd.DataFrame(
        {"asof_ns": [opened + 300], "session": [NEXT.label], "instrument_id": [42]}
    )
    both = load_trade_frame(path, SID, sessions=two, feed="databento")
    forming = forming_bars(decision, bars, both, observation_provenance=context)
    assert ids(forming, "forming_volume") == [8]
    window = trade_windows(decision, both, observation_provenance=context, lookback_ns=NS)
    assert window["trade_size"].tolist() == [5, 3]
    # Read for the first session alone, the second is absent, not empty.
    first = load_trade_frame(path, SID, sessions=DAYS, feed="databento")
    for answer in (
        lambda: forming_bars(decision, bars, first, observation_provenance=context),
        lambda: trade_windows(decision, first, observation_provenance=context, lookback_ns=NS),
    ):
        with pytest.raises(ValueError, match=re.escape("was loaded for 1 session(s) (2026-06-15)")):
            answer()
    # A session the frame was loaded for, in which nobody traded, has nothing in it.
    all_three = load_trade_frame(
        path, SID, sessions=replace(DAYS, days=(DAY, NEXT, quiet)), feed="databento"
    )
    silence = pd.DataFrame(
        {"asof_ns": [quiet.start_unix * NS + 5], "session": [quiet.label], "instrument_id": [42]}
    )
    assert trade_windows(silence, all_three, observation_provenance=context, lookback_ns=NS).empty
    with pytest.raises(ValueError, match=re.escape("session(s) ['2026-06-17'] cannot be answered")):
        trade_windows(silence, both, observation_provenance=context, lookback_ns=NS)


# ---------- trade windows ----------


def test_trade_windows_are_half_open_causal_and_in_archive_order(tmp_path: Path) -> None:
    _, _, trades = seeded(tmp_path)
    obs = decisions(T[2], T[4], private=["a", "b"])
    window = trade_windows(obs, trades, observation_provenance=OBSERVED, lookback_ns=150)
    # [T[2] - 150, T[2]) holds the second trade only; the window at T[4] is empty.
    assert window[["observation", "trade_ordinal"]].values.tolist() == [[0, 1]]
    assert "private" not in window.columns
    assert set(f"trade_{name}" for name in TRADE_COLUMNS) <= set(window.columns)
    # Attached trade columns keep the loader's dtypes, nullability included.
    assert {name: str(window[f"trade_{name}"].dtype) for name in TRADE_COLUMNS} == TRADE_COLUMNS
    inclusive = trade_windows(
        obs, trades, observation_provenance=OBSERVED, lookback_ns=150, allow_exact_matches=True
    )
    assert inclusive[["observation", "trade_ordinal"]].values.tolist() == [[0, 1], [0, 2], [1, 4]]
    wide = trade_windows(decisions(T[4]), trades, observation_provenance=OBSERVED, lookback_ns=2 * NS)
    assert wide["trade_ordinal"].tolist() == [0, 1, 2, 3]
    assert wide["trade_size"].tolist() == [400, 800, 400, 600]
    assert wide.attrs["trade_source"] is trades.attrs["trade_source"]
    assert wide.attrs["trade_provenance"] is trades.attrs["provenance"]
    # Nullable integer columns keep values float64 cannot hold, NULLs included.
    large = trades.copy()
    large["ts_event"] = pd.array([2**62 + 1, None, 2**62 + 3, 2**62 + 5, 2**62 + 7], dtype="Int64")
    events = trade_windows(decisions(T[4]), large, observation_provenance=OBSERVED, lookback_ns=2 * NS)
    assert str(events["trade_ts_event"].dtype) == "Int64"
    assert ids(events, "trade_ts_event") == [2**62 + 1, None, 2**62 + 3, 2**62 + 5]


def test_trade_windows_start_column_per_observation_and_null_means_nothing(tmp_path: Path) -> None:
    _, _, trades = seeded(tmp_path)
    obs = decisions(T[3] + 100, T[3] + 100)
    obs["from_ns"] = pd.array([T[2] - 50, None], dtype="Int64")
    window = trade_windows(obs, trades, observation_provenance=OBSERVED, start_column="from_ns")
    assert window[["observation", "trade_ordinal"]].values.tolist() == [[0, 2], [0, 3]]
    late = decisions(T[2], from_ns=[T[2] + 1])
    with pytest.raises(ValueError, match="window start after its decision time"):
        trade_windows(late, trades, observation_provenance=OBSERVED, start_column="from_ns")
    floats = decisions(T[2], from_ns=[float(T[1])])
    with pytest.raises(ValueError, match="integer dtype holding int64"):
        trade_windows(floats, trades, observation_provenance=OBSERVED, start_column="from_ns")
    with pytest.raises(ValueError, match="exactly one of lookback_ns"):
        trade_windows(late, trades, observation_provenance=OBSERVED)
    with pytest.raises(ValueError, match="exactly one of lookback_ns"):
        trade_windows(late, trades, observation_provenance=OBSERVED, lookback_ns=1, start_column="from_ns")
    with pytest.raises(ValueError, match="no column 'missing'"):
        trade_windows(late, trades, observation_provenance=OBSERVED, start_column="missing")


def test_trade_windows_must_lie_inside_a_bounded_frames_receive_window(tmp_path: Path) -> None:
    seeded(tmp_path)
    bounded = load_trade_frame(
        tmp_path / "trades.dbn", SID, sessions=DAYS, start_ns=T[1], end_ns=T[4], feed="databento"
    )
    assert bounded.attrs["receive_window"] == (T[1], T[4])
    inside = trade_windows(decisions(T[3]), bounded, observation_provenance=OBSERVED, lookback_ns=NS - 150)
    assert inside["trade_ordinal"].tolist() == [2]
    with pytest.raises(ValueError, match="reach outside the trade frame's receive-time bounds"):
        trade_windows(decisions(T[3]), bounded, observation_provenance=OBSERVED, lookback_ns=NS)
    # The frame holds ts_recv < T[4], so only an exact match at T[4] is uncovered.
    assert trade_windows(decisions(T[4]), bounded, observation_provenance=OBSERVED, lookback_ns=100).empty
    with pytest.raises(ValueError, match="reach outside"):
        trade_windows(decisions(T[4]), bounded, observation_provenance=OBSERVED, lookback_ns=100, allow_exact_matches=True)
    broken = bounded.copy()
    broken.attrs["receive_window"] = [T[1], T[4]]
    with pytest.raises(ValueError, match="receive_window"):
        trade_windows(decisions(T[3]), broken, observation_provenance=OBSERVED, lookback_ns=100)


def test_trade_windows_need_the_loaders_provenance_and_the_same_identity(tmp_path: Path) -> None:
    _, _, trades = seeded(tmp_path)
    unlabelled = load_trade_frame(tmp_path / "trades.dbn", SID, sessions=DAYS)
    with pytest.raises(ProvenanceError, match="trades carry no loader provenance"):
        trade_windows(decisions(T[2]), unlabelled, observation_provenance=OBSERVED, lookback_ns=NS)
    with pytest.raises(ProvenanceError, match=r"feed: observations 'nt8' \(attested\) vs trades 'databento' \(attested\)"):
        trade_windows(decisions(T[2]), trades, observation_provenance=replace(OBSERVED, feed="nt8"), lookback_ns=NS)
    with pytest.raises(ProvenanceError, match="lies outside its declared session"):
        trade_windows(decisions(DAY.end_unix * NS + 1), trades, observation_provenance=OBSERVED, lookback_ns=NS)
    moved = trades.copy()
    moved.loc[0, "ts_recv"] = DAY.start_unix * NS - 1
    with pytest.raises(ProvenanceError, match=re.escape("trade 0 of session '2026-06-15' has ts_recv=")):
        trade_windows(decisions(T[2]), moved, observation_provenance=OBSERVED, lookback_ns=NS)


# ---------- forming bars ----------


def test_forming_bar_is_the_replayed_state_before_each_decision(tmp_path: Path) -> None:
    _, bars, trades = seeded(tmp_path)
    close = DAY.end_unix * NS
    times = [T[0] + 50, FIRST, FIRST + 50, T[2] + 50, SECOND + 1, T[4] + 1, close]
    obs = decisions(*times, private=list(range(7)))
    forming = forming_bars(obs, bars, trades, observation_provenance=OBSERVED)
    assert forming["private"].tolist() == list(range(7))
    assert ids(forming, "forming_bar_id") == [0, 0, None, 1, None, 2, 2]
    assert ids(forming, "forming_volume") == [400, 400, None, 400, None, 250, 250]
    assert ids(forming, "forming_start_ns") == [T[0], T[0], None, T[2], None, T[4], T[4]]
    assert ids(forming, "forming_end_ns") == ids(forming, "forming_start_ns")
    assert ids(forming, "forming_input_count") == [1, 1, None, 1, None, 1, 1]
    closes = cast(list[Any], forming["forming_close"].tolist())
    assert [None if pd.isna(v) else v for v in closes] == [10, 10, None, 8, None, 9, 9]
    assert forming["forming_vwap"].tolist()[0] == 10.0
    assert str(forming["forming_volume"].dtype) == "Int64"
    assert str(forming["forming_open"].dtype) == "float64"
    # With exact matches, nothing is forming at FIRST (bar closed) or at session end.
    inclusive = forming_bars(obs, bars, trades, observation_provenance=OBSERVED, allow_exact_matches=True)
    assert ids(inclusive, "forming_bar_id") == [0, None, None, 1, None, 2, None]
    # The forming bar's id is always the next one after the history's lag 0.
    history = bar_history(obs, bars, observation_provenance=OBSERVED, depth=1, max_age_ns=HOUR)
    latest = cast(Any, history.set_index("observation"))["volume_bar_id"]
    for position, bar_id in enumerate(ids(forming, "forming_bar_id")):
        if bar_id is not None:
            assert bar_id == latest.get(position, -1) + 1
    assert forming.attrs["provenance"] == OBSERVED
    assert forming.attrs["bar_provenance"] is bars.attrs["provenance"]
    assert forming.attrs["trade_provenance"] is trades.attrs["provenance"]
    assert forming.attrs["trade_source"] is trades.attrs["trade_source"]
    assert forming.attrs["bar_definition"]["boundary"] == "whole_trade"


def test_forming_split_bar_begins_with_the_boundary_remainder(tmp_path: Path) -> None:
    _, bars, trades = seeded(tmp_path, VolumeSpec(1000, boundary="split_trade"))
    forming = forming_bars(decisions(FIRST + 50), bars, trades, observation_provenance=OBSERVED)
    row = cast(Any, forming.iloc[0])
    assert (int(row["forming_bar_id"]), int(row["forming_volume"])) == (1, 200)
    assert (int(row["forming_start_ns"]), int(row["forming_end_ns"])) == (FIRST, FIRST)
    assert (row["forming_open"], row["forming_close"], row["forming_vwap"]) == (12, 12, 12)
    assert int(row["forming_input_count"]) == 1
    # The accumulator's own view agrees, and emits nothing.
    builder = VolumeAccumulator(VolumeSpec(1000, boundary="split_trade"), DAY)
    builder.push(TRADES[0])
    assert len(builder.push(TRADES[1])) == 1
    assert builder.forming() == FormingBar(1, FIRST, FIRST, 12, 12, 12, 12, 200, 42, 1, 12.0)
    builder.finish()
    assert builder.forming() is None


def test_forming_replays_only_the_archive_the_bars_were_built_from(tmp_path: Path) -> None:
    spec, bars, trades = seeded(tmp_path)
    at_end = decisions(T[4] + 100)
    assert ids(forming_bars(at_end, bars, trades, observation_provenance=OBSERVED), "forming_volume") == [250]
    # Another archive with one more trade: every stored bar replays, only the forming one differs.
    extra = archive(tmp_path / "extra.dbn", [*TRADES, VolumeInput.trade(T[4] + 50, 9.0, 50, 42)])
    other = load_trade_frame(extra, SID, sessions=DAYS, feed="databento")
    with_partial = load_volume_frame(tmp_path, SID, spec, DAYS, feed="databento", include_partial=True)
    for stored in (bars, with_partial):
        with pytest.raises(ValueError, match="extra.dbn .*is not what these bars were built from"):
            forming_bars(at_end, stored, other, observation_provenance=OBSERVED)
    # Refused before any replay, so also where nothing yet contradicts it.
    with pytest.raises(ValueError, match="is not what these bars were built from"):
        forming_bars(decisions(T[0] + 1), bars, other, observation_provenance=OBSERVED)
    # The archive's bytes are its identity, not its name.
    renamed = tmp_path / "renamed.dbn"
    renamed.write_bytes((tmp_path / "trades.dbn").read_bytes())
    same = load_trade_frame(renamed, SID, sessions=DAYS, feed="databento")
    assert ids(forming_bars(at_end, bars, same, observation_provenance=OBSERVED), "forming_volume") == [250]
    # Each frame must say what it was loaded from.
    for frame, attr in ((bars, "bar_source"), (trades, "trade_source")):
        anonymous = frame.copy()
        del anonymous.attrs[attr]
        pair = (anonymous, trades) if attr == "bar_source" else (bars, anonymous)
        with pytest.raises(ValueError, match="need what each frame was loaded from"):
            forming_bars(at_end, *pair, observation_provenance=OBSERVED)


def test_forming_ties_a_continuous_series_to_its_contract_selection_too(tmp_path: Path) -> None:
    sid = replace(SID, series="local.v.0")
    seed_time_bars(tmp_path, sid)
    path = archive(tmp_path / "trades.dbn")
    spec = VolumeSpec(1000)
    build_volume_archive(path, tmp_path, sid, spec, DAYS)
    bars = load_volume_frame(tmp_path, sid, spec, DAYS, feed="databento")
    context = replace(OBSERVED, series=sid.series)

    def forming() -> list[int | None]:
        trades = load_trade_frame(path, sid, sessions=DAYS, lake_root=tmp_path, feed="databento")
        return ids(
            forming_bars(decisions(T[4] + 100), bars, trades, observation_provenance=context),
            "forming_volume",
        )

    assert forming() == [250]
    # Read after the 1s partition that chose the contract was rewritten.
    reference = session_path(tmp_path, sid, Timeframe.S1, DAY.label)
    write_day(tmp_path, sid, Timeframe.S1, DAY.label, read_partition(reference), "re-ingested")
    with pytest.raises(ValueError, match=":selection:bars.parquet:sha256:"):
        forming()


def test_forming_refuses_frames_altered_after_loading(tmp_path: Path) -> None:
    _, bars, trades = seeded(tmp_path)

    def sized(change: Callable[[Any], Any]) -> pd.DataFrame:
        altered = trades.copy()  # the loader's records of its source travel with it
        altered["size"] = change(altered["size"])
        return altered

    with pytest.raises(ValueError, match="does not reproduce stored bar 0"):
        forming_bars(decisions(SECOND + 1), bars, sized(lambda s: s * 2), observation_provenance=OBSERVED)
    # Conversely, a stored bar available by the decision must have been replayed.
    with pytest.raises(ValueError, match="stored bar 0 became available at .* but replaying the supplied trades has not completed it"):
        forming_bars(decisions(T[2] + 1), bars, sized(lambda s: s // 2), observation_provenance=OBSERVED)
    # With the last trade grown to a whole bar, the replay completes a bar 2 the frame lacks.
    grown = sized(lambda s: s.where(s != 250, 1000))
    with pytest.raises(ValueError, match="completes bar 2, which the stored bars do not hold"):
        forming_bars(decisions(DAY.end_unix * NS), bars, grown, observation_provenance=OBSERVED)
    unlabelled = load_trade_frame(tmp_path / "trades.dbn", SID, sessions=DAYS)
    with pytest.raises(ProvenanceError, match="trades carry no loader provenance"):
        forming_bars(decisions(T[0] + 1), bars, unlabelled, observation_provenance=OBSERVED)
    with pytest.raises(ValueError, match="reserved column name"):
        forming_bars(decisions(T[0] + 1, __observation=[0]), bars, trades, observation_provenance=OBSERVED)
    with pytest.raises(ValueError, match="would overwrite observation columns"):
        forming_bars(decisions(T[0] + 1, forming_open=[1.0]), bars, trades, observation_provenance=OBSERVED)


def test_forming_refuses_bounded_and_ineligible_trade_frames(tmp_path: Path) -> None:
    _, bars, _ = seeded(tmp_path)
    close = DAY.end_unix * NS
    # A receive-time window is refused outright: a missing tail could not be detected.
    bounded = load_trade_frame(
        tmp_path / "trades.dbn", SID, sessions=DAYS, start_ns=T[1], end_ns=close, feed="databento"
    )
    with pytest.raises(ValueError, match=re.escape(f"receive-time bounds ({T[1]}, {close})")):
        forming_bars(decisions(T[1] + 50), bars, bounded, observation_provenance=OBSERVED)
    # With that record stripped, the missing head is still caught at stored bar 0.
    bounded.attrs["receive_window"] = (None, None)
    with pytest.raises(ValueError, match="stored bar 0 became available"):
        forming_bars(decisions(T[1] + 50), bars, bounded, observation_provenance=OBSERVED)
    # An ineligible record is refused when the replay reaches it, never skipped.
    flagged = archive(
        tmp_path / "flagged.dbn",
        [TRADES[0], TRADES[1], VolumeInput.trade(T[1] + 10, 50.0, 0, 42), *TRADES[2:]],
    )
    zero = load_trade_frame(flagged, SID, sessions=DAYS, feed="databento")
    assert zero["eligible"].tolist() == [True, True, False, True, True, True]
    # Such a record can only reach a replay in a frame altered after loading.
    emptied = load_trade_frame(tmp_path / "trades.dbn", SID, sessions=DAYS, feed="databento")
    emptied.loc[2, "size"] = 0
    emptied.loc[2, "eligible"] = False
    emptied.loc[2, "issues"] = int(TradeIssue.ZERO_SIZE)
    before = forming_bars(decisions(T[0] + 50), bars, emptied, observation_provenance=OBSERVED)
    assert ids(before, "forming_volume") == [400]
    with pytest.raises(ValueError, match="trade 2 is ineligible for receive-time bars .*ZERO_SIZE"):
        forming_bars(decisions(T[2] + 50), bars, emptied, observation_provenance=OBSERVED)
    # Windows return the record as the reader retained it, flagged.
    window = trade_windows(decisions(T[2]), zero, observation_provenance=OBSERVED, lookback_ns=2 * NS)
    assert window["trade_eligible"].tolist() == [True, True, False]


def test_forming_at_the_close_treats_the_remainder_as_published_only_with_exact_matches(
    tmp_path: Path,
) -> None:
    spec, _, trades = seeded(tmp_path)
    full = load_volume_frame(tmp_path, SID, spec, DAYS, feed="databento", include_partial=True)
    at_close = decisions(DAY.end_unix * NS)
    history = bar_history(at_close, full, observation_provenance=OBSERVED, depth=5, max_age_ns=HOUR)
    assert history["volume_bar_id"].tolist() == [1, 0]
    forming = forming_bars(at_close, full, trades, observation_provenance=OBSERVED)
    assert ids(forming, "forming_bar_id") == [2] and ids(forming, "forming_volume") == [250]
    inclusive = bar_history(
        at_close, full, observation_provenance=OBSERVED, depth=5, max_age_ns=HOUR, allow_exact_matches=True
    )
    assert inclusive["volume_bar_id"].tolist() == [2, 1, 0]
    assert inclusive["volume_is_partial"].tolist() == [True, False, False]
    published = forming_bars(at_close, full, trades, observation_provenance=OBSERVED, allow_exact_matches=True)
    assert ids(published, "forming_bar_id") == [None]


def test_forming_needs_a_trade_built_definition_and_tolerates_unknown_contracts(tmp_path: Path) -> None:
    _, bars, trades = seeded(tmp_path)
    seed_time_bars(tmp_path)
    approximate = VolumeSpec(1000, "ohlcv-1s")
    derive_volume(tmp_path, SID, approximate, DAYS)
    seconds = load_volume_frame(tmp_path, SID, approximate, DAYS, feed="databento")
    with pytest.raises(ValueError, match="replayed from trades"):
        forming_bars(decisions(T[0] + 1), seconds, trades, observation_provenance=OBSERVED)
    # Seconds rebuilt from these very trades are still seconds, not a trade replay.
    build_volume_archive(
        tmp_path / "trades.dbn", tmp_path, SID, VolumeSpec(1000), DAYS, compare_1s=True
    )
    rebuilt = load_volume_frame(
        tmp_path, SID, VolumeSpec(1000, "trades-1s"), DAYS, feed="databento"
    )
    with pytest.raises(ValueError, match="replayed from trades"):
        forming_bars(decisions(T[0] + 1), rebuilt, trades, observation_provenance=OBSERVED)
    # Attributes edited without rehashing the definition column fail first.
    edited = bars.copy()
    edited.attrs["bar_definition"] = {**bars.attrs["bar_definition"], "version": 2}
    with pytest.raises(ProvenanceError, match="is not the sha256"):
        forming_bars(decisions(T[0] + 1), edited, trades, observation_provenance=OBSERVED)
    # A consistently rehashed definition this builder does not construct is refused.
    for change, message in (({"version": 2}, "builder version or policy"), ({"threshold": None}, "cannot be replayed")):
        foreign = bars.copy()
        definition: dict[str, object] = {**bars.attrs["bar_definition"], **change}
        foreign.attrs["bar_definition"] = definition
        foreign["definition"] = definition_key(definition)
        with pytest.raises(ValueError, match=message):
            forming_bars(decisions(T[0] + 1), foreign, trades, observation_provenance=OBSERVED)
    # An instrument with no trades has nothing forming; it is not an error.
    unknown = decisions(T[2] + 1).assign(instrument_id=43)
    assert ids(forming_bars(unknown, bars, trades, observation_provenance=OBSERVED), "forming_bar_id") == [None]
    # Declare integer keys: pandas would type the empty columns float.
    none = decisions().astype({"asof_ns": "int64", "instrument_id": "int64"})
    empty = forming_bars(none, bars, trades, observation_provenance=OBSERVED)
    assert empty.empty and list(empty.columns[-len(FORMING_COLUMNS):]) == [f"forming_{n}" for n in FORMING_COLUMNS]
    assert bar_history(none, bars, observation_provenance=OBSERVED, depth=1, max_age_ns=HOUR).empty
    assert trade_windows(none, trades, observation_provenance=OBSERVED, lookback_ns=NS).empty
