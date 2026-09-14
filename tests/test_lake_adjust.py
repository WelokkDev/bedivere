"""lake.adjust + BackAdjustedLakeSource — the read-time adjusted view.

The store stays raw; the same pin always reproduces the same numbers, and a
different pin is a DIFFERENT series that coexists rather than replacing it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.lake_support import SKIP_REASON

pytest.importorskip("duckdb", reason=SKIP_REASON)
pytest.importorskip("pandas", reason=SKIP_REASON)

from bedivere.config.spec import DataSpec  # noqa: E402
from bedivere.core.session_days import SessionDay, expected_close_stamps  # noqa: E402
from bedivere.core.types import Candle, Timeframe  # noqa: E402
from bedivere.data.build import build_candle_source  # noqa: E402
from bedivere.data.lake.adjust import AdjustmentError, back_adjustment  # noqa: E402
from bedivere.data.lake.layout import RollRule, local_continuous  # noqa: E402
from bedivere.data.lake.schema import BarBatch  # noqa: E402
from bedivere.data.lake.writer import write_day  # noqa: E402
from bedivere.data.port import CandleSource  # noqa: E402
from bedivere.data.source import (  # noqa: E402
    BackAdjustedLakeSource,
    BarSourceError,
    LakeCandleSource,
)
from tests.helpers import eth_session_days  # noqa: E402
from tests.lake_helpers import flat_bars  # noqa: E402

SID = local_continuous("GLBX.MDP3", "NQ", RollRule.VOLUME, 0)
TF = Timeframe.H1

LABELS = ["2026-06-11", "2026-06-12", "2026-06-15", "2026-06-16"]
DAYS = eth_session_days(LABELS)
STAMPS_PER_DAY = len(expected_close_stamps(DAYS.days[0], TF))

OLD_ID, NEW_ID = 4104058, 4104177
OLD_LEVEL, NEW_LEVEL = 100.0, 400.0
GAP = NEW_LEVEL - OLD_LEVEL  # the new era's first open minus the old era's last close


def _bars(day: SessionDay, level: float) -> list[Candle]:
    return flat_bars(expected_close_stamps(day, TF), level)


def _seed(root: Path) -> None:
    """Two eras: sessions 0-1 on the old contract at 100, 2-3 on the new at 400."""
    for i, day in enumerate(DAYS.days):
        iid, level = (OLD_ID, OLD_LEVEL) if i < 2 else (NEW_ID, NEW_LEVEL)
        write_day(
            root, SID, TF, day.label,
            BarBatch.from_candles(_bars(day, level), instrument_id=iid),
            source="vendor.dbn.zst",
        )


def _source(root: Path, as_of: str) -> BackAdjustedLakeSource:
    return BackAdjustedLakeSource(LakeCandleSource(SID.dataset, SID.series, root), as_of)


def _spec_source(root: Path, as_of: str | None = None, **extra: object) -> CandleSource:
    payload: dict[str, object] = {
        "source": "lake", "dataset": SID.dataset, "series": SID.series, "root": str(root),
        **extra,
    }
    if as_of is not None:
        payload["adjustment"] = {"method": "back_adjusted", "asOf": as_of}
    return build_candle_source(
        DataSpec.model_validate(payload), symbol="NQ", timeframe=TF
    )


# ---------- derivation ----------


def test_derives_the_boundary_and_the_close_to_open_gap(tmp_path: Path) -> None:
    _seed(tmp_path)
    adj = back_adjustment(tmp_path, SID, TF, LABELS[-1])

    assert len(adj.boundaries) == 1
    b = adj.boundaries[0]
    assert b.first_session == LABELS[2]  # the first session of the NEW era
    assert b.gap == GAP
    assert b.first_ts == expected_close_stamps(DAYS.days[2], TF)[0]
    assert adj.horizon_ts == expected_close_stamps(DAYS.days[-1], TF)[-1]


def test_an_earlier_as_of_excludes_the_later_roll(tmp_path: Path) -> None:
    # After the lake grows past a roll, the old pin still derives the old roll
    # set: the series never mutates under a caller holding yesterday's as-of.
    _seed(tmp_path)
    adj = back_adjustment(tmp_path, SID, TF, LABELS[1])
    assert adj.boundaries == ()
    assert adj.horizon_ts == expected_close_stamps(DAYS.days[1], TF)[-1]


def test_a_session_holding_two_contracts_refuses(tmp_path: Path) -> None:
    # A roll lands at a session START here, so two ids inside one session means
    # a malformed partition and a gap measured at the wrong pair of bars.
    for i, day in enumerate(DAYS.days[:3]):
        bars = _bars(day, OLD_LEVEL)
        batch = BarBatch.from_candles(bars, instrument_id=OLD_ID)
        if i == 1:
            half = len(bars) // 2
            batch.instrument_id[half:] = [NEW_ID] * (len(bars) - half)
        write_day(tmp_path, SID, TF, day.label, batch, source="vendor.dbn.zst")
    with pytest.raises(AdjustmentError, match="instrument ids"):
        back_adjustment(tmp_path, SID, TF, LABELS[2])


def test_nothing_stored_at_or_before_the_as_of_refuses(tmp_path: Path) -> None:
    _seed(tmp_path)
    with pytest.raises(AdjustmentError, match="holds nothing at or before"):
        back_adjustment(tmp_path, SID, TF, "2020-01-01")


def test_the_gap_is_first_open_minus_last_close_and_not_any_other_pair(tmp_path: Path) -> None:
    """Every price field differs per bar AND per position here, so swapping
    arg_min/arg_max or open/close changes the answer — which the flat fixtures
    elsewhere cannot detect."""
    old_day, new_day = DAYS.days[0], DAYS.days[1]

    def sloped(day: SessionDay, base: float) -> list[Candle]:
        out: list[Candle] = []
        for i, ts in enumerate(expected_close_stamps(day, TF)):
            o = base + i * 2.0  # opens walk one grid
            c = base + i * 2.0 + 0.75  # closes walk another
            out.append(
                Candle(timestamp=ts, open=o, high=max(o, c) + 1, low=min(o, c) - 1, close=c, volume=1.0)
            )
        return out

    old_bars, new_bars = sloped(old_day, 100.0), sloped(new_day, 500.0)
    write_day(tmp_path, SID, TF, old_day.label,
              BarBatch.from_candles(old_bars, instrument_id=OLD_ID), source="v.dbn.zst")
    write_day(tmp_path, SID, TF, new_day.label,
              BarBatch.from_candles(new_bars, instrument_id=NEW_ID), source="v.dbn.zst")

    adj = back_adjustment(tmp_path, SID, TF, new_day.label)
    expected = new_bars[0].open - old_bars[-1].close  # THE convention
    assert adj.boundaries[0].gap == expected
    # And it is not any of the plausible impostors.
    assert expected != new_bars[0].close - old_bars[-1].close
    assert expected != new_bars[-1].open - old_bars[-1].close
    assert expected != new_bars[0].open - old_bars[0].close


def test_a_calendar_hole_at_the_roll_refuses(tmp_path: Path) -> None:
    """A boundary measured across missing sessions would fold every absent day's
    move into the offset."""
    far_apart = eth_session_days(["2026-06-11", "2026-06-26"])  # 15 calendar days
    for day, (iid, level) in zip(
        far_apart.days, [(OLD_ID, OLD_LEVEL), (NEW_ID, NEW_LEVEL)], strict=True
    ):
        write_day(tmp_path, SID, TF, day.label,
                  BarBatch.from_candles(_bars(day, level), instrument_id=iid), source="v.dbn.zst")
    with pytest.raises(AdjustmentError, match="calendar days"):
        back_adjustment(tmp_path, SID, TF, "2026-06-26")


def test_a_single_missing_midweek_session_at_the_roll_refuses(tmp_path: Path) -> None:
    # Mon -> Wed is 2 calendar days: always a missing Tuesday, never a weekend,
    # so a rule that only rejected long holes would wave this through.
    days = eth_session_days(["2026-06-15", "2026-06-17"])
    for day, (iid, level) in zip(
        days.days, [(OLD_ID, OLD_LEVEL), (NEW_ID, NEW_LEVEL)], strict=True
    ):
        write_day(tmp_path, SID, TF, day.label,
                  BarBatch.from_candles(_bars(day, level), instrument_id=iid), source="v.dbn.zst")
    with pytest.raises(AdjustmentError, match="session is missing"):
        back_adjustment(tmp_path, SID, TF, "2026-06-17")


def test_non_canonical_iso_pins_refuse() -> None:
    # `date.fromisoformat` accepts both, but '20260612' sorts before every
    # dashed label, so the lexicographic session filter would include everything.
    for bad in ("20260612", "2026-W24-5"):
        with pytest.raises(BarSourceError, match="canonical ISO"):
            BackAdjustedLakeSource(LakeCandleSource("GLBX.MDP3", "local.v.0"), bad)
    with pytest.raises(AdjustmentError, match="canonical ISO"):
        back_adjustment(Path("."), SID, TF, "20260612")
    with pytest.raises(AdjustmentError, match="not a real calendar date"):
        back_adjustment(Path("."), SID, TF, "2026-02-30")


# ---------- the source view ----------


def test_the_old_era_shifts_by_the_gap_and_the_new_era_stays_raw(tmp_path: Path) -> None:
    _seed(tmp_path)
    src = _source(tmp_path, LABELS[-1])
    got = src.candles("NQ", TF, DAYS.days[0].start_unix, DAYS.days[-1].end_unix)

    assert len(got) == 4 * STAMPS_PER_DAY
    old_era, new_era = got[: 2 * STAMPS_PER_DAY], got[2 * STAMPS_PER_DAY :]
    # All four price fields shift; volume and stamps do not.
    assert all(c.open == OLD_LEVEL + GAP for c in old_era)
    assert all(c.high == OLD_LEVEL + 2 + GAP for c in old_era)
    assert all(c.low == OLD_LEVEL - 2 + GAP for c in old_era)
    assert all(c.close == OLD_LEVEL + GAP for c in old_era)
    assert all(c.volume == 7.0 for c in old_era)
    assert all(c.close == NEW_LEVEL for c in new_era)
    # No artificial cliff at the seam.
    assert old_era[-1].close == new_era[0].open


def test_two_rolls_accumulate_so_the_oldest_era_carries_both_gaps(tmp_path: Path) -> None:
    """Era-A bars carry BOTH gaps, era-B one, era-C none. The two gaps differ,
    so a swapped suffix order cannot pass."""
    levels = [100.0, 300.0, 1000.0]  # A->B gap 200, B->C gap 700
    ids = [111, 222, 333]
    for i, day in enumerate(DAYS.days[:3]):
        write_day(tmp_path, SID, TF, day.label,
                  BarBatch.from_candles(_bars(day, levels[i]), instrument_id=ids[i]),
                  source="vendor.dbn.zst")

    adj = back_adjustment(tmp_path, SID, TF, LABELS[2])
    assert [b.gap for b in adj.boundaries] == [200.0, 700.0]

    got = _source(tmp_path, LABELS[2]).candles("NQ", TF, DAYS.days[0].start_unix, DAYS.days[2].end_unix)
    era_a = got[:STAMPS_PER_DAY]
    era_b = got[STAMPS_PER_DAY : 2 * STAMPS_PER_DAY]
    era_c = got[2 * STAMPS_PER_DAY :]
    assert all(c.close == 100.0 + 200.0 + 700.0 for c in era_a)  # both gaps
    assert all(c.close == 300.0 + 700.0 for c in era_b)  # the later gap only
    assert all(c.close == 1000.0 for c in era_c)  # the newest era, raw
    # Continuity at BOTH seams.
    assert era_a[-1].close == era_b[0].open
    assert era_b[-1].close == era_c[0].open


def test_reading_past_the_as_of_horizon_refuses(tmp_path: Path) -> None:
    _seed(tmp_path)
    src = _source(tmp_path, LABELS[1])  # pinned before the roll
    # Within the pin: served, and raw — no boundary in the set.
    got = src.candles("NQ", TF, DAYS.days[0].start_unix, DAYS.days[1].end_unix)
    assert all(c.close == OLD_LEVEL for c in got)
    # Past the pin: refused, even though the lake holds those bars.
    with pytest.raises(BarSourceError, match="past the as-of horizon"):
        src.candles("NQ", TF, DAYS.days[0].start_unix, DAYS.days[2].end_unix)
    # The freshness surface agrees with the refusal.
    assert src.latest("NQ", TF) == expected_close_stamps(DAYS.days[1], TF)[-1]


def test_the_serve_guard_fires_inside_the_request_slack(tmp_path: Path) -> None:
    """The request check carries a day of slack; the SERVE check holds the pin.
    Ending the window early on the next session passes the request guard, so the
    serve guard must refuse with ITS message."""
    _seed(tmp_path)
    src = _source(tmp_path, LABELS[0])
    horizon = expected_close_stamps(DAYS.days[0], TF)[-1]
    next_first = expected_close_stamps(DAYS.days[1], TF)[0]
    assert next_first - horizon < 86_400  # inside the request slack, by construction
    with pytest.raises(BarSourceError, match="guarantees nothing beyond"):
        src.candles("NQ", TF, DAYS.days[0].start_unix, next_first)


def test_the_natural_session_close_read_serves_despite_a_late_last_trade(tmp_path: Path) -> None:
    """A session's last trade lands before its close, so `end = session close`
    overshoots the last stored bar. That read must SERVE, or the pinned session
    itself becomes unreadable."""
    for i, day in enumerate(DAYS.days[:2]):
        bars = _bars(day, OLD_LEVEL)
        if i == 1:
            bars = bars[:-2]  # last trade ~2 stamps before the close
        write_day(tmp_path, SID, TF, day.label,
                  BarBatch.from_candles(bars, instrument_id=OLD_ID), source="v.dbn.zst")
    src = _source(tmp_path, LABELS[1])
    got = src.candles("NQ", TF, DAYS.days[0].start_unix, DAYS.days[1].end_unix)
    assert len(got) == 2 * STAMPS_PER_DAY - 2
    assert src.latest("NQ", TF) == expected_close_stamps(DAYS.days[1], TF)[-3]


def test_two_pins_are_two_series_with_two_identities(tmp_path: Path) -> None:
    _seed(tmp_path)
    early, late = _source(tmp_path, LABELS[1]), _source(tmp_path, LABELS[-1])
    raw = LakeCandleSource(SID.dataset, SID.series, tmp_path)

    assert early.describe() == "lake:GLBX.MDP3/local.v.0@backadj:2026-06-12"
    assert late.describe() == "lake:GLBX.MDP3/local.v.0@backadj:2026-06-16"
    assert len({early.describe(), late.describe(), raw.describe()}) == 3
    assert raw.price_basis == "as_traded"
    assert early.price_basis == "back_adjusted"

    # Both readable side by side; neither mutates the other or the store.
    span = (DAYS.days[0].start_unix, DAYS.days[1].end_unix)
    assert [c.close for c in early.candles("NQ", TF, *span)] == [OLD_LEVEL] * 2 * STAMPS_PER_DAY
    assert [c.close for c in late.candles("NQ", TF, *span)] == [OLD_LEVEL + GAP] * 2 * STAMPS_PER_DAY
    assert [c.close for c in raw.candles("NQ", TF, *span)] == [OLD_LEVEL] * 2 * STAMPS_PER_DAY


def test_covered_days_is_truncated_at_the_pin(tmp_path: Path) -> None:
    # Reporting a stored-but-unservable day as covered would move the failure
    # from plan time to run time.
    _seed(tmp_path)
    assert _source(tmp_path, LABELS[1]).covered_days("NQ", TF, DAYS) == frozenset(LABELS[:2])
    assert LakeCandleSource(SID.dataset, SID.series, tmp_path).covered_days(
        "NQ", TF, DAYS
    ) == frozenset(LABELS)


def test_a_failed_derivation_is_cached_so_it_costs_one_scan(tmp_path: Path) -> None:
    days = eth_session_days(["2026-06-15", "2026-06-17"])
    for day, (iid, level) in zip(
        days.days, [(OLD_ID, OLD_LEVEL), (NEW_ID, NEW_LEVEL)], strict=True
    ):
        write_day(tmp_path, SID, TF, day.label,
                  BarBatch.from_candles(_bars(day, level), instrument_id=iid), source="v.dbn.zst")
    src = _source(tmp_path, "2026-06-17")
    with pytest.raises(BarSourceError, match="session is missing") as first:
        src.candles("NQ", TF, 0, days.days[-1].end_unix)
    with pytest.raises(BarSourceError) as second:
        src.candles("NQ", TF, 0, days.days[-1].end_unix)
    # Same exception object: the full-series scan was not repaid on the retry.
    assert second.value is first.value


# ---------- the spec spelling ----------


def test_a_spec_builds_the_adjusted_view(tmp_path: Path) -> None:
    _seed(tmp_path)
    src = _spec_source(tmp_path, LABELS[-1], allowStale=True)
    assert "@backadj:2026-06-16" in src.describe()
    got = src.candles("NQ", TF, DAYS.days[0].start_unix, DAYS.days[0].end_unix)
    assert all(c.close == OLD_LEVEL + GAP for c in got)


def test_an_adjustment_without_an_as_of_is_unspellable() -> None:
    with pytest.raises(ValueError, match="no safe default"):
        DataSpec.model_validate(
            {"source": "lake", "dataset": "D", "series": "local.v.0",
             "adjustment": {"method": "back_adjusted"}}
        )


def test_an_unknown_adjustment_method_refuses() -> None:
    with pytest.raises(ValueError, match="back_adjusted"):
        DataSpec.model_validate(
            {"source": "lake", "dataset": "D", "series": "local.v.0",
             "adjustment": {"method": "ratio", "asOf": "2026-06-16"}}
        )


def test_an_adjustment_on_a_file_source_is_unspellable() -> None:
    with pytest.raises(ValueError, match="lake-only"):
        DataSpec.model_validate(
            {"source": "csv", "path": "x.csv",
             "adjustment": {"method": "back_adjusted", "asOf": "2026-06-16"}}
        )
