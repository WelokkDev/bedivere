"""bedivere.engine.metrics — aggregates are honest: undefined ratios stay None."""

from __future__ import annotations

from typing import cast

from bedivere.core.pricing import spec_from_handoff
from bedivere.engine.metrics import compute_metrics
from bedivere.engine.portfolio import TradeRecord

SPEC = spec_from_handoff("DEMO", 0.25, 20)  # tick = 500 cents


def _dig(payload: object, *keys: str) -> object:
    """Walk a nested result block by key.

    The metrics envelope is `dict[str, object]` by design — it is JSON,
    not a typed model — so a chain of subscripts is unknown-typed under
    strict checking. One asserting walker keeps the tests readable and
    fails with the key that was missing rather than a KeyError."""
    for key in keys:
        assert isinstance(payload, dict), f"expected a dict before {key!r}"
        payload = cast("dict[str, object]", payload)[key]
    return payload


def _trade(net_cents: int, *, risk_ticks: int | None = None, ambiguous: bool = False) -> TradeRecord:
    return TradeRecord(
        bracket_id=1,
        symbol="DEMO",
        direction="long",
        qty=1,
        entry_ticks=400,
        exit_ticks=400,
        signal_ticks=400,
        exit_kind="target_fill",
        entry_ts=0,
        exit_ts=60,
        pnl_cents=net_cents,  # fees held at 0 → net == pnl
        fees_cents=0,
        ambiguous=ambiguous,
        tag="",
        stamps={},
        risk_ticks=risk_ticks,
    )


def test_empty_run_has_no_fake_numbers() -> None:
    m = compute_metrics([], SPEC)
    assert m["trades"] == 0
    assert m["winRate"] is None  # not 0.0 — undefined
    assert m["profitFactor"] is None
    assert m["avgTradeNetCents"] is None
    assert m["maxDrawdownCents"] == 0
    r = m["rMultiples"]
    assert isinstance(r, dict) and r["known"] == 0 and r["avg"] is None


def test_mixed_run_aggregates() -> None:
    trades = [
        _trade(1000, risk_ticks=4),  # W  (+1000 / 2000 risk → R +0.5)
        _trade(-500, risk_ticks=4, ambiguous=True),  # L  (R −0.25)
        _trade(0),  # flat, no risk recorded
        _trade(250),  # W
    ]
    m = compute_metrics(trades, SPEC)
    assert (m["wins"], m["losses"], m["flat"]) == (2, 1, 1)
    assert m["winRate"] == 0.5
    assert m["netCents"] == 750
    assert m["grossProfitCents"] == 1250
    assert m["grossLossCents"] == 500
    assert m["profitFactor"] == 2.5
    # Equity path 1000 → 500 → 500 → 750: worst peak-to-trough is 500.
    assert m["maxDrawdownCents"] == 500
    assert m["longestWinStreak"] == 1
    assert m["longestLossStreak"] == 1
    assert m["ambiguousFills"] == 1
    r = m["rMultiples"]
    assert isinstance(r, dict)
    assert r["known"] == 2
    assert r["avg"] == 0.125
    assert r["best"] == 0.5
    assert r["worst"] == -0.25


def test_no_losses_means_no_profit_factor() -> None:
    m = compute_metrics([_trade(100), _trade(200)], SPEC)
    assert m["profitFactor"] is None  # not infinity, not a big number
    assert m["longestWinStreak"] == 2


# ---------- latency: the six stamps, finally reported ----------


def _stamped(**stamps: int) -> TradeRecord:
    t = _trade(100, risk_ticks=4)
    t.stamps = dict(stamps)
    return t


def test_latency_reports_each_consecutive_stamp_pair() -> None:
    m = compute_metrics(
        [
            _stamped(
                ts_bar_close=1_000,
                ts_event_received=1_050,
                ts_decided=1_060,
                ts_submitted=1_060,
                ts_acked=1_310,
                ts_filled=1_400,
            )
        ],
        SPEC,
    )
    assert _dig(m, "latency", "closeToReceived", "mean") == 50
    assert _dig(m, "latency", "submittedToAcked", "mean") == 250
    assert _dig(m, "latency", "closeToFilled", "mean") == 400


def test_a_leg_no_broker_ever_stamped_is_absent_not_empty() -> None:
    """A stamp your venue never sets should be visibly missing, not reported
    as a distribution over nothing."""
    m = compute_metrics([_stamped(ts_bar_close=1_000, ts_filled=1_400)], SPEC)
    latency = m["latency"]
    assert isinstance(latency, dict)
    assert "closeToFilled" in latency
    assert "submittedToAcked" not in latency


def test_trades_without_stamps_produce_an_empty_latency_block() -> None:
    m = compute_metrics([_trade(100)], SPEC)
    assert m["latency"] == {}


# ---------- execution: two baselines, split by entry type ----------


def _filled(
    *, entry: int, open_ticks: int, signal: int, entry_type: str, direction: str = "long"
) -> TradeRecord:
    t = _trade(100, risk_ticks=4)
    t.direction = direction  # type: ignore[assignment]
    t.entry_ticks = entry
    t.entry_open_ticks = open_ticks
    t.signal_ticks = signal
    t.entry_type = entry_type
    return t


def test_execution_splits_slippage_by_entry_type() -> None:
    """Mixing a market entry's spread with a resting entry's price
    improvement produces a number that means nothing, so they never share a
    bucket."""
    m = compute_metrics(
        [
            _filled(entry=402, open_ticks=400, signal=400, entry_type="market"),
            _filled(entry=395, open_ticks=400, signal=395, entry_type="limit"),
        ],
        SPEC,
    )
    base = ("execution", "byEntryType")
    assert _dig(m, *base, "market", "entrySlipTicks", "mean") == 2
    # A buy limit that filled BELOW the open is price improvement — negative
    # by the "positive is adverse" sign convention, not a cost.
    assert _dig(m, *base, "limit", "entrySlipTicks", "mean") == -5


def test_signal_slip_is_the_measure_that_survives_a_resting_entry() -> None:
    """A stop entry filling past its trigger is a real taker cost — and it
    is exactly what a fill-derived target would have absorbed silently."""
    m = compute_metrics(
        [_filled(entry=406, open_ticks=390, signal=404, entry_type="stop")],
        SPEC,
    )
    assert _dig(m, "execution", "byEntryType", "stop", "signalSlipTicks", "mean") == 2


def test_slippage_signs_are_adverse_positive_for_a_short() -> None:
    m = compute_metrics(
        [
            _filled(
                entry=398, open_ticks=400, signal=400, entry_type="market", direction="short"
            )
        ],
        SPEC,
    )
    # A short receiving LESS than the open is adverse, so the sign is positive.
    assert _dig(m, "execution", "byEntryType", "market", "entrySlipTicks", "mean") == 2


# ---------- Sharpe, with its basis attached ----------


def _dated(net_cents: int, exit_ts: int) -> TradeRecord:
    t = _trade(net_cents)
    t.exit_ts = exit_ts
    return t


def test_sharpe_declares_its_basis() -> None:
    day = 86_400
    m = compute_metrics(
        [_dated(100, day), _dated(-50, 2 * day), _dated(300, 3 * day)], SPEC
    )
    assert _dig(m, "sharpeBasis", "observations") == 3
    assert _dig(m, "sharpeBasis", "annualisationDays") == 252
    assert _dig(m, "sharpeBasis", "dispersion") == "sample-stdev"
    assert _dig(m, "sharpeBasis", "grouping") == "utc-date"  # no SessionDays given
    assert _dig(m, "sharpeBasis", "tradelessDaysIncluded") is False
    assert isinstance(m["sharpeDaily"], float)


def test_one_observation_has_no_dispersion_so_no_sharpe() -> None:
    m = compute_metrics([_dated(100, 86_400)], SPEC)
    assert m["sharpeDaily"] is None  # not 0.0
    assert _dig(m, "sharpeBasis", "observations") == 1


def test_a_flat_daily_series_has_no_sharpe_either() -> None:
    day = 86_400
    m = compute_metrics([_dated(100, day), _dated(100, 2 * day)], SPEC)
    assert m["sharpeDaily"] is None  # zero dispersion, not an infinite ratio


# ---------- the second unit of account ----------


def _tagged(net_cents: int, tag: str, *, risk_ticks: int | None = 4) -> TradeRecord:
    t = _trade(net_cents, risk_ticks=risk_ticks)
    t.tag = tag
    return t


def test_untagged_trades_produce_no_groups_block() -> None:
    """A strategy that never tagged its brackets has not asked for this
    view; an empty block in every result is noise."""
    assert "groups" not in compute_metrics([_trade(100)], SPEC)


def test_groups_put_the_unit_of_account_back_on_the_idea() -> None:
    """Two brackets for one idea — a cancelled first attempt and a winning
    retry — is ONE decision, and per-trade win rate calls it 50%."""
    trades = [
        _tagged(-200, "idea-1"),
        _tagged(500, "idea-1"),
        _tagged(-100, "idea-2"),
    ]
    m = compute_metrics(trades, SPEC)
    assert m["winRate"] == 0.3333  # per trade: 1 of 3
    assert _dig(m, "groups", "distinct") == 2
    assert _dig(m, "groups", "winRate") == 0.5  # per idea: idea-1 netted +300
    assert _dig(m, "groups", "netCents") == 200
    assert _dig(m, "groups", "attemptsPerGroup", "max") == 2


def test_groups_count_untagged_trades_separately() -> None:
    m = compute_metrics([_tagged(100, "idea-1"), _trade(50)], SPEC)
    assert _dig(m, "groups", "taggedTrades") == 1
    assert _dig(m, "groups", "untaggedTrades") == 1


# ---------- the rest of the envelope ----------


def test_exit_kinds_are_counted() -> None:
    stopped = _trade(-100)
    stopped.exit_kind = "stop_fill"
    m = compute_metrics([_trade(100), stopped, _trade(200)], SPEC)
    assert m["exitKinds"] == {"stop_fill": 1, "target_fill": 2}


def test_net_over_max_drawdown_is_none_when_nothing_drew_down() -> None:
    m = compute_metrics([_trade(100), _trade(200)], SPEC)
    assert m["netOverMaxDrawdown"] is None  # undefined, not infinite


def test_expectancy_is_the_r_average() -> None:
    m = compute_metrics([_trade(1000, risk_ticks=4), _trade(-500, risk_ticks=4)], SPEC)
    assert _dig(m, "rMultiples", "expectancy") == 0.125
