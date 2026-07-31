"""bedivere.stats — aggregates are honest: undefined ratios stay None."""

from __future__ import annotations

from bedivere.core.pricing import spec_from_handoff
from bedivere.engine.portfolio import TradeRecord
from bedivere.stats import compute_stats

SPEC = spec_from_handoff("DEMO", 0.25, 20)  # tick = 500 cents


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
    s = compute_stats([], SPEC)
    assert s["trades"] == 0
    assert s["winRate"] is None  # not 0.0 — undefined
    assert s["profitFactor"] is None
    assert s["avgTradeNetCents"] is None
    assert s["maxDrawdownCents"] == 0
    r = s["rMultiples"]
    assert isinstance(r, dict) and r["known"] == 0 and r["avg"] is None


def test_mixed_run_aggregates() -> None:
    trades = [
        _trade(1000, risk_ticks=4),  # W  (+1000 / 2000 risk → R +0.5)
        _trade(-500, risk_ticks=4, ambiguous=True),  # L  (R −0.25)
        _trade(0),  # flat, no risk recorded
        _trade(250),  # W
    ]
    s = compute_stats(trades, SPEC)
    assert (s["wins"], s["losses"], s["flat"]) == (2, 1, 1)
    assert s["winRate"] == 0.5
    assert s["netCents"] == 750
    assert s["grossProfitCents"] == 1250
    assert s["grossLossCents"] == 500
    assert s["profitFactor"] == 2.5
    # Equity path 1000 → 500 → 500 → 750: worst peak-to-trough is 500.
    assert s["maxDrawdownCents"] == 500
    assert s["longestWinStreak"] == 1
    assert s["longestLossStreak"] == 1
    assert s["ambiguousFills"] == 1
    r = s["rMultiples"]
    assert isinstance(r, dict)
    assert r["known"] == 2
    assert r["avg"] == 0.125
    assert r["best"] == 0.5
    assert r["worst"] == -0.25


def test_no_losses_means_no_profit_factor() -> None:
    s = compute_stats([_trade(100), _trade(200)], SPEC)
    assert s["profitFactor"] is None  # not infinity, not a big number
    assert s["longestWinStreak"] == 2
