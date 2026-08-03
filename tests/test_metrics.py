"""bedivere.engine.metrics — aggregates are honest: undefined ratios stay None."""

from __future__ import annotations

from bedivere.core.pricing import spec_from_handoff
from bedivere.engine.metrics import compute_metrics
from bedivere.engine.portfolio import TradeRecord

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
