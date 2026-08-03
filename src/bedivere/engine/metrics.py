"""Run-level metrics — honest aggregates over completed trades.

Kept OUT of Portfolio on purpose: the portfolio owns trading state, this
owns interpretation, and interpretation changes far more often than state.

Ratios that are undefined stay None: a run with no losing trades has no
profit factor, not an infinite one; a run with no trades has no win rate,
not a 0% one. Faking a number where there isn't one is exactly the kind of
flattery this engine exists to refuse.

Money stays int cents until the final jsonable floats; R-multiples come
from the risk the venue ACTUALLY armed (fill-derived stop), recorded on the
TradeRecord — no join against config."""

from __future__ import annotations

from bedivere.core.pricing import InstrumentSpec
from bedivere.engine.portfolio import TradeRecord


def compute_metrics(trades: list[TradeRecord], spec: InstrumentSpec) -> dict[str, object]:
    """Aggregate metrics for one run's trade list. Deterministic; all fields
    JSON-serializable. Sums are exact int cents; ratios are rounded floats
    or None where undefined."""
    n = len(trades)
    nets = [t.pnl_cents - t.fees_cents for t in trades]
    wins = [x for x in nets if x > 0]
    losses = [x for x in nets if x < 0]
    gross_profit = sum(wins)
    gross_loss = -sum(losses)  # positive cents

    # Equity curve by trade close → max drawdown in cents.
    peak = 0
    equity = 0
    max_drawdown = 0
    for x in nets:
        equity += x
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_drawdown:
            max_drawdown = dd

    # Longest win/loss streaks (flat trades break both).
    best_win_streak = best_loss_streak = 0
    win_run = loss_run = 0
    for x in nets:
        if x > 0:
            win_run += 1
            loss_run = 0
        elif x < 0:
            loss_run += 1
            win_run = 0
        else:
            win_run = loss_run = 0
        best_win_streak = max(best_win_streak, win_run)
        best_loss_streak = max(best_loss_streak, loss_run)

    r_values = [r for t in trades if (r := t.r_multiple(spec)) is not None]

    return {
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "flat": n - len(wins) - len(losses),
        "winRate": _ratio(len(wins), n),
        "netCents": sum(nets),
        "grossProfitCents": gross_profit,
        "grossLossCents": gross_loss,
        "profitFactor": _ratio(gross_profit, gross_loss),
        "avgTradeNetCents": _ratio(sum(nets), n),
        "maxDrawdownCents": max_drawdown,
        "longestWinStreak": best_win_streak,
        "longestLossStreak": best_loss_streak,
        "ambiguousFills": sum(1 for t in trades if t.ambiguous),
        "rMultiples": {
            "known": len(r_values),
            "avg": _round4(sum(r_values) / len(r_values)) if r_values else None,
            "best": _round4(max(r_values)) if r_values else None,
            "worst": _round4(min(r_values)) if r_values else None,
            "sum": _round4(sum(r_values)) if r_values else None,
        },
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    """None when the denominator is zero — undefined is not zero."""
    if denominator == 0:
        return None
    return _round4(numerator / denominator)


def _round4(x: float) -> float:
    return round(x, 4)
