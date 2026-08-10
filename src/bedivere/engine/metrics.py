"""Run-level metrics — honest aggregates over completed trades.

Kept OUT of Portfolio on purpose: the portfolio owns trading state, this
owns interpretation, and interpretation changes far more often than state.

Ratios that are undefined stay None: a run with no losing trades has no
profit factor, not an infinite one; a run with no trades has no win rate,
not a 0% one. A Sharpe over three days is reported with the fact that it
covers three days attached to it. Faking a number where there isn't one —
or reporting one whose basis you cannot see — is exactly the kind of
flattery this engine exists to refuse.

Money stays int cents until the final jsonable floats; R-multiples come
from the risk the venue ACTUALLY armed (fill-derived stop), recorded on the
TradeRecord — no join against config.

THREE THINGS HERE ARE NOT PERFORMANCE STATS, and they are the reason this
module is worth reading:

- `latency` reports the six `ts_*` stamps every intent carries. They are
  MODELLED in a backtest and MEASURED live, under the same names, so the
  same field answers "what did I assume" and "what did I get".
- `execution` reports fill prices against two different baselines, split by
  entry type, because "slippage" means different things for a market order
  and a resting one (see `TradeRecord.entry_slip_ticks`).
- `groups` reports a second unit of account. One IDEA can be several
  bracket attempts — armed, cancelled, re-entered, filled — and per-trade
  statistics answer the wrong question for that shape.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from datetime import UTC, datetime

from bedivere.core.pricing import InstrumentSpec
from bedivere.core.session_days import SessionDays
from bedivere.engine.portfolio import TradeRecord

# Trading days per year, for annualising a daily Sharpe. Reported alongside
# every Sharpe this module emits — an annualisation factor you cannot see is
# the difference between a 1.4 and a 2.2.
ANNUALISATION_DAYS = 252

# The consecutive stamp pairs a latency block reports, in causal order. The
# last entry is the end-to-end span, which is the one worth alerting on.
_LATENCY_LEGS: tuple[tuple[str, str, str], ...] = (
    ("closeToReceived", "ts_bar_close", "ts_event_received"),
    ("receivedToDecided", "ts_event_received", "ts_decided"),
    ("decidedToSubmitted", "ts_decided", "ts_submitted"),
    ("submittedToAcked", "ts_submitted", "ts_acked"),
    ("ackedToFilled", "ts_acked", "ts_filled"),
    ("closeToFilled", "ts_bar_close", "ts_filled"),
)


def compute_metrics(
    trades: Sequence[TradeRecord],
    spec: InstrumentSpec,
    *,
    days: SessionDays | None = None,
) -> dict[str, object]:
    """Aggregate metrics for one run's trade list. Deterministic; all fields
    JSON-serializable. Sums are exact int cents; ratios are rounded floats
    or None where undefined.

    `days` is optional and only affects the Sharpe basis: with session days
    the daily series is grouped by SESSION day (the unit a futures trader
    actually books), without them by UTC calendar date. Which one was used
    is reported in `sharpeBasis.grouping` rather than assumed.
    """
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
    exit_kinds: dict[str, int] = {}
    for t in trades:
        exit_kinds[t.exit_kind] = exit_kinds.get(t.exit_kind, 0) + 1

    result: dict[str, object] = {
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
        "avgWinCents": _ratio(gross_profit, len(wins)),
        "avgLossCents": _ratio(gross_loss, len(losses)),
        "largestWinCents": max(nets) if nets else None,
        "largestLossCents": min(nets) if nets else None,
        "maxDrawdownCents": max_drawdown,
        # Net over the worst peak-to-trough. The one ratio that answers "was
        # the return worth the ride" without needing a position-size model.
        "netOverMaxDrawdown": _ratio(sum(nets), max_drawdown),
        "longestWinStreak": best_win_streak,
        "longestLossStreak": best_loss_streak,
        "ambiguousFills": sum(1 for t in trades if t.ambiguous),
        "exitKinds": dict(sorted(exit_kinds.items())),
        # The raw stop distances the venue actually armed. R-multiples
        # normalise these away, which is the point of R — but a run whose mean
        # risk drifted from 8 ticks to 40 is a different strategy wearing the
        # same config, and no R-based number will say so.
        "riskTicks": _stats([t.risk_ticks for t in trades if t.risk_ticks]),
        "rMultiples": {
            "known": len(r_values),
            # Expectancy in R is the comparable number across instruments and
            # stop distances — the one to quote when two strategies disagree
            # about what a "point" is worth.
            "expectancy": _round4(sum(r_values) / len(r_values)) if r_values else None,
            "avg": _round4(sum(r_values) / len(r_values)) if r_values else None,
            "best": _round4(max(r_values)) if r_values else None,
            "worst": _round4(min(r_values)) if r_values else None,
            "sum": _round4(sum(r_values)) if r_values else None,
        },
        "latency": _latency_block(trades),
        "execution": _execution_block(trades),
    }
    result.update(_sharpe_block(trades, days))
    groups = _groups_block(trades, spec)
    if groups is not None:
        result["groups"] = groups
    return result


# ---------- latency: the six stamps, finally reported ----------


def _latency_block(trades: Sequence[TradeRecord]) -> dict[str, object]:
    """Distributions over each consecutive stamp pair, in milliseconds.

    A backtest MODELS these (the sim's latency config is the only non-zero
    leg) and a live session MEASURES them, under the same names. Comparing
    the two blocks is how you find out whether the cost model you backtested
    resembles the venue you are trading.

    Legs with no complete pair on any trade are omitted rather than reported
    as empty — a stamp your broker never sets should be visibly absent.
    """
    out: dict[str, object] = {}
    for label, start, end in _LATENCY_LEGS:
        deltas: list[int] = []
        for t in trades:
            a, b = t.stamps.get(start), t.stamps.get(end)
            if a is not None and b is not None:
                deltas.append(b - a)
        stats = _stats(deltas)
        if stats is not None:
            out[label] = stats
    return out


# ---------- execution: fills against two baselines ----------


def _execution_block(trades: Sequence[TradeRecord]) -> dict[str, object]:
    """Fill quality, split by ENTRY TYPE because the two measures below mean
    different things per type and an aggregate over all of them is noise.

    `entrySlipTicks` is the fill against the open of the bar it filled on.
    For a market entry that is the spread crossed — the sim's answer is
    exactly `+half_spread_ticks`, so a live block that reads higher is the
    venue telling you your cost model is optimistic. For a resting entry it
    is not an execution cost at all (see `TradeRecord.entry_slip_ticks`).

    `signalSlipTicks` is the fill against the price the DECISION was made
    at, and it stays meaningful for every entry type. For a stop entry it is
    the trigger-to-fill cost — precisely the cost a fill-derived target
    would have absorbed silently, which is why those strategies fix their
    target instead.
    """
    by_type: dict[str, dict[str, object]] = {}
    for entry_type in sorted({t.entry_type for t in trades}):
        rows = [t for t in trades if t.entry_type == entry_type]
        entry_slip = [v for t in rows if (v := t.entry_slip_ticks()) is not None]
        signal_slip = [v for t in rows if (v := t.signal_slip_ticks()) is not None]
        block: dict[str, object] = {"trades": len(rows)}
        if (stats := _stats(entry_slip)) is not None:
            block["entrySlipTicks"] = stats
        if (stats := _stats(signal_slip)) is not None:
            block["signalSlipTicks"] = stats
        by_type[entry_type] = block
    return {
        "byEntryType": by_type,
        "feesCentsTotal": sum(t.fees_cents for t in trades),
    }


# ---------- Sharpe, with its basis attached ----------


def _sharpe_block(
    trades: Sequence[TradeRecord], days: SessionDays | None
) -> dict[str, object]:
    """Daily-net Sharpe plus the basis it was computed on.

    An undeclared Sharpe is not a number anyone can check: the series, the
    observation count and the annualisation factor all change it, and none
    of them is conventional enough to leave implicit. Fewer than two
    observations has no dispersion, so the ratio is None — not zero.

    The series is net cents per day over days that TRADED. Days with no
    trades are not zero-return days for a strategy that only takes a setup
    when it appears; treating them as such would flatter the dispersion.
    """
    buckets: dict[str, int] = {}
    grouping = "session-day" if days is not None else "utc-date"
    for t in trades:
        key = _day_key(t.exit_ts, days)
        buckets[key] = buckets.get(key, 0) + (t.pnl_cents - t.fees_cents)
    series = [buckets[k] for k in sorted(buckets)]

    sharpe: float | None = None
    if len(series) >= 2:
        spread = statistics.stdev(series)  # sample stdev (n−1)
        if spread > 0:
            sharpe = _round4(
                statistics.fmean(series) / spread * (ANNUALISATION_DAYS**0.5)
            )
    return {
        "sharpeDaily": sharpe,
        "sharpeBasis": {
            "series": "daily-net-cents",
            "grouping": grouping,
            "observations": len(series),
            "annualisationDays": ANNUALISATION_DAYS,
            "dispersion": "sample-stdev",
            "tradelessDaysIncluded": False,
        },
    }


def _day_key(unix_sec: int, days: SessionDays | None) -> str:
    if days is not None:
        day = days.day_containing(unix_sec)
        if day is not None:
            return day.label
    return datetime.fromtimestamp(unix_sec, UTC).strftime("%Y-%m-%d")


# ---------- the second unit of account ----------


def _groups_block(
    trades: Sequence[TradeRecord], spec: InstrumentSpec
) -> dict[str, object] | None:
    """Metrics per IDEA rather than per trade, keyed on `TradeRecord.tag`.

    A strategy that retries an entry produces several brackets for one idea,
    and per-trade statistics then count its bookkeeping as decisions. Three
    attempts for one win:

        win on attempt 3    -1 -1 +3  =  +1R on 3R risked
        win on attempt 1          +3  =  +3R on 1R risked

    `expectancyR` is per IDEA — what one SIGNAL is worth, and the one to
    headline. `expectancyRPerRiskUnit` is per unit of risk DEPLOYED, and is
    the only one that notices those two rows differ. Neither is the top-level
    `rMultiples.expectancy`, which averages ATTEMPTS: row one reads there as
    two losses and a win.

    Returns None when no trade carries a tag — that strategy has not asked
    for this view. `distinct` equal to the trade count is normal: it says
    every idea was a single attempt.
    """
    tagged = [t for t in trades if t.tag]
    if not tagged:
        return None
    nets: dict[str, int] = {}
    counts: dict[str, int] = {}
    r_sums: dict[str, float] = {}
    r_known: dict[str, int] = {}
    for t in tagged:
        nets[t.tag] = nets.get(t.tag, 0) + (t.pnl_cents - t.fees_cents)
        counts[t.tag] = counts.get(t.tag, 0) + 1
        r = t.r_multiple(spec)
        if r is not None:
            r_sums[t.tag] = r_sums.get(t.tag, 0.0) + r
            r_known[t.tag] = r_known.get(t.tag, 0) + 1

    values = [nets[k] for k in sorted(nets)]
    group_wins = sum(1 for v in values if v > 0)
    group_losses = sum(1 for v in values if v < 0)
    r_per_group = [r_sums[k] for k in sorted(r_sums)]
    total_r = sum(r_per_group)
    # Risk DEPLOYED: one unit per FILL that armed a measurable stop, not one
    # per idea. An idea that took three attempts spent three units.
    risk_units = sum(r_known.values())
    return {
        "key": "tag",
        "distinct": len(values),
        "taggedTrades": len(tagged),
        "untaggedTrades": len(trades) - len(tagged),
        "attemptsPerGroup": _stats([counts[k] for k in sorted(counts)]),
        "wins": group_wins,
        "losses": group_losses,
        "flat": len(values) - group_wins - group_losses,
        "winRate": _ratio(group_wins, len(values)),
        "netCents": sum(values),
        "avgNetCents": _ratio(sum(values), len(values)),
        "rMultipleSum": _round4(total_r) if r_per_group else None,
        "expectancyR": _ratio(total_r, len(r_per_group)),
        "groupsWithRisk": len(r_per_group),
        "riskUnits": risk_units,
        "expectancyRPerRiskUnit": _ratio(total_r, risk_units),
    }


# ---------- small shared helpers ----------


def _stats(values: Sequence[float]) -> dict[str, object] | None:
    """min / p50 / mean / max / n, or None for an empty series. Rounded so
    two runs of the same inputs produce byte-identical JSON."""
    if not values:
        return None
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": _round4(ordered[0]),
        "p50": _round4(statistics.median(ordered)),
        "mean": _round4(statistics.fmean(ordered)),
        "max": _round4(ordered[-1]),
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    """None when the denominator is zero — undefined is not zero."""
    if denominator == 0:
        return None
    return _round4(numerator / denominator)


def _round4(x: float) -> float:
    """Round to 4dp, and normalise −0.0 to 0.0 so a sign that carries no
    information cannot change a result hash."""
    r = round(x, 4)
    return 0.0 if r == 0 else r
