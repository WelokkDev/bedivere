"""Warm-up: declared lookbacks → session-aware load window + readiness gate.

Every component registered with the engine declares how many COMPLETED bars
of which TF it needs before the first tradeable instant. Two pieces:

- `lookback_load_start` turns those declarations into the load-window start
  the composition must fetch history from, by walking the resolved
  session-days BACKWARD from the tradeable start and counting each day's
  expected close-stamps per TF (pure arithmetic on resolved boundaries —
  no session policy here). It answers "how much history to load", assuming
  gapless session data; actual readiness is still checked live.

- `WarmupGate` is the runtime check against ACTUAL view state: ready when
  every requirement's completed count is met. The engine loop suppresses
  strategy callbacks until the tradeable start and HARD-FAILS via
  `assert_ready_for_trading` if that instant arrives with anything unready
  — identical rule in backtest and live.
"""

from __future__ import annotations

from dataclasses import dataclass

from bedivere.core.session_days import SessionDays, expected_close_stamps
from bedivere.core.types import Timeframe
from bedivere.view.market_view import MarketView


class WarmupError(RuntimeError):
    """Warm-up cannot be satisfied (declaration time) or was not satisfied
    when trading was about to start (runtime hard-fail)."""


@dataclass(frozen=True, slots=True)
class WarmupRequirement:
    tf: Timeframe
    bars: int  # completed bars required before the first tradeable instant

    def __post_init__(self) -> None:
        if self.bars < 0:
            raise ValueError("WarmupRequirement.bars must be >= 0")
        if self.tf == Timeframe.D1:
            raise ValueError(
                'WarmupRequirement: "1d" has no intraday chain in the engine'
            )


def lookback_load_start(
    days: SessionDays,
    tradeable_start_unix: int,
    requirements: list[WarmupRequirement],
) -> int:
    """The (exclusive) unix bound to load bars from — pair it with the run
    end in the project's half-open (start, end] query convention — so every
    requirement CAN be complete by `tradeable_start_unix` on gapless data.

    Raises WarmupError when the handed-over days can't cover the lookback:
    silently starting short is exactly the unready-start the standard bans.
    """
    remaining = {r.tf: r.bars for r in requirements if r.bars > 0}
    if not remaining:
        return tradeable_start_unix

    for day in reversed(days.days):
        if day.start_unix >= tradeable_start_unix:
            continue  # day opens at/after the tradeable start: contributes nothing
        for tf in list(remaining):
            if day.end_unix <= tradeable_start_unix:
                contributed = len(expected_close_stamps(day, tf))
            else:
                # The day containing the tradeable start contributes only
                # the close-stamps at or before it.
                contributed = sum(
                    1 for ts in expected_close_stamps(day, tf) if ts <= tradeable_start_unix
                )
            remaining[tf] -= contributed
            if remaining[tf] <= 0:
                del remaining[tf]
        if not remaining:
            return day.start_unix

    shortfall = ", ".join(f"{tf.value}: {n} more bar(s)" for tf, n in remaining.items())
    raise WarmupError(
        f"warm-up lookback exceeds the handed-over session-days (need {shortfall} before {tradeable_start_unix}) — hand over more days or shrink the declared lookbacks"
    )


class WarmupGate:
    """Runtime readiness over actual MarketView state."""

    __slots__ = ("_requirements", "_view")

    def __init__(self, view: MarketView, requirements: list[WarmupRequirement]) -> None:
        for r in requirements:
            if r.tf != view.base_tf and r.tf not in view.derived_tfs:
                raise WarmupError(
                    f'WarmupGate: requirement on "{r.tf.value}" but the view does not maintain it'
                )
        self._view = view
        self._requirements = list(requirements)

    def _count(self, tf: Timeframe) -> int:
        if tf == self._view.base_tf:
            return len(self._view.primary)
        return len(self._view.completed(tf))

    def is_ready(self) -> bool:
        return all(self._count(r.tf) >= r.bars for r in self._requirements)

    def missing(self) -> list[str]:
        """Human-readable shortfalls (empty when ready)."""
        out: list[str] = []
        for r in self._requirements:
            have = self._count(r.tf)
            if have < r.bars:
                out.append(f"{r.tf.value}: have {have}/{r.bars} completed bars")
        return out

    def assert_ready_for_trading(self, ts_unix: int) -> None:
        """The hard-fail: called by the loop when the first tradeable
        instant arrives. An unready component at this point is a data or
        composition bug — refuse to trade rather than degrade."""
        shortfalls = self.missing()
        if shortfalls:
            raise WarmupError(
                f"first tradeable instant {ts_unix} arrived unready: {'; '.join(shortfalls)}"
            )
