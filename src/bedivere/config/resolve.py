"""Resolves the string names in a spec to objects: the strategy plugin, the
candle source, the feed adapter, the broker adapter.

Two mechanisms, so a strategy never has to live inside this library:

    "examples.sma_cross:PLUGIN"   an import path: module, then attribute
    "sma_cross"                   an entry point in group "bedivere.strategies"

The import path needs nothing installed — `python -m bedivere.backtest` puts
the working directory on `sys.path`, so a module in your own repo resolves
as-is. Entry points are for strategies you package; declare one under

    [project.entry-points."bedivere.strategies"]
    my_strategy = "my_pkg.strategy:PLUGIN"

A `StrategyPlugin` carries the contract: the `kind` its config declares, the
config model to validate against, and a factory from validated config to
Strategy. Warm-up and the optional seams derive from the CONFIG rather than
the spec, because a lookback computed from a parameter cannot be a constant
in a JSON file without drifting from the code that reads it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from importlib.metadata import entry_points
from typing import Any, cast

from bedivere.config.base import StrategyConfig
from bedivere.core.pricing import InstrumentSpec
from bedivere.core.session_days import SessionDays
from bedivere.core.types import Timeframe
from bedivere.engine.loop import CloseSink, Strategy
from bedivere.engine.warmup import WarmupRequirement
from bedivere.streams.sparse import TriggerRule
from bedivere.view.market_view import ObserverFactory

STRATEGY_ENTRY_POINT_GROUP = "bedivere.strategies"


class ResolutionError(ValueError):
    """A spec named something that could not be imported, found, or used."""


@dataclass(frozen=True, slots=True)
class StrategyContext:
    """What a strategy factory is told about the run it is being built for.

    The instrument in particular: a strategy that computes a level in ticks
    needs the grid at CONSTRUCTION time, and digging it out of the first bar's
    `ctx.portfolio.spec` is a workaround for not having been told. Everything
    here is settled before the first bar and immutable afterwards — anything
    that changes during the run reaches the strategy through `RunContext`.
    """

    symbol: str
    instrument: InstrumentSpec
    base_timeframe: Timeframe
    derived_timeframes: tuple[Timeframe, ...]
    days: SessionDays


def _no_warmup(_cfg: StrategyConfig) -> Sequence[WarmupRequirement]:
    return ()


@dataclass(frozen=True, slots=True)
class StrategyPlugin[C: StrategyConfig]:
    """What a spec resolves a strategy name to.

    `kind` must match the `kind` in the spec's config block — the check is
    the one that catches a spec pointing at strategy A with strategy B's
    parameters, which otherwise validates cleanly right up to the first
    trade.

    `warmup` takes the RESOLVED config because lookbacks are usually derived
    from parameters; the composition uses it both to size the history load
    and to arm the run's readiness gate.
    """

    kind: str
    config_model: type[C]
    build: Callable[[C, StrategyContext], Strategy]
    warmup: Callable[[C], Sequence[WarmupRequirement]] = _no_warmup
    close_sink: Callable[[C], CloseSink | None] = field(default=lambda _cfg: None)
    observers: Callable[[C], ObserverFactory | None] = field(default=lambda _cfg: None)
    # The four below are optional seams; see docs/writing-a-strategy.md.
    #
    # The arming condition over COARSE bars only, enabling mixed-fidelity
    # replay. Publishing one is a CLAIM only the strategy's author can make,
    # which is why it lives here and not in a spec: a spec asking for
    # `replay.mode: sparse` against a plugin without one fails rather than
    # guessing.
    trigger_rule: Callable[[C], TriggerRule | None] = field(default=lambda _cfg: None)
    # The journal kinds this strategy emits — a typo net, so `--notify-on
    # entry_submited` fails at startup instead of alerting on nothing all
    # session. Empty (the default) disables the check.
    notify_kinds: Callable[[C], Iterable[str]] = field(default=lambda _cfg: ())
    # Extra runs-index columns. bedivere cannot know which of YOUR knobs are
    # worth sorting by, so by default it indexes none. Keep them FLAT (nested
    # objects are not sortable) and stable (an index is append-only history,
    # so a rename splits it into two shapes).
    index_columns: Callable[[C, dict[str, Any]], dict[str, Any]] = field(
        default=lambda _cfg, _result: {}
    )
    # Base timeframes this strategy can honestly run on; None means any.
    # Declare a set when a coarser base would change what the strategy IS
    # rather than how precisely it is measured — an intrabar strategy given a
    # base as coarse as its signal candle never sees an intrabar path, and
    # still produces a plausible run.
    base_timeframes: Callable[[C], Sequence[Timeframe] | None] = field(
        default=lambda _cfg: None
    )


# ---------- generic reference resolution ----------


def import_ref(ref: str, *, what: str) -> object:
    """Resolve `"package.module:attribute"` to the attribute itself.

    Errors name which half failed — a missing module and a missing attribute
    are different mistakes with different fixes, and "cannot resolve X" tells
    you neither.
    """
    module_name, sep, attr = ref.partition(":")
    if not sep or not module_name or not attr:
        raise ResolutionError(
            f'{what} "{ref}" must look like "package.module:attribute"'
        )
    try:
        module = import_module(module_name)
    except ImportError as e:
        raise ResolutionError(
            f'{what} "{ref}": cannot import module "{module_name}" ({e}) — '
            "is it on sys.path? (running from the repo root usually is enough)"
        ) from e
    try:
        return getattr(module, attr)
    except AttributeError as e:
        raise ResolutionError(
            f'{what} "{ref}": module "{module_name}" has no attribute "{attr}"'
        ) from e


def resolve_strategy(ref: str) -> StrategyPlugin[Any]:
    """A spec's `strategy` field → the plugin it names.

    A bare name goes to the entry-point group; anything containing ":" is an
    import path. Nothing else is accepted: a name that is neither installed
    nor importable must fail here, loudly, rather than at the first bar.
    """
    obj = _entry_point_object(ref) if ":" not in ref else import_ref(ref, what="strategy")
    if not isinstance(obj, StrategyPlugin):
        raise ResolutionError(
            f'strategy "{ref}" resolved to {type(obj).__name__}, not a StrategyPlugin — '
            "export a StrategyPlugin(kind=..., config_model=..., build=...)"
        )
    # The parameter is erased crossing the import boundary — a string cannot
    # carry it. `plugin.config_model` is what re-establishes the type, by
    # validating the spec's config block into a real instance.
    return cast(StrategyPlugin[Any], obj)


def _entry_point_object(name: str) -> object:
    found = entry_points(group=STRATEGY_ENTRY_POINT_GROUP, name=name)
    if not found:
        available = sorted(ep.name for ep in entry_points(group=STRATEGY_ENTRY_POINT_GROUP))
        raise ResolutionError(
            f'strategy "{name}" is not registered under the "{STRATEGY_ENTRY_POINT_GROUP}" '
            f"entry-point group (installed: {available or 'none'}) — either install a package "
            'that declares it, or name it as an import path "module:attribute"'
        )
    return next(iter(found)).load()


def resolve_factory(ref: str, *, what: str) -> Callable[..., Any]:
    """An import path that must resolve to something callable (a source, feed
    or broker factory). The callable's own signature is checked by calling
    it — a factory that refuses its options says so in its own words."""
    obj = import_ref(ref, what=what)
    if not callable(obj):
        raise ResolutionError(f'{what} "{ref}" resolved to {type(obj).__name__}, which is not callable')
    return obj
