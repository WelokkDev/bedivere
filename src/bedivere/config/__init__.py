"""Typed configuration: strategy parameters, the spec envelope, and the
name→object resolution that lets a spec point at YOUR code.

Three layers, each usable on its own:

    base.py     StrategyConfig + load_config + params_hash — the typed tree
    spec.py     RunSpec — one JSON document that fully describes a run
    resolve.py  StrategyPlugin + import-path / entry-point resolution
"""

from bedivere.config.base import (
    ConfigError,
    StrategyConfig,
    apply_overrides,
    canonical_json,
    config_jsonable,
    load_config,
    params_hash,
    parse_override,
)
from bedivere.config.resolve import (
    STRATEGY_ENTRY_POINT_GROUP,
    ResolutionError,
    StrategyPlugin,
    import_ref,
    resolve_factory,
    resolve_strategy,
)
from bedivere.config.spec import (
    RunSpec,
    SpecError,
    archived_spec,
    load_spec,
    parse_spec,
    resolve_backtest_window,
    resolve_instrument,
    resolve_live_window,
    resolve_sessions,
)

__all__ = [
    "STRATEGY_ENTRY_POINT_GROUP",
    "ConfigError",
    "ResolutionError",
    "RunSpec",
    "SpecError",
    "StrategyConfig",
    "StrategyPlugin",
    "apply_overrides",
    "archived_spec",
    "canonical_json",
    "config_jsonable",
    "import_ref",
    "load_config",
    "load_spec",
    "params_hash",
    "parse_override",
    "parse_spec",
    "resolve_backtest_window",
    "resolve_factory",
    "resolve_instrument",
    "resolve_live_window",
    "resolve_sessions",
    "resolve_strategy",
]
