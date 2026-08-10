"""Name → object resolution: can a spec reach code the library has never
heard of, and does it say something useful when it cannot?
"""

from __future__ import annotations

import pytest

from bedivere.config.resolve import (
    STRATEGY_ENTRY_POINT_GROUP,
    ResolutionError,
    StrategyPlugin,
    import_ref,
    resolve_factory,
    resolve_strategy,
)


def test_an_import_path_resolves_to_the_plugin() -> None:
    plugin = resolve_strategy("tests.strategy_fixture:PLUGIN")
    assert isinstance(plugin, StrategyPlugin)
    assert plugin.kind == "ramp"


def test_a_malformed_reference_says_what_the_shape_should_be() -> None:
    with pytest.raises(ResolutionError, match="package.module:attribute"):
        import_ref("tests.strategy_fixture", what="strategy")


def test_a_missing_module_and_a_missing_attribute_are_different_mistakes() -> None:
    """Different fixes, so different messages — "cannot resolve X" tells you
    neither."""
    with pytest.raises(ResolutionError, match="cannot import module"):
        import_ref("no_such_pkg.nowhere:PLUGIN", what="strategy")
    with pytest.raises(ResolutionError, match="has no attribute"):
        import_ref("tests.strategy_fixture:NOPE", what="strategy")


def test_something_that_is_not_a_plugin_is_refused() -> None:
    with pytest.raises(ResolutionError, match="not a StrategyPlugin"):
        resolve_strategy("tests.strategy_fixture:NOT_A_PLUGIN")


def test_a_bare_name_goes_to_the_entry_point_group() -> None:
    """A packaged strategy registers under an entry-point group; an
    unregistered bare name must name the group and list what IS installed,
    because "not found" is where a cloner gets stuck."""
    with pytest.raises(ResolutionError, match=STRATEGY_ENTRY_POINT_GROUP):
        resolve_strategy("definitely_not_installed")


def test_a_factory_must_be_callable() -> None:
    assert callable(resolve_factory("tests.strategy_fixture:build_ramp_source", what="data.factory"))
    with pytest.raises(ResolutionError, match="not callable"):
        resolve_factory("tests.strategy_fixture:NOT_A_PLUGIN", what="data.factory")
