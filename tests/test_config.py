"""Strategy config: typos fail, overrides reach everything, the hash is the
resolved tree.

The three properties worth pinning are the three that make a sweep
trustworthy — a knob you misspelled must not run, a knob you never mentioned
must still be reachable, and two runs must share a hash only when every knob
matches.
"""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import ValidationError

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
from bedivere.run.record import canonical_sha256


class Risk(StrategyConfig):
    rr: float = 1.5
    max_risk_ticks: int = 20
    exit_style: str = "fixed"


class Demo(StrategyConfig):
    kind: Literal["demo"] = "demo"
    qty: int = 1
    risk: Risk = Risk()
    timeframes: tuple[str, ...] = ("5m",)


REGISTRY: dict[str, type[Demo]] = {"demo": Demo}


# ---------- parse_override ----------


def test_override_parses_json_values_and_falls_back_to_strings() -> None:
    assert parse_override("risk.rr=2.5") == (["risk", "rr"], 2.5)
    assert parse_override("qty=3") == (["qty"], 3)
    assert parse_override('timeframes=["5m","15m"]') == (["timeframes"], ["5m", "15m"])
    # Unquoted words stay strings — a command line should not need JSON quoting
    # for the common case.
    assert parse_override("risk.exit_style=trailing") == (["risk", "exit_style"], "trailing")


@pytest.mark.parametrize("expr", ["norhs", "=2", " =2", "a..b=2"])
def test_malformed_overrides_are_refused(expr: str) -> None:
    with pytest.raises(ConfigError):
        parse_override(expr)


def test_apply_overrides_refuses_to_invent_containers() -> None:
    with pytest.raises(ConfigError, match="not an object"):
        apply_overrides({"qty": 1}, ["qty.rr=2"])


# ---------- load_config ----------


def test_unknown_kind_names_what_is_registered() -> None:
    with pytest.raises(ConfigError, match=r"\['demo'\]"):
        load_config({"kind": "nope"}, REGISTRY)


def test_a_typo_fails_instead_of_silently_no_opping() -> None:
    with pytest.raises(ConfigError, match="qtyy"):
        load_config({"kind": "demo", "qtyy": 4}, REGISTRY)


def test_a_typo_nested_one_level_down_also_fails() -> None:
    """extra="forbid" has to hold at EVERY level or the guarantee is
    decorative."""
    with pytest.raises(ConfigError, match="risk.rrr"):
        load_config({"kind": "demo", "risk": {"rrr": 2.0}}, REGISTRY)


def test_overrides_reach_paths_the_input_never_mentioned() -> None:
    """The whole reason validation runs twice: defaults materialize first, so
    a minimal config is still fully sweepable."""
    cfg = load_config({"kind": "demo"}, REGISTRY, ["risk.rr=3.0", "risk.exit_style=trailing"])
    assert cfg.risk.rr == 3.0
    assert cfg.risk.exit_style == "trailing"
    assert cfg.qty == 1  # untouched default survived


def test_an_override_outside_the_schema_fails_loudly() -> None:
    with pytest.raises(ConfigError, match="after --set"):
        load_config({"kind": "demo"}, REGISTRY, ["risk.rrr=3.0"])


def test_a_wrongly_typed_override_names_the_path() -> None:
    with pytest.raises(ConfigError, match="risk.rr"):
        load_config({"kind": "demo"}, REGISTRY, ["risk.rr=not_a_number"])


def test_the_config_is_frozen() -> None:
    cfg = load_config({"kind": "demo"}, REGISTRY)
    with pytest.raises(ValidationError):
        cfg.qty = 9  # pyright: ignore[reportAttributeAccessIssue]


# ---------- hashing ----------


def test_the_hash_covers_defaulted_knobs_too() -> None:
    """Two runs share a paramsHash iff every knob matches — including the
    ones nobody typed."""
    minimal = load_config({"kind": "demo"}, REGISTRY)
    spelled_out = load_config(
        {"kind": "demo", "qty": 1, "risk": {"rr": 1.5, "max_risk_ticks": 20, "exit_style": "fixed"}},
        REGISTRY,
    )
    assert params_hash(minimal) == params_hash(spelled_out)


def test_an_override_changes_the_hash() -> None:
    base = load_config({"kind": "demo"}, REGISTRY)
    swept = load_config({"kind": "demo"}, REGISTRY, ["risk.rr=2.5"])
    assert params_hash(base) != params_hash(swept)


def test_the_short_hash_is_the_prefix_of_the_envelope_hash() -> None:
    """The run directory is named with the short form and result.json carries
    the long one. They must be the same hash or the archive lies about which
    run produced which result."""
    cfg = load_config({"kind": "demo"}, REGISTRY, ["qty=2"])
    assert canonical_sha256(config_jsonable(cfg)).startswith(params_hash(cfg))


def test_canonical_json_is_stable_and_sorted() -> None:
    cfg = load_config({"kind": "demo"}, REGISTRY)
    text = canonical_json(cfg)
    assert text == canonical_json(load_config({"kind": "demo"}, REGISTRY))
    assert text.startswith('{"kind":')  # sorted keys, tight separators
