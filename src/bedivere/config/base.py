"""Strategy config: a typed frozen tree, dotted-path overrides, one hash.

    JSON → registry lookup by `kind` → validate (defaults MATERIALIZE)
         → dotted-path overrides → re-validate → canonicalize → params_hash

A typo'd key in an untyped config produces a run that succeeds under a
different `paramsHash` and looks like a legitimate variant of the experiment
you meant. Two rules prevent that:

  - `extra="forbid"` at EVERY level. A typo'd key is a validation error, not
    a silent no-op. Nested models must inherit `StrategyConfig` (or repeat
    its `model_config`) or the guarantee stops at the top level.
  - Overrides run against the RESOLVED tree, never the raw input. A minimal
    config `{"kind": "x"}` is therefore fully sweepable — `--set risk.rr=2.5`
    reaches a path the input never mentioned — while a path outside the
    schema still fails, because the re-validation forbids the extra key it
    would have created.

`params_hash` covers the resolved dump, so two runs share a hash iff every
knob matches, including the ones nobody typed. It is the 16-hex prefix of
`bedivere.run.record.canonical_sha256` over the same dump — the short form
names run directories, the full form rides in the result envelope.

The typed config is an UPGRADE, not a requirement: `run_backtest(params=...)`
still takes a plain dict for callers who want one.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, ValidationError

# The short id length. 16 hex = 64 bits: collision-free for any realistic
# sweep, and short enough to read in a directory listing.
_HASH_CHARS = 16


class StrategyConfig(BaseModel):
    """Base for every strategy config tree. Frozen after validation, unknown
    keys rejected at every level.

    Subclasses declare `kind: Literal["<name>"]`; the base deliberately does
    not, so the Literal narrowing stays clean and a subclass cannot inherit
    somebody else's discriminator by accident.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class ConfigError(ValueError):
    """A malformed config, an unknown `kind`, or a bad override expression."""


def parse_override(expr: str) -> tuple[list[str], object]:
    """One `--set` expression: `"risk.rr=2.5"` → `(["risk", "rr"], 2.5)`.

    The value is parsed as JSON when it parses, and left as a bare string
    when it does not — so `--set exit_style=trailing` needs no shell quoting
    while `--set timeframes=["5m","15m"]` still gets a real list.
    """
    path, sep, raw = expr.partition("=")
    if not sep or not path.strip():
        raise ConfigError(f'override "{expr}" must look like dotted.path=value')
    keys = [k.strip() for k in path.strip().split(".")]
    if any(not k for k in keys):
        raise ConfigError(f'override "{expr}" has an empty path segment')
    try:
        value: object = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return keys, value


def apply_overrides(data: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply dotted-path overrides to a plain JSON-clean dict.

    Intermediate containers must already exist — which is exactly why
    `load_config` materializes the model's defaults FIRST. Inventing missing
    objects here would let `--set rsik.rr=2` create a `rsik` block that
    validation then has to catch; refusing at the traversal names the
    segment that was wrong instead.
    """
    out: dict[str, Any] = json.loads(json.dumps(data))  # deep copy, JSON-clean
    for expr in overrides:
        keys, value = parse_override(expr)
        node: dict[str, Any] = out
        for k in keys[:-1]:
            nxt: Any = node.get(k)
            if not isinstance(nxt, dict):
                raise ConfigError(
                    f'override "{expr}": "{k}" is not an object in the resolved config'
                )
            node = cast(dict[str, Any], nxt)
        node[keys[-1]] = value
    return out


def load_config[C: StrategyConfig](
    data: Mapping[str, Any],
    registry: Mapping[str, type[C]],
    overrides: Sequence[str] | None = None,
) -> C:
    """dict → validate → overrides → re-validate, dispatched on `kind`.

    The double validation is the load-bearing part: the first pass fills in
    defaults so every schema path exists to be overridden, the second pass
    is what makes a typo'd override path fail (the key it wrote is `extra`).
    """
    kind = data.get("kind")
    if not isinstance(kind, str) or kind not in registry:
        raise ConfigError(
            f"config.kind must be one of {sorted(registry)} (got {kind!r})"
        )
    model = registry[kind]
    base = _validate(model, dict(data), what="config")
    if not overrides:
        return base
    merged = apply_overrides(base.model_dump(mode="json"), overrides)
    return _validate(model, merged, what="config (after --set)")


def _validate[C: StrategyConfig](
    model: type[C], data: dict[str, Any], *, what: str
) -> C:
    """Validate, turning pydantic's error list into one message that names
    each offending path — `risk.rr: Input should be a valid number` beats a
    traceback ending in "1 validation error"."""
    try:
        return model.model_validate(data)
    except ValidationError as e:
        raise ConfigError(f"{what}: {format_validation_error(e)}") from e


def format_validation_error(error: ValidationError) -> str:
    """Render a pydantic ValidationError as `path: message` lines. Kept
    public because the spec loader renders its own errors the same way."""
    lines: list[str] = []
    for item in error.errors():
        loc = ".".join(str(part) for part in item["loc"]) or "<root>"
        lines.append(f"{loc}: {item['msg']}")
    return "; ".join(lines)


def config_jsonable(cfg: StrategyConfig) -> dict[str, Any]:
    """The resolved parameter set as plain JSON data — what rides into the
    result envelope under `"params"` and into the archived spec."""
    return cfg.model_dump(mode="json")


def canonical_json(cfg: StrategyConfig) -> str:
    """The canonical form the hash covers: resolved dump, sorted keys, tight
    separators. Byte-identical to what `canonical_sha256` hashes."""
    return json.dumps(config_jsonable(cfg), sort_keys=True, separators=(",", ":"))


def params_hash(cfg: StrategyConfig) -> str:
    """Stable short hex id of the FULL resolved parameter set. Two runs share
    it iff every knob matches, defaulted ones included."""
    return hashlib.sha256(canonical_json(cfg).encode("utf-8")).hexdigest()[:_HASH_CHARS]
