"""Shared CLI machinery: spec → composition, and the boundary conventions.

Both runners do the same six things before they diverge, and they do them in
this order for a reason — each step can refuse the run, and the cheapest
refusals come first:

    1. parse the spec envelope          (a typo'd key fails here)
    2. resolve the strategy plugin      (a name nobody can import fails here)
    3. validate + override the config   (a typo'd --set path fails here)
    4. expand the session calendar
    5. build the instrument
    6. size the warm-up lookback

Only then does anything touch data or a venue.

BOUNDARY CONVENTIONS, held by every command in this package:

  - result JSON goes to STDOUT, everything else to STDERR. `... | jq` works
    while the progress commentary still reaches you.
  - exit 0 success, 1 failure, 130 hard abort (Ctrl+C).
  - a failure prints one line naming what was wrong, not a traceback. The
    traceback is for bedivere's bugs; a bad spec is yours, and telling you
    which path was bad is more useful than telling you which frame raised.
  - `./.env` is loaded before the command runs, and never over a variable the
    environment already holds.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Callable, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bedivere.config.base import (
    ConfigError,
    StrategyConfig,
    config_jsonable,
    load_config,
    params_hash,
)
from bedivere.config.resolve import (
    ResolutionError,
    StrategyContext,
    StrategyPlugin,
    resolve_strategy,
)
from bedivere.config.spec import (
    RunSpec,
    SpecError,
    load_spec,
    resolve_instrument,
    resolve_sessions,
)
from bedivere.core.pricing import InstrumentSpec
from bedivere.core.session_days import SessionDays
from bedivere.core.types import Candle, Timeframe
from bedivere.data.build import build_candle_source
from bedivere.data.port import CandleSource, assess_coverage
from bedivere.data.source import BarSourceError, require_timeframes
from bedivere.engine.loop import Strategy
from bedivere.engine.warmup import WarmupRequirement, lookback_load_start
from bedivere.notify.discord import DiscordNotifier
from bedivere.notify.port import ConsoleNotifier, Notifier

DISCORD_WEBHOOK_ENV = "BEDIVERE_DISCORD_WEBHOOK"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_ABORTED = 130


class CliError(RuntimeError):
    """Anything the operator can fix: a bad spec, a missing file, a refused
    run. Rendered as one stderr line and exit 1 — never a traceback."""


def note(text: str) -> None:
    """Progress commentary. Stderr, because stdout is the result."""
    sys.stderr.write(f"{text}\n")


# ---------- argument groups ----------


def add_spec_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--spec", type=Path, required=True, help="spec envelope JSON describing the run"
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="dotted.path=value",
        help="config override, repeatable (e.g. --set risk.rr=2.5). Applied to the "
        "RESOLVED config, so any schema path works; a path outside the schema fails.",
    )
    parser.add_argument(
        "--out",
        dest="runs_dir",
        type=Path,
        default=Path("runs"),
        help="runs root: one directory per run inside it, plus index.jsonl and the "
        "one-run lock (default: runs)",
    )
    parser.add_argument("--note", default="", help="label recorded in the run's index line")


# ---------- spec -> composition ----------


@dataclass(frozen=True, slots=True)
class ResolvedRun:
    """Everything both runners need, resolved and validated, before any data
    is read or any venue is contacted."""

    spec: RunSpec
    plugin: StrategyPlugin[Any]
    config: StrategyConfig
    config_json: dict[str, Any]
    params_hash: str
    days: SessionDays
    instrument: InstrumentSpec
    warmup: tuple[WarmupRequirement, ...]
    overrides: tuple[str, ...]

    @property
    def kind(self) -> str:
        return self.plugin.kind

    @property
    def symbol(self) -> str:
        return self.spec.symbol

    @property
    def base_tf(self) -> Timeframe:
        return self.spec.base_timeframe

    def strategy(self) -> Strategy:
        """Build the run's strategy. Called once, at the composition root —
        a Strategy is stateful and single-use, so this is deliberately a
        method rather than a field on a frozen dataclass."""
        return self.plugin.build(
            self.config,
            StrategyContext(
                symbol=self.symbol,
                instrument=self.instrument,
                base_timeframe=self.base_tf,
                derived_timeframes=self.spec.derived_timeframes,
                days=self.days,
            ),
        )


def _check_base_timeframe(
    plugin: StrategyPlugin[Any], config: StrategyConfig, base_tf: Timeframe
) -> None:
    """Refuse a base timeframe the strategy says it cannot honestly run on.

    Most strategies declare nothing and accept anything. The ones that do
    declare are the ones a coarser base would silently REDEFINE rather than
    merely measure less precisely: an intrabar strategy handed a base as
    coarse as its own signal candle never sees an intrabar path, arms
    nothing, and produces a clean-looking run of zero trades — or worse, a
    handful matched at the wrong resolution.

    Refusing by name, here, is the cheapest possible failure: before any bar
    is loaded, naming the timeframes that would have worked.
    """
    allowed = plugin.base_timeframes(config)
    if allowed is None or base_tf in allowed:
        return
    raise CliError(
        f'strategy "{plugin.kind}" does not run on a {base_tf.value} base — it declares '
        f"{', '.join(tf.value for tf in allowed)}. A base this coarse would change what "
        "the strategy is, not just how precisely it is measured."
    )


def resolve_run(
    spec_path: Path, overrides: Sequence[str], *, now_unix: int | None = None
) -> ResolvedRun:
    """Steps 1–6 of the header. `now_unix` is only used to expand a session
    calendar that declares no date range — the live case."""
    try:
        spec = load_spec(spec_path)
        plugin = resolve_strategy(spec.strategy)
        declared = spec.config.get("kind")
        if declared != plugin.kind:
            raise CliError(
                f'spec.config.kind is "{declared}" but strategy "{spec.strategy}" '
                f'registers kind "{plugin.kind}" — the spec is pointing at the wrong strategy '
                "(or the wrong parameters)"
            )
        config = load_config(spec.config, {plugin.kind: plugin.config_model}, list(overrides))
        _check_base_timeframe(plugin, config, spec.base_timeframe)
        days = resolve_sessions(spec, now_unix=now_unix)
        instrument = resolve_instrument(spec)
    except (SpecError, ConfigError, ResolutionError, ValueError) as e:
        raise CliError(str(e)) from e

    return ResolvedRun(
        spec=spec,
        plugin=plugin,
        config=config,
        config_json=config_jsonable(config),
        params_hash=params_hash(config),
        days=days,
        instrument=instrument,
        warmup=tuple(plugin.warmup(config)),
        overrides=tuple(overrides),
    )


def history_start(run: ResolvedRun, window_start: int) -> int:
    """The exclusive lower bound bars must be loaded from: far enough back
    that every declared lookback CAN be complete by the first tradeable
    instant. Refuses rather than starting short — an unready run is the one
    failure mode a warm-up gate exists to prevent."""
    try:
        return lookback_load_start(run.days, window_start, list(run.warmup))
    except ValueError as e:
        raise CliError(str(e)) from e


def build_source(run: ResolvedRun, *, timeframe: Timeframe | None = None) -> CandleSource:
    """The run's candle source. `timeframe` defaults to the base TF; a sparse
    replay also asks for one at its coarse TF — same store, different key."""
    if run.spec.data is None:
        raise CliError(
            "this spec has no `data` block, so there are no bars to run on — add one "
            '(e.g. {"source": "lake", "dataset": "GLBX.MDP3", "series": "local.v.0"} '
            'or {"source": "csv", "path": "data/DEMO-5m.csv"})'
        )
    tf = timeframe or run.base_tf
    try:
        source = build_candle_source(run.spec.data, symbol=run.symbol, timeframe=tf, days=run.days)
        # The cheapest possible refusal: "this store has no NQ 1s" costs one
        # directory listing, and finding it out an hour later instead looks like
        # a strategy that took no trades.
        require_timeframes(source, run.symbol, [tf])
    except (BarSourceError, ResolutionError, ValueError) as e:
        raise CliError(str(e)) from e
    return source


def load_bars(
    source: CandleSource,
    run: ResolvedRun,
    *,
    start_unix: int,
    end_unix: int,
    require_coverage: bool,
    timeframe: Timeframe | None = None,
) -> list[Candle]:
    """Fetch bars and REPORT COVERAGE. A hole in the range is stated out loud
    every time; `--require-coverage` promotes it to a refusal.

    Warning by default is the honest middle: real data has holidays the
    calendar did not list and sessions the exchange shortened, and a tool that
    refused all of them would be a tool nobody runs. A tool that never
    mentioned them would let a data outage masquerade as a strategy result.
    """
    tf = timeframe or run.base_tf
    try:
        bars = source.candles(run.symbol, tf, start_unix, end_unix)
    except BarSourceError as e:
        raise CliError(str(e)) from e
    except (OSError, ValueError) as e:
        raise CliError(f"{source.describe()}: {e}") from e

    coverage = assess_coverage(
        bars,
        days=run.days,
        symbol=run.symbol,
        timeframe=tf,
        start_unix=start_unix,
        end_unix=end_unix,
    )
    if coverage.complete:
        note(f"data: {coverage.describe()} — complete ({source.describe()})")
    elif require_coverage:
        raise CliError(f"incomplete data: {coverage.describe()} ({source.describe()})")
    else:
        note(f"data: ⚠ {coverage.describe()} ({source.describe()})")

    if not bars:
        raise CliError(
            f"no bars for {run.symbol} {tf.value} in "
            f"({start_unix}, {end_unix}] from {source.describe()}"
        )
    return bars


# ---------- notifications ----------


def build_notifier(transport: str) -> Notifier | None:
    """`auto` picks Discord when a webhook is configured, console otherwise.
    `off` returns None, which is what run_live reads as "no router at all"
    rather than a router that drops everything."""
    webhook = os.environ.get(DISCORD_WEBHOOK_ENV, "")
    if transport == "off":
        return None
    if transport == "console":
        return ConsoleNotifier(prefix="[alert]")
    if transport == "discord" or (transport == "auto" and webhook):
        if not webhook:
            raise CliError(
                f"--notify discord needs {DISCORD_WEBHOOK_ENV} in the environment or ./.env"
            )
        return DiscordNotifier(webhook)
    return ConsoleNotifier(prefix="[alert]")


# ---------- environment ----------


DOTENV_FILE = Path(".env")
"""Relative, so it is read from the working directory — where uv, Docker Compose
and most dotenv loaders look. Module-level so the test suite can point it away
from a real `.env` holding real keys."""

_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def load_dotenv(
    path: Path | None = None, *, environ: MutableMapping[str, str] = os.environ
) -> None:
    """Copy `KEY=VALUE` lines from `./.env` into the environment.

    A variable the environment already holds WINS, so
    `DATABENTO_API_KEY=... bedivere-data cost ...` still overrides the file for
    one command. A missing file is not an error.

    The small common subset, rather than a dependency: blank lines and `#`
    comments, an optional `export ` prefix, one pair of matching quotes around a
    value, and ` #` ending an unquoted one. No interpolation, no escapes, no
    multi-line values. A line outside that subset is skipped with a warning that
    names its line number and never its content, which is likely a secret.
    """
    target = DOTENV_FILE if path is None else path
    try:
        # utf-8-sig: an editor that writes a BOM would otherwise corrupt the first key.
        text = target.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, value = line.removeprefix("export ").partition("=")
        name = name.strip()
        if not sep or not _ENV_NAME.fullmatch(name):
            note(f"warning: {target}:{number} is not KEY=VALUE — skipped")
            continue
        if name not in environ:
            environ[name] = _dotenv_value(value.strip())


def _dotenv_value(value: str) -> str:
    if value[:1] in ('"', "'"):
        closing = value.find(value[0], 1)
        if closing > 0:
            return value[1:closing]
    comment = value.find(" #")
    return value if comment < 0 else value[:comment].rstrip()


# ---------- boundary ----------


def emit_result(result: dict[str, Any]) -> None:
    """The result envelope, to stdout, newline-terminated so a shell prompt
    lands on its own line."""
    sys.stdout.write(json.dumps(result, sort_keys=True, indent=2) + "\n")


def run_command(body: Callable[[], int]) -> int:
    """The boundary every command shares: `./.env` is loaded, then every
    outcome becomes an exit code."""
    try:
        load_dotenv()
        return body()
    except KeyboardInterrupt:
        note("\naborted (hard)")
        return EXIT_ABORTED
    except CliError as e:
        note(f"error: {e}")
        return EXIT_FAILED
    except Exception as e:  # noqa: BLE001 — CLI boundary: everything becomes exit 1
        note(f"error: {type(e).__name__}: {e}")
        return EXIT_FAILED
