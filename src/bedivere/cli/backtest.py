"""`python -m bedivere.backtest` — run a spec, archive the result.

    python -m bedivere.backtest --spec specs/sma_cross.json --out runs/
    python -m bedivere.backtest --spec specs/sma_cross.json --set rr=2.5

The composition is `bedivere.run.backtest.run_backtest`; this module's whole
job is turning a JSON document into its arguments and putting the answer
somewhere you can find it again. Nothing strategy-shaped appears here — the
strategy, its config model, its warm-up requirements and its bars all arrive
through names the spec chose.

`--set` reaches any path in the RESOLVED config, so a sweep is a shell loop
over one spec file; each override changes the config, which changes
`paramsHash`, which changes the run directory. A path outside the schema is
a validation error, not a new key.

Determinism is unchanged by any of this: the archived `result.json` carries
no wall-clock field, and the run's timestamp lives in the directory name and
the index line. Run the same spec twice and diff the two result.json files —
they are byte-identical.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from bedivere.cli.common import (
    CliError,
    ResolvedRun,
    add_spec_arguments,
    build_source,
    emit_result,
    history_start,
    load_bars,
    note,
    resolve_run,
    run_command,
)
from bedivere.config.spec import SpecError, archived_spec, resolve_backtest_window
from bedivere.core.clock import LiveClock
from bedivere.core.types import Candle
from bedivere.data.port import CandleSource
from bedivere.run.archive import run_id_for, utc_stamp, write_run
from bedivere.run.backtest import run_backtest
from bedivere.streams.sparse import SparseReplayStream


def _sparse_stream(
    run: ResolvedRun,
    fine_source: CandleSource,
    *,
    load_start: int,
    end_unix: int,
    require_coverage: bool,
) -> SparseReplayStream:
    """Build the mixed-fidelity stream a `replay.mode: sparse` spec asks for.

    The coarse series gets its own `CandleSource` at the coarse timeframe —
    the same store, a different key — and its coverage is reported like any
    other load, because a hole in the COARSE series is worse than a hole in
    the fine one: it silently removes windows the run would have descended
    into, and the result looks like a strategy that simply found fewer
    setups.
    """
    replay = run.spec.replay
    coarse_tf = replay.coarse_timeframe
    assert coarse_tf is not None  # ReplaySpec validates this pairing
    rule = run.plugin.trigger_rule(run.config)
    if rule is None:
        raise CliError(
            f'this spec asks for replay.mode "sparse", but the strategy plugin for '
            f'"{run.kind}" publishes no trigger_rule. Only the strategy\'s author can '
            "declare that its arming condition is decidable from coarse bars alone — "
            "see docs/backtest-fidelity.md"
        )
    if coarse_tf not in run.spec.derived_timeframes:
        raise CliError(
            f'replay.coarseTimeframe "{coarse_tf.value}" must also appear in '
            "derivedTimeframes — the agreement guard reads that series back off the view"
        )

    coarse_source = build_source(run, timeframe=coarse_tf)
    coarse_bars = load_bars(
        coarse_source,
        run,
        start_unix=load_start,
        end_unix=end_unix,
        require_coverage=require_coverage,
        timeframe=coarse_tf,
    )
    return SparseReplayStream(
        symbol=run.symbol,
        fine_timeframe=run.base_tf,
        coarse_timeframe=coarse_tf,
        coarse_bars=coarse_bars,
        rule=rule,
        fine_loader=lambda a, b: fine_source.candles(run.symbol, run.base_tf, a, b),
        days=run.days,
        window_seconds=replay.window_seconds,
        carry_across_sessions=replay.carry_across_sessions,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bedivere.backtest",
        description="Replay a spec and archive the run.",
    )
    add_spec_arguments(parser)
    parser.add_argument(
        "--require-coverage",
        action="store_true",
        help="refuse to run when the loaded range has a hole in it (default: warn)",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="run and print the result, but write nothing to disk",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_command(lambda: _run(args))


def _run(args: argparse.Namespace) -> int:
    overrides: list[str] = list(args.overrides)
    run = resolve_run(args.spec, overrides)

    try:
        window = resolve_backtest_window(run.spec, run.days)
    except SpecError as e:
        raise CliError(str(e)) from e

    load_start = history_start(run, window[0])
    source = build_source(run)

    bars: list[Candle] | None = None
    stream: SparseReplayStream | None = None
    if run.spec.replay.mode == "sparse":
        stream = _sparse_stream(
            run,
            source,
            load_start=load_start,
            end_unix=window[1],
            require_coverage=args.require_coverage,
        )
        note(
            f"backtest {run.kind} {run.symbol} {run.base_tf.value} · SPARSE over "
            f"{run.spec.replay.coarse_timeframe.value if run.spec.replay.coarse_timeframe else '?'}"  # noqa: E501
            f" · {len(stream.triggers)} trigger(s), {len(stream.windows)} window(s) · "
            f"params {run.params_hash}"
            + (f" · --set {' '.join(overrides)}" if overrides else "")
        )
    else:
        bars = load_bars(
            source,
            run,
            start_unix=load_start,
            end_unix=window[1],
            require_coverage=args.require_coverage,
        )
        note(
            f"backtest {run.kind} {run.symbol} {run.base_tf.value} · {len(bars)} bars · "
            f"params {run.params_hash}"
            + (f" · --set {' '.join(overrides)}" if overrides else "")
        )

    sim = run.spec.sim
    record = run_backtest(
        bars=bars,
        stream=stream,
        strict_fidelity=run.spec.replay.strict,
        symbol=run.symbol,
        base_timeframe=run.base_tf,
        derived_timeframes=list(run.spec.derived_timeframes),
        days=run.days,
        instrument=run.instrument,
        strategy=run.strategy(),
        window=window,
        latency_ms=sim.latency_ms,
        half_spread_ticks=sim.half_spread_ticks,
        commission_cents_per_side_per_contract=sim.commission_cents_per_side_per_contract,
        seed=sim.seed,
        defer_protection_one_bar=sim.defer_protection_one_bar,
        warmup=run.warmup,
        observer_factory=run.plugin.observers(run.config),
        close_sink=run.plugin.close_sink(run.config),
        params=run.config_json,
        # Deterministic context ONLY: the run id carries a timestamp and would
        # make journal.jsonl differ between two runs of the same spec.
        journal_context={"kind": run.kind},
    )

    result: dict[str, Any] = record.result
    if not args.no_archive:
        # The wall is read HERE and nowhere else in the run: the stamp names
        # the directory and the index line, never a field inside result.json.
        ran_at = utc_stamp(LiveClock().now_unix())
        run_dir = write_run(
            args.runs_dir,
            run_id_for(ran_at, run.params_hash),
            result=result,
            spec=archived_spec(
                run.spec, resolved_config=run.config_json, window=window
            ),
            journal=record.journal,
            ran_at_utc=ran_at,
            kind=run.kind,
            mode="backtest",
            note=args.note,
            overrides=overrides,
            index_extra=run.plugin.index_columns(run.config, result),
        )
        note(f"archived -> {run_dir}")

    summary = result["summary"]
    note(
        f"done — {summary['trades']} trade(s), net {summary['netCents'] / 100:.2f} USD · "
        f"result {result['resultHash'][:12]}"
    )
    emit_result(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
