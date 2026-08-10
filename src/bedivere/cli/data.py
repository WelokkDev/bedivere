"""`python -m bedivere.data` — get from "I have some CSVs" to "I can run a
backtest" without writing code.

    python -m bedivere.data import --db data/candles.db --symbol NQ \
                                   --timeframe 5m data/nq-5m.csv
    python -m bedivere.data list   --db data/candles.db
    python -m bedivere.data coverage --spec specs/x.json

The import is idempotent — the (symbol, timeframe, timestamp) key means
re-importing an overlapping file corrects the bars it overlaps instead of
duplicating them — so the answer to "did that file already go in?" is
always "run it again".

`coverage` is the command worth knowing about. It answers the question a bar
count cannot: not "how many bars do I have" but "which bars that should
exist are missing", measured against the session geometry the spec itself
declares. Run it before a backtest you intend to believe.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from bedivere.cli.common import (
    CliError,
    build_source,
    history_start,
    note,
    resolve_run,
    run_command,
)
from bedivere.config.spec import SpecError, resolve_backtest_window
from bedivere.core.types import Timeframe
from bedivere.data.csv import load_candles_csv
from bedivere.data.port import assess_coverage
from bedivere.data.sqlite import SqliteCandleStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bedivere.data", description="Candle store: import, inspect, verify."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    imp = sub.add_parser("import", help="import a CSV into the SQLite candle store")
    imp.add_argument("csv", type=Path, help="CSV with timestamp,open,high,low,close,volume")
    imp.add_argument("--db", type=Path, required=True, help="candle store file (created if needed)")
    imp.add_argument("--symbol", required=True, help="symbol these bars belong to")
    imp.add_argument(
        "--timeframe",
        required=True,
        choices=[tf.value for tf in Timeframe],
        help="timeframe these bars are (bedivere stamps bars at their CLOSE)",
    )

    lst = sub.add_parser("list", help="what is in a candle store")
    lst.add_argument("--db", type=Path, required=True, help="candle store file")

    cov = sub.add_parser("coverage", help="check a spec's data range against session geometry")
    cov.add_argument("--spec", type=Path, required=True, help="spec envelope JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_command(lambda: _dispatch(args))


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "import":
        return _import(args)
    if args.command == "list":
        return _list(args)
    return _coverage(args)


def _import(args: argparse.Namespace) -> int:
    try:
        bars = load_candles_csv(args.csv)
    except (OSError, ValueError) as e:
        raise CliError(f"{args.csv}: {e}") from e
    if not bars:
        raise CliError(f"{args.csv} has no rows")

    timeframe = Timeframe(args.timeframe)
    with SqliteCandleStore(args.db) as store:
        stats = store.upsert(args.symbol, timeframe, bars)
    note(
        f"imported {stats.written} bar(s) into {args.db}: {stats.inserted} new, "
        f"{stats.replaced} replaced · {args.symbol} {timeframe.value} "
        f"{_utc(bars[0].timestamp)} .. {_utc(bars[-1].timestamp)}"
    )
    return 0


def _list(args: argparse.Namespace) -> int:
    with SqliteCandleStore(args.db, read_only=True) as store:
        series = store.inventory()
    if not series:
        note(f"{args.db} is empty")
        return 0
    sys.stdout.write(f"{'SYMBOL':<12} {'TF':<5} {'BARS':>9}  FIRST .. LAST\n")
    for s in series:
        sys.stdout.write(
            f"{s.symbol:<12} {s.timeframe:<5} {s.bars:>9}  "
            f"{_utc(s.first_unix)} .. {_utc(s.last_unix)}\n"
        )
    return 0


def _coverage(args: argparse.Namespace) -> int:
    """Coverage over exactly the range the backtest would load — warm-up
    lookback included, because a run that cannot warm up is a run that cannot
    start, and finding that out here costs nothing."""
    run = resolve_run(args.spec, [])
    try:
        window = resolve_backtest_window(run.spec, run.days)
    except SpecError as e:
        raise CliError(str(e)) from e
    load_start = history_start(run, window[0])
    source = build_source(run)
    bars = source.candles(run.symbol, run.base_tf, load_start, window[1])
    coverage = assess_coverage(
        bars,
        days=run.days,
        symbol=run.symbol,
        timeframe=run.base_tf,
        start_unix=load_start,
        end_unix=window[1],
    )
    sys.stdout.write(f"{source.describe()}\n{coverage.describe()}\n")
    for gap in coverage.gaps:
        sys.stdout.write(
            f"  gap {_utc(gap.from_unix)} .. {_utc(gap.to_unix)}  ({gap.bars} bar(s))\n"
        )
    return 0 if coverage.complete else 1


def _utc(unix_sec: int) -> str:
    return datetime.fromtimestamp(unix_sec, UTC).strftime("%Y-%m-%d %H:%MZ")


if __name__ == "__main__":
    sys.exit(main())
