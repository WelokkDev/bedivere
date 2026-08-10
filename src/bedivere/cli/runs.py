"""`python -m bedivere.run` — query the run archive and the live lock.

    python -m bedivere.run list   --out runs --kind sma_cross --limit 20
    python -m bedivere.run show   --out runs 2026-08-05T14-30-02Z_b39d0b03cf25
    python -m bedivere.run status --out runs
    python -m bedivere.run stop   --out runs

Folder names are not a query language; `index.jsonl` is, and this is the tool
that reads it. `list` filters and sorts the append-only index without ever
rewriting it — the archive stays a record, not a database this command can
corrupt.

`status` and `stop` speak the supervision protocol from the other side: what
is running right now, and "please stop it" without a pid or a signal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from bedivere.cli.common import CliError, note, run_command
from bedivere.core.clock import LiveClock
from bedivere.run.archive import RESULT_NAME, read_index
from bedivere.run.supervise import STOP_NAME, is_stale, read_active, request_stop

_SORT_KEYS = ("ranAtUtc", "netCents", "trades", "bars")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bedivere.run", description="Query the run archive and the live lock."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    lst = sub.add_parser("list", help="list archived runs from index.jsonl")
    _add_runs_dir(lst)
    lst.add_argument("--kind", default=None, help="only this strategy kind")
    lst.add_argument("--symbol", default=None, help="only this symbol")
    lst.add_argument("--mode", default=None, help="only this mode (backtest, shadow, ...)")
    lst.add_argument("--params-hash", default=None, help="only this resolved-config hash")
    lst.add_argument("--limit", type=int, default=20, help="rows to show (default: 20)")
    lst.add_argument(
        "--sort", choices=_SORT_KEYS, default="ranAtUtc", help="sort key (default: ranAtUtc)"
    )
    lst.add_argument("--json", action="store_true", help="emit the matching index lines as JSON")

    show = sub.add_parser("show", help="print one run's result.json")
    _add_runs_dir(show)
    show.add_argument("run_id", help="run directory name")

    status = sub.add_parser("status", help="what is running right now (ACTIVE.json)")
    _add_runs_dir(status)

    stop = sub.add_parser("stop", help=f"request a graceful stop (creates <out>/{STOP_NAME})")
    _add_runs_dir(stop)
    return parser


def _add_runs_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--out", dest="runs_dir", type=Path, default=Path("runs"), help="runs root (default: runs)"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_command(lambda: _dispatch(args))


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "list":
        return _list(args)
    if args.command == "show":
        return _show(args)
    if args.command == "status":
        return _status(args)
    return _stop(args)


def _list(args: argparse.Namespace) -> int:
    rows = read_index(args.runs_dir)
    if not rows:
        note(f"no index at {args.runs_dir}/index.jsonl — nothing archived yet")
        return 0
    for field, wanted in (
        ("kind", args.kind),
        ("symbol", args.symbol),
        ("mode", args.mode),
        ("paramsHash", args.params_hash),
    ):
        if wanted is not None:
            rows = [r for r in rows if r.get(field) == wanted]
    # None sorts last regardless of direction: a row missing the key is a row
    # with nothing to say about it, not the smallest value.
    rows.sort(key=lambda r: (r.get(args.sort) is None, r.get(args.sort)), reverse=True)
    shown = rows[: args.limit]

    if args.json:
        sys.stdout.write(json.dumps(shown, indent=2, sort_keys=True) + "\n")
        return 0

    sys.stdout.write(
        f"{'RUN':<38} {'KIND':<16} {'MODE':<9} {'SYMBOL':<8} {'TRADES':>6} {'NET$':>10}  NOTE\n"
    )
    for row in shown:
        net = row.get("netCents")
        net_text = f"{net / 100:.2f}" if isinstance(net, int) else "-"
        sys.stdout.write(
            f"{_text(row.get('runId')):<38} {_text(row.get('kind')):<16} "
            f"{_text(row.get('mode')):<9} {_text(row.get('symbol')):<8} "
            f"{_text(row.get('trades')):>6} {net_text:>10}  {_text(row.get('note'))}\n"
        )
    note(f"{len(shown)} of {len(rows)} matching run(s)")
    return 0


def _show(args: argparse.Namespace) -> int:
    path = Path(args.runs_dir) / args.run_id / RESULT_NAME
    if not path.exists():
        raise CliError(f"no {RESULT_NAME} at {path}")
    sys.stdout.write(path.read_text(encoding="utf-8"))
    return 0


def _status(args: argparse.Namespace) -> int:
    active = read_active(args.runs_dir)
    if active is None:
        note(f"no active run in {args.runs_dir}")
        return 0
    sys.stdout.write(json.dumps(active.raw, indent=2, sort_keys=True) + "\n")
    # Staleness is judged against the LOCK's own heartbeat; the wall is read
    # only to measure its age, never to decide what the run should do.
    if is_stale(active, LiveClock().now_unix()):
        note(
            f"⚠ this lock is STALE (phase {active.phase}) — the next run will take it over"
        )
    return 0


def _stop(args: argparse.Namespace) -> int:
    active = read_active(args.runs_dir)
    path = request_stop(args.runs_dir)
    if active is None:
        note(f"no run is active — {path} will be cleared when the next one starts")
        return 0
    note(f"stop requested for {active.run_id} (phase {active.phase}) via {path}")
    return 0


def _text(value: Any) -> str:
    return "-" if value is None else str(value)


if __name__ == "__main__":
    sys.exit(main())
