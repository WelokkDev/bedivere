"""`bedivere-data` — fill the bar lake, then look at what is in it.

    # what would this history cost? (free to ask)
    bedivere-data cost --dataset GLBX.MDP3 --symbols NQ.FUT --schema ohlcv-1s \
                       --start 2026-01-01 --end 2026-07-01

    # buy it, with a limit you had to type; wait up to an hour, then download
    bedivere-data fetch --dataset GLBX.MDP3 --symbols NQ.FUT --schema ohlcv-1s \
                        --start 2026-01-01 --end 2026-07-01 \
                        --cost-limit 250 --wait 3600 --dest archives/

    # without --wait it submits and prints the job id; download that job later
    bedivere-data fetch --job GLBX-20260101-XXXXXXXXXX --dest archives/

    # each job lands in its own archives/<job id>/; decode it into the lake,
    # and derive every coarser rung
    bedivere-data ingest --spec my-spec.json \
        --archive archives/GLBX-20260101-XXXXXXXXXX/glbx-...ohlcv-1s.dbn.zst

    # what is in there, and is any derived timeframe short?
    bedivere-data list

    # ask it anything
    bedivere-data sql "SELECT session, count(*) FROM bars WHERE tf='5m' GROUP BY 1"

    # would the run in this spec actually have the bars it needs?
    bedivere-data coverage --spec my-spec.json

`coverage` answers the question a bar count cannot: not "how many bars do I
have" but "which bars that SHOULD exist are missing", measured against the
session geometry the spec itself declares — and, where the vendor's condition
feed has been stored, which of those days the vendor has flagged as degraded.

`fetch` is the only command here that can spend money. It prices the request
first, refuses above the `--cost-limit` you typed, and asks before submitting.
"""

from __future__ import annotations

import argparse
import io
import shlex
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bedivere.cli.common import (
    CliError,
    build_source,
    history_start,
    note,
    resolve_run,
    run_command,
)
from bedivere.config.spec import SpecError, load_spec, resolve_backtest_window, resolve_sessions
from bedivere.core.clock import LiveClock
from bedivere.core.session_days import SessionDays, parse_session_days
from bedivere.core.types import Timeframe
from bedivere.data.port import assess_coverage
from bedivere.data.quality import assess
from bedivere.data.source import BarSourceError, lake_module
from bedivere.data.vendor.databento.condition import VendorHttpUnavailable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bedivere-data", description="The bar lake: buy, ingest, inspect, verify."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def with_root(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument(
            "--root",
            type=Path,
            default=None,
            help="lake root (default: $BEDIVERE_LAKE_ROOT, else ./data/lake)",
        )
        return p

    # ---------- filling it ----------

    ing = with_root(sub.add_parser("ingest", help="decode a DBN archive into the lake"))
    ing.add_argument("--archive", type=Path, required=True, help="the .dbn or .dbn.zst file")
    ing.add_argument("--spec", type=Path, help="spec envelope: its session calendar and, when it "
                     "names a lake, its dataset/series are the defaults")
    ing.add_argument("--session", type=Path, help="resolved session-day rows as JSON, instead of a spec")
    ing.add_argument("--symbol", help="root symbol, e.g. NQ (default: the spec's)")
    ing.add_argument("--dataset", help="vendor dataset code (default: the spec's, else GLBX.MDP3)")
    ing.add_argument(
        "--series",
        help="local.v.0 (our causal front month) or raw.<CONTRACT> (default: the spec's, "
        "else local.v.0)",
    )
    ing.add_argument("--timeframe", default="1s", help="the archive's NATIVE timeframe (default 1s)")
    ing.add_argument(
        "--skip-verify",
        action="store_true",
        help="skip the truncated-download check, which costs a full extra pass over the archive",
    )
    ing.add_argument("--no-derive", action="store_true", help="ingest the raw timeframe only")
    ing.add_argument("--dry-run", action="store_true", help="report without writing")

    der = with_root(sub.add_parser("derive", help="resample a stored timeframe into coarser ones"))
    der.add_argument("--spec", type=Path, help="spec envelope for the session calendar")
    der.add_argument("--session", type=Path, help="resolved session-day rows as JSON")
    der.add_argument("--symbol", help="root symbol (default: the spec's)")
    der.add_argument("--dataset")
    der.add_argument("--series")
    der.add_argument("--from", dest="source_tf", default="1s", help="the stored timeframe to read")
    der.add_argument(
        "--to",
        dest="target_tf",
        default=None,
        help="one target timeframe; the default walks the whole ladder from --from",
    )
    der.add_argument("--dry-run", action="store_true")

    vol = with_root(sub.add_parser("volume", help="build volume bars for offline ML research"))
    vol.add_argument("--spec", type=Path, help="spec envelope for the session calendar")
    vol.add_argument("--session", type=Path, help="resolved session-day rows as JSON")
    vol.add_argument("--symbol", help="root symbol (default: the spec's)")
    vol.add_argument("--dataset")
    vol.add_argument("--series", help="raw.<CONTRACT>, or an existing series with 1s contract identity")
    source = vol.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path, help="local Databento trades DBN covering full sessions")
    source.add_argument("--from", dest="source_tf", choices=["1s"],
                        help="approximate from stored one-second bars (source ohlcv-1s)")
    vol.add_argument("--threshold", required=True, type=int, help="contracts per volume bar")
    vol.add_argument("--boundary", choices=["whole_trade", "split_trade", "nearest_second"],
                     default="whole_trade", help="nearest_second is only for --from 1s")
    vol.add_argument("--compare-1s", action="store_true",
                     help="also build and summarize the 1s approximation from the same trades "
                     "(source trades-1s: a dataset of its own, never a --from 1s one)")
    vol.add_argument("--approx-boundary", choices=["whole_trade", "nearest_second"],
                     default="whole_trade", help="boundary policy for --compare-1s")
    vol.add_argument("--dry-run", action="store_true", help="compute diagnostics without writing")
    vol.add_argument("--resume", action="store_true",
                     help="verify and skip partitions whose input fingerprints have not changed")

    prep = with_root(sub.add_parser("prepare-trades", help="extract and cache trades from local DBN files"))
    prep.add_argument("--archive", type=Path, required=True,
                      help="a trades/MBO file or directory of contiguous daily DBN files")

    # ---------- looking at it ----------

    with_root(sub.add_parser("list", help="every series in the lake, and its lineage"))

    q = with_root(sub.add_parser("sql", help="run SQL (views: bars, volume_bars, condition)"))
    q.add_argument("query", nargs="?", help="SQL to run (default: a coverage summary)")
    q.add_argument("--limit", type=int, default=40, help="max rows to print (default 40)")

    cov = sub.add_parser("coverage", help="check a spec's data range against session geometry")
    cov.add_argument("--spec", type=Path, required=True, help="spec envelope JSON")

    # ---------- the vendor edge ----------

    cond = with_root(sub.add_parser("conditions", help="the vendor's per-day data-quality feed"))
    cond.add_argument("--dataset", default="GLBX.MDP3")
    cond.add_argument("--fetch", action="store_true", help="pull from the API (free) and store it")
    cond.add_argument("--start", help="first UTC date to fetch, YYYY-MM-DD (inclusive)")
    cond.add_argument("--end", help="last UTC date to fetch, YYYY-MM-DD (inclusive)")
    cond.add_argument("--flagged-only", action="store_true", help="print only non-available days")

    cost = sub.add_parser("cost", help="price a history request. Free — nothing is submitted")
    _add_request_arguments(cost, required=True)

    fetch = sub.add_parser("fetch", help="submit a batch job and download it. THIS SPENDS MONEY")
    # Not required at the parser: `--job` downloads an existing job and needs
    # none of them. `_submit` refuses a new job that is missing any.
    _add_request_arguments(fetch, required=False)
    fetch.add_argument(
        "--cost-limit",
        type=float,
        help="refuse to submit above this many USD. Required unless --job is given",
    )
    fetch.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="directory to download into; each job gets its own <dest>/<job id>/, so one "
        "--dest serves any number of jobs",
    )
    fetch.add_argument(
        "--job",
        help="download an EXISTING job id instead of submitting a new one "
        "(no request arguments needed)",
    )
    fetch.add_argument(
        "--wait",
        type=float,
        default=0.0,
        help="seconds to wait for the job to finish before downloading (default 0: a new job "
        "is submitted and its id printed, for a later --job to download)",
    )
    fetch.add_argument("--yes", action="store_true", help="do not ask before submitting")

    jobs = sub.add_parser("jobs", help="batch jobs on this account")
    jobs.add_argument(
        "--states",
        help="comma-separated states, e.g. done,processing (default: every state but expired)",
    )

    return parser


_REQUEST_ARGUMENTS = ("dataset", "symbols", "schema", "start", "end")


def _add_request_arguments(p: argparse.ArgumentParser, *, required: bool) -> None:
    unless = "" if required else " (required unless --job)"
    p.add_argument("--dataset", required=required, help=f"e.g. GLBX.MDP3{unless}")
    p.add_argument(
        "--symbols", required=required, help=f"e.g. NQ.FUT, comma-separated for several{unless}"
    )
    p.add_argument("--schema", required=required, help=f"e.g. ohlcv-1s{unless}")
    p.add_argument(
        "--start",
        required=required,
        help=f"inclusive; YYYY-MM-DD or an ISO 8601 datetime, UTC unless it says otherwise{unless}",
    )
    p.add_argument(
        "--end",
        required=required,
        help="EXCLUSIVE, same forms — --start 2026-09-01 --end 2026-09-02 is exactly one "
        f"day, and --end equal to --start is an empty range{unless}",
    )
    p.add_argument(
        "--stype-in",
        default="parent",
        help="symbology of --symbols (default: parent, which buys every contract of a root "
        "so a front month can be re-ranked causally later)",
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # DuckDB draws result tables with box-drawing characters, and a Windows
    # console defaults to a codepage that raises on the first one.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # pyright: ignore[reportUnknownMemberType]
    return run_command(lambda: _run(args))


def _run(args: argparse.Namespace) -> int:
    """Render a refused seam as one operator-facing line.

    `lake_module` and `http_client` already say what is missing and which extra
    carries it; without this they would surface through the blanket handler with
    an exception class name bolted on the front, which reads like a bedivere bug
    rather than a machine that is missing a dependency.
    """
    try:
        return _dispatch(args)
    except (BarSourceError, VendorHttpUnavailable) as e:
        raise CliError(str(e)) from e


def _dispatch(args: argparse.Namespace) -> int:
    handlers = {
        "ingest": _ingest,
        "derive": _derive,
        "volume": _volume,
        "prepare-trades": _prepare_trades,
        "list": _list,
        "sql": _sql,
        "coverage": _coverage,
        "conditions": _conditions,
        "cost": _cost,
        "fetch": _fetch,
        "jobs": _jobs,
    }
    return handlers[args.command](args)


# ---------- shared resolution ----------


def _session_days(args: argparse.Namespace) -> SessionDays:
    """Session geometry, handed over — never derived here.

    A spec is the normal source: it already holds the calendar the run will use,
    so an ingest and the backtest that reads it cannot disagree about where a
    session starts. Raw resolved rows are the escape hatch.
    """
    if bool(args.spec) == bool(args.session):
        raise CliError("pass exactly one of --spec (a spec envelope) or --session (resolved rows)")
    try:
        if args.spec:
            return resolve_sessions(load_spec(args.spec))
        import json

        return parse_session_days(json.loads(Path(args.session).read_text(encoding="utf-8")))
    except (SpecError, OSError, ValueError) as e:
        raise CliError(str(e)) from e


def _series_defaults(args: argparse.Namespace) -> tuple[str, str, str]:
    """(dataset, symbol, series) — the flags, falling back to the spec's own
    `data` block so an ingest for a spec cannot land in a series it will never
    read."""
    dataset, symbol, series = args.dataset, args.symbol, args.series
    if args.spec:
        spec = load_spec(args.spec)
        symbol = symbol or spec.symbol
        if spec.data is not None and spec.data.source == "lake":
            dataset = dataset or spec.data.dataset
            series = series or spec.data.series
    if not symbol:
        raise CliError("--symbol is required (or pass a --spec that names one)")
    return dataset or "GLBX.MDP3", symbol, series or "local.v.0"


def _series_id(dataset: str, symbol: str, series: str) -> Any:
    """`local.v.0` by default — OUR front-month derivation, not the vendor's,
    which need not agree with it (see `lake.layout.local_continuous`)."""
    layout = lake_module("layout")
    try:
        if series.startswith("local."):
            _, rule, rank = series.split(".")
            return layout.local_continuous(dataset, symbol, layout.RollRule(rule), int(rank))
        if series.startswith("raw."):
            return layout.raw_contract(dataset, symbol, series.removeprefix("raw."))
        return layout.SeriesId(dataset=dataset, symbol=symbol, series=series)
    except ValueError as e:
        raise CliError(
            f"--series {series!r} is not a series identity bedivere can name ({e}). "
            "Use local.<rule>.<rank> (e.g. local.v.0), raw.<CONTRACT> (e.g. raw.NQZ5), "
            "or the vendor's own <rule>.<rank>."
        ) from e


def _lake_root(args: argparse.Namespace) -> Path:
    return lake_module("query").lake_root(getattr(args, "root", None))


# ---------- filling the lake ----------


def _ingest(args: argparse.Namespace) -> int:
    ingest = lake_module("ingest")
    resample = lake_module("resample")
    catalog = lake_module("catalog")

    days = _session_days(args)
    dataset, symbol, series = _series_defaults(args)
    sid = _series_id(dataset, symbol, series)
    timeframe = Timeframe(args.timeframe)
    root = _lake_root(args)

    try:
        declared = ingest.read_range(args.archive)
    except (ingest.IngestError, OSError) as e:
        raise CliError(str(e)) from e
    if declared.dataset != dataset:
        raise CliError(
            f"the archive says dataset={declared.dataset}, but this ingest is labelled "
            f"{dataset} — the dataset is part of the series identity, so this would "
            "mislabel every partition it writes"
        )

    note(f"archive:  {args.archive.name}  (dataset={declared.dataset} schema={declared.schema})")
    note(f"series:   {sid} {timeframe.value}  ->  {root}")
    note(f"session:  {len(days.days)} days, {days.days[0].label} .. {days.days[-1].label}")
    note(f"mode:     {'DRY RUN — nothing is written' if args.dry_run else 'writing'}")

    try:
        if not args.skip_verify:
            note("verify:   scanning for a truncated download ...")
            got = ingest.check_complete(args.archive)
            note(f"          {got.records:,} records, complete")

        symbology = ingest.build_symbology(ingest.read_metadata(args.archive))
        note(
            f"symbols:  {len(symbology)} instrument ids, "
            f"{len(symbology.ambiguous_ids)} reused across time"
        )

        front_months = None
        if not series.startswith("raw."):
            note("pass 1/2: ranking front months on D-1 volume ...")
            front_months = ingest.causal_front_months(
                args.archive, symbology, days, timeframe.period_seconds, symbol
            )
            note(f"          {len(front_months)} session-days ranked")

        note("pass 2/2: writing session-day partitions ...")
        report = ingest.ingest_dbn(
            args.archive, root, sid, timeframe, days,
            front_months=front_months, dry_run=args.dry_run,
        )
        note(f"          {report.describe()}")

        if not args.no_derive:
            for source_tf, target_tf in resample.DERIVE_CHAINS.get(timeframe, []):
                result = resample.resample_series(
                    root, sid, source_tf, target_tf, days, dry_run=args.dry_run
                )
                note(f"derive:   {result.describe()}")
    except (ingest.IngestError, resample.ResampleError, OSError) as e:
        raise CliError(str(e)) from e

    if args.dry_run:
        return 0
    return _report_inventory(catalog, root)


def _derive(args: argparse.Namespace) -> int:
    resample = lake_module("resample")
    catalog = lake_module("catalog")

    days = _session_days(args)
    dataset, symbol, series = _series_defaults(args)
    sid = _series_id(dataset, symbol, series)
    root = _lake_root(args)
    source_tf = Timeframe(args.source_tf)

    if args.target_tf:
        chain = [(source_tf, Timeframe(args.target_tf))]
    else:
        chain = resample.DERIVE_CHAINS.get(source_tf, [])
        if not chain:
            raise CliError(
                f"no derive ladder starts at {source_tf.value} — pass --to to name one target, "
                f"or start from {', '.join(tf.value for tf in resample.DERIVE_CHAINS)}"
            )

    try:
        for src, dst in chain:
            result = resample.resample_series(root, sid, src, dst, days, dry_run=args.dry_run)
            note(f"derive:   {result.describe()}")
    except resample.ResampleError as e:
        raise CliError(str(e)) from e

    if args.dry_run:
        return 0
    return _report_inventory(catalog, root)


def _volume(args: argparse.Namespace) -> int:
    import json

    build = lake_module("volume_build")
    volume = lake_module("volume")
    schema = lake_module("schema")
    import duckdb
    days = _session_days(args)
    sid = _series_id(*_series_defaults(args))
    root = _lake_root(args)
    try:
        if args.compare_1s and not args.archive:
            raise ValueError("--compare-1s requires --archive")
        if args.approx_boundary != "whole_trade" and not args.compare_1s:
            raise ValueError("--approx-boundary requires --compare-1s")
        spec = volume.VolumeSpec(
            args.threshold, "trades" if args.archive else f"ohlcv-{args.source_tf}", args.boundary,
        )
        if args.archive:
            reports = build.build_volume_archive(
                args.archive, root, sid, spec, days,
                compare_1s=args.compare_1s, dry_run=args.dry_run, resume=args.resume,
                approx_boundary=args.approx_boundary,
            )
        else:
            reports = build.derive_volume(root, sid, spec, days, dry_run=args.dry_run, resume=args.resume)
    except (ValueError, OSError, build.IngestError, schema.BarSchemaError, duckdb.Error) as e:
        raise CliError(str(e)) from e
    sys.stdout.write(json.dumps(reports, indent=2) + "\n")
    return 0


def _prepare_trades(args: argparse.Namespace) -> int:
    module = lake_module("trade_archive")
    try:
        path = module.prepare_trade_archive(
            args.archive, _lake_root(args) / "_meta" / "trade_cache", progress=note,
        )
    except (ValueError, OSError, module.IngestError) as e:
        raise CliError(str(e)) from e
    sys.stdout.write(str(path) + "\n")
    return 0


def _report_inventory(catalog: Any, root: Path) -> int:
    """The inventory, then the lineage check. Exit 1 on an incomplete rung —
    a derive that stopped early leaves every file valid and the series short."""
    _print_inventory(catalog, root)
    problems = catalog.coverage_gaps(root)
    if not problems:
        sys.stdout.write("\ncoverage: every derived timeframe covers exactly its source's days\n")
        return 0
    sys.stdout.write("\nINCOMPLETE derived timeframes:\n")
    for problem in problems:
        sys.stdout.write(f"  {problem}\n")
    return 1


def _print_inventory(catalog: Any, root: Path) -> None:
    entries = catalog.inventory(root)
    event_entries = lake_module("volume_store").volume_inventory(root)
    if not entries and not event_entries:
        sys.stdout.write(f"{root} holds no bars\n")
        return
    if entries:
        sys.stdout.write(f"{'SERIES':<34} {'TF':<5} {'DAYS':>6}  SOURCE\n")
    for entry in entries:
        sys.stdout.write(
            f"{str(entry.series):<34} {entry.timeframe.value:<5} "
            f"{entry.session_days:>6}  {entry.source}\n"
        )
    if event_entries:
        import json

        sys.stdout.write("\nVolume-bar datasets (offline research):\n")
        for entry in event_entries:
            sys.stdout.write(json.dumps(entry, sort_keys=True) + "\n")


# ---------- looking at it ----------


def _list(args: argparse.Namespace) -> int:
    catalog = lake_module("catalog")
    return _report_inventory(catalog, _lake_root(args))


def _sql(args: argparse.Namespace) -> int:
    query_mod = lake_module("query")
    import duckdb

    with query_mod.connect(args.root) as conn:
        try:
            relation = conn.sql(args.query or query_mod.COVERAGE_SQL)
        except duckdb.Error as e:
            raise CliError(f"query failed: {e}") from e
        # duckdb's stubs type `sql()` as always returning a relation, but at
        # runtime a statement that produces none returns None.
        if relation is not None:  # pyright: ignore[reportUnnecessaryComparison]
            relation.show(max_rows=args.limit)
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
    try:
        bars = source.candles(run.symbol, run.base_tf, load_start, window[1])
    except BarSourceError as e:
        raise CliError(str(e)) from e
    coverage = assess_coverage(
        bars,
        days=run.days,
        symbol=run.symbol,
        timeframe=run.base_tf,
        start_unix=load_start,
        end_unix=window[1],
    )
    basis = getattr(source, "price_basis", None)
    sys.stdout.write(f"{source.describe()}{f' [{basis}]' if basis else ''}\n")
    sys.stdout.write(f"{coverage.describe()}\n")
    for gap in coverage.gaps:
        sys.stdout.write(
            f"  gap {_utc(gap.from_unix)} .. {_utc(gap.to_unix)}  ({gap.bars} bar(s))\n"
        )

    flagged = _quality_lines(run, args)
    for line in flagged:
        sys.stdout.write(f"  {line}\n")
    return 0 if coverage.complete and not flagged else 1


def _quality_lines(run: Any, args: argparse.Namespace) -> list[str]:
    """The vendor's verdict on the run's session-days, when it has been stored.

    Silent when no condition history is present, because a bar-coverage check
    should not fail over a free, optional feed nobody fetched. But when the feed
    IS there and calls a day degraded, nothing else in this command can see it —
    those bars exist and look like bars.
    """
    if run.spec.data is None or run.spec.data.source != "lake":
        return []
    from bedivere.data.vendor.databento.condition import read_conditions

    root = lake_module("query").lake_root(
        Path(run.spec.data.root) if run.spec.data.root else None
    )
    rows = read_conditions(root, run.spec.data.dataset or "")
    if not rows:
        return []
    return [
        f"quality {verdict.describe()}"
        for verdict in assess(run.days, rows)
        if not verdict.clean
    ]


# ---------- the vendor edge ----------


def _conditions(args: argparse.Namespace) -> int:
    from bedivere.data.vendor.databento.condition import (
        DatabentoApiError,
        fetch_conditions,
        read_conditions,
        write_conditions,
    )

    root = _lake_root(args)
    if args.fetch:
        if not (args.start and args.end):
            raise CliError("--fetch needs --start and --end (inclusive UTC dates, YYYY-MM-DD)")
        try:
            rows = fetch_conditions(args.dataset, args.start, args.end)
        except (DatabentoApiError, ValueError, OSError) as e:
            raise CliError(str(e)) from e
        path = write_conditions(root, args.dataset, rows)
        note(f"stored {len(rows)} condition row(s) -> {path}")
    else:
        rows = read_conditions(root, args.dataset)
        if not rows:
            raise CliError(
                f"no stored conditions for {args.dataset} under {root} — "
                "run with --fetch --start ... --end ... (the endpoint is free)"
            )

    shown = [r for r in rows if not r.condition.usable] if args.flagged_only else rows
    sys.stdout.write(f"{'DATE':<12} {'CONDITION':<10} LAST MODIFIED\n")
    for row in shown:
        sys.stdout.write(f"{row.date:<12} {row.condition.value:<10} {row.last_modified or '-'}\n")
    flagged = sum(1 for r in rows if not r.condition.usable)
    sys.stdout.write(f"\n{len(rows)} day(s), {flagged} not available\n")
    return 0


def _check_range(start: str, end: str) -> None:
    """Refuse an empty range before it is priced. `end` is EXCLUSIVE, so
    `--start X --end X` asks for nothing — and `cost` would answer $0.00 for it,
    which reads as "free", not "empty"."""
    first, last = _parse_bound(start), _parse_bound(end)
    if first is None or last is None:
        return  # a form this CLI does not parse (UNIX nanoseconds): the API judges it
    if last <= first:
        raise CliError(
            f"--end {end} is not after --start {start}. --end is exclusive, so this range is "
            "empty; one day is --start 2026-09-01 --end 2026-09-02"
        )


def _parse_bound(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    # The API reads a bare timestamp as UTC; comparing a bare one with a zoned
    # one would otherwise raise.
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _cost(args: argparse.Namespace) -> int:
    from bedivere.data.vendor.databento import batch

    _check_range(args.start, args.end)
    try:
        usd = batch.get_cost(
            args.dataset, args.symbols, args.schema, args.start, args.end,
            stype_in=args.stype_in,
        )
    except (batch.DatabentoApiError, ValueError, OSError) as e:
        raise CliError(str(e)) from e
    sys.stdout.write(
        f"{args.dataset} {args.schema} {args.symbols} {args.start}..{args.end} "
        f"(stype_in={args.stype_in})\n${usd:,.2f}\n"
    )
    return 0


def _fetch(args: argparse.Namespace) -> int:
    from bedivere.data.vendor.databento import batch

    try:
        if args.job:
            job_id = args.job
        else:
            job_id = _submit(batch, args)
            if args.wait <= 0:
                # A job this new is still being prepared: downloading now would
                # only find no files and fail a command that did what it was asked.
                sys.stdout.write(f"{job_id}\n")
                note(f"submitted {job_id}, download it later with --job {job_id}:")
                note(f"  bedivere-data fetch --job {job_id} --dest {shlex.quote(str(args.dest))} --wait 3600")
                return 0
        if args.wait > 0:
            note(f"waiting up to {args.wait:.0f}s for job {job_id} ...")
            job = batch.wait_for_job(job_id, clock=LiveClock(), timeout_seconds=args.wait)
            note(f"          {job.describe()}")
        note(f"downloading job {job_id} -> {args.dest / job_id}")
        paths = batch.download_job(job_id, args.dest)
    except (batch.DatabentoApiError, ValueError, OSError) as e:
        raise CliError(str(e)) from e

    for path in paths:
        sys.stdout.write(f"{path}\n")
    note(
        f"{len(paths)} file(s). Keep the archive: a lake partition is a whole-file "
        "replacement derived from it, so it is what makes the lake reproducible."
    )
    return 0


def _submit(batch: Any, args: argparse.Namespace) -> str:
    """Price it, show it, ask, then submit. Never any two of those out of order."""
    missing = [f"--{name}" for name in _REQUEST_ARGUMENTS if getattr(args, name) is None]
    if missing:
        raise CliError(
            f"{', '.join(missing)} required to submit a job (--job downloads an existing one)"
        )
    if args.cost_limit is None:
        raise CliError("--cost-limit is required to submit a job (--job downloads an existing one)")
    _check_range(args.start, args.end)
    usd = batch.get_cost(
        args.dataset, args.symbols, args.schema, args.start, args.end, stype_in=args.stype_in
    )
    note(
        f"estimate: ${usd:,.2f} for {args.dataset} {args.schema} {args.symbols} "
        f"{args.start}..{args.end}  (limit ${args.cost_limit:,.2f})"
    )
    if not args.yes:
        if not sys.stdin.isatty():
            raise CliError(
                f"this would submit a billed job for ${usd:,.2f} and stdin is not a terminal — "
                "pass --yes to confirm non-interactively"
            )
        if input(f"submit this job for ${usd:,.2f}? [y/N] ").strip().lower() not in ("y", "yes"):
            raise CliError("not submitted")
    job = batch.submit_job(
        args.dataset, args.symbols, args.schema, args.start, args.end,
        cost_limit_usd=args.cost_limit, stype_in=args.stype_in,
    )
    note(f"submitted: {job.describe()}")
    return job.id


def _jobs(args: argparse.Namespace) -> int:
    from bedivere.data.vendor.databento import batch

    states = args.states.split(",") if args.states else None
    try:
        jobs = batch.list_jobs(states=states)
    except (batch.DatabentoApiError, ValueError, OSError) as e:
        raise CliError(str(e)) from e
    if not jobs:
        note("no batch jobs on this account")
        return 0
    for job in jobs:
        sys.stdout.write(f"{job.describe()}\n")
    return 0


def _utc(unix_sec: int) -> str:
    return datetime.fromtimestamp(unix_sec, UTC).strftime("%Y-%m-%d %H:%MZ")


if __name__ == "__main__":
    sys.exit(main())
