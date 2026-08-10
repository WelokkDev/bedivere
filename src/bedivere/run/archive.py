"""Run archive — one directory per run, plus an append-only index.

    runs/
      index.jsonl                          <- the query layer, one line per run
      2026-08-05T14-30-02Z_b39d0b03cf25/   <- UTC stamp + paramsHash
          spec.json      the envelope, config REPLACED by the resolved tree
          result.json    unchanged, still byte-deterministic
          journal.jsonl  the decision stream

Flat on purpose. A run varies along many axes at once (parameters, window,
symbol, timeframes, costs), so any folder-per-variant scheme forces a run
that is two things to live in one place. Folder names are not a query
language; `index.jsonl` is.

The archived spec has its `config` block replaced by the run's fully-resolved
config and its window materialized to stamps, so it re-runs identically with
NO `--set` flags. A spec plus a list of overrides you must remember to
re-apply is not a reproducible record.

WALL-CLOCK RULE: the timestamp lives in the directory name and the index
line, never inside result.json — that file's byte-identity across two runs of
the same spec is the determinism gate. `utc_stamp` therefore takes the
instant as DATA (from a Clock the caller owns); nothing here reads the wall.

CRASH SAFETY, two different mechanisms for two different failure modes:

  - Documents (`result.json`, `spec.json`) are written to a temp file and
    `os.replace`d. A reader sees the old file or the new one, never half.
  - The index is only ever APPENDED to, then flushed and fsync'd. There is no
    read-modify-write anywhere in this module: two runs finishing at once
    cannot lose each other's line, and a process killed mid-append truncates
    at most its own last line — which `read_index` skips rather than dying on.

The two hashes answer different questions:
  paramsHash  same STRATEGY SETTINGS (the config tree only — not the window)
  resultHash  same EXPERIMENT and same outcome (config + window + results)
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bedivere.engine.journal import DecisionJournal

INDEX_NAME = "index.jsonl"
RESULT_NAME = "result.json"
SPEC_NAME = "spec.json"
JOURNAL_NAME = "journal.jsonl"


def utc_stamp(unix_sec: int) -> str:
    """Render a GIVEN instant as a filename-safe UTC stamp. ':' is illegal in
    a Windows path, so the time separator is '-'; the result still sorts
    chronologically as plain text."""
    return datetime.fromtimestamp(unix_sec, UTC).strftime("%Y-%m-%dT%H-%M-%SZ")


def run_id_for(stamp: str, params_hash: str) -> str:
    """Directory name: sorts chronologically, and shows the config identity
    without opening anything."""
    return f"{stamp}_{params_hash}"


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Temp-file + `os.replace`, so a reader never sees a partial document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def append_index(runs_dir: Path, line: dict[str, Any]) -> Path:
    """Append one line to the runs index, flushed and fsync'd.

    Append-only and never read first: concurrent writers interleave whole
    lines rather than clobbering each other's view of the file. The fsync
    costs one syscall per RUN, which is not a budget worth optimising against
    the record of what the run did.
    """
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / INDEX_NAME
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(line, sort_keys=True, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return path


def read_index(runs_dir: Path) -> list[dict[str, Any]]:
    """Every index line, oldest first. A truncated final line (a run killed
    mid-append) is SKIPPED — the index must degrade to "one run missing",
    never to "the query tool crashes"."""
    path = runs_dir / INDEX_NAME
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not text:
            continue
        try:
            entry: object = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            out.append(entry)  # pyright: ignore[reportUnknownArgumentType]
    return out


def index_line(
    *,
    run_id: str,
    ran_at_utc: str,
    kind: str,
    mode: str,
    note: str,
    overrides: Sequence[str],
    result: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The queryable projection of a run: identity, the axes worth filtering
    on, and the headline numbers.

    Deliberately FLAT and deliberately strategy-agnostic — bedivere cannot
    know which of your knobs matter, so it indexes the ones every run has and
    leaves the rest to `paramsHash` (which covers all of them) and
    `spec.json` in the run directory (which spells them out).

    `extra` is your strategy's own columns (`StrategyPlugin.index_columns`),
    merged on top. The identity block above it stays byte-identical across
    kinds ON PURPOSE: one index file holds every strategy you run, so
    `--sort netCents` and `--where symbol=NQ` have to keep working on a mixed
    file. Your columns can never displace an identity field — a strategy that
    redefined `netCents` would corrupt exactly that guarantee.
    """
    summary: dict[str, Any] = result.get("summary", {})
    window: dict[str, Any] = result.get("window", {})
    identity: dict[str, Any] = {
        "runId": run_id,
        "ranAtUtc": ran_at_utc,
        "kind": kind,
        "mode": mode,
        "note": note,
        "overrides": list(overrides),
        "symbol": result.get("symbol"),
        "baseTimeframe": result.get("baseTimeframe"),
        "windowStart": window.get("startUnix"),
        "windowEnd": window.get("endUnix"),
        "bars": result.get("loop", {}).get("bars"),
        "paramsHash": result.get("paramsHash"),
        "resultHash": result.get("resultHash"),
        "trades": summary.get("trades"),
        "netCents": summary.get("netCents"),
        "openPositions": summary.get("openPositions"),
    }
    if not extra:
        return identity
    clashes = sorted(set(extra) & set(identity))
    if clashes:
        raise ValueError(
            f"index_columns for this strategy would overwrite the identity fields "
            f"{clashes} — pick different names. Those columns are what makes one index "
            "file queryable across every strategy you run."
        )
    return {**identity, **extra}


def write_run_files(
    run_dir: Path,
    *,
    result: dict[str, Any],
    journal: DecisionJournal | None = None,
    spec: dict[str, Any] | None = None,
) -> Path:
    """Write one run directory. `spec` is omitted by library callers who never
    had one — a `RunRecord.write("runs/mine")` from a hand-composed script is
    still a valid archive, just one whose provenance lives in your script.

    A live run has already streamed its journal into the directory; passing
    `journal=None` leaves that file alone rather than rewriting it from
    memory (the streamed copy is the crash-safe one).
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    if spec is not None:
        write_json_atomic(run_dir / SPEC_NAME, spec)
    write_json_atomic(run_dir / RESULT_NAME, result)
    if journal is not None:
        journal.write_jsonl(run_dir / JOURNAL_NAME)
    return run_dir


def write_run(
    runs_dir: Path,
    run_id: str,
    *,
    result: dict[str, Any],
    spec: dict[str, Any] | None,
    journal: DecisionJournal | None,
    ran_at_utc: str,
    kind: str,
    mode: str,
    note: str,
    overrides: Sequence[str] = (),
    index_extra: dict[str, Any] | None = None,
) -> Path:
    """Archive a run and append its index line. Returns the run directory.

    The index line goes LAST: a run directory without an index line is a run
    you can still find on disk, while an index line without a directory is a
    lie the query tool would repeat forever.
    """
    run_dir = write_run_files(runs_dir / run_id, result=result, journal=journal, spec=spec)
    append_index(
        runs_dir,
        index_line(
            run_id=run_id,
            ran_at_utc=ran_at_utc,
            kind=kind,
            mode=mode,
            note=note,
            overrides=overrides,
            result=result,
            extra=index_extra,
        ),
    )
    return run_dir
