"""The run archive: one directory per run, an append-only index, and no
read-modify-write anywhere.

The crash-safety claims are the ones with teeth. An index that is appended to
cannot lose a concurrent run's line, a document written through a temp file
cannot be read half-written, and a truncated final line must cost you one run
in a listing rather than the listing itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bedivere.engine.journal import DecisionJournal
from bedivere.run.archive import (
    INDEX_NAME,
    append_index,
    index_line,
    read_index,
    run_id_for,
    utc_stamp,
    write_json_atomic,
    write_run,
)

RESULT: dict[str, Any] = {
    "symbol": "DEMO",
    "baseTimeframe": "5m",
    "window": {"startUnix": 100, "endUnix": 200},
    "loop": {"bars": 42},
    "summary": {"trades": 3, "netCents": 1250, "openPositions": 0},
    "paramsHash": "abc123",
    "resultHash": "def456",
}


def test_the_stamp_is_filename_safe_and_sorts_chronologically() -> None:
    early = utc_stamp(1_785_000_000)
    late = utc_stamp(1_785_000_060)
    assert ":" not in early  # illegal in a Windows path
    assert early < late  # plain text sort is chronological
    assert run_id_for(early, "abc123") == f"{early}_abc123"


def test_a_run_directory_holds_the_three_files(tmp_path: Path) -> None:
    journal = DecisionJournal()
    journal.emit("signal", 100, score=1)
    run_dir = write_run(
        tmp_path,
        "run-1",
        result=RESULT,
        spec={"strategy": "x:PLUGIN", "config": {"kind": "ramp"}},
        journal=journal,
        ran_at_utc="2026-08-05T00-00-00Z",
        kind="ramp",
        mode="backtest",
        note="first",
    )
    assert {p.name for p in run_dir.iterdir()} == {"result.json", "spec.json", "journal.jsonl"}
    assert json.loads((run_dir / "result.json").read_text(encoding="utf-8")) == RESULT
    assert (run_dir / "journal.jsonl").read_text(encoding="utf-8").count("\n") == 1


def test_a_live_run_keeps_its_streamed_journal(tmp_path: Path) -> None:
    """journal=None means "already on disk, flushed per event" — rewriting it
    from memory would replace the crash-safe copy with one that only exists
    because we survived."""
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    (run_dir / "journal.jsonl").write_text('{"kind":"streamed"}\n', encoding="utf-8")
    write_run(
        tmp_path,
        "run-1",
        result=RESULT,
        spec=None,
        journal=None,
        ran_at_utc="2026-08-05T00-00-00Z",
        kind="ramp",
        mode="shadow",
        note="",
    )
    assert (run_dir / "journal.jsonl").read_text(encoding="utf-8") == '{"kind":"streamed"}\n'
    assert not (run_dir / "spec.json").exists()


def test_the_index_appends_and_never_rewrites(tmp_path: Path) -> None:
    for i in range(3):
        append_index(
            tmp_path,
            index_line(
                run_id=f"run-{i}",
                ran_at_utc=f"2026-08-05T00-00-0{i}Z",
                kind="ramp",
                mode="backtest",
                note="",
                overrides=[f"qty={i}"],
                result=RESULT,
            ),
        )
    rows = read_index(tmp_path)
    assert [r["runId"] for r in rows] == ["run-0", "run-1", "run-2"]
    assert rows[2]["overrides"] == ["qty=2"]
    # One line per run, appended in order — the file is a log, not a document.
    assert (tmp_path / INDEX_NAME).read_text(encoding="utf-8").count("\n") == 3


def test_the_index_line_carries_the_queryable_projection() -> None:
    line = index_line(
        run_id="run-1",
        ran_at_utc="2026-08-05T00-00-00Z",
        kind="ramp",
        mode="shadow",
        note="a note",
        overrides=["rr=2.5"],
        result=RESULT,
    )
    assert line["symbol"] == "DEMO"
    assert line["windowStart"] == 100 and line["windowEnd"] == 200
    assert line["trades"] == 3 and line["netCents"] == 1250
    assert line["paramsHash"] == "abc123" and line["resultHash"] == "def456"
    assert line["kind"] == "ramp" and line["mode"] == "shadow" and line["note"] == "a note"


def test_a_torn_final_line_costs_one_run_not_the_listing(tmp_path: Path) -> None:
    """A process killed mid-append truncates its own last line. read_index
    must degrade to "one run missing", never to "the query tool crashes"."""
    append_index(tmp_path, {"runId": "run-0"})
    with (tmp_path / INDEX_NAME).open("a", encoding="utf-8") as f:
        f.write('{"runId": "run-1", "netCe')  # killed here
    assert [r["runId"] for r in read_index(tmp_path)] == ["run-0"]


def test_reading_a_missing_index_is_empty_not_an_error(tmp_path: Path) -> None:
    assert read_index(tmp_path / "nothing-here") == []


def test_an_atomic_write_leaves_no_temp_file_behind(tmp_path: Path) -> None:
    target = tmp_path / "result.json"
    write_json_atomic(target, {"b": 2, "a": 1})
    write_json_atomic(target, {"b": 3, "a": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1, "b": 3}
    assert [p.name for p in tmp_path.iterdir()] == ["result.json"]


def test_run_record_write_is_the_same_archive(tmp_path: Path) -> None:
    """RunRecord.write is a thin wrapper, so a hand-composed script and the
    CLI produce byte-identical directories."""
    from bedivere.run.record import RunRecord

    journal = DecisionJournal()
    journal.emit("signal", 100)
    record = RunRecord(
        result=RESULT,
        journal=journal,
        portfolio=None,  # pyright: ignore[reportArgumentType] — unused by write
        view=None,  # pyright: ignore[reportArgumentType]
    )
    by_record = record.write(tmp_path / "a")
    by_archive = write_run(
        tmp_path / "root",
        "b",
        result=RESULT,
        spec=None,
        journal=journal,
        ran_at_utc="2026-08-05T00-00-00Z",
        kind="ramp",
        mode="backtest",
        note="",
    )
    for name in ("result.json", "journal.jsonl"):
        assert (by_record / name).read_bytes() == (by_archive / name).read_bytes()
