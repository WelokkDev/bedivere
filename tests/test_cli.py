"""The backtest CLI end to end: a spec file in, an archived run out.

`tests.strategy_fixture` sits outside the library on purpose: these tests
pass only if a spec can name a strategy bedivere has never heard of.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from bedivere.cli import backtest as backtest_cli
from bedivere.cli import data as data_cli
from bedivere.cli import runs as runs_cli
from bedivere.core.types import Timeframe
from bedivere.run.archive import INDEX_NAME, read_index

SPEC: dict[str, Any] = {
    "strategy": "tests.strategy_fixture:PLUGIN",
    "symbol": "DEMO",
    "note": "fixture",
    "config": {"kind": "ramp", "enter_after_bars": 2},
    "baseTimeframe": "5m",
    "derivedTimeframes": ["30m"],
    "session": {
        "template": "cme_us_index_futures_eth",
        "timezone": "America/New_York",
        "openTime": "18:00",
        "closeTime": "17:00",
        "firstCloseDate": "2026-07-14",
        "lastCloseDate": "2026-07-16",
    },
    "instruments": {"DEMO": {"tickSize": 0.25, "pointValue": 20}},
    "window": {"firstTradeDate": "2026-07-15", "lastTradeDate": "2026-07-16"},
    "sim": {
        "latencyMs": 250,
        "halfSpreadTicks": 1,
        "commissionCentsPerSidePerContract": 105,
        "seed": 7,
        "deferProtectionOneBar": False,
    },
    "data": {"source": "python", "factory": "tests.strategy_fixture:build_ramp_source"},
}


def write_spec(tmp_path: Path, **overrides: Any) -> Path:
    payload = {**json.loads(json.dumps(SPEC)), **overrides}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def run_backtest_cli(spec: Path, runs: Path, *extra: str) -> int:
    return backtest_cli.main(["--spec", str(spec), "--out", str(runs), *extra])


def stdout_json(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    return json.loads(capsys.readouterr().out)


# ---------- the happy path ----------


def test_a_spec_runs_end_to_end_and_archives(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = tmp_path / "runs"
    assert run_backtest_cli(write_spec(tmp_path), runs) == 0

    result = stdout_json(capsys)
    assert result["symbol"] == "DEMO"
    assert result["summary"]["trades"] > 0

    (run_dir,) = [p for p in runs.iterdir() if p.is_dir()]
    assert {p.name for p in run_dir.iterdir()} == {"result.json", "spec.json", "journal.jsonl"}
    assert json.loads((run_dir / "result.json").read_text(encoding="utf-8")) == result

    rows = read_index(runs)
    assert len(rows) == 1
    assert rows[0]["runId"] == run_dir.name
    assert rows[0]["kind"] == "ramp"
    assert rows[0]["mode"] == "backtest"
    assert rows[0]["note"] == ""
    assert rows[0]["resultHash"] == result["resultHash"]
    assert rows[0]["trades"] == result["summary"]["trades"]


def test_the_same_spec_twice_is_byte_identical(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The determinism gate, through the CLI: the run directory name carries
    the wall-clock stamp, result.json carries none."""
    spec = write_spec(tmp_path)
    run_backtest_cli(spec, tmp_path / "a")
    first = stdout_json(capsys)
    run_backtest_cli(spec, tmp_path / "b")
    second = stdout_json(capsys)
    assert first == second


def test_the_archived_spec_reruns_without_the_set_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The archived spec has the overrides baked in and the window
    materialized, so it reproduces the run on its own."""
    runs = tmp_path / "runs"
    run_backtest_cli(write_spec(tmp_path), runs, "--set", "enter_after_bars=5")
    swept = stdout_json(capsys)

    (run_dir,) = [p for p in runs.iterdir() if p.is_dir()]
    archived = run_dir / "spec.json"
    assert json.loads(archived.read_text(encoding="utf-8"))["config"]["enter_after_bars"] == 5

    assert run_backtest_cli(archived, tmp_path / "replay") == 0
    assert stdout_json(capsys)["resultHash"] == swept["resultHash"]


def test_set_changes_the_params_hash_and_the_run_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = tmp_path / "runs"
    run_backtest_cli(write_spec(tmp_path), runs)
    base = stdout_json(capsys)
    run_backtest_cli(write_spec(tmp_path), runs, "--set", "rr=2.5")
    swept = stdout_json(capsys)

    assert base["paramsHash"] != swept["paramsHash"]
    assert base["params"]["rr"] != swept["params"]["rr"] == 2.5
    assert len([p for p in runs.iterdir() if p.is_dir()]) == 2
    assert {r["paramsHash"] for r in read_index(runs)} == {base["paramsHash"], swept["paramsHash"]}


def test_no_archive_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runs = tmp_path / "runs"
    assert run_backtest_cli(write_spec(tmp_path), runs, "--no-archive") == 0
    assert stdout_json(capsys)["summary"]["trades"] > 0
    assert not runs.exists()


# ---------- the refusals ----------


def test_a_typod_set_path_fails_loudly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_backtest_cli(write_spec(tmp_path), tmp_path / "runs", "--set", "rrr=2.5") == 1
    captured = capsys.readouterr()
    assert captured.out == ""  # nothing on stdout: there is no result
    assert "rrr" in captured.err


def test_a_typod_spec_key_fails_loudly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = write_spec(tmp_path)
    payload = json.loads(spec.read_text(encoding="utf-8"))
    payload["baseTimefrmae"] = payload.pop("baseTimeframe")
    spec.write_text(json.dumps(payload), encoding="utf-8")
    assert run_backtest_cli(spec, tmp_path / "runs") == 1
    assert "baseTimefrmae" in capsys.readouterr().err


def test_an_unimportable_strategy_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = write_spec(tmp_path, strategy="no_such_pkg.nowhere:PLUGIN")
    assert run_backtest_cli(spec, tmp_path / "runs") == 1
    assert "cannot import module" in capsys.readouterr().err


def test_a_name_that_is_not_a_plugin_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = write_spec(tmp_path, strategy="tests.strategy_fixture:NOT_A_PLUGIN")
    assert run_backtest_cli(spec, tmp_path / "runs") == 1
    assert "not a StrategyPlugin" in capsys.readouterr().err


def test_a_spec_pointing_at_the_wrong_strategy_is_caught(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The config validates cleanly against the other plugin's model — only
    the kind check notices they are different strategies."""
    spec = write_spec(tmp_path, strategy="tests.strategy_fixture:OTHER_PLUGIN")
    assert run_backtest_cli(spec, tmp_path / "runs") == 1
    assert "wrong strategy" in capsys.readouterr().err


def test_a_warmup_the_calendar_cannot_cover_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Starting short is the one failure a warm-up gate exists to prevent, so
    it is refused before any bar is read."""
    spec = write_spec(tmp_path)
    payload = json.loads(spec.read_text(encoding="utf-8"))
    payload["config"]["warmup_bars"] = 10_000
    spec.write_text(json.dumps(payload), encoding="utf-8")
    assert run_backtest_cli(spec, tmp_path / "runs") == 1
    assert "warm-up lookback exceeds" in capsys.readouterr().err


def test_require_coverage_refuses_a_hole(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = write_spec(
        tmp_path,
        data={"source": "python", "factory": "tests.strategy_fixture:build_gappy_source"},
    )
    runs = tmp_path / "runs"
    assert run_backtest_cli(spec, runs, "--require-coverage") == 1
    assert "incomplete data" in capsys.readouterr().err
    # ...and warns rather than refusing by default.
    assert run_backtest_cli(spec, runs) == 0
    assert "gap(s)" in capsys.readouterr().err


# ---------- the data + runs commands ----------


def test_a_csv_spec_runs_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The zero-setup path a cloner walks first: one CSV, one spec, a backtest.
    No import step, no native dependency, no lake."""
    from tests.strategy_fixture import ramp_bars, ramp_days

    csv = tmp_path / "demo-5m.csv"
    csv.write_text(
        "timestamp,open,high,low,close,volume\n"
        + "".join(
            f"{b.timestamp},{b.open},{b.high},{b.low},{b.close},{b.volume}\n"
            for b in ramp_bars(ramp_days())
        ),
        encoding="utf-8",
    )
    spec = write_spec(tmp_path, data={"source": "csv", "path": str(csv)})
    assert run_backtest_cli(spec, tmp_path / "runs") == 0
    assert stdout_json(capsys)["summary"]["trades"] > 0


def test_a_lake_series_runs_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The research path: bars in the lake, inspected, then backtested.

    The point of the assertion on `list` is that the run and the inventory agree
    about WHICH series was read — the spec names `local.v.0`, and nothing else
    could have answered.
    """
    pytest.importorskip("duckdb", reason="the lake extra is not installed")
    from bedivere.data.lake.catalog import inventory
    from bedivere.data.lake.layout import RollRule, local_continuous
    from bedivere.data.lake.schema import BarBatch
    from bedivere.data.lake.writer import write_day
    from tests.strategy_fixture import ramp_bars, ramp_days

    lake = tmp_path / "lake"
    sid = local_continuous("TEST", "DEMO", RollRule.VOLUME, 0)
    days = ramp_days()
    bars = ramp_bars(days)
    for day in days.days:
        in_day = [b for b in bars if day.start_unix < b.timestamp <= day.end_unix]
        write_day(
            lake, sid, Timeframe.M5, day.label,
            BarBatch.from_candles(in_day, instrument_id=1),
            source="fixture",
        )
    assert [(e.timeframe, e.session_days) for e in inventory(lake)] == [(Timeframe.M5, 3)]

    assert data_cli.main(["list", "--root", str(lake)]) == 0
    listed = capsys.readouterr().out
    assert "TEST:DEMO:local.v.0" in listed and "fixture" in listed

    spec = write_spec(
        tmp_path,
        data={"source": "lake", "dataset": "TEST", "series": "local.v.0", "root": str(lake)},
    )
    assert run_backtest_cli(spec, tmp_path / "runs") == 0
    assert stdout_json(capsys)["summary"]["trades"] > 0

    assert data_cli.main(["coverage", "--spec", str(spec)]) == 0
    coverage_out = capsys.readouterr().out
    assert "lake:TEST/local.v.0 [as_traded]" in coverage_out
    assert "100.0%" in coverage_out


def test_a_lake_one_session_behind_the_window_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The reach gate is session-aware. A window ending at a close the lake does
    not hold refuses before a bar is read — rather than warning, running on the
    bars that were there, and archiving a result with a trade in it. A time
    tolerance cannot do this: 86 400 s is exactly one CME close-to-close.
    """
    pytest.importorskip("duckdb", reason="the lake extra is not installed")
    from bedivere.data.lake.layout import RollRule, local_continuous
    from bedivere.data.lake.schema import BarBatch
    from bedivere.data.lake.writer import write_day
    from tests.strategy_fixture import ramp_bars, ramp_days

    lake = tmp_path / "lake"
    sid = local_continuous("TEST", "DEMO", RollRule.VOLUME, 0)
    days = ramp_days()
    bars = ramp_bars(days)
    for day in days.days[:-1]:  # everything but the window's last session-day
        in_day = [b for b in bars if day.start_unix < b.timestamp <= day.end_unix]
        write_day(
            lake, sid, Timeframe.M5, day.label,
            BarBatch.from_candles(in_day, instrument_id=1),
            source="fixture",
        )
    data = {"source": "lake", "dataset": "TEST", "series": "local.v.0", "root": str(lake)}

    runs = tmp_path / "runs"
    assert run_backtest_cli(write_spec(tmp_path, data=data), runs) == 1
    err = capsys.readouterr().err
    assert "session-day 2026-07-16" in err and "silently short" in err
    assert not runs.exists() or not any(p.is_dir() for p in runs.iterdir())

    assert data_cli.main(["coverage", "--spec", str(write_spec(tmp_path, data=data))]) == 1
    assert "session-day 2026-07-16" in capsys.readouterr().err

    # Accepting what is there is a visible choice in the spec, not a default.
    stale = write_spec(tmp_path, data={**data, "allowStale": True})
    assert run_backtest_cli(stale, runs) == 0
    assert "50.0%" in capsys.readouterr().err


def test_a_lake_spec_naming_a_series_that_is_not_there_refuses_at_the_start(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No silent fallback, and the cheapest possible failure: before a bar is
    loaded, naming what was missing."""
    pytest.importorskip("duckdb", reason="the lake extra is not installed")
    spec = write_spec(
        tmp_path,
        data={"source": "lake", "dataset": "TEST", "series": "local.v.0", "root": str(tmp_path / "empty")},
    )
    assert run_backtest_cli(spec, tmp_path / "runs") == 1
    err = capsys.readouterr().err
    assert "has no DEMO bars for 5m" in err
    assert "will NOT fall back" in err


def test_coverage_command_reports_and_exits_nonzero_on_a_hole(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert data_cli.main(["coverage", "--spec", str(write_spec(tmp_path))]) == 0
    assert "100.0%" in capsys.readouterr().out

    gappy = write_spec(
        tmp_path,
        data={"source": "python", "factory": "tests.strategy_fixture:build_gappy_source"},
    )
    assert data_cli.main(["coverage", "--spec", str(gappy)]) == 1
    assert "gap " in capsys.readouterr().out


def test_runs_list_filters_the_index(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runs = tmp_path / "runs"
    run_backtest_cli(write_spec(tmp_path), runs, "--note", "base")
    run_backtest_cli(write_spec(tmp_path), runs, "--set", "rr=2.5", "--note", "swept")
    capsys.readouterr()

    assert runs_cli.main(["list", "--out", str(runs), "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {r["note"] for r in rows} == {"base", "swept"}

    assert runs_cli.main(["list", "--out", str(runs), "--kind", "nope", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_runs_show_prints_one_result(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runs = tmp_path / "runs"
    run_backtest_cli(write_spec(tmp_path), runs)
    expected = stdout_json(capsys)
    (run_dir,) = [p for p in runs.iterdir() if p.is_dir()]

    assert runs_cli.main(["show", "--out", str(runs), run_dir.name]) == 0
    assert json.loads(capsys.readouterr().out) == expected

    assert runs_cli.main(["show", "--out", str(runs), "not-a-run"]) == 1
    assert "no result.json" in capsys.readouterr().err


def test_a_missing_index_is_not_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert runs_cli.main(["list", "--out", str(tmp_path / "empty")]) == 0
    assert INDEX_NAME in capsys.readouterr().err


# ---------- bedivere-data fetch ----------

JOB_ID = "GLBX-20260909-JAWY6ACQWL"
REQUEST = [
    "--dataset", "GLBX.MDP3", "--symbols", "NQ.FUT", "--schema", "ohlcv-1m",
    "--start", "2026-09-01", "--end", "2026-09-02",
]


@pytest.fixture
def vendor(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stand in for every batch call `fetch` can make, recording each by name."""
    from bedivere.data.vendor.databento import batch

    calls: list[str] = []
    job = batch.BatchJob.from_payload({"id": JOB_ID, "state": "done"})

    def record(name: str, result: Any) -> Any:
        def fake(*_args: Any, **_kwargs: Any) -> Any:
            calls.append(name)
            return result

        return fake

    monkeypatch.setattr(batch, "get_cost", record("get_cost", 0.0))
    monkeypatch.setattr(batch, "submit_job", record("submit_job", job))
    monkeypatch.setattr(batch, "wait_for_job", record("wait_for_job", job))
    monkeypatch.setattr(batch, "download_job", record("download_job", [Path("nq.dbn.zst")]))
    return calls


def test_fetch_job_downloads_an_existing_job_with_no_request_arguments(
    tmp_path: Path, vendor: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert data_cli.main(["fetch", "--job", JOB_ID, "--dest", str(tmp_path)]) == 0
    assert vendor == ["download_job"]  # nothing priced, nothing submitted
    assert capsys.readouterr().out == "nq.dbn.zst\n"


def test_submitting_without_the_request_arguments_refuses_before_any_call(
    tmp_path: Path, vendor: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["fetch", "--dataset", "GLBX.MDP3", "--cost-limit", "5", "--yes", "--dest", str(tmp_path)]
    assert data_cli.main(argv) == 1
    assert vendor == []
    # Names exactly the missing ones: --dataset was given.
    assert "error: --symbols, --schema, --start, --end required to submit a job" in capsys.readouterr().err


def test_submitting_without_wait_prints_the_job_id_and_succeeds(
    tmp_path: Path, vendor: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    # Downloading a job this new would find no files; the submission succeeded,
    # so the command has to as well — and say how to collect it.
    argv = ["fetch", *REQUEST, "--cost-limit", "5", "--yes", "--dest", str(tmp_path)]
    assert data_cli.main(argv) == 0
    assert vendor == ["get_cost", "submit_job"]
    out, err = capsys.readouterr()
    assert out == f"{JOB_ID}\n"
    assert f"download it later with --job {JOB_ID}" in err


def test_submitting_with_wait_waits_then_downloads(
    tmp_path: Path, vendor: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["fetch", *REQUEST, "--cost-limit", "5", "--yes", "--wait", "60", "--dest", str(tmp_path)]
    assert data_cli.main(argv) == 0
    assert vendor == ["get_cost", "submit_job", "wait_for_job", "download_job"]
    assert capsys.readouterr().out == "nq.dbn.zst\n"


def test_cost_still_requires_the_request_arguments() -> None:
    with pytest.raises(SystemExit) as e:
        data_cli.main(["cost", "--dataset", "GLBX.MDP3"])
    assert e.value.code == 2


def test_an_empty_range_is_refused_before_it_is_priced(
    tmp_path: Path, vendor: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    # --end is exclusive: --start X --end X asks for nothing, and the API would
    # price it at $0.00 — which reads as "free", not "empty".
    same_day = [*REQUEST[:-2], "--end", "2026-09-01"]
    assert data_cli.main(["cost", *same_day]) == 1
    assert "--end is exclusive" in capsys.readouterr().err

    # A bare date is UTC; a zoned datetime before it is still "not after".
    backwards = [*REQUEST[:-2], "--end", "2026-08-31T23:00Z"]
    argv = ["fetch", *backwards, "--cost-limit", "5", "--yes", "--dest", str(tmp_path)]
    assert data_cli.main(argv) == 1
    assert "--end is exclusive" in capsys.readouterr().err
    assert vendor == []  # nothing priced, nothing submitted


def test_a_range_the_cli_cannot_parse_is_left_to_the_api(vendor: list[str]) -> None:
    # UNIX nanoseconds are a documented form; the CLI does not second-guess them.
    unix_ns = [*REQUEST[:-4], "--start", "1756684800000000000", "--end", "1756771200000000000"]
    assert data_cli.main(["cost", *unix_ns]) == 0
    assert vendor == ["get_cost"]
