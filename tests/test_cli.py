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


def test_csv_import_round_trips_through_a_backtest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The path a cloner actually walks: some CSVs, one import, a backtest."""
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
    db = tmp_path / "candles.db"
    assert data_cli.main(
        ["import", str(csv), "--db", str(db), "--symbol", "DEMO", "--timeframe", "5m"]
    ) == 0
    capsys.readouterr()
    assert data_cli.main(["list", "--db", str(db)]) == 0
    assert "DEMO" in capsys.readouterr().out

    spec = write_spec(tmp_path, data={"source": "sqlite", "path": str(db)})
    assert run_backtest_cli(spec, tmp_path / "runs") == 0
    assert stdout_json(capsys)["summary"]["trades"] > 0


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
