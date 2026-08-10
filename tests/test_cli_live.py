"""The live CLI end to end: a shadow session from a spec, and the
operational layer around it.

The clock is injected, so a whole session runs at a chosen instant and every
assertion is exact. That injection is the ONLY difference between these runs
and a real one — same lock, same supervisor, same feed adapter, same
preflight, same archive.

The spec here carries no `window`. That is the design goal being tested: a
backtested spec IS the run config, and the live runner derives its own window
from now and the session-day close.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from bedivere.brokers.port import BrokerContext
from bedivere.brokers.sim import SimBroker, SimBrokerConfig
from bedivere.cli import live as live_cli
from bedivere.cli.common import CliError
from bedivere.run.archive import read_index
from bedivere.run.supervise import ACTIVE_NAME, STATUS_NAME, RunSupervisor, request_stop
from tests.helpers import et
from tests.strategy_fixture import FIRST_CLOSE_DATE, LAST_CLOSE_DATE
from tests.test_supervise import FakeClock

# Mid-session on 2026-07-15: the CME ETH day opened 18:00 the evening before
# and closes 17:00, so there are hours of bars ahead of "now" to trade.
NOW = et("2026-07-15T10:00")
SESSION_CLOSE = et("2026-07-15T17:00")

SPEC: dict[str, Any] = {
    "strategy": "tests.strategy_fixture:PLUGIN",
    "symbol": "DEMO",
    "config": {"kind": "ramp", "enter_after_bars": 2, "warmup_bars": 4},
    "baseTimeframe": "5m",
    "derivedTimeframes": ["30m"],
    "session": {
        "template": "cme_us_index_futures_eth",
        "timezone": "America/New_York",
        "openTime": "18:00",
        "closeTime": "17:00",
        "firstCloseDate": FIRST_CLOSE_DATE,
        "lastCloseDate": LAST_CLOSE_DATE,
    },
    "instruments": {"DEMO": {"tickSize": 0.25, "pointValue": 20}},
    "sim": {
        "latencyMs": 250,
        "halfSpreadTicks": 1,
        "commissionCentsPerSidePerContract": 105,
        "seed": 7,
        "deferProtectionOneBar": False,
    },
    "notify": {"transport": "off"},
    "data": {"source": "python", "factory": "tests.strategy_fixture:build_ramp_source"},
    "feed": {"factory": "tests.strategy_fixture:build_synchronous_feed"},
}


def write_spec(tmp_path: Path, **overrides: Any) -> Path:
    payload = {**json.loads(json.dumps(SPEC)), **overrides}
    path = tmp_path / "live-spec.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def live_args(spec: Path, runs: Path, **overrides: Any) -> argparse.Namespace:
    defaults: dict[str, Any] = {
        "spec": spec,
        "overrides": [],
        "runs_dir": runs,
        "note": "",
        "mode": "shadow",
        "until": None,
        "notify": None,
        "notify_on": None,
    }
    return argparse.Namespace(**{**defaults, **overrides})


def run_live_cli(spec: Path, runs: Path, *, now: int = NOW, **overrides: Any) -> int:
    return live_cli.run(live_args(spec, runs, **overrides), clock=FakeClock(now))


def stdout_json(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    return json.loads(capsys.readouterr().out)


# ---------- the happy path ----------


def test_a_shadow_session_runs_end_to_end_and_archives(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = tmp_path / "runs"
    assert run_live_cli(write_spec(tmp_path), runs) == 0

    result = stdout_json(capsys)
    assert result["mode"] == "shadow"
    assert result["window"] == {"startUnix": NOW, "endUnix": SESSION_CLOSE}
    assert result["feed"]["liveBars"] > 0
    assert result["feed"]["historyBars"] > 0  # warm-up was backfilled
    assert result["summary"]["trades"] > 0

    (run_dir,) = [p for p in runs.iterdir() if p.is_dir()]
    assert {p.name for p in run_dir.iterdir()} == {
        "result.json",
        "spec.json",
        "journal.jsonl",
        STATUS_NAME,
    }
    # The journal STREAMED during the run — it is the crash-safe copy, and the
    # archive must not have replaced it with one written from memory.
    lines = (run_dir / "journal.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == result["journal"]["events"]
    assert json.loads(lines[0])["mode"] == "shadow"

    rows = read_index(runs)
    assert len(rows) == 1 and rows[0]["mode"] == "shadow" and rows[0]["kind"] == "ramp"


def test_the_lock_is_released_on_a_clean_exit(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    assert run_live_cli(write_spec(tmp_path), runs) == 0
    assert not (runs / ACTIVE_NAME).exists()

    (run_dir,) = [p for p in runs.iterdir() if p.is_dir()]
    status = json.loads((run_dir / STATUS_NAME).read_text(encoding="utf-8"))
    assert status["phase"] == "stopped"
    assert status["stoppedBy"] == "window"
    # The phase the loop knows exactly — the first tradeable instant — is in
    # the terminal record. NOW is itself on the 5m grid, so it qualifies.
    assert status["firstTradeableBarTs"] >= NOW


def test_until_shortens_the_window(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_live_cli(write_spec(tmp_path), tmp_path / "runs", until="12:30") == 0
    assert stdout_json(capsys)["window"]["endUnix"] == et("2026-07-15T12:30")


def test_a_backtest_spec_needs_no_edit_to_run_live(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The whole design goal in one assertion: add a window (as a backtest
    must have) and the live runner still derives its own."""
    spec = write_spec(tmp_path, window={"firstTradeDate": "2026-07-15", "lastTradeDate": "2026-07-16"})
    assert run_live_cli(spec, tmp_path / "runs") == 0
    assert stdout_json(capsys)["window"] == {"startUnix": NOW, "endUnix": SESSION_CLOSE}


def test_the_phase_wrapper_is_invisible_in_the_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The live runner wraps the strategy to report its "running" phase. That
    wrapper must not show up in the envelope — a strategy's own counters ride
    through unchanged, and one WITHOUT counters must not acquire an empty
    block just because it ran live."""
    assert run_live_cli(write_spec(tmp_path), tmp_path / "runs") == 0
    result = stdout_json(capsys)
    assert result["strategy"]["submitted"] == 1
    assert result["strategy"]["fills"] >= 1


def test_the_shipped_replay_feed_drives_a_session(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`bedivere.streams.feed:build_replay_feed` is what a spec names to
    rehearse the live path before a real feed exists — pushing from its own
    thread, exactly as a websocket callback would."""
    spec = write_spec(tmp_path, feed={"factory": "bedivere.streams.feed:build_replay_feed"})
    assert run_live_cli(spec, tmp_path / "runs") == 0
    result = stdout_json(capsys)
    assert result["feed"]["liveBars"] > 0
    assert result["summary"]["trades"] > 0


def test_set_works_live_too(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runs = tmp_path / "runs"
    run_live_cli(write_spec(tmp_path), runs)
    base = stdout_json(capsys)
    run_live_cli(write_spec(tmp_path), runs, overrides=["rr=3.0"])
    swept = stdout_json(capsys)
    assert base["paramsHash"] != swept["paramsHash"]
    assert swept["params"]["rr"] == 3.0


# ---------- the one-run lock ----------


def test_two_concurrent_live_runs_are_refused(tmp_path: Path) -> None:
    """Two live runs on one symbol is a double position waiting for a signal.
    The lock makes it impossible rather than unlikely."""
    runs = tmp_path / "runs"
    clock = FakeClock(NOW)
    holder = RunSupervisor(runs_dir=runs, run_id="already-running", run_dir=runs / "x", clock=clock)
    holder.acquire(mode="shadow")

    with pytest.raises(CliError, match="another run is active: already-running"):
        live_cli.run(live_args(write_spec(tmp_path), runs), clock=clock)

    # The refused run archived nothing and did not disturb the holder.
    assert read_index(runs) == []
    assert json.loads((runs / ACTIVE_NAME).read_text(encoding="utf-8"))["runId"] == "already-running"


def test_a_dead_runs_lock_does_not_block_forever(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    dead_clock = FakeClock(NOW - 3600)
    RunSupervisor(
        runs_dir=runs, run_id="crashed", run_dir=runs / "crashed", clock=dead_clock
    ).acquire(mode="shadow")

    assert run_live_cli(write_spec(tmp_path), runs) == 0
    assert not (runs / ACTIVE_NAME).exists()


def test_a_failing_run_releases_the_lock(tmp_path: Path) -> None:
    """finalize is on every exit path, including the failing ones — a lock
    that outlives its run is a lock nobody can explain."""
    runs = tmp_path / "runs"
    spec = write_spec(tmp_path, feed={"factory": "tests.strategy_fixture:build_exploding_feed"})
    with pytest.raises(RuntimeError, match="exploding"):
        live_cli.run(live_args(spec, runs), clock=FakeClock(NOW))
    assert not (runs / ACTIVE_NAME).exists()


# ---------- the stop sentinel ----------


def test_the_stop_sentinel_ends_a_run_gracefully(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Touch a file, and the run archives instead of dying — no signal, no
    pid, nothing that needs to know how the run was started."""
    runs = tmp_path / "runs"
    spec = write_spec(
        tmp_path,
        feed={
            "factory": "tests.strategy_fixture:build_stopping_feed",
            "options": {"stop_after": 3, "runs_dir": str(runs)},
        },
    )
    assert run_live_cli(spec, runs) == 0

    result = stdout_json(capsys)
    assert result["feed"]["liveBars"] == 3  # ended early, not at the window end
    (run_dir,) = [p for p in runs.iterdir() if p.is_dir()]
    status = json.loads((run_dir / STATUS_NAME).read_text(encoding="utf-8"))
    assert status["phase"] == "stopped"
    assert status["stoppedBy"] == "sentinel"
    assert read_index(runs)[0]["runId"] == run_dir.name


def test_requesting_a_stop_with_no_run_is_harmless(tmp_path: Path) -> None:
    request_stop(tmp_path / "runs")
    assert run_live_cli(write_spec(tmp_path), tmp_path / "runs") == 0


# ---------- preflight ----------


def test_a_dirty_venue_refuses_the_run(tmp_path: Path) -> None:
    """The arming gate: a position or working order this run did not create
    is not something to trade around."""
    runs = tmp_path / "runs"
    spec = write_spec(
        tmp_path,
        broker={
            "factory": "tests.strategy_fixture:build_blocked_broker",
            "options": {"reason": "open DEMO position of 2 this run did not create"},
        },
    )
    with pytest.raises(CliError, match="refused to start"):
        live_cli.run(live_args(spec, runs, mode="paper"), clock=FakeClock(NOW))
    assert not (runs / ACTIVE_NAME).exists()
    assert read_index(runs) == []


def test_a_clean_venue_starts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    spec = write_spec(
        tmp_path, broker={"factory": "tests.strategy_fixture:build_blocked_broker"}
    )
    assert run_live_cli(spec, tmp_path / "runs", mode="paper") == 0
    assert stdout_json(capsys)["mode"] == "paper"


def test_the_sim_broker_has_nothing_to_reconcile() -> None:
    from bedivere.core.pricing import spec_from_handoff

    broker = SimBroker(
        spec_from_handoff("DEMO", 0.25, 20),
        SimBrokerConfig(
            bar_period_seconds=300,
            latency_ms=0,
            half_spread_ticks=0,
            commission_cents_per_side_per_contract=0,
            seed=1,
            defer_protection_one_bar=False,
        ),
    )
    assert broker.preflight() is None


# ---------- refusals that keep a real order path honest ----------


def test_a_real_mode_without_a_broker_adapter_is_refused(tmp_path: Path) -> None:
    """bedivere ships no venue adapter, and `--mode paper` must not quietly
    fall back to the sim venue — the label would then be a lie."""
    with pytest.raises(CliError, match="ships no venue adapter"):
        live_cli.run(live_args(write_spec(tmp_path), tmp_path / "runs", mode="paper"), clock=FakeClock(NOW))


def test_a_spec_without_a_feed_is_refused(tmp_path: Path) -> None:
    payload = json.loads(json.dumps(SPEC))
    del payload["feed"]
    path = tmp_path / "no-feed.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CliError, match="no bars would ever arrive"):
        live_cli.run(live_args(path, tmp_path / "runs"), clock=FakeClock(NOW))


def test_a_run_outside_any_session_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CliError, match="no session-day contains"):
        run_live_cli(write_spec(tmp_path), tmp_path / "runs", now=et("2026-07-15T17:30"))


def test_a_broker_context_carries_the_operators_label(tmp_path: Path) -> None:
    """The engine knows which BROKER it was given, never which ACCOUNT that
    broker points at — so the label rides through to the adapter."""
    seen: list[BrokerContext] = []
    spec = write_spec(
        tmp_path,
        broker={"factory": "tests.strategy_fixture:build_blocked_broker"},
    )
    from tests import strategy_fixture

    strategy_fixture.BROKER_CONTEXTS = seen
    try:
        assert run_live_cli(spec, tmp_path / "runs", mode="funded") == 0
    finally:
        strategy_fixture.BROKER_CONTEXTS = []
    assert [c.mode for c in seen] == ["funded"]
    assert seen[0].symbol == "DEMO"
