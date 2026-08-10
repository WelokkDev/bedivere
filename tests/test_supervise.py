"""Supervision: the one-run lock, the stale takeover, the stop sentinel, and
a finalize that runs on every exit path.

The lock is the safety property — two live runs on one symbol is a
double-position waiting for a signal — so the tests are written as the four
ways it could go wrong: it lets a second run in, it refuses forever after a
crash, it obeys a sentinel nobody created for it, or it outlives the run that
took it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bedivere.run.supervise import (
    ACTIVE_NAME,
    STALE_AFTER_S,
    STATUS_NAME,
    STOP_NAME,
    ActiveRunError,
    RunSupervisor,
    is_stale,
    read_active,
    request_stop,
)


class FakeClock:
    """A clock the test moves by hand — heartbeat age is the liveness signal,
    so the tests need to control it exactly."""

    def __init__(self, now: int = 1_785_000_000) -> None:
        self.unix = now

    def now_unix(self) -> int:
        return self.unix

    def now_ms(self) -> int:
        return self.unix * 1000


def supervisor(runs: Path, run_id: str, clock: FakeClock) -> RunSupervisor:
    return RunSupervisor(runs_dir=runs, run_id=run_id, run_dir=runs / run_id, clock=clock)


# ---------- the lock ----------


def test_acquiring_writes_the_status_and_the_run_directory(tmp_path: Path) -> None:
    clock = FakeClock()
    sup = supervisor(tmp_path, "run-1", clock)
    sup.acquire(mode="shadow", symbol="DEMO")

    active = read_active(tmp_path)
    assert active is not None
    assert active.run_id == "run-1"
    assert active.phase == "starting"
    assert active.heartbeat_unix == clock.unix
    assert active.raw["mode"] == "shadow"
    assert (tmp_path / "run-1").is_dir()


def test_a_second_run_is_refused_while_the_first_is_alive(tmp_path: Path) -> None:
    clock = FakeClock()
    supervisor(tmp_path, "run-1", clock).acquire()
    with pytest.raises(ActiveRunError, match="another run is active: run-1"):
        supervisor(tmp_path, "run-2", clock).acquire()


def test_a_stale_lock_is_taken_over(tmp_path: Path) -> None:
    """A crashed run leaves its lock behind. Liveness is heartbeat AGE, never
    pid probing — os.kill(pid, 0) terminates the target on Windows."""
    clock = FakeClock()
    supervisor(tmp_path, "dead", clock).acquire()

    clock.unix += STALE_AFTER_S + 1
    taken = supervisor(tmp_path, "fresh", clock)
    taken.acquire()

    active = read_active(tmp_path)
    assert active is not None and active.run_id == "fresh"


def test_a_terminal_lock_is_stale_regardless_of_age(tmp_path: Path) -> None:
    clock = FakeClock()
    finished = supervisor(tmp_path, "run-1", clock)
    finished.acquire()
    # Simulate a finalize that set the phase but could not remove the file.
    finished.set_phase("stopped")
    (tmp_path / ACTIVE_NAME).write_text(
        json.dumps({**finished.state(), "phase": "stopped"}), encoding="utf-8"
    )
    supervisor(tmp_path, "run-2", clock).acquire()  # must not raise
    active = read_active(tmp_path)
    assert active is not None and active.run_id == "run-2"


def test_a_corrupt_lock_is_stale(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / ACTIVE_NAME).write_text("{not json", encoding="utf-8")
    supervisor(tmp_path, "run-1", FakeClock()).acquire()  # must not raise
    assert is_stale(None, 0)


def test_finalize_releases_the_lock_and_is_idempotent(tmp_path: Path) -> None:
    clock = FakeClock()
    sup = supervisor(tmp_path, "run-1", clock)
    sup.acquire()
    sup.finalize("stopped", summary={"trades": 2})
    assert not (tmp_path / ACTIVE_NAME).exists()
    sup.finalize("failed")  # first call wins; no resurrection
    assert not (tmp_path / ACTIVE_NAME).exists()
    # ...and the next run walks straight in.
    supervisor(tmp_path, "run-2", clock).acquire()


def test_finalize_after_a_takeover_does_not_delete_the_new_holders_lock(tmp_path: Path) -> None:
    """A supervisor that was taken over while stalled must not release a lock
    it no longer holds."""
    clock = FakeClock()
    stalled = supervisor(tmp_path, "stalled", clock)
    stalled.acquire()
    clock.unix += STALE_AFTER_S + 1
    supervisor(tmp_path, "successor", clock).acquire()

    stalled.finalize("failed")
    active = read_active(tmp_path)
    assert active is not None and active.run_id == "successor"


def test_a_failing_run_still_releases(tmp_path: Path) -> None:
    clock = FakeClock()
    sup = supervisor(tmp_path, "run-1", clock)
    sup.acquire()
    with pytest.raises(RuntimeError):
        try:
            raise RuntimeError("strategy exploded")
        finally:
            sup.finalize("failed", error="strategy exploded")
    assert not (tmp_path / ACTIVE_NAME).exists()
    # The lock is gone; the explanation is not. A terminal status lands in the
    # RUN directory, because "the lock vanished" is not a diagnosis.
    status = json.loads((tmp_path / "run-1" / STATUS_NAME).read_text(encoding="utf-8"))
    assert status["phase"] == "failed"
    assert status["error"] == "strategy exploded"


# ---------- status ----------


def test_the_snapshot_lands_in_the_status_file(tmp_path: Path) -> None:
    clock = FakeClock()
    sup = supervisor(tmp_path, "run-1", clock)
    sup.acquire()
    sup.snapshot = lambda: {"trades": 3, "openPositions": 1}
    sup.set_phase("running")
    sup.tick(heartbeat=True)

    active = read_active(tmp_path)
    assert active is not None
    assert active.phase == "running"
    assert active.raw["snapshot"] == {"trades": 3, "openPositions": 1}


def test_a_broken_snapshot_does_not_break_the_run(tmp_path: Path) -> None:
    sup = supervisor(tmp_path, "run-1", FakeClock())
    sup.acquire()

    def explode() -> dict[str, object]:
        raise RuntimeError("nope")

    sup.snapshot = explode
    sup.tick(heartbeat=True)  # must not raise
    assert read_active(tmp_path) is not None


# ---------- the stop sentinel ----------


def test_the_sentinel_requests_a_graceful_stop(tmp_path: Path) -> None:
    sup = supervisor(tmp_path, "run-1", FakeClock())
    sup.acquire()
    stopped: list[str] = []
    sup.on_stop_requested = lambda: stopped.append("closed")

    sup.tick(heartbeat=False)
    assert stopped == [] and not sup.stop_requested

    request_stop(tmp_path)
    sup.tick(heartbeat=False)
    assert stopped == ["closed"]
    assert sup.stop_requested
    active = read_active(tmp_path)
    assert active is not None and active.phase == "stopping"

    sup.tick(heartbeat=False)  # fires once, not once per poll
    assert stopped == ["closed"]


def test_a_leftover_sentinel_does_not_kill_the_next_run(tmp_path: Path) -> None:
    """Yesterday's stop request must not end today's run in its first
    second."""
    request_stop(tmp_path)
    sup = supervisor(tmp_path, "run-1", FakeClock())
    sup.acquire()
    assert not (tmp_path / STOP_NAME).exists()
    sup.tick(heartbeat=False)
    assert not sup.stop_requested


def test_finalize_clears_the_sentinel(tmp_path: Path) -> None:
    sup = supervisor(tmp_path, "run-1", FakeClock())
    sup.acquire()
    request_stop(tmp_path)
    sup.tick(heartbeat=False)
    sup.finalize("stopped")
    assert not (tmp_path / STOP_NAME).exists()
