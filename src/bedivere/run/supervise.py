"""Run supervision — the file protocol between a runner and any launcher.

    <runs-dir>/ACTIVE.json   the one-run lock AND the live status snapshot
    <runs-dir>/STOP          sentinel: create it and the run stops gracefully

Files only, no process handles. A shell, a cron job, a container supervisor
and a remote host all speak this protocol unchanged, and every one of them
can read a run's status without being the thing that started it.

ONE file, not two. `ACTIVE.json` is simultaneously the lock (its existence)
and the report (its contents) — which means a lock can never exist without a
status to explain it, and the launcher that finds a lock finds out why in the
same read:

    {"runId": "...", "runDir": "...", "pid": 4812, "startedAtUnix": ...,
     "phase": "running", "heartbeatUnix": ..., "mode": "shadow",
     "snapshot": {"trades": 3, "openPositions": 1, "feed": {...}}}

LIVENESS IS HEARTBEAT AGE, never pid probing. `os.kill(pid, 0)` is a liveness
check on POSIX and TERMINATES the target on Windows, and a pid is meaningless
across a container boundary anyway. A lock whose heartbeat has gone quiet for
`STALE_AFTER_S`, or whose phase is terminal, belongs to a dead run and may be
taken over; a fresh one refuses the start. Takeover re-reads after writing to
confirm it actually won — a small CAS window that is acceptable for a single
operator on one machine and should be revisited before anything fans out.

The stop sentinel lives at the RUNS ROOT rather than inside the run
directory, because the lock already guarantees there is at most one run to
stop: `touch runs/STOP` needs no knowledge of a directory name that did not
exist when you decided to stop it. A leftover sentinel is cleared when a run
acquires the lock — otherwise yesterday's stop request would kill today's run
in its first second.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from bedivere.core.clock import Clock
from bedivere.run.archive import write_json_atomic

ACTIVE_NAME = "ACTIVE.json"
STOP_NAME = "STOP"
# The terminal status, written into the RUN directory as the lock is released.
# ACTIVE.json must not survive its run — but the reason a run ended should
# survive with the run, and "the lock vanished" is not a diagnosis.
STATUS_NAME = "status.json"

# Heartbeats flush every ~10s; 90s of silence means the holder is presumed
# dead. Generous on purpose — a paused debugger or a stalled disk should not
# hand your lock to a second run.
STALE_AFTER_S = 90
HEARTBEAT_EVERY_S = 10
POLL_EVERY_S = 1.0

# The phases the shipped runners report. Not enforced — this file is a
# report, not a state machine the library polices — but a launcher can rely
# on these names appearing in this order.
PHASES = (
    "starting",
    "connecting",
    "backfilling",
    "warming",
    "running",
    "stopping",
    "stopped",
    "failed",
)
TERMINAL_PHASES = frozenset({"stopped", "failed"})


class ActiveRunError(RuntimeError):
    """Another run holds the lock and its heartbeat is fresh."""

    def __init__(self, run_id: str, phase: str, heartbeat_age_s: int) -> None:
        super().__init__(
            f"another run is active: {run_id} (phase {phase}, heartbeat {heartbeat_age_s}s ago) — "
            f"stop it first (touch <runs-dir>/{STOP_NAME}) or wait for it to finish"
        )
        self.run_id = run_id
        self.phase = phase


@dataclass(frozen=True, slots=True)
class ActiveRun:
    """A parsed ACTIVE.json. Unknown/missing fields degrade to placeholders —
    supervision must answer "unknown", never raise, or a corrupt status file
    becomes an outage in every tool that reads it."""

    run_id: str
    run_dir: str
    pid: int
    started_at_unix: int
    phase: str
    heartbeat_unix: int
    raw: dict[str, Any] = field(default_factory=dict[str, Any])


def read_json(path: Path) -> dict[str, Any] | None:
    """Tolerant read: missing or corrupt is None, never an exception."""
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return cast(dict[str, Any], raw) if isinstance(raw, dict) else None


def read_active(runs_dir: Path) -> ActiveRun | None:
    """The current lock holder, or None when the lock is free."""
    raw = read_json(runs_dir / ACTIVE_NAME)
    if raw is None:
        return None
    run_id = raw.get("runId")
    if not isinstance(run_id, str):
        return None
    return ActiveRun(
        run_id=run_id,
        run_dir=raw["runDir"] if isinstance(raw.get("runDir"), str) else "",
        pid=raw["pid"] if isinstance(raw.get("pid"), int) else -1,
        started_at_unix=raw["startedAtUnix"] if isinstance(raw.get("startedAtUnix"), int) else 0,
        phase=raw["phase"] if isinstance(raw.get("phase"), str) else "?",
        heartbeat_unix=raw["heartbeatUnix"] if isinstance(raw.get("heartbeatUnix"), int) else 0,
        raw=raw,
    )


def is_stale(active: ActiveRun | None, now_unix: int, *, threshold_s: int = STALE_AFTER_S) -> bool:
    """A lock that is unreadable, terminal, or heartbeat-silent is dead."""
    if active is None:
        return True
    if active.phase in TERMINAL_PHASES:
        return True
    return now_unix - active.heartbeat_unix > threshold_s


def request_stop(runs_dir: Path) -> Path:
    """Create the stop sentinel — the graceful-stop request any process can
    make without a signal or a pid."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / STOP_NAME
    path.write_text("stop\n", encoding="utf-8")
    return path


class RunSupervisor:
    """Owns the lock, the status file, the heartbeat thread and the stop
    sentinel for exactly one run.

    Thread-safety: `_state` mutates under `_lock`. The snapshot callable reads
    ENGINE objects from the supervisor thread — plain int/dict reads that the
    GIL makes atomic individually. That is observability-grade, not
    transactional: a snapshot may catch a trade count from just before a fill
    and a position count from just after. Reporting is allowed to be a beat
    behind; it is not allowed to lock the trading loop to be exact.
    """

    def __init__(
        self,
        *,
        runs_dir: Path,
        run_id: str,
        run_dir: Path,
        clock: Clock,
        poll_every_s: float = POLL_EVERY_S,
    ) -> None:
        self._runs_dir = runs_dir
        self._run_id = run_id
        self._run_dir = run_dir
        self._clock = clock
        self._poll_every_s = poll_every_s
        # Wired post-construction: both point at objects built AFTER the lock
        # is taken (there is no point building a run you may not be allowed
        # to start).
        self.on_stop_requested: Callable[[], None] | None = None
        self.snapshot: Callable[[], dict[str, Any]] | None = None
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {}
        self._thread: threading.Thread | None = None
        self._shutdown = threading.Event()
        self._stop_seen = False
        self._finalized = False

    # ---------- lock ----------

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    @property
    def active_path(self) -> Path:
        return self._runs_dir / ACTIVE_NAME

    @property
    def stop_path(self) -> Path:
        return self._runs_dir / STOP_NAME

    @property
    def stop_requested(self) -> bool:
        """True once the sentinel has been seen — the runner reports WHY it
        ended, and "the operator asked" is a different ending from "the
        window closed"."""
        return self._stop_seen

    def acquire(self, **initial: Any) -> None:
        """Take the one-run lock or raise `ActiveRunError`.

        Exclusive create (`open(..., "x")`) is the fast path and the only one
        with no race at all. A pre-existing lock is inspected: fresh means
        refuse, stale means take over and confirm.
        """
        now = self._clock.now_unix()
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._state = {
            **initial,
            "runId": self._run_id,
            "runDir": str(self._run_dir),
            "pid": os.getpid(),
            "phase": "starting",
            "startedAtUnix": now,
            "heartbeatUnix": now,
        }
        payload = json.dumps(self._state, sort_keys=True, indent=2)
        try:
            with self.active_path.open("x", encoding="utf-8") as f:
                f.write(payload)
            self._clear_stale_sentinel()
            return
        except FileExistsError:
            pass

        existing = read_active(self._runs_dir)
        if not is_stale(existing, now):
            assert existing is not None  # is_stale(None) is True
            raise ActiveRunError(
                existing.run_id, existing.phase, max(now - existing.heartbeat_unix, 0)
            )
        # Stale or corrupt holder: take over, then confirm we actually won the
        # race against any other process doing the same thing right now.
        write_json_atomic(self.active_path, self._state)
        confirm = read_active(self._runs_dir)
        if confirm is None or confirm.run_id != self._run_id:
            raise ActiveRunError(confirm.run_id if confirm else "?", "starting", 0)
        if existing is not None:
            sys.stderr.write(
                f"[supervise] took over a stale lock from {existing.run_id} "
                f"(phase {existing.phase})\n"
            )
        self._clear_stale_sentinel()

    def _clear_stale_sentinel(self) -> None:
        """A sentinel left by a previous run is not a request to stop THIS
        one. Clearing it at acquire is the only moment the distinction is
        unambiguous."""
        try:
            if self.stop_path.exists():
                self.stop_path.unlink()
                sys.stderr.write(f"[supervise] cleared a leftover {STOP_NAME} sentinel\n")
        except OSError:
            pass

    # ---------- status ----------

    def set_phase(self, phase: str, **extra: Any) -> None:
        with self._lock:
            self._state["phase"] = phase
            self._state.update(extra)
            self._state["heartbeatUnix"] = self._clock.now_unix()
            self._flush()

    def note(self, **fields: Any) -> None:
        with self._lock:
            self._state.update(fields)
            self._flush()

    def state(self) -> dict[str, Any]:
        """A copy of the current status — what ACTIVE.json holds right now."""
        with self._lock:
            return dict(self._state)

    def _flush(self) -> None:
        """Write the status — but ONLY while we still hold the lock.

        A supervisor that stalled long enough to be taken over is still a
        live thread with a stale `_state`. Without this check its next
        heartbeat overwrites the successor's pointer with its own runId, and
        its `finalize` then reads that pointer, recognises itself, and
        deletes the lock of the run that legitimately replaced it. Losing a
        stalled run's last status line is the cheap half of that trade.
        """
        try:
            holder = read_active(self._runs_dir)
            if holder is not None and holder.run_id != self._run_id:
                return
            write_json_atomic(self.active_path, self._state)
        except OSError as e:
            sys.stderr.write(f"[supervise] status flush failed: {e}\n")

    # ---------- heartbeat / sentinel thread ----------

    def start(self) -> None:
        """Begin the background supervision loop. Daemon: a supervisor thread
        must never be the reason a finished process refuses to exit."""
        if self._thread is not None:
            raise RuntimeError("RunSupervisor.start() called twice")
        self._thread = threading.Thread(
            target=self._run, name="bedivere-supervisor", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        ticks = 0
        while not self._shutdown.wait(self._poll_every_s):
            ticks += 1
            self.tick(heartbeat=ticks % HEARTBEAT_EVERY_S == 0)

    def tick(self, *, heartbeat: bool) -> None:
        """One supervision step. The thread loop calls this; tests drive it
        directly, which is why the sentinel check and the heartbeat are one
        synchronous method rather than two threads' worth of timing."""
        if not self._stop_seen and self.stop_path.exists():
            self._stop_seen = True
            self.set_phase("stopping", stopRequestedBy="sentinel")
            callback = self.on_stop_requested
            if callback is not None:
                try:
                    callback()
                except Exception as e:  # noqa: BLE001 — supervision may not kill the run
                    sys.stderr.write(f"[supervise] stop callback failed: {e}\n")
        if heartbeat:
            with self._lock:
                self._state["heartbeatUnix"] = self._clock.now_unix()
                if self.snapshot is not None:
                    try:
                        self._state["snapshot"] = self.snapshot()
                    except Exception as e:  # noqa: BLE001 — a bad snapshot is not a bad run
                        sys.stderr.write(f"[supervise] snapshot failed: {e}\n")
                self._flush()

    # ---------- teardown ----------

    def finalize(self, status: str, **extra: Any) -> None:
        """Terminal status, then release the lock. Idempotent — the first call
        wins — so it is safe on every exit path, and it MUST be on every exit
        path: a lock outliving its run is a lock nobody can explain.

        The lock is released only if we still hold it. A supervisor that was
        taken over while stalled must not delete the pointer of the run that
        legitimately replaced it.
        """
        if self._finalized:
            return
        self._finalized = True
        self._shutdown.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self.set_phase(status, **extra)
        try:
            write_json_atomic(self._run_dir / STATUS_NAME, self.state())
        except OSError as e:
            sys.stderr.write(f"[supervise] terminal status not written: {e}\n")
        holder = read_active(self._runs_dir)
        if holder is not None and holder.run_id == self._run_id:
            for path in (self.active_path, self.stop_path):
                try:
                    path.unlink(missing_ok=True)
                except OSError as e:
                    sys.stderr.write(f"[supervise] could not remove {path.name}: {e}\n")
