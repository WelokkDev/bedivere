"""CI gate: no wall-clock reads anywhere under bedivere/.

The engine's only time source is the injected Clock; the ONLY module allowed
to touch the wall is bedivere/core/clock.py (LiveClock internals). This is
a plain textual scan, kept deliberately dumb so it cannot be argued with.

If this test fails on a new module, the fix is to take a Clock (or an
explicit `now` value handed over by the caller), never to extend the
allow-list — LiveClock is the single point where wall time enters bedivere.
"""

from __future__ import annotations

from pathlib import Path

import bedivere

# Textual patterns whose presence means a wall-clock (or process-clock) read.
_BANNED = (
    "time.time(",
    "time.time_ns(",
    "time.monotonic",
    "time.perf_counter",
    "time.localtime(",
    "time.gmtime(",
    "datetime.now(",
    "datetime.utcnow(",
    "date.today(",
    ".today()",
)

# The one module allowed to read the wall (posix-style relative path).
_ALLOWED = frozenset({"core/clock.py"})


def test_no_wall_clock_reads_outside_clock_py() -> None:
    package_root = Path(bedivere.__file__).resolve().parent
    offenders: list[str] = []

    for path in sorted(package_root.rglob("*.py")):
        rel = path.relative_to(package_root).as_posix()
        if rel in _ALLOWED:
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            for pattern in _BANNED:
                if pattern in line:
                    offenders.append(f"bedivere/{rel}:{line_no}: {stripped} [{pattern}]")

    assert not offenders, (
        "wall-clock reads outside core/clock.py (inject a Clock or take `now` as data):\n"
        + "\n".join(offenders)
    )
