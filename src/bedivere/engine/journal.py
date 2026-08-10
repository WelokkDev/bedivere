"""Decision journal — the run's event-level narrative.

Summary stats answer "how many"; the journal answers "why (not) at 10:30":
one JSON object per decision-relevant event — signals seen, entries taken,
entries blocked, setups expired, whatever kinds and fields the strategy
emits (scores, feature vectors, indicator readings), plus the venue
outcomes the loop journals automatically — appended in loop order, so the
file reads as the run's timeline. Collection is always on (a month is ~a
thousand small dicts); writing is the composition's choice (`journal_path`
on the runner).

Events are pure functions of bar data, so the journal is as deterministic
as the result JSON: same spec + same bars → byte-identical JSONL. The
run-level `context` is stamped into every line at write time — deterministic
fields ONLY (a wall-clock run id belongs in the directory name, never in a
line).
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class DecisionJournal:
    events: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    # Run-level provenance merged into every written line (an event field of
    # the same name wins). Kept OFF the in-memory events so emit stays lean.
    context: dict[str, object] = field(default_factory=dict[str, object])
    # Live observability tap (composition-supplied; backtests leave it None).
    # Gets a context-merged copy after the event is recorded; failures are
    # contained — observability must never sink a trading decision.
    sink: Callable[[dict[str, object]], None] | None = None

    def emit(self, kind: str, ts: int, **fields: object) -> None:
        event: dict[str, object] = {"kind": kind, "ts": ts, **fields}
        self.events.append(event)
        if self.sink is not None:
            try:
                self.sink({**self.context, **event})
            except Exception as e:  # noqa: BLE001 — the tap may not break the loop
                sys.stderr.write(f"[journal] sink failed on {kind}: {e}\n")

    def serialized_lines(self) -> Iterator[str]:
        """THE canonical serialization. `write_jsonl` and `sha256_hex` share
        it, so the digest covers exactly the bytes on disk by construction
        rather than by two implementations agreeing."""
        for event in self.events:
            line = {**self.context, **event} if self.context else event
            yield json.dumps(line, sort_keys=True, separators=(",", ":")) + "\n"

    def sha256_hex(self) -> str:
        """Digest of the canonical JSONL, stamped into the result so
        `resultHash` covers the journal too.

        Without it a fingerprint sees trades and counters only: a refactor
        that changed a rejection reason or reordered emissions, leaving the
        trade list untouched, would reproduce the old hash exactly.
        """
        digest = hashlib.sha256()
        for line in self.serialized_lines():
            digest.update(line.encode("utf-8"))
        return digest.hexdigest()

    def write_jsonl(self, path: Path) -> int:
        """One sorted-keys JSON object per line; returns the event count."""
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for line in self.serialized_lines():
                f.write(line)
        return len(self.events)
