"""The journal's sha256, and why the result envelope carries it.

`resultHash` without it sees only trades and counters. A refactor that
changed a rejection reason, renamed a field, or reordered emissions — while
leaving the trade list untouched — would reproduce the old hash exactly, and
the one artifact that explains WHY the run did what it did would drift with
nothing to notice.
"""

from __future__ import annotations

from pathlib import Path

from bedivere.engine.journal import DecisionJournal


def _journal(**context: object) -> DecisionJournal:
    j = DecisionJournal(context=dict(context))
    j.emit("signal", 100, score=0.5)
    j.emit("blocked_in_position", 160)
    return j


def test_the_digest_covers_exactly_the_bytes_on_disk(tmp_path: Path) -> None:
    """`write_jsonl` and `sha256_hex` share one serializer, so this holds by
    construction rather than by two implementations agreeing."""
    import hashlib

    j = _journal(kind="demo")
    path = tmp_path / "journal.jsonl"
    j.write_jsonl(path)

    assert j.sha256_hex() == hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_digest_is_stable_across_identical_journals() -> None:
    assert _journal().sha256_hex() == _journal().sha256_hex()


def test_a_changed_field_moves_the_digest() -> None:
    a, b = _journal(), DecisionJournal()
    b.emit("signal", 100, score=0.6)  # one value differs
    b.emit("blocked_in_position", 160)
    assert a.sha256_hex() != b.sha256_hex()


def test_a_renamed_kind_moves_the_digest() -> None:
    """The case a trade-list comparison is blind to: same trades, different
    explanation."""
    a, b = _journal(), DecisionJournal()
    b.emit("signal", 100, score=0.5)
    b.emit("blocked", 160)  # renamed reason
    assert a.sha256_hex() != b.sha256_hex()


def test_reordering_moves_the_digest() -> None:
    a, b = _journal(), DecisionJournal()
    b.emit("blocked_in_position", 160)
    b.emit("signal", 100, score=0.5)
    assert a.sha256_hex() != b.sha256_hex()


def test_run_level_context_is_part_of_the_digest() -> None:
    """Context is stamped into every written line, so it is part of the
    bytes and must be part of the hash."""
    assert _journal().sha256_hex() != _journal(paramsHash="abc123").sha256_hex()


def test_an_event_field_beats_a_context_field_of_the_same_name() -> None:
    """Documented precedence, and worth pinning because it is invisible: a
    context `kind` never reaches the file, so it cannot move the digest
    either. Name run-level provenance so it cannot collide with an event."""
    assert _journal().sha256_hex() == _journal(kind="demo").sha256_hex()


def test_an_empty_journal_has_the_empty_digest() -> None:
    import hashlib

    assert DecisionJournal().sha256_hex() == hashlib.sha256(b"").hexdigest()
