"""Archive builds publish only what a fully validated read produced."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from tests.lake_support import SKIP_REASON

pytest.importorskip("duckdb", reason=SKIP_REASON)
pytest.importorskip("pandas", reason=SKIP_REASON)
dbn = pytest.importorskip("databento_dbn", reason=SKIP_REASON)

from bedivere.data.lake import trades, volume_build  # noqa: E402
from bedivere.data.lake.ingest import IngestError, ZstdFrames  # noqa: E402
from bedivere.data.lake.volume import VolumeSpec  # noqa: E402
from bedivere.data.lake.volume_build import build_volume_archive  # noqa: E402
from bedivere.data.lake.volume_store import (  # noqa: E402
    StagedPartition,
    checked_volume_partition,
    volume_path,
)
from tests.test_trades import (  # noqa: E402
    DAY,
    DAYS,
    NEXT,
    NS,
    SID,
    add_receipt_trade,
    edit_receipt,
    prepared,
    seed_reference,
    trade,
    write,
    zstd,
)

SPEC = VolumeSpec(5)


@pytest.fixture(autouse=True)
def small_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deliver trades one at a time, so earlier sessions are staged before EOF."""
    monkeypatch.setattr(trades, "DEFAULT_BATCH_RECORDS", 1)


def two_sessions() -> list[Any]:
    return [
        trade(101 * NS, size=3),
        trade(102 * NS, size=4),
        trade(111 * NS, size=6),
        trade(112 * NS, size=2),
    ]


def snapshot(root: Path) -> dict[Path, bytes]:
    return {p: p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file() and "event_bars" in p.parts}


def leftovers(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if ".volume-staging-" in p.name or p.suffix == ".tmp"]


def test_a_late_failure_publishes_no_staged_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged: list[StagedPartition] = []
    real: Callable[..., StagedPartition] = volume_build.stage_volume_day

    def recording(*args: Any, **kwargs: Any) -> StagedPartition:
        staged.append(real(*args, **kwargs))
        assert staged[-1].staged.is_file() and not staged[-1].path.exists()
        return staged[-1]

    monkeypatch.setattr(volume_build, "stage_volume_day", recording)
    root = tmp_path / "new-lake"
    # Session 1 is staged long before the late record reaches EOF.
    path = write(tmp_path / "t.dbn", [*two_sessions(), trade(105 * NS, size=1)])
    with pytest.raises(ValueError, match="out of receive-time order"):
        build_volume_archive(path, root, SID, SPEC, DAYS, compare_1s=True)
    assert [s.path.parent.name for s in staged] == [f"session={DAY.label}"] * 2
    assert not root.exists()  # staging, and the root it had to create, are gone


def test_existing_partitions_survive_a_failed_read(tmp_path: Path) -> None:
    path = write(tmp_path / "t.dbn", two_sessions())
    build_volume_archive(path, tmp_path, SID, SPEC, DAYS, compare_1s=True)
    before = snapshot(tmp_path)
    assert len(before) == 4
    write(path, [*two_sessions(), trade(113 * NS, size=9), trade(110 * NS, size=1)])
    for resume in (False, True):
        with pytest.raises(ValueError, match="out of receive-time order"):
            build_volume_archive(path, tmp_path, SID, SPEC, DAYS, compare_1s=True, resume=resume)
        assert snapshot(tmp_path) == before and not leftovers(tmp_path)


def test_ineligible_trade_in_a_later_session_publishes_nothing(tmp_path: Path) -> None:
    records = two_sessions()
    records[2] = trade(111 * NS, size=6, flags=dbn.F_BAD_TS_RECV)
    path = write(tmp_path / "t.dbn", records)
    with pytest.raises(ValueError, match="F_BAD_TS_RECV"):
        build_volume_archive(path, tmp_path, SID, SPEC, DAYS)
    assert not (tmp_path / "event_bars").exists() and not leftovers(tmp_path)


def test_truncated_compressed_archive_publishes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "t.dbn.zst"
    write(path, two_sessions())
    path.write_bytes(path.read_bytes()[:-2])
    with pytest.raises(IngestError, match="incomplete trailing"):
        build_volume_archive(path, tmp_path, SID, SPEC, DAYS)
    assert not (tmp_path / "event_bars").exists() and not leftovers(tmp_path)


def test_resume_refuses_a_truncated_archive_it_would_not_have_decoded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "t.dbn.zst"
    write(path, two_sessions())
    path.write_bytes(path.read_bytes()[:-2])

    def unchecked(_frames: ZstdFrames) -> bool:
        return True

    with monkeypatch.context() as before:
        # Partitions as a build made them before frames were checked.
        before.setattr(ZstdFrames, "complete", unchecked)
        build_volume_archive(path, tmp_path, SID, SPEC, DAYS)
    kept = snapshot(tmp_path)
    assert len(kept) == 2
    # Their recorded fingerprint still matches the archive; only its frames say it is short.
    for resume in (True, False):
        with pytest.raises(IngestError, match="incomplete trailing zstd frame"):
            build_volume_archive(path, tmp_path, SID, SPEC, DAYS, resume=resume)
    assert snapshot(tmp_path) == kept and not leftovers(tmp_path)


def test_receipt_counts_found_wrong_at_eof_publish_nothing(tmp_path: Path) -> None:
    cache = prepared(tmp_path)
    root = tmp_path / "lake"
    reports = build_volume_archive(cache, root, SID, VolumeSpec(3), DAYS)
    assert [r["volume"] for r in reports] == [12, 7]
    before = snapshot(root)
    edit_receipt(cache, add_receipt_trade)  # consistent with itself, not with the archive
    with pytest.raises(IngestError, match="its receipt records 7"):
        build_volume_archive(cache, root, SID, VolumeSpec(3), DAYS)
    with pytest.raises(IngestError, match="its receipt records 7"):
        build_volume_archive(cache, root, SID, VolumeSpec(4), DAYS)
    assert snapshot(root) == before and not leftovers(root)
    # A receipt contradicting the digest fails even when resume skips decoding.
    edit_receipt(cache, lambda p: p.update(sha256="0" * 64))
    with pytest.raises(IngestError, match="SHA256 differs"):
        build_volume_archive(cache, root, SID, VolumeSpec(3), DAYS, resume=True)


def test_publication_is_atomic_per_partition_not_across_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    published: list[Path] = []
    real: Callable[[StagedPartition], Path] = volume_build.publish_staged

    def flaky(staged: StagedPartition) -> Path:
        if published:
            raise OSError("disk full")
        published.append(real(staged))
        return published[-1]

    monkeypatch.setattr(volume_build, "publish_staged", flaky)
    path = write(tmp_path / "t.dbn", two_sessions())
    with pytest.raises(OSError, match="disk full"):
        build_volume_archive(path, tmp_path, SID, SPEC, DAYS)
    # The validated first session stays published; the second never appears.
    assert published == [volume_path(tmp_path, SID, SPEC, DAYS, DAY.label)]
    assert checked_volume_partition(tmp_path, SID, SPEC, DAYS, DAY)[0].volume == 7
    assert not volume_path(tmp_path, SID, SPEC, DAYS, NEXT.label).exists()
    assert not leftovers(tmp_path)


def test_reports_say_what_was_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write(tmp_path / "t.dbn", two_sessions())
    first = build_volume_archive(path, tmp_path, SID, SPEC, DAYS, compare_1s=True)
    assert {(r["status"], r["source_check"]) for r in first} == {("written", "full_read")}
    dry = build_volume_archive(path, tmp_path, SID, SPEC, DAYS, dry_run=True)
    assert {(r["status"], r["source_check"]) for r in dry} == {("computed", "full_read")}

    def no_decode(*_args: object) -> None:
        raise AssertionError("current outputs must not be decoded again")

    monkeypatch.setattr(volume_build, "_selected_trades", no_decode)
    resumed = build_volume_archive(path, tmp_path, SID, SPEC, DAYS, compare_1s=True, resume=True)
    assert {(r["status"], r["source_check"]) for r in resumed} == {("unchanged", "fingerprint")}
    assert [r["session"] for r in resumed] == [DAY.label, DAY.label, NEXT.label, NEXT.label]


def test_a_selection_that_moves_mid_build_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = replace(SID, series="local.v.0")
    seed_reference(tmp_path, sid, DAY, [42])
    seed_reference(tmp_path, sid, NEXT, [42])
    path = write(tmp_path / "t.dbn", two_sessions())
    real = volume_build._fingerprint  # pyright: ignore[reportPrivateUsage]

    def stale(target: Path) -> str:
        # What the resume check saw before the reference partition was rewritten.
        return "bars.parquet:sha256:" + "0" * 64 if "tf=1s" in target.parts else real(target)

    monkeypatch.setattr(volume_build, "_fingerprint", stale)
    with pytest.raises(ValueError, match="reference 1s partition changed during the build"):
        build_volume_archive(path, tmp_path, sid, SPEC, DAYS)
    assert not (tmp_path / "event_bars").exists() and not leftovers(tmp_path)


def test_compressed_and_plain_archives_publish_identical_bars(tmp_path: Path) -> None:
    plain = write(tmp_path / "t.dbn", two_sessions())
    zipped = tmp_path / "t.dbn.zst"
    zipped.write_bytes(zstd(plain.read_bytes()))
    build_volume_archive(plain, tmp_path, SID, SPEC, DAYS)
    expected = [checked_volume_partition(tmp_path, SID, SPEC, DAYS, day) for day in DAYS.days]
    build_volume_archive(zipped, tmp_path, SID, SPEC, DAYS)
    assert [checked_volume_partition(tmp_path, SID, SPEC, DAYS, day) for day in DAYS.days] == expected
