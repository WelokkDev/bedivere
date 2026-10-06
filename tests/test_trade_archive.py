"""MBO extraction does not double-count resting fills or midnight snapshots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest

from tests.lake_support import SKIP_REASON

pytest.importorskip("duckdb", reason=SKIP_REASON)
pytest.importorskip("pandas", reason=SKIP_REASON)
dbn = pytest.importorskip("databento_dbn", reason=SKIP_REASON)

from bedivere.data.lake.ingest import (  # noqa: E402
    IngestError,
    ZstdFrames,
    read_metadata,
    stream_dbn,
)
from bedivere.data.lake.trade_archive import prepare_trade_archive  # noqa: E402
from tests.test_trades import NS, bare_decode, zstd  # noqa: E402
from tests.test_trades import header as dbn_header  # noqa: E402
from tests.test_volume_lake import archive  # noqa: E402

TRADES = 20_000


def mbo(path: Path, start: int = 100, end: int = 110) -> Path:
    archive(path, [], schema=dbn.Schema.MBO, start=start * 10**9, end=end * 10**9)
    header = path.read_bytes()
    records = [
        dbn.MBOMsg(
            publisher_id=1,
            instrument_id=42,
            ts_event=(start + 1) * 10**9,
            ts_recv=(start + 1) * 10**9,
            order_id=i,
            price=10 * 10**9,
            size=7,
            action=action,
            side=dbn.Side.BID,
            flags=flags,
        )
        for i, (action, flags) in enumerate(
            [
                (dbn.Action.ADD, dbn.F_SNAPSHOT),
                (dbn.Action.TRADE, 0),
                (dbn.Action.FILL, 0),
                (dbn.Action.CANCEL, 0),
                (dbn.Action.TRADE, dbn.F_SNAPSHOT),
            ]
        )
    ]
    path.write_bytes(header + b"".join(bytes(r) for r in records))
    return path


def test_only_trade_actions_contribute_and_cache_reuse_is_verified(tmp_path: Path) -> None:
    path = mbo(tmp_path / "input.dbn")
    target = prepare_trade_archive(path, tmp_path / "cache")
    trades = [cast(Any, r) for r in stream_dbn(target) if isinstance(r, dbn.TradeMsg)]
    assert len(trades) == 1 and trades[0].size == 7
    before = target.stat().st_mtime_ns
    assert prepare_trade_archive(path, tmp_path / "cache") == target
    assert target.stat().st_mtime_ns == before
    target.write_bytes(b"corrupt cache")
    assert prepare_trade_archive(path, tmp_path / "cache") == target
    assert len([r for r in stream_dbn(target) if isinstance(r, dbn.TradeMsg)]) == 1


def test_daily_inputs_join_without_losing_metadata_or_boundary_volume(tmp_path: Path) -> None:
    inputs = tmp_path / "input"
    inputs.mkdir()
    mbo(inputs / "first.dbn", 100, 110)
    mbo(inputs / "second.dbn", 110, 120)
    target = prepare_trade_archive(inputs, tmp_path / "cache")
    assert read_metadata(target).start == 100 * 10**9
    assert read_metadata(target).end == 120 * 10**9
    assert "NQZ5" in read_metadata(target).mappings
    assert sum(cast(Any, r).size for r in stream_dbn(target) if isinstance(r, dbn.TradeMsg)) == 14


@pytest.mark.parametrize("start", [109, 111])
def test_overlapping_and_missing_daily_ranges_are_refused(tmp_path: Path, start: int) -> None:
    inputs = tmp_path / "input"
    inputs.mkdir()
    mbo(inputs / "first.dbn", 100, 110)
    mbo(inputs / "second.dbn", start, 120)
    with pytest.raises(IngestError, match="overlap or have a gap"):
        prepare_trade_archive(inputs, tmp_path / "cache")


def test_vendor_manifest_is_checked_before_extraction(tmp_path: Path) -> None:
    path = mbo(tmp_path / "input.dbn")
    manifest = tmp_path / "manifest.json"
    payload = {
        "files": [
            {
                "filename": path.name,
                "size": path.stat().st_size,
                "hash": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        ]
    }
    manifest.write_text(json.dumps(payload))
    target = prepare_trade_archive(path, tmp_path / "cache")
    receipt = json.loads(target.with_suffix(target.suffix + ".json").read_text())
    assert receipt["policy"]["inputs"][0]["manifest_verified"]
    path.write_bytes(path.read_bytes() + b"broken")
    with pytest.raises(IngestError, match="vendor manifest"):
        prepare_trade_archive(path, tmp_path / "cache")


def compressed_mbo() -> bytes:
    """One compressed MBO day of `TRADES` one-lot trades, several zstd blocks long."""
    records = b"".join(
        bytes(
            dbn.MBOMsg(
                publisher_id=1,
                instrument_id=42,
                ts_event=101 * NS + i,
                ts_recv=101 * NS + i,
                order_id=i,
                price=10 * NS,
                size=1,
                action=dbn.Action.TRADE,
                side=dbn.Side.BID,
            )
        )
        for i in range(TRADES)
    )
    return zstd(dbn_header(dbn.Schema.MBO, 100 * NS, 110 * NS) + records)


def cut_points(whole: bytes) -> list[int]:
    return [len(whole) * k // 24 for k in range(1, 24)] + [len(whole) - 1]


def silently_short(whole: bytes) -> list[int]:
    """Cuts the vendor decoder alone takes for a complete, smaller file: every
    block that arrived decodes, and the last of them ends on a record boundary."""
    return [
        cut
        for cut in cut_points(whole)
        for records, leftover in [bare_decode(whole[:cut])]
        if 1 < records <= TRADES and not leftover
    ]


def trades_in(path: Path) -> int:
    return sum(isinstance(record, dbn.TradeMsg) for record in stream_dbn(path))


def test_a_compressed_source_cut_short_is_refused_wherever_it_is_cut(tmp_path: Path) -> None:
    whole = compressed_mbo()
    source = tmp_path / "day.mbo.dbn.zst"
    source.write_bytes(whole)
    assert trades_in(prepare_trade_archive(source, tmp_path / "whole")) == TRADES
    silent = silently_short(whole)
    assert silent  # or this fixture no longer holds the dangerous case
    for cut in [*silent, *cut_points(whole)]:
        source.write_bytes(whole[:cut])
        with pytest.raises(IngestError, match="incomplete trailing zstd frame"):
            prepare_trade_archive(source, tmp_path / "cache")
    # Refused before anything was extracted: not even the cache directory exists.
    assert not (tmp_path / "cache").exists()


def test_a_cache_made_from_a_truncated_source_is_not_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    whole = compressed_mbo()
    source = tmp_path / "day.mbo.dbn.zst"
    source.write_bytes(whole[: silently_short(whole)[0]])

    def unchecked(_frames: ZstdFrames) -> bool:
        return True

    with monkeypatch.context() as before:
        # Preparation as it ran before frames were checked.
        before.setattr(ZstdFrames, "complete", unchecked)
        stale = prepare_trade_archive(source, tmp_path / "cache")
    # A cache whose receipt and checksum agree with it, holding part of a day.
    assert 0 < trades_in(stale) < TRADES
    with pytest.raises(IngestError, match="incomplete trailing zstd frame"):
        prepare_trade_archive(source, tmp_path / "cache")
    source.write_bytes(whole)
    fresh = prepare_trade_archive(source, tmp_path / "cache")
    assert fresh != stale and trades_in(fresh) == TRADES
