"""Prepare a reusable trades archive from local daily trades/MBO downloads.

Only MBO action T contributes volume: fills (F) are the resting side of the
same executions. Sources are read only, and a compressed one that stops short
is refused. The receipt written beside each cache is parsed here too, so
writer and reader agree on its shape; no recorded source path is ever opened.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final, NoReturn, cast

import databento_dbn as dbn
from databento_dbn.metadata import SymbolMapping

from bedivere.data.lake.ingest import (
    IngestError,
    ZstdFrames,
    is_zstd,
    read_metadata,
    stream_dbn,
)

POLICY_VERSION: Final[int] = 1
POLICY_SELECTION: Final[str] = "action_T_without_snapshot"

# What `prepare_trade_archive` names a cache: its policy key, then the format.
MANAGED_NAME: Final[re.Pattern[str]] = re.compile(r"^([0-9a-f]{64})\.trades\.dbn\.zst$")

_SHA256: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _source_hash(path: Path) -> str:
    """SHA256 of one source file, whose last zstd frame must be complete: a cut
    download can otherwise decode cleanly as the records that arrived."""
    digest = hashlib.sha256()
    frames = ZstdFrames(path.name) if is_zstd(path) else None
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
            if frames is not None:
                frames.feed(chunk)
    if frames is not None:
        frames.require_complete()
    return digest.hexdigest()


def policy_key(policy: Mapping[str, object]) -> str:
    """The cache key: SHA256 of the policy as recorded, full source paths included."""
    return hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()


def receipt_path(archive: Path) -> Path:
    """Where `prepare_trade_archive` writes the receipt for `archive`."""
    return archive.with_name(archive.name + ".json")


def verify_manifest(path: Path, digest: str, size: int | None = None) -> bool:
    """Check `path` against a vendor `manifest.json` beside it.

    False means no manifest, which verifies nothing; one that is unreadable or
    disagrees raises.
    """
    manifest = path.parent / "manifest.json"
    if not manifest.exists():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        matches = [row for row in payload["files"] if row["filename"] == path.name]
    except (ValueError, KeyError, TypeError) as e:
        raise IngestError(f"{path.name}: unreadable vendor manifest ({e!r})") from e
    if len(matches) != 1:
        raise IngestError(f"{path.name}: missing or duplicate entry in vendor manifest")
    row = matches[0]
    expected_size = path.stat().st_size if size is None else size
    if row.get("size") != expected_size or row.get("hash") != f"sha256:{digest}":
        raise IngestError(f"{path.name}: size or SHA256 differs from vendor manifest")
    return True


@dataclass(frozen=True, slots=True)
class ReceiptInput:
    """One source file as the preparation step recorded it.

    Only the basename is kept. `manifest_verified` describes that step, not a
    fresh check of the source file.
    """

    name: str
    sha256: str
    bytes: int
    schema: str
    start_ns: int
    end_ns: int
    manifest_verified: bool
    records: int
    trades: int
    volume: int
    snapshot_trades_skipped: int


@dataclass(frozen=True, slots=True)
class TradeReceipt:
    """A parsed, self-consistent preparation receipt: recorded lineage only.

    It does not authenticate the inputs or prove the market data complete.
    """

    policy_key: str
    version: int
    selection: str
    dataset: str
    sha256: str
    trades: int
    volume: int
    inputs: tuple[ReceiptInput, ...]
    """In extraction order: ascending, contiguous `[start_ns, end_ns)` ranges."""

    @property
    def input_schemas(self) -> tuple[str, ...]:
        return tuple(sorted({i.schema for i in self.inputs}))

    def verify_artifact(self, name: str, sha256: str) -> None:
        if self.sha256 != sha256:
            raise IngestError(f"{name}: SHA256 differs from its preparation receipt")
        managed = MANAGED_NAME.fullmatch(name)
        if managed is not None and managed.group(1) != self.policy_key:
            raise IngestError(
                f"{name}: managed cache name does not match its receipt's extraction-policy "
                f"hash ({self.policy_key})"
            )

    def verify_header(self, dataset: str, start_ns: int, end_ns: int) -> None:
        if dataset != self.dataset:
            raise IngestError(
                f"receipt policy dataset {self.dataset!r} differs from archive dataset {dataset!r}"
            )
        if (start_ns, end_ns) != (self.inputs[0].start_ns, self.inputs[-1].end_ns):
            raise IngestError("archive range differs from the receipt's recorded input ranges")

    def verify_counts(self, trades: int, volume: int, per_input: list[tuple[int, int]]) -> None:
        """Full-archive totals: every decoded trade record, before any selection."""
        if (trades, volume) != (self.trades, self.volume):
            raise IngestError(
                f"archive holds {trades:,} trade records / {volume:,} units; its receipt "
                f"records {self.trades:,} / {self.volume:,}"
            )
        recorded = [(i.trades, i.volume) for i in self.inputs]
        if per_input != recorded:
            raise IngestError("per-input trade counts differ from the receipt")


def read_receipt(path: Path) -> TradeReceipt:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise IngestError(f"{path.name}: receipt is not valid JSON ({e})") from e
    return parse_receipt(payload, where=path.name)


def parse_receipt(payload: object, *, where: str) -> TradeReceipt:
    """Structure and internal consistency of a version-1 receipt; raises on any doubt."""

    def fail(problem: str) -> NoReturn:
        raise IngestError(f"{where}: {problem}")

    def fields(value: object, keys: set[str], what: str) -> dict[str, object]:
        if not isinstance(value, dict) or set(cast(dict[str, object], value)) != keys:
            fail(f"{what} must be an object with exactly {sorted(keys)}")
        return cast(dict[str, object], value)

    def count(value: object, what: str) -> int:
        if type(value) is not int or value < 0:
            fail(f"{what} must be a non-negative integer, not {value!r}")
        return value

    def digest(value: object, what: str) -> str:
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            fail(f"{what} must be a lowercase hex SHA256")
        return value

    receipt = fields(payload, {"policy", "sha256", "trades", "volume", "files"}, "receipt")
    policy = fields(receipt["policy"], {"version", "selection", "dataset", "inputs"}, "policy")
    version = policy["version"]
    if type(version) is not int or version != POLICY_VERSION:
        fail(f"unsupported extraction-policy version {version!r}")
    if policy["selection"] != POLICY_SELECTION:
        fail(f"unknown extraction selection {policy['selection']!r}")
    dataset = policy["dataset"]
    if not isinstance(dataset, str) or not dataset:
        fail("policy dataset must be a non-empty string")
    raw_inputs = policy["inputs"]
    if not isinstance(raw_inputs, list) or not raw_inputs:
        fail("policy inputs must be a non-empty list")
    recorded: list[dict[str, Any]] = []
    input_keys = {"path", "sha256", "bytes", "manifest_verified", "start_ns", "end_ns", "schema"}
    for i, raw in enumerate(cast(list[object], raw_inputs)):
        entry = fields(raw, input_keys, f"policy input {i}")
        source = entry["path"]
        # Basename only, whichever separator the preparing platform used.
        name = re.split(r"[\\/]", source)[-1] if isinstance(source, str) else ""
        if name in ("", ".", ".."):
            fail(f"policy input {i} has no file name")
        start, end = entry["start_ns"], entry["end_ns"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end < 2**63:
            fail(f"policy input {i} needs integer nanoseconds with start < end")
        if not isinstance(entry["manifest_verified"], bool):
            fail(f"policy input {i} manifest_verified must be true or false")
        if entry["schema"] not in ("mbo", "trades"):
            fail(f"policy input {i} schema must be mbo or trades")
        recorded.append(
            {
                "name": name,
                "sha256": digest(entry["sha256"], f"policy input {i} sha256"),
                "bytes": count(entry["bytes"], f"policy input {i} bytes"),
                "schema": entry["schema"],
                "start_ns": start,
                "end_ns": end,
                "manifest_verified": entry["manifest_verified"],
            }
        )
    # Extraction ran in range order; `inputs` keeps the directory's name order.
    recorded.sort(key=lambda entry: entry["start_ns"])
    for left, right in zip(recorded, recorded[1:], strict=False):
        if left["end_ns"] != right["start_ns"]:
            fail("recorded input ranges overlap or leave a gap")
    if len({entry["name"] for entry in recorded}) != len(recorded):
        fail("recorded input file names are not unique")
    raw_files = receipt["files"]
    if not isinstance(raw_files, list) or len(cast(list[object], raw_files)) != len(recorded):
        fail("files must list one count record per input")
    file_keys = {"file", "records", "trades", "volume", "snapshot_trades_skipped"}
    inputs: list[ReceiptInput] = []
    for entry, raw in zip(recorded, cast(list[object], raw_files), strict=True):
        counts = fields(raw, file_keys, f"file record {entry['name']!r}")
        if counts["file"] != entry["name"]:
            fail("per-file counts are not in the recorded inputs' extraction order")
        records = count(counts["records"], "records")
        trades = count(counts["trades"], "trades")
        skipped = count(counts["snapshot_trades_skipped"], "snapshot_trades_skipped")
        if trades + skipped > records:
            fail(f"{entry['name']}: more extracted and skipped trades than records scanned")
        inputs.append(
            ReceiptInput(
                **entry,
                records=records,
                trades=trades,
                volume=count(counts["volume"], "volume"),
                snapshot_trades_skipped=skipped,
            )
        )
    trades = count(receipt["trades"], "trades")
    volume = count(receipt["volume"], "volume")
    if not trades:
        fail("a prepared cache holds at least one trade record")
    if (sum(i.trades for i in inputs), sum(i.volume for i in inputs)) != (trades, volume):
        fail("per-file counts do not add up to the recorded totals")
    return TradeReceipt(
        policy_key=policy_key(policy),
        version=version,
        selection=POLICY_SELECTION,
        dataset=dataset,
        sha256=digest(receipt["sha256"], "sha256"),
        trades=trades,
        volume=volume,
        inputs=tuple(inputs),
    )


def prepare_trade_archive(
    source: Path,
    cache_root: Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Accept a single archive or a directory of contiguous daily files.

    The cache key covers input content and extraction policy, and a matching
    cache is reused only after its own hash verifies. A truncated compressed
    input is refused before any extraction or reuse.
    """
    paths = (
        sorted([*source.glob("*.dbn"), *source.glob("*.dbn.zst")]) if source.is_dir() else [source]
    )
    if not paths:
        raise IngestError(f"{source}: no DBN files")
    inputs: list[dict[str, Any]] = []
    headers: dict[Path, dbn.Metadata] = {}
    for path in paths:
        if progress:
            progress(f"verify {path.name}")
        meta = read_metadata(path)
        if str(meta.schema) not in ("mbo", "trades") or meta.limit:
            raise IngestError("trade preparation requires unlimited trades or MBO archives")
        if not meta.end or not 0 <= meta.start < meta.end < 2**63:
            raise IngestError(f"{path.name}: explicit finite archive range is required")
        digest = _source_hash(path)
        verified = verify_manifest(path, digest)
        inputs.append(
            {
                "path": str(path.resolve()),
                "sha256": digest,
                "bytes": path.stat().st_size,
                "manifest_verified": verified,
                "start_ns": meta.start,
                "end_ns": meta.end,
                "schema": str(meta.schema),
            }
        )
        headers[path] = meta
    paths.sort(key=lambda path: headers[path].start)
    for left, right in zip(paths, paths[1:], strict=False):
        if headers[left].end != headers[right].start:
            raise IngestError("daily archive ranges overlap or have a gap; supply contiguous files")
    dataset = headers[paths[0]].dataset
    if any(meta.dataset != dataset for meta in headers.values()):
        raise IngestError("cannot combine archives from different datasets")
    policy = {
        "version": POLICY_VERSION,
        "selection": POLICY_SELECTION,
        "dataset": dataset,
        "inputs": inputs,
    }
    key = policy_key(policy)
    cache_root.mkdir(parents=True, exist_ok=True)
    target = cache_root / f"{key}.trades.dbn.zst"
    receipt = receipt_path(target)
    if target.exists() and receipt.exists():
        previous = json.loads(receipt.read_text(encoding="utf-8"))
        if previous.get("policy") == policy and previous.get("sha256") == file_hash(target):
            if progress:
                progress(f"reuse verified trade cache {target.name}")
            return target

    mapping: dict[str, list[Any]] = defaultdict(list)
    for path in paths:
        for symbol, intervals in headers[path].mappings.items():
            mapping[symbol].extend(SimpleNamespace(**interval) for interval in intervals)
    meta = dbn.Metadata(
        dataset=dataset,
        start=headers[paths[0]].start,
        end=headers[paths[-1]].end,
        schema=cast("dbn.Schema", dbn.Schema.TRADES),
        stype_in=cast("dbn.SType", dbn.SType.RAW_SYMBOL),
        stype_out=cast("dbn.SType", dbn.SType.INSTRUMENT_ID),
        symbols=sorted(mapping),
        mappings=[
            cast(SymbolMapping, SimpleNamespace(raw_symbol=symbol, intervals=intervals))
            for symbol, intervals in mapping.items()
        ],
    )
    fd, name = tempfile.mkstemp(dir=cache_root, prefix="trades-", suffix=".tmp")
    os.close(fd)
    tmp = Path(name)
    count = volume = 0
    per_file: list[dict[str, object]] = []
    try:
        with tmp.open("wb") as output:
            encoder = dbn.Transcoder(
                output,
                cast("dbn.Encoding", dbn.Encoding.DBN),
                cast("dbn.Compression", dbn.Compression.ZSTD),
            )
            encoder.write(meta.encode())
            buffer = bytearray()
            for path in paths:
                if progress:
                    progress(f"extract trade events from {path.name}")
                records = trades = snapshots = quantity = 0
                for rec in stream_dbn(path):
                    records += 1
                    if isinstance(rec, dbn.ErrorMsg):
                        raise IngestError(f"{path.name}: {rec}")
                    if not isinstance(rec, (dbn.TradeMsg, dbn.MBOMsg)):
                        continue
                    if str(rec.action) != "T":
                        continue
                    if rec.flags & dbn.F_SNAPSHOT:
                        snapshots += 1
                        continue
                    if not headers[path].start <= rec.ts_recv < headers[path].end:
                        raise IngestError(
                            f"{path.name}: trade lies outside its declared file range"
                        )
                    if isinstance(rec, dbn.MBOMsg):
                        trade = dbn.TradeMsg(
                            publisher_id=rec.publisher_id,
                            instrument_id=rec.instrument_id,
                            ts_event=rec.ts_event,
                            ts_recv=rec.ts_recv,
                            price=rec.price,
                            size=rec.size,
                            action=cast("dbn.Action", rec.action),
                            side=cast("dbn.Side", rec.side),
                            depth=0,
                            flags=rec.flags,
                            ts_in_delta=rec.ts_in_delta,
                            sequence=rec.sequence,
                        )
                    else:
                        trade = rec
                    buffer.extend(bytes(trade))
                    trades += 1
                    quantity += rec.size
                    if len(buffer) >= 1 << 20:
                        encoder.write(bytes(buffer))
                        buffer.clear()
                if progress:
                    progress(f"{path.name}: {trades:,} trade records, {quantity:,} units")
                per_file.append(
                    {
                        "file": path.name,
                        "records": records,
                        "trades": trades,
                        "volume": quantity,
                        "snapshot_trades_skipped": snapshots,
                    }
                )
                count += trades
                volume += quantity
            if buffer:
                encoder.write(bytes(buffer))
            encoder.finish()
        if not count:
            raise IngestError("archives contain no non-snapshot trade events")
        # Validate the compact output before publishing it, not just its header.
        decoded_count = decoded_volume = 0
        for rec in stream_dbn(tmp, zstd=True):
            if isinstance(rec, dbn.TradeMsg):
                decoded_count += 1
                decoded_volume += rec.size
        if (decoded_count, decoded_volume) != (count, volume):
            raise IngestError("extracted trade cache failed read-back verification")
        digest = file_hash(tmp)
        os.replace(tmp, target)
        payload = {
            "policy": policy,
            "sha256": digest,
            "trades": count,
            "volume": volume,
            "files": per_file,
        }
        receipt_tmp = receipt.with_suffix(".tmp")
        try:
            receipt_tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            os.replace(receipt_tmp, receipt)
        finally:
            receipt_tmp.unlink(missing_ok=True)
        return target
    finally:
        tmp.unlink(missing_ok=True)
