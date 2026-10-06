# Trade records for offline research

`bedivere.data.lake.trades` reads retained records from local Databento
`trades` archives, including [prepared MBO caches](volume-bars.md#build).
The volume builder uses the same reader. No downloads, order-book reconstruction,
trade-correction handling, features, labels, or sizing are included.

## Reading

```python
import json
from pathlib import Path

from bedivere.core.session_days import parse_session_days
from bedivere.data.lake.layout import SeriesId
from bedivere.data.lake.trades import TradeTotals, read_trades

archive = Path("data/lake/_meta/trade_cache/<key>.trades.dbn.zst")
days = parse_session_days(json.loads(Path("sessions.json").read_text()))
sid = SeriesId("GLBX.MDP3", "NQ", "raw.NQZ5")  # your actual contract

totals = TradeTotals()
with read_trades(archive, sid, sessions=days) as read:
    for batch in read:  # tuples of Trade; provisional until EOF validation
        totals += TradeTotals.of(batch)
    assert read.validated
    source = read.source
```

For pandas or a summary, results return only after validation:

```python
from bedivere.data.lake.trades import summarize_trades
from bedivere.data.lake.trades_frame import load_trade_frame

frame = load_trade_frame(archive, sid, sessions=days, feed="databento")
summary = summarize_trades(archive, sid, sessions=days)
buy_volume = summary.totals.buy_volume
```

The pandas loader requires sessions or both `start_ns` and `end_ns`.
Streaming bounds memory by batch size, but every read scans the whole archive:
combine sessions in one read. `open_trade_archive(path).read(...)` supports
sequential reads on one handle, reusing the upfront hash verification.

`TradeTotals` counts eligible records and quantities by buy/sell/unknown side
using integers. Side totals add up; `ineligible_records` is counted separately.
Empty selections give zeros. These are record counts, not unique executions,
orders, or participants.

## The record

Each `Trade` is one decoded record, without repair or sorting.
[Trade and TradeSource definitions](../src/bedivere/data/lake/trades.py)
and [TRADE_COLUMNS dtypes](../src/bedivere/data/lake/trades_frame.py) are the
full API reference.

| Fields | Meaning |
|---|---|
| `ordinal` | Zero-based decoded-trade position, assigned before selection |
| `ts_recv`, `ts_event` | Capture and venue timestamps, respectively; integer Unix ns |
| `session`, `contract` | Caller-calendar label; raw symbol resolved from archive symbology at `ts_recv` |
| `instrument_id`, `publisher_id` | Stored vendor identifiers |
| `price_fixed`, `price` | Exact integer in units of 1e-9; float convenience conversion |
| `size`, `sequence` | Stored quantity and venue message sequence |
| `side_code`, `aggressor` | Raw vendor side; `buy`, `sell`, or `unknown` |
| `flags`, `issues`, `eligible` | All raw flag bits, interpreted quality notes, and bar eligibility |

Undefined price/event time becomes `None` (nullable integers in pandas), not
a sentinel or float. Select and order by `ts_recv`, never `ts_event`.
Ties preserve archive order; sequences are not globally unique and are never
used to sort or deduplicate.

Databento side codes map `A → sell`, `B → buy`, `N → unknown`; other codes
stay unknown. No side is inferred from price moves. MBO preparation drops
`order_id`/`channel_id` and writes a placeholder depth, so none is exposed
and sequence channel scope cannot be recovered.

## Quality and eligibility

Flagged selected records remain visible:

| `TradeIssue` | Eligible for bars? |
|---|---|
| `UNDEFINED_PRICE`, `ZERO_SIZE`, `BAD_TS_RECV`, `NON_TRADE_ACTION` | No |
| `UNDEFINED_SIDE`, `UNDEFINED_TS_EVENT` | Yes |

The volume builder **refuses**, rather than silently drops, ineligible selected
records. Unknown flag bits are retained without changing eligibility.
`BAD_TS_RECV` means the vendor flagged capture time as inaccurate; those
records are exempt from ordering checks.

A read fails on invalid `ts_recv` (0 or outside int64), records outside the
header range, error/non-trade records, or selected symbols that are not
outrights of the root. Integrity checks cover all records; ordering,
publisher, symbology, and quality checks apply to the selection.

## Selection

- `raw.<CONTRACT>` selects that outright.
- Other series, such as `local.v.0`, require `sessions=` and `lake_root=`.
  Each session uses its existing `1s` contract selection, without future-volume
  ranking. Missing, damaged, or ambiguous references fail.
  `TradeSource.selection_dependencies` retains their fingerprints.
- `sessions` selects and labels `[open, close)`; `start_ns`/`end_ns` further
  restrict `[start, end)` on `ts_recv`. The archive must declare coverage of
  the requested range without a record limit. Covered quiet ranges are valid.
- Selected records must be ordered and have one publisher per session
  (per read without a calendar). Multi-symbol files need not be globally sorted,
  so reaching a window's end does not stop the scan.

`ts_recv` is feed capture time, not when your application processed the record.
`ts_event` uses the venue's clock; the two clocks need not agree. Neither
models processing or order latency.

## Completion and verification

**Every yielded batch is provisional. An exception invalidates everything
yielded by that read; stopping early leaves it unvalidated.**

Verification uses two passes on one open handle:

1. Hash every byte, check compressed-frame completeness, and verify any receipt
   or vendor manifest before decoding.
2. Hash the bytes fed to the decoder. At EOF, require matching digests, complete
   trailing records/frames, receipt counts, and unchanged file identity
   (device, inode, size, mtime).

Status moves `pending → reading → validated | failed | incomplete`.
Only `validated` permits `read.source` and `read.provenance(...)`.
Inputs must stay unchanged: these checks are not a lock or filesystem snapshot.
Upfront hashing costs an extra full pass of I/O; narrow windows are not indexed.

### Receipts and manifests

A present receipt or `manifest.json` must agree; it is never ignored on failure.

- Receipts check structure/policy, contiguous input ranges, archive hash,
  managed cache-name policy hash, header identity/range, and total/per-input
  counts and volume.
- Vendor manifests check the filename's unique entry, size, and SHA256.
  `fetch --dest archives/` isolates each job under `archives/<job-id>/`, keeping
  its manifest with its files.
- Recorded source paths are not opened or exposed; public lineage keeps
  basenames and hashes. Receipt `manifest_verified` describes preparation,
  not fresh source verification. Missing metadata means unavailable lineage,
  not verified origin.

Incomplete zstd frames and partial DBN records fail. Without a receipt or
manifest, truncation exactly at a valid record/frame boundary can remain
undetectable. Caches prepared before compressed-source completeness checks
existed, from sources without manifests, must be rechecked by rerunning
`prepare-trades` on the originals. Valid hashes do not prove complete market
data or authentic origin; also review vendor condition reports.

## Identity and provenance

`(TradeSource.archive_sha256, ordinal)` identifies a record in those exact
bytes, independent of selection and batch size. Recompression or cache
regeneration can change that identity even at the same path. Keep archives if
you need persistent record references.

`TradeSource` separates checked artifact facts (hash, size, schema, range,
counts, selection dependencies), recorded lineage, caller attestations, and
unavailable facts. A prepared cache's stored schema is `trades`, even for MBO
inputs. Original source schema comes from the receipt; original request
symbology is unavailable for prepared caches. A manifest-verified direct vendor
file can supply both from its header. Neither is inferred without evidence.

Join `Provenance` is separate from artifact identity; see the
[shared contract](volume-bars.md#provenance-contract). For trade frames:

- `sessions=` plus `feed=` attaches `attrs["provenance"]`. Feed, its derived
  namespace, and calendar are attestations; dataset, selection identity,
  `as_traded`, `ns`, `unix`, and `ts_recv` are verified.
- `attrs["trade_source"]` records the artifact;
  `attrs["receive_window"]` records applied time bounds, or `(None, None)`
  for whole sessions.
- A calendar-free read has source metadata but no join provenance; omit `feed`.

To use trades as observations, rename `ts_recv` to `asof_ns` and pass their
loader provenance. For per-decision windows and forming bars, see
[decision-time context](volume-bars.md#decision-time-context): load every
observed session; forming replay also needs whole-session trades from the
same archive and contract selection as the bars.
