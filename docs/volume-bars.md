# Volume bars for ML research

Build session-partitioned volume bars from local Databento trades, or
approximate them from stored `1s` OHLCV. Requires the `lake` extra; these
commands never fetch or purchase data. Engine replay and live feeds still use
fixed-duration candles. Features, labels, models, and sizing remain private.

## Build

```bash
# Optional: extract trades from a vendor job's contiguous daily MBO/trades files.
bedivere-data prepare-trades --root data/lake --archive archives/GLBX-20260101-XXXXXXXXXX

# Approximate from an existing one-second lake.
bedivere-data volume --root data/lake --spec my-spec.json \
  --from 1s --threshold 5000

# Build from trades, plus an approximation from those same trades.
bedivere-data volume --root data/lake --spec my-spec.json \
  --series raw.NQZ5 --archive archives/trades.dbn.zst \
  --threshold 5000 --compare-1s
```

Use your actual contract and archive path. `prepare-trades` prints a cache
path under `data/lake/_meta/trade_cache`; it extracts non-snapshot MBO
`T` actions, not resting-side `F` fills, and leaves sources untouched.
It verifies and reuses existing caches. See [trade verification](trade-data.md#completion-and-verification).

No strategy spec is required: replace `--spec` with
`--session sessions.json --symbol NQ --series raw.NQZ5`.
The default dataset is `GLBX.MDP3`. For a continuous series such as
`local.v.0`, trade builds reuse each session's existing `1s` contract
selection; they never rerank using future volume. Missing, ambiguous, or
intraday-changing identities fail.

Request complete exchange sessions. UTC-midnight download bounds often cut
the first and last overnight sessions short.

- Trade archives must declare full requested coverage without a record limit;
  every built session must contain selected trades.
- `--from 1s` requires source-range metadata covering each session from open
  to close, including under `--resume` and `--dry-run`. Sparse rows alone
  cannot distinguish a quiet open from missing hours. Re-ingest older
  partitions without this metadata.
- Preparation refuses gaps, overlaps, mixed datasets, and record-limited inputs.

The two approximations stay separate:

| Input | Definition `source` | Purpose |
|---|---|---|
| Trade archive | `trades` | Trade-built reference |
| Stored seconds (`--from 1s`) | `ohlcv-1s` | Approximation using the existing lake |
| Rebuilt seconds (`--compare-1s`) | `trades-1s` | Isolate aggregation loss using identical trades |

`--compare-1s` does not establish that vendor OHLCV uses identical trade
eligibility rules. Reports include counts, volume, remainders, durations, and
threshold deviations. `--dry-run` computes them without publishing datasets.

## Construction policy

OHLC describes prices within each volume-selected interval; volume determines
when to close it. Seconds preserve OHLCV for their grouped intervals, but cannot
recover within-second ordering, trade counts, or exact VWAP.

| Boundary | Behaviour |
|---|---|
| `whole_trade` (default) | Close after the trade/second reaching or exceeding Q; retain overshoot in that bar and restart the counter. |
| `split_trade` | Split a boundary trade's quantity at its observed price/time; full bars equal Q. Trades only. |
| `nearest_second` | Target cumulative Q, 2Q, …; choose the nearer preceding/crossing positive-volume second's end. Earlier wins ties; repeated boundaries coalesce. Seconds only. |

For nearest-second stored bars, add `--boundary nearest_second`.
For the comparison approximation, use `--approx-boundary nearest_second`;
`--boundary` still selects the trade reference. Whole seconds are never split,
and nearest-second bars may be smaller or larger than Q.

All policies:

- Use `ts_recv` in Unix nanoseconds and preserve archive order for ties.
  Out-of-order selected trades fail; trades belong to `[session open, close)`.
  Source OHLCV intervals use close stamps.
- Reset at exchange-session boundaries, not decoder chunks. Retain the
  remainder as `is_partial=true`; under nearest-second it need not be below Q.
- Set `end_ns` to the last contributing trade or interval end.
  **Join on `available_ns`**, not `end_ns`: full default bars are available at
  their end, nearest-second bars at the crossing second's end, and remainders
  at session close. Sparse trading can delay nearest-second availability by
  more than a second; several bars can become available together.
- Refuse ineligible selected trades, multiple publishers per session, and
  synthetic seconds. Zero-volume seconds contribute nothing.
  [Trade eligibility](trade-data.md#quality-and-eligibility) is shared with the
  reader; no aggressor-side inference or trade-break reconciliation occurs.

The threshold is one fixed integer, not an adaptive estimate. Choose it using
training data or completed prior sessions. Construction diagnostics are not
evidence of ML performance.

### Measured nearest-second comparison

On four inspected NQU6 ETH sessions (August 25–28, 2026; 2,063,880 contracts),
seconds rebuilt from the same trades gave these results at Q=5,000:

| Metric | Default seconds | Nearest second |
|---|---:|---:|
| p95 absolute volume deviation (contracts) | 95 | 74 |
| Median matched endpoint gap (seconds) | 51.55 | 1.28 |
| Median volume-interval intersection / union | 63.5% | 98.8% |

Matches maximize cumulative-volume overlap within a session, exclude reference
remainders, and may be many-to-one; these are offline comparisons, not causal
joins. Percentiles use nearest rank. Both policies conserve volume.

Nearest-second is not always better: Q=20,000 worsened p95 volume deviation,
and causal median close agreement worsened against split-trade bars despite
better interval matching. This was not a holdout, vendor-OHLCV, rollover, or ML
test. Keep it opt-in; prefer trade-built bars when trades are available.
Local reproduction files are untracked:
`data/volume_research/compare_nearest.py` and `nearest_comparison.{json,md}`
in that directory.

## Storage and reading

```text
event_bars/dataset=GLBX.MDP3/symbol=NQ/series=raw.NQZ5/
  definition=<policy-and-calendar SHA256>/session=2026-06-15/bars.parquet
```

The definition includes source, threshold, boundary policy, clock, calendar
template/timezone, reset/remainder rules, and builder version. Footers record
session geometry, source fingerprint, content checksum, identity, and
`as_traded` price basis. Conflicting session geometry is refused.

Archive builds stage output until the full trade read validates; a read failure
publishes nothing. Publication is atomic per partition, not across the whole
build. `--from 1s` publishes day by day, so later failures can leave earlier
days built.

`--resume` validates outputs and current source fingerprints, including
continuous-series selection partitions. Missing, changed-source, or legacy
checksum-free outputs rebuild; damaged outputs fail. Reports distinguish
`source_check="fingerprint"` (no trade decoding) from `"full_read"` (validated
through EOF). A missing comparison side recomputes both sides for that session.

```python
import json
from pathlib import Path

from bedivere.core.session_days import parse_session_days
from bedivere.data.lake.layout import SeriesId
from bedivere.data.lake.volume import VolumeSpec
from bedivere.data.lake.volume_research import load_volume_frame, volume_coverage

root = Path("data/lake")
days = parse_session_days(json.loads(Path("sessions.json").read_text()))
sid = SeriesId("GLBX.MDP3", "NQ", "raw.NQZ5")
spec = VolumeSpec(threshold=5000, source="trades", boundary="whole_trade")

coverage = volume_coverage(root, sid, spec, days)  # valid / missing / invalid
bars = load_volume_frame(root, sid, spec, days, feed="databento")
```

The loader checks schema, identity, session bounds, bar invariants, and content
checksums. Use the exact definition and sessions.

- Remainders require `include_partial=True`.
- Missing sessions fail unless `allow_missing=True`; corrupt ones always fail.
- `start_ns`/`end_ns` filter the half-open **availability** window.
  `attrs["availability_window"]` retains it; joins reject uncovered requests.
- Attributes also retain `bar_definition`, `missing_sessions`, `price_basis`,
  `provenance`, and `bar_source`. The latter records source kind and per-session
  input fingerprints at build time, not fresh verification.
- `volume_coverage` checks stored integrity; only `--resume` checks current
  input freshness. SQL is lower-level and does not run these full checks.

Row identity is `(dataset, symbol, series, definition, session, bar_id)`, not
timestamp. `input_count` counts contributing trades or seconds; split trades
can count toward several bars. Trade-built bars have VWAP; approximations have
NULL VWAP. Read integer timestamps column-first (e.g.
`bars["available_ns"].iloc[i]`); a mixed-type pandas row may coerce them to floats.

```bash
bedivere-data sql --root data/lake \
  "SELECT definition, session, count(*) bars, sum(volume) volume
   FROM volume_bars GROUP BY definition, session ORDER BY session"
```

## Availability-aware research alignment

Observations need `asof_ns` (decision time, int64 Unix nanoseconds),
`session` (string label), and positive `instrument_id`. Declare what actually
produced them; do not copy provenance from the bars to make a join pass.

```python
from bedivere.data.lake.volume_research import Provenance, align_available

observed = Provenance(
    feed="databento", dataset="GLBX.MDP3", symbol="NQ", series="raw.NQZ5",
    instrument_namespace="databento:GLBX.MDP3", price_basis="as_traded",
    sessions=days, time_unit="ns", epoch="unix", clock="ts_recv",
)
aligned = align_available(
    observations, bars, observation_provenance=observed, max_age_ns=60_000_000_000
)
```

The result preserves observation rows, index, private values, and dtypes,
attaching raw bar columns under `volume_`. The latest sufficiently recent
`available_ns` wins within the same session/instrument; no match yields NULLs.
Timestamp ties choose the largest `bar_id`.

Exact matches default off. Set `allow_exact_matches=True` only if your timing
convention supports inclusive matching. The illustrative age above is not a
recommended feature lookback. For windowed bar frames, the requested lookback
—from the later of session open and `asof_ns - max_age_ns` to the decision—
must be covered. At `end_ns`, strict matching is safe; inclusive matching is not.

### Provenance contract

All identity fields in the example are required. Feed, dataset, root/series,
instrument-ID namespace, price basis, calendar template/timezone, and clock
compare exactly and case-sensitively; shared session labels must have identical
bounds. Units must be `ns` and epoch `unix`. Threshold and boundary policy are
not compared.

Frames must agree with their declarations, including identity columns and
attributes. Observations must lie in their declared sessions; bars must match
their definition and session bounds. Unknown, mixed, contradictory, or
incompatible provenance raises `ProvenanceError`. No automatic NT8/Databento
mapping, price adjustment, timezone conversion, or clock reconciliation occurs.

For loaded bars, `feed` and its derived namespace are caller attestations;
other fields are verified against stored partitions. All observation fields
are attestations. `Provenance.verified` records the distinction, not authenticity.
Keep loader attributes intact and reload instead of editing frames: these
checks cannot prove arbitrary in-memory edits safe.

Convert timestamps explicitly before declaring them:

```python
import pandas as pd

utc = pd.to_datetime(observations["decided_at"], utc=True)  # already-UTC input
observations["asof_ns"] = utc.dt.as_unit("ns").astype("int64")
```

For naive exchange-local timestamps, localize to their real timezone first;
`utc=True` would assume UTC. Pin nanosecond resolution before taking integers,
and never pass through floats.

## Decision-time context

These APIs share the alignment rules above: causal availability, compatible
provenance, one session/instrument, and opt-in exact matches. The examples
continue from the loading/alignment setup; `archive` must be the original
trade archive used to build `bars`.

```python
from bedivere.data.lake.decision_context import bar_history, forming_bars, trade_windows
from bedivere.data.lake.trades_frame import load_trade_frame

archive = Path("archives/trades.dbn.zst")
trades = load_trade_frame(archive, sid, sessions=days, feed="databento")
history = bar_history(
    observations, bars, observation_provenance=observed,
    depth=20, max_age_ns=3_600_000_000_000,
)
windows = trade_windows(
    observations, trades, observation_provenance=observed, lookback_ns=60_000_000_000,
)
forming = forming_bars(observations, bars, trades, observation_provenance=observed)
```

| Function | Result and requirements |
|---|---|
| `bar_history` | Long frame keyed by positional `observation` and `lag` (0 = latest); bar columns use `volume_`. Requires explicit depth/age, covered lookback, and contiguous bar IDs. No qualifying bars means no rows. |
| `trade_windows` | One row per observation/trade in archive order, with `trade_` columns. Supply exactly one of `lookback_ns` or an int64 `start_column`; NULL starts select nothing, starts after the decision fail. Windows are `[start, asof_ns)` unless exact matches are enabled. |
| `forming_bars` | Original observations/index plus `forming_` ID, timestamps, OHLC, volume, input count, and VWAP. Replays the trade-built policy; NULL when nothing is forming. |

Long-frame `observation` is the input row's zero-based position, not its index
label. Join private columns back by position. Window output can be large:
it holds one row per observation/trade pair.

Trade windows must fit `attrs["receive_window"]` and the loaded sessions.
An unloaded session is an error; a loaded but quiet session is a valid empty
answer.

Forming replay additionally requires:

- `source="trades"`; neither second-based approximation is supported.
- Whole-session trades loaded without `start_ns`/`end_ns`, plus bar coverage
  from each observed session's open through its decision.
- The same archive SHA256 and continuous-series selection fingerprints as the
  stored bars. Renaming is fine; different bytes require rebuilding.
- Every emitted full bar must equal a stored bar, and every stored full bar due
  by the decision must have been emitted. Later bars are not checked.

Fields are NULL while the accumulator is empty. At session close with exact
matches, the remainder counts as published and nothing is forming.
Full API definitions: [research joins](../src/bedivere/data/lake/volume_research.py)
and [decision context](../src/bedivere/data/lake/decision_context.py).

## Older data and limits

- Re-ingest old `1s` partitions lacking source-range metadata; rebuild volume
  bars made from partial sessions. Existing incorrect outputs are not repaired
  automatically.
- Old `--compare-1s` outputs labelled `ohlcv-1s` are invalid. Rebuild them as
  `trades-1s`, then remove the obsolete partitions; raw SQL can still show them.
- Reload older in-memory frames missing required loader metadata.
- For older prepared caches and truncation checks, see
  [receipts and manifests](trade-data.md#receipts-and-manifests).

Neither provenance nor these diagnostics prove feed completeness or a
leakage-free experiment. Review vendor condition reports. `ts_recv` is a
capture-time proxy, not application/order latency; `ts_event` is a different
clock. Session bounds catch some unit errors, not every timezone mistake.
Instrument IDs are matched only within a session and namespace, never mapped
across feeds. Training splits, fitted transforms, labels, and execution remain
outside this layer.
