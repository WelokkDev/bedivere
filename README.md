# bedivere

A public version of my trading engine, built so that any strategy I backtest is
the same strategy I deploy.

There is exactly one loop, and the three things that genuinely differ between a
backtest and a live session (where bars come from, what time it is, and who
fills the orders) are the only three things that get swapped.

```
for event in stream:
    clock.advance(event.ts)      # ReplayClock in a backtest; LiveClock IS time
    venue = broker.drain(event)  # fills settle BEFORE the strategy sees the bar
    portfolio.apply(venue)
    view.update(event.candle)    # multi-timeframe buckets; closes → observers
    strategy.on_order_event(...)
    strategy.on_bar(ctx)         # may submit brackets
```

That loop is `bedivere.engine.loop`, and it is the whole engine. A backtest runs
it with `{ReplayStream, ReplayClock, SimBroker}`; a live session runs it with
`{LiveBarStream, LiveClock, YourBroker}`. Everything downstream is shared code.

The engine is public; the strategies I run on it are not. My private working
copy adds strategy modules, and some other stuff, which I hope to port
eventually. The one strategy included
([`examples/sma_cross.py`](examples/sma_cross.py)) is just a demo of the loop.
It runs on fake data that trends up. If you want to run it on historical or live
data, you'd need to wrap it in a strategy plugin (see
[`tests/strategy_fixture.py`](tests/strategy_fixture.py)), write a spec for it,
and load bars into the lake (see
[Where the bars come from](#where-the-bars-come-from)). Live data also needs a
[feed adapter](#how-i-run-it-live-ninjatrader-mcp), which bedivere doesn't ship.

**Status:** this grows stale quickly, since I make changes fairly actively on my
own main and they land here in batches. I'll try to keep it maintained and ship
updates every now and then. Expect sharp edges, and defaults tuned to CME index
futures (NQ in particular).

## Quickstart

No data files and no network needed:

```bash
uv sync
python examples/sma_cross.py       # backtest on synthetic bars; run it twice,
                                   # the resultHash is identical
python examples/shadow_session.py  # a live-shaped session: pushed bars, shadow
                                   # venue, alerts on fills
```

Real work goes through a spec file and the CLIs:

```bash
bedivere-backtest --spec my-spec.json --out runs/     # run + archive
bedivere-backtest --spec my-spec.json --set rr=2.5    # a sweep is a shell loop
bedivere-live     --spec my-spec.json --mode shadow   # same file, live bars
bedivere-runs     list --out runs                     # query the archive
bedivere-data     coverage --spec my-spec.json        # would this run have bars?
```

### Where the bars come from

Research ranges live in the **bar lake**: one Parquet file per (series,
timeframe, session-day), laid out Hive-style and read in place by DuckDB. That
is the store `bedivere-data` fills and the store a real backtest reads:
`ingest`, `derive`, `list` and `sql` read and write it, `cost` and `fetch` buy
the vendor archives that fill it, and `conditions` records what the vendor
thinks of the days it sold you. Of the nine subcommands only `coverage` runs
without the extra. The lake is what carries the things a flat file cannot: per-bar contract identity (so a roll
is detectable), footer provenance, coverage that is exact by construction, and a
back-adjusted continuous view derived at read time and pinned to an as-of date.

A spec names it:

```json
"data": {"source": "lake", "dataset": "GLBX.MDP3", "series": "local.v.0"}
```

It is packaged as a separate extra (`uv sync --extra lake`) for one reason: the
CSV path exists, so `uv sync` alone is enough to clone this repo, run both
examples and get a green test suite with no native wheel on the machine. That is
a courtesy to a reviewer, not a statement that the lake is peripheral. A CSV is
one (symbol, timeframe) read whole into memory, with no provenance and no way to
say which session-days it should have held. Anything past a demo wants the lake.

`cost`, `fetch`, `jobs` and `conditions --fetch` call Databento, and read the
key from `DATABENTO_API_KEY`. Every bedivere CLI also loads a `.env` in the
working directory (gitignored), without overriding anything already set:

```bash
echo 'DATABENTO_API_KEY=db-...' >> .env
```

## Architecture

One JSON envelope fully describes a run: strategy, symbol, config, timeframes,
session calendar, instrument specs, sim costs, window, data source, and
optionally a feed and broker factory. `window` is optional precisely because a
live run cannot inherit one: it derives its own from *now* and the session-day
close, and everything else in the file means what it meant in the backtest.

```
  spec (JSON)  ──>  composition root  ──>  the one loop
                    run_backtest / run_live / run_shadow
```

| Layer | What it owns |
|---|---|
| `core/` | Timeframes, candles, session-days, bucket arithmetic, clocks, tick/price math |
| `data/` | The `CandleSource` port, CSV, the Parquet **lake**, and the gates a run passes before reading |
| `view/` | `MarketView`, incremental multi-timeframe aggregation with an observer seam per derived TF |
| `engine/` | The loop, intents/brackets, portfolio, warm-up gate, journal, metrics |
| `brokers/` | The `Broker` port + `SimBroker` (the modelled venue) |
| `streams/` | `ReplayStream`, `LiveBarStream`, and `SparseReplayStream` (coarse everywhere, fine where it matters) |
| `run/` | Composition roots, the run archive, live supervision |
| `config/` | The spec envelope and reference resolution |
| `cli/` | `bedivere-backtest`, `bedivere-live`, `bedivere-data`, `bedivere-runs` |
| `notify/` | Journal → notifier routing (console, Discord) |

A few choices worth knowing about, because they are the ones that keep a
backtest honest:

- **Costs are never defaulted.** `latencyMs`, `halfSpreadTicks`, commission,
  seed and the naked-window treatment are required fields. A frictionless
  backtest has to be a zero you can see in your own file.
- **Brackets are the only entry API.** The two-step reality (entry order →
  fill → protection, with a naked window in between) is implemented identically
  inside every broker, so that risk is modelled in the backtest rather than
  discovered live.
- **Every broker method is a command.** Outcomes arrive as `OrderEvent`s via
  `drain`, never as return values. A broker that answered synchronously would
  model a latency of zero.
- **Warm-up fails closed.** Components declare their lookbacks; if the first
  tradeable instant arrives with anything unready, the run dies. Same rule in
  backtest and live.
- **No silent data fallback.** A source that cannot serve a timeframe raises
  rather than handing back something adjacent. Two series that select contracts
  by different rules are two different price series, and mixing them is not
  detectable downstream.
- **Runs are reproducible records.** One directory per run plus an append-only
  index; the archived spec has its config resolved and its window materialized,
  so it re-runs with no remembered flags. `result.json` carries no wall-clock
  field, so you can run the same spec twice and diff the two files.

## How I run it live: ninjatrader-mcp

bedivere ships no market-data connection and no venue adapter. Both are small
ports, deliberately left out:

```python
adapter.start(stream)          # FeedAdapter: push closed bars, don't block
adapter.stop()

broker.submit_bracket(intent)  # Broker: commands in, OrderEvents out via drain()
broker.drain(event)
```

Each is named in the spec as a factory (`"feed"`, `"broker"`), so connecting a
different feed or a different broker is a new module and a spec edit, not a
change to the engine.

What I currently use for both is
[**ninjatrader-mcp**](https://github.com/WelokkDev/ninjatrader-mcp), my local MCP
server for NinjaTrader 8. It reaches NT8 through a loopback-only WebSocket
bridge (a small NT8 AddOn), and it exposes the two things this engine needs:

- **Live bars.** A `/feed` WebSocket channel that local bots can consume with
  the same bearer token. My private `FeedAdapter` subscribes to the
  `(symbol, timeframe)` streams it needs and pushes closed bars into the
  `LiveBarStream`, reconnecting with backoff and self-healing missed bars from
  the cache.
- **The order path.** A single audited `ExecutionService` submit path (place,
  OCO exit pairs, change, cancel, flatten), default-off behind independent
  fail-closed gates. My private `Broker` implementation places the entry there
  and arms the OCO protection the moment the entry fill broadcast arrives.

An order submitted from Python takes roughly **150-250ms** to actually be placed
(Python → MCP server → loopback WebSocket → NT8 AddOn → broker). This is fine
for the strategy I run, but for obvious reasons would rule out any strategy
trying to compete on microstructure.

## Requirements

Python 3.12+. Two runtime dependencies (`pydantic`, `tzdata`); the `lake` extra
adds DuckDB, the Rust DBN decoder, numpy/pandas and requests.

```bash
uv sync --extra lake        # everything
uv run pytest               # 490 tests (408 of them without the extra)
uv run ruff check .
uv run pyright
```

## License

MIT, see [LICENSE](LICENSE).
