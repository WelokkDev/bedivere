"""The Databento edge: data-quality metadata, batch jobs, DBN decoding.

    condition.py  the per-UTC-date quality feed, and its Parquet store
    batch.py      cost estimates, batch job submission, archive download

DBN decoding itself lives in `bedivere.data.lake.ingest`, because normalising a
vendor's bar conventions into the lake's is an ingest concern, not an API one.

Every HTTP call goes through `condition.http_client()`, which imports `requests`
lazily and turns its absence into a sentence naming the extra that carries it,
and every response through `condition.raise_for_status()`, which quotes the
API's own error message and refuses a partially resolved symbol list.
So the conditions vocabulary — the enum, the parser, the severity fold
`bedivere.data.quality` runs — stays usable on a machine with no network stack
and no API key, and a base install that reaches for the API is told why.
"""
