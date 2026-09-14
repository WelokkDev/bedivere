"""Buying history: cost estimate, batch job, download — the billed edge.

The only module in bedivere that can spend money, and written accordingly.

Every submission is priced first. `submit_job` takes a mandatory
`cost_limit_usd`, calls the free `metadata.get_cost` endpoint, and REFUSES
before the POST if the estimate exceeds it. There is deliberately no spelling
that submits an unpriced job: a fat-fingered `start` date is a five-figure
difference on a tick schema.

Downloads are verified and atomic — temp name, size and published hash checked,
then renamed — because a truncated archive keeps its truthful header and every
ingest downstream would succeed over the fraction that arrived. Keep the
archives: a partition is a whole-file replacement derived from one, so the
`.dbn.zst` is what makes the lake reproducible.

Against the documented v0 Historical REST API (`hist.databento.com/v0`), auth by
API key as the basic-auth username. Every call here — pricing, the submit,
polling, file listing and the verified download — has been run end to end
against a live account from this repository. Expect a small job to sit in the
vendor's batch queue for tens of minutes before it is `done`.

`start` is inclusive and `end` is EXCLUSIVE, the API's convention: a range of
`2026-09-01..2026-09-02` is one day, and `2026-09-01..2026-09-01` is nothing.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

from bedivere.core.clock import Clock
from bedivere.data.vendor.databento.condition import (
    DatabentoApiError,
    PartialSymbolsError,
    http_client,
    raise_for_status,
    resolve_api_key,
)

HIST_BASE: Final[str] = "https://hist.databento.com/v0"

_TIMEOUT_SECONDS: Final[int] = 60
_DOWNLOAD_CHUNK: Final[int] = 1 << 20

DONE: Final[str] = "done"
"""The job state in which files can be listed and downloaded."""

EXPIRED: Final[str] = "expired"
"""Past `ts_expiration`: the files are gone from the Download center."""

_TERMINAL_STATES: Final[frozenset[str]] = frozenset({DONE, EXPIRED})


class CostLimitError(DatabentoApiError):
    """The priced estimate exceeded the caller's stated limit. NOTHING was
    submitted and nothing was billed."""


def _get(path: str, params: dict[str, Any], api_key: str | None) -> Any:
    response = http_client().get(
        f"{HIST_BASE}/{path}",
        params=params,
        auth=(resolve_api_key(api_key), ""),
        timeout=_TIMEOUT_SECONDS,
    )
    raise_for_status(response, path)
    return response.json()


def _post(path: str, data: dict[str, Any], api_key: str | None) -> Any:
    response = http_client().post(
        f"{HIST_BASE}/{path}",
        data=data,
        auth=(resolve_api_key(api_key), ""),
        timeout=_TIMEOUT_SECONDS,
    )
    raise_for_status(response, path)
    return response.json()


def _symbol_list(symbols: str | Iterable[str]) -> str:
    return symbols if isinstance(symbols, str) else ",".join(symbols)


# ---------- pricing ----------


def get_cost(
    dataset: str,
    symbols: str | Iterable[str],
    schema: str,
    start: str,
    end: str,
    *,
    stype_in: str = "parent",
    api_key: str | None = None,
) -> float:
    """What this request would cost, in USD. This endpoint is FREE.

    On a subscription plan the estimate reflects the plan, so data it includes
    prices at $0.00. There is no billing-mode parameter: Databento's own client
    deprecated `mode` and no longer sends it.
    """
    payload = _get(
        "metadata.get_cost",
        {
            "dataset": dataset,
            "symbols": _symbol_list(symbols),
            "schema": schema,
            "start": start,
            "end": end,
            "stype_in": stype_in,
        },
        api_key,
    )
    value: object = payload
    if isinstance(value, dict):
        # The endpoint returns a bare number, but a wrapped {"cost": ...} must
        # not be read as 0.0.
        wrapper = cast("dict[str, object]", value)
        for key in ("cost", "cost_usd", "usd"):
            if key in wrapper:
                value = wrapper[key]
                break
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise DatabentoApiError(f"metadata.get_cost answered {value!r}, not a number")
    try:
        return float(value)
    except ValueError as e:
        raise DatabentoApiError(f"metadata.get_cost answered {value!r}, not a number") from e


# ---------- jobs ----------


@dataclass(frozen=True, slots=True)
class BatchJob:
    """One batch job, as the API reports it.

    Only the fields this module acts on are modelled; `raw` keeps the whole
    payload so a caller can read one bedivere has not needed yet.
    """

    id: str
    state: str
    dataset: str
    schema: str
    symbols: str
    start: str
    end: str
    cost_usd: float | None
    record_count: int | None
    ts_expiration: str | None
    progress: int | None
    """Percent processed; meaningful while `processing`, null when just submitted."""

    raw: dict[str, Any]

    @property
    def done(self) -> bool:
        return self.state == DONE

    @property
    def short(self) -> bool:
        """A short-form listing entry (`id`, `state`, `ts_received` only), the
        format the API has announced `batch.list_jobs` will move to."""
        return "dataset" not in self.raw

    def describe(self) -> str:
        cost = f"${self.cost_usd:,.2f}" if self.cost_usd is not None else "cost unknown"
        records = f"{self.record_count:,} records" if self.record_count is not None else "?"
        state = self.state
        if state == "processing" and self.progress is not None:
            state = f"{state} {self.progress}%"
        return (
            f"{self.id}  {state:<15} {self.dataset} {self.schema} "
            f"{self.symbols} {self.start}..{self.end}  {records}, {cost}"
        )

    @classmethod
    def from_payload(cls, payload: object) -> BatchJob:
        if not isinstance(payload, dict):
            raise DatabentoApiError(f"batch job payload must be an object, got {type(payload)}")
        job = cast("dict[str, object]", payload)
        job_id = job.get("id")
        state = job.get("state")
        if not isinstance(job_id, str) or not isinstance(state, str):
            raise DatabentoApiError(f"batch job payload has no id/state: {sorted(job)}")
        return cls(
            id=job_id,
            state=state,
            dataset=str(job.get("dataset", "")),
            schema=str(job.get("schema", "")),
            symbols=str(job.get("symbols", "")),
            start=str(job.get("start", "")),
            end=str(job.get("end", "")),
            cost_usd=_optional_float(job.get("cost_usd")),
            record_count=_optional_int(job.get("record_count")),
            ts_expiration=_optional_str(job.get("ts_expiration")),
            progress=_optional_int(job.get("progress")),
            raw=job,
        )


def _optional_float(value: object) -> float | None:
    return None if value is None else float(cast("float", value))


def _optional_int(value: object) -> int | None:
    return None if value is None else int(cast("int", value))


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def submit_job(
    dataset: str,
    symbols: str | Iterable[str],
    schema: str,
    start: str,
    end: str,
    *,
    cost_limit_usd: float,
    stype_in: str = "parent",
    split_duration: str = "none",
    compression: str = "zstd",
    encoding: str = "dbn",
    extra_params: dict[str, Any] | None = None,
    api_key: str | None = None,
) -> BatchJob:
    """Price the request, refuse it if it is over budget, then submit it.

    `cost_limit_usd` has no default on purpose: a default is either so low that
    everyone overrides it reflexively or high enough to be no limit at all.

    Defaults chosen for the lake: `dbn` + `zstd` is what `lake.ingest` decodes,
    `split_duration="none"` keeps one archive per job so one file is the whole
    provenance root, and `stype_in="parent"` buys every contract of a root so a
    front month can be re-ranked causally later (see `causal_front_months`) —
    continuous symbology is API-only and cannot be re-derived after the fact.
    """
    if cost_limit_usd <= 0:
        raise ValueError(f"cost_limit_usd must be positive, got {cost_limit_usd}")

    try:
        estimate = get_cost(
            dataset, symbols, schema, start, end, stype_in=stype_in, api_key=api_key
        )
    except PartialSymbolsError as e:
        raise PartialSymbolsError(f"{e}\nNOTHING was submitted.", status=e.status) from e
    if estimate > cost_limit_usd:
        raise CostLimitError(
            f"this request is priced at ${estimate:,.2f}, above the stated limit of "
            f"${cost_limit_usd:,.2f} — NOTHING was submitted.\n"
            f"  {dataset} {schema} {_symbol_list(symbols)} {start}..{end} (stype_in={stype_in})\n"
            "Narrow the range, coarsen the schema, or raise cost_limit_usd deliberately."
        )

    data: dict[str, Any] = {
        "dataset": dataset,
        "symbols": _symbol_list(symbols),
        "schema": schema,
        "start": start,
        "end": end,
        "stype_in": stype_in,
        "encoding": encoding,
        "compression": compression,
        "split_duration": split_duration,
        "delivery": "download",
    }
    data.update(extra_params or {})
    try:
        payload = _post("batch.submit_job", data, api_key)
    except PartialSymbolsError as e:
        # 206 is a SUCCESS code: the job exists, for the symbols that resolved.
        raise PartialSymbolsError(
            f"{e}\nA 206 from batch.submit_job means the job WAS created for the symbols "
            "that did resolve, and is billed as such — see `bedivere-data jobs` before "
            "submitting again.",
            status=e.status,
        ) from e
    return BatchJob.from_payload(payload)


def list_jobs(
    *,
    states: Sequence[str] | None = None,
    since: str | None = None,
    api_key: str | None = None,
) -> list[BatchJob]:
    """Every batch job on the account, oldest first (the API sorts by `ts_received`).

    Expired jobs are left out unless `states` names them — the API's default.

    The API has announced that this listing will shrink to `id`, `state` and
    `ts_received`. An entry that arrives in that short form is filled in from
    `batch.get_job_details`, so callers always get the full record.
    """
    params: dict[str, Any] = {}
    if states:
        params["states"] = ",".join(states)
    if since:
        params["since"] = since
    payload = _get("batch.list_jobs", params, api_key)
    if not isinstance(payload, list):
        raise DatabentoApiError(f"batch.list_jobs answered {type(payload)}, expected a list")
    jobs = [BatchJob.from_payload(entry) for entry in cast("list[object]", payload)]
    return [find_job(job.id, api_key=api_key) if job.short else job for job in jobs]


def find_job(job_id: str, *, api_key: str | None = None) -> BatchJob:
    """One job by id, via `batch.get_job_details` — which, unlike the listing,
    still answers for an expired job."""
    try:
        payload = _get("batch.get_job_details", {"job_id": job_id}, api_key)
    except DatabentoApiError as e:
        if e.status == 404:
            raise DatabentoApiError(
                f"no batch job {job_id!r} on this account (the API answered 404)", status=404
            ) from e
        raise
    return BatchJob.from_payload(payload)


def wait_for_job(
    job_id: str,
    *,
    clock: Clock,
    timeout_seconds: float,
    poll_seconds: float = 30.0,
    api_key: str | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> BatchJob:
    """Poll until the job reaches a terminal state, or the timeout expires.

    `clock` has no default, like every other time source in bedivere: nothing
    outside `core.clock` reads the wall, which is what makes the timeout path
    testable without waiting for it.

    A timeout does NOT cancel anything — the job is still being prepared, and
    `download_job` will find it later.
    """
    deadline = clock.now_unix() + timeout_seconds
    while True:
        job = find_job(job_id, api_key=api_key)
        if job.state in _TERMINAL_STATES:
            return job
        remaining = deadline - clock.now_unix()
        if remaining <= 0:
            raise DatabentoApiError(
                f"batch job {job_id} is still {job.state!r} after {timeout_seconds:.0f}s. "
                "It is still being prepared — re-run the download later; nothing was lost."
            )
        sleep(min(poll_seconds, remaining))


# ---------- files ----------


@dataclass(frozen=True, slots=True)
class BatchFile:
    """One downloadable file of a completed job."""

    filename: str
    size: int
    hash: str | None
    """As published, `"<algorithm>:<hexdigest>"` — e.g. `sha256:ab12...`."""

    url: str

    @classmethod
    def from_payload(cls, payload: object) -> BatchFile:
        if not isinstance(payload, dict):
            raise DatabentoApiError(f"batch file payload must be an object, got {type(payload)}")
        entry = cast("dict[str, object]", payload)
        filename = entry.get("filename")
        urls = entry.get("urls")
        url = cast("dict[str, object]", urls).get("https") if isinstance(urls, dict) else None
        if not isinstance(filename, str) or not isinstance(url, str):
            raise DatabentoApiError(f"batch file payload has no filename/https url: {sorted(entry)}")
        # Joined onto a local directory below, so a traversal must not survive.
        if "/" in filename or "\\" in filename or filename in ("", ".", ".."):
            raise DatabentoApiError(f"refusing a batch filename that is not a plain name: {filename!r}")
        return cls(
            filename=filename,
            size=_optional_int(entry.get("size")) or 0,
            hash=_optional_str(entry.get("hash")),
            url=url,
        )


def list_files(job_id: str, *, api_key: str | None = None) -> list[BatchFile]:
    """The files of a completed job, with the URLs to fetch them from."""
    payload = _get("batch.list_files", {"job_id": job_id}, api_key)
    if not isinstance(payload, list):
        raise DatabentoApiError(f"batch.list_files answered {type(payload)}, expected a list")
    return [BatchFile.from_payload(entry) for entry in cast("list[object]", payload)]


def download_job(
    job_id: str,
    dest_dir: Path,
    *,
    api_key: str | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Download every file of a completed job into `dest_dir`, verified.

    The job's state is read first, so a job still being prepared is refused by
    name rather than as an empty file list, and an expired one — whose files
    are gone — is not mistaken for one that is not finished.

    A file already present and verifying is SKIPPED rather than re-fetched, so an
    interrupted download is resumed by re-running the same command. `overwrite`
    re-fetches regardless, for the case where the vendor reissued the range.
    """
    job = find_job(job_id, api_key=api_key)
    if job.state == EXPIRED:
        raise DatabentoApiError(
            f"batch job {job_id} has expired from the Download center — its files are gone. "
            "Submit the request again to rebuild them."
        )
    if not job.done:
        progress = f" ({job.progress}%)" if job.progress is not None else ""
        raise DatabentoApiError(
            f"batch job {job_id} is {job.state!r}{progress}, not {DONE!r} — it is still being "
            "prepared. wait_for_job (or `fetch --wait`) will say when it is; nothing was lost."
        )
    files = list_files(job_id, api_key=api_key)
    if not files:
        raise DatabentoApiError(
            f"batch job {job_id} is {DONE!r} but lists no files — nothing to download"
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    return [_download_one(f, dest_dir, api_key, overwrite) for f in files]


def _download_one(
    file: BatchFile, dest_dir: Path, api_key: str | None, overwrite: bool
) -> Path:
    requests = http_client()

    target = dest_dir / file.filename
    if target.is_file() and not overwrite:
        try:
            _verify(target, file)
            return target
        except DatabentoApiError:
            pass  # present but wrong: fall through and re-fetch

    tmp = target.with_name(target.name + ".part")
    try:
        with requests.get(
            file.url,
            auth=(resolve_api_key(api_key), ""),
            timeout=_TIMEOUT_SECONDS,
            stream=True,
        ) as response:
            raise_for_status(response, f"download {file.filename}")
            with tmp.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK):
                    if chunk:
                        fh.write(chunk)
        _verify(tmp, file)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return target


def _verify(path: Path, file: BatchFile) -> None:
    """Size and (when published) content hash. Raises on any mismatch.

    Size alone would pass a cached body of the right length, but it catches the
    common failure for free, so it runs first.
    """
    if file.size and path.stat().st_size != file.size:
        raise DatabentoApiError(
            f"{file.filename}: downloaded {path.stat().st_size} bytes, API published {file.size}"
        )
    if not file.hash:
        return
    algorithm, _, expected = file.hash.partition(":")
    if not expected:
        algorithm, expected = "sha256", file.hash
    try:
        digest = hashlib.new(algorithm)
    except ValueError:
        return  # an algorithm this Python cannot compute: size check stands alone
    with path.open("rb") as fh:
        while chunk := fh.read(_DOWNLOAD_CHUNK):
            digest.update(chunk)
    got = digest.hexdigest()
    if got.lower() != expected.lower():
        raise DatabentoApiError(
            f"{file.filename}: {algorithm} mismatch — got {got}, API published {expected}"
        )
