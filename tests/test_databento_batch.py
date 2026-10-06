"""bedivere.data.vendor.databento.batch — the only module here that spends money.

Every test drives a stand-in for `requests`, so the suite never touches the
network and never needs a key. The discipline under test: the request is PRICED
before it is submitted, a price over the stated limit submits nothing at all, and
a download that does not match what the API published never becomes visible.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from bedivere.core.clock import ReplayClock
from bedivere.data.vendor.databento import batch
from bedivere.data.vendor.databento.batch import (
    BatchFile,
    BatchJob,
    CostLimitError,
    DatabentoApiError,
    PartialSymbolsError,
)

KEY = "db-test-key"

JOB: dict[str, Any] = {
    "id": "GLBX-20260610-ABCDEF",
    "state": "done",
    "dataset": "GLBX.MDP3",
    "schema": "ohlcv-1s",
    "symbols": "NQ.FUT",
    "start": "2026-01-01",
    "end": "2026-07-01",
    "cost_usd": 42.5,
    "record_count": 123_456,
    "ts_expiration": "2026-07-08",
    "split_duration": "none",
}


class FakeResponse:
    def __init__(
        self,
        payload: Any = None,
        *,
        status: int = 200,
        body: bytes = b"",
        text: str | None = None,
    ) -> None:
        self._payload = payload
        self.status_code = status
        self.text = text if text is not None else ("refused: bad dataset" if status >= 400 else "")
        self._body = body

    def json(self) -> Any:
        return self._payload

    def iter_content(self, chunk_size: int = 0) -> Any:
        yield self._body

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


class FakeRequests:
    """Records every call, so auth and parameters are assertable."""

    def __init__(self, **routes: Any) -> None:
        self.routes: dict[str, Any] = routes
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def _answer(self, method: str, url: str, kwargs: dict[str, Any]) -> FakeResponse:
        self.calls.append((method, url, kwargs))
        for name, response in self.routes.items():
            if name in url:
                built = response() if callable(response) else response
                assert isinstance(built, FakeResponse)
                return built
        raise AssertionError(f"unrouted {method} {url}")

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        return self._answer("GET", url, kwargs)

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        return self._answer("POST", url, kwargs)


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Install a stand-in `requests` and a key, and hand back an installer."""
    monkeypatch.setenv("DATABENTO_API_KEY", KEY)

    def install(**routes: Any) -> FakeRequests:
        fake = FakeRequests(**routes)
        module = types.ModuleType("requests")
        module.get = fake.get  # pyright: ignore[reportAttributeAccessIssue]
        module.post = fake.post  # pyright: ignore[reportAttributeAccessIssue]
        monkeypatch.setitem(sys.modules, "requests", module)
        return fake

    return install


DETAILS: dict[str, Any] = {"batch.get_job_details": FakeResponse(JOB)}
"""The single-job endpoint answering "done" — where every download starts."""


# ---------- pricing ----------


def test_get_cost_prices_a_request_and_authenticates_with_the_key(api: Any) -> None:
    fake = api(**{"metadata.get_cost": FakeResponse(42.5)})
    assert batch.get_cost("GLBX.MDP3", "NQ.FUT", "ohlcv-1s", "2026-01-01", "2026-07-01") == 42.5

    _, url, kwargs = fake.calls[0]
    assert url.endswith("/metadata.get_cost")
    assert kwargs["auth"] == (KEY, "")  # the key is the basic-auth USERNAME
    assert kwargs["params"]["stype_in"] == "parent"
    assert "mode" not in kwargs["params"]  # deprecated upstream; Databento's client no longer sends it


def test_a_symbol_iterable_becomes_one_comma_separated_parameter(api: Any) -> None:
    fake = api(**{"metadata.get_cost": FakeResponse(1.0)})
    batch.get_cost("GLBX.MDP3", ["NQ.FUT", "ES.FUT"], "ohlcv-1s", "a", "b")
    assert fake.calls[0][2]["params"]["symbols"] == "NQ.FUT,ES.FUT"


def test_a_wrapped_cost_is_not_read_as_zero(api: Any) -> None:
    # Reading a wrapped {"cost": ...} as 0.0 would make every limit check pass.
    api(**{"metadata.get_cost": FakeResponse({"cost": 17.25})})
    assert batch.get_cost("GLBX.MDP3", "NQ.FUT", "ohlcv-1s", "a", "b") == 17.25


def test_an_unreadable_cost_refuses_rather_than_defaulting(api: Any) -> None:
    api(**{"metadata.get_cost": FakeResponse({"unexpected": True})})
    with pytest.raises(DatabentoApiError, match="not a number"):
        batch.get_cost("GLBX.MDP3", "NQ.FUT", "ohlcv-1s", "a", "b")


def test_an_api_refusal_surfaces_the_bodys_own_message(api: Any) -> None:
    api(**{"metadata.get_cost": FakeResponse(status=400)})
    with pytest.raises(DatabentoApiError, match="refused: bad dataset"):
        batch.get_cost("NOPE", "NQ.FUT", "ohlcv-1s", "a", "b")


def test_partially_resolved_symbols_are_refused_not_priced(api: Any) -> None:
    # HTTP 206 is documented as "successful request, with partially resolved
    # symbols": a price for a subset of what was asked, presented as the price.
    api(**{"metadata.get_cost": FakeResponse(12.0, status=206)})
    with pytest.raises(PartialSymbolsError, match="only some of the requested symbols") as e:
        batch.get_cost("GLBX.MDP3", ["NQ.FUT", "NQQ.FUT"], "ohlcv-1s", "a", "b")
    assert isinstance(e.value, DatabentoApiError) and e.value.status == 206


def test_a_missing_key_refuses_before_any_request(api: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    api(**{"metadata.get_cost": FakeResponse(1.0)})
    monkeypatch.delenv("DATABENTO_API_KEY")
    with pytest.raises(ValueError, match="no Databento API key"):
        batch.get_cost("GLBX.MDP3", "NQ.FUT", "ohlcv-1s", "a", "b")


# ---------- submission ----------


def test_a_job_is_priced_before_it_is_submitted(api: Any) -> None:
    fake = api(**{"metadata.get_cost": FakeResponse(42.5), "batch.submit_job": FakeResponse(JOB)})
    job = batch.submit_job(
        "GLBX.MDP3", "NQ.FUT", "ohlcv-1s", "2026-01-01", "2026-07-01", cost_limit_usd=100.0
    )
    assert [url.rsplit("/", 1)[-1] for _, url, _ in fake.calls] == [
        "metadata.get_cost",
        "batch.submit_job",
    ]
    assert job.id == JOB["id"]
    assert job.done and job.cost_usd == 42.5
    assert "$42.50" in job.describe() and "123,456 records" in job.describe()


def test_a_price_over_the_limit_submits_nothing(api: Any) -> None:
    # The refusal has to happen before the POST, not after.
    fake = api(**{"metadata.get_cost": FakeResponse(5_000.0), "batch.submit_job": FakeResponse(JOB)})
    with pytest.raises(CostLimitError) as e:
        batch.submit_job(
            "GLBX.MDP3", "NQ.FUT", "mbo", "2020-01-01", "2026-07-01", cost_limit_usd=250.0
        )
    assert "NOTHING was submitted" in str(e.value)
    assert "$5,000.00" in str(e.value) and "$250.00" in str(e.value)
    assert [url.rsplit("/", 1)[-1] for _, url, _ in fake.calls] == ["metadata.get_cost"]


def test_a_partial_price_submits_nothing(api: Any) -> None:
    fake = api(
        **{"metadata.get_cost": FakeResponse(1.0, status=206), "batch.submit_job": FakeResponse(JOB)}
    )
    with pytest.raises(PartialSymbolsError, match="NOTHING was submitted"):
        batch.submit_job("GLBX.MDP3", "NQ.FUT,NQQ.FUT", "ohlcv-1s", "a", "b", cost_limit_usd=10.0)
    assert [url.rsplit("/", 1)[-1] for _, url, _ in fake.calls] == ["metadata.get_cost"]


def test_a_partial_submit_says_the_job_was_created(api: Any) -> None:
    # 206 is a SUCCESS code: the POST went through, for the symbols that resolved.
    api(
        **{"metadata.get_cost": FakeResponse(1.0), "batch.submit_job": FakeResponse(JOB, status=206)}
    )
    with pytest.raises(PartialSymbolsError, match="WAS created"):
        batch.submit_job("GLBX.MDP3", "NQ.FUT,NQQ.FUT", "ohlcv-1s", "a", "b", cost_limit_usd=10.0)


def test_a_limit_is_mandatory_and_must_be_positive(api: Any) -> None:
    api(**{"metadata.get_cost": FakeResponse(1.0)})
    with pytest.raises(TypeError):
        batch.submit_job("GLBX.MDP3", "NQ.FUT", "ohlcv-1s", "a", "b")  # pyright: ignore[reportCallIssue]
    with pytest.raises(ValueError, match="must be positive"):
        batch.submit_job("GLBX.MDP3", "NQ.FUT", "ohlcv-1s", "a", "b", cost_limit_usd=0)


def test_the_lake_defaults_are_what_ingest_can_decode(api: Any) -> None:
    fake = api(**{"metadata.get_cost": FakeResponse(1.0), "batch.submit_job": FakeResponse(JOB)})
    batch.submit_job("GLBX.MDP3", "NQ.FUT", "ohlcv-1s", "a", "b", cost_limit_usd=10.0)
    data = fake.calls[1][2]["data"]
    assert data["encoding"] == "dbn" and data["compression"] == "zstd"
    # One archive per job, so one file is the whole provenance root.
    assert data["split_duration"] == "none"
    # `parent` buys every contract of a root, which is what makes a causal
    # front-month re-ranking possible later.
    assert data["stype_in"] == "parent"
    assert data["delivery"] == "download"


def test_a_job_payload_without_an_id_refuses(api: Any) -> None:
    api(**{"metadata.get_cost": FakeResponse(1.0), "batch.submit_job": FakeResponse({"state": "done"})})
    with pytest.raises(DatabentoApiError, match="no id/state"):
        batch.submit_job("GLBX.MDP3", "NQ.FUT", "ohlcv-1s", "a", "b", cost_limit_usd=10.0)


def test_list_jobs_passes_the_state_filter_through(api: Any) -> None:
    fake = api(**{"batch.list_jobs": FakeResponse([JOB])})
    assert [j.id for j in batch.list_jobs(states=["done", "expired"])] == [JOB["id"]]
    assert fake.calls[0][2]["params"] == {"states": "done,expired"}
    assert len(fake.calls) == 1  # a full-form listing needs no per-job lookups


def test_a_short_form_listing_is_filled_in_from_job_details(api: Any) -> None:
    # The API has announced the listing will shrink to id/state/ts_received.
    short = {"id": JOB["id"], "state": "done", "ts_received": "2026-06-10T00:00:00Z"}
    fake = api(**{"batch.list_jobs": FakeResponse([short]), **DETAILS})
    (job,) = batch.list_jobs()
    assert not job.short and job.dataset == "GLBX.MDP3" and job.cost_usd == 42.5
    assert [url.rsplit("/", 1)[-1] for _, url, _ in fake.calls] == [
        "batch.list_jobs",
        "batch.get_job_details",
    ]
    assert fake.calls[1][2]["params"] == {"job_id": JOB["id"]}


def test_find_job_uses_the_single_job_endpoint(api: Any) -> None:
    fake = api(**DETAILS)
    assert batch.find_job(JOB["id"]).done
    [(_, url, kwargs)] = fake.calls
    assert url.endswith("/batch.get_job_details") and kwargs["params"] == {"job_id": JOB["id"]}


def test_an_unknown_job_is_named_in_the_refusal(api: Any) -> None:
    not_found = '{"detail":{"case":"batch_job_not_found","message":"Job not found."}}'
    api(**{"batch.get_job_details": FakeResponse(status=404, text=not_found)})
    with pytest.raises(DatabentoApiError, match="no batch job 'not-a-job'") as e:
        batch.find_job("not-a-job")
    assert e.value.status == 404


def test_an_expired_job_is_still_found(api: Any) -> None:
    # The listing omits expired jobs by default; the single-job endpoint does
    # not, which is what makes `expired` a state wait_for_job can actually return.
    api(**{"batch.get_job_details": FakeResponse({**JOB, "state": "expired"})})
    assert batch.find_job(JOB["id"]).state == "expired"
    waited = batch.wait_for_job(JOB["id"], clock=ReplayClock(0), timeout_seconds=0.0)
    assert waited.state == "expired"


def test_wait_for_job_returns_on_a_terminal_state(api: Any) -> None:
    api(**DETAILS)
    assert batch.wait_for_job(JOB["id"], clock=ReplayClock(0), timeout_seconds=60.0).done


def test_wait_for_job_polls_until_the_job_finishes(api: Any) -> None:
    states = iter(["queued", "processing", "done"])
    api(**{"batch.get_job_details": lambda: FakeResponse({**JOB, "state": next(states)})})
    clock = ReplayClock(0)
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        clock.advance(clock.now_unix() + int(seconds))

    job = batch.wait_for_job(
        JOB["id"], clock=clock, timeout_seconds=600.0, poll_seconds=30.0, sleep=sleep
    )
    assert job.done
    assert slept == [30.0, 30.0]


def test_wait_for_job_times_out_without_cancelling_anything(api: Any) -> None:
    # The message has to say so, or someone re-submits and pays twice.
    api(**{"batch.get_job_details": FakeResponse({**JOB, "state": "queued"})})
    with pytest.raises(DatabentoApiError, match="nothing was lost"):
        batch.wait_for_job(JOB["id"], clock=ReplayClock(0), timeout_seconds=0.0)


# ---------- download ----------


def _file_payload(body: bytes, *, name: str = "nq.dbn.zst") -> dict[str, Any]:
    return {
        "filename": name,
        "size": len(body),
        "hash": "sha256:" + hashlib.sha256(body).hexdigest(),
        "urls": {"https": f"https://hist.databento.com/v0/batch/download/{name}"},
    }


def test_a_verified_download_lands_under_its_real_name(api: Any, tmp_path: Path) -> None:
    body = b"\x01\x02\x03 pretend this is DBN"
    api(
        **{
            "batch.list_files": FakeResponse([_file_payload(body)]),
            "batch/download": FakeResponse(body=body),
            **DETAILS,
        }
    )
    (path,) = batch.download_job(JOB["id"], tmp_path / "archives")
    assert path.read_bytes() == body
    assert path == tmp_path / "archives" / JOB["id"] / "nq.dbn.zst"  # the job's own directory
    assert list(path.parent.glob("*.part")) == []  # no temp left behind


def test_a_truncated_download_never_becomes_visible(api: Any, tmp_path: Path) -> None:
    body = b"the whole file"
    api(
        **{
            "batch.list_files": FakeResponse([_file_payload(body)]),
            "batch/download": FakeResponse(body=b"the wh"),
            **DETAILS,
        }
    )
    with pytest.raises(DatabentoApiError, match="downloaded 6 bytes"):
        batch.download_job(JOB["id"], tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_a_corrupted_download_is_caught_by_the_hash(api: Any, tmp_path: Path) -> None:
    # Right length, wrong bytes: the size check cannot see this one.
    body = b"the whole file"
    api(
        **{
            "batch.list_files": FakeResponse([_file_payload(body)]),
            "batch/download": FakeResponse(body=b"the whole FILE"),
            **DETAILS,
        }
    )
    with pytest.raises(DatabentoApiError, match="sha256 mismatch"):
        batch.download_job(JOB["id"], tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_an_already_verified_file_is_not_re_fetched(api: Any, tmp_path: Path) -> None:
    body = b"already here"
    fake = api(
        **{
            "batch.list_files": FakeResponse([_file_payload(body)]),
            "batch/download": FakeResponse(body=body),
            **DETAILS,
        }
    )
    (tmp_path / JOB["id"]).mkdir()
    (tmp_path / JOB["id"] / "nq.dbn.zst").write_bytes(body)
    batch.download_job(JOB["id"], tmp_path)
    assert [url for _, url, _ in fake.calls if "download" in url] == []


def test_a_present_but_wrong_file_is_re_fetched(api: Any, tmp_path: Path) -> None:
    body = b"the right bytes"
    api(
        **{
            "batch.list_files": FakeResponse([_file_payload(body)]),
            "batch/download": FakeResponse(body=body),
            **DETAILS,
        }
    )
    (tmp_path / JOB["id"]).mkdir()
    (tmp_path / JOB["id"] / "nq.dbn.zst").write_bytes(b"stale")
    (path,) = batch.download_job(JOB["id"], tmp_path)
    assert path.read_bytes() == body


def _job_files(job_id: str, archive: str, data: bytes) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    """A job as the vendor lists it: its archive, and the manifest naming that archive."""
    entry = _file_payload(data, name=archive)
    manifest = json.dumps({"job_id": job_id, "files": [entry]}).encode()
    bodies = {archive: data, "manifest.json": manifest}
    listing = [entry, _file_payload(manifest, name="manifest.json")]
    for payload in listing:
        payload["urls"] = {"https": f"https://hist/v0/batch/download/{job_id}/{payload['filename']}"}
    return listing, bodies


SAME_NAME = "glbx-mdp3-20260824.mbo.dbn.zst"


def _two_jobs_fetched(monkeypatch: pytest.MonkeyPatch, dest: Path) -> dict[str, dict[str, bytes]]:
    """Two jobs downloaded into `dest`, one after the other: what each one holds."""
    monkeypatch.setenv("DATABENTO_API_KEY", KEY)
    jobs = {
        "GLBX-20260909-AAAAAAAAAA": _job_files("GLBX-20260909-AAAAAAAAAA", SAME_NAME, b"NQ"),
        "GLBX-20260910-BBBBBBBBBB": _job_files("GLBX-20260910-BBBBBBBBBB", SAME_NAME, b"ES, longer"),
    }

    def get(url: str, **kwargs: Any) -> FakeResponse:
        asked = str(dict(kwargs.get("params") or {}).get("job_id"))
        if "batch.list_files" in url:
            return FakeResponse(jobs[asked][0])
        if "batch/download" in url:
            _, job_id, name = url.rsplit("/", 2)
            return FakeResponse(body=jobs[job_id][1][name])
        return FakeResponse({**JOB, "id": asked})

    module = types.ModuleType("requests")
    module.get = get  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setitem(sys.modules, "requests", module)

    for job_id in jobs:
        batch.download_job(job_id, dest)
    # Fetched again, the first job finds its files where it left them.
    first = next(iter(jobs))
    assert all(path.parent == dest / first for path in batch.download_job(first, dest))
    return {job_id: bodies for job_id, (_, bodies) in jobs.items()}


def test_two_jobs_into_one_destination_keep_their_own_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dest = tmp_path / "archives"
    jobs = _two_jobs_fetched(monkeypatch, dest)
    assert sorted(path.name for path in dest.iterdir()) == sorted(jobs)
    for job_id, bodies in jobs.items():
        assert (dest / job_id / SAME_NAME).read_bytes() == bodies[SAME_NAME]
        manifest = json.loads((dest / job_id / "manifest.json").read_text())
        (listed,) = manifest["files"]
        assert manifest["job_id"] == job_id
        assert listed["hash"] == "sha256:" + hashlib.sha256(bodies[SAME_NAME]).hexdigest()


def test_each_jobs_archive_still_passes_its_vendor_manifest_after_another_fetch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("databento_dbn", reason="the lake extra is not installed")
    from bedivere.data.lake.trade_archive import verify_manifest

    dest = tmp_path / "archives"
    for job_id, bodies in _two_jobs_fetched(monkeypatch, dest).items():
        # The strict check, unchanged: listed exactly once, same size and SHA256.
        digest = hashlib.sha256(bodies[SAME_NAME]).hexdigest()
        assert verify_manifest(dest / job_id / SAME_NAME, digest) is True


def test_a_job_id_that_is_not_a_plain_name_is_refused_before_any_request(
    api: Any, tmp_path: Path
) -> None:
    fake = api()
    for job_id in ("../escape", "sub/job", "..", ""):
        with pytest.raises(DatabentoApiError, match="not a plain name"):
            batch.download_job(job_id, tmp_path)
    assert fake.calls == [] and list(tmp_path.iterdir()) == []


def test_a_filename_that_is_not_a_plain_name_is_refused(api: Any, tmp_path: Path) -> None:
    # The filename is joined onto a local directory.
    for name in ("../escape.dbn", "sub/dir.dbn", ".."):
        api(**{"batch.list_files": FakeResponse([_file_payload(b"x", name=name)]), **DETAILS})
        with pytest.raises(DatabentoApiError, match="not a plain name"):
            batch.download_job(JOB["id"], tmp_path)


def test_downloading_an_unfinished_job_names_its_state_and_asks_for_nothing(
    api: Any, tmp_path: Path
) -> None:
    fake = api(
        **{
            "batch.get_job_details": FakeResponse({**JOB, "state": "processing", "progress": 40}),
            "batch.list_files": FakeResponse([]),
        }
    )
    with pytest.raises(DatabentoApiError, match=r"is 'processing' \(40%\), not 'done'"):
        batch.download_job(JOB["id"], tmp_path / "archives")
    assert [url for _, url, _ in fake.calls if "list_files" in url] == []
    assert not (tmp_path / "archives").exists()


def test_downloading_an_expired_job_says_the_files_are_gone(api: Any, tmp_path: Path) -> None:
    api(**{"batch.get_job_details": FakeResponse({**JOB, "state": "expired"})})
    with pytest.raises(DatabentoApiError, match="expired.*files are gone"):
        batch.download_job(JOB["id"], tmp_path)


def test_a_done_job_with_no_files_is_reported_rather_than_guessed_at(
    api: Any, tmp_path: Path
) -> None:
    api(**{"batch.list_files": FakeResponse([]), **DETAILS})
    with pytest.raises(DatabentoApiError, match="'done' but lists no files"):
        batch.download_job(JOB["id"], tmp_path)


def test_a_file_payload_without_a_url_refuses() -> None:
    with pytest.raises(DatabentoApiError, match="no filename/https url"):
        BatchFile.from_payload({"filename": "nq.dbn.zst", "size": 1})
    with pytest.raises(DatabentoApiError, match="must be an object"):
        BatchJob.from_payload(["not", "an", "object"])


def test_describe_shows_progress_only_while_processing() -> None:
    processing = BatchJob.from_payload({**JOB, "state": "processing", "progress": 40})
    assert "processing 40%" in processing.describe()
    assert "100%" not in BatchJob.from_payload({**JOB, "progress": 100}).describe()


# ---------- conditions ----------


def test_conditions_use_inclusive_dates_and_a_refusal_quotes_the_body(api: Any) -> None:
    from bedivere.data.vendor.databento.condition import fetch_conditions

    day = {"date": "2026-09-01", "condition": "available", "last_modified_date": "2026-09-02"}
    fake = api(**{"metadata.get_dataset_condition": FakeResponse([day])})
    (row,) = fetch_conditions("GLBX.MDP3", "2026-09-01", "2026-09-01")
    assert row.date == "2026-09-01"
    assert fake.calls[0][2]["params"] == {
        "dataset": "GLBX.MDP3",
        "start_date": "2026-09-01",
        "end_date": "2026-09-01",
    }

    refusal = '{"detail":"start_date must be an ISO 8601 date"}'
    api(**{"metadata.get_dataset_condition": FakeResponse(status=400, text=refusal)})
    with pytest.raises(DatabentoApiError, match="HTTP 400 — .*start_date must be"):
        fetch_conditions("GLBX.MDP3", "yesterday", "2026-09-01")


def test_the_conditions_command_reports_an_api_refusal_in_one_line(
    api: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("duckdb", reason="the lake extra is not installed")
    from bedivere.cli import data as data_cli

    refusal = '{"detail":"start_date must be an ISO 8601 date"}'
    api(**{"metadata.get_dataset_condition": FakeResponse(status=400, text=refusal)})
    argv = ["conditions", "--fetch", "--start", "x", "--end", "y", "--root", str(tmp_path)]
    assert data_cli.main(argv) == 1
    err = capsys.readouterr().err
    assert "error: metadata.get_dataset_condition: HTTP 400 — " in err
    assert "start_date must be" in err
    assert "DatabentoApiError" not in err  # a refusal, not a bedivere bug


# ---------- the missing-dependency seam ----------


def test_a_missing_requests_names_the_extra_rather_than_raising_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`requests` is in the lake extra, so a base install reaching for the API is
    a normal state, not a bug. It has to be told which extra carries it — the
    same courtesy `lake_module` gives the DuckDB stack."""
    from bedivere.data.vendor.databento.condition import (
        VendorHttpUnavailable,
        fetch_conditions,
        http_client,
    )

    monkeypatch.setenv("DATABENTO_API_KEY", KEY)
    monkeypatch.setitem(sys.modules, "requests", None)  # makes `import requests` raise

    for call in (
        http_client,
        lambda: fetch_conditions("GLBX.MDP3", "2026-01-01", "2026-01-02"),
        lambda: batch.get_cost("GLBX.MDP3", "NQ.FUT", "ohlcv-1s", "a", "b"),
        lambda: batch.list_jobs(),
        lambda: batch.list_files("job-1"),
    ):
        with pytest.raises(VendorHttpUnavailable) as e:
            call()
        assert "bedivere[lake]" in str(e.value)
        assert "requests" in str(e.value)


def test_the_dependency_refusal_is_an_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Subclassing ImportError keeps any pre-existing `except ImportError` around
    # a vendor call working, rather than turning a soft probe into a hard failure.
    from bedivere.data.vendor.databento.condition import VendorHttpUnavailable, http_client

    monkeypatch.setitem(sys.modules, "requests", None)
    with pytest.raises(ImportError):
        http_client()
    assert issubclass(VendorHttpUnavailable, ImportError)
