"""bedivere.notify — transports never block or raise; routing is loud."""

from __future__ import annotations

import urllib.error
from email.message import Message
from typing import Any

import pytest

from bedivere.notify.discord import DiscordNotifier
from bedivere.notify.format import format_event
from bedivere.notify.port import ConsoleNotifier, NullNotifier
from bedivere.notify.router import DEFAULT_KINDS, VENUE_KINDS, NotificationRouter


class FakeNotifier:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed = False

    def send(self, text: str) -> None:
        self.sent.append(text)

    def close(self) -> None:
        self.closed = True


# ---------- port basics ----------


def test_console_and_null_never_raise(capsys: pytest.CaptureFixture[str]) -> None:
    NullNotifier().send("dropped")
    console = ConsoleNotifier(prefix="[t]")
    console.send("hello")
    console.close()
    assert "[t] hello" in capsys.readouterr().err


# ---------- discord worker ----------


def _http_error(code: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError("https://x", code, "err", headers, None)


def _drained(notifier: DiscordNotifier) -> DiscordNotifier:
    notifier.close()  # close() joins the worker → deterministic assertions
    return notifier


def test_delivers_and_counts() -> None:
    posts: list[bytes] = []

    def post(_url: str, body: bytes, _timeout: float) -> int:
        posts.append(body)
        return 204

    n = DiscordNotifier("https://hook", post=post, sleeper=lambda _s: None)
    n.send("alpha")
    n.send("beta")
    _drained(n)
    assert n.delivered == 2 and n.dropped == 0
    assert b"alpha" in posts[0] and b"beta" in posts[1]
    n.close()  # idempotent


def test_429_sleeps_out_retry_after_then_delivers() -> None:
    calls: dict[str, Any] = {"n": 0}
    naps: list[float] = []

    def post(_url: str, _body: bytes, _timeout: float) -> int:
        calls["n"] += 1
        if calls["n"] == 1:
            raise _http_error(429, retry_after="3")
        return 204

    n = DiscordNotifier("https://hook", post=post, sleeper=naps.append)
    n.send("rate-limited once")
    _drained(n)
    assert n.delivered == 1
    assert 3.0 in naps  # honored the server's Retry-After


def test_non_429_client_error_drops_at_once() -> None:
    def post(_url: str, _body: bytes, _timeout: float) -> int:
        raise _http_error(404)

    n = DiscordNotifier("https://hook", post=post, sleeper=lambda _s: None)
    n.send("bad hook")
    _drained(n)
    assert n.dropped == 1 and n.delivered == 0


def test_network_failure_retries_then_drops_loudly(
    capsys: pytest.CaptureFixture[str],
) -> None:
    attempts: dict[str, int] = {"n": 0}

    def post(_url: str, _body: bytes, _timeout: float) -> int:
        attempts["n"] += 1
        raise OSError("connection reset")

    n = DiscordNotifier("https://hook", post=post, sleeper=lambda _s: None)
    n.send("doomed")
    _drained(n)
    assert attempts["n"] == 4  # 1 try + 3 retries
    assert n.dropped == 1
    assert "dropping" in capsys.readouterr().err


def test_full_queue_drops_without_blocking() -> None:
    import threading

    release = threading.Event()

    def post(_url: str, _body: bytes, _timeout: float) -> int:
        release.wait(5)
        return 204

    n = DiscordNotifier("https://hook", max_queue=1, post=post, sleeper=lambda _s: None)
    n.send("worker takes this")
    n.send("queued")
    n.send("this one must drop — queue full")
    assert n.dropped >= 1
    release.set()
    n.close()


def test_https_required() -> None:
    with pytest.raises(ValueError, match="https"):
        DiscordNotifier("http://insecure")


# ---------- routing ----------


def test_router_filters_formats_and_counts() -> None:
    sink = FakeNotifier()
    router = NotificationRouter({"entry_fill", "custom_signal"}, sink)
    router({"kind": "entry_fill", "ts": 1_784_930_400, "symbol": "NQ", "direction": "long", "qty": 1, "priceTicks": 404})
    router({"kind": "stop_fill", "ts": 1, "symbol": "NQ"})  # not subscribed
    router({"kind": "custom_signal", "ts": 1_784_930_400, "symbol": "NQ", "score": 0.8})
    assert router.routed == 2
    assert "ENTRY" in sink.sent[0]
    assert "custom_signal" in sink.sent[1] and "score=0.8" in sink.sent[1]


def test_router_typo_protection_when_kinds_declared() -> None:
    with pytest.raises(ValueError, match="unknown notify kind"):
        NotificationRouter({"entry_fil"}, FakeNotifier(), known_kinds={"my_signal"})
    # Venue kinds are always known; declared strategy kinds extend the set.
    NotificationRouter({"entry_fill", "my_signal"}, FakeNotifier(), known_kinds={"my_signal"})


def test_router_contains_notifier_failures(capsys: pytest.CaptureFixture[str]) -> None:
    class Exploding:
        def send(self, text: str) -> None:
            raise RuntimeError("boom")

        def close(self) -> None:
            return

    router = NotificationRouter({"entry_fill"}, Exploding())
    router({"kind": "entry_fill", "ts": 1})  # must not raise
    assert router.routed == 0
    assert "routing entry_fill failed" in capsys.readouterr().err


def test_default_and_venue_kind_sets() -> None:
    assert DEFAULT_KINDS <= VENUE_KINDS
    assert "entry_fill" in VENUE_KINDS and "cancelled" in VENUE_KINDS


# ---------- formatting ----------


def test_format_event_is_total() -> None:
    # A malformed event degrades to the fallback, never an exception.
    assert format_event({}) .startswith("ℹ")
    line = format_event({"kind": "target_fill", "ts": 1_784_930_400, "symbol": "NQ", "priceTicks": 412, "ambiguous": True})
    assert "TARGET" in line and "412t" in line and "ambiguous" in line
    fallback = format_event({"kind": "made_up", "ts": 1_784_930_400, "vec": [1, 2], "note": "x"})
    assert "made_up" in fallback and "note=x" in fallback and "vec" not in fallback
