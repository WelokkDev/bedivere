"""Discord webhook Notifier — a queue and one daemon worker.

send() only enqueues; the worker does the HTTP, so the engine loop never
waits on the network and never sees an exception. A full queue drops loudly
rather than stall a trading decision. 429s sleep out Retry-After; other
transient failures retry with backoff then drop loudly; non-429 4xx drops
at once. No wall-clock reads: pacing is sleep-based and Retry-After
arrives as data.
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from time import sleep as _sleep

# Discord caps content at 2000 chars; leave headroom for safety.
_MAX_CONTENT = 1900
_MAX_RETRIES = 3
_POST_TIMEOUT_S = 10.0
# Polite inter-post pause; the webhook bucket is ~30/min with small bursts.
_PACE_S = 0.4
_RETRY_AFTER_CAP_S = 30.0

_SENTINEL = object()


def _default_post(url: str, body: bytes, timeout_s: float) -> int:
    """POST JSON, return the HTTP status. Raises urllib errors on failure."""
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "bedivere/0.1"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as res:  # noqa: S310 — operator-configured webhook URL
        return int(res.status)


class DiscordNotifier:
    """Queue + daemon worker over one webhook URL; `post`/`sleeper` are
    injectable for tests."""

    def __init__(
        self,
        webhook_url: str,
        *,
        max_queue: int = 256,
        post: Callable[[str, bytes, float], int] | None = None,
        sleeper: Callable[[float], None] = _sleep,
    ) -> None:
        if not webhook_url.startswith("https://"):
            raise ValueError("Discord webhook URL must be https")
        self._url = webhook_url
        self._post = post or _default_post
        self._sleep = sleeper
        self._q: queue.Queue[object] = queue.Queue(maxsize=max_queue)
        self.dropped = 0
        self.delivered = 0
        self._closed = False
        self._worker = threading.Thread(
            target=self._run, name="bedivere-discord-notify", daemon=True
        )
        self._worker.start()

    # ---------- port ----------

    def send(self, text: str) -> None:
        if self._closed:
            return
        try:
            self._q.put_nowait(text[:_MAX_CONTENT])
        except queue.Full:
            self.dropped += 1
            sys.stderr.write("[notify] discord queue full — dropping a message\n")

    def close(self) -> None:
        """Flush best-effort: stop accepting, let the worker drain, join."""
        if self._closed:
            return
        self._closed = True
        try:
            self._q.put(_SENTINEL, timeout=1.0)
        except queue.Full:
            # Worker is wedged behind retries; the daemon thread dies with us.
            sys.stderr.write("[notify] discord queue full at close — undelivered messages lost\n")
            return
        self._worker.join(timeout=15.0)
        if self._worker.is_alive():
            sys.stderr.write("[notify] discord worker did not drain before timeout\n")

    # ---------- worker ----------

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is _SENTINEL:
                return
            self._deliver(str(item))
            self._sleep(_PACE_S)

    def _deliver(self, text: str) -> None:
        body = json.dumps({"content": text}).encode("utf-8")
        backoff = 1.0
        for attempt in range(1 + _MAX_RETRIES):
            try:
                self._post(self._url, body, _POST_TIMEOUT_S)
                self.delivered += 1
                return
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    # Spends an attempt — bounded even under endless 429s.
                    self._sleep(self._retry_after(e))
                    continue
                if 400 <= e.code < 500:
                    self.dropped += 1
                    sys.stderr.write(f"[notify] discord rejected message (HTTP {e.code})\n")
                    return
                # 5xx: transient, fall through to backoff.
            except Exception as e:  # noqa: BLE001 — network errors are expected here
                if attempt == _MAX_RETRIES:
                    self.dropped += 1
                    sys.stderr.write(f"[notify] discord delivery failed, dropping: {e}\n")
                    return
            if attempt < _MAX_RETRIES:
                self._sleep(backoff)
                backoff = min(backoff * 2, 8.0)
        self.dropped += 1
        sys.stderr.write("[notify] discord delivery failed after retries, dropping\n")

    @staticmethod
    def _retry_after(e: urllib.error.HTTPError) -> float:
        headers = e.headers
        raw = headers.get("Retry-After") if headers else None
        try:
            return min(max(float(raw), 0.5), _RETRY_AFTER_CAP_S) if raw else 2.0
        except ValueError:
            return 2.0
