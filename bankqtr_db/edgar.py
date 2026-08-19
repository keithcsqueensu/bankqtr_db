"""Rate-limited, on-disk-cached HTTP access to SEC EDGAR."""

from __future__ import annotations

import gzip
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from .config import (
    COMPANYFACTS_URL,
    EDGAR_HEADERS,
    MAX_REQUESTS_PER_SECOND,
    RAW_FACTS,
    SUBMISSIONS_URL,
)

log = logging.getLogger(__name__)


class _RateLimiter:
    """Global token-spacing limiter shared by every EDGAR call in the process."""

    def __init__(self, rps: float) -> None:
        self._min_interval = 1.0 / rps
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            sleep_for = self._min_interval - (now - self._last)
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last = time.monotonic()


_limiter = _RateLimiter(MAX_REQUESTS_PER_SECOND)

_client: httpx.Client | None = None


def client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(
            headers=EDGAR_HEADERS,
            timeout=httpx.Timeout(60.0, connect=20.0),
            follow_redirects=True,
        )
    return _client


def get(url: str, *, retries: int = 4) -> httpx.Response:
    """GET with rate limiting and backoff on 429/5xx."""
    delay = 1.0
    last_exc: Exception | None = None
    for attempt in range(retries):
        _limiter.wait()
        try:
            resp = client().get(url)
        except httpx.HTTPError as exc:  # transport-level failure
            last_exc = exc
            log.warning("GET %s failed (%s), retry %d", url, exc, attempt + 1)
        else:
            if resp.status_code == 200:
                return resp
            if resp.status_code == 404:
                resp.raise_for_status()
            last_exc = httpx.HTTPStatusError(
                f"{resp.status_code} for {url}", request=resp.request, response=resp
            )
            log.warning("GET %s -> %d, retry %d", url, resp.status_code, attempt + 1)
        time.sleep(delay)
        delay *= 2
    assert last_exc is not None
    raise last_exc


# --------------------------------------------------------------------------
# Company facts (cached gzipped, since each payload is 2-30 MB of JSON)
# --------------------------------------------------------------------------


def _facts_path(cik: str) -> Path:
    return RAW_FACTS / f"CIK{cik}.json.gz"


def company_facts(cik: str, *, refresh: bool = False) -> dict[str, Any]:
    path = _facts_path(cik)
    if path.exists() and not refresh:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)

    path.parent.mkdir(parents=True, exist_ok=True)
    resp = get(COMPANYFACTS_URL.format(cik=cik))
    payload = resp.json()
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    tmp.replace(path)
    return payload


def submissions_shard(name: str, *, refresh: bool = False) -> dict[str, Any]:
    """One overflow shard of a submissions index, cached like the main file.

    ``filings.list_filings`` walks every shard for every bank on every call,
    and a universe run calls it once per bank per stage.  Uncached that is
    dozens of requests for data that only changes when a bank files -- and it
    is what draws a 429 first, because the shards are fetched in a tight loop
    while the main files are all coming from disk.

    Cached on the same terms as :func:`submissions`: a build sees the filing
    index as of the last refresh, which is already true of the main file, so
    this does not make the window any staler than it was.
    """
    safe = "".join(c for c in name if c.isalnum() or c in "-_.")
    path = RAW_FACTS / f"SUBSHARD_{safe}.gz"
    if path.exists() and not refresh:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = get(f"https://data.sec.gov/submissions/{name}").json()
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    tmp.replace(path)
    return payload


def submissions(cik: str, *, refresh: bool = False) -> dict[str, Any]:
    path = RAW_FACTS / f"SUB{cik}.json.gz"
    if path.exists() and not refresh:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)

    path.parent.mkdir(parents=True, exist_ok=True)
    resp = get(SUBMISSIONS_URL.format(cik=cik))
    payload = resp.json()
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    tmp.replace(path)
    return payload
