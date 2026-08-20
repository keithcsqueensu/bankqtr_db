"""Rate-limited, on-disk-cached access to FFIEC CDR bulk Call Report data.

The download is not an API
-------------------------
``cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx`` is an ASP.NET WebForm, and
there is no REST endpoint behind it.  A download is four POSTs, each carrying
the ``__VIEWSTATE`` the previous response returned:

1. select the product (``ReportingSeriesSinglePeriod`` -- all schedules, one
   quarter),
2. select the period, by the server's own opaque numeric id,
3. select the format (tab-delimited; the alternative is XBRL, which is the
   same numbers wrapped in several times the bytes),
4. press Download.

Steps 2 and 3 look redundant -- the final POST names the period and format
again -- but the server rejects the download unless its ``__VIEWSTATE`` says
it has already seen those choices, so the round trips are load-bearing.

The period ids are **not stable and not derivable**: 2025Q3 is 149 and 2025Q2
is 147, with 148 skipped entirely.  They are read from the dates dropdown on
every run and mapped by the label, which is a real date.

Each zip is ~6 MB and holds every schedule for every filer in that quarter --
around 4,400 institutions.  Cached under ``data/raw/call`` keyed on the period,
so a rebuild is offline and reproducible in the same way ``build_panel --ir``
is.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import threading
import time
from html import unescape
from pathlib import Path

import httpx

from .config import (
    CDR_BULK_URL,
    FFIEC_HEADERS,
    MAX_REQUESTS_PER_SECOND,
    RAW_CALL,
)

log = logging.getLogger(__name__)

PRODUCT_ALL_SCHEDULES = "ReportingSeriesSinglePeriod"

_PREFIX = "ctl00$MainContentHolder$"
_F_PRODUCT = _PREFIX + "ListBox1"
_F_DATES = _PREFIX + "DatesDropDownList"
_F_FORMAT = _PREFIX + "FormatType"
_F_DOWNLOAD = _PREFIX + "TabStrip1$Download_0"
_FORMAT_TSV = "TSVRadioButton"

_HIDDEN_RE = re.compile(r'<input type="hidden" name="([^"]+)"[^>]*value="([^"]*)"')
_DATES_SELECT_RE = re.compile(r'id="DatesDropDownList".*?</select>', re.DOTALL)
_OPTION_RE = re.compile(r'<option[^>]*value="([^"]*)"[^>]*>([^<]*)</option>')

# A good period zip is several megabytes; anything near-empty is an error page
# or a truncated stream, and must not be cached as if it were data.
MIN_ZIP_BYTES = 100_000


class _RateLimiter:
    """Token spacing shared by every FFIEC call in the process."""

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


def _hidden_fields(html: str) -> dict[str, str]:
    """The ``__VIEWSTATE`` family, unescaped so it survives a round trip.

    The viewstate is HTML-escaped in the page and has to be posted back in its
    raw form; posting the escaped text yields a 500 with no explanation.
    """
    return {k: unescape(v) for k, v in _HIDDEN_RE.findall(html)}


def new_client() -> httpx.Client:
    return httpx.Client(
        headers=FFIEC_HEADERS,
        timeout=httpx.Timeout(300.0, connect=30.0),
        follow_redirects=True,
    )


def _post(client: httpx.Client, data: dict[str, str]) -> httpx.Response:
    _limiter.wait()
    resp = client.post(CDR_BULK_URL, data=data)
    resp.raise_for_status()
    return resp


def quarter_end(period: str) -> dt.date:
    """``2025Q4`` -> the date the quarter ends on."""
    year, q = int(period[:4]), int(period[-1])
    month = q * 3
    day = 31 if month in (3, 12) else 30
    return dt.date(year, month, day)


def period_label(period: str) -> str:
    """``2025Q4`` -> ``12/31/2025``, the label CDR's dropdown uses."""
    d = quarter_end(period)
    return f"{d.month:02d}/{d.day:02d}/{d.year}"


def _label_to_period(label: str) -> str | None:
    m = re.fullmatch(r"(\d{2})/(\d{2})/(\d{4})", label.strip())
    if not m:
        return None
    month, year = int(m.group(1)), m.group(3)
    if month % 3:
        return None
    return f"{year}Q{month // 3}"


def available_periods(client: httpx.Client | None = None) -> dict[str, str]:
    """Map ``2025Q4`` -> the server's opaque period id, for every period offered.

    Read on every run rather than cached: the ids are assigned by the server,
    are not contiguous, and a new quarter shifts nothing but adds one.
    """
    own = client is None
    session = client if client is not None else new_client()
    try:
        _limiter.wait()
        resp = session.get(CDR_BULK_URL)
        resp.raise_for_status()
        data = _hidden_fields(resp.text)
        data["__EVENTTARGET"] = _F_PRODUCT
        data["__EVENTARGUMENT"] = ""
        data[_F_PRODUCT] = PRODUCT_ALL_SCHEDULES
        resp = _post(session, data)
        block = _DATES_SELECT_RE.search(resp.text)
        if block is None:
            raise RuntimeError("CDR returned no period list; the form changed")
        out: dict[str, str] = {}
        for value, label in _OPTION_RE.findall(block.group(0)):
            period = _label_to_period(unescape(label))
            if period:
                out[period] = value
        if not out:
            raise RuntimeError("CDR period list parsed to nothing")
        return out
    finally:
        if own:
            session.close()


def zip_path(period: str) -> Path:
    return RAW_CALL / f"call_{period}.zip"


def fetch_period(
    period: str,
    *,
    refresh: bool = False,
    client: httpx.Client | None = None,
    periods: dict[str, str] | None = None,
) -> Path:
    """Download one quarter's all-schedules bulk zip, or return the cached one."""
    path = zip_path(period)
    if path.exists() and path.stat().st_size >= MIN_ZIP_BYTES and not refresh:
        return path

    own = client is None
    session = client if client is not None else new_client()
    try:
        ids = periods if periods is not None else available_periods(session)
        if period not in ids:
            raise KeyError(
                f"{period} is not offered by CDR (available {min(ids)}..{max(ids)})"
            )
        period_id = ids[period]

        _limiter.wait()
        resp = session.get(CDR_BULK_URL)
        resp.raise_for_status()

        # 1. product
        data = _hidden_fields(resp.text)
        data["__EVENTTARGET"] = _F_PRODUCT
        data[_F_PRODUCT] = PRODUCT_ALL_SCHEDULES
        resp = _post(session, data)

        # 2. period
        data = _hidden_fields(resp.text)
        data["__EVENTTARGET"] = _F_DATES
        data[_F_PRODUCT] = PRODUCT_ALL_SCHEDULES
        data[_F_DATES] = period_id
        resp = _post(session, data)

        # 3. format
        data = _hidden_fields(resp.text)
        data["__EVENTTARGET"] = _PREFIX + _FORMAT_TSV
        data[_F_PRODUCT] = PRODUCT_ALL_SCHEDULES
        data[_F_DATES] = period_id
        data[_F_FORMAT] = _FORMAT_TSV
        resp = _post(session, data)

        # 4. download
        data = _hidden_fields(resp.text)
        data.pop("__EVENTTARGET", None)
        data[_F_PRODUCT] = PRODUCT_ALL_SCHEDULES
        data[_F_DATES] = period_id
        data[_F_FORMAT] = _FORMAT_TSV
        data[_F_DOWNLOAD] = "Download"

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        _limiter.wait()
        with session.stream("POST", CDR_BULK_URL, data=data) as stream:
            stream.raise_for_status()
            ctype = stream.headers.get("content-type", "")
            if "html" in ctype:
                raise RuntimeError(
                    f"CDR served a page rather than a file for {period} "
                    f"({ctype}); the form flow changed"
                )
            written = 0
            with tmp.open("wb") as fh:
                for chunk in stream.iter_bytes():
                    fh.write(chunk)
                    written += len(chunk)
        if written < MIN_ZIP_BYTES or tmp.read_bytes()[:2] != b"PK":
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"CDR returned {written} bytes for {period}, not a zip")
        tmp.replace(path)
        log.info("fetched %s (%.1f MB)", period, written / 1e6)
        return path
    finally:
        if own:
            session.close()


def periods_between(start: dt.date, end: dt.date) -> list[str]:
    """Every quarter end in ``[start, end]``, as ``YYYYQn``."""
    out: list[str] = []
    for year in range(start.year, end.year + 1):
        for q in (1, 2, 3, 4):
            if start <= quarter_end(f"{year}Q{q}") <= end:
                out.append(f"{year}Q{q}")
    return out


def cached_periods() -> list[str]:
    """Every period already on disk, oldest first."""
    if not RAW_CALL.exists():
        return []
    found = []
    for p in RAW_CALL.glob("call_*.zip"):
        if p.stat().st_size >= MIN_ZIP_BYTES:
            found.append(p.stem.removeprefix("call_"))
    return sorted(found)
