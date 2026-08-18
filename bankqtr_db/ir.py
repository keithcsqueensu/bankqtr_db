"""Investor-relations supplement discovery and download.

Why this module exists
----------------------
Office CRE exposure, leveraged lending, small-business balances and the
criticized/classified ladder are disclosed in quarterly *investor supplements*
and earnings presentations.  A large part of that never reaches 10-K/10-Q XBRL
-- see the coverage table in the README -- so the only way to fill those
columns is to go and read the supplement.

Getting the documents is the hard part, and it is hard in an unglamorous way:
there is no standard for how a bank publishes them.  Three access patterns
cover the universe, and the registry below records which one applies to whom.

``q4``
    Most large-cap IR sites are hosted by Q4 Inc.  Their pages are rendered
    client-side -- fetching the HTML returns zero document links -- but the
    widget behind them reads a stable JSON feed at
    ``/feed/FinancialReport.svc/GetFinancialReportList``, which returns every
    quarter's document set going back a decade with titles and types attached.
    That feed is the cleanest source in this module.

``page``
    A handful of banks serve a plain server-rendered index of quarterly PDFs
    (Bank of America, Wells Fargo, Truist, Ally, BNY).  Those are scraped with
    a link pattern.

``edgar8k``
    The rest either render the document list in JavaScript with no feed, sit
    behind a bot filter that returns 403 to a scripted client, or use opaque
    per-document UUID paths that cannot be enumerated.  For those the *same*
    supplement is recoverable from EDGAR: a bank furnishes its quarterly
    supplement and earnings deck as exhibits to the Item 2.02 earnings 8-K.
    ``exhibit_index`` reads the filing index page, which carries the exhibit
    type and description ("EX-99.2  QUARTERLY FINANCIAL SUPPLEMENT"), so the
    supplement can be told apart from a debt-issuance exhibit filed the same
    week.

The 8-K route is a deliberate fallback, not a shortcut: it is the identical
document, it reuses the rate limiter and cache in :mod:`edgar`, and it is
frequently HTML where the IR site serves a PDF -- which extracts far more
reliably.  :func:`discovery_report` prints which route served each bank so the
provenance of a run is never implicit.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import httpx

from . import edgar, filings
from .config import DATA, Bank, universe

log = logging.getLogger(__name__)

IR_DIR = DATA / "ir"

# IR hosts are ordinary marketing sites, not an API with a published fair-use
# limit.  Stay conservative and identify the client.
IR_REQUESTS_PER_SECOND = 2.0
IR_HEADERS = {
    "User-Agent": ("bankqtr-db/0.1 (research; peer benchmarking) python-httpx"),
    "Accept": "text/html,application/xhtml+xml,application/json,*/*",
    "Accept-Encoding": "gzip, deflate",
}

# Documents worth downloading, most useful first.  The supplement carries the
# statistical schedules; the presentation carries the office-CRE and
# leveraged-lending slides that never appear anywhere else.
DOC_PRIORITY = ("supplement", "presentation", "release", "credit_metrics")


# --------------------------------------------------------------------------
# Quarter handling
# --------------------------------------------------------------------------

QUARTER_ENDS = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}

# Bounds for expanding a two-digit year in a filename such as "2Q26".
MIN_YEAR, MAX_YEAR = 2000, 2099


def quarter_end(year: int, q: int) -> dt.date:
    month, day = QUARTER_ENDS[q]
    return dt.date(year, month, day)


def quarter_label(year: int, q: int) -> str:
    return f"{year}Q{q}"


def previous_quarter_end(as_of: dt.date) -> dt.date:
    """The most recent calendar quarter end strictly before ``as_of``.

    An earnings 8-K is filed a few weeks after the quarter it reports, so the
    quarter end preceding the filing date is the reporting period.
    """
    q = (as_of.month - 1) // 3 + 1
    start = dt.date(as_of.year, 3 * (q - 1) + 1, 1)
    if as_of >= start:
        prev_q, prev_year = (q - 1, as_of.year) if q > 1 else (4, as_of.year - 1)
    else:  # pragma: no cover - defensive
        prev_q, prev_year = q, as_of.year
    return quarter_end(prev_year, prev_q)


# Document names separate words with underscores and '+' as often as spaces
# ("The+Supplemental+Information_2Q26_ADA.pdf"), and '_' is a regex word
# character, so ``\b`` does not fire around "_2Q26_".  Normalising every
# separator to a space up front is what makes one pattern set work across
# filenames, URL paths and human-written document titles alike.
_SEPARATORS = re.compile(r"[+_\-/\\.]+")


def normalize(text: str) -> str:
    return _SEPARATORS.sub(" ", unquote(text or "")).strip()


# "2Q26", "Q2 2026", "2Q 2026", "second quarter 2026", "2026 Q2", "4Q2020"
#
# ``\b`` is the wrong boundary for these.  A quarter tag is routinely butted
# straight against the next word -- "2Q26EarningsSupplement.pdf" is how US
# Bancorp names every one of its supplements -- and ``\b`` does not fire
# between "26" and "E".  Only the digit side needs guarding, so the boundaries
# are written as explicit lookarounds.
_Q_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?<!\d)([1-4])q ?((?:19|20)\d{2})(?!\d)", re.IGNORECASE), "qy"),
    (re.compile(r"(?<!\w)q([1-4]) ?((?:19|20)\d{2})(?!\d)", re.IGNORECASE), "qy"),
    (re.compile(r"(?<!\d)((?:19|20)\d{2}) ?q([1-4])(?!\d)", re.IGNORECASE), "yq"),
    (re.compile(r"(?<!\d)((?:19|20)\d{2}) ?([1-4])q(?!\w)", re.IGNORECASE), "yq"),
    (re.compile(r"(?<!\d)([1-4])q(\d{2})(?!\d)", re.IGNORECASE), "qy2"),
)

_WORD_QUARTER = re.compile(
    r"\b(first|second|third|fourth) quarter\b.{0,20}?\b((?:19|20)\d{2})\b",
    re.IGNORECASE | re.DOTALL,
)
_WORD_QUARTER_REV = re.compile(
    r"\b((?:19|20)\d{2})\b.{0,20}?\b(first|second|third|fourth) quarter\b",
    re.IGNORECASE | re.DOTALL,
)
_WORD_TO_Q = {"first": 1, "second": 2, "third": 3, "fourth": 4}

# A bare 8-digit date in a filename ("hban20260630_8kex992.htm") pins the
# period exactly, which beats inferring it from the filing date.
_YMD = re.compile(r"(?<!\d)(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)")


def parse_quarter(text: str) -> tuple[int, int] | None:
    """Recover (year, quarter) from a document title, filename or URL."""
    if not text:
        return None
    s = normalize(text)

    m = _WORD_QUARTER.search(s) or _WORD_QUARTER_REV.search(s)
    if m:
        a, b = m.group(1), m.group(2)
        word, year = (a, b) if a.lower() in _WORD_TO_Q else (b, a)
        return int(year), _WORD_TO_Q[word.lower()]

    for pattern, order in _Q_PATTERNS:
        m = pattern.search(s)
        if not m:
            continue
        a, b = m.group(1), m.group(2)
        if order == "qy":
            return int(b), int(a)
        if order == "yq":
            return int(a), int(b)
        year = 2000 + int(b)
        if MIN_YEAR <= year <= MAX_YEAR:
            return year, int(a)

    m = _YMD.search(s)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        if month in (3, 6, 9, 12):
            return year, month // 3
    return None


# --------------------------------------------------------------------------
# Document classification
# --------------------------------------------------------------------------

# Ordered: the first pattern that matches wins, so the more specific document
# names are tested before the generic "earnings" catch.
_DOC_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    # Exhibit filenames run the words together: "ex991june2026creditmetrics".
    ("credit_metrics", re.compile(r"credit ?metrics|monthly charge", re.IGNORECASE)),
    (
        "supplement",
        re.compile(
            r"supplement|quarterly performance summary|\bqps\b"
            r"|financial highlights|statistical (supplement|summary)"
            r"|financial (schedules|tables|data)",
            re.IGNORECASE,
        ),
    ),
    (
        "presentation",
        re.compile(
            r"presentation|slides|\bdeck\b|investor day"
            r"|conference call|earnings call",
            re.IGNORECASE,
        ),
    ),
    (
        "release",
        re.compile(
            r"earnings release|press release|news release"
            r"|reports? (first|second|third|fourth) quarter"
            r"|\bresults\b|\bearnings\b",
            re.IGNORECASE,
        ),
    ),
)

# Things that look like a quarterly document but are not one.
_DOC_EXCLUDE = re.compile(
    r"proxy|annual report|\b10 ?[kq]\b|prospectus|indenture|code of conduct"
    r"|privacy|sustainab|\besg\b|citizenship|charter|by ?laws|underwriting"
    r"|dividend announce|\bccar\b|dfast disclosure|stress test"
    r"|pillar 3|resolution plan|tax form|\bw ?9\b|transcript"
    r"|\baudio\b|\bxbrl\b",
    re.IGNORECASE,
)


def classify_document(text: str) -> str | None:
    """Map a document title/filename to one of :data:`DOC_PRIORITY`."""
    if not text:
        return None
    s = normalize(text)
    if _DOC_EXCLUDE.search(s):
        return None
    for kind, pattern in _DOC_RULES:
        if pattern.search(s):
            return kind
    return None


# --------------------------------------------------------------------------
# The document record
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class IRDoc:
    bank: str
    ticker: str
    cik: str
    year: int
    quarter: int
    doc_type: str
    url: str
    origin: str  # q4 | page | edgar8k
    title: str = ""
    accn: str | None = None

    @property
    def label(self) -> str:
        return quarter_label(self.year, self.quarter)

    @property
    def period(self) -> dt.date:
        return quarter_end(self.year, self.quarter)

    @property
    def ext(self) -> str:
        path = urlparse(self.url).path.lower()
        for candidate in (".pdf", ".xlsx", ".xls", ".htm", ".html"):
            if path.endswith(candidate):
                return candidate.lstrip(".")
        return "htm"

    @property
    def path(self) -> Path:
        """Stable on-disk location: ``data/ir/<TICKER>/<TICKER>_<Q>_<type>.<ext>``."""
        return (
            IR_DIR
            / self.ticker
            / f"{self.ticker}_{self.label}_{self.doc_type}.{self.ext}"
        )


# --------------------------------------------------------------------------
# HTTP for non-EDGAR hosts
# --------------------------------------------------------------------------


class _Limiter:
    def __init__(self, rps: float) -> None:
        self._min = 1.0 / rps
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            gap = self._min - (time.monotonic() - self._last)
            if gap > 0:
                time.sleep(gap)
            self._last = time.monotonic()


_limiter = _Limiter(IR_REQUESTS_PER_SECOND)
_client: httpx.Client | None = None


def client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(
            headers=IR_HEADERS,
            timeout=httpx.Timeout(90.0, connect=20.0),
            follow_redirects=True,
        )
    return _client


def fetch(url: str, *, retries: int = 3) -> httpx.Response:
    """GET an IR host with rate limiting and backoff."""
    delay = 1.5
    last: Exception | None = None
    for attempt in range(retries):
        _limiter.wait()
        try:
            resp = client().get(url)
        except httpx.HTTPError as exc:
            last = exc
            log.debug("IR GET %s failed (%s) try %d", url, exc, attempt + 1)
        else:
            if resp.status_code == 200:
                return resp
            if resp.status_code in (403, 404):
                resp.raise_for_status()
            last = httpx.HTTPStatusError(
                f"{resp.status_code} for {url}", request=resp.request, response=resp
            )
        time.sleep(delay)
        delay *= 2
    assert last is not None
    raise last


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class IRSource:
    """How to reach one bank's quarterly documents."""

    ticker: str
    provider: str  # q4 | page | edgar8k
    # q4: IR hostname.  page: one or more index URLs to scrape.
    host: str = ""
    pages: tuple[str, ...] = field(default=())
    note: str = ""

    @property
    def providers(self) -> tuple[str, ...]:
        """Routes to try, best first.

        ``edgar8k`` is appended to every entry rather than used only as a last
        resort.  An IR index is usually pruned -- Truist's page lists earnings
        decks going back six years but no Quarterly Performance Summary at all,
        and Bank of America's drops older quarters -- so the 8-K exhibits fill
        the (quarter, document type) slots the site does not serve.  Where both
        routes offer the same slot the IR-site copy wins.
        """
        if self.provider == "edgar8k":
            return ("edgar8k",)
        return (self.provider, "edgar8k")


# Q4 hosts were confirmed by probing the feed endpoint directly; a wrong host
# does not silently degrade, it raises, and ``discovery_report`` shows it.
SOURCES: tuple[IRSource, ...] = (
    # ---- Q4-hosted IR sites (JSON feed) ----------------------------------
    IRSource("USB", "q4", host="ir.usbank.com"),
    IRSource("RF", "q4", host="ir.regions.com"),
    IRSource("FITB", "q4", host="ir.53.com"),
    IRSource("KEY", "q4", host="investor.key.com"),
    IRSource("FCNCA", "q4", host="ir.firstcitizens.com"),
    IRSource("AXP", "q4", host="ir.americanexpress.com"),
    IRSource("AMP", "q4", host="ir.ameriprise.com"),
    IRSource("STT", "q4", host="investors.statestreet.com"),
    # ---- server-rendered index pages --------------------------------------
    IRSource(
        "BAC", "page", pages=("https://investor.bankofamerica.com/quarterly-earnings",)
    ),
    IRSource(
        "WFC",
        "page",
        pages=(
            "https://www.wellsfargo.com/about/investor-relations/quarterly-earnings/",
        ),
    ),
    IRSource("TFC", "page", pages=("https://ir.truist.com/earnings",)),
    IRSource("ALLY", "page", pages=("https://www.ally.com/about/investor/",)),
    IRSource(
        "BNY",
        "page",
        pages=(
            "https://www.bny.com/corporate/global/en/investor-relations/overview.html",
        ),
    ),
    # ---- IR site not machine-reachable; same documents via the 8-K --------
    IRSource("JPM", "edgar8k", note="IR documents live under opaque UUID paths"),
    IRSource("C", "edgar8k", note="client-rendered document list, no feed"),
    IRSource("PNC", "edgar8k", note="IR document list rendered client-side"),
    IRSource("CFG", "edgar8k", note="IR host returns 403 to scripted clients"),
    IRSource("HBAN", "edgar8k", note="client-rendered document list, no feed"),
    IRSource("MTB", "edgar8k", note="IR host times out on scripted requests"),
    IRSource("ZION", "edgar8k", note="client-rendered document list, no feed"),
    IRSource("COF", "edgar8k", note="IR host times out on scripted requests"),
    IRSource("SYF", "edgar8k", note="client-rendered document list, no feed"),
    IRSource("DFS", "edgar8k", note="IR host retired after the Capital One deal"),
    IRSource("SHUSA", "edgar8k", note="no public quarterly document index"),
    IRSource("HSBCUSA", "edgar8k", note="no US-entity IR document index"),
    IRSource("MUFGA", "edgar8k", note="filer inactive since 2021"),
    IRSource("GS", "edgar8k", note="IR host returns 403 to scripted clients"),
    IRSource("MS", "edgar8k", note="IR host refuses scripted connections"),
    IRSource("NTRS", "edgar8k", note="IR host refuses scripted connections"),
    IRSource("SCHW", "edgar8k", note="IR host returns 403 to scripted clients"),
    IRSource("RJF", "edgar8k", note="IR host times out on scripted requests"),
)

BY_TICKER: dict[str, IRSource] = {s.ticker: s for s in SOURCES}


def source_for(bank: Bank) -> IRSource:
    return BY_TICKER.get(
        bank.ticker, IRSource(bank.ticker, "edgar8k", note="no registry entry")
    )


# --------------------------------------------------------------------------
# Provider: Q4 financial-report feed
# --------------------------------------------------------------------------

Q4_FEED = (
    "https://{host}/feed/FinancialReport.svc/GetFinancialReportList"
    "?LanguageId=1&year=-1&pageSize=500"
)

_SUBTYPE_TO_Q = {
    "first quarter": 1,
    "second quarter": 2,
    "third quarter": 3,
    "fourth quarter": 4,
}


def _q4_docs(bank: Bank, source: IRSource, since: dt.date) -> list[IRDoc]:
    payload = fetch(Q4_FEED.format(host=source.host)).json()
    reports = payload.get("GetFinancialReportListResult") or []
    out: list[IRDoc] = []
    for report in reports:
        subtype = (report.get("ReportSubType") or "").strip().lower()
        year = report.get("ReportYear")
        quarter = _SUBTYPE_TO_Q.get(subtype)
        if quarter is None or not year:
            # Annual reports and one-off supplemental filings; the quarter, if
            # any, has to come from the title.
            parsed = parse_quarter(report.get("ReportTitle") or "")
            if parsed is None:
                continue
            year, quarter = parsed
        if quarter_end(int(year), int(quarter)) < since:
            continue

        for doc in report.get("Documents") or []:
            if (doc.get("DocumentType") or "") != "File":
                continue
            url = doc.get("DocumentPath") or ""
            if not url:
                continue
            title = doc.get("DocumentTitle") or ""
            category = doc.get("DocumentCategory") or ""
            kind = (
                classify_document(title)
                or classify_document(category)
                or classify_document(Path(urlparse(url).path).name)
            )
            if kind is None:
                continue
            out.append(
                IRDoc(
                    bank=bank.name,
                    ticker=bank.ticker,
                    cik=bank.cik,
                    year=int(year),
                    quarter=int(quarter),
                    doc_type=kind,
                    url=url,
                    origin="q4",
                    title=title or category,
                )
            )
    return out


# --------------------------------------------------------------------------
# Provider: server-rendered index page
# --------------------------------------------------------------------------

_HREF = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)
_DOC_EXT = re.compile(r"\.(pdf|xlsx|xls)(\?|#|$)", re.IGNORECASE)


def _page_docs(bank: Bank, source: IRSource, since: dt.date) -> list[IRDoc]:
    out: list[IRDoc] = []
    seen: set[str] = set()
    for page in source.pages:
        try:
            html = fetch(page).text
        except Exception as exc:  # noqa: BLE001 - a dead index is a reportable gap
            log.warning("IR page fetch failed %s %s: %s", bank.ticker, page, exc)
            continue
        for href in _HREF.findall(html):
            if not _DOC_EXT.search(href):
                continue
            url = urljoin(page, href)
            if url in seen:
                continue
            seen.add(url)
            name = unquote(Path(urlparse(url).path).name)
            kind = classify_document(name)
            if kind is None:
                continue
            parsed = parse_quarter(name) or parse_quarter(url)
            if parsed is None:
                continue
            year, quarter = parsed
            if quarter_end(year, quarter) < since:
                continue
            out.append(
                IRDoc(
                    bank=bank.name,
                    ticker=bank.ticker,
                    cik=bank.cik,
                    year=year,
                    quarter=quarter,
                    doc_type=kind,
                    url=url,
                    origin="page",
                    title=name,
                )
            )
    return out


# --------------------------------------------------------------------------
# Provider: earnings 8-K exhibits
# --------------------------------------------------------------------------

# The filing index page lists each attachment with its exhibit type and the
# description the filer gave it, which is what makes the supplement separable
# from an unrelated exhibit filed the same week.
_INDEX_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_CELL = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_DOC_HREF = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", _TAG.sub(" ", html)).strip()


def _index_cache_path(accn: str) -> Path:
    return IR_DIR / "_index_cache" / f"{accn}.json.gz"


def exhibit_index(folder_url: str, accn: str) -> list[tuple[str, str, str]]:
    """(document name, exhibit type, description) for one filing.

    Read from the filing index page rather than ``index.json`` because only
    the index page carries the type and description columns.

    Cached on disk: discovery walks every 8-K a bank filed in the window --
    several thousand across the universe -- and only a handful are earnings
    filings, so re-running the scraper must not re-fetch them all.
    """
    cache = _index_cache_path(accn)
    if cache.exists():
        try:
            with gzip.open(cache, "rt", encoding="utf-8") as fh:
                return [tuple(row) for row in json.load(fh)]  # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001 - a corrupt entry just re-fetches
            log.debug("bad index cache %s: %s", accn, exc)

    url = f"{folder_url}/{accn}-index.htm"
    try:
        html = edgar.get(url).text
    except Exception as exc:  # noqa: BLE001
        log.debug("exhibit index failed %s: %s", accn, exc)
        return []

    rows: list[tuple[str, str, str]] = []
    for row_html in _INDEX_ROW.findall(html):
        cells = _CELL.findall(row_html)
        if len(cells) < 4:
            continue
        href = _DOC_HREF.search(row_html)
        if not href:
            continue
        name = Path(urlparse(href.group(1)).path).name
        texts = [_text(c) for c in cells]
        # Layout is: Seq | Description | Document | Type | Size
        description = texts[1] if len(texts) > 1 else ""
        ex_type = ""
        for value in texts:
            if re.fullmatch(r"EX-\d+(\.\d+)*", value, re.IGNORECASE):
                ex_type = value
                break
        rows.append((name, ex_type, description))

    cache.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(cache, gzip.compress(json.dumps(rows).encode("utf-8")))
    return rows


# An earnings 8-K lands a couple of weeks after the quarter it reports and
# never months later, so this window separates it from the debt-issuance and
# governance 8-Ks that also carry EX-99 exhibits.
EARNINGS_WINDOW_DAYS = (8, 60)

# Large banks report roughly a fortnight to three weeks after quarter end.
# Used only to rank competing 8-Ks inside one window, never to exclude one.
TYPICAL_EARNINGS_LAG_DAYS = 18

# Phrases from the first page of each document type.  Modern EDGAR index pages
# put only the exhibit number in the description column ("EX-99.2"), and
# filenames like ``ex992-qpsx2q26.htm`` say nothing either, so the document has
# to be classified from its own opening text.
# Order is by how distinctive the signal is, not by document importance.  Every
# one of these documents mentions the others' vocabulary somewhere, so the
# unmistakable openings have to be tested before the generic ones:
#
# * A press release opens with a dateline and *also* says "financial
#   highlights" further down.  Testing the supplement patterns first labelled
#   every Truist press release a supplement.
# * A slide deck says "financial highlights" on its first slide.  Same trap:
#   JPMorgan's earnings deck claimed the supplement slot for every quarter,
#   and its actual financial supplement was never read.
#
# Hence two presentation rules -- the self-identifying titles first, the
# weaker hints after the supplement patterns.
_PRESENTATION_TITLE = re.compile(
    r"earnings presentation|investor presentation|presentation slides"
    r"|company update presentation|earnings deck|investor day",
    re.IGNORECASE,
)

_CONTENT_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "release",
        re.compile(
            r"for immediate release|news release|press release"
            r"|reports? (first|second|third|fourth).quarter (\w+ ){0,3}results"
            r"|media contact|investor contact.{0,200}media",
            re.IGNORECASE,
        ),
    ),
    ("presentation", _PRESENTATION_TITLE),
    (
        "supplement",
        re.compile(
            r"financial supplement|quarterly performance summary"
            r"|supplemental (information|financial)|selected financial data"
            r"|quarterly financial (data|schedules|supplement)"
            r"|selected quarterly (financial )?data"
            r"|statistical (supplement|information)|financial highlights",
            re.IGNORECASE,
        ),
    ),
    (
        "presentation",
        re.compile(
            r"conference call presentation|earnings (call|conference)",
            re.IGNORECASE,
        ),
    ),
)

# Exhibit ordering is a weak but real prior when the text is inconclusive:
# banks overwhelmingly file the release as .1, the supplement as .2 and the
# slide deck as .3.
_EXHIBIT_PRIOR = {
    "EX-99.1": "release",
    "EX-99.2": "supplement",
    "EX-99.3": "presentation",
}

_STRIP_TAGS = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)


def _document_head(content: bytes, *, chars: int = 6000) -> str:
    """Readable opening text of an exhibit, for content classification."""
    if content[:5] == b"%PDF-":
        try:
            from io import BytesIO

            import pdfplumber  # imported lazily; only PDF exhibits need it

            with pdfplumber.open(BytesIO(content)) as pdf:
                pages = pdf.pages[:2]
                return " ".join((p.extract_text() or "") for p in pages)[:chars]
        except Exception as exc:  # noqa: BLE001
            log.debug("pdf head failed: %s", exc)
            return ""
    text = content[: chars * 12].decode("utf-8", errors="replace")
    text = _STRIP_TAGS.sub(" ", text)
    return re.sub(r"\s+", " ", _TAG.sub(" ", text))[:chars]


# Below this much readable text, an exhibit is a deck of slide *images* with
# no extractable content -- JPMorgan files its investor-day presentation that
# way.  Such a document can never populate a column, so it must not be allowed
# to claim a quarter's slot and shut out the real supplement.
MIN_READABLE_CHARS = 400


def classify_content(content: bytes) -> str | None:
    """Document type read out of the document's own opening text."""
    head = _document_head(content)
    if len(head.strip()) < MIN_READABLE_CHARS:
        return None
    for kind, pattern in _CONTENT_RULES:
        if pattern.search(head):
            return kind
    return None


# Bumped whenever the content classifier changes, so cached verdicts made by
# an older rule set are re-derived instead of being trusted forever.
SNIFF_CACHE_VERSION = 3


def _sniff_cache_path(cik: str) -> Path:
    return IR_DIR / "_index_cache" / f"sniff_v{SNIFF_CACHE_VERSION}_{cik}.json"


def _load_sniffs(cik: str) -> dict[str, str]:
    path = _sniff_cache_path(cik)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.debug("bad sniff cache %s: %s", cik, exc)
        return {}


def _save_sniffs(cik: str, sniffs: dict[str, str]) -> None:
    path = _sniff_cache_path(cik)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(path, json.dumps(sniffs, indent=0, sort_keys=True).encode("utf-8"))


def _edgar8k_docs(bank: Bank, source: IRSource, since: dt.date) -> list[IRDoc]:
    """Quarterly documents recovered from earnings-8-K EX-99 exhibits.

    Exhibits whose type cannot be read off the filename are fetched once and
    classified from their own text; the verdict is cached per CIK so a re-run
    costs nothing.  The fetched bytes are written straight to the document's
    final location, so the later download step finds it already present.
    """
    del source  # the 8-K route needs no per-bank configuration
    lookback = since - dt.timedelta(days=EARNINGS_WINDOW_DAYS[1])
    sniffs = _load_sniffs(bank.cik)
    dirty = False

    # More than one 8-K can fall inside a quarter's earnings window, and the
    # first slot-filler wins, so the order they are visited in decides which
    # document the panel reads.  Newest-first is the wrong order: JPMorgan's
    # February investor-day deck lands in the same window as its January
    # earnings release and claimed the 2025Q4 supplement slot with a file that
    # is nothing but slide images.  Visiting them in order of how close the
    # filing sits to a typical earnings lag puts the real earnings 8-K first.
    scheduled: list[tuple[int, dt.date, filings.Filing]] = []
    for filing in filings.list_filings(bank, forms=("8-K",), since=lookback):
        period = previous_quarter_end(filing.filing_date)
        lag = (filing.filing_date - period).days
        if not EARNINGS_WINDOW_DAYS[0] <= lag <= EARNINGS_WINDOW_DAYS[1]:
            continue
        if period < since:
            continue
        scheduled.append((abs(lag - TYPICAL_EARNINGS_LAG_DAYS), period, filing))
    scheduled.sort(key=lambda item: (-item[1].toordinal(), item[0]))

    by_key: dict[tuple[int, int, str], IRDoc] = {}
    for _, period, filing in scheduled:
        rows = [
            (name, ex_type)
            for name, ex_type, _ in exhibit_index(filing.folder_url, filing.accn)
            if ex_type.upper().startswith("EX-99")
            and name.lower().endswith((".htm", ".html", ".pdf", ".xlsx", ".xls"))
        ]
        if not rows:
            continue

        year, quarter = period.year, (period.month - 1) // 3 + 1
        picked: list[tuple[IRDoc, bytes | None]] = []
        for name, ex_type in rows:
            url = f"{filing.folder_url}/{name}"
            kind = classify_document(name)
            content: bytes | None = None
            if kind is None:
                cached = sniffs.get(f"{filing.accn}/{name}")
                if cached is not None:
                    kind = cached or None
                else:
                    try:
                        content = edgar.get(url).content
                    except Exception as exc:  # noqa: BLE001
                        log.debug("exhibit fetch failed %s: %s", url, exc)
                        content = None
                    kind = classify_content(content) if content else None
                    kind = kind or _EXHIBIT_PRIOR.get(ex_type.upper())
                    sniffs[f"{filing.accn}/{name}"] = kind or ""
                    dirty = True
            if kind is None:
                continue
            picked.append(
                (
                    IRDoc(
                        bank=bank.name,
                        ticker=bank.ticker,
                        cik=bank.cik,
                        year=year,
                        quarter=quarter,
                        doc_type=kind,
                        url=url,
                        origin="edgar8k",
                        title=f"{ex_type} {name}",
                        accn=filing.accn,
                    ),
                    content,
                )
            )

        for doc, content in picked:
            key = (doc.year, doc.quarter, doc.doc_type)
            if key in by_key:
                continue
            by_key[key] = doc
            # Already in hand from the classification fetch -- keep it rather
            # than asking EDGAR for the same bytes again at download time.
            if content and not doc.path.exists():
                doc.path.parent.mkdir(parents=True, exist_ok=True)
                _write_atomic(doc.path, content)

    if dirty:
        _save_sniffs(bank.cik, sniffs)
    return list(by_key.values())


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

_PROVIDERS = {"q4": _q4_docs, "page": _page_docs, "edgar8k": _edgar8k_docs}


def discover(
    bank: Bank, *, since: dt.date, source: IRSource | None = None
) -> list[IRDoc]:
    """Every quarterly document for a bank at or after ``since``.

    Routes are tried in order and merged into one slot per (quarter, document
    type).  The first route to fill a slot keeps it, so the IR site wins where
    it publishes and the 8-K exhibits fill in the rest.
    """
    src = source or source_for(bank)
    ranked: dict[tuple[int, int, str], IRDoc] = {}

    for name in src.providers:
        provider = _PROVIDERS.get(name)
        if provider is None:
            log.warning("unknown IR provider %r for %s", name, bank.ticker)
            continue
        try:
            docs = provider(bank, src, since)
        except Exception as exc:  # noqa: BLE001 - one dead site must not stop the run
            log.warning("IR discovery failed %s (%s): %s", bank.ticker, name, exc)
            continue
        for doc in docs:
            key = (doc.year, doc.quarter, doc.doc_type)
            current = ranked.get(key)
            # Within one route, prefer the format that extracts more reliably;
            # across routes, the earlier (more authoritative) route keeps the
            # slot it already filled.
            better_format = (
                current is not None
                and current.origin == doc.origin
                and _format_rank(doc) > _format_rank(current)
            )
            if current is None or better_format:
                ranked[key] = doc

    return sorted(
        ranked.values(),
        key=lambda d: (-d.year, -d.quarter, DOC_PRIORITY.index(d.doc_type)),
    )


def _format_rank(doc: IRDoc) -> int:
    """HTML and Excel extract more reliably than PDF, so prefer them."""
    return {"htm": 3, "html": 3, "xlsx": 2, "xls": 2, "pdf": 1}.get(doc.ext, 0)


# The on-disk name a document was stored under, read back:
# ``USB_2026Q2_supplement.pdf``.
_CACHED_NAME = re.compile(
    r"^(?P<ticker>[A-Z]+)_(?P<year>20\d{2})Q(?P<quarter>[1-4])_(?P<kind>[a-z_]+)"
    r"\.(?P<ext>pdf|htm|html|xlsx|xls)$"
)


def discover_cached(bank: Bank, *, since: dt.date) -> list[IRDoc]:
    """Documents already downloaded under ``data/ir/``, without any network I/O.

    ``build_panel`` uses this rather than :func:`discover` so that a build is
    reproducible and offline: fetching is ``scripts/fetch_ir.py``'s job, and a
    rebuild reads exactly the corpus that script left behind.  The original
    URL is not recoverable from the filename, so ``url`` is the local path --
    every field the extractor needs (bank, quarter, document type) is.
    """
    folder = IR_DIR / bank.ticker
    if not folder.is_dir():
        return []

    # A quarter can end up with the same document in two formats -- the Q4
    # feed's PDF and the 8-K's HTML of the identical supplement -- so keep one
    # per slot rather than parsing both and letting an arbitrary one win.
    best: dict[tuple[int, int, str], IRDoc] = {}
    for path in sorted(folder.iterdir()):
        match = _CACHED_NAME.match(path.name)
        if not match or path.stat().st_size == 0:
            continue
        year, quarter = int(match.group("year")), int(match.group("quarter"))
        if quarter_end(year, quarter) < since:
            continue
        doc = IRDoc(
            bank=bank.name,
            ticker=bank.ticker,
            cik=bank.cik,
            year=year,
            quarter=quarter,
            doc_type=match.group("kind"),
            url=path.as_uri(),
            origin="cached",
            title=path.name,
        )
        key = (year, quarter, doc.doc_type)
        current = best.get(key)
        if current is None or _format_rank(doc) > _format_rank(current):
            best[key] = doc
    return sorted(best.values(), key=lambda d: (-d.year, -d.quarter, d.doc_type))


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------


def download(doc: IRDoc, *, refresh: bool = False) -> Path | None:
    """Fetch one document into ``data/ir/`` and return its path."""
    path = doc.path
    if path.exists() and path.stat().st_size > 0 and not refresh:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)

    getter = edgar.get if doc.origin == "edgar8k" else fetch
    try:
        content = getter(doc.url).content
    except Exception as exc:  # noqa: BLE001
        log.warning("IR download failed %s %s: %s", doc.ticker, doc.url, exc)
        return None
    if not content:
        return None

    return _write_atomic(path, content)


def _write_atomic(path: Path, content: bytes) -> Path | None:
    """Write via a per-process temp file, tolerating a locked destination.

    ``os.replace`` is atomic on POSIX but raises ``PermissionError`` on Windows
    when anything else holds the destination open -- including a second copy of
    this scraper.  The temp name carries the process id so two runs cannot
    collide on it, and a locked destination degrades to a plain overwrite
    rather than aborting the whole fetch.
    """
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(content)
        tmp.replace(path)
        return path
    except OSError as exc:
        log.debug("atomic write failed %s (%s); writing in place", path.name, exc)
        try:
            path.write_bytes(content)
            return path
        except OSError as exc2:
            log.warning("could not write %s: %s", path, exc2)
            return None
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                log.debug("could not remove temp file %s", tmp)


def download_all(
    docs: list[IRDoc],
    *,
    refresh: bool = False,
    doc_types: tuple[str, ...] = DOC_PRIORITY,
) -> list[tuple[IRDoc, Path]]:
    out: list[tuple[IRDoc, Path]] = []
    for doc in docs:
        if doc.doc_type not in doc_types:
            continue
        path = download(doc, refresh=refresh)
        if path is not None:
            out.append((doc, path))
    return out


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def discovery_report(
    results: dict[str, list[IRDoc]], *, banks: tuple[Bank, ...] | None = None
) -> str:
    """Human-readable summary of which route served which bank."""
    lines = [
        f"{'bank':7s} {'route':9s} {'docs':>5s} {'quarters':>9s}  {'span':21s} note"
    ]
    for bank in banks or universe():
        docs = results.get(bank.ticker, [])
        src = source_for(bank)
        quarters = sorted({(d.year, d.quarter) for d in docs})
        span = (
            f"{quarter_label(*quarters[0])}..{quarter_label(*quarters[-1])}"
            if quarters
            else "-"
        )
        lines.append(
            f"{bank.ticker:7s} {src.provider:9s} {len(docs):5d} {len(quarters):9d}  "
            f"{span:21s} {src.note}"
        )
    return "\n".join(lines)


def manifest_rows(pairs: list[tuple[IRDoc, Path]]) -> list[dict[str, object]]:
    return [
        {
            "bank": doc.bank,
            "ticker": doc.ticker,
            "cik": doc.cik,
            "quarter": doc.label,
            "period": doc.period,
            "doc_type": doc.doc_type,
            "origin": doc.origin,
            "ext": doc.ext,
            "title": doc.title[:160],
            "url": doc.url,
            "accn": doc.accn,
            "path": str(path.relative_to(DATA)),
            "bytes": path.stat().st_size if path.exists() else 0,
        }
        for doc, path in pairs
    ]
