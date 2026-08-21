"""FFIEC NIC structure data: the entity graph, succession, and the bridge to EDGAR.

Four bulk files at ``ffiec.gov/npw`` describe who exists, who owns whom, and
who became whom: ``attributes-active``, ``attributes-closed``,
``relationships`` and ``transformations``.  Together they answer the three
questions this package asks of NIC.

**Which entity became which.**  The transformations table records every
event in which one entity's assets passed to another: a merger, a failure
resolved by the FDIC, a charter re-issued under a new RSSD.  It is what lets
the panel follow Wachovia into Wells Fargo, National City into PNC and
Washington Mutual into JPMorgan Chase -- see :mod:`callrpt_db.lineage`.  The
codes are read off the NIC data dictionary, not guessed: ``1`` is a merger or
purchase-and-assumption, ``9`` a charter retained under a new RSSD with 95%
or more of the assets, and ``50`` a failure with government assistance.
Codes ``5`` and ``7`` are partial -- a split or a sale of 40-94% of the
assets, after which the predecessor continues to exist -- and are not
succession in any sense a panel can sum.

**Which charter belongs to which parent.**  A Call Report is filed by a bank,
not by a holding company, so rolling up to something comparable with a 10-K
means walking down from the top-tier parent to its depository subsidiaries.
The walk is over ``CTRL_IND = 1`` (control) relationships and is transitive:
Truist Bank sits under Truist Financial through intermediate holdcos, and a
one-level lookup finds nothing.

**Which charter is not really a subsidiary any more.**  Relationship rows are
dated.  ``DT_END = 99991231`` is an open relationship; anything else closed on
that date, and reading the graph as though it were current puts SunTrust Bank
under Truist in 2014.  :func:`hierarchy` therefore takes the quarter it is
being asked about and honours both ends of the date range -- which is what
makes a 2013-start panel possible at all.

Bridging to EDGAR
-----------------
NIC carries no CIK.  It carries ``ID_TAX``, the EIN, and EDGAR's submissions
payload carries ``ein`` -- so the join is on a federal tax identifier that both
registries collect independently, not on a company name.  That matters more
than it sounds: NIC's legal names are upper-cased, abbreviated and often
punctuated differently from EDGAR's ("JPMORGAN CHASE & CO." against "JPMORGAN
CHASE & CO"), and several holdcos share a name stem with a subsidiary that has
a different EIN.  Name matching is used only for the IHCs, which have no SEC
registration to carry an EIN, and every such match is recorded with its
evidence in ``data/out/rssd_resolution.csv``.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import logging
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import httpx

from .config import FFIEC_HEADERS, NPW_BASE, NPW_FILES, NPW_REFERER, RAW_NIC

log = logging.getLogger(__name__)

OPEN_END = "99991231"


def fetch_structure(*, refresh: bool = False) -> dict[str, Path]:
    """Download the NIC bulk structure files, or return the cached ones.

    ``www.ffiec.gov`` answers a bare scripted client with 403 and wants a
    referer pointing at the download page, so both are set in
    :data:`config.FFIEC_HEADERS` and here rather than left to defaults.
    """
    RAW_NIC.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    todo = {
        key: endpoint
        for key, endpoint in NPW_FILES.items()
        if refresh or not (RAW_NIC / f"{key}.zip").exists()
    }
    for key in NPW_FILES:
        out[key] = RAW_NIC / f"{key}.zip"
    if not todo:
        return out

    headers = dict(FFIEC_HEADERS)
    headers["Referer"] = NPW_REFERER
    with httpx.Client(
        headers=headers,
        timeout=httpx.Timeout(300.0, connect=30.0),
        follow_redirects=True,
    ) as client:
        for key, endpoint in todo.items():
            path = out[key]
            resp = client.get(f"{NPW_BASE}/{endpoint}")
            resp.raise_for_status()
            if resp.content[:2] != b"PK":
                raise RuntimeError(
                    f"NIC returned {resp.headers.get('content-type')} for {key}, "
                    "not a zip; the download endpoints changed"
                )
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(resp.content)
            tmp.replace(path)
            log.info("fetched NIC %s (%.1f MB)", key, len(resp.content) / 1e6)
    return out


def _read_csv(path: Path) -> list[dict[str, str]]:
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        with z.open(name) as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8-sig", errors="replace")
            return list(csv.DictReader(text))


@dataclass(frozen=True)
class Entity:
    rssd: int
    name: str
    entity_type: str
    ein: str | None
    lei: str | None
    fdic_cert: int | None
    is_bhc: bool
    is_ihc: bool
    active: bool
    # ``YYYYMMDD`` strings, as the file carries them; ``None`` where NIC has
    # no date.  ``ended`` is ``None`` for an entity that still exists.
    opened: str | None = None
    ended: str | None = None
    # REASON_TERM_CD: 0 continues, 1 voluntary liquidation, 2 closed after a
    # merger, 3 no longer regulated, 4 failed but continues, 5 failed and
    # ceased to exist.  4 and 5 are the failures.
    reason_term: str = "0"

    @property
    def failed(self) -> bool:
        return self.reason_term in FAILURE_TERM_CODES


# REASON_TERM_CD values that mean the entity failed, per the NIC dictionary.
FAILURE_TERM_CODES = frozenset({"4", "5"})


def _clean(value: str) -> str | None:
    """NIC writes an absent identifier as ``0``, not as an empty field."""
    v = (value or "").strip()
    return None if v in ("", "0") else v


def _stamp(value: str) -> str | None:
    """A ``YYYYMMDD`` field, or ``None`` where NIC wrote ``0`` or the open end."""
    v = _clean(value)
    return None if v is None or v == OPEN_END else v


@lru_cache(maxsize=1)
def entities() -> dict[int, Entity]:
    """Every entity NIC knows, active and closed, keyed on RSSD.

    Closed entities are loaded too, and deliberately do not overwrite an active
    row of the same RSSD: an institution that was closed and whose id was later
    reused would otherwise be read as defunct.
    """
    paths = fetch_structure()
    out: dict[int, Entity] = {}
    for key, active in (("attributes_active", True), ("attributes_closed", False)):
        for row in _read_csv(paths[key]):
            rssd = int(row["#ID_RSSD"])
            if rssd in out:
                continue
            fdic = _clean(row.get("ID_FDIC_CERT", ""))
            out[rssd] = Entity(
                rssd=rssd,
                name=row["NM_LGL"].strip(),
                entity_type=row["ENTITY_TYPE"].strip(),
                ein=_clean(row.get("ID_TAX", "")),
                lei=_clean(row.get("ID_LEI", "")),
                fdic_cert=int(fdic) if fdic and fdic.isdigit() else None,
                is_bhc=row.get("BHC_IND", "0").strip() == "1",
                is_ihc=row.get("IHC_IND", "0").strip() == "1",
                active=active,
                opened=_stamp(row.get("DT_OPEN", "")),
                ended=_stamp(row.get("DT_END", "")),
                reason_term=(row.get("REASON_TERM_CD") or "0").strip() or "0",
            )
    return out


# --------------------------------------------------------------------------
# Succession
# --------------------------------------------------------------------------

# TRNSFM_CD values under which the predecessor's whole balance sheet passed
# to the successor.  See the module docstring for the partial codes left out.
ABSORPTION_CODES = frozenset({"1", "9", "50"})
FAILURE_CODE = "50"


@dataclass(frozen=True)
class Succession:
    """One predecessor -> successor event from the transformations table."""

    predecessor: int
    successor: int
    date: str  # YYYYMMDD
    code: str  # TRNSFM_CD
    # ACCT_METHOD 1: pooling of interests / entities under common control.
    # The survivor's year-to-date income statement is restated as though the
    # two had been combined from the start of the year, which matters for
    # differencing flows -- see ``panel.quarterize``.
    pooled: bool

    @property
    def failure(self) -> bool:
        return self.code == FAILURE_CODE

    @property
    def on(self) -> dt.date:
        return stamp_to_date(self.date)


def stamp_to_date(stamp: str) -> dt.date:
    """``YYYYMMDD`` -> date.  Raises ``ValueError`` on anything else."""
    if len(stamp) != 8 or not stamp.isdigit():
        raise ValueError(f"not a YYYYMMDD stamp: {stamp!r}")
    return dt.date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:]))


@lru_cache(maxsize=1)
def successions() -> tuple[Succession, ...]:
    """Every whole-entity succession NIC records, oldest first.

    Only the absorption codes are kept.  A split (``5``) or a sale of assets
    (``7``) leaves the predecessor standing, and summing it into the successor
    would count a bank that still files its own Call Report.
    """
    paths = fetch_structure()
    out: list[Succession] = []
    for row in _read_csv(paths["transformations"]):
        code = (row.get("TRNSFM_CD") or "").strip()
        if code not in ABSORPTION_CODES:
            continue
        try:
            pred = int(row["#ID_RSSD_PREDECESSOR"])
            succ = int(row["ID_RSSD_SUCCESSOR"])
        except (KeyError, ValueError):
            continue
        date = (row.get("DT_TRANS") or "").strip()
        if len(date) != 8 or not date.isdigit():
            continue
        out.append(
            Succession(
                predecessor=pred,
                successor=succ,
                date=date,
                code=code,
                pooled=(row.get("ACCT_METHOD") or "").strip() == "1",
            )
        )
    out.sort(key=lambda s: (s.date, s.predecessor, s.successor))
    return tuple(out)


@lru_cache(maxsize=1)
def by_successor() -> dict[int, tuple[Succession, ...]]:
    """Successor -> the events in which it absorbed something."""
    grouped: dict[int, list[Succession]] = defaultdict(list)
    for s in successions():
        grouped[s.successor].append(s)
    return {k: tuple(v) for k, v in grouped.items()}


@lru_cache(maxsize=1)
def failed_rssds() -> frozenset[int]:
    """Every entity that failed, by either of NIC's two independent records.

    The transformations table marks the resolution (``TRNSFM_CD = 50``) and
    the attributes table marks the termination (``REASON_TERM_CD`` 4 or 5).
    They agree on every case checked -- Washington Mutual Bank, Colonial Bank,
    First Republic Bank, Park National Bank -- and the union is taken so that
    a failure recorded in only one of them is still a failure.
    """
    out = {s.predecessor for s in successions() if s.failure}
    out.update(rssd for rssd, ent in entities().items() if ent.failed)
    return frozenset(out)


@lru_cache(maxsize=1)
def _relationships() -> list[tuple[int, int, str, str]]:
    """``(parent, offspring, start, end)`` for every control relationship.

    Filtered to ``CTRL_IND = 1`` here rather than at the walk, because the file
    also carries minority stakes and reporting relationships, and a
    non-control row is not a subsidiary in any sense the Call Report cares
    about.  Dates are kept as the file's ``YYYYMMDD`` strings, which sort
    correctly and avoid parsing 4 million rows into date objects.
    """
    paths = fetch_structure()
    out: list[tuple[int, int, str, str]] = []
    for row in _read_csv(paths["relationships"]):
        if row.get("CTRL_IND", "").strip() != "1":
            continue
        start = (row.get("DT_START") or "").strip()
        end = (row.get("DT_END") or "").strip() or OPEN_END
        try:
            parent = int(row["#ID_RSSD_PARENT"])
            child = int(row["ID_RSSD_OFFSPRING"])
        except (KeyError, ValueError):
            continue
        out.append((parent, child, start, end))
    return out


@lru_cache(maxsize=1)
def control_parents() -> dict[int, tuple[tuple[int, str, str], ...]]:
    """Offspring -> every ``(parent, start, end)`` control relationship it has had.

    The dated inverse of :func:`hierarchy`.  It answers when an entity
    *joined* an organisation, which the transformations table cannot: an
    acquisition that keeps the acquired company alive as a subsidiary --
    Countrywide, Merrill Lynch, Bear Stearns, MUFG Americas -- is recorded only
    here, as a relationship that begins on the closing date.
    """
    out: dict[int, list[tuple[int, str, str]]] = defaultdict(list)
    for parent, child, start, end in _relationships():
        out[child].append((parent, start, end))
    return {k: tuple(v) for k, v in out.items()}


def parents_on(rssd: int, as_of: dt.date) -> list[int]:
    """Controlling parents of ``rssd`` as the graph stood on ``as_of``."""
    stamp = as_of.strftime("%Y%m%d")
    out: list[int] = []
    for parent, start, end in control_parents().get(rssd, ()):
        if start and start > stamp:
            continue
        if end < stamp:
            continue
        out.append(parent)
    return out


def hierarchy(as_of: dt.date | None = None) -> dict[int, list[int]]:
    """Parent -> direct controlled offspring, as the graph stood on ``as_of``.

    Passing ``None`` uses every open relationship, which is the graph as it
    stands today.  Passing a date is what keeps a historical quarter honest:
    SunTrust Bank was not a Truist subsidiary in 2014, and the undated graph
    says it was.

    Cached per date.  The lineage walk and the universe build both ask for
    every quarter end in the window, and the answer does not change between
    the two.  Callers must treat the result as read-only.
    """
    return _hierarchy_at(None if as_of is None else as_of.strftime("%Y%m%d"))


@lru_cache(maxsize=512)
def _hierarchy_at(stamp: str | None) -> dict[int, list[int]]:
    kids: dict[int, list[int]] = defaultdict(list)
    for parent, child, start, end in _relationships():
        if stamp is None:
            if end != OPEN_END:
                continue
        else:
            # A relationship counts if it had begun and had not yet ended.
            # A missing start date is treated as "always been true", which is
            # how NIC represents relationships predating its own records.
            if start and start > stamp:
                continue
            if end < stamp:
                continue
        kids[parent].append(child)
    return dict(kids)


def descendants(root: int, kids: dict[int, list[int]], *, max_depth: int = 12) -> set[int]:
    """Every entity controlled by ``root``, transitively.

    Depth-bounded and cycle-guarded.  NIC's graph is *not* a tree -- an entity
    can appear under two parents, and a badly dated pair can form a cycle --
    and an unguarded walk over a large BHC does not terminate.
    """
    seen: set[int] = set()
    frontier = [(root, 0)]
    while frontier:
        node, depth = frontier.pop()
        if depth >= max_depth:
            continue
        for child in kids.get(node, ()):
            if child in seen or child == root:
                continue
            seen.add(child)
            frontier.append((child, depth + 1))
    return seen


def call_filers(
    root: int,
    roster: set[int],
    *,
    as_of: dt.date | None = None,
    kids: dict[int, list[int]] | None = None,
) -> list[int]:
    """The RSSDs under ``root`` that filed a Call Report in the given quarter.

    ``roster`` is that quarter's own POR file -- the authoritative list of who
    filed -- so the result is an intersection of two facts rather than a guess
    from entity type.  A holding company that *is itself* a filer (rare, but a
    few industrial banks are structured that way) is included.
    """
    graph = kids if kids is not None else hierarchy(as_of)
    under = descendants(root, graph)
    if root in roster:
        under.add(root)
    return sorted(under & roster)


def normalise_ein(value: str | None) -> str | None:
    """A federal tax id in a form both registries agree on.

    NIC stores ``ID_TAX`` as a number and drops the leading zero; EDGAR reports
    the same id as nine characters with the zero intact.  Joining the raw
    strings therefore misses every filer whose EIN begins with 0 -- which is
    Citizens Financial Group (050412693 against 50412693), State Street, and
    Santander USA.  Citizens is the one that shows why it matters: with the EIN
    join failing it fell through to a name match, and there is an unrelated
    company also called CITIZENS FINANCIAL GROUP, INC. that it matched instead.
    """
    digits = "".join(c for c in (value or "") if c.isdigit())
    return digits.zfill(9) if digits and digits.strip("0") else None


def by_ein() -> dict[str, list[Entity]]:
    """EIN -> entities carrying it.  A list, because NIC is not unique on EIN."""
    out: dict[str, list[Entity]] = defaultdict(list)
    for ent in entities().values():
        key = normalise_ein(ent.ein)
        if key:
            out[key].append(ent)
    return dict(out)


def top_holders(as_of: dt.date | None = None) -> dict[int, int]:
    """Offspring -> its highest controlling ancestor.

    Only used for diagnostics: it answers "whose bank is this?" for a charter
    that turned up in the roster but under no holding company we track.
    """
    kids = hierarchy(as_of)
    parent_of: dict[int, int] = {}
    for parent, children in kids.items():
        for child in children:
            parent_of.setdefault(child, parent)
    out: dict[int, int] = {}
    for node in parent_of:
        seen = {node}
        cur = node
        while cur in parent_of and parent_of[cur] not in seen:
            cur = parent_of[cur]
            seen.add(cur)
        out[node] = cur
    return out
