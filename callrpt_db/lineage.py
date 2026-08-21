"""Predecessor lineage: what each 2026 organisation used to be.

Why the 2013 panel could ignore this and a 2001 panel cannot
------------------------------------------------------------
The universe is today's DFAST list, keyed on each firm's 2026 RSSD, and the
existing build resolves that RSSD's charters quarter by quarter against the
dated NIC graph.  That is exact for the entity that exists today -- and it is
the wrong entity for most of the window.  Wells Fargo in 2007 was Wells Fargo
*and Wachovia*, two organisations of similar size that would not share a
balance sheet for another year; PNC was PNC and National City; Truist was
BB&T and SunTrust and Colonial.  A panel that carries only the survivor's own
RSSD back to 2001 shows Wells Fargo's loan book doubling at 2008Q4, reads as
growth, and loses exactly the institutions whose 2007-2009 deterioration is
the reason to go back that far.

What a predecessor is
---------------------
NIC's transformations table records every event in which one entity's whole
balance sheet passed to another: a merger (``TRNSFM_CD = 1``), a charter
retained under a new RSSD (``9``), a failure resolved with government
assistance (``50``).  A predecessor of organisation *O* is any entity that was
absorbed, by one of those events, into something that was part of *O* at the
time -- transitively, so SouthTrust (into Wachovia, 2004) is a Wells Fargo
predecessor, and Providian (into Washington Mutual Bank, 2005) would be a
JPMorgan one.

"Into something that was part of *O* at the time" is the clause that matters,
and it is why the walk runs over *members* rather than over the top entity.
Countrywide was not merged into Bank of America Corporation; it was merged
into a subsidiary.  Washington Mutual Bank was sold by the FDIC to JPMorgan
Chase Bank, N.A., a charter, not to the holding company.  Walking only the
holding company's own transformations finds Fleet and MBNA and misses both.
So for every quarter the organisation's members -- the current roots, every
predecessor still standing, and everything the dated graph puts beneath them
-- are the set whose absorptions are examined, and an event is claimed only
when its successor was a member at the quarter end it falls in.

Three kinds of succession
-------------------------
``merger``
    An outside organisation absorbed whole.  Its Call Reports before the date
    are history the survivor did not file but now owns.
``fdic_assisted``
    The predecessor failed and the FDIC arranged the disposition.  These are
    the stress observations: Colonial Bank's 2009Q2 Call Report is the
    balance sheet of an institution one quarter from failure, and it is in
    Truist's lineage.  Identified by the transformation code **and** checked
    against the attributes table's own termination reason, which agrees on
    every case examined.
``reorg``
    The predecessor was already a member of the organisation when it was
    absorbed -- Chase Bank USA into JPMorgan Chase Bank, Zions' holding company
    into its own bank.  No history is added; the charter's reports were
    already being summed.  Listed because the lineage should be complete, and
    because these are the events that restate year-to-date flows (see
    ``panel.quarterize``).

What is deliberately not done
-----------------------------
A predecessor that is itself a tracked firm in the panel -- Discover before
Capital One bought it, MUFG Americas before US Bancorp -- stays its own row
for as long as it is tracked, and is claimed by the successor's lineage only
afterwards.  That rule is applied by :func:`panel.universe_filers`, not
here; the lineage records the succession regardless, so the CSV is a map of
what happened and the panel is a statement of what was summed.

Thrifts are the gap this cannot close.  Washington Mutual, Golden West,
Countrywide Bank, Sovereign, IndyMac and every other OTS-supervised
institution filed a Thrift Financial Report, not a Call Report, until 2012Q1,
and CDR has nothing for them.  Their lineage rows are written anyway -- the
succession is real -- and :func:`panel.unfiled_depositories` counts, per
bank-quarter, the insured depositories in the organisation's tree that filed
nothing CDR holds, so a reader can see where the synthetic history is thin.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

import polars as pl

from . import config, nic

log = logging.getLogger(__name__)

# Earliest quarter CDR offers; nothing before it can contribute a row, so the
# walk does not chase successions that completed before it.
DEFAULT_FLOOR = dt.date(2001, 1, 1)

MERGER = "merger"
FDIC_ASSISTED = "fdic_assisted"
REORG = "reorg"
# The organisation's own top entity, in the lineage file only.
SELF = "self"

# Entity types worth listing in the lineage file: depositories, and the
# holding companies that own them.  The walk itself passes through every
# entity type -- a succession can route through a nonbank intermediate -- but
# a dissolved audit-services LLC is not a predecessor anyone needs to read.
HOLDING_TYPES: frozenset[str] = frozenset({"BHC", "FHD", "SLHC", "IHC", "FBH", "FBO"})
DEPOSITORY_TYPES: frozenset[str] = config.DEPOSITORY_ENTITY_TYPES | {"SAL", "FSB", "SSB"}


def quarter_end_on_or_after(day: dt.date) -> dt.date:
    """The quarter end a date falls in."""
    month = ((day.month - 1) // 3 + 1) * 3
    last = 31 if month in (3, 12) else 30
    return dt.date(day.year, month, last)


def previous_quarter_end(day: dt.date) -> dt.date:
    """The last day of the quarter before the one ``day`` falls in."""
    start_month = day.month - (day.month - 1) % 3
    return dt.date(day.year, start_month, 1) - dt.timedelta(days=1)


@dataclass(frozen=True)
class Predecessor:
    """One entity in an organisation's lineage, and how it got there."""

    rssd: int
    name: str
    entity_type: str
    successor: int  # the member it was absorbed into
    effective_from: dt.date
    effective_to: dt.date  # the succession date
    succession_type: str  # merger | fdic_assisted | reorg
    code: str  # TRNSFM_CD
    pooled: bool

    def active_on(self, day: dt.date) -> bool:
        """Whether the predecessor still stood on its own on ``day``.

        Inclusive at both ends.  A merger dated on a quarter end is one where
        both the predecessor's last filing and the successor's first combined
        one can carry the same date, and the dated graph may put the charter
        under either parent; claiming it from both sides and letting the set
        union dedupe is safer than an off-by-one day that loses a quarter.
        """
        return self.effective_from <= day <= self.effective_to

    @property
    def reportable(self) -> bool:
        if self.code == ACQUISITION_CODE:
            return True  # passed the owns-a-depository test to get here
        ent = nic.entities().get(self.rssd)
        if ent is None:
            return False
        return (
            ent.fdic_cert is not None
            or ent.is_bhc
            or ent.is_ihc
            or ent.entity_type in HOLDING_TYPES
            or ent.entity_type in DEPOSITORY_TYPES
        )


@dataclass
class Lineage:
    holding: config.Holding
    predecessors: dict[int, Predecessor] = field(default_factory=dict)

    def active(self, day: dt.date) -> list[Predecessor]:
        return [p for p in self.predecessors.values() if p.active_on(day)]

    def roots_on(self, day: dt.date) -> list[int]:
        """Top entities to walk down from on ``day``: own RSSDs plus live predecessors."""
        return [*self.holding.rssds, *(p.rssd for p in self.active(day))]


def _opened(rssd: int, floor: dt.date) -> dt.date:
    ent = nic.entities().get(rssd)
    if ent is None or ent.opened is None:
        return floor
    try:
        opened = nic.stamp_to_date(ent.opened)
    except ValueError:
        return floor
    return max(floor, opened)


ACQUISITION_CODE = "ACQ"


def resolve(
    holding: config.Holding,
    quarter_ends: list[dt.date],
    *,
    floor: dt.date = DEFAULT_FLOOR,
    filers: set[int] | None = None,
) -> Lineage:
    """Walk the organisation's members backwards through every quarter.

    ``quarter_ends`` are the dates to examine, in any order; they are sorted
    newest first so that a predecessor discovered in 2008 is already a root
    when 2007 is examined.  The walk is over the dated graph, so a subsidiary
    that was sold in 2003 is a member in 2002 and not in 2004.

    Two kinds of event make a predecessor.  A **transformation** -- the
    member absorbed something, and NIC says so.  An **acquisition** -- an
    entity that was not in the organisation at the previous quarter end is
    in it at this one, joined through a control relationship that began in
    between, existed before the quarter, and is or owns a depository.  The
    second is what catches Countrywide, Merrill Lynch, Bear Stearns, MUFG
    Americas and Golden West, all of which were bought and kept as
    subsidiaries; the transformations table records nothing until the shell
    is dissolved years later, by which time the banks beneath it have long
    since moved.  An acquired entity is recorded with ``code = "ACQ"`` and
    ``succession_type = "merger"``.

    ``filers`` -- every RSSD that has ever filed a Call Report in the window
    -- is how "owns a depository" is decided for entities NIC's attribute
    files leave out, which include Discover Bank and National City Bank.
    """
    lineage = Lineage(holding)
    own = set(holding.rssds)
    succ_index = nic.by_successor()
    entities = nic.entities()
    failed = nic.failed_rssds()
    parents = nic.control_parents()
    filers = filers or set()

    def is_depository(rssd: int) -> bool:
        if rssd in filers:
            return True
        ent = entities.get(rssd)
        return ent is not None and (
            ent.fdic_cert is not None or ent.entity_type in DEPOSITORY_TYPES
        )

    def was_member_before(rssd: int, day: dt.date) -> bool:
        """Whether ``rssd`` was already part of the organisation before ``day``.

        Checked at the quarter end preceding the event, against the roots
        that outlive the event.  SunTrust Bank was a member on 2019-09-30
        only through SunTrust Banks, Inc., which ceased on the same day the
        bank did -- so the bank's absorption is a merger, not a reorg.  Chase
        Bank USA on 2019-03-31 was a member through JPMorgan Chase & Co.,
        which is still here, so its absorption is a reorg.
        """
        if rssd in own:
            return True
        before = previous_quarter_end(quarter_end_on_or_after(day))
        graph = nic.hierarchy(before)
        for root in own:
            if rssd in nic.descendants(root, graph):
                return True
        for pred in lineage.predecessors.values():
            if pred.effective_to <= day or pred.rssd == rssd:
                continue
            if pred.active_on(before) and rssd in nic.descendants(pred.rssd, graph):
                return True
        return False

    def record(
        rssd: int, successor: int, day: dt.date, code: str, pooled: bool, failure: bool
    ) -> None:
        """Record a predecessor and chase its own predecessors, depth first."""
        stack: list[tuple[int, int, dt.date, str, bool, bool]] = [
            (rssd, successor, day, code, pooled, failure)
        ]
        while stack:
            pred, succ, when, cd, pool, failed_event = stack.pop()
            if pred in own or pred in lineage.predecessors or when < floor:
                continue
            ent = entities.get(pred)
            if failed_event or pred in failed:
                kind = FDIC_ASSISTED
            elif cd != ACQUISITION_CODE and was_member_before(pred, when):
                kind = REORG
            else:
                kind = MERGER
            lineage.predecessors[pred] = Predecessor(
                rssd=pred,
                name=ent.name if ent else "",
                entity_type=ent.entity_type if ent else "",
                successor=succ,
                effective_from=_opened(pred, floor),
                effective_to=when,
                succession_type=kind,
                code=cd,
                pooled=pool,
            )
            # Everything absorbed into the predecessor before it was itself
            # absorbed is ours too.  Dates after its own succession would be a
            # data defect and are not followed.
            for earlier in succ_index.get(pred, ()):
                if floor <= earlier.on <= when:
                    stack.append(
                        (earlier.predecessor, pred, earlier.on, earlier.code, earlier.pooled, earlier.failure)
                    )

    def members_on(day: dt.date) -> set[int]:
        graph = nic.hierarchy(day)
        out: set[int] = set()
        for root in lineage.roots_on(day):
            out.add(root)
            out |= nic.descendants(root, graph)
        return out

    for q_end in sorted(set(quarter_ends), reverse=True):
        if q_end < floor:
            continue
        q_start = previous_quarter_end(q_end)
        start_stamp = q_start.strftime("%Y%m%d")
        end_stamp = q_end.strftime("%Y%m%d")
        members = members_on(q_end)

        # 1. Transformations into a member, dated inside the quarter.
        for member in sorted(members):
            for ev in succ_index.get(member, ()):
                if q_start < ev.on <= q_end and ev.on >= floor:
                    record(ev.predecessor, member, ev.on, ev.code, ev.pooled, ev.failure)

        # 2. Acquisitions: members that were outside the organisation at the
        #    previous quarter end and joined through a relationship that
        #    began since.  ``members`` is re-read because step 1 may have
        #    added roots whose subtrees now count as already ours.
        before = members_on(q_start)
        members = members_on(q_end)
        graph_start = nic.hierarchy(q_start)
        # A member whose only claim to prior membership is being a reorg root
        # itself (MUFG Union Bank, a root until its 2023 merger into U.S.
        # Bank) was not inside the organisation before it was acquired.
        self_rooted = {
            m
            for m in members & before
            if m not in own
            and (p := lineage.predecessors.get(m)) is not None
            and p.succession_type == REORG
            and not any(
                m in nic.descendants(r, graph_start)
                for r in lineage.roots_on(q_start)
                if r != m
            )
        }
        for member in sorted((members - before) | self_rooted):
            if member in own:
                continue
            known = lineage.predecessors.get(member)
            if known is not None and known.succession_type != REORG:
                continue
            # A reorg already on file is the shell's eventual dissolution
            # (Merrill Lynch & Co. into Bank of America, 2013; MUFG Union Bank
            # into U.S. Bank, 2023).  The acquisition that brought it in is
            # the event that matters, and it is dated here instead.
            joins = [
                start
                for parent, start, _end in parents.get(member, ())
                if parent in members and start and start_stamp < start <= end_stamp
            ]
            if not joins:
                continue
            ent = entities.get(member)
            if ent is not None:
                # Formed inside the quarter: a new subsidiary, not a purchase.
                # NIC has no opening date for many older entities (Merrill
                # Lynch & Co. among them); an unknown date is not evidence of
                # youth.
                if ent.opened and ent.opened > start_stamp:
                    continue
            elif member not in filers:
                continue  # unknown to NIC and never a filer: cannot judge
            if any(p in before and p != member for p in nic.parents_on(member, q_start)):
                continue  # was beneath us already through another path
            subtree = {member} | nic.descendants(member, nic.hierarchy(q_start))
            if not any(is_depository(e) for e in subtree):
                continue  # a nonbank purchase contributes no Call Report
            joined = nic.stamp_to_date(min(joins))
            if known is not None:
                if known.effective_to <= joined:
                    continue
                del lineage.predecessors[member]
            acquirer = max(
                (p for p, s, _e in parents.get(member, ()) if p in members and s == min(joins)),
                default=holding.rssd,
            )
            record(member, acquirer, joined, ACQUISITION_CODE, False, False)
    return lineage


def resolve_all(
    holdings: tuple[config.Holding, ...],
    quarter_ends: list[dt.date],
    *,
    floor: dt.date = DEFAULT_FLOOR,
    filers: set[int] | None = None,
) -> dict[str, Lineage]:
    out: dict[str, Lineage] = {}
    for holding in holdings:
        lineage = resolve(holding, quarter_ends, floor=floor, filers=filers)
        out[holding.ticker] = lineage
        kinds = {k: 0 for k in (MERGER, FDIC_ASSISTED, REORG)}
        for p in lineage.predecessors.values():
            kinds[p.succession_type] += 1
        log.info(
            "%s: %d predecessors (%d merger, %d fdic_assisted, %d reorg)",
            holding.ticker,
            len(lineage.predecessors),
            kinds[MERGER],
            kinds[FDIC_ASSISTED],
            kinds[REORG],
        )
    return out


LINEAGE_SCHEMA: dict[str, type[pl.DataType] | pl.DataType] = {
    "bhc_name": pl.Utf8,
    "bhc_rssd_2026": pl.Int64,
    "predecessor_rssd": pl.Int64,
    "predecessor_name": pl.Utf8,
    "effective_from": pl.Date,
    "effective_to": pl.Date,
    "succession_type": pl.Utf8,
    "ticker": pl.Utf8,
    "predecessor_type": pl.Utf8,
    "via_rssd": pl.Int64,
    "via_name": pl.Utf8,
    "transformation_code": pl.Utf8,
    "pooled": pl.Boolean,
    "tracked_separately": pl.Boolean,
    "charter_quarters": pl.Int64,
}


def to_frame(
    lineages: dict[str, Lineage],
    *,
    contributed: dict[tuple[str, int], int] | None = None,
    tracked: set[int] | None = None,
    names: dict[int, str] | None = None,
) -> pl.DataFrame:
    """The lineage as one row per (organisation, predecessor).

    ``contributed`` maps ``(ticker, predecessor_rssd)`` to the number of
    charter-quarters that predecessor supplied to the panel; a predecessor
    that supplied any is listed whatever its entity type, and one that
    supplied none is listed only if it is a depository or a holding company.
    ``tracked`` is the set of RSSDs that are top entities of firms in the
    panel in their own right.  ``names`` supplies names for entities NIC's
    attribute files omit -- the Call Report roster knows Discover Bank and
    MUFG Union Bank even where NIC's attributes do not.
    """
    contributed = contributed or {}
    tracked = tracked or set()
    names = names or {}
    entities = nic.entities()

    def name_of(rssd: int) -> str:
        ent = entities.get(rssd)
        if ent is not None and ent.name:
            return ent.name
        return names.get(rssd, "")

    rows: list[dict] = []
    for ticker, lineage in lineages.items():
        # The organisation itself, so that a firm with no predecessors still
        # has a row and the file is complete over the universe.  A second
        # own RSSD (Zions' dissolved holding company) is a ``self`` row too.
        for own in lineage.holding.rssds:
            ent = entities.get(own)
            rows.append(
                {
                    "bhc_name": lineage.holding.name,
                    "bhc_rssd_2026": lineage.holding.rssd,
                    "predecessor_rssd": own,
                    "predecessor_name": name_of(own),
                    "effective_from": _opened(own, DEFAULT_FLOOR),
                    "effective_to": nic.stamp_to_date(ent.ended) if ent and ent.ended else None,
                    "succession_type": SELF,
                    "ticker": ticker,
                    "predecessor_type": ent.entity_type if ent else "",
                    "via_rssd": None,
                    "via_name": "",
                    "transformation_code": "",
                    "pooled": False,
                    "tracked_separately": False,
                    "charter_quarters": contributed.get((ticker, own), 0),
                }
            )
        for pred in sorted(lineage.predecessors.values(), key=lambda p: (p.effective_to, p.rssd)):
            quarters = contributed.get((ticker, pred.rssd), 0)
            if quarters == 0 and not (pred.reportable or pred.rssd in names):
                continue
            rows.append(
                {
                    "bhc_name": lineage.holding.name,
                    "bhc_rssd_2026": lineage.holding.rssd,
                    "predecessor_rssd": pred.rssd,
                    "predecessor_name": pred.name or name_of(pred.rssd),
                    "effective_from": pred.effective_from,
                    "effective_to": pred.effective_to,
                    "succession_type": pred.succession_type,
                    "ticker": ticker,
                    "predecessor_type": pred.entity_type,
                    "via_rssd": pred.successor,
                    "via_name": name_of(pred.successor),
                    "transformation_code": pred.code,
                    "pooled": pred.pooled,
                    "tracked_separately": pred.rssd in tracked,
                    "charter_quarters": quarters,
                }
            )
    if not rows:
        return pl.DataFrame(schema=LINEAGE_SCHEMA)
    return pl.DataFrame(rows, schema=LINEAGE_SCHEMA).sort(
        ["bhc_name", "effective_to", "predecessor_rssd"]
    )


def pooled_events(rssds: set[int]) -> pl.DataFrame:
    """Common-control mergers between two charters in ``rssds``.

    One row per (successor, predecessor, quarter the merger fell in).  Used by
    ``panel.quarterize`` to undo the year-to-date restatement that a pooling
    merger imposes on the survivor's income statement.
    """
    rows = [
        (s.successor, s.predecessor, quarter_end_on_or_after(s.on))
        for s in nic.successions()
        if s.pooled and s.successor in rssds and s.predecessor in rssds
    ]
    return pl.DataFrame(
        rows,
        schema={"rssd": pl.Int64, "predecessor": pl.Int64, "period": pl.Date},
        orient="row",
    )
