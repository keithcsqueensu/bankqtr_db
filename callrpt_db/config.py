"""Static configuration: FFIEC endpoints, cache layout, holding-company universe.

Why a second source at all
--------------------------
``bankqtr_db`` reads SEC filings, and its README names the gap that leaves:
seven of the eight US intermediate holding companies of foreign banking
organisations file **no** 10-K, so they cannot appear in an EDGAR-sourced
panel at all.  They do file a Call Report -- or rather their bank subsidiaries
do -- and that is public, bulk, and reaches back to 2001.

What the entity is
------------------
The Call Report (FFIEC 031/041/051) is filed by an **insured depository**, not
by its holding company.  ``JPMORGAN CHASE BANK, NATIONAL ASSOCIATION`` is not
``JPMORGAN CHASE & CO.``: the bank excludes the broker-dealer, the card
funding trusts and the non-bank subsidiaries, so its balance sheet is smaller
than the 10-K's and the two will not tie.  The panel therefore carries both
levels -- see :mod:`callrpt_db.panel` -- and every holding-company row records
how many bank charters were summed into it.

RSSD identifiers below were resolved by ``scripts/resolve_rssd.py`` against the
FFIEC NIC bulk attribute file, bridging on **EIN** (NIC ``ID_TAX`` against
EDGAR ``ein``) rather than on company name, and the evidence for each is in
``data/out/rssd_resolution.csv``.  Names are matched only where an entity has
no SEC registration to carry an EIN.
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass, field
from pathlib import Path

# FFIEC does not publish a fair-access policy the way SEC does, but the same
# courtesy applies: declare a contact.  Shared with bankqtr_db's BANKQTR_UA.
USER_AGENT = os.environ.get(
    "BANKQTR_UA",
    os.environ.get("CALLRPT_UA", "research@example.com"),
)

# FFIEC's public sites reject a bare scripted client (403 on www.ffiec.gov),
# so the contact rides inside a browser-shaped string rather than replacing it.
FFIEC_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        f"callrpt_db/0.1 (+{USER_AGENT})"
    ),
    "Accept-Encoding": "gzip, deflate",
}

# The CDR bulk-download page is an ASP.NET WebForm; see cdr.py for the flow.
CDR_BULK_URL = "https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx"

# NIC structure data: entity attributes and the parent/offspring graph.
NPW_BASE = "https://www.ffiec.gov/npw/FinancialReport"
NPW_REFERER = f"{NPW_BASE}/DataDownload"
NPW_FILES = {
    "attributes_active": "ReturnAttributesActiveZipFileCSV",
    "attributes_closed": "ReturnAttributesClosedZipFileCSV",
    "relationships": "ReturnRelationshipsZipFileCSV",
    # Predecessor -> successor events: mergers, failures, re-charterings.
    # Singular "Transformation" in the endpoint name, plural in the file.
    "transformations": "ReturnTransformationZipFileCSV",
}

# One bulk zip is ~6 MB and the server is not a CDN; stay gentle.
MAX_REQUESTS_PER_SECOND = 1.0

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW_CALL = DATA / "raw" / "call"  # one zip per reporting period
RAW_NIC = DATA / "raw" / "nic"  # NPW structure bulk files
OUT = DATA / "out"


# --------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Holding:
    """A banking organisation, keyed on the RSSD at the top of it.

    ``cik`` is present only where the firm also files with the SEC, and is what
    joins this panel to ``bankqtr_db``'s.  ``ticker`` is carried for the same
    reason; the IHCs have no ticker and get a synthetic one so the two panels
    can be concatenated without a null key.

    ``rssds`` holds **every** top entity the organisation has had across the
    window, because one firm is not one RSSD for all time.  Zions is the case
    that forces it: ZIONS BANCORPORATION (RSSD 1027004) was the holding company
    until it was merged into its own bank in 2018, and from then on the top
    entity of the organisation is ZIONS BANCORPORATION, NATIONAL ASSOCIATION
    (276579) -- a different RSSD, a different EIN, and no parent at all.  The
    quarter's filers are the union of what sits under each, which is a single
    charter in both eras.
    """

    name: str
    rssd: int
    ticker: str
    cik: str | None = None
    category: str = "regional"
    tier: str = "dfast"
    last_filing: dt.date | None = None
    # Additional top entities for the same organisation, in other eras.
    also_rssd: tuple[int, ...] = field(default=())
    # RSSDs of the depository subsidiaries whose Call Reports roll up here.
    # Empty means "discover from the NIC relationship graph at build time",
    # which is the normal case; a non-empty tuple pins the set.
    subsidiaries: tuple[int, ...] = field(default=())

    @property
    def rssds(self) -> tuple[int, ...]:
        return (self.rssd, *self.also_rssd)


# Populated by scripts/resolve_rssd.py.  Do not hand-edit: rerun the resolver.
HOLDINGS: tuple[Holding, ...] = (
    Holding("Ally Financial", 1562859, "ALLY", cik="0000040729", category="card_consumer", tier="dfast"),
    Holding("Ameriprise", 2433312, "AMP", cik="0000820027", category="broker", tier="dfast"),
    Holding("American Express", 1275216, "AXP", cik="0000004962", category="card_consumer", tier="dfast"),
    Holding("Bank of America", 1073757, "BAC", cik="0000070858", category="universal", tier="dfast"),
    Holding("BNY Mellon", 3587146, "BNY", cik="0001390777", category="custody", tier="dfast"),
    Holding("Citigroup", 1951350, "C", cik="0000831001", category="universal", tier="dfast"),
    Holding("Citizens Financial", 1132449, "CFG", cik="0000759944", category="regional", tier="dfast"),
    Holding("Capital One", 2277860, "COF", cik="0000927628", category="card_consumer", tier="dfast"),
    Holding("Discover", 3846375, "DFS", cik="0001393612", category="card_consumer", tier="dfast", last_filing=dt.date(2025, 5, 18)),
    Holding("First Citizens BancShares", 1075612, "FCNCA", cik="0000798941", category="regional", tier="dfast"),
    Holding("Fifth Third", 1070345, "FITB", cik="0000035527", category="regional", tier="dfast"),
    Holding("Goldman Sachs", 2380443, "GS", cik="0000886982", category="ibank", tier="dfast"),
    Holding("Huntington", 1068191, "HBAN", cik="0000049196", category="regional", tier="dfast"),
    Holding("JPMorgan Chase", 1039502, "JPM", cik="0000019617", category="universal", tier="dfast"),
    Holding("KeyCorp", 1068025, "KEY", cik="0000091576", category="regional", tier="dfast"),
    Holding("Morgan Stanley", 2162966, "MS", cik="0000895421", category="ibank", tier="dfast"),
    Holding("M&T Bank", 1037003, "MTB", cik="0000036270", category="regional", tier="dfast"),
    Holding("Northern Trust", 1199611, "NTRS", cik="0000073124", category="custody", tier="dfast"),
    Holding("PNC Financial", 1069778, "PNC", cik="0000713676", category="regional", tier="dfast"),
    Holding("Regions Financial", 3242838, "RF", cik="0001281761", category="regional", tier="dfast"),
    Holding("Raymond James", 3815157, "RJF", cik="0000720005", category="broker", tier="dfast"),
    Holding("Charles Schwab", 1026632, "SCHW", cik="0000316709", category="broker", tier="dfast"),
    Holding("State Street", 1111435, "STT", cik="0000093751", category="custody", tier="dfast"),
    Holding("Synchrony Financial", 4504654, "SYF", cik="0001601712", category="card_consumer", tier="dfast"),
    Holding("Truist Financial", 1074156, "TFC", cik="0000092230", category="regional", tier="dfast"),
    Holding("US Bancorp", 1119794, "USB", cik="0000036104", category="regional", tier="dfast"),
    Holding("Wells Fargo", 1120754, "WFC", cik="0000072971", category="universal", tier="dfast"),
    Holding("Zions Bancorporation", 1027004, "ZION", cik="0000109380", category="regional", tier="dfast", also_rssd=(276579,)),
    Holding("Barclays US LLC", 5006575, "BCSUS", cik=None, category="regional", tier="ihc"),
    Holding("BMO Financial Corp", 1245415, "BMOUS", cik=None, category="regional", tier="ihc"),
    Holding("BNP Paribas USA, Inc.", 1575569, "BNPUS", cik=None, category="regional", tier="ihc"),
    Holding("Credit Suisse Holdings (USA), Inc.", 1574834, "CSUS", cik=None, category="regional", tier="ihc"),
    Holding("DB USA Corporation", 2816906, "DBUS", cik=None, category="regional", tier="ihc"),
    Holding("HSBC USA", 1020201, "HSBCUSA", cik="0000083246", category="regional", tier="ihc"),
    Holding("RBC US Group Holdings LLC", 5280254, "RBCUS", cik=None, category="regional", tier="ihc"),
    Holding("Santander Holdings USA", 3981856, "SHUSA", cik="0000811830", category="regional", tier="ihc"),
    Holding("TD Group US Holdings LLC", 3606542, "TDUS", cik=None, category="regional", tier="ihc"),
    Holding("UBS Americas Holding LLC", 4846998, "UBSUS", cik=None, category="regional", tier="ihc"),
    Holding("MUFG Americas Holdings", 1378434, "MUFGA", cik="0001011659", category="regional", tier="inactive", last_filing=dt.date(2021, 12, 1)),
    Holding("Popular", 1129382, "BPOP", cik="0000763901", category="regional", tier="comparator"),
    Holding("Cullen/Frost", 1102367, "CFR", cik="0000039263", category="regional", tier="comparator"),
    Holding("Comerica", 1029259, "CMA", cik="0000028412", category="regional", tier="comparator", also_rssd=(1199844,)),
    Holding("East West Bancorp", 2734233, "EWBC", cik="0001069157", category="regional", tier="comparator"),
    Holding("First Horizon", 1094640, "FHN", cik="0000036966", category="regional", tier="comparator"),
    Holding("Prosperity Bancshares", 1109599, "PB", cik="0001068851", category="regional", tier="comparator"),
    Holding("Synovus Financial", 1078846, "SNV", cik="0000018349", category="regional", tier="comparator"),
    Holding("Valley National", 1048773, "VLY", cik="0000714310", category="regional", tier="comparator"),
    Holding("Western Alliance", 2349815, "WAL", cik="0001212545", category="regional", tier="comparator"),
    Holding("Webster Financial", 1145476, "WBS", cik="0000801337", category="regional", tier="comparator"),
)

# NIC entity types that actually appear in a Call Report roster, counted from
# the 2025Q4 POR file rather than taken from the NIC code list -- five of the
# codes a reasonable person would guess (SLA, IBK, NTC, CSB, SBD) file no Call
# Report, and three that do (SAL, MTC, CSA, CPB) are not obvious.
#
# This is documentation and a sanity check, **not** the filter.  The authority
# on who filed in a given quarter is that quarter's own POR roster, which
# ``nic.call_filers`` intersects the descendant walk against; an entity-type
# allowlist would silently drop a charter whose type NIC later recodes.
DEPOSITORY_ENTITY_TYPES: frozenset[str] = frozenset(
    {"NMB", "SMB", "NAT", "SSB", "SAL", "FSB", "MTC", "CSA", "CPB"}
)

# Categories whose loan books are thin by construction -- mirrors
# bankqtr_db.config.THIN_LOAN_BOOK so the two coverage reports agree on what
# counts as a real gap.
THIN_LOAN_BOOK = ("custody", "broker")

CECL_ADOPTION_DEFAULT = dt.date(2020, 1, 1)


def cecl_adoption(ticker: str) -> dt.date:
    """First Call Report period reported under CECL.

    Unlike the SEC panel there is no per-filer exception here: the Call Report
    instructions tie adoption to the same fiscal-year rule, and every filer in
    this universe is a calendar-year bank.  Raymond James' September year-end
    applies to the holding company, not to Raymond James Bank.
    """
    return CECL_ADOPTION_DEFAULT


def ensure_dirs() -> None:
    for p in (RAW_CALL, RAW_NIC, OUT):
        p.mkdir(parents=True, exist_ok=True)
