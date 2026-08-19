"""Static configuration: bank universe, EDGAR endpoints, on-disk cache layout.

Universe definition
-------------------
The target set is "DFAST BHCs" -- firms subject to the Federal Reserve's
supervisory stress test.  That population is larger than the 19 names the
project started with, and it is not fully reachable from EDGAR:

* ``dfast`` -- supervisory stress-test participants across the recent cycles
  that file 10-K/10-Q with the SEC.  These are the benchmarkable peers.
* ``ihc`` -- US intermediate holding companies of foreign banking
  organisations.  Most (BMO Financial Corp, TD Group US Holdings, UBS Americas
  Holding, DB USA, Barclays US LLC, RBC US Group Holdings, BNP Paribas USA)
  file **no** 10-K with the SEC -- they report only on FR Y-9C -- so they
  cannot appear in an EDGAR-sourced panel at all.  See ``NON_SEC_IHCS``.
  The two that do file are carried here.
* ``inactive`` -- firms that were in the universe but stopped filing after an
  acquisition.  Kept so historical quarters remain in the panel.

CIKs below were resolved against EDGAR (company_tickers.json and full-text
search), not from memory, and every entry was confirmed to have 10-K/10-Q
filings.
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass, field
from pathlib import Path

# SEC requires a declared contact in the User-Agent. Override with BANKQTR_UA.
USER_AGENT = os.environ.get("BANKQTR_UA", "research@example.com")

EDGAR_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

# SEC fair-access limit is 10 requests/second; we stay well under it.
MAX_REQUESTS_PER_SECOND = 6.0

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW_FACTS = DATA / "raw" / "companyfacts"
RAW_HTML = DATA / "raw" / "html"
RAW_INSTANCES = DATA / "raw" / "instances"
OUT = DATA / "out"


@dataclass(frozen=True)
class Bank:
    name: str
    cik: str  # zero-padded to 10 digits
    ticker: str
    # Business model drives which variables are even meaningful: a custody
    # bank has no CRE book, a card issuer has no construction book.
    category: str = "regional"
    tier: str = "dfast"
    # Set when the filer stopped reporting (acquisition), so the panel can
    # distinguish "no disclosure" from "no longer exists".
    last_filing: dt.date | None = None
    aliases: tuple[str, ...] = field(default=())

    @property
    def cik_int(self) -> int:
        return int(self.cik)


BANKS: tuple[Bank, ...] = (
    # ---- GSIBs / universal banks -----------------------------------------
    Bank("JPMorgan Chase", "0000019617", "JPM", "universal"),
    Bank("Bank of America", "0000070858", "BAC", "universal"),
    Bank("Wells Fargo", "0000072971", "WFC", "universal"),
    Bank("Citigroup", "0000831001", "C", "universal"),
    Bank("Goldman Sachs", "0000886982", "GS", "ibank"),
    Bank("Morgan Stanley", "0000895421", "MS", "ibank"),
    # ---- Large regionals --------------------------------------------------
    Bank("US Bancorp", "0000036104", "USB", "regional"),
    Bank("PNC Financial", "0000713676", "PNC", "regional"),
    Bank("Truist Financial", "0000092230", "TFC", "regional"),
    Bank("Citizens Financial", "0000759944", "CFG", "regional"),
    Bank("Regions Financial", "0001281761", "RF", "regional"),
    Bank("Huntington", "0000049196", "HBAN", "regional"),
    Bank("KeyCorp", "0000091576", "KEY", "regional"),
    Bank("Fifth Third", "0000035527", "FITB", "regional"),
    Bank("M&T Bank", "0000036270", "MTB", "regional"),
    Bank("First Citizens BancShares", "0000798941", "FCNCA", "regional"),
    Bank("Zions Bancorporation", "0000109380", "ZION", "regional"),
    # ---- Consumer / card / auto ------------------------------------------
    Bank("Capital One", "0000927628", "COF", "card_consumer"),
    Bank("American Express", "0000004962", "AXP", "card_consumer"),
    Bank("Ally Financial", "0000040729", "ALLY", "card_consumer"),
    Bank("Synchrony Financial", "0001601712", "SYF", "card_consumer"),
    Bank(
        "Discover",
        "0001393612",
        "DFS",
        "card_consumer",
        last_filing=dt.date(2025, 5, 18),  # acquired by Capital One
    ),
    # ---- Custody / trust / brokerage --------------------------------------
    Bank("BNY Mellon", "0001390777", "BNY", "custody"),
    Bank("State Street", "0000093751", "STT", "custody"),
    Bank("Northern Trust", "0000073124", "NTRS", "custody"),
    Bank("Charles Schwab", "0000316709", "SCHW", "broker"),
    Bank("Raymond James", "0000720005", "RJF", "broker"),
    Bank("Ameriprise", "0000820027", "AMP", "broker"),
    # ---- US IHCs of foreign banks that DO file with the SEC ---------------
    Bank("Santander Holdings USA", "0000811830", "SHUSA", "regional", tier="ihc"),
    Bank("HSBC USA", "0000083246", "HSBCUSA", "regional", tier="ihc"),
    Bank(
        "MUFG Americas Holdings",
        "0001011659",
        "MUFGA",
        "regional",
        tier="inactive",
        last_filing=dt.date(2021, 12, 1),  # Union Bank sold to US Bancorp
    ),
)

# DFAST participants with no SEC periodic filings -- documented so the gap in
# the panel is explicit rather than an unexplained absence.
NON_SEC_IHCS: tuple[str, ...] = (
    "BMO Financial Corp",
    "TD Group US Holdings LLC",
    "UBS Americas Holding LLC",
    "DB USA Corporation",
    "Barclays US LLC",
    "RBC US Group Holdings LLC",
    "BNP Paribas USA, Inc.",
    "Credit Suisse Holdings (USA), Inc.",
)

# Not DFAST ($100B threshold) but common regional benchmarking comparators.
# Off by default; opt in with ``universe(include_comparators=True)``.
COMPARATORS: tuple[Bank, ...] = (
    Bank("Comerica", "0000028412", "CMA", "regional", tier="comparator"),
    Bank("Synovus Financial", "0000018349", "SNV", "regional", tier="comparator"),
    Bank("First Horizon", "0000036966", "FHN", "regional", tier="comparator"),
    Bank("Webster Financial", "0000801337", "WBS", "regional", tier="comparator"),
    Bank("Valley National", "0000714310", "VLY", "regional", tier="comparator"),
    Bank("Western Alliance", "0001212545", "WAL", "regional", tier="comparator"),
    Bank("East West Bancorp", "0001069157", "EWBC", "regional", tier="comparator"),
    Bank("Cullen/Frost", "0000039263", "CFR", "regional", tier="comparator"),
    Bank("Popular", "0000763901", "BPOP", "regional", tier="comparator"),
    Bank("Prosperity Bancshares", "0001068851", "PB", "regional", tier="comparator"),
)

# Categories whose loan disclosures are thin by construction. Useful for
# suppressing false "coverage gap" alarms in the reconciliation report.
THIN_LOAN_BOOK = ("custody", "broker")

# --------------------------------------------------------------------------
# Accounting regime
# --------------------------------------------------------------------------

# The incurred-loss -> CECL transition changes what ``acl``, ``provision``,
# ``reserve_coverage`` and ``reserve_to_nonaccrual`` *mean*, and the move from
# recorded investment to amortised cost shifts loan and nonaccrual levels
# slightly.  Splicing the two regimes into one column without saying so
# produces a discontinuity that reads as a credit event, so the panel carries
# a ``basis`` column keyed on these dates instead.
#
# This is the standard effective date for large SEC filers -- the first fiscal
# year beginning after 15 December 2019 -- not a per-filer confirmation from
# each 10-K.  It is right for every filer in this universe except Raymond
# James, whose September fiscal year puts its first CECL quarter at 2020Q4.
CECL_ADOPTION_DEFAULT = dt.date(2020, 1, 1)
CECL_ADOPTION: dict[str, dt.date] = {
    "RJF": dt.date(2020, 10, 1),
}


def cecl_adoption(ticker: str) -> dt.date:
    """First period end reported under CECL for a filer."""
    return CECL_ADOPTION.get(ticker, CECL_ADOPTION_DEFAULT)


BY_CIK = {b.cik: b for b in BANKS + COMPARATORS}
BY_TICKER = {b.ticker: b for b in BANKS + COMPARATORS}
BY_NAME = {b.name: b for b in BANKS + COMPARATORS}


def universe(
    *,
    include_inactive: bool = True,
    include_comparators: bool = False,
    categories: tuple[str, ...] | None = None,
) -> tuple[Bank, ...]:
    """Select the working bank set."""
    banks = list(BANKS)
    if include_comparators:
        banks.extend(COMPARATORS)
    if not include_inactive:
        banks = [b for b in banks if b.tier != "inactive"]
    if categories:
        banks = [b for b in banks if b.category in categories]
    return tuple(banks)


def ensure_dirs() -> None:
    for p in (RAW_FACTS, RAW_HTML, RAW_INSTANCES, OUT):
        p.mkdir(parents=True, exist_ok=True)
