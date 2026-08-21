"""HTML table fallback for disclosures that never reach XBRL.

Two distinct gaps this covers:

1. **Pre-XBRL and thinly tagged filings.**  Older 10-Ks, and small filers who
   tag only the face financial statements, leave the ACL rollforward and the
   loan-portfolio table untagged.

2. **Narrative-only disclosures.**  Criticized and classified balances,
   office-CRE exposure and leveraged-lending exposure are frequently disclosed
   only in MD&A prose and tables that carry no XBRL element at all.  No amount
   of tag hunting recovers those -- they have to be read out of the HTML.

Tables are located by matching row *labels*, not by position, because table
ordering is not stable across banks or across years for the same bank.  Parsed
values keep a ``confidence`` and the matched label so that a reconciliation
step can prefer XBRL and show what the fallback actually matched.
"""

from __future__ import annotations

import gzip
import logging
import re
import warnings
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from . import edgar, parallel, taxonomy
from .config import RAW_HTML
from .filings import Filing

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Table identification
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TableSpec:
    """How to recognise a table and what to pull out of it."""

    name: str
    # A table qualifies when at least ``min_hits`` of these appear in its text.
    signals: tuple[str, ...]
    min_hits: int = 2
    # Row-label patterns mapped to the canonical variable they populate.
    row_map: dict[str, str] = None  # type: ignore[assignment]


ACL_ROLLFORWARD = TableSpec(
    name="acl_rollforward",
    signals=(
        "allowance for credit losses",
        "beginning balance",
        "charge-offs",
        "recoveries",
        "provision",
        "ending balance",
    ),
    min_hits=3,
    # Order is load-bearing: the first pattern to match a row label wins, and
    # "net charge-offs" contains "charge-offs".  Listed the other way round --
    # as it was -- the net rule is unreachable and every net figure is read as
    # a gross one, which understates nothing and overstates charge_offs by the
    # whole recovery rate.  Same trap as the net/gross element split in
    # ``variables``, on the other side of the pipeline.
    row_map={
        r"beginning balance|balance,? (at )?beginning": "acl_beginning",
        # 'net' is not always adjacent: Zions writes 'net loan and lease
        # charge-offs', which the adjacent form leaves in the gross column.
        r"net\b.{0,30}charge.?offs": "nco",
        r"gross charge.?offs|charge.?offs": "charge_offs",
        r"recoveries": "recoveries",
        r"provision": "provision",
        r"ending balance|balance,? (at )?end": "acl_ending",
    },
)

LOAN_PORTFOLIO = TableSpec(
    name="loan_portfolio",
    signals=(
        "commercial and industrial",
        "commercial real estate",
        "residential",
        "total loans",
        "construction",
        "home equity",
        "credit card",
    ),
    min_hits=3,
    row_map={
        r"commercial real estate|commercial mortgage": "loans_cre_total",
        r"construction|land development": "loans_construction",
        r"commercial and industrial|c&i\b": "loans_ci",
        r"multifamily|multi-family": "loans_multifamily",
        r"owner.?occupied": "loans_cre_owner_occupied",
        r"non.?owner.?occupied|investor (commercial )?real estate": "loans_cre_investor",
        r"small business": "loans_small_business",
        r"residential mortgage|residential real estate|1-4 family": "loans_resi_mortgage",
        r"home equity": "loans_home_equity",
        r"credit card": "loans_credit_card",
        r"automobile|auto\b|indirect auto": "loans_auto",
        r"lease financing|leases": "loans_lease",
        r"total loans( and leases)?": "loans_total",
    },
)

CREDIT_QUALITY = TableSpec(
    name="credit_quality",
    signals=(
        "pass",
        "special mention",
        "substandard",
        "criticized",
        "classified",
        "doubtful",
        "risk rating",
        "internal risk",
    ),
    min_hits=2,
    row_map={
        r"special mention": "cq_special_mention",
        r"substandard": "cq_substandard",
        r"doubtful": "cq_doubtful",
        r"^loss$": "cq_loss",
        r"criticized": "cq_criticized",
        r"classified": "cq_classified",
        r"^pass": "cq_pass",
        r"total (criticized|classified)": "cq_criticized",
    },
)

NONPERFORMING = TableSpec(
    name="nonperforming",
    signals=(
        "nonaccrual",
        "nonperforming",
        "other real estate owned",
        "90 days",
        "past due",
    ),
    min_hits=2,
    row_map={
        r"total nonaccrual|nonaccrual loans": "nonaccrual_total",
        r"other real estate owned|oreo|foreclosed": "oreo",
        r"total non.?performing assets": "npa_total",
        r"90 days.*(past due|accruing)": "pd_dpd_90_plus",
        r"30.{0,4}89 days": "pd_dpd_30_89",
    },
)

# Labels that look like a target row but describe a different population.
# "Total loans to borrowers experiencing financial difficulty" is a
# modification disclosure, not the loan book, and matches a naive
# "total loans" pattern.
# Labels whose row must never be read at all.  Two kinds.
#
# The first is a different population, or a different unit: held-for-sale,
# unfunded, modified -- and anything stating a *ratio*, which belongs beside
# 'percent' and '%'.  Zions prints 'Ratio of net charge-offs to average loans
# and leases' in the same table as the dollar rollforward, and a percentage
# read as a dollar amount is the quietest error this file can make.
#
# The second is a label that *negates* the concept it names, which is the same
# trap ``taxonomy.LOAN_RULES`` guards three times over on the member side and
# which this matcher had no notion of.  JPMorgan's 10-K cost all three:
# "total consumer, excluding credit card loans" was read as the card book,
# "noncriticized" as criticized, and "pre-provision profit" as the provision.
EXCLUDE_LABEL = re.compile(
    r"experiencing financial difficulty|troubled debt|modification"
    r"|held for sale|unfunded|commitment|guarantee|servicing"
    r"|fair value|weighted average|per share|percent|%"
    r"|\bexcluding\b|\bexcept\b|\bnon-?criticized\b|\bpre-?provision\b"
    r"|\bratio\b",
    re.IGNORECASE,
)

# Tables whose rows carry the right *labels* against the wrong *population*.
# ``EXCLUDE_LABEL`` cannot reach these, because there is nothing wrong with the
# label: it is the table that is wrong, and every row in it.
#
# The EX-13 path is what made this necessary.  An annual report is the whole of
# MD&A, so beside the loan schedule it prints a rate/volume analysis whose rows
# are "Total loans", "Commercial real estate", "Credit card" -- the loan
# schedule's labels exactly -- against the *change in interest income* those
# books produced.  US Bancorp's 2025 report put it first, and it read total
# loans as 549 against a real book near 390,000, with the mix in proportion so
# nothing looked odd.  It is the same trap as the average/period-end pair
# ``ir_extract`` guards on the supplement side, and it survives scale
# inference: a ratio of 710 snaps to no power of a thousand, so the rows are
# dropped rather than corrected, and the filing looks like one with no
# disclosure instead of one that was misread.
EXCLUDE_TABLE = re.compile(
    # Rate/volume analysis: an income effect attributed to volume and to rate.
    r"increase \(decrease\) in interest income"
    r"|yields?\s*/\s*rates?"
    r"|rate/volume"
    # Average balances, not period-end ones.
    r"|average balance"
    # Unfunded commitments, split by the term left to run on the line.  The
    # rows are loan classes and the numbers are lines *available*, so US
    # Bancorp's card book reads 143,354 against a real 32,234.  ``EXCLUDE_LABEL``
    # already refuses "unfunded" and "commitment" as row labels and cannot help
    # here: the caption sits outside the table and every row inside it is
    # labelled like an ordinary loan schedule.
    r"|greater than one year"
    # ...and the same table named in its own caption.  Wells Fargo's "Table
    # 3.4: Unfunded Credit Commitments" reports credit card at 180,563 against
    # a book of 56,262 -- the balance plus the unused lines behind it.
    r"|unfunded|credit commitments",
    re.IGNORECASE,
)

# The CRE book split by what the property is used for, which is where office
# exposure lives when a bank discloses it at all in a filing.  It is tagged in
# no XBRL instance -- Bank of America's carries no office member and no
# property-type axis anywhere -- and it is absent from the 10-Q, so this table
# in the 10-K is the only place the number exists.
#
# Bank of America's Table 33, "Outstanding Commercial Real Estate Loans by
# Geographic Region and Property Type", gives Office at 12,447 for 2025, which
# reconciles two independent ways against the MD&A prose beside it: "Office
# loans decreased $2.6 billion, or 17 percent, during 2025" (15.0 -> 12.4) and
# "represented approximately one percent of total loans" (12.4 / 1,090).
#
# The signals are the *other* property types on purpose.  "Office" alone is a
# word that appears in every filing ("principal executive offices", "deposits
# in U.S. offices"); a table that also names industrial/warehouse, multi-family
# rental and shopping centres is unambiguously this schedule.
CRE_PROPERTY_TYPE = TableSpec(
    name="cre_property_type",
    signals=(
        "by property type",
        "industrial / warehouse",
        "industrial/warehouse",
        "multi-family rental",
        "shopping centers",
        "hotel",
        "non-residential",
        "multi-use",
    ),
    min_hits=3,
    row_map={
        # Anchored: the same table has an "Other" row and a residential block,
        # and the geographic half above it must not contribute anything.
        r"^office$|^office loans?$|^office buildings?$": "loans_office_cre",
        r"^multi.?family rental": "loans_multifamily",
        r"^total outstanding commercial real estate loans?$": "loans_cre_total",
    },
)

# Two banks publish this table and they lay it out differently, which is worth
# knowing before extending the spec.  Bank of America's columns are years, so
# the left-most number is the current one -- the rule this module already uses.
# M&T's are maturity buckets followed by a total ("Office 408 809 872 1,024 310
# $3,423 14%"), so the useful number is the total.  Both come out right, and
# both were checked row by row against the filings: BAC 2019-2025 and M&T
# 2023-2025, whose buckets sum to the total taken.  A third layout should be
# verified the same way rather than assumed.
#
# M&T's population is its "permanent finance" CRE, which is a narrower book
# than the consolidated CRE line, so its office share of loans is not strictly
# comparable with Bank of America's.

# --------------------------------------------------------------------------
# Grids: schedules whose *columns* carry the variable
# --------------------------------------------------------------------------
#
# Every ``TableSpec`` above reads a row label and takes ``_first_numeric`` --
# the left-most number, which in a comparative table is the current period.
# Two disclosures cannot be read that way, and measuring them is what showed
# it: the quantity wanted is named by the **column**, and the row names only
# which slice of the book it belongs to.
#
#   Past-due aging.  Rows are loan classes, columns are the buckets:
#   "Current | 30-89 days past due | 90 days or more past due | Nonaccruing |
#   Total".  The left-most number is *Current* -- the performing balance -- so
#   a row-label spec matching "commercial and industrial" would report a
#   bank's performing C&I book as its 30-89 day delinquency.  Right label,
#   right magnitude, wrong column, and nothing raises.
#
#   CECL vintage.  Rows are credit-quality grades under a loan-class heading,
#   columns are origination years.  Measured over the 231 cached documents,
#   633 of 711 candidate tables put the years in columns and exactly one puts
#   them in rows.
#
# So these carry a ``column_map`` as well as a ``row_map`` and are read by
# :func:`extract_from_grid`.  The row-addressed specs are untouched.


@dataclass(frozen=True)
class GridSpec:
    """A schedule addressed by column as well as by row.

    ``caption`` is required, unlike a ``TableSpec``'s signals.  A reader keyed
    on year headers will otherwise read an ordinary comparative table -- Ally's
    "December 31, | 2019 | 2018 | 2017" -- as a vintage disclosure and report
    three *reporting periods* as three origination years.  115 of the 615
    candidate tables measured are that shape.  The caption is the only thing
    that separates them, so it has to match before a table is read at all.
    """

    name: str
    signals: tuple[str, ...]
    caption: re.Pattern[str]
    row_map: dict[str, str]
    column_map: dict[str, str]
    min_hits: int = 2


# How far down the table the header band is looked for.  Headers run to three
# rows where a bank spans a group label ("Amortized cost basis by origination
# year") over the year cells beneath it.
_HEADER_ROWS = 6

# Populations that wear the aging table's own headings.  A TDR or modification
# schedule is laid out identically -- the same loan classes down the side, the
# same buckets across the top -- over the restructured book alone, so Fifth
# Third's "Table 56: Accruing and nonaccruing portfolio TDRs" reports
# commercial at 411 against a book near 60,000.  ``EXCLUDE_LABEL`` cannot see
# it, because every row label in it is an ordinary loan class.
_GRID_EXCLUDE = re.compile(
    r"troubled debt|\btdrs?\b|modificat|experiencing financial difficulty"
    r"|held for sale|purchased credit",
    re.IGNORECASE,
)

DELINQUENCY_BY_CATEGORY = GridSpec(
    name="delinquency_by_category",
    signals=(
        "past due",
        "30-89",
        "90 days or more",
        "accruing",
        "current",
        "nonaccrual",
    ),
    # "Aging", or a heading that names the past-due schedule.  A bare "past
    # due" appears in the prose above half the credit tables in a 10-K, so the
    # caption has to name the schedule rather than merely mention the concept.
    # ``and`` is deliberately not one of these.  Citizens captions a five-year
    # comparative "Table 13: Nonaccrual loans and leases, accruing and 90 days
    # or more past due **and** restructured loans", whose columns are years,
    # and it qualified on that word alone.
    caption=re.compile(
        r"age analysis|aging|past.?due (status|analysis|financing|loans)"
        r"|analysis of past.?due|delinquenc",
        re.IGNORECASE,
    ),
    # Categories follow ``callrpt_db.mdrm.RCN_CATEGORIES`` so that the two
    # panels' delinquency columns line up.  They are spelled out rather than
    # imported: ``callrpt_db`` imports ``bankqtr_db``, and the reverse would
    # close the loop.
    # Order is load-bearing here for the same reason it is in
    # ``ACL_ROLLFORWARD``: the first pattern to match a row label wins, and
    # "total loans and leases" contains "leases".  Listed the other way round
    # -- as it was -- Bank of America's whole-book 30-89 figure of 5,555 was
    # reported as its *lease* delinquency, against a lease book a fraction of
    # that size.  The whole-book row is therefore matched first, and the lease
    # pattern is anchored so it cannot reach inside a total.
    row_map={
        r"^total loans( and leases)?\b": "total",
        r"commercial real estate|commercial mortgage": "cre_nonfarm_nonres",
        r"construction|land development": "construction",
        r"commercial and industrial|c&i\b": "ci",
        r"multifamily|multi-family": "multifamily",
        r"residential mortgage|residential real estate|1-4 family": "resi_mortgage",
        r"home equity": "home_equity",
        r"credit card": "credit_card",
        r"automobile|auto\b|indirect auto": "auto",
        r"^lease financing|^leases\b|^direct financing lease": "lease",
    },
    # 30-59 and 60-89 are separate columns at some filers and one 30-89 column
    # at others.  Both map here and are summed, which ``_grid_columns`` does by
    # header *text*, so a cell spanned across three columns contributes once
    # rather than three times.
    #
    # Only the explicit day ranges match.  A looser "30 days past due" was
    # tried and is wrong in the one way that matters: Bank of America's aging
    # table heads its performing column "Total current or less than 30 days
    # past due", which contains the phrase, so its whole $225bn residential
    # book was being reported as 30-89 day delinquency against a real 9,136.
    # The rollup column beside it, "Total past due 30 days or more", is left
    # unmatched on purpose -- it is the sum of the three buckets and adding it
    # would double the row.
    column_map={
        r"30\s*[-–to]+\s*89|30\s*[-–to]+\s*59|60\s*[-–to]+\s*89": "dpd_30_89",
        r"90 days? or (more|greater)|90\+|greater than 90 days"
        r"|90 days? and over": "dpd_90_plus",
    },
    min_hits=3,
)

VINTAGE_ANALYSIS = GridSpec(
    name="vintage_analysis",
    signals=(
        "origination year",
        "revolving",
        "pass",
        "special mention",
        "substandard",
        "amortized cost",
    ),
    caption=re.compile(
        r"origination year|year of origination|\bvintage\b", re.IGNORECASE
    ),
    # Only the grade rows are read, and the totals are built from them.  These
    # tables state a *nested* set of totals: "Total", "Total commercial" and
    # "Total commercial and industrial" all appear in one table, so summing the
    # rows labelled "total" multiplies the book by two to four.  It is the same
    # hazard ``mdrm.ItemSpec`` documents on the Call Report side, where a filer
    # reports a rollup beside the detail that sums to it.
    #
    # The grades do not nest.  Each appears once per loan class, so summing
    # them across the classes is a partition rather than a double count -- and
    # the total built from them and the criticised part of it then describe the
    # same population by construction, which a total read from a "total" row
    # and a criticised figure summed from grade rows would not.
    #
    # Criticised is taken as everything that is not pass, rather than as a
    # fixed list of grade names, because the ladder is not uniform: Truist
    # writes pass / special mention / substandard / nonperforming where others
    # write doubtful and loss.  Total less pass is the one definition every
    # filer's own table supports.
    row_map={
        r"^pass\b": "pass",
        r"^(special mention|substandard|doubtful|loss|criticized|classified"
        r"|nonperforming|non-performing|nonaccrual)\b": "criticized",
    },
    column_map={r"^(20[0-2]\d)(\s*\(\d+\))?$": r"\1"},
    min_hits=3,
)

GRID_SPECS: tuple[GridSpec, ...] = (DELINQUENCY_BY_CATEGORY, VINTAGE_ANALYSIS)

# How many origination years are kept, most recent first.  A vintage table runs
# five discrete years and then a "Prior" column, which is not a year and is not
# comparable across filings, so it is left out.
_VINTAGE_YEARS = 5

# Every year ``VINTAGE_ANALYSIS``'s column pattern can match, and so the
# complete set of ``vintage_*`` columns the spec can emit.  Tied to the
# ``^(20[0-2]\d)$`` in that pattern: widen both or neither.
VINTAGE_YEAR_RANGE = range(2000, 2030)


def grid_columns() -> tuple[str, ...]:
    """Every column the grid specs can populate, for the reconcile allowlist.

    ``reconcile.HTML_NEW_COLUMNS`` names the columns the HTML path is allowed
    to *create*, and these are all of that kind: no filing tags a past-due
    bucket by loan class or an origination-year balance, so there is no XBRL
    column waiting for them -- the same reason ``loans_office_cre`` is on that
    list.

    Derived here rather than transcribed there so the allowlist cannot drift
    out of step with the specs.  A column missing from it is not an error that
    raises; it is a reading that is parsed, scaled, and then silently dropped
    at the merge, which is the kind of gap this package tries not to have.

    The whole-book row is deliberately absent.  It fills ``pd_dpd_30_89`` and
    ``pd_dpd_90_plus``, which the panel already carries from XBRL, so it needs
    no permission to be created.
    """
    names: list[str] = []
    for category in dict.fromkeys(DELINQUENCY_BY_CATEGORY.row_map.values()):
        if category == "total":
            continue
        names.append(f"dpd_30_89_{category}")
        names.append(f"dpd_90_plus_{category}")
    for year in VINTAGE_YEAR_RANGE:
        names.append(f"vintage_total_{year}")
        names.append(f"vintage_criticized_{year}")
    return tuple(names)


SPECS: tuple[TableSpec, ...] = (
    ACL_ROLLFORWARD,
    LOAN_PORTFOLIO,
    CREDIT_QUALITY,
    NONPERFORMING,
    CRE_PROPERTY_TYPE,
)


# --------------------------------------------------------------------------
# Fetching and parsing
# --------------------------------------------------------------------------


def _html_cache_path(filing: Filing) -> Path:
    return RAW_HTML / f"{filing.ticker}_{filing.accn_nodash}.html.gz"


def ensure_cached(filing: Filing) -> bool:
    """Put a filing's primary document on disk if it is not already there.

    Same contract as ``instance.ensure_cached`` and for the same two reasons:
    it keeps fetching serial and behind the one rate limiter while parsing runs
    in a pool, and it makes a rebuild cost nothing.  Re-parsing 210 filings was
    dominated by re-downloading them, because this cache was declared in
    ``config.RAW_HTML`` and never written.
    """
    path = _html_cache_path(filing)
    if path.exists():
        return True
    content = _download_primary_document(filing)
    if content is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wb") as fh:
        fh.write(content)
    tmp.replace(path)
    return True


def fetch_primary_document(filing: Filing) -> bytes | None:
    """Primary 10-K/10-Q HTML as **bytes**, from cache where one exists."""
    path = _html_cache_path(filing)
    if path.exists():
        with gzip.open(path, "rb") as fh:
            return fh.read()
    return _download_primary_document(filing)


def _download_primary_document(filing: Filing) -> bytes | None:
    """Fetch the primary document, resolving it via the index when needed.

    Bytes rather than text on purpose: inline-XBRL filings begin with an XML
    declaration, and lxml refuses to parse a ``str`` that carries an encoding
    declaration.  Handing it the raw bytes keeps the declaration meaningful
    and avoids a double decode.
    """
    if filing.primary_doc:
        try:
            return edgar.get(filing.primary_doc_url).content
        except Exception as exc:  # noqa: BLE001
            log.warning("primary doc fetch failed %s: %s", filing.accn, exc)

    try:
        index = edgar.get(f"{filing.folder_url}/index.json").json()
    except Exception as exc:  # noqa: BLE001
        log.warning("index fetch failed %s: %s", filing.accn, exc)
        return None
    names = [
        i["name"] for i in index["directory"]["item"] if i["name"].endswith(".htm")
    ]
    main = [n for n in names if not re.search(r"ex(hibit)?[-_]?\d", n, re.IGNORECASE)]
    if not main:
        return None
    try:
        return edgar.get(f"{filing.folder_url}/{main[0]}").content
    except Exception as exc:  # noqa: BLE001
        log.warning("fallback doc fetch failed %s: %s", filing.accn, exc)
        return None


# --------------------------------------------------------------------------
# The financial statements are not always in the 10-K
# --------------------------------------------------------------------------
#
# Three banks -- Wells Fargo, US Bancorp and BNY Mellon -- file a 10-K whose
# Item 8 reads "Information in response to this Item can be found in the 2025
# Annual Report to Shareholders", and file the annual report itself as an
# **EX-13** exhibit in the same submission.  The primary document is a
# 1.5 MB wrapper of cross-references; the statements are in an 11.6 MB exhibit
# beside it.  All 21 of the cached filings that yield no table at all are these
# three banks, and the reason is not that their tables are hard to find -- it
# is that the document being parsed does not contain a single occurrence of
# "allowance", "nonaccrual" or "commercial and industrial".
#
# So this is not a parsing problem and no parser can fix it.  The fallback is
# to fetch the right document.
#
# It runs as a genuine second pass, on the filings the first pass got nothing
# from, for two reasons.  A gate on the *parsed result* is exact where a text
# heuristic is not -- American Express names none of the phrases a wrapper also
# lacks, and scores identically to one while parsing perfectly well -- and
# resolving an exhibit costs an extra index fetch per filing, which is worth
# paying on 21 filings and not on 210.
_EXHIBIT_TYPE = re.compile(r"^EX-13(\.\d+)?$", re.IGNORECASE)


def _exhibit_cache_path(filing: Filing) -> Path:
    return RAW_HTML / f"{filing.ticker}_{filing.accn_nodash}.ex13.html.gz"


def financial_statement_exhibit(filing: Filing) -> str | None:
    """The EX-13 attachment's filename, if this filing has one.

    EX-13 *is* the annual report to shareholders, so its presence is the
    disclosure that the statements were incorporated by reference rather than
    printed in the 10-K.  A filing that carries its own statements has no such
    exhibit, which makes this self-limiting: nothing changes for the filings
    that already parse.
    """
    from .ir import exhibit_index

    for name, ex_type, _ in exhibit_index(filing.folder_url, filing.accn):
        if _EXHIBIT_TYPE.match(ex_type) and name.endswith((".htm", ".html")):
            return name
    return None


def ensure_exhibit_cached(filing: Filing) -> bool:
    """Put a filing's EX-13 on disk, next to the primary document."""
    path = _exhibit_cache_path(filing)
    if path.exists():
        return True
    name = financial_statement_exhibit(filing)
    if name is None:
        return False
    try:
        content = edgar.get(f"{filing.folder_url}/{name}").content
    except Exception as exc:  # noqa: BLE001
        log.warning("ex-13 fetch failed %s: %s", filing.accn, exc)
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wb") as fh:
        fh.write(content)
    tmp.replace(path)
    return True


# --------------------------------------------------------------------------
# Tables, with the caption printed above them
# --------------------------------------------------------------------------
#
# The caption is load-bearing rather than decoration, and this module learned
# it the same way ``ir_extract`` did.  A bank's allowance rollforward, its
# charge-off table, its nonaccrual schedule and its loan schedule all list the
# loan classes down the side, so on body text alone they score identically --
# and an EX-13 is a whole annual report, so it prints all of them plus the
# segment, average-balance, commitment and total-exposure variants of each.
# Wells Fargo's alone offers "total loans" at 927,491, 986,167 and 223,399:
# the book, the book plus its unused commitments, and one segment of it.
#
# No rule over the numbers can separate those -- taking the first is a property
# of the typesetting, taking the largest reliably finds the commitment table --
# but the text above each one names it outright.  Tables are therefore walked
# one at a time through lxml, which keeps each paired with its own preceding
# text; a document lxml cannot parse falls back to reading them in bulk without
# context, exactly as before.
Table = tuple[pd.DataFrame, str]

# How much of the text above a table is kept as its heading.  ``_CAPTION_CHARS``
# is the stopping rule -- collecting ends as soon as this much text has been
# gathered -- and ``_CONTEXT_CHARS`` trims whatever that last chunk dragged in.
_CAPTION_CHARS = 60
_CONTEXT_CHARS = 200


def _html_tables_with_context(payload: bytes) -> list[Table]:
    try:
        from lxml import html as lxml_html
    except ImportError:  # pragma: no cover
        return []
    try:
        tree = lxml_html.fromstring(payload)
    except Exception as exc:  # noqa: BLE001 - malformed markup is the norm
        log.debug("lxml parse failed: %s", exc)
        return []

    out: list[Table] = []
    for element in tree.iter("table"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.read_html(
                    BytesIO(lxml_html.tostring(element)), flavor="lxml"
                )
        except Exception as exc:  # noqa: BLE001 - a rejected table is skipped
            log.debug("table read_html failed: %s", exc)
            continue
        if not parsed:
            continue
        out.append((parsed[0], _text_before(element)))
    return out


def _text_before(element: Any) -> str:
    """The caption printed immediately above an element.

    Deliberately short-sighted.  Reading back a fixed number of characters
    walks straight off the top of the section and picks up the footnotes of the
    page before -- which is how Wells Fargo's "COMMERCIAL LOAN PORTFOLIO" table
    came to be captioned with the previous page's past-due commentary and was
    classified as a nonperforming schedule.  Only the nearest heading is taken:
    enough to name the schedule, not enough to reach the one before it.
    """
    chunks: list[str] = []
    total = 0
    node = element
    while node is not None and total < _CAPTION_CHARS:
        previous = node.getprevious()
        while previous is not None and total < _CAPTION_CHARS:
            # Skip other tables (their contents are not this table's heading)
            # and comments/processing instructions, whose tag is a callable
            # rather than a string and which carry no text content.
            if isinstance(previous.tag, str) and previous.tag != "table":
                text = " ".join(previous.text_content().split())
                if text:
                    chunks.insert(0, text)
                    total += len(text)
            previous = previous.getprevious()
        node = node.getparent()
    return " ".join(chunks)[-_CONTEXT_CHARS:]


def read_tables(html: bytes | str) -> list[Table]:
    """All tables in a filing, each paired with the caption printed above it.

    Walked one at a time through lxml so the pairing survives; the bulk
    ``read_html`` path is kept as a backstop for documents lxml rejects, and
    yields tables with an empty caption rather than none at all.

    Failures are logged rather than swallowed: a filing that silently yields
    zero tables looks identical to a filing with no relevant disclosure, and
    that is exactly the kind of gap this module exists to close.
    """
    raw = html if isinstance(html, bytes) else html.encode("utf-8", "ignore")
    with_context = _html_tables_with_context(raw)
    if with_context:
        return with_context

    payload: Any = BytesIO(html) if isinstance(html, bytes) else StringIO(html)
    errors: list[str] = []
    for flavor in ("lxml", "bs4"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tables = pd.read_html(payload, flavor=flavor)
        except Exception as exc:  # noqa: BLE001 - malformed markup is the norm
            errors.append(f"{flavor}: {type(exc).__name__} {exc}")
            payload = BytesIO(html) if isinstance(html, bytes) else StringIO(html)
            continue
        if tables:
            return [(t, "") for t in tables]
        payload = BytesIO(html) if isinstance(html, bytes) else StringIO(html)
    if errors:
        log.warning("read_html found no tables (%s)", "; ".join(errors)[:300])
    return []


def _table_text(df: pd.DataFrame) -> str:
    return " ".join(str(v).lower() for v in df.head(60).to_numpy().ravel())


# How much more a signal counts for when it appears in the table's own caption
# rather than anywhere in its body.  Same weight, and for the same reason, as
# ``ir_extract._HEADER_WEIGHT``.
_HEADER_WEIGHT = 3

# A segment schedule repeats the consolidated row labels over a fraction of the
# book, and an annual report prints it before the consolidated one.  Wells
# Fargo's reports total loans of 223,399 and C&I of 167,207 for one segment
# against 927,491 and 335,405 consolidated; taking the first table wins handed
# the panel the segment figure.  The panel is consolidated, so these are
# skipped -- the same rule, on the same evidence, as ``ir_extract``'s.
_SEGMENT_TABLE = re.compile(
    # Not a bare "segment": a *portfolio* segment is a loan class, not a
    # business unit, and it is what the consolidated schedule is named after
    # -- "Table 16: Total Loans Outstanding by Portfolio Segment and Class of
    # Financing Receivable".  Excluding on the bare word threw away the very
    # table this module most wants and left C&I reading 157.
    r"reportable segment|operating segment|line of business|business line"
    # A segment is not always called one.  Wells Fargo captions these
    # "Table 9d: Commercial Banking - Balance Sheet", naming the segment
    # and never the word, so the generic pattern above walks straight past
    # them.  ``ir_extract`` carries the same list for the same filer.
    r"|commercial banking|corporate and investment banking"
    r"|consumer banking and lending|banking and lending\b"
    r"|wealth and investment|\bwim\b",
    re.IGNORECASE,
)


def classify_table(df: pd.DataFrame, context: str = "") -> TableSpec | None:
    """Which schedule a table is, weighting its caption above its body.

    Body text alone cannot tell these apart: every schedule in a credit
    disclosure lists the loan classes down the side.  What names the table is
    the text printed above it, so caption matches count toward the threshold
    *and* are weighted well above body matches.
    """
    body = _table_text(df)
    header = f"{context} {_table_text(df.head(4))}".lower()
    if EXCLUDE_TABLE.search(body) or EXCLUDE_TABLE.search(header):
        return None
    if _SEGMENT_TABLE.search(header):
        return None

    best: tuple[int, TableSpec] | None = None
    for spec in SPECS:
        body_hits = sum(1 for s in spec.signals if s in body)
        header_hits = sum(1 for s in spec.signals if s in header)
        if body_hits + header_hits < spec.min_hits:
            continue
        score = body_hits + header_hits + _HEADER_WEIGHT * header_hits
        if best is None or score > best[0]:
            best = (score, spec)
    return best[1] if best else None


_NUM = re.compile(r"^\(?\$?\s*-?[\d,]+(\.\d+)?\)?$")


def _to_number(value: Any) -> float | None:
    """Parse a financial-statement cell, honouring parenthesised negatives."""
    if value is None:
        return None
    # The trailing strip is load-bearing.  A cell reading '$ (3,573)' leaves a
    # space where the currency marker was, so ``startswith("(")`` is False and
    # a balanced, unambiguous negative parses as None -- a whole row dropped
    # for a space.
    s = (
        str(value)
        .strip()
        .replace("$", "")
        .replace(",", "")
        .replace("—", "")
        .strip()
    )
    if not s or s in {"-", "--", "nan", "None"}:
        return None
    negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        num = float(s)
    except ValueError:
        return None
    return -num if negative else num


# A filing typesets a negative as separate cells -- the opening parenthesis
# glued to the digits, then the closer on its own -- so the minus sign lands in
# a different ``<td>`` from the number it belongs to and ``read_html``
# faithfully reports what the markup says.  Ally writes a charge-off of (1,423)
# as ``('(1,423', '(1,423', ')')``: the fragment, ``read_html``'s duplicate of
# it for the spanned column, and the closer alone.
#
# ``_to_number`` only treats parentheses as a minus when it sees *both* halves,
# so the fragment reads as **positive** 1,423.  That is a sign flip rather than
# a parse failure, which makes it the quiet kind: nothing raises, nothing comes
# out null, and a $1.4bn charge-off enters the panel as a $1.4bn recovery.
# 4,799 cells across the 210 cached filings are split this way, in 26 banks.
#
# Reassembly is deliberately narrow.  A fragment is joined to its closer only
# when everything between the two is padding -- an empty cell, a currency
# marker, or ``read_html``'s own duplicate of the fragment.  Anything else
# stops the scan and the fragment is left alone, so a number can never absorb a
# parenthesis belonging to a different column.  The 1.2% of fragments with no
# closer to be found keep exactly today's behaviour.
_OPEN_FRAGMENT = re.compile(r"^\$?\s*\(\s*[\d,]+(\.\d+)?$")
_PADDING = frozenset({"", "nan", "none", "$", "—", "-", "--"})


def _merge_split_parens(values: Any) -> list[str]:
    """Reassemble parenthesised negatives that the markup split across cells."""
    cells = ["" if v is None else str(v).strip() for v in values]
    for i, cell in enumerate(cells):
        if not _OPEN_FRAGMENT.match(cell):
            continue
        duplicates: list[int] = []
        for j in range(i + 1, len(cells)):
            nxt = cells[j]
            if nxt.lower() in _PADDING:
                continue
            if nxt == cell:
                duplicates.append(j)
                continue
            if nxt == ")" or nxt == cell + ")":
                for k in (i, *duplicates):
                    cells[k] = cell + ")"
                # Drop the bare closer: it carries nothing, and ``_row_label``
                # would otherwise be free to read it as a row label.
                if nxt == ")":
                    cells[j] = ""
            break
    return cells


# Balances that cannot be negative in fact, only in presentation.  An
# allowance rollforward writes the closing allowance and the charge-offs as
# deductions -- "(14,407)", "(876)" -- and carrying that sign into the panel
# makes a bank look like it holds a negative reserve or a negative loan book.
# Flows are left alone: a provision release and a net recovery are both real
# and both genuinely negative.
#
# ``ir_extract`` had this rule and this module did not, which went unnoticed
# only because the split-parenthesis defect above was cancelling it out --
# reading '(1,423' as +1,423 accidentally produced the magnitude the panel
# wanted.  Reassembling the parentheses without also applying the convention
# turns 33 impossible negative balances into 42.  The two fixes belong
# together, and the rule now lives here, on the lower layer, so both readers
# share one definition of it rather than one having a copy.
# ``dpd_`` and ``vintage_`` join the list for the same reason the rest are on
# it: both are balances, and a vintage grid does print one as a deduction --
# Truist's "Other" column carries (579) and (69) beside the year columns.
_NON_NEGATIVE_PREFIXES = (
    "loans_",
    "acl_",
    "nonaccrual_",
    "cq_",
    "pd_",
    "dpd_",
    "vintage_",
)
_NON_NEGATIVE_EXACT = ("oreo", "npa_total")


def _signed(variable: str, value: float) -> float:
    if variable in _NON_NEGATIVE_EXACT or variable.startswith(_NON_NEGATIVE_PREFIXES):
        return abs(float(value))
    return float(value)


def _row_label(row: Any) -> str:
    for value in row:
        s = str(value).strip()
        if s and s.lower() != "nan" and not _NUM.match(s):
            return s.lower()
    return ""


def _first_numeric(row: Any) -> float | None:
    """Left-most numeric cell -- the current period in a comparative table."""
    for value in row:
        num = _to_number(value)
        if num is not None:
            return num
    return None


def extract_from_table(
    df: pd.DataFrame, spec: TableSpec, filing: Filing
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not spec.row_map:
        return rows
    compiled = [(re.compile(p, re.IGNORECASE), var) for p, var in spec.row_map.items()]

    for _, row in df.iterrows():
        cells = _merge_split_parens(row)
        label = _row_label(cells)
        if not label or EXCLUDE_LABEL.search(label):
            continue
        for pattern, variable in compiled:
            if not pattern.search(label):
                continue
            value = _first_numeric(cells)
            if value is None:
                continue
            rows.append(
                {
                    "bank": filing.bank,
                    "ticker": filing.ticker,
                    "cik": filing.cik,
                    "accn": filing.accn,
                    "form": filing.form,
                    "period": filing.report_date,
                    "filed": filing.filing_date,
                    "table": spec.name,
                    "variable": variable,
                    "label": label[:120],
                    "value": _signed(variable, value),
                    "source": "html",
                    # read_html cannot recover the table's scale header, so the
                    # magnitude is unresolved until reconciliation sees it
                    # alongside the XBRL value.
                    "scale_resolved": False,
                    "confidence": 0.6,
                }
            )
            break
    return rows


def _norm_cell(value: Any) -> str:
    return " ".join(str(value).lower().split())


# Header cells that name a population rather than a bucket.  The performing
# column is the dangerous one: it sits in the same table, carries the same row
# labels, and is two orders of magnitude larger than the bucket beside it, so
# reading it as a delinquency is the quiet kind of wrong.  A percentage column
# is refused here for the same reason ``EXCLUDE_LABEL`` refuses one on the row
# side -- a rate read as a balance raises nothing.
_GRID_COLUMN_EXCLUDE = re.compile(
    r"\bcurrent\b|outstanding|fair value|percent|%|\bratio\b|\btotal loans?\b",
    re.IGNORECASE,
)


def _grid_columns(df: pd.DataFrame, spec: GridSpec) -> dict[str, list[list[int]]]:
    """Column key -> one group of column indices per distinct header text.

    Grouping by the header's own text is what makes the two shapes of the
    past-due schedule read alike.  ``read_html`` repeats a cell across every
    column it spans, so "30-89 days past due" spanning three columns arrives as
    three identical headers over one value; grouped by text that contributes
    once.  A filer that instead prints "30-59 days" and "60-89 days" as two
    separate columns produces two *different* texts, so those are summed.
    Taking the left-most match would undercount the second filer and taking
    every match would treble-count the first.
    """
    compiled = [
        (re.compile(pattern, re.IGNORECASE), key)
        for pattern, key in spec.column_map.items()
    ]
    found: dict[str, dict[str, list[int]]] = {}
    for r in range(min(_HEADER_ROWS, df.shape[0])):
        for c in range(df.shape[1]):
            cell = _norm_cell(df.iat[r, c])
            if not cell or cell == "nan" or _GRID_COLUMN_EXCLUDE.search(cell):
                continue
            for pattern, key in compiled:
                match = pattern.search(cell)
                if match is None:
                    continue
                # ``expand`` resolves a template like ``\1`` against the match,
                # which is how a vintage column names itself after the year it
                # carries; a literal key passes through unchanged.
                found.setdefault(match.expand(key), {}).setdefault(cell, []).append(c)
                break
    return {
        key: groups
        for key, texts in found.items()
        if (
            groups := [
                sorted(set(idx))
                for text, idx in texts.items()
                if not _is_banner(text, idx, df.shape[1])
            ]
        )
    }


# A cell that names a column, and a cell that names the whole table, are the
# same kind of object to ``read_html``: it repeats a spanned cell across every
# column the span covers.  Citizens' "Table 13: Nonaccrual loans and leases,
# accruing and 90 days or more past due and restructured loans and leases" is
# one cell spanned over all thirty columns, and it matched the 90-day pattern
# -- mapping that bucket onto the entire width of the table, so every row
# returned whichever number happened to come first in it.
#
# Two things separate a banner from a header, and a banner trips both: it
# covers most of the table, and it reads like a sentence rather than a label.
_BANNER_SHARE = 0.5
_BANNER_COLUMNS = 8
_BANNER_CHARS = 60


def _is_banner(text: str, columns: list[int], width: int) -> bool:
    spans = len(set(columns))
    if spans >= _BANNER_COLUMNS and spans > width * _BANNER_SHARE:
        return True
    return len(text) > _BANNER_CHARS


def _grid_value(cells: list[str], groups: list[list[int]]) -> float | None:
    """Sum one value per header group, taking the first number in each."""
    total = 0.0
    seen = False
    for group in groups:
        for c in group:
            if c >= len(cells):
                continue
            value = _to_number(cells[c])
            if value is not None:
                total += value
                seen = True
                break
    return total if seen else None


def classify_grid(df: pd.DataFrame, context: str = "") -> GridSpec | None:
    """Which grid a table is, or ``None``.

    Stricter than :func:`classify_table` in one way that matters: the caption
    must match.  Signals alone cannot separate a vintage disclosure from an
    ordinary comparative table whose columns are reporting years, and a column
    reader that gets that wrong reports reporting periods as origination years.
    """
    body = _table_text(df)
    header = f"{context} {_table_text(df.head(4))}".lower()
    if EXCLUDE_TABLE.search(body) or EXCLUDE_TABLE.search(header):
        return None
    if _SEGMENT_TABLE.search(header) or _GRID_EXCLUDE.search(header):
        return None

    best: tuple[int, GridSpec] | None = None
    for spec in GRID_SPECS:
        if not spec.caption.search(header):
            continue
        hits = sum(1 for s in spec.signals if s in body or s in header)
        if hits < spec.min_hits:
            continue
        if best is None or hits > best[0]:
            best = (hits, spec)
    return best[1] if best else None


# A filing prints the same grid twice: once as of the period it reports, and
# again as of the prior year end for comparison.  Both carry the same headers
# and the same row labels, so both read, and the two disagree in a way no rule
# over the numbers can settle -- an older vintage has amortised *down*, so
# Truist's 2017 origination year is 780 in the 2020 table and 590 in the 2021
# one.  "Largest wins", which settles every other contest in this module,
# picks the stale one here, because the stale one is genuinely larger.
#
# The header says which is which.  A vintage or aging grid is headed with a
# single date -- "December 31, 2021" -- and the bare four-digit cells beside
# it are origination years, not dates, so a date pattern separates them.  When
# the band names a date at all, it has to name the one the filing reports.
_HEADER_DATE = re.compile(
    r"(?:january|february|march|april|may|june|july|august|september|october"
    r"|november|december)\s+\d{1,2},?\s*(\d{4})",
    re.IGNORECASE,
)


def _grid_period_matches(df: pd.DataFrame, context: str, filing: Filing) -> bool:
    """Is this grid stated as of the period the filing reports?

    The header band is read as one string rather than cell by cell, because a
    filing may typeset "December 31," and "2021" into adjacent cells and
    neither half carries a date on its own.

    A grid whose band names no date at all is kept.  Some filers put the date
    only in the caption above, and refusing those would throw away the
    disclosure to avoid a comparative that may not be there.
    """
    if filing.report_date is None:
        return True
    band = " ".join(
        _norm_cell(df.iat[r, c])
        for r in range(min(_HEADER_ROWS, df.shape[0]))
        for c in range(df.shape[1])
    )
    years = {int(y) for y in _HEADER_DATE.findall(f"{context} {band}")}
    return not years or filing.report_date.year in years


def _grid_row(
    filing: Filing, spec: GridSpec, variable: str, label: str, value: float
) -> dict[str, Any]:
    return {
        "bank": filing.bank,
        "ticker": filing.ticker,
        "cik": filing.cik,
        "accn": filing.accn,
        "form": filing.form,
        "period": filing.report_date,
        "filed": filing.filing_date,
        "table": spec.name,
        "variable": variable,
        "label": label[:120],
        "value": _signed(variable, value),
        "source": "html",
        "scale_resolved": False,
        # Below a row-addressed reading.  A grid depends on a header band as
        # well as a row label, so there is one more way for it to be wrong, and
        # ``reconcile`` should prefer an ordinary table's answer to this one.
        "confidence": 0.5,
    }


def _extract_delinquency(
    df: pd.DataFrame, spec: GridSpec, filing: Filing, columns: dict[str, list[list[int]]]
) -> list[dict[str, Any]]:
    """One reading per loan class per bucket, addressed by row *and* column."""
    compiled = [(re.compile(p, re.IGNORECASE), key) for p, key in spec.row_map.items()]
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        cells = _merge_split_parens(row)
        label = _row_label(cells)
        if not label or EXCLUDE_LABEL.search(label):
            continue
        category = next((k for p, k in compiled if p.search(label)), None)
        if category is None:
            continue
        for bucket, groups in columns.items():
            value = _grid_value(cells, groups)
            if value is None:
                continue
            # The whole-book row fills the totals the panel already carries
            # rather than inventing a "_total" category beside them.
            variable = f"pd_{bucket}" if category == "total" else f"{bucket}_{category}"
            rows.append(_grid_row(filing, spec, variable, label, value))
    return rows


def _extract_vintage(
    df: pd.DataFrame, spec: GridSpec, filing: Filing, columns: dict[str, list[list[int]]]
) -> list[dict[str, Any]]:
    """Origination-year totals, built from the grade rows rather than read.

    See ``VINTAGE_ANALYSIS`` for why the rows labelled "total" are not summed:
    they nest, and adding them multiplies the book.  The grades partition each
    loan class, so accumulating them down the table gives the year's balance
    over every class the table covers, and the criticised share of it is the
    same population less pass.
    """
    years = sorted((k for k in columns if k.isdigit()), reverse=True)[:_VINTAGE_YEARS]
    if len(years) < 2:
        return []
    compiled = [(re.compile(p, re.IGNORECASE), key) for p, key in spec.row_map.items()]

    totals = {y: {"pass": 0.0, "criticized": 0.0} for y in years}
    grade_rows = 0
    for _, row in df.iterrows():
        cells = _merge_split_parens(row)
        label = _row_label(cells)
        if not label or EXCLUDE_LABEL.search(label):
            continue
        grade = next((k for p, k in compiled if p.search(label)), None)
        if grade is None:
            continue
        matched = False
        for year in years:
            value = _grid_value(cells, columns[year])
            if value is None:
                continue
            # A balance, so a presentation minus is not a negative book -- the
            # same convention ``_signed`` applies to the row-addressed specs.
            totals[year][grade] += abs(value)
            matched = True
        grade_rows += matched

    # One grade row is a stray label rather than a risk ladder.
    if grade_rows < 2:
        return []

    rows: list[dict[str, Any]] = []
    for year in years:
        book = totals[year]["pass"] + totals[year]["criticized"]
        if book <= 0:
            continue
        label = f"origination year {year}, {grade_rows} grade rows"
        rows.append(_grid_row(filing, spec, f"vintage_total_{year}", label, book))
        if totals[year]["criticized"] > 0:
            rows.append(
                _grid_row(
                    filing,
                    spec,
                    f"vintage_criticized_{year}",
                    label,
                    totals[year]["criticized"],
                )
            )
    return rows


_GRID_READERS = {
    DELINQUENCY_BY_CATEGORY.name: _extract_delinquency,
    VINTAGE_ANALYSIS.name: _extract_vintage,
}


def extract_from_grid(
    df: pd.DataFrame, spec: GridSpec, filing: Filing, context: str = ""
) -> list[dict[str, Any]]:
    if not _grid_period_matches(df, context, filing):
        return []
    columns = _grid_columns(df, spec)
    if not columns:
        return []
    return _GRID_READERS[spec.name](df, spec, filing, columns)


HTML_SCHEMA: dict[str, Any] = {
    "bank": pl.String,
    "ticker": pl.String,
    "cik": pl.String,
    "accn": pl.String,
    "form": pl.String,
    "period": pl.Date,
    "filed": pl.Date,
    "table": pl.String,
    "variable": pl.String,
    "label": pl.String,
    "value": pl.Float64,
    "source": pl.String,
    "scale_resolved": pl.Boolean,
    "confidence": pl.Float64,
}


def _extract_document(filing: Filing, html: bytes | None) -> pl.DataFrame:
    """Parse one document's tables into a long frame."""
    if not html:
        return pl.DataFrame(schema=HTML_SCHEMA)

    rows: list[dict[str, Any]] = []
    for table, context in read_tables(html):
        if table.empty or table.shape[1] < 2:
            continue
        # The two passes are independent, not alternatives.  A past-due aging
        # table classifies as ``nonperforming`` on its signals and yields
        # nothing there -- its rows are loan classes, and that spec matches on
        # bucket labels -- while the grid reader gets the whole schedule out of
        # it.  Running both costs one extra classification and lets a table
        # answer whichever reader can address it.
        spec = classify_table(table, context)
        if spec is not None:
            rows.extend(extract_from_table(table, spec, filing))
        grid = classify_grid(table, context)
        if grid is not None:
            rows.extend(extract_from_grid(table, grid, filing, context))

    if not rows:
        return pl.DataFrame(schema=HTML_SCHEMA)

    df = pl.DataFrame(rows, schema_overrides=HTML_SCHEMA, infer_schema_length=None)
    # A label can match in several tables, and on an EX-13 -- a whole annual
    # report rather than a single filing -- it matches in dozens.  Ties went to
    # whichever table came first in the document, which is a property of the
    # typesetting and not of the disclosure: Wells Fargo's 2025 report offers
    # "total loans" at 927,491 and again at 223,399, the second being one
    # segment of the first, and document order picks between them arbitrarily.
    #
    # Largest wins, which is ``ir_extract``'s rule for the same problem and for
    # the same reason: every sub-schedule is a subset of the consolidated
    # figure, so the consolidated one is the largest.  It also disposes of the
    # misreadings that survive classification, which are almost always a
    # fragment of the real number rather than a multiple of it.
    return (
        df.with_columns(pl.col("value").abs().alias("_size"))
        .sort(["confidence", "_size"], descending=True)
        .unique(
            subset=["ticker", "period", "variable", "table"],
            keep="first",
            maintain_order=True,
        )
        .drop("_size")
    )


def extract_filing(filing: Filing) -> pl.DataFrame:
    """Parse one filing's primary document."""
    return _extract_document(filing, fetch_primary_document(filing))


def extract_exhibit(filing: Filing) -> pl.DataFrame:
    """Parse one filing's EX-13, for filings whose 10-K carries no statements.

    A module-level function rather than a partial because the pool spawns on
    Windows and a closure cannot cross that boundary.
    """
    path = _exhibit_cache_path(filing)
    if not path.exists():
        return pl.DataFrame(schema=HTML_SCHEMA)
    with gzip.open(path, "rb") as fh:
        return _extract_document(filing, fh.read())


def extract_filings(filing_list: list[Filing]) -> pl.DataFrame:
    """Parse many filings, downloading serially and parsing across cores.

    Two passes.  The second one exists because a 10-K need not contain the
    financial statements it reports -- see the EX-13 note above -- and it is
    gated on the first pass returning *nothing* for a filing, which is the only
    signal that separates a wrapper from a filing whose disclosures this module
    simply does not cover.  Downloading stays in the parent on both passes; the
    workers only ever read from disk.
    """
    cached = [f for f in filing_list if ensure_cached(f)]
    first = parallel.map_frames(
        extract_filing,
        cached,
        schema=HTML_SCHEMA,
        label="html extract",
        describe=lambda f: f"{f.ticker} {f.accn}",
    )

    parsed = set(first["accn"].unique()) if not first.is_empty() else set()
    empty = [f for f in cached if f.accn not in parsed]
    with_exhibit = [f for f in empty if ensure_exhibit_cached(f)]
    if not with_exhibit:
        return first

    log.info("html extract: %d filings retried from EX-13", len(with_exhibit))
    second = parallel.map_frames(
        extract_exhibit,
        with_exhibit,
        schema=HTML_SCHEMA,
        label="html extract (ex-13)",
        describe=lambda f: f"{f.ticker} {f.accn} EX-13",
    )
    return pl.concat([first, second]) if not second.is_empty() else first


def canonical_loan_columns() -> set[str]:
    """Loan columns the HTML specs can populate, for coverage reporting."""
    return {f"loans_{c}" for c in taxonomy.LOAN_CATEGORIES}
