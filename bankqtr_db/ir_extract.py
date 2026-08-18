"""Extract panel variables from downloaded IR supplements and presentations.

Three document shapes, three readers
------------------------------------
* **HTML tables** (most supplements).  ``pandas.read_html`` handles these well
  and they hold the loan-portfolio schedule, the asset-quality summary and the
  criticized/classified ladder -- columns the panel already has but that many
  banks never tag dimensionally.
* **PDF lines** (the rest of the supplements).  Table *detection* is the wrong
  tool for a PDF supplement and :func:`pdf_line_rows` explains why at length;
  the short version is that these documents are typeset without cell borders,
  so column boundaries have to be guessed from whitespace and the guess is not
  even consistent between two blocks on the same page.  A schedule line parses
  unambiguously without any of that.
* **Phrases** (presentations).  Office-CRE and leveraged-lending exposure exist
  nowhere in XBRL and in no table either -- Regions publishes office exposure
  as a slide caption reading "Balances $858 % of Total Loans 0.9% NPL $121 ...
  ACL $35" -- so they are read with a section-scoped label/value scan.

The traps
---------
Everything below produced a plausible, wrong, non-raising number during
development.  The pattern they share is that a supplement says the same thing
several times over, in different senses, using the same words each time -- so
matching on the words alone picks an arbitrary one of them.

**Average vs period-end.**  Supplements publish both, on facing pages, with
identical row labels.  Regions' end-of-period owner-occupied CRE is 4,890 and
its six-month average is 4,882.  Section state is tracked while reading.

**Sub-schedules.**  The same variable is stated for the whole bank and again
per segment, per property type, per risk grade.  Segment tables are skipped;
where readings still compete the largest wins, since every sub-schedule is a
subset of the consolidated figure.

**Schedules that share row labels.**  A bank's allowance rollforward, its
charge-off table and its loan schedule all list the loan classes down the side.
The caption is what separates them, and in these documents the caption sits
outside the table -- hence the context string threaded through every reader.

**Percentage twins.**  The portfolio mix is published in dollars and again as a
share of loans, with the same row labels.  A value whose column carries "%" is
refused.

**Contents pages.**  A supplement's table of contents pairs section titles with
page numbers, and those titles contain every phrase a row pattern looks for.

**Footnote markers.**  "(1)" after a row label is a footnote, and reading it as
a parenthesised negative turned a $3.2bn home-equity book into -$1.

**Year-ago comparatives.**  Every schedule carries 2Q26, 1Q26 and 2Q25 side by
side, so only the left-most number belongs to the document's own quarter --
the same rule the HTML fallback uses.

**Presentation-only signs.**  A rollforward writes the closing allowance as a
deduction.  Balances are taken as magnitudes; flows are left alone, because a
provision release and a net recovery are both real.

Scale
-----
Everything emitted carries ``source="ir_supplement"``.  Unlike a 10-K table,
an IR document usually *declares* its unit where a parser can see it -- in the
table header, in the row label, or in a slide caption -- so ``declared_scale``
is filled where one is stated and :func:`reconcile.resolve_ir_scale` prefers it
over inference.  That matters most for office CRE and leveraged lending, which
have no XBRL counterpart to infer a scale against at all.
"""

from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from .html_fallback import EXCLUDE_LABEL, _to_number
from .ir import IRDoc

log = logging.getLogger(__name__)

# Columns this module can populate that do not exist in the XBRL path at all.
# Kept out of ``variables.ALL_VARS`` on purpose: those definitions drive tag
# lookup in the XBRL extractor, and a tagless entry there would be dead weight.
IR_ONLY_COLUMNS: tuple[str, ...] = (
    "loans_office_cre",
    "nonaccrual_office_cre",
    "acl_office_cre",
    "nco_office_cre",
    "loans_leveraged",
    "loans_cre_unsecured",
)


# --------------------------------------------------------------------------
# Table specifications
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class IRTableSpec:
    name: str
    signals: tuple[str, ...]
    min_hits: int
    row_map: dict[str, str]
    # Rows in the "average balances" block of a table are dropped rather than
    # mapped; see the module docstring.
    period_sensitive: bool = False
    # A caption that identifies this schedule outright.  Signal counting is not
    # enough on its own: an allowance table, a charge-off table and the loan
    # schedule all list the loan classes down the side, so they score alike on
    # body text.  Where a caption names the schedule, it decides.
    header_pattern: re.Pattern[str] | None = None


LOAN_SCHEDULE = IRTableSpec(
    name="ir_loan_schedule",
    signals=(
        "commercial and industrial",
        "commercial real estate",
        "total loans",
        "residential",
        "construction",
        "credit card",
        "home equity",
        "consumer",
        "loans outstanding",
        "average loans",
    ),
    min_hits=3,
    period_sensitive=True,
    header_pattern=re.compile(
        r"loans outstanding|loan portfolio|loans? by (type|class|category|product)"
        r"|loans and leases|period.?end (loans|balances)|loan balances"
        r"|composition of loans|loans held for investment",
        re.IGNORECASE,
    ),
    row_map={
        r"^total loans( and leases)?$|^loans( and leases)?, total$|^total loans outstanding": "loans_total",
        r"owner.?occupied": "loans_cre_owner_occupied",
        r"non.?owner.?occupied|investor real estate|investor commercial real estate"
        r"|income.producing (commercial )?real estate|^ire\b": "loans_cre_investor",
        r"construction|land (acquisition|development)": "loans_construction",
        r"multi.?family|apartment": "loans_multifamily",
        r"^commercial real estate|^commercial mortgage|^cre\b|real estate.*commercial": "loans_cre_total",
        r"small business": "loans_small_business",
        r"commercial and industrial|^c&i\b|^c and i\b": "loans_ci",
        r"lease financing|equipment financ": "loans_lease",
        r"^commercial$|^total commercial( loans)?$": "loans_commercial_total",
        r"residential mortgage|residential real estate|1-4 family|first mortgage": "loans_resi_mortgage",
        r"home equity": "loans_home_equity",
        r"credit card": "loans_credit_card",
        r"^auto|automobile|indirect auto|retail auto": "loans_auto",
        r"^consumer$|^total consumer( loans)?$": "loans_consumer_total",
    },
)

# One "asset quality" spec is too coarse.  A bank prints three schedules that
# all sit under that heading and all list the loan classes down the side, and
# merging them lets the allowance rollforward answer a nonaccrual question: on
# Wells Fargo's 2Q26 supplement the rollforward yielded a nonaccrual total of
# 182 while the actual nonperforming-assets page said 7,643, and the
# rollforward is printed first.  Each schedule therefore gets its own spec,
# pinned by its own caption, and only claims the rows it really carries.

ACL_ROLLFORWARD = IRTableSpec(
    name="ir_acl_rollforward",
    signals=(
        "allowance for credit losses",
        "provision",
        "charge-offs",
        "recoveries",
        "balance, beginning",
        "balance, end",
    ),
    min_hits=3,
    period_sensitive=True,
    header_pattern=re.compile(
        r"allowance for (credit loss|loan)|changes in (the )?allowance"
        r"|allocation of (the )?allowance|provision for credit",
        re.IGNORECASE,
    ),
    row_map={
        r"^(total )?allowance for credit losses|^total allowance"
        r"|^balance,? (at )?end|^ending balance": "acl_total",
        r"^(total )?provision for credit losses": "provision_total",
    },
)

CHARGE_OFFS = IRTableSpec(
    name="ir_charge_offs",
    signals=(
        "net charge-offs",
        "charge-offs",
        "recoveries",
        "commercial and industrial",
        "credit card",
    ),
    min_hits=2,
    period_sensitive=True,
    header_pattern=re.compile(r"net (loan )?charge.?offs|charge.?offs", re.IGNORECASE),
    row_map={
        r"^total net (loan )?charge.?offs|^net (loan )?charge.?offs"
        r"|^total charge.?offs": "nco_total",
    },
)

NONPERFORMING = IRTableSpec(
    name="ir_nonperforming",
    signals=(
        "nonperforming",
        "nonaccrual",
        "other real estate",
        "foreclosed",
        "past due",
        "delinquen",
    ),
    min_hits=2,
    period_sensitive=True,
    header_pattern=re.compile(
        r"nonperforming|non.?accrual|foreclosed asset|delinquen|past due",
        re.IGNORECASE,
    ),
    row_map={
        r"^total nonaccrual|^nonaccrual loans|^total non.?accrual"
        r"|^total non.?performing loans|^non.?performing loans": "nonaccrual_total",
        r"^total non.?performing assets|^non.?performing assets": "npa_total",
        r"^(total )?foreclosed asset|^other real estate owned|^oreo\b": "oreo",
        r"30.{0,4}89 days|accruing loans past due 30": "pd_dpd_30_89",
        r"90 days or more|accruing loans past due 90": "pd_dpd_90_plus",
    },
)

CREDIT_LADDER = IRTableSpec(
    name="ir_credit_ladder",
    signals=(
        "criticized",
        "classified",
        "special mention",
        "substandard",
        "doubtful",
        "pass",
        "risk rating",
    ),
    min_hits=2,
    header_pattern=re.compile(
        r"criticized|classified|risk rating|credit quality indicator"
        r"|internal risk|regulatory rating",
        re.IGNORECASE,
    ),
    row_map={
        r"^total criticized|^criticized( commercial)?( loans)?$|criticized assets": "cq_criticized",
        r"^total classified|^classified( commercial)?( loans)?$|classified assets": "cq_classified",
        r"special mention": "cq_special_mention",
        r"substandard": "cq_substandard",
        r"doubtful": "cq_doubtful",
        r"^pass\b": "cq_pass",
    },
)

TABLE_SPECS: tuple[IRTableSpec, ...] = (
    LOAN_SCHEDULE,
    ACL_ROLLFORWARD,
    CHARGE_OFFS,
    NONPERFORMING,
    CREDIT_LADDER,
)


# --------------------------------------------------------------------------
# Phrase specifications (slide-style disclosures)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PhraseSpec:
    """A labelled value inside a section of a presentation.

    ``section`` scopes the search so that "Balances $858" is only read as
    office exposure when it sits under an office-portfolio heading; the same
    words appear on the trucking and healthcare slides that follow it.
    """

    name: str
    section: re.Pattern[str]
    # variable -> pattern whose first group is the number
    fields: dict[str, re.Pattern[str]] = field(default_factory=dict)
    window: int = 1400
    # variable -> words that, between the heading and the number, mean the
    # match has wandered onto a neighbouring caption.  Keyed per field because
    # the words that disqualify a *balance* ("ACL", "non-performing") are the
    # very labels the allowance and nonaccrual fields match on.
    disqualify: dict[str, re.Pattern[str]] = field(default_factory=dict)
    # Banks whose slide layout this spec has actually been checked against.
    # Empty would mean "every bank", which is what the first version did and
    # why it had to be restricted -- see the note above :data:`PHRASE_SPECS`.
    tickers: tuple[str, ...] = field(default=())
    # Where the section stops.  A character window cannot know where a slide
    # ends, and running past the end is how the office block came to be read
    # from the multi-family slide that follows it.
    boundary: re.Pattern[str] | None = None
    # Variables this spec reads as a *percentage* rather than an amount.  They
    # need no scale and must never be given one.
    percent_fields: tuple[str, ...] = field(default=())
    # (pattern, low, high): the numbers matching ``pattern`` anywhere in the
    # window must sum to within [low, high], or nothing is taken from it.  This
    # is how a scrambled PDF is caught -- see :data:`USB_OFFICE_SHARE`.
    sum_check: tuple[str, float, float] | None = None
    # A variable without which the section is not the section.  Regions' block
    # always leads with its balance, so a quarter that yields only a
    # charge-off has matched something else -- 4Q24 produced a lone nco of 32
    # off an "NPL Paying Current" caption.
    require: str | None = None


# A currency-marked number: "$858", "$ 7.4", "$1,832".  Requiring the "$" is
# most of what separates a disclosed balance from a number that merely sits
# nearby -- an axis label, a percentage, a share count -- on a slide where
# proximity means nothing.
# The currency marker is sometimes doubled where a template leaves an empty
# cell before the figure -- Regions' 1Q24 office block reads "Balances $ $1,504"
# -- so a second "$" is tolerated rather than ending the match.
_MONEY = r"\$\s*\$?\s*(\d[\d,]*(?:\.\d+)?)"

# Regions gives every property-type slide the same heading shape, so the next
# one of these is where the office slide stops.  Without that boundary the
# office block was read from the multi-family slide printed after it: office
# exposure of $1,574m came out as $4,279m, which is multi-family's balance.
# The slide *number* is what makes this a boundary rather than a description.
# Matching on "Portfolio (Outstanding balances as of" alone would fire on the
# office slide's own subtitle and truncate the window to nothing.
_RF_SLIDE_BOUNDARY = re.compile(
    r"(?<!\d)\d{1,2} CRE[\s\-]+[\w\-]+ Portfolio", re.IGNORECASE
)

OFFICE_CRE = PhraseSpec(
    name="office_cre",
    section=re.compile(
        r"(cre|commercial real estate)[\s\-]*(office|office portfolio)"
        r"|office (cre|portfolio|exposure)"
        r"|office (commercial real estate|real estate) portfolio",
        re.IGNORECASE,
    ),
    # A generous cap; the slide boundary below is what actually ends the
    # section, and it fires long before this does.
    window=1400,
    boundary=_RF_SLIDE_BOUNDARY,
    # Regions prints a fixed "Key Portfolio Metrics" block under its office
    # heading -- Balances / % of Total Loans / NPL / Charge-offs / ACL, in that
    # order, every quarter.  No other bank in the universe does, which is the
    # whole reason this is an allowlist; see :data:`PHRASE_SPECS`.
    tickers=("RF",),
    require="loans_office_cre",
    fields={
        # Anchored on the block, not on the word "Balances" alone.  The slide's
        # own prose says "Outstanding balances as of June 30" in its subtitle,
        # and its footnote says "except for charge-offs" -- both of which a
        # bare label match walks straight into.
        "loans_office_cre": re.compile(
            rf"Key Portfolio Metrics.{{0,12}}?Balances\s*{_MONEY}",
            re.IGNORECASE,
        ),
        "nonaccrual_office_cre": re.compile(
            rf"\bNPL\b(?! ?/)[\s:]*{_MONEY}", re.IGNORECASE
        ),
        "acl_office_cre": re.compile(
            rf"\bACL\b(?! ?/)[\s:]*{_MONEY}", re.IGNORECASE
        ),
        "nco_office_cre": re.compile(
            rf"Charge.?offs\b(?! ?/)[\s:]*{_MONEY}", re.IGNORECASE
        ),
    },
)

# Leveraged lending is *defined* on these slides far more often than it is
# measured -- in a risk-appetite bullet, a footnote, a list of portfolios a
# bank has been reducing.  Every pattern tried against it read something else:
#
#   USB    the next "total" on the slide            -> $1.18tn vs a $381bn book
#   GS     "Results Snapshot Net Revenues 2025 $58.28 billion"
#   CFG    "granular hold positions with an average outstanding of ~$12 million"
#   FCNCA  "CIT has built risk management ... for a $100 billion" (rail)
#   PNC    "~90% are CLO securitizations $33 billion"
#   STT    "Average loan size of ..."
#
# None of those is a leveraged-lending balance, and the only reading that was
# right -- Regions' "Leveraged Lending Definition - $6.8B" -- is a prose
# fragment stating a definition, not a reported balance.  The spec is kept for
# the record and left out of :data:`PHRASE_SPECS`: the column stays null rather
# than carrying six banks' worth of unrelated numbers.
LEVERAGED = PhraseSpec(
    name="leveraged_lending",
    section=re.compile(
        r"leveraged (lending|loans?|finance|credit)|sponsor(ed)? finance", re.IGNORECASE
    ),
    window=160,
    fields={
        "loans_leveraged": re.compile(
            r"(?:balances?|outstanding(?: balances?)?|exposure|portfolio)?\s*"
            r"(?:of|:|is|was|at|totall?(?:ed|ing)?)?\s*"
            r"\$\s*(\d[\d,]*(?:\.\d+)?)\s*"
            r"(?:billion|bn\b|million|mm?\b|b\b)",
            re.IGNORECASE,
        ),
    },
)

# Citizens states its office balance in a sentence rather than a metrics block:
# "CRE General Office portfolio of $3.4 billion, down ~$200 million, or 6%, QoQ".
# The sentence is the anchor, so the window is only as long as the clause.
#
# Its ACL is deliberately *not* taken.  That line reads "Memo: General Office
# ACL $ 370 10.2 % $ 364 10.6 %" -- two periods side by side with the **prior**
# quarter first, the reverse of every other schedule in this module, and the
# accompanying prose confirms 10.6% is the current one.  A number that means
# the opposite of what the surrounding convention implies is not worth taking
# for one extra column.
CFG_OFFICE = PhraseSpec(
    name="office_cre_cfg",
    section=re.compile(
        r"(?:CRE )?General Office portfolio(?: of)?", re.IGNORECASE
    ),
    window=60,
    tickers=("CFG",),
    fields={
        "loans_office_cre": re.compile(
            r"^[\s,]*(?:of|totall?(?:ing|ed)?|was|is|at)?\s*"
            # A lookahead, not a consuming group: the magnitude word has to
            # still be there for ``_magnitude_after`` to read the unit off it.
            # Consuming it left "$3.4 billion" scaled as millions, because the
            # only unit left to find was the document's own caption.
            r"\$\s*([\d.,]+)\s*(?=billion|million|bn\b|mm?\b)",
            re.IGNORECASE,
        ),
    },
)

# US Bancorp stopped publishing an office *balance* after 2Q23.  What it does
# publish every quarter is the CRE book split by property class:
#
#   "CRE by Property Class SFR Construction 7% Owner Occupied 21%
#    Multi-Family 38% Office 9% Industrial 11% Other 14%"
#
# so the share is what gets extracted -- the disclosure as made, rather than a
# balance multiplied out of it against an XBRL CRE total that may not cover the
# same population.
#
# ``sum_check`` is what makes this safe.  US Bancorp's later decks are PDFs
# whose text extraction interleaves characters from neighbouring boxes
# ("oSnFfRid =en Stiinaglle Family Residential"), and out of that soup the
# office share read 21% and 19% against a real trend of 13, 13, 13, 12, 10, 9.
# A legend whose slices no longer sum to 100 has not been read correctly, and
# nothing is taken from it.
USB_OFFICE_SHARE = PhraseSpec(
    name="office_share_usb",
    section=re.compile(r"CRE by Property Class", re.IGNORECASE),
    window=220,
    tickers=("USB",),
    percent_fields=("office_cre_share_of_cre",),
    sum_check=(r"([\d.]+)\s*%", 95.0, 105.0),
    fields={
        "office_cre_share_of_cre": re.compile(
            r"\bOffice\s+([\d.]+)\s*%", re.IGNORECASE
        ),
    },
)

# Phrase specs are an **allowlist**, not a general reader, and that is the
# hard-won conclusion of this module.
#
# A statistical schedule has structure a parser can lean on -- a caption, row
# labels, aligned columns.  A slide has none: the only evidence that a number
# belongs to a heading is that it is printed near it, and "near" is worth
# nothing when the neighbouring caption is a share count or a revenue line.
# Run across all 31 banks, one generic office spec put Bank of America's *share
# count* in the office column, Citizens' *allowance* in place of its balance,
# and First Citizens' *venture* book in place of its offices -- four of five
# banks wrong, every value plausible, nothing raised.
#
# So a spec applies only to banks whose layout has been read and checked, and
# the three below are three genuinely different disclosures, not one pattern
# stretched over three banks:
#
#   RF   a fixed "Key Portfolio Metrics" block, balance and NPL and ACL
#   CFG  a sentence stating the balance, and nothing else usable
#   USB  no balance at all since 2Q23, only the share of CRE
#
# Adding a bank means reading its deck and confirming the block, not loosening
# a pattern until something matches.
PHRASE_SPECS: tuple[PhraseSpec, ...] = (OFFICE_CRE, CFG_OFFICE, USB_OFFICE_SHARE)


# --------------------------------------------------------------------------
# Document readers
# --------------------------------------------------------------------------

_TAG = re.compile(r"<[^>]+>")
_SCRIPT = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)


def _decode_entities(text: str) -> str:
    import html as _html

    return _html.unescape(text)


def document_text(path: Path) -> str:
    """Flat readable text of a document, for the phrase scanner."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            import pdfplumber

            with pdfplumber.open(path) as pdf:
                return "\n".join((page.extract_text() or "") for page in pdf.pages)
        except Exception as exc:  # noqa: BLE001 - malformed PDFs are routine
            log.debug("pdf text failed %s: %s", path.name, exc)
            return ""
    if suffix in (".htm", ".html"):
        raw = path.read_bytes().decode("utf-8", errors="replace")
        raw = _SCRIPT.sub(" ", raw)
        return re.sub(r"\s+", " ", _decode_entities(_TAG.sub(" ", raw)))
    return ""


# A table plus the text printed immediately above it.  That context is where a
# PDF keeps the things the table itself never repeats: the unit caption
# ("$ amounts in millions") and the block heading ("Average Balances / Six
# Months Ended June 30") that says these are not period-end figures.
Table = tuple[pd.DataFrame, str]


def document_tables(path: Path) -> list[Table]:
    """Every table in a document, whatever its container format."""
    suffix = path.suffix.lower()
    if suffix in (".htm", ".html"):
        return _html_tables(path.read_bytes())
    if suffix in (".xlsx", ".xls"):
        return _excel_tables(path)
    # PDFs deliberately do not come through here; see :func:`pdf_line_rows`.
    return []


def _html_tables(payload: bytes) -> list[Table]:
    """Each table with the heading printed above it.

    The heading is not decoration.  A bank's allowance rollforward, its
    charge-off table and its nonaccrual table all use the loan classes as row
    labels, so on body text alone every one of them looks exactly like the loan
    schedule -- Wells Fargo's supplement produced five "loan schedules" per
    quarter, four of them charge-offs.  What tells them apart is the caption,
    and in these documents the caption sits *outside* the ``<table>``.

    Tables are therefore walked one at a time through lxml, which keeps each
    one paired with its own preceding text; a document lxml cannot parse falls
    back to reading the tables in bulk without context.
    """
    tables = _html_tables_with_context(payload)
    if tables:
        return tables
    for flavor in ("lxml", "bs4"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                found = pd.read_html(BytesIO(payload), flavor=flavor)
        except Exception as exc:  # noqa: BLE001 - malformed markup is the norm
            log.debug("read_html %s failed: %s", flavor, exc)
            continue
        if found:
            return [(t, "") for t in found]
    return []


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


def _excel_tables(path: Path) -> list[Table]:
    try:
        sheets = pd.read_excel(path, sheet_name=None, header=None)
    except Exception as exc:  # noqa: BLE001
        log.debug("excel read failed %s: %s", path.name, exc)
        return []
    return [
        (df, str(name))
        for name, df in sheets.items()
        if not df.empty and df.shape[1] >= 2
    ]


# --------------------------------------------------------------------------
# Table extraction
# --------------------------------------------------------------------------


def _table_text(df: pd.DataFrame) -> str:
    return " ".join(str(v).lower() for v in df.head(80).to_numpy().ravel())


# A supplement's contents page is a two-column table of section titles and page
# numbers, and those titles contain every phrase the loan schedule matches on.
_CONTENTS_TABLE = re.compile(r"(^|\s)page(\s|$)", re.IGNORECASE)

# How much more a signal counts for when it appears in the table's own heading
# rather than anywhere in its body.
_HEADER_WEIGHT = 3

# Business-segment schedules repeat the consolidated row labels over a subset
# of the book.  The panel is a consolidated one, so they are skipped outright
# rather than merged.
_SEGMENT_TABLE = re.compile(
    r"\bsegment\b|line of business|business line|reportable segment"
    r"|banking and lending\b|\bwim\b",
    re.IGNORECASE,
)


def classify_table(df: pd.DataFrame, context: str = "") -> IRTableSpec | None:
    """Which schedule a table is, weighting its heading above its body.

    Body text alone is not enough to tell them apart.  A bank's allowance
    rollforward, its charge-off table and its nonaccrual table all use the loan
    *classes* as row labels, so every one of them scores full marks against the
    loan-schedule signals and reports charge-offs as loan balances.  What
    separates them is the heading -- "Allowance for Credit Losses", "Net
    Charge-offs" -- so heading matches are weighted well above body matches.
    """
    body = _table_text(df)
    header = f"{context} {_table_text(df.head(4))}"
    if _CONTENTS_TABLE.search(header):
        return None

    if _SEGMENT_TABLE.search(header):
        # A segment schedule uses the same row labels as the consolidated one
        # but covers a fraction of the book, and it is printed first: Wells
        # Fargo's Commercial Banking segment reports C&I of 181,739 against a
        # consolidated 487,630, and first-hit-wins handed the panel the
        # segment figure.  The panel is consolidated, so these are skipped.
        return None

    # Where a caption names the schedule outright it decides, and the more
    # specific schedules are tested first: every one of them lists the loan
    # classes down the side, so "loan portfolio" would otherwise capture the
    # allowance and charge-off tables too.
    for spec in (
        CREDIT_LADDER,
        ACL_ROLLFORWARD,
        CHARGE_OFFS,
        NONPERFORMING,
        LOAN_SCHEDULE,
    ):
        if spec.header_pattern is not None and spec.header_pattern.search(header):
            return spec

    best: tuple[int, IRTableSpec] | None = None
    for spec in TABLE_SPECS:
        body_hits = sum(1 for s in spec.signals if s in body)
        header_hits = sum(1 for s in spec.signals if s in header)
        # Heading matches count toward the threshold as well as the score.
        if body_hits + header_hits < spec.min_hits:
            continue
        score = body_hits + header_hits + _HEADER_WEIGHT * header_hits
        if best is None or score > best[0]:
            best = (score, spec)
    return best[1] if best else None


# Section markers inside a loan or asset-quality schedule.  "Average" blocks
# carry the same row labels as the period-end block directly below them.
_AVERAGE_MARKER = re.compile(r"^\s*average(\s+balances?|\s+loans?)?\s*$", re.IGNORECASE)
_PERIOD_END_MARKER = re.compile(
    r"^\s*(quarter|period|year)[\s-]*end(\s+balances?)?\s*$|^\s*(ending|end of period)\s*$",
    re.IGNORECASE,
)
_AVERAGE_LABEL = re.compile(r"^average\b|\baverage (loans|balances)\b", re.IGNORECASE)
# Header text that marks an entire table as average rather than period-end
# balances, including the year-to-date variants ("Six Months Ended June 30").
_AVERAGE_TABLE = re.compile(
    r"average balances?|average loans?|average (deposits|assets|earning)"
    r"|\baverages\b",
    re.IGNORECASE,
)

# A caption naming *both* bases describes a single table that holds both, with
# the period-end block first -- Wells Fargo's loan schedule is headed
# "CONSOLIDATED LOANS OUTSTANDING - PERIOD-END BALANCES, AVERAGE BALANCES, AND
# AVERAGE INTEREST RATES".  Skipping it on the word "average" discarded the
# loan schedule itself, so where both appear the row-level markers decide.
_PERIOD_END_HEADER = re.compile(
    r"period.?end|end of period|quarter.?end|end of quarter|balances at",
    re.IGNORECASE,
)

# Supplements interleave balance rows with ratio rows that reuse the balance
# row's wording -- "Nonperforming assets to total loans, OREO, foreclosed and
# other assets" sits directly under the OREO balance and matched it, loading a
# 0.66% ratio into a dollar column.  A ratio row always names its denominator.
_RATIO_LABEL = re.compile(
    r"\bto (total |average )?(loans|assets|deposits|equity|nonperforming|nco)"
    r"|\bratio\b|\bcoverage\b|\bas a percent|\bper (share|common share)\b"
    r"|\bannualized\b.*\brate\b|\byield\b|\bmargin\b|\brate\b$",
    re.IGNORECASE,
)


def _period_mode(label: str, current: str) -> str:
    """Track whether following rows are average or period-end balances."""
    if _AVERAGE_MARKER.match(label):
        return "average"
    if _PERIOD_END_MARKER.match(label):
        return "period_end"
    return current


# --------------------------------------------------------------------------
# Declared units
# --------------------------------------------------------------------------

# Unlike a 10-K table, where the "(in millions)" caption sits outside the
# <table> and read_html cannot see it, an IR supplement states its unit inside
# the table -- and PNC states it *per row*: a millions-denominated schedule
# whose loan lines read "Average loans In billions".  Reading the table header
# alone would put those rows out by a factor of a thousand, so the row label is
# checked first and the table header only as a default.
_SCALE_PHRASE = re.compile(
    r"\(?\$?\s*in\s+(thousands|millions|billions)\b|\((thousands|millions|billions)\)",
    re.IGNORECASE,
)
_SCALE_VALUE = {"thousands": 1e3, "millions": 1e6, "billions": 1e9}


def declared_scale(text: str) -> float | None:
    """Reporting multiplier stated in ``text``, if it states one."""
    match = _SCALE_PHRASE.search(text or "")
    if not match:
        return None
    word = (match.group(1) or match.group(2) or "").lower()
    return _SCALE_VALUE.get(word)


def _strip_scale(label: str) -> str:
    """Row label without its unit qualifier, so anchored patterns still match."""
    return re.sub(r"\s{2,}", " ", _SCALE_PHRASE.sub(" ", label)).strip(" ,;:")


# --------------------------------------------------------------------------
# Reading a value out of a row
# --------------------------------------------------------------------------

# Two hazards the plain "left-most numeric cell" rule gets wrong on IR
# documents:
#
# 1. *Merged cells.*  pdfplumber's text strategy often returns a row as one
#    cell -- "Commercial real estate mortgage-owner-occupied 4,890 4,849" --
#    so the value never appears as a cell of its own and the scan runs on to
#    the following column, silently reporting a prior-quarter balance.
# 2. *Percentage twins.*  Regions publishes the portfolio mix twice: once in
#    dollars and once as a percentage of loans, with identical row labels.
#    Whichever table is met first wins, so a 5.1% share can land in a dollar
#    column.  A value whose own cell or next cell carries "%" is refused.

# The label of a merged cell, then the *first* number after it.  Anchoring on
# the first number rather than the last is what makes this work on a row that
# carries the whole comparative table -- "Commercial and industrial $ 49,586 $
# 48,879 $ 49,671 ... 1.4 %" -- where the last number is a percentage change
# and only the first belongs to the document's own quarter.
_LABEL_THEN_NUMBER = re.compile(
    r"^(?P<label>.*?[a-z\)\]%])[\s:]*\$?\s*(?P<num>\(?-?\d[\d,]*(?:\.\d+)?\)?)"
    r"(?P<after>\s*%?)",
    re.IGNORECASE | re.DOTALL,
)
_PCT_CELL = re.compile(r"%")


# "(1)" and "(2)" after a row label are footnote markers, and a parenthesised
# number is also how these documents write a negative.  Regions' home-equity
# line reads "Home equity-lines of credit (1) 3,184 3,130 ..." and parsing the
# marker as the value turned a $3.2bn book into -$1.
_FOOTNOTE_MARKER = re.compile(r"^\(\d{1,2}\)$")

# The only number on a contents line is the page it points at.  A real balance
# is written with a thousands separator, a decimal point or a currency symbol,
# so a bare one- or two-digit integer standing alone is not one.
_FINANCIAL_NUMBER = re.compile(r"[,.]|^\d{3,}$")

_NUMERIC_TOKEN = re.compile(r"\(?-?\$?\s?\d[\d,]*(?:\.\d+)?\)?")


def _split_merged(cell: str) -> tuple[str, float | None]:
    """Split ``"Total loans 96,477 96,122"`` into its label and first value.

    Scans past footnote markers rather than stopping at the first parenthesised
    token, refuses a percentage column, and refuses a lone contents-page number.
    """
    text = str(cell).strip()
    match = _LABEL_THEN_NUMBER.match(text)
    if not match:
        return text, None

    label = match.group("label").strip()
    rest = text[match.start("num") :]
    tokens = [
        t
        for t in _NUMERIC_TOKEN.finditer(rest)
        if not _FOOTNOTE_MARKER.match(t.group(0))
    ]
    if not tokens:
        return label, None

    first = tokens[0]
    if "%" in rest[first.end() : first.end() + 2]:
        return label, None  # a ratio line, not a balance
    if len(tokens) == 1 and not _FINANCIAL_NUMBER.search(first.group(0).strip("()$ ")):
        return label, None  # a contents entry pointing at a page
    return label, _to_number(first.group(0))


_PUNCTUATION_ONLY = re.compile(r"^[^\w]+$")

# Longest a genuine row label runs to.  Anything longer is a heading or a
# contents entry that happens to contain the words a row pattern looks for.
_MAX_LABEL_CHARS = 90

# Balances that cannot be negative in fact, only in presentation.  An allowance
# rollforward writes the closing allowance and the charge-offs as deductions --
# "(14,407)", "(876)" -- and carrying that sign into the panel makes a bank
# look like it holds a negative reserve.  Flows are left alone: a provision
# release and a net recovery are both real and both genuinely negative.
_NON_NEGATIVE_PREFIXES = ("loans_", "acl_", "nonaccrual_", "cq_", "pd_")
_NON_NEGATIVE_EXACT = ("oreo", "npa_total")


def _signed(variable: str, value: float) -> float:
    if variable in _NON_NEGATIVE_EXACT or variable.startswith(_NON_NEGATIVE_PREFIXES):
        return abs(float(value))
    return float(value)


def row_label_and_value(row: pd.Series) -> tuple[str, float | None]:
    """The row's label and the value belonging to the document's own quarter.

    Returns ``(label, None)`` when the row has no usable number -- including
    when the only numbers present are percentages, and when the only number is
    a contents-page reference.
    """
    cells = [str(v).strip() for v in row]
    parts: list[str] = []
    value: float | None = None
    numbers: set[float] = set()

    for index, cell in enumerate(cells):
        if cell.lower() in ("", "nan", "none"):
            continue
        number = _to_number(cell)
        if number is None:
            # Either a label cell, or a label with its value merged into it.
            text, merged = _split_merged(cell)
            # A label can be split across several cells -- one extraction of
            # Regions' schedule yields ["Commercial", "and industrial", "$",
            # "49,586"], and reading only the first cell turns a C&I line into
            # the commercial *total*.  Leading text cells are joined, skipping
            # the currency markers and the duplicates that ``read_html`` emits
            # for a cell spanning several columns.
            is_new_label_part = (
                value is None
                and text
                and not _PUNCTUATION_ONLY.match(text)
                and (not parts or parts[-1].lower() != text.lower())
            )
            if is_new_label_part:
                parts.append(text)
            if merged is not None and value is None and not _PCT_CELL.search(cell):
                value = merged
                numbers.add(merged)
            continue

        numbers.add(number)
        if value is not None or not parts:
            # A numeric column before any label is not this row's value.
            continue
        # The unit marker for a percentage sits in the *next* cell as often as
        # in this one.
        following = next(
            (c for c in cells[index + 1 :] if c.lower() not in ("", "nan", "none")), ""
        )
        if _PCT_CELL.search(cell) or _PCT_CELL.match(following):
            continue
        value = number

    label = " ".join(parts).strip().lower()
    # A supplement's contents page is a two-column table of section titles and
    # page numbers, and those titles contain every row label the loan schedule
    # uses -- Wells Fargo's reads "Commercial Loan Portfolio - Commercial and
    # Industrial Loans ... by Property Type | 23", which put a $23m C&I book on
    # a bank with $849bn of loans.
    #
    # The test is on *distinct* values rather than on how many cells hold them:
    # ``read_html`` repeats a cell that spans several columns, so the page
    # number arrives three times over and a count-based rule never fires.
    if (
        value is not None
        and len(numbers) == 1
        and not _FINANCIAL_NUMBER.search(f"{value:g}")
    ):
        return label, None
    # A label this long is a section title, not a line item.  Real schedule
    # labels are a few words; contents entries run to a full sentence.
    if len(label) > _MAX_LABEL_CHARS:
        return label, None
    return label, value


def extract_table(
    df: pd.DataFrame, spec: IRTableSpec, doc: IRDoc, context: str = ""
) -> list[dict[str, Any]]:
    compiled = [(re.compile(p, re.IGNORECASE), var) for p, var in spec.row_map.items()]
    rows: list[dict[str, Any]] = []
    # A table with no explicit "quarter end" marker is assumed to be
    # period-end, which is what a standalone schedule is.
    mode = "period_end"
    seen: set[str] = set()
    header = f"{context} {_table_text(df.head(6))}"
    table_scale = declared_scale(_table_text(df.head(12))) or declared_scale(context)

    # A whole table of averages is a separate schedule with the same row
    # labels as the period-end one -- Regions publishes a "Average Balances /
    # Six Months Ended June 30" block whose owner-occupied CRE line reads
    # 4,882 against a period-end 4,890.  Row-level marker tracking cannot see
    # that when the header lands in its own extracted table, so the table's
    # own opening text is checked as well.
    if (
        spec.period_sensitive
        and _AVERAGE_TABLE.search(header)
        and not _PERIOD_END_HEADER.search(header)
    ):
        return rows

    for _, row in df.iterrows():
        raw_label, value = row_label_and_value(row)
        if not raw_label:
            continue
        label = _strip_scale(raw_label)
        if not label:
            continue
        if spec.period_sensitive:
            mode = _period_mode(label, mode)
            if mode == "average" or _AVERAGE_LABEL.match(label):
                continue
        if EXCLUDE_LABEL.search(label) or _RATIO_LABEL.search(label):
            continue
        if value is None:
            continue
        for pattern, variable in compiled:
            if not pattern.search(label):
                continue
            if variable in seen:
                break  # first hit in a table wins; later rows are subtotals
            seen.add(variable)
            rows.append(
                {
                    "variable": variable,
                    "label": label[:120],
                    "value": _signed(variable, value),
                    "table": spec.name,
                    "declared_scale": declared_scale(raw_label) or table_scale,
                    "confidence": 0.55,
                }
            )
            break
    return rows


# --------------------------------------------------------------------------
# PDF line parsing
# --------------------------------------------------------------------------


def _compiled_row_maps() -> dict[str, list[tuple[re.Pattern[str], str]]]:
    """Every spec's row patterns, compiled once per run."""
    return {
        spec.name: [
            (re.compile(p, re.IGNORECASE), var) for p, var in spec.row_map.items()
        ]
        for spec in TABLE_SPECS
    }


def pdf_line_rows(path: Path) -> list[dict[str, Any]]:
    """Parse a PDF supplement line by line rather than as tables.

    Table detection is the wrong tool for these documents.  A supplement is
    typeset without cell borders, so pdfplumber's lattice strategy sees nothing
    and its text strategy has to guess column boundaries from whitespace -- and
    it guesses differently for adjacent blocks on the same page.  On Regions'
    Q2 2025 supplement that produced two contradictory readings of the same
    schedule and silently returned the six-month *average* owner-occupied CRE
    balance (4,882) in place of the period-end one (4,890).

    A schedule line, on the other hand, is unambiguous::

        Commercial real estate mortgage-owner-occupied 4,890 4,849 4,841 ...

    -- a text label, then the document's own quarter, then comparatives.  So
    the label and the first number are taken straight off the line, and the
    surrounding lines supply the section context (which schedule this is, what
    unit it is in, and whether the block is averages) that a bare table would
    have thrown away.
    """
    try:
        import pdfplumber
    except ImportError:  # pragma: no cover
        log.warning("pdfplumber missing; PDF supplements will not be parsed")
        return []

    compiled = _compiled_row_maps()
    rows: list[dict[str, Any]] = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                rows.extend(_page_line_rows(page.extract_text() or "", compiled))
    except Exception as exc:  # noqa: BLE001 - malformed PDFs are routine
        log.debug("pdf line parse failed %s: %s", path.name, exc)
    return rows


def _page_line_rows(
    text: str, compiled: dict[str, list[tuple[re.Pattern[str], str]]]
) -> list[dict[str, Any]]:
    lines = text.splitlines()
    lowered = text.lower()
    # A row label only counts for a schedule the page actually contains; the
    # word "consumer" appears on a capital-ratios page too.
    active = {
        spec.name
        for spec in TABLE_SPECS
        if sum(1 for signal in spec.signals if signal in lowered) >= spec.min_hits
    }
    if not active:
        return []

    rows: list[dict[str, Any]] = []
    scale: float | None = None
    mode = "period_end"
    seen: set[tuple[str, str]] = set()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        scale = declared_scale(stripped) or scale
        if _AVERAGE_TABLE.search(stripped):
            mode = "average"
        elif _PERIOD_END_MARKER.match(stripped):
            mode = "period_end"

        raw_label, value = _split_merged(stripped)
        if value is None:
            continue
        label = _strip_scale(raw_label).lower()
        if not label or EXCLUDE_LABEL.search(label) or _RATIO_LABEL.search(label):
            continue

        for spec in TABLE_SPECS:
            if spec.name not in active:
                continue
            if spec.period_sensitive and (
                mode == "average" or _AVERAGE_LABEL.match(label)
            ):
                continue
            matched = next(
                (var for pattern, var in compiled[spec.name] if pattern.search(label)),
                None,
            )
            if matched is None or (spec.name, matched) in seen:
                continue
            seen.add((spec.name, matched))
            rows.append(
                {
                    "variable": matched,
                    "label": label[:120],
                    "value": float(value),
                    "table": spec.name,
                    "declared_scale": declared_scale(raw_label) or scale,
                    "confidence": 0.55,
                }
            )
            break
    return rows


# --------------------------------------------------------------------------
# Phrase extraction
# --------------------------------------------------------------------------

# A magnitude word straight after the figure, as prose writes it: "$3.4
# billion".  This is a stronger unit statement than any caption -- it is
# attached to the number itself -- so it wins where both are present.
_MAGNITUDE_AFTER = re.compile(
    r"^\s*(billion|bn|million|mm|thousand|[kmb])\b", re.IGNORECASE
)
_MAGNITUDE_VALUE = {
    "billion": 1e9,
    "bn": 1e9,
    "b": 1e9,
    "million": 1e6,
    "mm": 1e6,
    "m": 1e6,
    "thousand": 1e3,
    "k": 1e3,
}


def _magnitude_after(window: str, end: int) -> float | None:
    """The unit written immediately after a figure, if there is one."""
    match = _MAGNITUDE_AFTER.match(window[end:])
    return _MAGNITUDE_VALUE.get(match.group(1).lower()) if match else None


def _sums_check_out(window: str, spec: PhraseSpec) -> bool:
    """Whether a window's parts add up to what the spec says they should.

    A pie legend's slices sum to 100.  When they do not, the text was not read
    correctly -- which is exactly what happens to US Bancorp's later decks,
    where PDF extraction interleaves characters from adjacent boxes and the
    office slice came out at 21% against a real trend of 13, 13, 13, 12, 10, 9.
    """
    if spec.sum_check is None:
        return True
    pattern, low, high = spec.sum_check
    values = [_to_number(v) for v in re.findall(pattern, window)]
    total = sum(v for v in values if v is not None)
    return low <= total <= high


# A "%" glued to the number means the phrase matched a ratio, not a balance.
# It has to be glued: Regions' office slide reads "Balances $858 % of Total
# Loans 0.9%", where the "%" after 858 opens the *next* caption rather than
# qualifying 858, and rejecting on it discarded office exposure entirely.
_PCT_SUFFIX = re.compile(r"^\s*%(?!\s*of\b)")


def extract_phrases(text: str, doc: IRDoc) -> list[dict[str, Any]]:
    """Read slide-style disclosures out of a presentation's flat text.

    Only the *first* section that yields values is used.  Regions' office
    slide is followed by a "Transportation - Trucking" slide with the same
    "Balances / NPL / ACL" caption layout and a heading that still matches the
    office pattern, so scanning on would overwrite office exposure with
    trucking exposure.
    """
    rows: list[dict[str, Any]] = []
    for spec in PHRASE_SPECS:
        if spec.tickers and doc.ticker not in spec.tickers:
            continue
        for match in spec.section.finditer(text):
            window = text[match.end() : match.end() + spec.window]
            # Stop at the next slide rather than at an arbitrary character
            # count -- the following slide's metrics block looks exactly like
            # this one's and is the likeliest thing to be captured by mistake.
            if spec.boundary is not None:
                edge = spec.boundary.search(window)
                if edge is not None:
                    window = window[: edge.start()]
            # A legend whose parts no longer add up has not been read
            # correctly, whatever it looks like.
            if not _sums_check_out(window, spec):
                continue
            # A slide states its unit in its own caption ("$ in Millions"),
            # which is better evidence than anything inferable later.
            scale = declared_scale(window[:400]) or declared_scale(
                text[max(0, match.start() - 400) : match.start()]
            )
            found_rows: list[dict[str, Any]] = []
            for variable, pattern in spec.fields.items():
                found = pattern.search(window)
                if not found:
                    continue
                raw = next((g for g in found.groups() if g), None)
                if raw is None:
                    continue
                is_percent = variable in spec.percent_fields
                if not is_percent and _PCT_SUFFIX.match(window[found.end() :]):
                    continue
                # Whatever stands between the heading and the number decides
                # whether the number is still about the heading.  On a slide,
                # a neighbouring caption is the likeliest thing to have been
                # captured, and it names itself.
                blocker = spec.disqualify.get(variable)
                if blocker is not None and blocker.search(window[: found.start()]):
                    continue
                value = _to_number(raw)
                if value is None or value <= 0:
                    continue
                found_rows.append(
                    {
                        "variable": variable,
                        "label": window[: found.end()][-80:].strip()[:120],
                        "value": float(value),
                        "table": spec.name,
                        # A percentage has no scale and must never be given
                        # one; a prose figure carries its own magnitude word
                        # ("$3.4 billion"), which beats any caption.
                        "declared_scale": (
                            1.0
                            if is_percent
                            else (_magnitude_after(window, found.end()) or scale)
                        ),
                        # Prose is read less confidently than a ruled table.
                        "confidence": 0.45,
                    }
                )
            if spec.require is not None and not any(
                r["variable"] == spec.require for r in found_rows
            ):
                continue
            if found_rows:
                rows.extend(found_rows)
                break  # this section is the disclosure; later ones are neighbours
    return rows


# --------------------------------------------------------------------------
# Per-document extraction
# --------------------------------------------------------------------------

IR_SCHEMA: dict[str, Any] = {
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
    "declared_scale": pl.Float64,
    "doc_type": pl.String,
    "origin": pl.String,
}


def extract_document(doc: IRDoc, path: Path) -> pl.DataFrame:
    """Parse one downloaded supplement or presentation into a long frame."""
    if not path.exists() or path.stat().st_size == 0:
        return pl.DataFrame(schema=IR_SCHEMA)

    rows: list[dict[str, Any]] = []
    if path.suffix.lower() == ".pdf":
        rows.extend(pdf_line_rows(path))
    for table, context in document_tables(path):
        if table.empty or table.shape[1] < 2:
            continue
        spec = classify_table(table, context)
        if spec is None:
            continue
        rows.extend(extract_table(table, spec, doc, context))

    text = document_text(path)
    if text:
        rows.extend(extract_phrases(text, doc))

    if not rows:
        return pl.DataFrame(schema=IR_SCHEMA)

    for row in rows:
        row.update(
            bank=doc.bank,
            ticker=doc.ticker,
            cik=doc.cik,
            # ``accn`` is the reconciliation key for per-filing scale
            # inference; documents pulled off an IR site have no accession
            # number, so the stored filename stands in as the unit of scale.
            accn=doc.accn or path.name,
            form=f"ir_{doc.doc_type}",
            period=doc.period,
            filed=doc.period,
            source="ir_supplement",
            scale_resolved=False,
            doc_type=doc.doc_type,
            origin=doc.origin,
        )

    df = pl.DataFrame(rows, schema_overrides=IR_SCHEMA, infer_schema_length=None)
    return df.sort("confidence", descending=True).unique(
        subset=["ticker", "period", "variable", "table"],
        keep="first",
        maintain_order=True,
    )


def _consolidated_first(df: pl.DataFrame) -> pl.DataFrame:
    """Resolve competing readings of one variable in favour of the largest.

    A supplement states the same variable several times over: once for the
    whole bank and again for each segment, each property type, each risk
    grade.  Every one of those sub-schedules is a strict subset of the
    consolidated figure, so the consolidated figure is the largest -- US
    Bancorp's total loans read 410,300 on the consolidated schedule and 538 on
    a sub-schedule printed above it, and taking whichever came first took the
    538.  Magnitude, not position, decides.

    This also disposes of the misreadings that survive classification: a
    charge-off table read as a loan schedule, or a ratio read as a balance,
    loses to the real balance for the same reason.
    """
    return (
        df.with_columns(pl.col("value").abs().alias("_size"))
        .sort(["_rank", "_size", "confidence"], descending=[False, True, True])
        .unique(
            subset=["ticker", "period", "variable"], keep="first", maintain_order=True
        )
        .drop("_size")
    )


def extract_documents(pairs: list[tuple[IRDoc, Path]]) -> pl.DataFrame:
    frames = []
    for doc, path in pairs:
        try:
            df = extract_document(doc, path)
        except Exception as exc:  # noqa: BLE001 - one bad PDF must not stop the run
            log.warning("IR extract failed %s %s: %s", doc.ticker, path.name, exc)
            continue
        if not df.is_empty():
            frames.append(df)
    if not frames:
        return pl.DataFrame(schema=IR_SCHEMA)

    combined = pl.concat(frames, how="vertical_relaxed")
    # A quarter's supplement and its presentation often state the same figure.
    # Prefer the supplement: it is the audited schedule, the deck is a summary.
    order = {"supplement": 0, "presentation": 1, "release": 2, "credit_metrics": 3}
    ranked = combined.with_columns(
        pl.col("doc_type")
        .replace_strict(order, default=9, return_dtype=pl.Int32)
        .alias("_rank")
    )
    return _consolidated_first(ranked).drop("_rank")
