"""MDRM item codes mapped onto the panel's canonical variables.

An MDRM code is a prefix plus a four-character item: ``RCFD2122`` is item 2122
read on the consolidated basis.  Three things about that are easy to get wrong
and expensive to get wrong quietly.

**The prefix is a reporting basis, not decoration.**  ``RCFD`` is the fully
consolidated bank -- domestic and foreign offices -- and ``RCON`` is domestic
offices only.  A bank with no foreign offices files ``RCON`` and no ``RCFD`` at
all, so a build that insists on ``RCFD`` drops Zions entirely; a build that
prefers ``RCON`` reads JPMorgan's loan book as $1,340bn instead of $1,497bn, an
11% understatement that raises no error and looks perfectly plausible.  Both
prefixes are therefore listed per item and the first one *present* wins.

The preference is not uniform across a filing, which is why it is per item
rather than per schedule.  Schedule RC-N reports its totals and its C&I lines
on ``RCFD`` but its entire real-estate breakdown on ``RCON``, in the same
filing, for the same bank.

**There are two "total loans" and they differ by held-for-sale.**  Schedule
RC-C item 12 (``2122``) is loans and leases including those held for sale;
Schedule RC item 4.b (``B528``) is held for investment only.  The EDGAR panel's
``loans_total`` is held for investment, so ``B528`` is the comparable one --
but the RC-C *category* detail is on the ``2122`` basis, so that is what the
categories must be checked against.  Verified on four banks at 2025Q4:
``B528 + 5369 (held for sale) == 2122`` exactly, to the dollar, every time.

**Flows are year-to-date and reset in Q1.**  JPMorgan's charge-offs run
730 -> 3,346 -> 5,022 -> 6,810 through 2019 and then restart at 1,902 in
2020Q1.  Every ``RIAD`` item behaves this way, for every bank -- unlike the
XBRL panel, where cumulative-only tagging is one bank's quirk.  Differencing
happens in :mod:`callrpt_db.panel`; this module only records which items are
flows.

Everything below was read off the schedules themselves -- ``mdrm_labels`` in
the bulk file gives each code's label -- and every partition is checked against
the schedule's own total by :func:`partition_checks`, not asserted from the
form instructions.  Two codes in RC-C carry *identical* truncated labels
(``K137`` and ``K207`` are both "LN TO IND HH OTHR AUTO LN"); they were told
apart by the data, since Ally Bank reports $76.5bn under ``K137`` and zero
under ``K207``, and Ally without an auto book is not a possible reading.

The form changes under you
--------------------------
A 2001 start crosses four redesigns of the schedules, and each one retired
codes and issued new ones.  Every span below was *measured* -- the first and
last quarter each code carries a value, counted over all ~8,900 filers in
every one of the 102 bulk files -- not taken from the instructions:

======  =======================================================================
2002Q1  Closed-end 1-4 family split into first and junior liens on RC-N and
        RI-B (``5401-5403`` and ``5413/5414`` become ``C229/C230``,
        ``C236-C239``, ``C234/C235``, ``C217/C218``); fed funds purchased and
        repos split (``2800`` becomes ``B993`` + ``B995``).
2007Q1  Construction and nonfarm nonresidential split (``1415`` becomes
        ``F158`` + ``F159``; ``1480`` becomes ``F160`` owner-occupied +
        ``F161`` other), with matching splits on RC-N (``2759/2769/3492``,
        ``3502-3504``), RI-B (``3582/3583``, ``3590/3591``) and RC-L
        (``3816``).  Leases split into ``F162`` + ``F163``.  Through 2007 the
        old codes are carried as derived totals beside the new detail: 1415
        equals F158 + F159 to the dollar for every one of the 4,722 filers
        reporting both, so the "most complete variant wins" rule picks the
        detail and the transition costs nothing.
2010Q1  Item 9 "other loans" (``1563``, with ``1545``/``2081``/``1564``
        detail on form 031) becomes loans to nondepository financial
        institutions ``J454`` plus ``J464`` (``1545``/``2081``/``J451``).
        RC-L's other commitments ``3818`` split into ``J457``-``J459``.
2011Q1  "Other loans to individuals" ``2011`` split into automobile ``K137``
        and other ``K207``, with RC-N (``B578-B580`` -> ``K213-K218``) and
        RI-B (``B516/B517`` -> ``K129/K133`` + ``K205/K206``) following.  TDRs
        by category (``K158``...) replace the single ``1616``.
2015Q1  Basel III RC-R: ``RCFA``/``RCOA`` prefixes on schedule RCRI replace
        ``RCFD``/``RCON`` on RCR; CET1 (``P859``) appears.  2014 is a
        transition year on RCRIA/RCRIB.
2017Q1  RC-N gains a *total* row (``1403``/``1406``/``1407``).  Before that
        the form states no total and one must be built from the categories.
2018Q2  Goodwill ``3163`` moves from RC to RC-M; other intangibles ``0426``
        is replaced by total intangibles ``2143``.
======  =======================================================================

The rule for all of these is the same: the modern codes are ``items``, the
retired ones are ``alternatives``, and a column's *meaning* never changes
across the break.  Where the old form cannot support the modern meaning --
owner-occupied against investor CRE before 2007, auto against other consumer
before 2011 -- the modern column stays null and a coarser column that *is*
consistent across the whole window is added beside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Kind = Literal["stock", "flow"]
Prefer = Literal["complete", "first"]

# Call Report amounts are in thousands of dollars; the panel is in dollars.
UNIT_SCALE = 1_000.0

# Prefix families, in the order they are tried.  A stock item is looked for on
# the consolidated basis first and the domestic one second; an income item has
# only one basis; a regulatory-capital item is reported on RCFA/RCOA.
BALANCE_PREFIXES = ("RCFD", "RCON")
INCOME_PREFIXES = ("RIAD",)
CAPITAL_PREFIXES = ("RCFA", "RCOA")
# The pre-2015 RC-R used the balance-sheet prefixes; the 2014 transition
# schedule for advanced-approaches banks already used the new ones.
LEGACY_CAPITAL_PREFIXES = ("RCFD", "RCON", "RCFA", "RCOA")
# Foreign-office columns, used only where an item exists nowhere else.
FOREIGN_PREFIXES = ("RCFN",)


# The three Call Report forms.  Which one a bank files is stated in that
# quarter's POR roster:
#
#   031  consolidated, banks with foreign offices  (80 filers at 2025Q4)
#   041  domestic only                             (987)
#   051  the short form, under $5bn assets         (3,327; exists from 2017Q2)
#
# Carried on every row as metadata.  It is deliberately *not* what selects
# between item lists -- see ``rollup`` below for why.
FORMS = ("031", "041", "051")


@dataclass(frozen=True)
class ItemSpec:
    """One panel column, as a sum of MDRM items on the first available prefix.

    ``items`` are summed.  A component that is absent for a given bank is
    treated as zero *provided at least one component is present* -- a bank with
    no foreign C&I leaves ``1764`` blank rather than writing 0, and requiring
    every component would drop its C&I book altogether.  If no component is
    present the column is null, not zero: "not reported" and "nothing" are
    different answers and only one of them is safe to rank on.

    ``alternatives`` are coarser lines some filers report *instead of* the
    detail, ordered coarsest first; the first group any component of which is
    present wins, and ``items`` is the fallback.  Which filers report which is
    not a clean function of the form: the short form collapses C&I into 1766,
    leases into 2165 and other loans into J464, but so does a small 041 filer
    before 2017, when no short form existed.  Keying on the form got 5,368 of
    6,974 filers wrong in 2013Q1 for exactly that reason.

    Preferring the coarse line also fixes the opposite trap.  Zions files 041
    and reports the five-way split of loans to depository institutions as
    explicit **zeros** next to the single line 1288 carrying the real $59m --
    so "prefer the detail when present" reads its interbank book as nothing.
    A rollup is the total of its detail by construction, so taking it whenever
    it exists is right in both directions.

    More than one level of rollup exists, which is why this is a list.  RC-C
    item 9 can arrive as 1563 (the whole item), as J454 plus J464 (its two
    halves), or as J454 plus the three lines inside 9.b.  Morgan Stanley Bank
    reports 1563 **and** the detail underneath it, and counting both put its
    loan book 62% over its own reported total.

    ``prefer`` is how the winning variant is chosen.  ``complete`` -- the
    default, and right for every partition leaf -- takes the variant the filer
    reported most completely, ties to the coarser.  ``first`` takes the first
    variant with anything present, in the order listed, and exists for items
    whose variants are *successive definitions* rather than levels of detail:
    the restructured-loan total is ``HK25`` from 2017, ``1616`` before 2011 and
    the sum of eight category lines between, and the eight lines still exist
    after 2017, so completeness would pick them over the form's own total.

    ``schedule`` may name several schedules.  Goodwill is item 3163 on RC
    until 2018Q1 and on RC-M from 2018Q2; an item that appears in more than one
    of the listed schedules in the same filing is taken from the first listed,
    never summed across them.
    """

    name: str
    schedule: str | tuple[str, ...]
    items: tuple[str, ...]
    kind: Kind = "stock"
    prefixes: tuple[str, ...] = BALANCE_PREFIXES
    description: str = ""
    # Items summed on a *second* prefix family and added to the first.  Total
    # deposits is the only case: RCON2200 is domestic and RCFN2200 is foreign,
    # there is no RCFD2200, and a bank's deposits are the sum of the two.
    plus: tuple[tuple[str, tuple[str, ...]], ...] = field(default=())
    # Coarser item groups replacing ``items``, tried in order before it.
    alternatives: tuple[tuple[str, ...], ...] = field(default=())
    prefer: Prefer = "complete"
    # First quarter the modern ``items`` exist; documentation only.
    since: str = "2001Q1"

    @property
    def schedules(self) -> tuple[str, ...]:
        return (self.schedule,) if isinstance(self.schedule, str) else self.schedule

    def variants(self) -> tuple[tuple[str, ...], ...]:
        """Item groups in the order they are tried; first one present wins."""
        return (*self.alternatives, self.items)

    def all_items(self) -> tuple[str, ...]:
        seen: list[str] = []
        for group in self.variants():
            seen.extend(i for i in group if i not in seen)
        return tuple(seen)


# --------------------------------------------------------------------------
# Schedule RC-C Part I -- the loan book, by category
# --------------------------------------------------------------------------
#
# These groups are a *partition* of RC-C item 12 (2122).  Verified against
# JPMorgan at 2025Q4: the groups sum to 1,497,490,000 exactly, which is 2122 to
# the dollar.  Three codes that look like they belong are memoranda and must
# stay out of the sum, because each is a subset of a line already counted:
#
#   1410  real-estate loans, total   -- the rollup of the RE group
#   B837  loans secured by RE in non-US offices -- a subset of the RE group
#   2746  loans to finance CRE, construction and land development *not*
#         secured by real estate -- reported inside C&I and other loans
#
# Adding B837 alone put JPMorgan's total 4,825,000 over its own reported total,
# which is how all three were found.
#
# Two leaves are coarser than the columns the EDGAR panel carries, because the
# pre-2007 and pre-2011 forms cannot support the finer split and a partition
# has to close in every quarter of the window:
#
#   loans_cre_nonfarm_nonres     F160 + F161, or 1480 before 2007
#   loans_consumer_installment   K137 + K207, or 2011 before 2011
#
# The finer columns (owner-occupied / investor CRE, auto / other consumer) are
# still produced, in RCC_DETAIL_ITEMS, and are null where the form did not
# break them out.

RCC_LOAN_ITEMS: tuple[ItemSpec, ...] = (
    # --- real estate ------------------------------------------------------
    ItemSpec(
        "loans_construction",
        "RCCI",
        ("F158", "F159"),
        alternatives=(("1415",),),
        description="1-4 family residential construction, and other construction and land development",
        since="2007Q1",
    ),
    ItemSpec("loans_farmland", "RCCI", ("1420",), description="Secured by farmland"),
    ItemSpec(
        "loans_home_equity",
        "RCCI",
        ("1797",),
        description="Revolving open-end lines secured by 1-4 family (HELOC)",
    ),
    ItemSpec(
        "loans_resi_mortgage",
        "RCCI",
        ("5367", "5368"),
        description="Closed-end 1-4 family, first and junior liens",
    ),
    ItemSpec("loans_multifamily", "RCCI", ("1460",), description="Secured by 5+ family"),
    ItemSpec(
        "loans_cre_nonfarm_nonres",
        "RCCI",
        ("F160", "F161"),
        alternatives=(("1480",),),
        description="Nonfarm nonresidential, owner-occupied and other together; 1480 before 2007",
    ),
    # --- commercial -------------------------------------------------------
    ItemSpec(
        "loans_ci",
        "RCCI",
        ("1763", "1764"),
        alternatives=(("1766",),),
        description="C&I, split by US and non-US addressee where the filer reports it",
    ),
    ItemSpec(
        "loans_depository_institutions",
        "RCCI",
        ("B532", "B533", "B534", "B536", "B537"),
        alternatives=(("1288",),),
        description="Loans to depository institutions and acceptances of other banks",
    ),
    ItemSpec(
        "loans_agricultural",
        "RCCI",
        ("1590",),
        description="Loans to finance agricultural production",
    ),
    ItemSpec(
        "loans_municipal",
        "RCCI",
        ("2107",),
        description="Obligations of states and political subdivisions",
    ),
    ItemSpec(
        "loans_lease",
        "RCCI",
        ("F162", "F163"),
        alternatives=(("2165",),),
        description="Lease financing receivables",
        since="2007Q1",
    ),
    # --- consumer ---------------------------------------------------------
    ItemSpec("loans_credit_card", "RCCI", ("B538",), description="Credit cards"),
    ItemSpec(
        "loans_consumer_revolving_other",
        "RCCI",
        ("B539",),
        description="Other revolving credit plans (not credit cards)",
    ),
    ItemSpec(
        "loans_consumer_installment",
        "RCCI",
        ("K137", "K207"),
        alternatives=(("2011",),),
        description="Other loans to individuals: automobile and other consumer together; 2011 before 2011",
    ),
    # --- everything else --------------------------------------------------
    ItemSpec(
        "loans_other_total",
        "RCCI",
        # RC-C item 9 in full: 9.a loans to nondepository financial
        # institutions, plus 9.b other loans.  Three reporting shapes exist and
        # a filer can publish more than one of them at once:
        #   1563              the whole of item 9
        #   J454 + J464       its two halves
        #   J454 + the three lines inside 9.b
        # Morgan Stanley Bank reports 1563 (57,175) alongside J454 (51,150),
        # J451 (5,066) and 1545 (959) -- which sum to 1563 exactly -- so the
        # coarsest present has to win or the same $57bn is counted twice.
        # Before 2010 form 031 broke 1563 into 2081, 1545 and 1564 ("all other
        # loans"); 1564 is listed so that a pre-2010 filer reporting the detail
        # is read in full rather than as 1545 + 2081 alone.
        ("J454", "1545", "2081", "J451", "1564"),
        alternatives=(("1563",), ("J454", "J464")),
        description="Loans to nondepository financial institutions, and all other loans",
    ),
)

# Detail available inside a leaf above, carried for comparison with the EDGAR
# panel but **not** part of the partition -- adding it would double count.
# ``loans_securities_based`` is a component of ``loans_other``, and is null
# rather than zero for a filer that reports only the J464 rollup: its
# securities lending is real but not separable from the rest of item 9.b.
RCC_DETAIL_ITEMS: tuple[ItemSpec, ...] = (
    ItemSpec(
        "loans_cre_owner_occupied",
        "RCCI",
        ("F160",),
        description="Owner-occupied nonfarm nonresidential; null before 2007",
        since="2007Q1",
    ),
    ItemSpec(
        "loans_cre_investor",
        "RCCI",
        ("F161",),
        description="Other (non-owner-occupied) nonfarm nonresidential; null before 2007",
        since="2007Q1",
    ),
    ItemSpec(
        "loans_auto",
        "RCCI",
        ("K137",),
        description="Automobile loans; null before 2011",
        since="2011Q1",
    ),
    ItemSpec(
        "loans_consumer_other",
        "RCCI",
        ("B539", "K207"),
        description="Other revolving credit plans, and other consumer loans; null before 2011",
        since="2011Q1",
    ),
    ItemSpec(
        "loans_securities_based",
        "RCCI",
        ("1545",),
        description="Loans for purchasing or carrying securities",
    ),
    ItemSpec(
        "loans_nondepository_fi",
        "RCCI",
        ("J454",),
        description="Loans to nondepository financial institutions; null before 2010",
        since="2010Q1",
    ),
    ItemSpec(
        "loans_other",
        "RCCI",
        ("1545", "2081", "J451", "1564"),
        alternatives=(("J464",),),
        description="Other loans, excluding nondepository financial institutions",
    ),
    ItemSpec(
        "loans_tdr_accruing",
        "RCCI",
        (
            "K158", "K159", "K160", "K161", "K162",
            "K163", "K164", "K256", "K165",
        ),
        alternatives=(("HK25",), ("1616",)),
        prefer="first",
        description=(
            "Restructured loans in compliance with modified terms (accruing TDRs): "
            "HK25 from 2017, 1616 before 2011, the category lines between. "
            "From 2023Q1 the form reports modifications to borrowers in financial "
            "difficulty (ASU 2022-02), a broader population than a TDR."
        ),
    ),
)

# The partition's own total, and the two rollups that are *not* summable.
RCC_TOTAL = ItemSpec(
    "loans_rcc_total",
    "RCCI",
    ("2122",),
    description="RC-C total loans and leases, net of unearned income (includes held for sale)",
)
RCC_UNEARNED = ItemSpec("loans_unearned_income", "RCCI", ("2123",))

# Columns assembled from the leaves above rather than read from a code.
# ``cre_total`` follows bankqtr_db.taxonomy, where construction and
# multifamily are children of CRE.
DERIVED_LOAN_GROUPS: dict[str, tuple[str, ...]] = {
    "loans_cre_total": (
        "loans_cre_nonfarm_nonres",
        "loans_construction",
        "loans_multifamily",
    ),
    "loans_financial_institutions": (
        "loans_depository_institutions",
        "loans_nondepository_fi",
    ),
    "loans_commercial_total": (
        "loans_cre_total",
        "loans_ci",
        "loans_depository_institutions",
        "loans_nondepository_fi",
        "loans_municipal",
        "loans_lease",
    ),
    "loans_consumer_total": (
        "loans_resi_mortgage",
        "loans_home_equity",
        "loans_credit_card",
        "loans_consumer_revolving_other",
        "loans_consumer_installment",
    ),
    # Everything consumer that is not a card or a mortgage, on both eras'
    # terms: the denominator for ``charge_offs_consumer_noncard``.
    "loans_consumer_noncard": (
        "loans_consumer_revolving_other",
        "loans_consumer_installment",
    ),
}

# --------------------------------------------------------------------------
# Schedule RC -- balance sheet, with RC-B, RC-E, RC-K and RC-M
# --------------------------------------------------------------------------

RC_ITEMS: tuple[ItemSpec, ...] = (
    ItemSpec(
        "loans_total",
        "RC",
        ("B528",),
        description="Loans and leases held for investment, net of unearned income",
    ),
    ItemSpec("loans_held_for_sale", "RC", ("5369",)),
    ItemSpec("acl_total", "RC", ("3123",), description="Allowance for credit losses on loans"),
    ItemSpec("assets", "RC", ("2170",)),
    ItemSpec("equity", "RC", ("3210",)),
    ItemSpec("oreo", "RC", ("2150",), description="Other real estate owned"),
    ItemSpec(
        "deposits",
        "RC",
        ("2200",),
        prefixes=("RCON",),
        plus=(("RCFN", ("2200",)),),
        description="Total deposits, domestic offices plus foreign offices",
    ),
    ItemSpec("trading_assets", "RC", ("3545",)),
    ItemSpec("securities_afs", "RC", ("1773",), description="Available-for-sale securities, fair value"),
    # --- added for the 2001 window --------------------------------------------
    ItemSpec(
        "securities_htm",
        "RCB",
        ("1754",),
        description="Held-to-maturity securities, amortised cost (RC-B, which keeps the code after RC dropped it in 2019)",
    ),
    ItemSpec(
        "securities_htm_fair_value",
        "RCB",
        ("1771",),
        description="Held-to-maturity securities, fair value",
    ),
    ItemSpec(
        "securities_afs_amortized_cost",
        "RCB",
        ("1772",),
        description="Available-for-sale securities, amortised cost",
    ),
    ItemSpec(
        "goodwill",
        ("RC", "RCM"),
        ("3163",),
        description="Goodwill: RC item 10.a to 2018Q1, RC-M item 2.a from 2018Q2",
    ),
    ItemSpec(
        "intangibles_total",
        ("RC", "RCM"),
        ("2143",),
        alternatives=(("3163", "0426"),),
        prefer="first",
        description="Goodwill plus other intangible assets: 2143 from 2018Q2, 3163 + 0426 before",
    ),
    ItemSpec(
        "preferred_stock",
        "RC",
        ("3838",),
        description="Perpetual preferred stock and related surplus (TARP preferred lands here)",
    ),
    ItemSpec(
        "fed_funds_repo_purchased",
        "RC",
        ("B993", "B995"),
        alternatives=(("2800",),),
        description="Federal funds purchased and securities sold under repurchase agreements; 2800 in 2001",
    ),
    ItemSpec(
        "borrowings_other",
        "RC",
        ("3190",),
        description="Other borrowed money, including FHLB advances",
    ),
    ItemSpec(
        "deposits_brokered",
        ("RCE", "RCEI"),
        ("2365",),
        prefixes=("RCON",),
        description="Total brokered deposits (RC-E memorandum 1.b)",
    ),
    ItemSpec(
        "assets_average",
        "RCK",
        ("3368",),
        description="Quarterly average of total assets (RC-K)",
    ),
    ItemSpec(
        "loans_average",
        "RCK",
        ("3360",),
        prefixes=("RCON",),
        plus=(("RCFN", ("3360",)),),
        description="Quarterly average of total loans, domestic plus foreign offices (RC-K)",
    ),
)

# --------------------------------------------------------------------------
# Schedule RC-L -- unused commitments and standby letters of credit
# --------------------------------------------------------------------------
#
# Off-balance-sheet exposure is what a stress scenario draws on, and every line
# here has been on the form since before 2001; only the codes moved.

RCL_ITEMS: tuple[ItemSpec, ...] = (
    ItemSpec("commitments_home_equity", "RCL", ("3814",), description="Unused HELOC commitments"),
    ItemSpec("commitments_credit_card", "RCL", ("3815",), description="Unused credit card lines"),
    ItemSpec(
        "commitments_cre_construction",
        "RCL",
        ("F164", "F165"),
        alternatives=(("3816",),),
        description="Commitments to fund CRE, construction and land development loans secured by real estate; 3816 before 2007",
    ),
    ItemSpec(
        "commitments_cre_not_secured",
        "RCL",
        ("6550",),
        description="Commitments to fund CRE, construction and land development loans not secured by real estate",
    ),
    ItemSpec(
        "commitments_securities_underwriting", "RCL", ("3817",)
    ),
    ItemSpec(
        "commitments_other",
        "RCL",
        ("J457", "J458", "J459"),
        alternatives=(("3818",),),
        description="Other unused commitments (C&I, financial institutions, all other); 3818 before 2010",
    ),
    ItemSpec(
        "standby_letters_of_credit",
        "RCL",
        ("3819",),
        description="Financial standby letters of credit",
    ),
)

COMMITMENT_COMPONENTS: tuple[str, ...] = (
    "commitments_home_equity",
    "commitments_credit_card",
    "commitments_cre_construction",
    "commitments_cre_not_secured",
    "commitments_securities_underwriting",
    "commitments_other",
)

# --------------------------------------------------------------------------
# Schedule RC-N -- past due and nonaccrual
# --------------------------------------------------------------------------
#
# The three columns are: A past due 30-89 and still accruing, B past due 90 or
# more and still accruing, C nonaccrual.  Note the prefix split inside a single
# filing -- JPMorgan reports 1403/1406/1407 and its C&I lines on RCFD, and its
# entire real-estate breakdown on RCON.
#
# **The form had no total row before 2017Q1.**  1403/1406/1407 exist from then
# and not before, so a 2013 build that reads only them carries nothing for its
# first four years.  Before 2017 the total is the sum of the category rows, and
# the category grid below is complete enough to reproduce the form's own total
# from 2017 on -- which is how it is checked (``tests/test_callrpt.py``).
#
# The RC-N row for nonfarm nonresidential lays its three columns out as
# F178/F180/F182 (owner-occupied) and F179/F181/F183 (other).  An earlier
# version of this file read F180/F181 -- the *90-days-past-due* column -- as
# the nonaccrual one, and JPMorgan's nonaccrual owner-occupied CRE came out at
# $1m.  Nonaccrual is F182/F183.

RCN_TOTALS: tuple[ItemSpec, ...] = (
    ItemSpec("pd_dpd_30_89", "RCN", ("1406",), description="Past due 30-89 days, accruing; 1406 from 2017Q1, the category sum before", since="2017Q1"),
    ItemSpec("pd_dpd_90_plus", "RCN", ("1407",), description="Past due 90+ days, accruing; 1407 from 2017Q1, the category sum before", since="2017Q1"),
    ItemSpec("nonaccrual_total", "RCN", ("1403",), description="Nonaccrual; 1403 from 2017Q1, the category sum before", since="2017Q1"),
)


def _rcn(
    category: str,
    dpd30: tuple[str, ...],
    dpd90: tuple[str, ...],
    nonaccrual: tuple[str, ...],
    *,
    alt30: tuple[tuple[str, ...], ...] = (),
    alt90: tuple[tuple[str, ...], ...] = (),
    altna: tuple[tuple[str, ...], ...] = (),
    since: str = "2001Q1",
) -> tuple[ItemSpec, ...]:
    return (
        ItemSpec(f"dpd_30_89_{category}", "RCN", dpd30, alternatives=alt30, since=since),
        ItemSpec(f"dpd_90_plus_{category}", "RCN", dpd90, alternatives=alt90, since=since),
        ItemSpec(f"nonaccrual_{category}", "RCN", nonaccrual, alternatives=altna, since=since),
    )


# The fourteen rows that partition RC-N's loan total, in form order.
#
# Agricultural loans (1594/1597/1583) are *not* among them.  They look like a
# row and they are a memorandum: the same dollars sit inside "all other loans"
# (5459/5460/5461), and for 877 of the 1,017 filers reporting agricultural
# past-dues in 2001Q1 the two lines are identical to the dollar -- the rural
# bank whose other loans are all farm loans.  Counting both put 523 filers
# over the form's own 2017Q1 total; leaving it out, 99.3% tie.
#
# Nor are real-estate loans to non-US addressees (1248-1250): Mercantil Bank's
# 2017Q1 sum came out 60% over its own total by exactly that line.  What form
# 031 does add is the foreign-office column: its real-estate rows are domestic
# only, and the loans booked abroad are one line on the ``RCFN`` prefix
# (B572-B574).  With it, 49 of the 70 031 filers that were short tie exactly.
RCN_CATEGORIES: tuple[str, ...] = (
    "construction",
    "farmland",
    "home_equity",
    "resi_mortgage",
    "multifamily",
    "cre_nonfarm_nonres",
    "foreign_office",
    "depository_institutions",
    "ci",
    "credit_card",
    "consumer_installment",
    "foreign_govt",
    "other_loans",
    "lease",
)

RCN_BY_CATEGORY: tuple[ItemSpec, ...] = (
    *_rcn("construction", ("F172", "F173"), ("F174", "F175"), ("F176", "F177"),
          alt30=(("2759",),), alt90=(("2769",),), altna=(("3492",),), since="2007Q1"),
    *_rcn("farmland", ("3493",), ("3494",), ("3495",)),
    *_rcn("home_equity", ("5398",), ("5399",), ("5400",)),
    *_rcn("resi_mortgage", ("C236", "C238"), ("C237", "C239"), ("C229", "C230"),
          alt30=(("5401",),), alt90=(("5402",),), altna=(("5403",),), since="2002Q1"),
    *_rcn("multifamily", ("3499",), ("3500",), ("3501",)),
    *_rcn("cre_nonfarm_nonres", ("F178", "F179"), ("F180", "F181"), ("F182", "F183"),
          alt30=(("3502",),), alt90=(("3503",),), altna=(("3504",),), since="2007Q1"),
    # Form 031 only: loans booked in foreign offices, the consolidating line
    # for a schedule whose real-estate rows are domestic.
    ItemSpec("dpd_30_89_foreign_office", "RCN", ("B572",), prefixes=FOREIGN_PREFIXES),
    ItemSpec("dpd_90_plus_foreign_office", "RCN", ("B573",), prefixes=FOREIGN_PREFIXES),
    ItemSpec("nonaccrual_foreign_office", "RCN", ("B574",), prefixes=FOREIGN_PREFIXES),
    # A memorandum, carried as detail: real estate loans to non-US addressees.
    *_rcn("re_foreign", ("1248",), ("1249",), ("1250",)),
    # Form 041 carries a derived single line; 031 splits US banks from
    # foreign banks.
    *_rcn("depository_institutions", ("5377", "5380"), ("5378", "5381"), ("5379", "5382"),
          alt30=(("B834",),), alt90=(("B835",),), altna=(("B836",),)),
    *_rcn("agricultural", ("1594",), ("1597",), ("1583",)),
    *_rcn("ci", ("1251", "1254"), ("1252", "1255"), ("1253", "1256"),
          alt30=(("1606",),), alt90=(("1607",),), altna=(("1608",),)),
    *_rcn("credit_card", ("B575",), ("B576",), ("B577",)),
    *_rcn("consumer_installment", ("K213", "K216"), ("K214", "K217"), ("K215", "K218"),
          alt30=(("B578",),), alt90=(("B579",),), altna=(("B580",),), since="2011Q1"),
    # Folded into "all other loans" from 2017; null thereafter, which the
    # total sum treats as zero.
    *_rcn("foreign_govt", ("5389",), ("5390",), ("5391",)),
    *_rcn("other_loans", ("5459",), ("5460",), ("5461",)),
    # Leases: a derived single line on 041; on 031 the 2007+ split into
    # leases to individuals and all other, and before 2007 US against non-US.
    *_rcn("lease", ("F166", "F169"), ("F167", "F170"), ("F168", "F171"),
          alt30=(("1226",), ("1257", "1271")),
          alt90=(("1227",), ("1258", "1272")),
          altna=(("1228",), ("1259", "1791")),
          since="2007Q1"),
    # --- finer splits, not part of the total -----------------------------
    *_rcn("cre_owner_occupied", ("F178",), ("F180",), ("F182",), since="2007Q1"),
    *_rcn("cre_investor", ("F179",), ("F181",), ("F183",), since="2007Q1"),
    *_rcn("auto", ("K213",), ("K214",), ("K215",), since="2011Q1"),
    *_rcn("consumer_other", ("K216",), ("K217",), ("K218",), since="2011Q1"),
)

RCN_TOTAL_COMPONENTS: dict[str, tuple[str, ...]] = {
    "pd_dpd_30_89": tuple(f"dpd_30_89_{c}" for c in RCN_CATEGORIES),
    "pd_dpd_90_plus": tuple(f"dpd_90_plus_{c}" for c in RCN_CATEGORIES),
    "nonaccrual_total": tuple(f"nonaccrual_{c}" for c in RCN_CATEGORIES),
}

# --------------------------------------------------------------------------
# Schedule RI-B Part I -- charge-offs and recoveries, and Part II -- allowance
# --------------------------------------------------------------------------


def _flow(name: str, schedule: str, items: tuple[str, ...], **kw) -> ItemSpec:
    return ItemSpec(name, schedule, items, kind="flow", prefixes=INCOME_PREFIXES, **kw)


RIB_ITEMS: tuple[ItemSpec, ...] = (
    _flow("charge_offs_total", "RIBI", ("4635",)),
    _flow("recoveries_total", "RIBI", ("4605",)),
    # Form 041 reports C&I on a single line (4638/4608); 031 splits by
    # addressee.  Without the single line every 041 filer's C&I charge-offs
    # were null.
    _flow("charge_offs_ci", "RIBI", ("4645", "4646"), alternatives=(("4638",),)),
    _flow("recoveries_ci", "RIBI", ("4617", "4618"), alternatives=(("4608",),)),
    _flow("charge_offs_construction", "RIBI", ("C891", "C893"), alternatives=(("3582",),), since="2007Q1"),
    _flow("recoveries_construction", "RIBI", ("C892", "C894"), alternatives=(("3583",),), since="2007Q1"),
    _flow("charge_offs_multifamily", "RIBI", ("3588",)),
    _flow("recoveries_multifamily", "RIBI", ("3589",)),
    _flow("charge_offs_cre_nonfarm_nonres", "RIBI", ("C895", "C897"), alternatives=(("3590",),),
          description="Nonfarm nonresidential charge-offs, both occupancy classes; 3590 before 2007"),
    _flow("recoveries_cre_nonfarm_nonres", "RIBI", ("C896", "C898"), alternatives=(("3591",),)),
    _flow("charge_offs_cre_owner_occupied", "RIBI", ("C895",), since="2007Q1"),
    _flow("recoveries_cre_owner_occupied", "RIBI", ("C896",), since="2007Q1"),
    _flow("charge_offs_cre_investor", "RIBI", ("C897",), since="2007Q1"),
    _flow("recoveries_cre_investor", "RIBI", ("C898",), since="2007Q1"),
    _flow("charge_offs_resi_mortgage", "RIBI", ("C234", "C235"), alternatives=(("5413",),), since="2002Q1"),
    _flow("recoveries_resi_mortgage", "RIBI", ("C217", "C218"), alternatives=(("5414",),), since="2002Q1"),
    _flow("charge_offs_home_equity", "RIBI", ("5411",)),
    _flow("recoveries_home_equity", "RIBI", ("5412",)),
    _flow("charge_offs_credit_card", "RIBI", ("B514",)),
    _flow("recoveries_credit_card", "RIBI", ("B515",)),
    _flow("charge_offs_auto", "RIBI", ("K129",), since="2011Q1"),
    _flow("recoveries_auto", "RIBI", ("K133",), since="2011Q1"),
    # Consumer other than cards and mortgages, on both eras' terms: auto plus
    # other consumer from 2011, the single "other loans to individuals" line
    # before.  The denominator is ``loans_consumer_noncard``.
    _flow("charge_offs_consumer_noncard", "RIBI", ("K129", "K205"), alternatives=(("B516",),)),
    _flow("recoveries_consumer_noncard", "RIBI", ("K133", "K206"), alternatives=(("B517",),)),
    # 4266/4267 are carried as a derived total for every filer in every
    # quarter; the 2007+ detail is preferred where a filer reports it.
    _flow("charge_offs_lease", "RIBI", ("F185", "C880"), alternatives=(("4266",),)),
    _flow("recoveries_lease", "RIBI", ("F187", "F188"), alternatives=(("4267",),)),
    # --- Part II: the allowance rollforward -------------------------------
    _flow("provision_total", "RIBII", ("4230",), description="Provision for credit losses"),
    _flow(
        "acl_beginning",
        "RIBII",
        ("B522",),
        description="Allowance balance most recently reported for the prior period",
    ),
)

# --------------------------------------------------------------------------
# Schedule RI -- income, and RC-R Part I -- regulatory capital
# --------------------------------------------------------------------------

RI_ITEMS: tuple[ItemSpec, ...] = (
    _flow("net_income", "RI", ("4340",)),
    _flow("net_interest_income", "RI", ("4074",)),
    _flow("noninterest_income", "RI", ("4079",)),
    _flow("noninterest_expense", "RI", ("4093",)),
    _flow("interest_income", "RI", ("4107",)),
    _flow("interest_expense", "RI", ("4073",)),
)

# Basel III, 2015Q1 onward.  Left null before that rather than approximated
# from the pre-Basel III items -- the definitions differ, and a CET1 ratio
# that silently becomes a Tier 1 ratio in 2014 is worse than a gap.
RCR_ITEMS: tuple[ItemSpec, ...] = (
    ItemSpec("cet1_capital", "RCRI", ("P859",), prefixes=CAPITAL_PREFIXES, since="2015Q1"),
    ItemSpec("tier1_capital", "RCRI", ("8274",), prefixes=CAPITAL_PREFIXES, since="2015Q1"),
    ItemSpec("total_capital", "RCRI", ("3792",), prefixes=CAPITAL_PREFIXES, since="2015Q1"),
    ItemSpec("risk_weighted_assets", "RCRI", ("A223",), prefixes=CAPITAL_PREFIXES, since="2015Q1"),
)

# The regime before it, under its own names.  Same item numbers, different
# schedule and prefixes, and -- the reason for the separate columns -- a
# different definition of capital and of risk weights.  Schedule RCR carries
# 2001Q1-2013Q4; 2014 is the transition year, on RCRIA for most banks and
# RCRIB for the twelve advanced-approaches banks.
RCR_LEGACY_ITEMS: tuple[ItemSpec, ...] = (
    ItemSpec(
        "tier1_capital_basel1",
        ("RCR", "RCRIA", "RCRIB"),
        ("8274",),
        prefixes=LEGACY_CAPITAL_PREFIXES,
        description="Tier 1 capital under the pre-2015 general risk-based rules",
    ),
    ItemSpec(
        "total_capital_basel1",
        ("RCR", "RCRIA", "RCRIB"),
        ("3792",),
        prefixes=LEGACY_CAPITAL_PREFIXES,
        description="Total risk-based capital under the pre-2015 rules",
    ),
    ItemSpec(
        "risk_weighted_assets_basel1",
        ("RCR", "RCRIA", "RCRIB"),
        ("A223",),
        prefixes=LEGACY_CAPITAL_PREFIXES,
        description="Risk-weighted assets under the pre-2015 rules",
    ),
)

ALL_ITEMS: tuple[ItemSpec, ...] = (
    *RCC_LOAN_ITEMS,
    *RCC_DETAIL_ITEMS,
    RCC_TOTAL,
    RCC_UNEARNED,
    *RC_ITEMS,
    *RCL_ITEMS,
    *RCN_TOTALS,
    *RCN_BY_CATEGORY,
    *RIB_ITEMS,
    *RI_ITEMS,
    *RCR_ITEMS,
    *RCR_LEGACY_ITEMS,
)

BY_NAME: dict[str, ItemSpec] = {spec.name: spec for spec in ALL_ITEMS}
assert len(BY_NAME) == len(ALL_ITEMS), "duplicate ItemSpec name"

FLOW_COLUMNS: frozenset[str] = frozenset(
    spec.name for spec in ALL_ITEMS if spec.kind == "flow"
)


def wanted_codes() -> dict[str, set[str]]:
    """Schedule -> the full MDRM codes to keep when reading a bulk file.

    Reading every code for every filer is 3,843 columns by 4,400 institutions
    by 102 quarters, and almost none of it is wanted.  Filtering at the parse
    is what keeps a full build in memory.
    """
    out: dict[str, set[str]] = {}
    for spec in ALL_ITEMS:
        for schedule in spec.schedules:
            codes = out.setdefault(schedule, set())
            for prefix in spec.prefixes:
                codes.update(prefix + item for item in spec.all_items())
            for prefix, items in spec.plus:
                codes.update(prefix + item for item in items)
    return out


# --------------------------------------------------------------------------
# Arithmetic checks
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PartitionCheck:
    """A set of columns that must sum to another column, within tolerance.

    A Call Report schedule states its own total, so a category mapping can be
    *checked* rather than trusted -- which is the one advantage this source has
    over the XBRL panel, where nothing cross-foots and ``mix_coverage_pct`` is
    the best that can be done.  A failure here means the mapping is wrong, not
    that a bank disclosed oddly.
    """

    name: str
    parts: tuple[str, ...]
    total: str
    tolerance: float = 1e-6


PARTITION_CHECKS: tuple[PartitionCheck, ...] = (
    PartitionCheck(
        "rcc_loan_partition",
        tuple(spec.name for spec in RCC_LOAN_ITEMS),
        "loans_rcc_total",
    ),
    PartitionCheck(
        "held_for_investment_plus_held_for_sale",
        ("loans_total", "loans_held_for_sale"),
        "loans_rcc_total",
    ),
    # Only checkable from 2017Q1, when the form first states the total.
    PartitionCheck("rcn_nonaccrual_partition", RCN_TOTAL_COMPONENTS["nonaccrual_total"], "nonaccrual_total"),
    PartitionCheck("rcn_dpd_30_89_partition", RCN_TOTAL_COMPONENTS["pd_dpd_30_89"], "pd_dpd_30_89"),
    PartitionCheck("rcn_dpd_90_plus_partition", RCN_TOTAL_COMPONENTS["pd_dpd_90_plus"], "pd_dpd_90_plus"),
)


def partition_checks() -> tuple[PartitionCheck, ...]:
    return PARTITION_CHECKS
