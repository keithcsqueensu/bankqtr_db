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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Kind = Literal["stock", "flow"]

# Call Report amounts are in thousands of dollars; the panel is in dollars.
UNIT_SCALE = 1_000.0

# Prefix families, in the order they are tried.  A stock item is looked for on
# the consolidated basis first and the domestic one second; an income item has
# only one basis; a regulatory-capital item is reported on RCFA/RCOA.
BALANCE_PREFIXES = ("RCFD", "RCON")
INCOME_PREFIXES = ("RIAD",)
CAPITAL_PREFIXES = ("RCFA", "RCOA")
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
    """

    name: str
    schedule: str
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
# These nine groups are a *partition* of RC-C item 12 (2122).  Verified against
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

RCC_LOAN_ITEMS: tuple[ItemSpec, ...] = (
    # --- real estate ------------------------------------------------------
    ItemSpec(
        "loans_construction",
        "RCCI",
        ("F158", "F159"),
        description="1-4 family residential construction, and other construction and land development",
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
        "loans_cre_owner_occupied",
        "RCCI",
        ("F160",),
        description="Owner-occupied nonfarm nonresidential",
    ),
    ItemSpec(
        "loans_cre_investor",
        "RCCI",
        ("F161",),
        description="Other (non-owner-occupied) nonfarm nonresidential",
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
    ),
    # --- consumer ---------------------------------------------------------
    ItemSpec("loans_credit_card", "RCCI", ("B538",), description="Credit cards"),
    ItemSpec("loans_auto", "RCCI", ("K137",), description="Automobile loans"),
    ItemSpec(
        "loans_consumer_other",
        "RCCI",
        ("B539", "K207"),
        description="Other revolving credit plans, and other consumer loans",
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
        ("J454", "1545", "2081", "J451"),
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
        "loans_securities_based",
        "RCCI",
        ("1545",),
        description="Loans for purchasing or carrying securities",
    ),
    ItemSpec(
        "loans_nondepository_fi",
        "RCCI",
        ("J454",),
        description="Loans to nondepository financial institutions",
    ),
    ItemSpec(
        "loans_other",
        "RCCI",
        ("1545", "2081", "J451"),
        alternatives=(("J464",),),
        description="Other loans, excluding nondepository financial institutions",
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
        "loans_cre_owner_occupied",
        "loans_cre_investor",
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
        "loans_auto",
        "loans_consumer_other",
    ),
}

# --------------------------------------------------------------------------
# Schedule RC -- balance sheet
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
    ItemSpec("securities_afs", "RC", ("1773",)),
)

# --------------------------------------------------------------------------
# Schedule RC-N -- past due and nonaccrual
# --------------------------------------------------------------------------
#
# The three columns are: A past due 30-89 and still accruing, B past due 90 or
# more and still accruing, C nonaccrual.  Note the prefix split inside a single
# filing -- JPMorgan reports 1403/1406/1407 and its C&I lines on RCFD, and its
# entire real-estate breakdown on RCON.

RCN_TOTALS: tuple[ItemSpec, ...] = (
    ItemSpec("pd_dpd_30_89", "RCN", ("1406",), description="Past due 30-89 days, accruing"),
    ItemSpec("pd_dpd_90_plus", "RCN", ("1407",), description="Past due 90+ days, accruing"),
    ItemSpec("nonaccrual_total", "RCN", ("1403",), description="Nonaccrual"),
)

# Per-category delinquency, as (column, category) -> items.  Only the
# categories the Call Report actually breaks out appear; the rest stay null in
# the panel rather than being filled from a neighbouring line.
RCN_BY_CATEGORY: tuple[ItemSpec, ...] = (
    ItemSpec("nonaccrual_ci", "RCN", ("1253", "1256")),
    ItemSpec("nonaccrual_credit_card", "RCN", ("B577",)),
    ItemSpec("nonaccrual_auto", "RCN", ("K215",)),
    ItemSpec("nonaccrual_multifamily", "RCN", ("3501",)),
    ItemSpec("nonaccrual_construction", "RCN", ("F176", "F177")),
    ItemSpec("nonaccrual_cre_owner_occupied", "RCN", ("F180",)),
    ItemSpec("nonaccrual_cre_investor", "RCN", ("F181",)),
    ItemSpec("nonaccrual_resi_mortgage", "RCN", ("C229", "C230")),
    ItemSpec("nonaccrual_home_equity", "RCN", ("5400",)),
    ItemSpec("dpd_30_89_ci", "RCN", ("1251", "1254")),
    ItemSpec("dpd_30_89_credit_card", "RCN", ("B575",)),
    ItemSpec("dpd_90_plus_credit_card", "RCN", ("B576",)),
)

# --------------------------------------------------------------------------
# Schedule RI-B Part I -- charge-offs and recoveries, and Part II -- allowance
# --------------------------------------------------------------------------

RIB_ITEMS: tuple[ItemSpec, ...] = (
    ItemSpec("charge_offs_total", "RIBI", ("4635",), kind="flow", prefixes=INCOME_PREFIXES),
    ItemSpec("recoveries_total", "RIBI", ("4605",), kind="flow", prefixes=INCOME_PREFIXES),
    ItemSpec(
        "charge_offs_ci", "RIBI", ("4645", "4646"), kind="flow", prefixes=INCOME_PREFIXES
    ),
    ItemSpec(
        "recoveries_ci", "RIBI", ("4617", "4618"), kind="flow", prefixes=INCOME_PREFIXES
    ),
    ItemSpec(
        "charge_offs_construction",
        "RIBI",
        ("C891", "C893"),
        kind="flow",
        prefixes=INCOME_PREFIXES,
    ),
    ItemSpec(
        "recoveries_construction",
        "RIBI",
        ("C892", "C894"),
        kind="flow",
        prefixes=INCOME_PREFIXES,
    ),
    ItemSpec(
        "charge_offs_multifamily", "RIBI", ("3588",), kind="flow", prefixes=INCOME_PREFIXES
    ),
    ItemSpec(
        "recoveries_multifamily", "RIBI", ("3589",), kind="flow", prefixes=INCOME_PREFIXES
    ),
    ItemSpec(
        "charge_offs_cre_owner_occupied",
        "RIBI",
        ("C895",),
        kind="flow",
        prefixes=INCOME_PREFIXES,
    ),
    ItemSpec(
        "recoveries_cre_owner_occupied",
        "RIBI",
        ("C896",),
        kind="flow",
        prefixes=INCOME_PREFIXES,
    ),
    ItemSpec(
        "charge_offs_cre_investor", "RIBI", ("C897",), kind="flow", prefixes=INCOME_PREFIXES
    ),
    ItemSpec(
        "recoveries_cre_investor", "RIBI", ("C898",), kind="flow", prefixes=INCOME_PREFIXES
    ),
    ItemSpec(
        "charge_offs_resi_mortgage",
        "RIBI",
        ("C234", "C235"),
        kind="flow",
        prefixes=INCOME_PREFIXES,
    ),
    ItemSpec(
        "recoveries_resi_mortgage",
        "RIBI",
        ("C217", "C218"),
        kind="flow",
        prefixes=INCOME_PREFIXES,
    ),
    ItemSpec(
        "charge_offs_home_equity", "RIBI", ("5411",), kind="flow", prefixes=INCOME_PREFIXES
    ),
    ItemSpec(
        "recoveries_home_equity", "RIBI", ("5412",), kind="flow", prefixes=INCOME_PREFIXES
    ),
    ItemSpec(
        "charge_offs_credit_card", "RIBI", ("B514",), kind="flow", prefixes=INCOME_PREFIXES
    ),
    ItemSpec(
        "recoveries_credit_card", "RIBI", ("B515",), kind="flow", prefixes=INCOME_PREFIXES
    ),
    ItemSpec("charge_offs_auto", "RIBI", ("K129",), kind="flow", prefixes=INCOME_PREFIXES),
    ItemSpec("recoveries_auto", "RIBI", ("K133",), kind="flow", prefixes=INCOME_PREFIXES),
    ItemSpec(
        "charge_offs_lease",
        "RIBI",
        ("F185", "C880"),
        kind="flow",
        prefixes=INCOME_PREFIXES,
    ),
    ItemSpec(
        "recoveries_lease",
        "RIBI",
        ("F187", "F188"),
        kind="flow",
        prefixes=INCOME_PREFIXES,
    ),
    # --- Part II: the allowance rollforward -------------------------------
    ItemSpec(
        "provision_total",
        "RIBII",
        ("4230",),
        kind="flow",
        prefixes=INCOME_PREFIXES,
        description="Provision for credit losses",
    ),
    ItemSpec(
        "acl_beginning",
        "RIBII",
        ("B522",),
        kind="flow",
        prefixes=INCOME_PREFIXES,
        description="Allowance balance most recently reported for the prior period",
    ),
)

# --------------------------------------------------------------------------
# Schedule RI -- income, and RC-R Part I -- regulatory capital
# --------------------------------------------------------------------------

RI_ITEMS: tuple[ItemSpec, ...] = (
    ItemSpec("net_income", "RI", ("4340",), kind="flow", prefixes=INCOME_PREFIXES),
    ItemSpec("net_interest_income", "RI", ("4074",), kind="flow", prefixes=INCOME_PREFIXES),
    ItemSpec("noninterest_income", "RI", ("4079",), kind="flow", prefixes=INCOME_PREFIXES),
    ItemSpec("noninterest_expense", "RI", ("4093",), kind="flow", prefixes=INCOME_PREFIXES),
)

RCR_ITEMS: tuple[ItemSpec, ...] = (
    ItemSpec("cet1_capital", "RCRI", ("P859",), prefixes=CAPITAL_PREFIXES),
    ItemSpec("tier1_capital", "RCRI", ("8274",), prefixes=CAPITAL_PREFIXES),
    ItemSpec("total_capital", "RCRI", ("3792",), prefixes=CAPITAL_PREFIXES),
    ItemSpec("risk_weighted_assets", "RCRI", ("A223",), prefixes=CAPITAL_PREFIXES),
)

ALL_ITEMS: tuple[ItemSpec, ...] = (
    *RCC_LOAN_ITEMS,
    *RCC_DETAIL_ITEMS,
    RCC_TOTAL,
    RCC_UNEARNED,
    *RC_ITEMS,
    *RCN_TOTALS,
    *RCN_BY_CATEGORY,
    *RIB_ITEMS,
    *RI_ITEMS,
    *RCR_ITEMS,
)

BY_NAME: dict[str, ItemSpec] = {spec.name: spec for spec in ALL_ITEMS}

FLOW_COLUMNS: frozenset[str] = frozenset(
    spec.name for spec in ALL_ITEMS if spec.kind == "flow"
)


def wanted_codes() -> dict[str, set[str]]:
    """Schedule -> the full MDRM codes to keep when reading a bulk file.

    Reading every code for every filer is 3,843 columns by 4,400 institutions
    by 54 quarters, and almost none of it is wanted.  Filtering at the parse is
    what keeps a full build in memory.
    """
    out: dict[str, set[str]] = {}
    for spec in ALL_ITEMS:
        codes = out.setdefault(spec.schedule, set())
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
)


def partition_checks() -> tuple[PartitionCheck, ...]:
    return PARTITION_CHECKS
