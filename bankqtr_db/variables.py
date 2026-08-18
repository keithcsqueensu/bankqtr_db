"""Declarative definitions of the panel's canonical variables.

Each :class:`VarDef` says *what* to look for, not how to fetch it, so the same
spec drives the XBRL path, the HTML fallback and the coverage report.

Tag candidates are priority-ordered and were chosen from a frequency survey
across the whole universe rather than from the taxonomy documentation -- e.g.
the net-charge-off element exists under two different names depending on
whether a bank adopted the "ExcludingAccruedInterest" variant, and roughly a
tenth of the universe tags neither, so NCO also has a derivation from gross
write-offs minus recoveries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Kind = Literal["stock", "flow"]

# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VarDef:
    name: str
    kind: Kind
    tags: tuple[str, ...]
    # Which dimension axis the variable is sliced on.  ``None`` means the
    # consolidated, undimensioned fact -- the only thing companyfacts exposes.
    axis: Literal["loan", "credit_quality", "past_due"] | None = None
    unit: str = "USD"
    description: str = ""
    # Alternative construction when no tag matches, expressed over other
    # variable names: ("subtract", "a", "b") or ("sum", "a", "b", ...).
    derive: tuple[str, ...] | None = None
    # Row-label patterns for the HTML fallback path.
    html_labels: tuple[str, ...] = field(default=())


# --------------------------------------------------------------------------
# Balance-sheet stocks
# --------------------------------------------------------------------------

LOAN_TAGS = (
    "FinancingReceivableExcludingAccruedInterestBeforeAllowanceForCreditLoss",
    "LoansAndLeasesReceivableNetReportedAmount",
    "NotesReceivableGross",
)

ACL_TAGS = (
    "FinancingReceivableAllowanceForCreditLosses",
    "FinancingReceivableAllowanceForCreditLoss",
    "FinancingReceivableAllowanceForCreditLossExcludingAccruedInterest",
    "AllowanceForLoanAndLeaseLossesRealEstate",
)

NONACCRUAL_TAGS = (
    "FinancingReceivableExcludingAccruedInterestNonaccrual",
    "FinancingReceivableRecordedInvestmentNonaccrualStatus",
    "FinancingReceivableNonaccrualNoAllowance",
    "FinancingReceivableExcludingAccruedInterestNonaccrualNoAllowance",
)

STOCKS: tuple[VarDef, ...] = (
    VarDef(
        "loans",
        "stock",
        LOAN_TAGS,
        axis="loan",
        description="Loans held for investment, gross of allowance",
        html_labels=(
            "total loans",
            "total loans and leases",
            "loans and leases outstanding",
        ),
    ),
    VarDef(
        "acl",
        "stock",
        ACL_TAGS,
        axis="loan",
        description="Allowance for credit losses on loans",
        html_labels=(
            "allowance for credit losses",
            "allowance for loan",
            "balance, end of period",
            "ending balance",
        ),
    ),
    VarDef(
        "nonaccrual",
        "stock",
        NONACCRUAL_TAGS,
        axis="loan",
        description="Nonaccrual loans",
        html_labels=("nonaccrual", "non-accrual", "nonperforming loans"),
    ),
    VarDef(
        "oreo",
        "stock",
        (
            "OtherRealEstateAndForeclosedAssets",
            "RealEstateAcquiredThroughForeclosure",
            "ForeclosedAssets",
            "OtherRealEstate",
        ),
        description="Other real estate owned / foreclosed assets",
        html_labels=("other real estate", "foreclosed", "oreo"),
    ),
    VarDef(
        "loans_by_credit_quality",
        "stock",
        LOAN_TAGS,
        axis="credit_quality",
        description="Loans sliced by internal risk rating / credit quality indicator",
        html_labels=(
            "pass",
            "special mention",
            "substandard",
            "criticized",
            "classified",
            "doubtful",
        ),
    ),
    VarDef(
        "loans_by_past_due",
        "stock",
        LOAN_TAGS,
        axis="past_due",
        description="Loans sliced by delinquency bucket",
        html_labels=("30-89 days", "90 days or more", "past due", "current"),
    ),
    # Context variables for scaling and per-share work.
    VarDef("assets", "stock", ("Assets",), description="Total assets"),
    VarDef(
        "equity",
        "stock",
        ("StockholdersEquity",),
        description="Common + preferred equity",
    ),
    VarDef("deposits", "stock", ("Deposits",), description="Total deposits"),
    VarDef(
        "unfunded_commitments",
        "stock",
        ("OffBalanceSheetCreditLossLiability", "CommitmentsToExtendCredit"),
        description="Off-balance-sheet lending commitments",
    ),
)

# --------------------------------------------------------------------------
# Income-statement / rollforward flows
# --------------------------------------------------------------------------

FLOWS: tuple[VarDef, ...] = (
    VarDef(
        "nco",
        "flow",
        (
            "FinancingReceivableExcludingAccruedInterestAllowanceForCreditLossWriteoffAfterRecovery",
            "FinancingReceivableAllowanceForCreditLossWriteoffAfterRecovery",
        ),
        axis="loan",
        derive=("subtract", "charge_offs", "recoveries"),
        description="Net charge-offs",
        html_labels=("net charge-offs", "net charge offs", "net chargeoffs"),
    ),
    VarDef(
        "charge_offs",
        "flow",
        (
            "FinancingReceivableExcludingAccruedInterestAllowanceForCreditLossWriteoff",
            "FinancingReceivableAllowanceForCreditLossWriteoff",
        ),
        axis="loan",
        description="Gross charge-offs",
        html_labels=("gross charge-offs", "charge-offs", "chargeoffs"),
    ),
    VarDef(
        "recoveries",
        "flow",
        (
            "FinancingReceivableExcludingAccruedInterestAllowanceForCreditLossRecovery",
            "FinancingReceivableAllowanceForCreditLossesRecovery",
            "FinancingReceivableAllowanceForCreditLossesRecoveries",
            "AllowanceForLoanAndLeaseLossesRecoveriesOfBadDebts",
        ),
        axis="loan",
        description="Recoveries",
        html_labels=("recoveries",),
    ),
    VarDef(
        "provision",
        "flow",
        (
            "ProvisionForLoanLeaseAndOtherLosses",
            "ProvisionForLoanAndLeaseLosses",
            "ProvisionForLoanLossesExpensed",
            "ProvisionForCreditLosses",
        ),
        axis="loan",
        description="Provision for credit losses",
        html_labels=("provision for credit losses", "provision for loan"),
    ),
    VarDef("net_income", "flow", ("NetIncomeLoss",), description="Net income"),
    VarDef(
        "nii", "flow", ("InterestIncomeExpenseNet",), description="Net interest income"
    ),
    VarDef("revenue", "flow", ("Revenues",), description="Total revenue"),
)

ALL_VARS: tuple[VarDef, ...] = STOCKS + FLOWS
BY_NAME: dict[str, VarDef] = {v.name: v for v in ALL_VARS}

# Variables sliced by loan category produce one column per category.
LOAN_SLICED = tuple(v.name for v in ALL_VARS if v.axis == "loan")


# --------------------------------------------------------------------------
# Derived ratios computed once the raw panel exists
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RatioDef:
    name: str
    numerator: str
    denominator: str
    annualize: bool = False
    description: str = ""


RATIOS: tuple[RatioDef, ...] = (
    # ---- portfolio mix ----------------------------------------------------
    RatioDef(
        "cre_pct",
        "loans_cre_total",
        "loans_total",
        description="CRE as % of total loans",
    ),
    RatioDef(
        "construction_pct",
        "loans_construction",
        "loans_total",
        description="Construction as % of total loans",
    ),
    RatioDef(
        "ci_pct", "loans_ci", "loans_total", description="C&I as % of total loans"
    ),
    RatioDef("multifamily_pct", "loans_multifamily", "loans_total"),
    RatioDef("cre_investor_pct", "loans_cre_investor", "loans_total"),
    RatioDef("cre_owner_occupied_pct", "loans_cre_owner_occupied", "loans_total"),
    RatioDef("small_business_pct", "loans_small_business", "loans_total"),
    RatioDef("credit_card_pct", "loans_credit_card", "loans_total"),
    RatioDef("resi_pct", "loans_resi_mortgage", "loans_total"),
    RatioDef("consumer_pct", "loans_consumer_total", "loans_total"),
    RatioDef("commercial_pct", "loans_commercial_total", "loans_total"),
    # ---- asset quality ----------------------------------------------------
    RatioDef(
        "nco_rate",
        "nco_total",
        "loans_total",
        annualize=True,
        description="Annualised net charge-off rate",
    ),
    RatioDef("nco_rate_cre", "nco_cre_total", "loans_cre_total", annualize=True),
    RatioDef("nco_rate_ci", "nco_ci", "loans_ci", annualize=True),
    RatioDef("nco_rate_card", "nco_credit_card", "loans_credit_card", annualize=True),
    RatioDef(
        "npa_ratio",
        "npa_total",
        "loans_total",
        description="(Nonaccrual + OREO) / loans",
    ),
    RatioDef("nonaccrual_ratio", "nonaccrual_total", "loans_total"),
    RatioDef("reserve_coverage", "acl_total", "loans_total", description="ACL / loans"),
    RatioDef(
        "reserve_to_nonaccrual",
        "acl_total",
        "nonaccrual_total",
        description="ACL / nonaccrual loans",
    ),
    RatioDef("reserve_coverage_cre", "acl_cre_total", "loans_cre_total"),
    RatioDef("provision_to_nco", "provision_total", "nco_total"),
    # ---- risk appetite ----------------------------------------------------
    RatioDef(
        "criticized_pct",
        "cq_criticized",
        "loans_total",
        description="Criticized (special mention + classified) / loans",
    ),
    RatioDef(
        "classified_pct",
        "cq_classified",
        "loans_total",
        description="Classified (substandard + doubtful + loss) / loans",
    ),
    RatioDef("special_mention_pct", "cq_special_mention", "loans_total"),
    RatioDef("substandard_pct", "cq_substandard", "loans_total"),
    RatioDef(
        "investment_grade_pct",
        "cq_investment_grade",
        "cq_rated_total",
        description="Investment-grade share of internally rated exposure",
    ),
    # ---- delinquency ------------------------------------------------------
    RatioDef("dpd_30_89_pct", "pd_dpd_30_89", "loans_total"),
    RatioDef("dpd_90_plus_pct", "pd_dpd_90_plus", "loans_total"),
    # ---- scale ------------------------------------------------------------
    RatioDef("loans_to_assets", "loans_total", "assets"),
    RatioDef("loans_to_deposits", "loans_total", "deposits"),
    # ---- IR-supplement exposures -------------------------------------------
    # These have no XBRL element at any bank: office CRE and leveraged lending
    # are disclosed only in quarterly investor supplements and earnings decks.
    # The ratios are computed wherever ``ir_extract`` manages to populate their
    # inputs, and stay null everywhere else, exactly like every other ratio.
    RatioDef(
        "office_cre_pct",
        "loans_office_cre",
        "loans_total",
        description="Office CRE as % of total loans (IR supplement)",
    ),
    # ``office_cre_share_of_cre`` is deliberately NOT a RatioDef.  Deriving it
    # needs a complete CRE denominator and there often is not one: Regions'
    # ``loans_cre_total`` exists in two quarters out of nine, and in those two
    # it comes from the HTML fallback at $4.84bn against an investor-real-estate
    # book nearer $9bn -- which turned a ~15% office share into 31%.  The column
    # is still populated, but only where a bank *discloses* the share itself
    # (US Bancorp prints its CRE split by property class every quarter).
    # ``office_cre_pct`` carries the cross-bank comparison instead, because
    # ``loans_total`` is populated for 99% of the panel.
    RatioDef(
        "office_nonaccrual_ratio",
        "nonaccrual_office_cre",
        "loans_office_cre",
        description="Nonaccrual office CRE / office CRE",
    ),
    RatioDef(
        "office_reserve_coverage",
        "acl_office_cre",
        "loans_office_cre",
        description="ACL on office CRE / office CRE",
    ),
    RatioDef(
        "leveraged_pct",
        "loans_leveraged",
        "loans_total",
        description="Leveraged lending as % of total loans (IR supplement)",
    ),
)

BY_RATIO: dict[str, RatioDef] = {r.name: r for r in RATIOS}
