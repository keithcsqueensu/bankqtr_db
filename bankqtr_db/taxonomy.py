"""Normalise heterogeneous XBRL dimension members onto canonical categories.

Banks do not share a disclosure vocabulary.  A survey of one 10-K per firm
across the universe found **106 distinct** loan-class members for a single
element -- ``CommercialRealEstateMember`` appears at only a third of them,
while others say ``CommercialRealEstateLoansMember``, ``RealEstateLoanMember``,
or push CRE into a bank-specific extension member such as
``WholesaleRealEstateCommercialLessorsMember``.  Any cross-bank comparison has
to pass through a mapping layer; the alternative is a panel whose columns mean
different things in different rows.

The rules below are ordered and first-match-wins, because the member strings
overlap (``ConstructionLoansMember`` must be tested before any ``RealEstate``
rule, ``HomeEquity`` before ``Residential``).  Anything unmatched is reported
by :func:`unmapped` rather than silently dropped -- an unmapped member is a
data-quality finding, not a no-op.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

# --------------------------------------------------------------------------
# Canonical categories
# --------------------------------------------------------------------------

# Loan categories.  ``parent`` lets the panel roll components up and check that
# a bank's parts reconcile to its reported total.
LOAN_CATEGORIES: dict[str, str | None] = {
    "total": None,
    "commercial_total": "total",
    "cre_total": "commercial_total",
    "cre_investor": "cre_total",
    "cre_owner_occupied": "cre_total",
    "construction": "cre_total",
    "multifamily": "cre_total",
    "ci": "commercial_total",
    "small_business": "commercial_total",
    "lease": "commercial_total",
    "municipal": "commercial_total",
    "fund_finance": "commercial_total",
    "financial_institutions": "commercial_total",
    "commercial_other": "commercial_total",
    "consumer_total": "total",
    "resi_mortgage": "consumer_total",
    "home_equity": "consumer_total",
    "credit_card": "consumer_total",
    "auto": "consumer_total",
    "student": "consumer_total",
    "securities_based": "consumer_total",
    "consumer_other": "consumer_total",
    "other": "total",
}

# Members that qualify the *measurement basis* rather than name a loan class.
# Held-for-sale balances must not be folded into a held-for-investment
# portfolio mix, and unfunded commitments are off-balance-sheet entirely.
SCOPE_MEMBERS: dict[str, str] = {
    "LoansHeldForSaleMember": "held_for_sale",
    "LoansHeldForInvestmentMember": "held_for_investment",
    "UnfundedLoanCommitmentMember": "unfunded_commitment",
    "LoansReceivableHeldForSaleMember": "held_for_sale",
    # Vintage tables split term vs revolving; that is a presentation axis,
    # not a loan class, and summing across it double-counts.
    "TermLoansMember": "term",
    "TermMember": "term",
    "RevolvingLoansMember": "revolving",
}


@dataclass(frozen=True)
class Rule:
    pattern: str
    category: str
    note: str = ""


# Order is load-bearing: first match wins.
LOAN_RULES: tuple[Rule, ...] = (
    # --- "excluding X" members must never map to X ------------------------
    # M&T tags CommercialRealEstateExcludingResidentialBuilderAndDeveloper
    # AndOtherConstruction -- a CRE line that deliberately *excludes*
    # construction, yet contains the word.  Same trap as the credit-card
    # member below; it inflated M&T construction to 33% of loans.
    Rule(r"Excluding.{0,80}Construction", "cre_total"),
    # --- construction / development (before any generic CRE rule) ---------
    Rule(
        r"Construction|Land(Acquisition)?And?Development|LandDevelopment",
        "construction",
    ),
    # --- multifamily (a CRE subclass some banks break out) ----------------
    Rule(r"Multifamily|MultiFamily|ApartmentBuilding", "multifamily"),
    Rule(r"REITLoans|RealEstateInvestmentTrust", "cre_investor"),
    # --- CRE split by occupancy (before generic CRE) ----------------------
    Rule(r"OwnerOccupied|Occupied.*Commercial|CommercialOwner", "cre_owner_occupied"),
    Rule(
        r"InvestorRealEstate|NonOwnerOccupied|IncomeProducing|InvestorCommercial"
        r"|CommercialLessors|CommercialMortgage(?!Backed)",
        "cre_investor",
    ),
    # --- generic CRE -------------------------------------------------------
    Rule(
        r"CommercialRealEstate|CommercialMortgage|RealEstateCommercial"
        r"|^RealEstateLoan|CommercialPropert",
        "cre_total",
    ),
    # --- consumer classes (before C&I, which is a broad catch) ------------
    # "ConsumerExcludingCreditCardLoanPortfolioSegmentMember" contains the
    # substring "CreditCard" but means the opposite; exclude it explicitly.
    Rule(r"Excluding.{0,20}CreditCard|CreditCard.{0,20}Excluded", "consumer_other"),
    Rule(r"CreditCard|(?i:cardmember)|CardReceivable", "credit_card"),
    Rule(r"Automobile|AutoLoan|AutoFinanc|VehicleLoan|RetailAutomotive", "auto"),
    Rule(r"HomeEquity|SecondMortgage|HELOC", "home_equity"),
    Rule(r"ReverseMortgage", "resi_mortgage"),
    Rule(
        r"ResidentialMortgage|ResidentialRealEstate|Residential(?!.*Commercial)"
        r"|^Mortgages?Member|OneToFourFamily|FirstMortgage|FirstLien",
        "resi_mortgage",
    ),
    Rule(r"Student|Education", "student"),
    Rule(
        r"SecuritiesBased|MarginLoan|SecuritiesLending|PledgedAssetLines"
        r"|SecuredLendingFacilities",
        "securities_based",
    ),
    # --- commercial classes ------------------------------------------------
    Rule(r"SmallBusiness", "small_business"),
    Rule(r"Municipal|TaxExempt|PublicFinance", "municipal"),
    # Capital-call / subscription / fund finance is a fast-growing commercial
    # book that several banks now break out; it is not ordinary C&I.
    Rule(
        r"CapitalCall|SubscriptionFinance|FundFinance|PrivateEquityCapital",
        "fund_finance",
    ),
    Rule(
        r"LoansToFinancialInstitutions|^FinancialInstitutionsMember"
        r"|CollateralizedLoanObligation|SecuritiesFinanceLoans",
        "financial_institutions",
    ),
    Rule(
        r"CommercialAndIndustrial|^CorporateLoan|CommercialAndFinancial"
        r"|BusinessBanking|BusinessandCorporateBanking|SyndicatedLoans",
        "ci",
    ),
    Rule(r"Lease(Financing|s)?(PortfolioSegment)?Member|DirectFinancingLease", "lease"),
    # --- rollups -----------------------------------------------------------
    Rule(
        r"^CommercialPortfolioSegment|^CommercialLoan(AndLease)?|^WholesaleMember"
        r"|^Commercial(?!.*(RealEstate|Mortgage))",
        "commercial_total",
    ),
    Rule(
        r"^ConsumerPortfolioSegment|^ConsumerAndResidential|^RetailMember"
        r"|^Consumer(?!.*(Other|Revolving|Installment|Retail))",
        "consumer_total",
    ),
    Rule(r"^ResidentialPortfolioSegment", "resi_mortgage"),
    # --- residual consumer buckets (after the rollup test) ----------------
    Rule(
        r"ConsumerOther|OtherConsumer|ConsumerRevolving|ConsumerInstallment"
        r"|ConsumerRetail|OtherRetail|PersonalLoan|PersonalUnsecured"
        r"|RVandMarine|RecreationalFinance|SolarEnergy|IndirectSecuredConsumer"
        r"|BankcardAndOtherRevolving|Overdraft|PrivateClient"
        r"|LoansToFinancialAdvisors|WealthManagementLoans|ConsumerLoanOther"
        r"|DirectandIndirectFinancingReceivable|DirectAndIndirect",
        "consumer_other",
    ),
    Rule(r"^RetailBankingMember", "consumer_total"),
    Rule(r"InvestorDependentEarlyStage|EarlyStage|VentureLending", "ci"),
    Rule(r"WealthManagementMortgages|RevolvingMortgage|FHAVA", "resi_mortgage"),
    Rule(r"ForeignCommercial", "commercial_other"),
    Rule(r"ForeignConsumer", "consumer_other"),
    Rule(r"OtherLoans|OtherFinancingReceivable|^Other", "other"),
)

# --------------------------------------------------------------------------
# Credit quality
# --------------------------------------------------------------------------

# Regulatory ladder: criticized = special mention + classified;
# classified = substandard + doubtful + loss.
CREDIT_QUALITY_CATEGORIES: dict[str, str | None] = {
    "pass": None,
    "special_mention": "criticized",
    "substandard": "classified",
    "doubtful": "classified",
    "loss": "classified",
    "classified": "criticized",
    "criticized": None,
    "nonaccrual": None,
    "performing": None,
    "nonperforming": None,
    "investment_grade": None,
    "non_investment_grade": None,
    "unrated": None,
    # Numeric internal ladders (1-9 style). Banks that use these do not tag
    # pass/special-mention/substandard, so criticized has to be derived from
    # the ladder position instead -- see INTERNAL_GRADE_ORDER.
    "internal_grade_low": None,
    "internal_grade_mid": None,
    "internal_grade_high": None,
    # Consumer FICO bands. Deliberately a *separate family* from the
    # commercial ladder: folding "prime" into investment_grade would silently
    # inflate the investment-grade share of a commercial book.
    "fico_prime": None,
    "fico_near_prime": None,
    "fico_subprime": None,
}

# Which credit-quality families may be mixed within one ratio.
CREDIT_QUALITY_FAMILY: dict[str, str] = {
    **{
        k: "regulatory"
        for k in (
            "pass",
            "special_mention",
            "substandard",
            "doubtful",
            "loss",
            "classified",
            "criticized",
            "nonaccrual",
            "performing",
            "nonperforming",
            "unrated",
        )
    },
    **{k: "internal_ig" for k in ("investment_grade", "non_investment_grade")},
    **{
        k: "internal_ladder"
        for k in ("internal_grade_low", "internal_grade_mid", "internal_grade_high")
    },
    **{k: "fico" for k in ("fico_prime", "fico_near_prime", "fico_subprime")},
}

CREDIT_QUALITY_RULES: tuple[Rule, ...] = (
    # Nonaccrual qualifiers ride on top of criticized labels at some banks
    # (CriticizedNonaccrualMember); test the compound forms first.
    Rule(r"CriticizedNonaccru", "criticized", "nonaccrual subset"),
    Rule(r"CriticizedAccru|CriticizedPerforming", "criticized", "accruing subset"),
    Rule(r"Criticized", "criticized"),
    Rule(r"SpecialMention|OtherAssetsEspeciallyMentioned", "special_mention"),
    Rule(r"Substandard", "substandard"),
    Rule(r"Doubtful", "doubtful"),
    Rule(r"^LossMember|ClassifiedLoss", "loss"),
    Rule(r"Classified", "classified"),
    Rule(r"^PassMember|^Pass(?!Due)|Acceptable|WatchMember", "pass"),
    Rule(r"NonAccrual|Nonaccrual", "nonaccrual"),
    Rule(r"NonperformingFinancing|NonPerforming", "nonperforming"),
    Rule(r"^PerformingFinancing|^Performing", "performing"),
    # Non-investment grade must be tested before investment grade: the string
    # "InternalNoninvestmentGrade" contains "investmentGrade".
    Rule(
        r"InternalNoninvestmentGrade|NonInvestmentGrade|NoninvestmentGrade",
        "non_investment_grade",
    ),
    Rule(
        r"InternalInvestmentGrade|InvestmentGradeMember|AOrBetter", "investment_grade"
    ),
    Rule(r"Unrated|NotRated|Ungraded|NotAssigned|UsingFicoCreditMetric", "unrated"),
    # --- consumer FICO bands (own family; never mixed with the ladder) ----
    # FICO member spellings vary in case ("FICOScore" vs "FicoScore") and in
    # wording, so these few patterns are matched case-insensitively.
    Rule(
        r"(?i:fico).{0,24}(?i:below|less\s?than)\s?\d{3}|(?i:score)less(?i:than)\d{3}",
        "fico_subprime",
    ),
    Rule(r"(?i:fico).{0,24}(?i:not\s?available|no\s?score)", "unrated"),
    Rule(
        r"(?i:fico).{0,24}(740|greater\s?than\s?\d{3}|(?i:and\s?above)|660)"
        r"|EqualToOrGreaterThan660|FicoScoresEqualToOrGreaterThan",
        "fico_prime",
    ),
    Rule(
        r"FicoScore620Through679|FICOScore620Through679"
        r"|FicoScore680Through739|FICOScore680Through739",
        "fico_near_prime",
    ),
    Rule(r"Subprime", "fico_subprime"),
    Rule(r"Prime", "fico_prime"),
    # --- numeric internal ladders ------------------------------------------
    Rule(
        r"OneToThreeInternalGrade|InternalRiskRating[123]Member|RiskLevelLow",
        "internal_grade_low",
    ),
    Rule(
        r"FourToFiveInternalGrade|InternalRiskRating[45]Member|RiskLevelMedium",
        "internal_grade_mid",
    ),
    Rule(
        r"SixToNineInternalGrade|InternalRiskRating[6789]Member|RiskLevelHigh",
        "internal_grade_high",
    ),
)

# For banks using a numeric ladder, the criticized cut-off. Ladder grades at
# or worse than this are the criticized equivalent.
INTERNAL_GRADE_ORDER = (
    "internal_grade_low",
    "internal_grade_mid",
    "internal_grade_high",
)
INTERNAL_GRADE_CRITICIZED = ("internal_grade_high",)

# --------------------------------------------------------------------------
# Past due
# --------------------------------------------------------------------------

# Matched case-insensitively: banks tag these as both
# "FinancialAsset30To59DaysPastDueMember" and "A3059dayspastdueMember".
PAST_DUE_RULES: tuple[Rule, ...] = (
    # Nonaccrual/nonperforming ride on the past-due axis at several banks.
    Rule(r"nonaccrual", "nonaccrual"),
    Rule(r"nonperforming", "nonperforming"),
    Rule(r"foreclosure", "in_foreclosure"),
    Rule(
        r"currentand(less|lessthan)?30|notpastdue|^currentmember|financialassetcurrent",
        "current",
    ),
    Rule(r"30to59|30through59|3059", "dpd_30_59"),
    Rule(r"60to89|60through89|6089", "dpd_60_89"),
    Rule(r"30to89|30through89|3089|30to149", "dpd_30_89"),
    Rule(r"90|150|greaterthan90|equaltogreaterthan90", "dpd_90_plus"),
    Rule(r"pastdue", "past_due_total"),
)


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------


def _compile(rules: tuple[Rule, ...]) -> tuple[tuple[re.Pattern[str], str], ...]:
    return tuple((re.compile(r.pattern), r.category) for r in rules)


def _compile_ci(rules: tuple[Rule, ...]) -> tuple[tuple[re.Pattern[str], str], ...]:
    return tuple((re.compile(r.pattern, re.IGNORECASE), r.category) for r in rules)


_LOAN = _compile(LOAN_RULES)
_CQ = _compile(CREDIT_QUALITY_RULES)
_PD = _compile_ci(PAST_DUE_RULES)

# Bank-specific overrides for members whose generic reading is wrong.
# Keyed by (ticker, member); consulted before the rule list.
OVERRIDES: dict[tuple[str, str], str] = {
    # Regions reports an explicit investor-CRE rollup as a portfolio segment.
    ("RF", "TotalInvestorRealEstateMember"): "cre_investor",
    # Wells Fargo's extension members carry the CRE split.
    ("WFC", "WholesaleRealEstateCommercialLessorsMember"): "cre_investor",
    (
        "WFC",
        "WholesaleRealEstateCommercialConstructionAndDevelopmentMember",
    ): "construction",
    # Capital One's "Other Loans" segment is commercial, not a residual.
    ("COF", "OtherLoansMember"): "commercial_other",
    # Morgan Stanley / Schwab securities-based lending sits in consumer.
    ("MS", "SecuritiesBasedLendingandOtherLoansMember"): "securities_based",
    ("MS", "SecuritiesBasedLendingMember"): "securities_based",
}

_unmapped: Counter[tuple[str, str]] = Counter()


def _apply(
    member: str | None,
    compiled: tuple[tuple[re.Pattern[str], str], ...],
    kind: str,
    ticker: str | None = None,
) -> str | None:
    if not member:
        return None
    if ticker is not None:
        hit = OVERRIDES.get((ticker, member))
        if hit is not None:
            return hit
    for pattern, category in compiled:
        if pattern.search(member):
            return category
    _unmapped[(kind, member)] += 1
    return None


def loan_category(member: str | None, ticker: str | None = None) -> str | None:
    """Map a loan-class or portfolio-segment member to a canonical category."""
    if member in SCOPE_MEMBERS:
        return None  # a basis qualifier, not a loan class
    return _apply(member, _LOAN, "loan", ticker)


def scope_of(member: str | None) -> str | None:
    """Measurement basis implied by a member (held-for-sale, unfunded, ...)."""
    return SCOPE_MEMBERS.get(member or "")


def credit_quality(member: str | None, ticker: str | None = None) -> str | None:
    """Map a credit-quality-indicator member to a canonical rating bucket."""
    return _apply(member, _CQ, "credit_quality", ticker)


def past_due(member: str | None, ticker: str | None = None) -> str | None:
    """Map a past-due-period member to a canonical delinquency bucket."""
    return _apply(member, _PD, "past_due", ticker)


def unmapped(min_count: int = 1) -> list[tuple[str, str, int]]:
    """Members seen but not mapped, most frequent first.

    Surfacing these is the point: an unmapped member means a bank's disclosure
    is silently missing from the panel.
    """
    return [
        (kind, member, n)
        for (kind, member), n in _unmapped.most_common()
        if n >= min_count
    ]


def reset_unmapped() -> None:
    _unmapped.clear()


def rollup_children(category: str) -> list[str]:
    """Direct children of a loan category in the rollup tree."""
    return [k for k, v in LOAN_CATEGORIES.items() if v == category]


# Categories that must not be summed together because banks disclose them at
# overlapping levels of detail (a bank reporting cre_total AND cre_investor
# would double-count if both were added into commercial_total).
OVERLAPPING = {
    "cre_total": ("cre_investor", "cre_owner_occupied", "multifamily", "construction"),
    "consumer_total": ("resi_mortgage", "home_equity", "credit_card", "auto"),
    "commercial_total": ("cre_total", "ci", "small_business", "lease"),
}
