"""Regression tests for the extraction and panel logic.

Every test here corresponds to a defect that produced plausible-looking but
wrong numbers during development -- the dangerous kind, since nothing raises
and the panel still renders.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from bankqtr_db import (
    edgar,
    html_fallback,
    instance,
    panel,
    parallel,
    taxonomy,
    variables,
    xbrl,
)

# --------------------------------------------------------------------------
# Taxonomy
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("member", "expected"),
    [
        # The substring trap: this member means consumer loans *excluding*
        # cards, but contains "CreditCard".
        ("ConsumerExcludingCreditCardLoanPortfolioSegmentMember", "consumer_other"),
        ("CreditCardReceivablesMember", "credit_card"),
        ("CommercialRealEstateMember", "cre_total"),
        # Construction must beat the generic real-estate rule.
        ("CommercialRealEstateConstructionMember", "construction"),
        ("ConstructionLoansMember", "construction"),
        # Home equity must beat the residential rule.
        ("HomeEquityLoanMember", "home_equity"),
        ("ResidentialMortgageMember", "resi_mortgage"),
        ("CommercialAndIndustrialMember", "ci"),
        ("CommercialPortfolioSegmentMember", "commercial_total"),
        ("CapitalCallFinancingMember", "fund_finance"),
        # A CRE line that deliberately *excludes* construction must not be
        # read as construction, despite containing the word.
        (
            (
                "CommercialRealEstateExcludingResidentialBuilderAndDeveloper"
                "AndOtherConstructionMember"
            ),
            "cre_total",
        ),
        ("OtherCommercialConstructionMember", "construction"),
        ("CardmemberLoansMember", "credit_card"),
        ("MunicipalLoansMember", "municipal"),
        # The lease line, whose members mostly end "...FinancingReceivable
        # Member" rather than in the lease word itself.  The commercial one
        # is the trap: unmatched here it falls through to the ^Commercial
        # rollup and is summed into commercial_total beside it.
        ("FinanceLeasesFinancingReceivableMember", "lease"),
        ("FinanceLeaseFinancingReceivableMember", "lease"),
        ("LeaseFinancingsMember", "lease"),
        ("CommercialLeaseFinancingReceivableMember", "lease"),
        ("FinanceLeasesPortfolioSegmentMember", "lease"),
        ("DirectFinancingLeaseMember", "lease"),
    ],
)
def test_loan_category(member: str, expected: str) -> None:
    assert taxonomy.loan_category(member) == expected


@pytest.mark.parametrize(
    "member",
    [
        # Contains "Lease" but is an allowance component, a VIE, or too vague
        # to read either way.  Reported as unmapped rather than guessed at.
        "AllowanceForLoanAndLeaseLossesMember",
        "InvestmentVehiclesAndLeveragedLeaseTrustsMember",
        "LeaseAgreementsMember",
        "ExclusionFromImpairedLoansPursuantToAuthoritativeLeaseAccountingGuidanceMember",
    ],
)
def test_lease_rule_does_not_overreach(member: str) -> None:
    assert taxonomy.loan_category(member) != "lease"


@pytest.mark.parametrize(
    ("member", "expected"),
    [
        # "Loans *and* leases" names a rollup, not the lease class.  US
        # Bancorp's is its entire $139bn commercial book against a lease line
        # of about $4bn, so reading it as leases moves the mix by 37 points.
        ("CommercialLoanAndLeaseFinancingLoanMember", "commercial_total"),
        ("CommercialLoansAndLeasesMember", "commercial_total"),
        ("OtherConsumerLoansAndLeasesMember", "consumer_other"),
        # The grand total and an accrual-status cut: neither is a loan class,
        # and both must stay out of a partition sum.
        ("TotalLoansandLeasesMember", None),
        ("NonaccrualPortfolioLoansAndLeasesMember", None),
    ],
)
def test_loans_and_leases_is_a_rollup_not_the_lease_class(
    member: str, expected: str | None
) -> None:
    assert taxonomy.loan_category(member) == expected


@pytest.mark.parametrize(
    ("member", "expected"),
    [
        ("SpecialMentionMember", "special_mention"),
        ("SubstandardMember", "substandard"),
        ("CriticizedMember", "criticized"),
        # "InternalNoninvestmentGrade" contains "investmentGrade"; order matters.
        ("InternalNoninvestmentGradeMember", "non_investment_grade"),
        ("InternalInvestmentGradeMember", "investment_grade"),
        # FICO bands are their own family, never the commercial ladder.
        ("FICOScoreGreaterThan740Member", "fico_prime"),
        ("RefreshedFicoScoresLessThan660Member", "fico_subprime"),
    ],
)
def test_credit_quality(member: str, expected: str) -> None:
    assert taxonomy.credit_quality(member) == expected


def test_fico_never_maps_to_commercial_grade() -> None:
    """A consumer FICO band must not inflate investment-grade share."""
    for member in ("PrimeMember", "SubprimeMember", "FICOScore660andAboveMember"):
        family = taxonomy.CREDIT_QUALITY_FAMILY[taxonomy.credit_quality(member)]
        assert family == "fico"


def test_scope_members_are_not_loan_categories() -> None:
    """Held-for-sale must stay out of the held-for-investment mix."""
    assert taxonomy.loan_category("LoansHeldForSaleMember") is None
    assert taxonomy.scope_of("LoansHeldForSaleMember") == "held_for_sale"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

BASE = {
    "bank": "Test Bank",
    "ticker": "TB",
    "cik": "0000000001",
    "namespace": "us-gaap",
    "tag": variables.LOAN_TAGS[0],
    "unit": "USD",
    "form": "10-Q",
    "fy": 2024,
    "fp": "Q1",
    "accn": "acc-1",
    "filed": dt.date(2024, 4, 30),
    "source": "instance",
    "segment": None,
    "loan_class": None,
    "credit_quality": None,
    "past_due": None,
    "other_dims": "",
    "dim_axes": "",
    "n_dims": 0,
    "instant": True,
    "period_start": None,
}

SEG = "FinancingReceivablePortfolioSegmentAxis"
CLS = "FinancingReceivableRecordedInvestmentByClassOfFinancingReceivableAxis"
PD = "FinancingReceivablesPeriodPastDueAxis"

# Pre-2018 spellings.
LEGACY_CLS = "AccountsNotesLoansAndFinancingReceivableByReceivableTypeAxis"
LEGACY_SEG = "PortfolioSegmentAxis"
LEGACY_CQ = "CreditQualityIndicatorAxis"
LEGACY_PD = "FinancingReceivableByDelinquencyStatusAxis"


def fact(**kw) -> dict:
    row = dict(BASE)
    row.update(kw)
    return row


def frame(rows: list[dict]) -> pl.DataFrame:
    df = pl.DataFrame(rows, schema_overrides=xbrl.LONG_SCHEMA, infer_schema_length=None)
    return xbrl.attach_categories(df)


# --------------------------------------------------------------------------
# Dimensional selection
# --------------------------------------------------------------------------


def test_alias_members_are_not_double_counted() -> None:
    """The JPMorgan case: one balance tagged under two member names."""
    rows = [
        fact(
            period=dt.date(2024, 3, 31),
            value=100.0,
            segment="CreditCardReceivablesMember",
            dim_axes=SEG,
            n_dims=1,
        ),
        fact(
            period=dt.date(2024, 3, 31),
            value=100.0,
            segment="CreditCardLoanPortfolioSegmentMember",
            dim_axes=SEG,
            n_dims=1,
        ),
    ]
    out = panel.build_panel(frame(rows), derived=False)
    assert out["loans_credit_card"][0] == 100.0


def test_signatures_are_never_mixed() -> None:
    """Segment-level and class-level cuts of the same book must not be summed."""
    rows = [
        fact(
            period=dt.date(2024, 3, 31),
            value=600.0,
            segment="CommercialPortfolioSegmentMember",
            dim_axes=SEG,
            n_dims=1,
        ),
        fact(
            period=dt.date(2024, 3, 31),
            value=400.0,
            segment="CommercialPortfolioSegmentMember",
            loan_class="CommercialLoanMember",
            dim_axes=f"{SEG}|{CLS}",
            n_dims=2,
        ),
        fact(
            period=dt.date(2024, 3, 31),
            value=200.0,
            segment="CommercialPortfolioSegmentMember",
            loan_class="CommercialRealEstateMember",
            dim_axes=f"{SEG}|{CLS}",
            n_dims=2,
        ),
    ]
    out = panel.build_panel(frame(rows), derived=False)
    # 600 (segment level) or 400 (class level) -- never 1000.
    assert out["loans_commercial_total"][0] in (600.0, 400.0)


def test_plain_balance_not_read_off_delinquency_table() -> None:
    rows = [
        fact(
            period=dt.date(2024, 3, 31),
            value=500.0,
            loan_class="CommercialRealEstateMember",
            dim_axes=CLS,
            n_dims=1,
        ),
        fact(
            period=dt.date(2024, 3, 31),
            value=5.0,
            loan_class="CommercialRealEstateMember",
            past_due="FinancialAsset30To59DaysPastDueMember",
            dim_axes=f"{CLS}|{PD}",
            n_dims=2,
        ),
    ]
    out = panel.build_panel(frame(rows), derived=False)
    assert out["loans_cre_total"][0] == 500.0


def test_total_uses_most_complete_partition() -> None:
    """The Wells Fargo case: the shallow cut is an incomplete partition."""
    rows = [
        # Only one member at segment level -- covers part of the book.
        fact(
            period=dt.date(2024, 3, 31),
            value=375.0,
            segment="ConsumerPortfolioSegmentMember",
            dim_axes=SEG,
            n_dims=1,
        ),
        # The complete book appears one level deeper.
        fact(
            period=dt.date(2024, 3, 31),
            value=375.0,
            segment="ConsumerPortfolioSegmentMember",
            loan_class="ResidentialMortgageMember",
            dim_axes=f"{SEG}|{CLS}",
            n_dims=2,
        ),
        fact(
            period=dt.date(2024, 3, 31),
            value=534.0,
            segment="CommercialPortfolioSegmentMember",
            loan_class="CommercialAndIndustrialMember",
            dim_axes=f"{SEG}|{CLS}",
            n_dims=2,
        ),
    ]
    out = panel.build_panel(frame(rows), derived=False)
    assert out["loans_total"][0] == pytest.approx(909.0)


def test_undimensioned_total_wins_over_partition() -> None:
    rows = [
        fact(period=dt.date(2024, 3, 31), value=1000.0, dim_axes="", n_dims=0),
        fact(
            period=dt.date(2024, 3, 31),
            value=400.0,
            segment="CommercialPortfolioSegmentMember",
            dim_axes=SEG,
            n_dims=1,
        ),
    ]
    out = panel.build_panel(frame(rows), derived=False)
    assert out["loans_total"][0] == 1000.0


# --------------------------------------------------------------------------
# Flow quarterisation
# --------------------------------------------------------------------------


NCO_TAG = variables.BY_NAME["nco"].tags[0]


def ytd(end: dt.date, value: float, start: dt.date = dt.date(2024, 1, 1)) -> dict:
    return fact(
        tag=NCO_TAG,
        instant=False,
        period_start=start,
        period=end,
        value=value,
        dim_axes="",
        n_dims=0,
        form="10-Q",
    )


def test_ytd_only_flows_are_differenced_into_quarters() -> None:
    """The JPMorgan case: no quarterly charge-off fact exists in the filing.

    Q2 in particular is lost if the Q1 fact is excluded from the cumulative
    series merely because its span looks quarterly.
    """
    rows = [
        ytd(dt.date(2024, 3, 31), 100.0),
        ytd(dt.date(2024, 6, 30), 250.0),
        ytd(dt.date(2024, 9, 30), 420.0),
        ytd(dt.date(2024, 12, 31), 600.0),
    ]
    out = panel.build_panel(frame(rows), derived=False).sort("period")
    assert out["nco_total"].to_list() == [100.0, 150.0, 170.0, 180.0]


def test_reported_quarterly_flow_beats_derived() -> None:
    rows = [
        ytd(dt.date(2024, 3, 31), 100.0),
        ytd(dt.date(2024, 6, 30), 250.0),
        # A natively reported Q2 that disagrees with the difference.
        fact(
            tag=NCO_TAG,
            instant=False,
            period_start=dt.date(2024, 4, 1),
            period=dt.date(2024, 6, 30),
            value=149.0,
            dim_axes="",
            n_dims=0,
        ),
    ]
    out = panel.build_panel(frame(rows), derived=False).sort("period")
    assert out.filter(pl.col("period") == dt.date(2024, 6, 30))["nco_total"][0] == 149.0


# --------------------------------------------------------------------------
# Deduplication and restatement
# --------------------------------------------------------------------------


def test_latest_filing_wins_and_restatement_is_flagged() -> None:
    rows = [
        fact(
            period=dt.date(2024, 3, 31),
            value=100.0,
            accn="a",
            filed=dt.date(2024, 4, 30),
        ),
        fact(
            period=dt.date(2024, 3, 31),
            value=110.0,
            accn="b",
            filed=dt.date(2024, 7, 31),
        ),
    ]
    out = xbrl.deduplicate(frame(rows))
    assert out.height == 1
    assert out["value"][0] == 110.0
    assert bool(out["restated"][0]) is True


def test_rounding_difference_is_not_a_restatement() -> None:
    rows = [
        fact(
            period=dt.date(2024, 3, 31),
            value=100_000.0,
            accn="a",
            filed=dt.date(2024, 4, 30),
        ),
        fact(
            period=dt.date(2024, 3, 31),
            value=100_001.0,
            accn="b",
            filed=dt.date(2024, 7, 31),
        ),
    ]
    out = xbrl.deduplicate(frame(rows))
    assert bool(out["restated"][0]) is False


# --------------------------------------------------------------------------
# Derived ratios
# --------------------------------------------------------------------------


def test_nco_rate_is_annualised() -> None:
    rows = [
        fact(period=dt.date(2024, 3, 31), value=1000.0, dim_axes="", n_dims=0),
        ytd(dt.date(2024, 3, 31), 10.0),
    ]
    out = panel.build_panel(frame(rows))
    # 10 / 1000 = 1% per quarter -> 4% annualised.
    assert out["nco_rate"][0] == pytest.approx(4.0)


def test_criticized_rolls_up_from_components() -> None:
    cq_axis = "InternalCreditAssessmentAxis"
    rows = [
        fact(period=dt.date(2024, 3, 31), value=1000.0, dim_axes="", n_dims=0),
        fact(
            period=dt.date(2024, 3, 31),
            value=30.0,
            credit_quality="SpecialMentionMember",
            dim_axes=cq_axis,
            n_dims=1,
        ),
        fact(
            period=dt.date(2024, 3, 31),
            value=15.0,
            credit_quality="SubstandardMember",
            dim_axes=cq_axis,
            n_dims=1,
        ),
        fact(
            period=dt.date(2024, 3, 31),
            value=5.0,
            credit_quality="DoubtfulMember",
            dim_axes=cq_axis,
            n_dims=1,
        ),
    ]
    out = panel.build_panel(frame(rows))
    assert out["cq_classified"][0] == pytest.approx(20.0)
    assert out["cq_criticized"][0] == pytest.approx(50.0)
    assert out["criticized_pct"][0] == pytest.approx(5.0)


def test_missing_nonaccrual_does_not_become_zero_npa() -> None:
    """A bank that does not disclose nonaccrual must not read as pristine.

    The realistic shape: one peer in the panel discloses it, another does not.
    The second must come out null, not 0.0 -- a fabricated zero would rank a
    non-discloser as the cleanest book in the peer set.
    """
    nonaccrual_tag = variables.NONACCRUAL_TAGS[0]
    rows = [
        fact(period=dt.date(2024, 3, 31), value=1000.0, dim_axes="", n_dims=0),
        fact(
            period=dt.date(2024, 3, 31),
            value=10.0,
            tag=nonaccrual_tag,
            dim_axes="",
            n_dims=0,
        ),
        fact(
            period=dt.date(2024, 3, 31),
            value=2000.0,
            dim_axes="",
            n_dims=0,
            ticker="TB2",
            bank="Other Bank",
            cik="0000000002",
        ),
    ]
    out = panel.build_panel(frame(rows)).sort("ticker")
    discloser = out.filter(pl.col("ticker") == "TB")
    silent = out.filter(pl.col("ticker") == "TB2")
    assert discloser["npa_total"][0] == pytest.approx(10.0)
    assert silent["npa_total"][0] is None
    assert silent["npa_ratio"][0] is None


# --------------------------------------------------------------------------
# The pre-CECL dialect
#
# Everything below is about reading 2013-2019 filings without disturbing the
# 2020-2026 window.  Each test is one of the four ways that goes wrong
# silently -- a legacy name displacing a modern one, an unaliased axis, a
# delinquency shape the axis builder cannot see, and a scope qualifier summed
# as if it were a loan class.
# --------------------------------------------------------------------------


LEGACY_LOAN_TAG = "LoansAndLeasesReceivableNetOfDeferredIncome"


def test_legacy_tag_never_displaces_a_modern_one() -> None:
    """A quarter carrying both elements must be read off the modern one.

    The legacy names are appended to the tag tuples, and ranking is by
    position, so this is the property that keeps the 2020-2026 panel exactly
    as it was.  Prepending would silently rewrite it.
    """
    assert LEGACY_LOAN_TAG in variables.LOAN_TAGS
    assert variables.LOAN_TAGS.index(LEGACY_LOAN_TAG) > variables.LOAN_TAGS.index(
        "FinancingReceivableExcludingAccruedInterestBeforeAllowanceForCreditLoss"
    )

    rows = [
        fact(
            period=dt.date(2024, 3, 31),
            value=500.0,
            loan_class="CommercialRealEstateMember",
            dim_axes=CLS,
            n_dims=1,
        ),
        fact(
            period=dt.date(2024, 3, 31),
            value=900.0,
            tag=LEGACY_LOAN_TAG,
            loan_class="CommercialRealEstateMember",
            dim_axes=CLS,
            n_dims=1,
        ),
    ]
    out = panel.build_panel(frame(rows), derived=False)
    assert out["loans_cre_total"][0] == 500.0


def test_legacy_tag_is_used_where_the_modern_one_is_absent() -> None:
    rows = [
        fact(
            period=dt.date(2013, 12, 31),
            value=900.0,
            tag=LEGACY_LOAN_TAG,
            loan_class="CommercialRealEstateMember",
            dim_axes=CLS,
            n_dims=1,
        )
    ]
    out = panel.build_panel(frame(rows), derived=False)
    assert out["loans_cre_total"][0] == 900.0


def test_net_and_gross_writeoff_elements_are_not_interchanged() -> None:
    """The one assignment here that fails silently rather than loudly.

    A net element read as gross leaves ``nco_rate`` understated by the
    recovery rate, and nothing raises: the panel still renders.
    """
    nco = set(variables.BY_NAME["nco"].tags)
    gross = set(variables.BY_NAME["charge_offs"].tags)
    assert "AllowanceForLoanAndLeaseLossesWriteoffsNet" in nco
    assert "FinancingReceivableAllowanceForCreditLossesNetChargeOffs" in nco
    assert "AllowanceForLoanAndLeaseLossesWriteOffs" in gross
    assert "FinancingReceivableAllowanceForCreditLossesWriteOffs" in gross
    # The two spellings differ by one capital letter; they must not collide.
    assert not nco & gross


@pytest.mark.parametrize(
    ("axis", "column", "member"),
    [
        (LEGACY_CLS, "loan_class", "CommercialRealEstateMember"),
        (LEGACY_SEG, "segment", "CommercialPortfolioSegmentMember"),
        (LEGACY_CQ, "credit_quality", "SpecialMentionMember"),
        (LEGACY_PD, "past_due", "CurrentAndLessThan90DaysPastDueMember"),
    ],
)
def test_legacy_axes_are_promoted(axis: str, column: str, member: str) -> None:
    """Unaliased, a pre-2018 axis lands in ``other_dims`` and the column is null."""
    promoted = instance._explode_axes([{"dims": {axis: member}}])
    assert promoted[column] == [member]
    assert promoted["other_dims"] == [""]


def test_legacy_class_axis_carries_a_whole_loan_book() -> None:
    """Wells Fargo's FY2013 instance has no class-of-financing-receivable axis."""
    rows = [
        fact(
            period=dt.date(2013, 12, 31),
            value=300.0,
            tag=LEGACY_LOAN_TAG,
            loan_class="CommercialRealEstateMember",
            dim_axes=LEGACY_CLS,
            n_dims=1,
        ),
        fact(
            period=dt.date(2013, 12, 31),
            value=700.0,
            tag=LEGACY_LOAN_TAG,
            loan_class="CommercialAndIndustrialMember",
            dim_axes=LEGACY_CLS,
            n_dims=1,
        ),
    ]
    out = panel.build_panel(frame(rows), derived=False)
    assert out["loans_cre_total"][0] == 300.0
    assert out["loans_total"][0] == pytest.approx(1000.0)


def test_legacy_signature_never_outranks_a_modern_one() -> None:
    """A filing carrying both spellings must be read off the modern axis.

    The legacy axes are aliases of last resort, exactly as the legacy tags
    are.  Wells Fargo keeps tagging the old receivable-type axis into the
    2020s and does *not* partition the book on it -- residential mortgage
    appears there as the parent and its first-lien child at once -- so
    summing its members counts the first lien twice.
    """
    rows = [
        fact(
            period=dt.date(2021, 12, 31),
            value=260.0,
            loan_class="ResidentialMortgageMember",
            dim_axes=LEGACY_CLS,
            n_dims=1,
        ),
        fact(
            period=dt.date(2021, 12, 31),
            value=242.0,
            loan_class="FirstMortgageMember",
            dim_axes=LEGACY_CLS,
            n_dims=1,
        ),
        fact(
            period=dt.date(2021, 12, 31),
            value=260.0,
            loan_class="ResidentialMortgageMember",
            dim_axes=CLS,
            n_dims=1,
        ),
    ]
    out = panel.build_panel(frame(rows), derived=False)
    assert out["loans_resi_mortgage"][0] == 260.0


# ---- legacy delinquency is element-shaped ---------------------------------


def legacy_pd(tag: str, value: float, member: str = "CommercialLoanMember") -> dict:
    return fact(
        tag=tag,
        period=dt.date(2013, 12, 31),
        value=value,
        loan_class=member,
        dim_axes=CLS,
        n_dims=1,
    )


def test_each_legacy_delinquency_bucket_survives() -> None:
    """One element per bucket, so the one-tag-per-key rule would keep one bucket.

    :func:`panel._past_due_columns` fixes a single tag per bank-quarter, which
    is what keeps the axis members mutually exclusive.  Applied to this shape
    it would silently drop every bucket but one.
    """
    rows = [
        legacy_pd("FinancingReceivableRecordedInvestment30To59DaysPastDue", 30.0),
        legacy_pd("FinancingReceivableRecordedInvestment60To89DaysPastDue", 20.0),
        legacy_pd(
            "FinancingReceivableRecordedInvestmentEqualToGreaterThan90DaysPastDue", 10.0
        ),
        legacy_pd("FinancingReceivableRecordedInvestmentCurrent", 940.0),
        legacy_pd("FinancingReceivableRecordedInvestmentPastDue", 60.0),
    ]
    out = panel.build_panel(frame(rows), derived=False)
    assert out["pd_dpd_30_59"][0] == 30.0
    assert out["pd_dpd_60_89"][0] == 20.0
    assert out["pd_dpd_90_plus"][0] == 10.0
    assert out["pd_current"][0] == 940.0
    assert out["pd_past_due_total"][0] == 60.0


def test_alternative_spellings_of_one_bucket_are_not_added_together() -> None:
    """Within a bucket the candidates are alternatives, not components."""
    rows = [
        legacy_pd(
            "FinancingReceivableRecordedInvestmentEqualToGreaterThan90DaysPastDue", 10.0
        ),
        legacy_pd(
            "FinancingReceivableRecordedInvestment90DaysPastDueAndStillAccruing", 6.0
        ),
    ]
    out = panel.build_panel(frame(rows), derived=False)
    assert out["pd_dpd_90_plus"][0] == 10.0


def test_legacy_delinquency_does_not_overwrite_the_axis_reading() -> None:
    """The element path fills gaps behind the axis path, never over it."""
    rows = [
        fact(
            period=dt.date(2019, 12, 31),
            value=7.0,
            loan_class="CommercialLoanMember",
            past_due="FinancialAssetEqualToOrGreaterThan90DaysPastDueMember",
            dim_axes=f"{CLS}|{PD}",
            n_dims=2,
        ),
        legacy_pd(
            "FinancingReceivableRecordedInvestmentEqualToGreaterThan90DaysPastDue",
            99.0,
        ),
    ]
    rows[1]["period"] = dt.date(2019, 12, 31)
    out = panel.build_panel(frame(rows), derived=False)
    assert out["pd_dpd_90_plus"][0] == 7.0


def test_legacy_delinquency_elements_stay_out_of_the_loan_columns() -> None:
    """They are not loan tags, so they cannot leak into the mix."""
    assert not set(variables.LEGACY_PAST_DUE_BY_TAG) & set(variables.LOAN_TAGS)


# ---- era-specific scope members -------------------------------------------


@pytest.mark.parametrize(
    "member",
    [
        "PurchasedCreditImpairedMember",
        "PurchaseCreditImpairedLoansMember",
        "FinancingReceivableAcquiredWithDeterioratedCreditQualityMember",
        "CoveredFinancingReceivableMember",
        "FdicSupportedLoansMember",
        "CollectivelyEvaluatedForImpairmentMember",
        "IndividuallyEvaluatedForImpairmentMember",
        "AcquiredFinancingReceivableMember",
        "NonAcquiredFinancingReceivableMember",
        "UnallocatedFinancingReceivablesMember",
        "ImpairedFinancingReceivableWithRelatedAllowanceRecordedMember",
        "NonperformingAssetsMember",
    ],
)
def test_legacy_scope_members_are_not_loan_categories(member: str) -> None:
    assert taxonomy.scope_of(member) is not None
    assert taxonomy.loan_category(member) is None


@pytest.mark.parametrize(
    ("member", "ticker", "expected"),
    [
        # "Excluding X" must never be read as X -- the same trap the loan rules
        # already guard, now on the scope side.  BB&T spells every FY2013 class
        # "<class>ExcludingCovered"; JPMorgan spells residential "excluding
        # purchased credit impaired"; and Wells Fargo carries its *whole* book
        # on an axis whose only member says "excluding ... credit
        # deterioration".  A loose pattern erases all three.
        ("CreOtherExcludingCoveredMember", "TFC", "cre_total"),
        ("CreResidentialAdcExcludingCoveredMember", "TFC", "construction"),
        (
            "ResidentialRealEstateExcludingPurchasedCreditImpairedMember",
            "JPM",
            "resi_mortgage",
        ),
        ("StudentLoansGovernmentGuaranteedMember", "WFC", "student"),
        # Regions' two top-level rollups.
        ("TotalCommercialMember", "RF", "commercial_total"),
        ("TotalConsumerMember", "RF", "consumer_total"),
        # ...without swallowing the members that merely start the same way.
        ("TotalCommercialInvestorRealEstateConstructionMember", "RF", "construction"),
    ],
)
def test_scope_patterns_do_not_swallow_loan_classes(
    member: str, ticker: str, expected: str
) -> None:
    assert taxonomy.loan_category(member, ticker) == expected


def test_wells_fargo_main_book_is_not_read_as_a_scope_cut() -> None:
    """It is carried alone on its own axis, so a false positive drops the bank."""
    member = (
        "LoansExcludingCertainLoansAcquiredInTransferWith"
        "EvidenceOfCreditDeteriorationMember"
    )
    assert taxonomy.scope_of(member) is None
    assert (
        taxonomy.scope_of(
            "FinancingReceivableAcquiredWithDeterioratedCreditQualityMember"
        )
        == "purchased_credit_impaired"
    )


def test_scope_members_are_excluded_from_a_partition_total() -> None:
    """Left unmapped they are dropped; mapped as classes they double-count.

    The purchased-credit-impaired balance sits *beside* the classes it is
    already part of, so summing it into the partition counts it twice.
    """
    rows = [
        fact(
            period=dt.date(2013, 12, 31),
            value=600.0,
            tag=LEGACY_LOAN_TAG,
            loan_class="CommercialAndIndustrialMember",
            dim_axes=CLS,
            n_dims=1,
        ),
        fact(
            period=dt.date(2013, 12, 31),
            value=400.0,
            tag=LEGACY_LOAN_TAG,
            loan_class="ResidentialMortgageMember",
            dim_axes=CLS,
            n_dims=1,
        ),
        fact(
            period=dt.date(2013, 12, 31),
            value=250.0,
            tag=LEGACY_LOAN_TAG,
            loan_class="PurchasedCreditImpairedMember",
            dim_axes=CLS,
            n_dims=1,
        ),
    ]
    out = panel.build_panel(frame(rows), derived=False)
    assert out["loans_total"][0] == pytest.approx(1000.0)


# ---- the CECL break -------------------------------------------------------


def test_basis_marks_the_cecl_transition() -> None:
    """Two regimes spliced into one column read as a credit event otherwise."""
    rows = [
        fact(period=p, value=1000.0, dim_axes="", n_dims=0)
        for p in (
            dt.date(2019, 12, 31),
            dt.date(2020, 3, 31),
            dt.date(2020, 12, 31),
        )
    ]
    out = panel.build_panel(frame(rows), derived=False).sort("period")
    assert out["basis"].to_list() == ["incurred", "cecl", "cecl"]


def test_september_fiscal_year_adopts_cecl_a_quarter_late() -> None:
    """Raymond James' first CECL quarter is 2020Q4, not 2020Q1."""
    rows = [
        fact(
            ticker="RJF",
            bank="Raymond James",
            cik="0000720005",
            period=p,
            value=1000.0,
            dim_axes="",
            n_dims=0,
        )
        for p in (dt.date(2020, 6, 30), dt.date(2020, 12, 31))
    ]
    out = panel.build_panel(frame(rows), derived=False).sort("period")
    assert out["basis"].to_list() == ["incurred", "cecl"]


# ---- a modern-named element used for something much narrower -------------


def test_a_fragment_does_not_win_on_tag_priority() -> None:
    """The Citigroup FY2013 case: the top-ranked tag holds $113m of $19.6bn.

    Tag priority assumes every candidate measures the same concept.  Where a
    filer used a modern-*named* element for something far narrower, priority
    alone reports a 0.02% reserve ratio for a bank whose real one is 3%.
    Reordering ``ACL_TAGS`` would rewrite the modern window instead.
    """
    top, legacy = variables.ACL_TAGS[0], "LoansAndLeasesReceivableAllowance"
    assert variables.ACL_TAGS.index(legacy) > 0
    rows = [
        fact(period=dt.date(2013, 12, 31), value=113.0, tag=top, dim_axes="", n_dims=0),
        fact(
            period=dt.date(2013, 12, 31),
            value=19648.0,
            tag=legacy,
            dim_axes="",
            n_dims=0,
        ),
    ]
    out = panel.build_panel(frame(rows), derived=False)
    assert out["acl_total"][0] == 19648.0


def test_a_genuine_alternative_still_loses_to_tag_priority() -> None:
    """The guard is an order of magnitude wide, so close readings are untouched."""
    top, legacy = variables.ACL_TAGS[0], "LoansAndLeasesReceivableAllowance"
    rows = [
        fact(period=dt.date(2013, 12, 31), value=900.0, tag=top, dim_axes="", n_dims=0),
        fact(
            period=dt.date(2013, 12, 31),
            value=1000.0,
            tag=legacy,
            dim_axes="",
            n_dims=0,
        ),
    ]
    out = panel.build_panel(frame(rows), derived=False)
    assert out["acl_total"][0] == 900.0


def test_a_nil_flow_is_not_treated_as_a_fragment() -> None:
    """Raymond James reports a nil provision beside a $10m release.

    A flow is legitimately small, zero or negative in a quarter, so its
    magnitude says nothing about which tag is the right one.
    """
    provision = variables.BY_NAME["provision"]
    rows = [
        fact(
            tag=provision.tags[0],
            instant=False,
            period_start=dt.date(2024, 4, 1),
            period=dt.date(2024, 6, 30),
            value=0.0,
            dim_axes="",
            n_dims=0,
        ),
        fact(
            tag=provision.tags[2],
            instant=False,
            period_start=dt.date(2024, 4, 1),
            period=dt.date(2024, 6, 30),
            value=-10.0,
            dim_axes="",
            n_dims=0,
        ),
    ]
    out = panel.build_panel(frame(rows), derived=False)
    assert out["provision_total"][0] == 0.0


def test_unfunded_commitments_keeps_its_small_first_choice() -> None:
    """Its tuple ranks a reserve element above a notional one deliberately."""
    var = variables.BY_NAME["unfunded_commitments"]
    assert var.axis is None
    rows = [
        fact(
            period=dt.date(2024, 3, 31),
            value=500.0,
            tag=var.tags[0],
            dim_axes="",
            n_dims=0,
        ),
        fact(
            period=dt.date(2024, 3, 31),
            value=400_000.0,
            tag=var.tags[1],
            dim_axes="",
            n_dims=0,
        ),
    ]
    out = panel.build_panel(frame(rows), derived=False)
    assert out["unfunded_commitments"][0] == 500.0


# ---- a member that is a class on one axis and a rollup on another --------


def test_a_rollup_is_suppressed_only_where_its_components_are_disclosed() -> None:
    """A rollup is only a rollup when its parts are there.

    Wells Fargo tags residential mortgage as the parent and both liens at
    once, so reading the parent as a class counts the junior lien in both
    ``resi_mortgage`` and ``home_equity``.  In its recent quarters it stopped
    tagging the split, and there the same member is the only residential
    figure published -- so this cannot be decided from the member, or from the
    axis, but only from what the bank-quarter actually contains.
    """
    assert (
        taxonomy.superseded_scope(
            "ResidentialMortgageMember", "WFC", frozenset({"FirstMortgageMember"})
        )
        == "rollup"
    )
    assert (
        taxonomy.superseded_scope("ResidentialMortgageMember", "WFC", frozenset())
        is None
    )
    # ...and never for a bank that tags the member as an ordinary class.
    assert (
        taxonomy.superseded_scope(
            "ResidentialMortgageMember", "PNC", frozenset({"FirstMortgageMember"})
        )
        is None
    )


def test_a_rollup_survives_where_the_split_is_not_tagged() -> None:
    """The 15 recent Wells Fargo quarters: the parent is all there is."""
    rows = [
        fact(
            ticker="WFC",
            period=dt.date(2026, 6, 30),
            value=240.77,
            loan_class="ResidentialMortgageMember",
            dim_axes=CLS,
            n_dims=1,
        )
    ]
    out = panel.build_panel(frame(rows), derived=False)
    assert out["loans_resi_mortgage"][0] == pytest.approx(240.77)


def test_wells_fargo_first_lien_is_not_double_counted() -> None:
    """The parent must drop out, leaving the two liens to partition."""
    rows = [
        fact(
            ticker="WFC",
            period=dt.date(2021, 9, 30),
            value=258.89,
            loan_class="ResidentialMortgageMember",
            dim_axes=LEGACY_CLS,
            n_dims=1,
        ),
        fact(
            ticker="WFC",
            period=dt.date(2021, 9, 30),
            value=242.27,
            loan_class="FirstMortgageMember",
            dim_axes=LEGACY_CLS,
            n_dims=1,
        ),
        fact(
            ticker="WFC",
            period=dt.date(2021, 9, 30),
            value=16.62,
            loan_class="SecondMortgageMember",
            dim_axes=LEGACY_CLS,
            n_dims=1,
        ),
    ]
    out = panel.build_panel(frame(rows), derived=False)
    assert out["loans_resi_mortgage"][0] == pytest.approx(242.27)
    assert out["loans_home_equity"][0] == pytest.approx(16.62)


def test_wells_fargo_junior_lien_is_not_counted_in_both_columns() -> None:
    """The 10-K shape: the parent is on the modern axis, the liens on the old one.

    258.89 = 242.27 + 16.62 to the dollar, so reading the parent as
    ``resi_mortgage`` beside ``home_equity`` counts the junior lien twice.
    """
    rows = [
        fact(
            ticker="WFC",
            period=dt.date(2021, 12, 31),
            value=258.89,
            loan_class="ResidentialMortgageMember",
            segment="ConsumerPortfolioSegmentMember",
            dim_axes=f"{SEG}|{CLS}",
            n_dims=2,
        ),
        fact(
            ticker="WFC",
            period=dt.date(2021, 12, 31),
            value=242.27,
            loan_class="FirstMortgageMember",
            dim_axes=LEGACY_CLS,
            n_dims=1,
        ),
        fact(
            ticker="WFC",
            period=dt.date(2021, 12, 31),
            value=16.62,
            loan_class="SecondMortgageMember",
            dim_axes=LEGACY_CLS,
            n_dims=1,
        ),
    ]
    out = panel.build_panel(frame(rows), derived=False)
    assert out["loans_resi_mortgage"][0] == pytest.approx(242.27)
    assert out["loans_home_equity"][0] == pytest.approx(16.62)


def test_purchased_credit_impaired_is_caught_in_either_casing() -> None:
    """Truist spells it PCI; the guard on "Excluding" still has to hold."""
    for member in (
        "PCIMember",
        "ConsumerPortfolioSegmentPCIMember",
        "CommercialPortfolioSegmentPCIMember",
    ):
        assert taxonomy.scope_of(member) == "purchased_credit_impaired"
        assert taxonomy.loan_category(member) is None
    # A book defined by *excluding* PCI is the main book, not the PCI cut.
    assert taxonomy.scope_of("ConsumerPortfolioSegmentExcludingPCIMember") is None
    assert (
        taxonomy.loan_category("ConsumerPortfolioSegmentExcludingPCIMember")
        == "consumer_total"
    )


def test_a_rollup_is_kept_for_flows() -> None:
    """Wells Fargo reports residential charge-offs on the parent alone.

    The overlap the rollup test prevents is a balance-sheet one: the liens
    partition the parent's *balance*.  They carry no charge-offs of their own,
    so suppressing the parent for flows would drop the only figure there is.
    """
    co = variables.BY_NAME["charge_offs"]
    rows = [
        fact(
            ticker="WFC",
            period=dt.date(2021, 12, 31),
            value=242.27,
            loan_class="FirstMortgageMember",
            dim_axes=LEGACY_CLS,
            n_dims=1,
        ),
        fact(
            ticker="WFC",
            tag=co.tags[0],
            instant=False,
            period_start=dt.date(2021, 10, 1),
            period=dt.date(2021, 12, 31),
            value=0.04,
            loan_class="ResidentialMortgageMember",
            dim_axes=CLS,
            n_dims=1,
        ),
    ]
    out = panel.build_panel(frame(rows), derived=False)
    assert out["loans_resi_mortgage"][0] == pytest.approx(242.27)
    assert out["charge_offs_resi_mortgage"][0] == pytest.approx(0.04)


# ---- a disclosure table that states a rollup and its own parts -----------


def test_a_partition_drops_a_rollup_that_contains_its_parts() -> None:
    """The Wells Fargo 2022Q3 case: 1.33tn of loans against a 931bn book.

    The receivable-type table carries the consumer segment beside the five
    consumer classes underneath it, and the five come to the segment exactly.
    ``prefer="coverage"`` selects that signature *because* it is the largest,
    so the flat sum is the one thing the total must not be.
    """
    rows = [
        fact(
            ticker="WFC",
            period=dt.date(2022, 9, 30),
            value=v,
            loan_class=m,
            dim_axes=CLS,
            n_dims=1,
        )
        for m, v in (
            ("ConsumerPortfolioSegmentMember", 395.94),
            ("FirstMortgageMember", 254.16),
            ("AutomobileLoanMember", 54.55),
            ("CreditCardReceivablesMember", 43.56),
            ("ConsumerLoanMember", 29.77),
            ("SecondMortgageMember", 13.90),
            ("CommercialLoanMember", 379.69),
        )
    ]
    out = panel.build_panel(frame(rows), derived=False)
    # 395.94 + 379.69, with the five consumer classes folded into the segment
    # they add up to rather than counted a second time beside it.
    assert out["loans_total"][0] == pytest.approx(775.63)


def test_a_sibling_named_like_a_parent_is_still_summed() -> None:
    """Wells Fargo tags CRE mortgage as RealEstateLoanMember and construction
    as its sibling, but the category tree calls construction a child of
    cre_total.  Dropping every descendant of a present ancestor would lose the
    construction book; only the arithmetic may decide.
    """
    rows = [
        fact(period=dt.date(2022, 9, 30), value=v, loan_class=m, dim_axes=CLS, n_dims=1)
        for m, v in (
            ("RealEstateLoanMember", 133.77),
            ("ConstructionLoansMember", 21.89),
        )
    ]
    out = panel.build_panel(frame(rows), derived=False)
    assert out["loans_cre_total"][0] == pytest.approx(133.77)
    assert out["loans_construction"][0] == pytest.approx(21.89)
    assert out["loans_total"][0] == pytest.approx(155.66)


def test_siblings_that_nearly_sum_to_a_fourth_are_not_a_rollup() -> None:
    """Raymond James tags six sibling segments, not a hierarchy.

    Three of them come to within 0.97% of a fourth by coincidence.  The
    category tree calls CRE, REIT and tax-exempt descendants of commercial, so
    a loose tolerance reads the coincidence as containment and deletes a tenth
    of the bank's loan book.  Containment is an identity, not a near-miss.
    """
    rows = [
        fact(
            ticker="RJF",
            period=dt.date(2023, 9, 30),
            value=v,
            segment=m,
            dim_axes=SEG,
            n_dims=1,
        )
        for m, v in (
            ("SecuritiesBasedLoanMember", 14.582),
            ("CommercialPortfolioSegmentMember", 10.135),
            ("ResidentialPortfolioSegmentMember", 8.636),
            ("CommercialRealEstatePortfolioSegmentMember", 7.024),
            ("REITLoansPortfolioSegmentMember", 1.668),
            ("TaxExemptLoanPortfolioSegmentMember", 1.541),
        )
    ]
    out = panel.build_panel(frame(rows), derived=False)
    assert out["loans_total"][0] == pytest.approx(43.586)


# ---- CRE disclosed only as its classes ----------------------------------


def test_cre_total_is_built_from_its_classes_when_no_total_is_tagged() -> None:
    """JPMorgan tags no CRE rollup member, only the classes under it.

    The CRE book was in the panel all along, spread across three columns,
    while ``cre_total`` -- the one ``cre_pct`` and the coherence metric read --
    stayed empty.
    """
    rows = [
        fact(
            period=dt.date(2025, 12, 31), value=v, loan_class=m, dim_axes=CLS, n_dims=1
        )
        for m, v in (
            ("MultifamilyMember", 105.13),
            ("WholesaleRealEstateCommercialLessorsMember", 60.41),
            ("WholesaleRealEstateCommercialConstructionAndDevelopmentMember", 12.40),
        )
    ]
    rows = [dict(r, ticker="JPM") for r in rows]
    out = panel.build_panel(frame(rows))
    assert out["loans_cre_total"][0] == pytest.approx(177.94)


def test_a_single_class_is_not_promoted_to_the_cre_total() -> None:
    """One class is a part of the book, not a partition of it.

    Zions tags construction alone at $18m against a $57bn book; calling that
    its commercial real estate reports a 0.0% CRE share for one of the most
    CRE-concentrated banks in the panel.
    """
    rows = [
        fact(
            ticker="ZION",
            period=dt.date(2023, 6, 30),
            value=0.018,
            loan_class="ConstructionLoansMember",
            dim_axes=CLS,
            n_dims=1,
        )
    ]
    out = panel.build_panel(frame(rows))
    assert out["loans_construction"][0] == pytest.approx(0.018)
    # Absent entirely, or present and null: either way it is not claiming that
    # Zions' commercial real estate book is $18m.
    assert "loans_cre_total" not in out.columns or out["loans_cre_total"][0] is None


def test_a_tagged_cre_total_is_never_overwritten() -> None:
    rows = [
        fact(period=dt.date(2024, 3, 31), value=v, loan_class=m, dim_axes=CLS, n_dims=1)
        for m, v in (
            ("CommercialRealEstateMember", 500.0),
            ("MultifamilyMember", 100.0),
            ("ConstructionLoansMember", 50.0),
        )
    ]
    out = panel.build_panel(frame(rows))
    assert out["loans_cre_total"][0] == pytest.approx(500.0)


# ---- the HTML fallback's own label traps --------------------------------


@pytest.mark.parametrize(
    "label",
    [
        # Each of these was read as the concept it negates, from one JPMorgan
        # 10-K: the card book, criticized loans, and the provision.
        "total consumer, excluding credit card loans",
        "noncriticized",
        "pre-provision profit",
    ],
)
def test_a_label_that_negates_a_concept_is_not_read_as_it(label: str) -> None:
    assert html_fallback.EXCLUDE_LABEL.search(label)


def test_net_charge_offs_are_not_read_as_gross() -> None:
    """Row patterns are first-match-wins and every net label contains the gross
    one, so the net rule has to be listed first or it is unreachable -- the
    same net/gross trap as the element split in ``variables``."""
    spec = next(s for s in html_fallback.SPECS if s.name == "acl_rollforward")
    keys = list(spec.row_map)
    net = next(k for k in keys if spec.row_map[k] == "nco")
    gross = next(k for k in keys if spec.row_map[k] == "charge_offs")
    assert keys.index(net) < keys.index(gross)


@pytest.mark.parametrize(
    "label",
    [
        # A ratio is not a dollar amount, and Zions prints one in the same
        # table as the rollforward it belongs to.
        "ratio of net charge-offs to average loans and leases",
        "ratio of total net charge-offs to average total loans and leases",
    ],
)
def test_a_ratio_row_is_not_read_as_an_amount(label: str) -> None:
    assert html_fallback.EXCLUDE_LABEL.search(label)


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("net charge-offs", "nco"),
        ("net charge-offs (recoveries)", "nco"),
        # "net" is not always adjacent to the noun it qualifies.
        ("net loan and lease charge-offs", "nco"),
        ("gross charge-offs", "charge_offs"),
        ("charge-offs", "charge_offs"),
    ],
)
def test_net_and_gross_charge_offs_land_in_their_own_columns(
    label: str, expected: str
) -> None:
    import re as _re

    spec = next(s for s in html_fallback.SPECS if s.name == "acl_rollforward")
    assert not html_fallback.EXCLUDE_LABEL.search(label)
    matched = next(
        var
        for pattern, var in spec.row_map.items()
        if _re.search(pattern, label, _re.IGNORECASE)
    )
    assert matched == expected


# --------------------------------------------------------------------------
# Running the parse across cores
# --------------------------------------------------------------------------


def _one_row(n: int) -> pl.DataFrame:
    """Module level so it can be pickled to a worker."""
    return pl.DataFrame({"n": [n]})


def _explodes(n: int) -> pl.DataFrame:
    if n == 2:
        raise ValueError("bad document")
    return pl.DataFrame({"n": [n]})


PARALLEL_SCHEMA = {"n": pl.Int64}


def test_results_come_back_in_input_order() -> None:
    """Not merely equivalent to the serial frame -- identical to it.

    A panel that reshuffled between builds would make every diff unreadable,
    so ``Executor.map`` is used for its ordering guarantee rather than
    ``as_completed``.
    """
    out = parallel.map_frames(_one_row, range(25), schema=PARALLEL_SCHEMA)
    assert out["n"].to_list() == list(range(25))


def test_one_bad_document_does_not_end_the_run() -> None:
    """Same contract as the serial loops this replaced."""
    out = parallel.map_frames(_explodes, range(5), schema=PARALLEL_SCHEMA)
    assert out["n"].to_list() == [0, 1, 3, 4]


def test_an_unavailable_pool_falls_back_to_one_core(monkeypatch) -> None:
    """Windows spawns workers, so a caller with no main guard breaks the pool.

    It must degrade to the serial path rather than take the build down with a
    BrokenProcessPool that names none of the cause.
    """
    monkeypatch.setattr(parallel, "_POOL_UNAVAILABLE", True)
    out = parallel.map_frames(_one_row, range(6), schema=PARALLEL_SCHEMA)
    assert out["n"].to_list() == list(range(6))


def test_the_empty_batch_is_a_frame_not_a_crash() -> None:
    out = parallel.map_frames(_one_row, [], schema=PARALLEL_SCHEMA)
    assert out.is_empty()
    assert out.columns == ["n"]


# --------------------------------------------------------------------------
# The submissions overflow shards
# --------------------------------------------------------------------------


def test_a_submissions_shard_is_fetched_once_and_then_cached(
    tmp_path, monkeypatch
) -> None:
    """JPMorgan alone has 69 of them, walked on every list_filings call.

    Uncached that is dozens of requests per build for an index that changes
    only when a bank files, and it is what draws a 429 first: the shards go out
    in a tight loop while every main file is coming from disk.
    """
    calls: list[str] = []

    class _Resp:
        @staticmethod
        def json() -> dict:
            return {"filings": {"recent": {}}}

    def _fake_get(url: str, **_: object) -> _Resp:
        calls.append(url)
        return _Resp()

    monkeypatch.setattr(edgar, "RAW_FACTS", tmp_path)
    monkeypatch.setattr(edgar, "get", _fake_get)

    name = "CIK0000019617-submissions-001.json"
    first = edgar.submissions_shard(name)
    second = edgar.submissions_shard(name)

    assert first == second
    assert len(calls) == 1, "the second call must come from disk"
    assert (tmp_path / f"SUBSHARD_{name}.gz").exists()


# ---- the HTML fallback's split-parenthesis trap --------------------------
#
# A filing typesets a negative as separate cells, so the opening parenthesis
# arrives glued to the digits and the closer arrives alone.  ``_to_number``
# only honoured a parenthesis when it saw both halves, so the fragment parsed
# as *positive* -- a sign flip, not a parse failure, which is why nothing ever
# raised.  4,799 cells across the 210 cached filings are split this way.


@pytest.mark.parametrize(
    ("cells", "expected"),
    [
        # Ally: the fragment, read_html's duplicate of it for the spanned
        # column, then the closer alone.  Read as +1,423 before.
        (["Charge-offs (a)", "nan", "(1,423", "(1,423", ")", "nan"], -1423.0),
        # Goldman: the currency marker leaves a space where it was stripped,
        # which defeated the parenthesis check even with both halves present --
        # so this one came out None and the row was dropped entirely.
        (["Allowance for loan losses", "nan", "$ (3,573", ")"], -3573.0),
        # Northern Trust, with a decimal.
        (["Net charge-offs (recoveries)", "(0.7", "(0.7", ")", "1.1"], -0.7),
        # M&T, with the fragment not adjacent to the label.
        (["Unearned discount", "nan", "(509,993", ")", "nan"], -509993.0),
        # A positive number is untouched, duplicate and all.
        (["Commercial real estate", "28110", "28110", "nan"], 28110.0),
    ],
)
def test_a_parenthesised_negative_survives_being_split_across_cells(
    cells: list[str], expected: float
) -> None:
    merged = html_fallback._merge_split_parens(cells)
    assert html_fallback._first_numeric(merged) == expected


def test_a_fragment_with_no_closer_is_left_alone() -> None:
    """Reassembly is a repair, not a guess.  1.2% of fragments have no closer
    to be found, and inventing one for them would be the same class of error
    in the other direction."""
    cells = html_fallback._merge_split_parens(["Something", "(47", "1,234"])
    assert html_fallback._first_numeric(cells) == 47.0


def test_a_number_does_not_absorb_a_parenthesis_from_another_column() -> None:
    """Only padding, a currency marker or read_html's own duplicate may sit
    between a fragment and its closer.  A real number in between means the two
    belong to different columns."""
    cells = html_fallback._merge_split_parens(["Provision", "(52", "981", ")"])
    assert html_fallback._first_numeric(cells) == 52.0


# ---- ...and the sign convention it exposed -------------------------------


@pytest.mark.parametrize(
    ("variable", "parsed", "expected"),
    [
        # A rollforward writes the closing allowance and the loan lines as
        # deductions.  Carrying that into the panel makes a bank hold a
        # negative reserve, so balances are magnitudes.
        ("loans_lease", -5646.0, 5646.0),
        ("acl_ending", -14407.0, 14407.0),
        ("oreo", -2.0, 2.0),
        ("pd_dpd_90_plus", -1.0, 1.0),
        # Flows are left alone: a provision release and a net recovery are
        # both real and both genuinely negative.
        ("provision", -52.0, -52.0),
        ("nco", -927.0, -927.0),
        ("recoveries", -13.0, -13.0),
    ],
)
def test_presentation_negatives_are_magnitudes_only_where_a_balance_cannot_be_negative(
    variable: str, parsed: float, expected: float
) -> None:
    assert html_fallback._signed(variable, parsed) == expected


def test_both_readers_share_one_sign_convention() -> None:
    """``ir_extract`` had this rule and the HTML fallback did not, which went
    unnoticed only because the split-parenthesis defect was cancelling it out.
    One definition, on the lower layer, so the two cannot drift apart."""
    from bankqtr_db import ir_extract

    assert ir_extract._signed is html_fallback._signed


# ---- the caption is what names a table ----------------------------------
#
# Every schedule in a credit disclosure lists the loan classes down the side,
# so on body text alone a loan schedule, an unfunded-commitment table and a
# business segment's balance sheet score identically.  What separates them is
# the text printed above the table.


def _table(rows: list[list[str]]):
    import pandas as pd

    return pd.DataFrame(rows)


LOAN_ROWS = [
    ["Commercial and industrial", "452068"],
    ["Commercial real estate", "132284"],
    ["Credit card", "59540"],
    ["Total loans", "986167"],
]


def test_a_caption_counts_toward_classification() -> None:
    spec = html_fallback.classify_table(
        _table(LOAN_ROWS), "Table 16: Total Loans Outstanding by Portfolio Segment"
    )
    assert spec is not None and spec.name == "loan_portfolio"


@pytest.mark.parametrize(
    "caption",
    [
        # Wells Fargo's Table 3.4.  The rows are loan classes and the numbers
        # are the lines available behind them, so credit card reads 180,563
        # against a book of 59,540.
        "Table 3.4: Unfunded Credit Commitments",
        # US Bancorp's rate/volume analysis: the same row labels against the
        # change in interest income, which read total loans as 549.
        "Increase (decrease) in Interest Income",
        # Average balances carry the period-end labels one block below them.
        "Average Balance Sheets and Interest Yields/Rates",
    ],
)
def test_a_table_that_is_not_a_balance_sheet_is_refused(caption: str) -> None:
    assert html_fallback.classify_table(_table(LOAN_ROWS), caption) is None


@pytest.mark.parametrize(
    "caption",
    [
        # A segment repeats the consolidated labels over a fraction of the
        # book, and is printed first.  Wells Fargo names the segment and never
        # the word "segment".
        "Table 9d: Commercial Banking - Balance Sheet",
        "Table 9f: Corporate and Investment Banking - Balance Sheet",
        "Consumer Banking and Lending - Balance Sheet",
        "Results by reportable segment",
    ],
)
def test_a_business_segment_schedule_is_refused(caption: str) -> None:
    assert html_fallback.classify_table(_table(LOAN_ROWS), caption) is None


def test_a_portfolio_segment_is_not_a_business_segment() -> None:
    """The consolidated loan schedule is *named* after portfolio segments --
    "Total Loans Outstanding by Portfolio Segment and Class of Financing
    Receivable" -- so excluding on the bare word threw away the one table this
    module most wants, and left Wells Fargo's C&I reading 157."""
    spec = html_fallback.classify_table(
        _table(LOAN_ROWS),
        "Table 16: Total Loans Outstanding by Portfolio Segment and Class of "
        "Financing Receivable",
    )
    assert spec is not None and spec.name == "loan_portfolio"


def test_tables_are_paired_with_the_text_above_them() -> None:
    html = b"""
    <html><body>
      <p>Table 11: Loan Portfolios</p>
      <table><tr><td>Commercial and industrial</td><td>452068</td></tr>
             <tr><td>Total loans</td><td>986167</td></tr></table>
    </body></html>"""
    tables = html_fallback.read_tables(html)
    assert tables, "the document has a table"
    _, caption = tables[0]
    assert "Loan Portfolios" in caption


# ---- which document gets parsed ------------------------------------------


def test_the_ex13_is_recognised_as_the_statements(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wells Fargo, US Bancorp and BNY Mellon file a 10-K whose Item 8 points at
    the annual report, and file the annual report as an EX-13 beside it.  All
    21 of the cached filings that yield no table at all are these three banks,
    and no parser can fix that -- the document being parsed does not contain
    the numbers."""
    from bankqtr_db import ir

    monkeypatch.setattr(
        ir,
        "exhibit_index",
        lambda folder, accn: [
            ("wfc-1231x2025xex4c.htm", "", "EXHIBIT 4.C"),
            ("wfc-20251231.htm", "EX-13", "EXHIBIT 13"),
            ("wfc-1231x2025xex21.htm", "EX-21", "EXHIBIT 21"),
        ],
    )
    filing = _filing()
    assert html_fallback.financial_statement_exhibit(filing) == "wfc-20251231.htm"


def test_a_filing_that_carries_its_own_statements_has_no_ex13(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate is self-limiting on purpose: EX-13 *is* the annual report, so a
    filing that prints its statements inline has no such exhibit and the second
    pass changes nothing for it."""
    from bankqtr_db import ir

    monkeypatch.setattr(
        ir,
        "exhibit_index",
        lambda folder, accn: [("bac-20251231.htm", "", "10-K"), ("ex21.htm", "EX-21", "")],
    )
    assert html_fallback.financial_statement_exhibit(_filing()) is None


def test_the_ex13_is_cached_beside_the_primary_document() -> None:
    """A rebuild must not re-render or re-fetch; the two documents are kept
    under distinct names so the 189 filings that already parse keep their
    existing cache entries."""
    filing = _filing()
    primary = html_fallback._html_cache_path(filing)
    exhibit = html_fallback._exhibit_cache_path(filing)
    assert primary != exhibit
    assert exhibit.name.endswith(".ex13.html.gz")
    assert primary.parent == exhibit.parent


def _filing():
    import datetime as dt

    from bankqtr_db.filings import Filing

    return Filing(
        bank="Wells Fargo",
        ticker="WFC",
        cik="0000072971",
        accn="0000072971-26-000133",
        form="10-K",
        filing_date=dt.date(2026, 2, 20),
        report_date=dt.date(2025, 12, 31),
        primary_doc="wfc-20251231x10k.htm",
    )


# --------------------------------------------------------------------------
# Grids: schedules addressed by column as well as by row
# --------------------------------------------------------------------------
#
# Every case below is a reading that came out plausible and wrong while the
# two grid specs were being measured against the 231 cached documents.  None
# of them raises, and each one lands a number of the right magnitude in the
# wrong column.

# Bank of America's aging table, cut down to the columns that matter.  Note
# the two headers that are *not* the 30-89 bucket and read as though they
# were: the performing column names "30 days past due" inside a longer
# phrase, and the rollup beside it is the sum of the three buckets.
AGING_HEADER = [
    "",
    "30-59 days past due",
    "60-89 days past due",
    "90 days or more past due",
    "Total past due 30 days or more",
    "Total current or less than 30 days past due",
    "Total outstandings",
]
AGING_ROWS = [
    AGING_HEADER,
    ["Residential mortgage", "667", "972", "1,639", "3,278", "225,205", "228,483"],
    ["Commercial and industrial", "120", "40", "30", "190", "300,000", "300,190"],
]


def _aging_caption() -> str:
    return "Age analysis of past due loans and leases at December 31, 2025"


def test_a_grid_reads_the_column_not_the_left_most_number() -> None:
    """The bucket is a column; ``_first_numeric`` would take the performing one.

    This is the whole reason ``GridSpec`` exists.  A row-addressed spec
    matching "residential mortgage" takes the left-most number in the row,
    which in an aging table is whichever column the filer printed first.
    """
    spec = html_fallback.classify_grid(_table(AGING_ROWS), _aging_caption())
    assert spec is html_fallback.DELINQUENCY_BY_CATEGORY

    rows = html_fallback.extract_from_grid(
        _table(AGING_ROWS), spec, _filing(), _aging_caption()
    )
    got = {r["variable"]: r["value"] for r in rows}
    # 30-59 and 60-89 are separate columns here and are summed.
    assert got["dpd_30_89_resi_mortgage"] == 667 + 972
    assert got["dpd_90_plus_resi_mortgage"] == 1639


def test_the_performing_column_is_not_read_as_a_delinquency() -> None:
    """"Total current or less than 30 days past due" contains "30 days past due".

    A looser column pattern matched it, and Bank of America's entire $225bn
    residential book was reported as its 30-89 day delinquency against a real
    1,639.  The magnitude is plausible for a balance and absurd for a bucket,
    and nothing in the pipeline would have raised.
    """
    rows = html_fallback.extract_from_grid(
        _table(AGING_ROWS),
        html_fallback.DELINQUENCY_BY_CATEGORY,
        _filing(),
        _aging_caption(),
    )
    values = [r["value"] for r in rows]
    assert 225205 not in values
    assert 228483 not in values


def test_the_past_due_rollup_column_is_not_added_to_its_own_parts() -> None:
    """"Total past due 30 days or more" is 30-59 + 60-89 + 90+, already counted."""
    rows = html_fallback.extract_from_grid(
        _table(AGING_ROWS),
        html_fallback.DELINQUENCY_BY_CATEGORY,
        _filing(),
        _aging_caption(),
    )
    got = {r["variable"]: r["value"] for r in rows}
    assert got["dpd_30_89_resi_mortgage"] == 1639
    assert 3278 not in [r["value"] for r in rows]


def test_every_grid_column_may_be_created_by_the_merge() -> None:
    """A column off ``HTML_NEW_COLUMNS`` is parsed, scaled, and then dropped.

    ``fill_gaps`` creates an absent column only if the allowlist names it, and
    nothing raises when it does not -- the reading simply never reaches the
    panel, which looks identical to a filing that disclosed nothing. Both grid
    specs populate columns no filing tags, so every one of them has to be
    named there.
    """
    from bankqtr_db import reconcile

    missing = set(html_fallback.grid_columns()) - set(reconcile.HTML_NEW_COLUMNS)
    assert not missing


def test_a_grid_column_is_bounded_by_the_loan_book() -> None:
    """The check that caught leveraged lending at $1.18 trillion.

    A past-due bucket and an origination-year balance are both slices of the
    book, and both arrive under name prefixes none of the original entries in
    ``_SUBSET_OF_LOANS`` covers -- so without adding them they would skip the
    one guard written for columns with no XBRL counterpart to sanity-check
    against.
    """
    from bankqtr_db import reconcile

    for column in html_fallback.grid_columns():
        assert column.startswith(reconcile._SUBSET_OF_LOANS), column


def test_the_whole_book_row_is_not_read_as_the_lease_book() -> None:
    """"Total loans and leases" contains "leases", and matched it first.

    The same order-is-load-bearing trap ``ACL_ROLLFORWARD`` documents for
    net-versus-gross charge-offs.  Bank of America's whole-book 30-89 figure
    of 5,555 was reported as its lease delinquency, against a lease book a
    fraction of that size -- and the total row is what the panel's own
    ``pd_dpd_30_89`` column wants anyway.
    """
    rows = [
        ["", "30-59 days past due", "60-89 days past due", "90 days or more past due"],
        ["Lease financing", "10", "5", "2"],
        ["Total loans and leases (7)", "3,000", "2,555", "3,819"],
    ]
    got = {
        r["variable"]: r["value"]
        for r in html_fallback.extract_from_grid(
            _table(rows),
            html_fallback.DELINQUENCY_BY_CATEGORY,
            _filing(),
            _aging_caption(),
        )
    }
    assert got["pd_dpd_30_89"] == 5555
    assert got["pd_dpd_90_plus"] == 3819
    assert got["dpd_30_89_lease"] == 15


def test_a_spanned_header_cell_counts_once_not_once_per_column() -> None:
    """``read_html`` repeats a spanned cell across every column it covers.

    Grouping the matched indices by the header's own text is what keeps the
    single value under a three-column span from being added to itself.
    """
    rows = [
        ["", "30-89 days past due", "30-89 days past due", "90 days or more"],
        ["Commercial and industrial", "150", "150", "20"],
    ]
    got = {
        r["variable"]: r["value"]
        for r in html_fallback.extract_from_grid(
            _table(rows),
            html_fallback.DELINQUENCY_BY_CATEGORY,
            _filing(),
            _aging_caption(),
        )
    }
    assert got["dpd_30_89_ci"] == 150


def test_a_table_title_is_not_a_column_header() -> None:
    """A banner spanned over the whole table matched the 90-day pattern.

    Citizens' "Table 13: Nonaccrual loans and leases, accruing and 90 days or
    more past due and restructured loans and leases" is one cell spanned over
    thirty columns, so the bucket was mapped onto the entire width and every
    row returned whichever number came first in it.  The table's columns are
    years; it is not an aging schedule at all.
    """
    title = (
        "Table 13: Nonaccrual loans and leases, accruing and 90 days or more "
        "past due and restructured loans and leases"
    )
    rows = [
        [title] * 5,
        ["", "2025", "2024", "2023", "2022"],
        ["Commercial and industrial", "280", "240", "210", "190"],
    ]
    columns = html_fallback._grid_columns(
        _table(rows), html_fallback.DELINQUENCY_BY_CATEGORY
    )
    assert "dpd_90_plus" not in columns


def test_a_tdr_aging_table_is_refused() -> None:
    """The restructured book wears the same headings as the whole book.

    Fifth Third's "Table 56: Accruing and nonaccruing portfolio TDRs" is laid
    out identically -- loan classes down the side, buckets across the top --
    over a population two orders of magnitude smaller.  ``EXCLUDE_LABEL``
    cannot see it, because every row label in it is an ordinary loan class.
    """
    caption = "Table 56: Accruing and nonaccruing portfolio TDRs by loan type and delinquency status"
    assert html_fallback.classify_grid(_table(AGING_ROWS), caption) is None


# --- vintage -------------------------------------------------------------

VINTAGE_ROWS = [
    ["December 31, 2025", "2025", "2024", "2023", "2022", "2021", "Prior", "Total"],
    ["Commercial and industrial:", "", "", "", "", "", "", ""],
    ["Pass", "900", "800", "700", "600", "500", "400", "3,900"],
    ["Special mention", "40", "30", "20", "10", "5", "5", "110"],
    ["Substandard", "60", "20", "10", "10", "5", "5", "110"],
    ["Total", "1,000", "850", "730", "620", "510", "410", "4,120"],
    ["Commercial real estate:", "", "", "", "", "", "", ""],
    ["Pass", "100", "90", "80", "70", "60", "50", "450"],
    ["Special mention", "10", "5", "5", "5", "5", "5", "35"],
    ["Substandard", "10", "5", "5", "5", "5", "5", "35"],
    ["Total", "120", "100", "90", "80", "70", "60", "520"],
]
VINTAGE_CAPTION = (
    "The following table presents the amortized cost basis of loans by "
    "origination year and credit quality indicator"
)


def test_vintage_totals_are_built_from_the_grades_not_the_total_rows() -> None:
    """Summing the rows labelled "total" double counts; the grades do not.

    A vintage table states a nested set of totals -- "Total", "Total
    commercial", "Total commercial and industrial" all appear in one table --
    so adding the rows labelled "total" multiplies the book.  The grades
    appear once per loan class, so they partition it.
    """
    spec = html_fallback.classify_grid(_table(VINTAGE_ROWS), VINTAGE_CAPTION)
    assert spec is html_fallback.VINTAGE_ANALYSIS

    got = {
        r["variable"]: r["value"]
        for r in html_fallback.extract_from_grid(
            _table(VINTAGE_ROWS), spec, _filing(), VINTAGE_CAPTION
        )
    }
    # 2025: C&I 900 + 40 + 60, CRE 100 + 10 + 10.  The "Total" rows carry
    # 1,000 and 120 for the same year and must not be added on top.
    assert got["vintage_total_2025"] == 1120
    assert got["vintage_criticized_2025"] == 120
    assert got["vintage_total_2024"] == 850 + 100


def test_criticized_is_everything_that_is_not_pass() -> None:
    """The ladder is not uniform across filers, so it is defined by exclusion.

    Truist writes pass / special mention / substandard / nonperforming where
    others write doubtful and loss.  Total less pass is the one definition
    every filer's own table supports, and it keeps the total and the
    criticised share describing the same population.
    """
    rows = [
        ["December 31, 2025", "2025", "2024", "Total"],
        ["Pass", "900", "800", "1,700"],
        ["Nonperforming", "100", "50", "150"],
    ]
    got = {
        r["variable"]: r["value"]
        for r in html_fallback.extract_from_grid(
            _table(rows), html_fallback.VINTAGE_ANALYSIS, _filing(), VINTAGE_CAPTION
        )
    }
    assert got["vintage_criticized_2025"] == 100
    assert got["vintage_total_2025"] == 1000


def test_a_comparative_table_is_not_read_as_a_vintage_one() -> None:
    """Ally's "December 31, | 2019 | 2018 | 2017" is three reporting periods.

    115 of the 615 candidate tables measured are that shape.  A column reader
    keyed on year headers reports them as three origination years, so the
    caption has to name the disclosure before the table is read at all.
    """
    rows = [
        ["December 31,", "2019", "2018", "2017"],
        ["Pass", "900", "800", "700"],
        ["Substandard", "40", "30", "20"],
    ]
    caption = "Finance receivables and loans at December 31, 2019 and 2018"
    assert html_fallback.classify_grid(_table(rows), caption) is None


def test_the_prior_year_grid_in_the_same_filing_is_skipped() -> None:
    """A filing prints the grid twice, and "largest wins" picks the stale one.

    An older vintage amortises *down*, so Truist's 2017 origination year is
    780 in the 2020 table and 590 in the 2021 one.  The rule that settles
    every other contest in this module picks 780 here, because 780 is
    genuinely larger.  The header date is what separates them.
    """
    prior = [["December 31, 2024", "2024", "2023", "Total"]] + VINTAGE_ROWS[2:5]
    filing = _filing()  # reports 2025-12-31
    assert (
        html_fallback.extract_from_grid(
            _table(prior), html_fallback.VINTAGE_ANALYSIS, filing, VINTAGE_CAPTION
        )
        == []
    )
    # ...and the grid stated as of the filing's own period is kept.
    assert html_fallback.extract_from_grid(
        _table(VINTAGE_ROWS), html_fallback.VINTAGE_ANALYSIS, filing, VINTAGE_CAPTION
    )


def test_portfolio_segment_still_reaches_the_grid_reader() -> None:
    """A *portfolio* segment is a loan class, not a business unit.

    Excluding on the bare word once threw away the table this module most
    wants; the grid reader inherits ``_SEGMENT_TABLE`` and must not regress
    it.
    """
    caption = (
        "Total loans outstanding by portfolio segment and class of financing "
        "receivable, by origination year"
    )
    assert (
        html_fallback.classify_grid(_table(VINTAGE_ROWS), caption)
        is html_fallback.VINTAGE_ANALYSIS
    )


def test_a_business_segment_grid_is_still_refused() -> None:
    caption = "Commercial Banking - past due status by origination year"
    assert html_fallback.classify_grid(_table(VINTAGE_ROWS), caption) is None
