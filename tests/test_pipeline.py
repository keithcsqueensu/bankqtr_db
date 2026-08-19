"""Regression tests for the extraction and panel logic.

Every test here corresponds to a defect that produced plausible-looking but
wrong numbers during development -- the dangerous kind, since nothing raises
and the panel still renders.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from bankqtr_db import instance, panel, taxonomy, variables, xbrl

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


def test_a_rollup_is_suppressed_only_on_the_axis_that_carries_its_children() -> None:
    """Wells Fargo tags residential mortgage as parent and children together.

    On the pre-2018 receivable-type axis the parent sits beside its first-lien
    and junior-lien children, so reading it as a class counts the first lien
    twice.  On the modern class axis it is tagged alone and is the right
    reading.  A plain ``(ticker, member)`` override cannot tell those apart.
    """
    assert (
        taxonomy.axis_scope_of("ResidentialMortgageMember", "WFC", LEGACY_CLS)
        == "rollup"
    )
    # ...but not on the modern axis, nor on a fact carrying both, nor for a
    # bank that tags the member as an ordinary class.
    assert taxonomy.axis_scope_of("ResidentialMortgageMember", "WFC", CLS) is None
    assert (
        taxonomy.axis_scope_of(
            "ResidentialMortgageMember", "WFC", f"{LEGACY_CLS}|{CLS}"
        )
        is None
    )
    assert (
        taxonomy.axis_scope_of("ResidentialMortgageMember", "PNC", LEGACY_CLS) is None
    )


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
