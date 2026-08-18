"""Regression tests for the extraction and panel logic.

Every test here corresponds to a defect that produced plausible-looking but
wrong numbers during development -- the dangerous kind, since nothing raises
and the panel still renders.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from bankqtr_db import panel, taxonomy, variables, xbrl

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
    ],
)
def test_loan_category(member: str, expected: str) -> None:
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
