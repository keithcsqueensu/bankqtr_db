"""Regression tests for IR supplement discovery, extraction and merging.

Same rule as ``test_pipeline.py``: every case here is a defect that produced a
plausible-looking but wrong number while the scraper was being built.  None of
them raised anything -- the panel rendered, the column was populated, and the
value was a thousand times too small, an average instead of a period end, or a
percentage instead of a balance.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import polars as pl
import pytest

from bankqtr_db import ir, ir_extract, reconcile

# --------------------------------------------------------------------------
# Quarter and document-name parsing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Bank of America's filenames separate words with '+' (an encoded
        # space) and '_'.  '_' is a regex word character, so "\b2Q26\b" never
        # fires on "_2Q26_" and every one of these documents was skipped.
        ("The+Supplemental+Information_2Q26_ADA.pdf", (2026, 2)),
        ("The+Presentation+Materials_4Q25_ADA.pdf", (2025, 4)),
        ("TFC_1Q26_Earnings_Deck.pdf", (2026, 1)),
        ("2Q26EarningsSupplement.pdf", (2026, 2)),
        ("Second Quarter 2026", (2026, 2)),
        ("Fourth Quarter 2020", (2020, 4)),
        ("q3 2024 results", (2024, 3)),
        # A period embedded as a bare date in the exhibit filename.
        ("hban20260630_8kex992.htm", (2026, 2)),
        # A month that is not a quarter end must not be read as one.
        ("hban20260515_8kex991.htm", None),
        ("annual report 2024", None),
    ],
)
def test_parse_quarter(text: str, expected: tuple[int, int] | None) -> None:
    assert ir.parse_quarter(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The+Supplemental+Information_2Q26_ADA.pdf", "supplement"),
        ("Quarterly Performance Summary", "supplement"),
        ("TFC_2Q26_Earnings_Deck.pdf", "presentation"),
        ("Earnings Call Presentation", "presentation"),
        # '+'-separated, which the pre-normalisation patterns missed.
        ("The+Press+Release_2Q26_ADA.pdf", "release"),
        ("ex991june2026creditmetrics.htm", "credit_metrics"),
        # Documents that look quarterly but are not.
        ("2026 Proxy Statement", None),
        ("2025 Annual Report", None),
        ("jpmc2026ccarresultsex99.htm", None),
        ("Q2 2026 earnings call transcript", None),
    ],
)
def test_classify_document(text: str, expected: str | None) -> None:
    assert ir.classify_document(text) == expected


def test_press_release_is_not_classified_as_supplement() -> None:
    """A press release names "financial highlights" too.

    Testing the supplement patterns first labelled every Truist press release a
    supplement, so the panel read its Quarterly Performance Summary slot from
    the wrong document.  Release detection has to come first.
    """
    filler = b" Net income available to common shareholders was reported for the period. " * 12
    release = (
        b"<html><body>FOR IMMEDIATE RELEASE Truist reports second quarter 2026 "
        b"results. Financial highlights follow." + filler
    )
    assert ir.classify_content(release) == "release"

    supplement = (
        b"<html><body>Quarterly Performance Summary Second Quarter 2026" + filler
    )
    assert ir.classify_content(supplement) == "supplement"


def test_image_only_slide_deck_is_not_classified() -> None:
    """JPMorgan files its investor-day deck as a page of slide images.

    It carries no extractable text, so it can never populate a column -- but it
    lands in the same earnings window as the January earnings 8-K, and once it
    claimed the quarter's supplement slot the real supplement was shut out.
    """
    deck = b"<html><body>" + b'<IMG src="firmoverview%03d.jpg" title="slide">' % 1 * 40
    assert ir.classify_content(deck) is None


def test_previous_quarter_end() -> None:
    # An earnings 8-K filed mid-July reports the June quarter.
    assert ir.previous_quarter_end(dt.date(2026, 7, 14)) == dt.date(2026, 6, 30)
    # Filed in January, it reports the December quarter of the prior year.
    assert ir.previous_quarter_end(dt.date(2026, 1, 21)) == dt.date(2025, 12, 31)


def test_every_registry_ticker_is_a_real_bank() -> None:
    """A typo'd ticker in the IR registry silently disables that bank."""
    from bankqtr_db.config import BY_TICKER

    unknown = [s.ticker for s in ir.SOURCES if s.ticker not in BY_TICKER]
    assert unknown == []


def test_every_bank_has_a_route() -> None:
    """A bank with no registry entry must still fall back, not vanish."""
    from bankqtr_db.config import universe

    for bank in universe():
        assert ir.source_for(bank).providers, bank.ticker


# --------------------------------------------------------------------------
# Reading a value off a line
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "label", "value"),
    [
        # The ordinary case: label, this quarter, then comparatives.
        (
            "Commercial and industrial $ 49,586 $ 48,879 $ 49,671 $ 707 1.4 %",
            "Commercial and industrial",
            49586.0,
        ),
        # Regions marks this row with a footnote.  "(1)" reads as a
        # parenthesised negative, so a $3.2bn home-equity book was recorded
        # as -$1.
        (
            "Home equity—lines of credit (1) 3,184 3,130 3,150 54 1.7 %",
            "Home equity—lines of credit",
            3184.0,
        ),
        # The percentage twin of the loan schedule: same row labels, shares
        # instead of balances.  5.1 must not land in a dollar column.
        ("Commercial real estate mortgage—owner-occupied 5.1 % 5.1 % 5.0 %", None, None),
        ("Total Loans 100.0 % 100.0 %", None, None),
        # A contents line: the only number is the page it points at.
        ("Allowance for Credit Losses and Related Ratios 2", None, None),
        # A genuine single-value row is still read.
        ("Total loans $ 96,723", "Total loans", 96723.0),
        # Parenthesised negatives survive.
        ("Allowance for credit losses (1,743) (1,730)", None, -1743.0),
        # A heading has no value at all.
        ("End of Period Loans", "End of Period Loans", None),
    ],
)
def test_split_merged(line: str, label: str | None, value: float | None) -> None:
    got_label, got_value = ir_extract._split_merged(line)
    assert got_value == value
    if label is not None:
        assert got_label == label


def test_contents_row_is_refused_even_when_read_html_repeats_the_cell() -> None:
    """Wells Fargo's contents page put a $23m C&I book on an $849bn bank.

    ``read_html`` emits a column-spanning cell once per column it spans, so the
    page number arrives three times over and a "only one number on the row"
    rule never fires.  The test has to be on distinct values.
    """
    title = (
        "Commercial Loan Portfolio - Commercial and Industrial Loans and "
        "Lease Financing by Industry and Commercial Real Estate Loans by "
        "Property Type"
    )
    _, value = ir_extract.row_label_and_value(pd.Series([title, "23", "23", "23"]))
    assert value is None


def test_a_section_title_is_not_a_row_label() -> None:
    title = (
        "Allocation of the Allowance for Credit Losses by Loan Class and "
        "Portfolio Segment for the Periods Presented"
    )
    label, value = ir_extract.row_label_and_value(pd.Series([title, "4,834"]))
    assert len(label) > 90
    assert value is None


def test_average_block_is_not_read_as_period_end() -> None:
    """Regions publishes both, with identical row labels.

    "End of Period Loans" gives owner-occupied CRE of 4,890; the "Average
    Balances of Loans" page two pages later gives 4,900 for the quarter and
    4,882 for the six months.  Reading the average page silently substitutes an
    average balance into a period-end panel.
    """
    period_end = """\
End of Period Loans
($ amounts in millions) 6/30/2025 3/31/2025 12/31/2024
Commercial and industrial $ 49,586 $ 48,879 $ 49,671
Commercial real estate mortgage—owner-occupied 4,890 4,849 4,841
Total Loans $ 96,723 $ 95,733 $ 96,727"""
    averages = """\
Average Balances of Loans
Average Balances
($ amounts in millions) 2Q25 1Q25 4Q24
Commercial and industrial $ 49,033 $ 49,209 $ 49,357
Commercial real estate mortgage—owner-occupied 4,900 4,863 4,869
Total Loans $ 96,077 $ 96,122 $ 96,408"""
    compiled = ir_extract._compiled_row_maps()

    rows = ir_extract._page_line_rows(period_end, compiled)
    by_var = {r["variable"]: r["value"] for r in rows}
    assert by_var["loans_cre_owner_occupied"] == 4890.0
    assert by_var["loans_total"] == 96723.0

    assert ir_extract._page_line_rows(averages, compiled) == []


def test_segment_schedules_are_skipped() -> None:
    """A segment schedule looks exactly like the consolidated one.

    Wells Fargo's Commercial Banking segment reports C&I of 181,739 against a
    consolidated 487,630, and its page is printed first.
    """
    table = pd.DataFrame(
        [
            ["Commercial and industrial", "$", "181,739"],
            ["Commercial real estate", "", "40,179"],
            ["Total loans", "$", "237,127"],
        ]
    )
    caption = "Wells Fargo & Company and Subsidiaries COMMERCIAL BANKING SEGMENT"
    assert ir_extract.classify_table(table, caption) is None


def test_caption_decides_between_schedules_that_share_row_labels() -> None:
    """An allowance table lists the loan classes exactly as the schedule does."""
    table = pd.DataFrame(
        [
            ["Commercial and industrial", "$", "4,834"],
            ["Commercial real estate", "", "2,349"],
            ["Credit card", "", "4,974"],
            ["Total", "$", "14,407"],
        ]
    )
    schedule = ir_extract.classify_table(
        table, "CONSOLIDATED LOANS OUTSTANDING - PERIOD-END BALANCES"
    )
    allowance = ir_extract.classify_table(
        table, "ALLOCATION OF ALLOWANCE FOR CREDIT LOSSES FOR LOANS"
    )
    assert schedule is not None and schedule.name == "ir_loan_schedule"
    assert allowance is not None and allowance.name == "ir_acl_rollforward"


def test_period_end_caption_survives_the_word_average() -> None:
    """One caption can name both bases; the table still holds period-end rows.

    Wells Fargo's loan schedule is headed "PERIOD-END BALANCES, AVERAGE
    BALANCES, AND AVERAGE INTEREST RATES", and skipping it on the word
    "average" threw away the loan schedule itself.
    """
    table = pd.DataFrame(
        [
            ["Commercial and industrial", "$", "487,630"],
            ["Commercial real estate", "", "132,986"],
            ["Total loans", "$", "1,031,115"],
        ]
    )
    caption = (
        "CONSOLIDATED LOANS OUTSTANDING - PERIOD-END BALANCES, "
        "AVERAGE BALANCES, AND AVERAGE INTEREST RATES (in millions)"
    )
    spec = ir_extract.classify_table(table, caption)
    assert spec is not None
    rows = ir_extract.extract_table(table, spec, _doc(), caption)
    by_var = {r["variable"]: r["value"] for r in rows}
    assert by_var["loans_ci"] == 487630.0


def test_largest_reading_wins_across_sub_schedules() -> None:
    """Sub-schedules are subsets, so the consolidated figure is the largest.

    US Bancorp's total loans read 410,300 on the consolidated schedule and 538
    on a sub-schedule printed above it; taking whichever came first took 538.
    """
    frame = pl.DataFrame(
        {
            "ticker": ["USB", "USB"],
            "period": [dt.date(2026, 6, 30)] * 2,
            "variable": ["loans_total", "loans_total"],
            "value": [538.0, 410_300.0],
            "confidence": [0.55, 0.55],
            "_rank": [0, 0],
        }
    )
    out = ir_extract._consolidated_first(frame)
    assert out.height == 1
    assert out["value"][0] == 410_300.0


def test_balances_are_taken_as_magnitudes_flows_are_not() -> None:
    """A rollforward writes the closing allowance as a deduction."""
    assert ir_extract._signed("acl_total", -14_407.0) == 14_407.0
    assert ir_extract._signed("loans_cre_total", -16.0) == 16.0
    # A provision release and a net recovery are real, and really negative.
    assert ir_extract._signed("provision_total", -54.0) == -54.0
    assert ir_extract._signed("nco_total", -12.0) == -12.0


def test_declared_scale_is_read_per_row_not_only_per_table() -> None:
    """PNC states "In millions" for the table and overrides it on some rows.

    "Average loans In billions" sits inside a millions-denominated schedule.
    Applying the table's unit to that row is wrong by a factor of a thousand.
    """
    assert ir_extract.declared_scale("In millions, except per share data") == 1e6
    assert ir_extract.declared_scale("Interest-bearing deposits In billions") == 1e9
    assert ir_extract.declared_scale("($ amounts in millions)") == 1e6
    assert ir_extract.declared_scale("(in thousands)") == 1e3
    assert ir_extract.declared_scale("Commercial and industrial") is None
    # The unit qualifier must come off the label, or anchored patterns miss.
    assert ir_extract._strip_scale("Total loans In billions") == "Total loans"


# --------------------------------------------------------------------------
# Slide-style disclosures
# --------------------------------------------------------------------------

# Condensed from Regions' 2Q26 deck: the office slide is immediately followed
# by a trucking slide using the same caption layout, and the office section
# pattern matches a heading on that later slide too.
_OFFICE_DECK = (
    "23 CRE- Office Portfolio (Outstanding balances as of June 30, 2026) "
    "$ in Millions. Amounts include IRE and CRE Unsecured loans. "
    "Key Portfolio Metrics Balances $858 % of Total Loans 0.9% "
    "NPL $121 NPL / Loans 14.1% Charge-offs $1 Charge-offs / Loans 0.1% "
    "ACL $35 ACL / Loans 4.1% Ongoing Portfolio Surveillance "
    "Investor Real Estate Office Portfolio Overview 79% 21% Suburban Urban "
    "24 Transportation - Trucking (Outstanding balances as of June 30, 2026) "
    "Balances $1,076 NPL $45 ACL $83"
)


def test_office_phrases_stop_at_the_first_section() -> None:
    rows = ir_extract.extract_phrases(_OFFICE_DECK, _doc())
    by_var = {r["variable"]: r["value"] for r in rows}
    assert by_var["loans_office_cre"] == 858.0
    assert by_var["nonaccrual_office_cre"] == 121.0
    assert by_var["acl_office_cre"] == 35.0
    # The trucking slide's $1,076 must not overwrite office exposure.
    assert by_var["loans_office_cre"] != 1076.0


def test_office_phrases_take_the_slide_declared_unit() -> None:
    """Office CRE has no XBRL counterpart to infer a scale against.

    The slide's own "$ in Millions" caption is the only evidence available, so
    without reading it the row is dropped for want of a scale.
    """
    rows = ir_extract.extract_phrases(_OFFICE_DECK, _doc())
    assert {r["declared_scale"] for r in rows} == {1e6}


def test_office_percentage_is_not_read_as_a_balance() -> None:
    """"% of Total Loans 0.9%" must not populate an exposure column."""
    rows = ir_extract.extract_phrases(_OFFICE_DECK, _doc())
    assert all(r["value"] not in (0.9, 14.1, 4.1) for r in rows)


def _doc() -> ir.IRDoc:
    return ir.IRDoc(
        bank="Regions Financial",
        ticker="RF",
        cik="0001281761",
        year=2026,
        quarter=2,
        doc_type="presentation",
        url="https://example.invalid/rf.htm",
        origin="q4",
    )


# --------------------------------------------------------------------------
# Merging into the panel
# --------------------------------------------------------------------------


def _panel() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "bank": ["Regions Financial"],
            "ticker": ["RF"],
            "cik": ["0001281761"],
            "quarter": ["2026Q2"],
            "period": [dt.date(2026, 6, 30)],
            "loans_total": [100_000_000_000.0],
            "loans_cre_total": [None],
        },
        schema_overrides={"loans_cre_total": pl.Float64},
    )


def _ir_row(variable: str, value: float, scale: float | None) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "bank": ["Regions Financial"],
            "ticker": ["RF"],
            "cik": ["0001281761"],
            "accn": ["RF_2026Q2_presentation.htm"],
            "form": ["ir_presentation"],
            "period": [dt.date(2026, 6, 30)],
            "filed": [dt.date(2026, 6, 30)],
            "table": ["office_cre"],
            "variable": [variable],
            "label": ["balances"],
            "value": [value],
            "source": ["ir_supplement"],
            "scale_resolved": [False],
            "confidence": [0.45],
            "declared_scale": [scale],
            "doc_type": ["presentation"],
            "origin": ["q4"],
        },
        schema_overrides={"declared_scale": pl.Float64},
    )


def test_merge_ir_adds_columns_that_no_filing_carries() -> None:
    """Office CRE has no XBRL column to fill, so the merge must create one."""
    merged = reconcile.merge_ir(
        reconcile.reconcile(_panel()), _ir_row("loans_office_cre", 858.0, 1e6)
    )
    assert merged["loans_office_cre"][0] == 858e6
    assert merged["loans_office_cre__source"][0] == "ir_supplement"


def test_merge_ir_never_overwrites_an_xbrl_value() -> None:
    merged = reconcile.merge_ir(
        reconcile.reconcile(_panel()), _ir_row("loans_total", 96_723.0, 1e6)
    )
    assert merged["loans_total"][0] == 100_000_000_000.0
    assert merged["loans_total__source"][0] == "xbrl"


def test_unscaled_ir_rows_are_dropped_not_guessed() -> None:
    """A row with no declared unit and nothing to infer against stays out.

    Defaulting to millions would be right for most banks and wrong by a factor
    of a thousand for the ones reporting in thousands.
    """
    merged = reconcile.merge_ir(
        reconcile.reconcile(_panel()), _ir_row("loans_office_cre", 858.0, None)
    )
    assert "loans_office_cre" not in merged.columns or merged["loans_office_cre"][0] is None


def test_a_slice_of_the_book_cannot_exceed_the_book() -> None:
    """A loose phrase pattern read US Bancorp's leveraged book at $1.18tn.

    It scaled cleanly, it was the only reading of that slide, and the panel
    rendered it without complaint at 310% of total loans.
    """
    merged = reconcile.merge_ir(
        reconcile.reconcile(_panel()),
        _ir_row("loans_leveraged", 1_181_000.0, 1e6),
    )
    assert (
        "loans_leveraged" not in merged.columns
        or merged["loans_leveraged"][0] is None
    )


def test_a_plausible_slice_is_still_accepted() -> None:
    merged = reconcile.merge_ir(
        reconcile.reconcile(_panel()), _ir_row("loans_cre_total", 20_000.0, 1e6)
    )
    assert merged["loans_cre_total"][0] == 20_000e6


def test_refresh_ratios_fills_only_what_is_null() -> None:
    """Ratios are computed before the IR merge, so new inputs have none yet."""
    merged = reconcile.merge_ir(
        reconcile.reconcile(_panel()), _ir_row("loans_office_cre", 1_000.0, 1e6)
    )
    out = reconcile.refresh_ratios(merged)
    assert out["office_cre_pct"][0] == pytest.approx(1.0)
    assert out["office_cre_pct__source"][0] == "derived_ir"


def test_refresh_ratios_does_not_disturb_an_existing_ratio() -> None:
    panel = reconcile.reconcile(_panel()).with_columns(
        pl.lit(42.0).alias("loans_to_assets"),
        pl.lit("xbrl").alias("loans_to_assets__source"),
    )
    out = reconcile.refresh_ratios(panel)
    assert out["loans_to_assets"][0] == 42.0
    assert out["loans_to_assets__source"][0] == "xbrl"
