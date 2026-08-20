"""Regression tests for the FFIEC Call Report path.

Every test here corresponds to a defect that produced plausible-looking but
wrong numbers while this package was built -- the dangerous kind, since nothing
raises and the panel still renders.  Where a test names a bank and a quarter,
those are the real filings the defect was found in.

The unit tests build a bulk file in memory rather than reading a cached
quarter, so they run without ``data/raw/call`` and state the exact shape they
are about.  The integration tests at the end are skipped when no quarter is
cached.
"""

from __future__ import annotations

import datetime as dt
import io
import zipfile
from pathlib import Path

import polars as pl
import pytest

from callrpt_db import cdr, config, mdrm, nic, panel, schedules

# --------------------------------------------------------------------------
# Fixture construction
# --------------------------------------------------------------------------


def make_zip(
    tmp_path: Path,
    period: str,
    schedules_data: dict[str, dict[int, dict[str, str]]],
    forms: dict[int, str] | None = None,
) -> Path:
    """Build a bulk zip with the real file shape: codes, labels, then data."""
    stamp = cdr.quarter_end(period).strftime("%m%d%Y")
    path = tmp_path / f"call_{period}.zip"
    rssds = sorted(
        {rssd for rows in schedules_data.values() for rssd in rows}
        | set(forms or {})
    )
    with zipfile.ZipFile(path, "w") as z:
        por = io.StringIO()
        por.write(
            '"IDRSSD"\tFDIC Certificate Number\tOCC Charter Number\t'
            "OTS Docket Number\tPrimary ABA Routing Number\t"
            "Financial Institution Name\tFinancial Institution Address\t"
            "Financial Institution City\tFinancial Institution State\t"
            "Financial Institution Zip Code\tFinancial Institution Filing Type\t"
            "Last Date/Time Submission Updated On\n"
        )
        for rssd in rssds:
            por.write(
                "\t".join(
                    [
                        str(rssd),
                        "0",
                        "0",
                        "0",
                        "0",
                        f"TEST BANK {rssd}",
                        "",
                        "",
                        "XX",
                        "00000",
                        (forms or {}).get(rssd, "031"),
                        "",
                    ]
                )
                + "\n"
            )
        z.writestr(f"FFIEC CDR Call Bulk POR {stamp}.txt", por.getvalue())

        for schedule, rows in schedules_data.items():
            codes = sorted({code for values in rows.values() for code in values})
            body = io.StringIO()
            body.write("\t".join(['"IDRSSD"', *codes]) + "\n")
            # The second row is labels, not data.  A reader that treats it as
            # data acquires an institution whose every field is a caption.
            body.write("\t".join(["", *[f"LABEL FOR {c}" for c in codes]]) + "\n")
            for rssd in sorted(rows):
                values = rows[rssd]
                body.write(
                    "\t".join(
                        [str(rssd), *[values.get(c, "") for c in codes]]
                    )
                    + "\n"
                )
            z.writestr(
                f"FFIEC CDR Call Schedule {schedule} {stamp}.txt", body.getvalue()
            )
    return path


def resolve(path: Path, specs: tuple[mdrm.ItemSpec, ...]) -> pl.DataFrame:
    roster = schedules.roster(path)
    long = schedules.read_period(path)
    return schedules.resolve_items(long, specs, forms=roster)


def value(frame: pl.DataFrame, rssd: int, column: str) -> float | None:
    row = frame.filter(pl.col("rssd") == rssd)
    assert row.height == 1, f"expected one row for {rssd}, got {row.height}"
    return row[column][0]


SPEC_BY_NAME = mdrm.BY_NAME


# --------------------------------------------------------------------------
# The reporting basis: RCFD against RCON
# --------------------------------------------------------------------------


def test_consolidated_basis_beats_domestic(tmp_path: Path) -> None:
    """JPMorgan's loan book is $1,497bn consolidated and $1,340bn domestic.

    Preferring RCON understates it by 11%, raises nothing, and looks entirely
    plausible next to its peers.
    """
    path = make_zip(
        tmp_path,
        "2025Q4",
        {"RCCI": {852218: {"RCFD2122": "1497490000", "RCON2122": "1339500000"}}},
    )
    frame = resolve(path, (mdrm.RCC_TOTAL,))
    assert value(frame, 852218, "loans_rcc_total") == 1_497_490_000 * 1_000


def test_domestic_basis_used_when_there_is_no_consolidated_one(
    tmp_path: Path,
) -> None:
    """Zions files no RCFD at all; insisting on it drops the bank entirely."""
    path = make_zip(
        tmp_path,
        "2025Q4",
        {"RCCI": {276579: {"RCON2122": "61118055"}}},
        forms={276579: "041"},
    )
    frame = resolve(path, (mdrm.RCC_TOTAL,))
    assert value(frame, 276579, "loans_rcc_total") == 61_118_055 * 1_000


def test_prefix_is_resolved_per_item_not_per_group(tmp_path: Path) -> None:
    """On form 031 an item can exist in only one of the two RC-C columns.

    A bank at 2020Q2 reports item 2081 consolidated but 1545 and J451
    domestic-only.  Choosing one prefix for the whole group takes whichever the
    first item used and drops the rest, reading $16.7bn of other loans as zero.
    """
    path = make_zip(
        tmp_path,
        "2020Q2",
        {
            "RCCI": {
                210434: {
                    "RCFD2081": "0",
                    "RCON1545": "2826591",
                    "RCONJ451": "13830195",
                }
            }
        },
    )
    frame = resolve(path, (SPEC_BY_NAME["loans_other_total"],))
    assert value(frame, 210434, "loans_other_total") == (
        2_826_591 + 13_830_195
    ) * 1_000


# --------------------------------------------------------------------------
# Choosing between reporting shapes
# --------------------------------------------------------------------------


def test_rollup_beats_a_detail_reported_as_zeros(tmp_path: Path) -> None:
    """Zions reports the five-way interbank split as zeros beside item 1288.

    "Prefer the detail where present" reads its interbank book as nothing;
    the real balance is the $59m on the single line.
    """
    path = make_zip(
        tmp_path,
        "2025Q4",
        {"RCCI": {276579: {"RCONB534": "0", "RCON1288": "59172"}}},
        forms={276579: "041"},
    )
    frame = resolve(path, (SPEC_BY_NAME["loans_depository_institutions"],))
    assert value(frame, 276579, "loans_depository_institutions") == 59_172 * 1_000


def test_detail_beats_a_rollup_when_more_completely_reported(
    tmp_path: Path,
) -> None:
    """BNY Mellon reports J454 -- half of one variant -- and all four of another.

    Matching a variant on "any component present" took the pair, kept J454's
    $7bn and dropped the $16bn of securities lending and other loans beside it.
    """
    path = make_zip(
        tmp_path,
        "2025Q4",
        {
            "RCCI": {
                541101: {
                    "RCFDJ454": "7182000",
                    "RCFD1545": "13891000",
                    "RCFD2081": "79000",
                    "RCFDJ451": "2585000",
                }
            }
        },
    )
    frame = resolve(path, (SPEC_BY_NAME["loans_other_total"],))
    expected = (7_182_000 + 13_891_000 + 79_000 + 2_585_000) * 1_000
    assert value(frame, 541101, "loans_other_total") == expected


def test_item_9_rollup_is_not_added_to_its_own_components(
    tmp_path: Path,
) -> None:
    """A filer reports J464 next to the very lines it totals.

    Its J464 of 11,441,353 is 1545 plus J451 to the dollar; counting both put
    the loan book 64% above the bank's own reported total.
    """
    path = make_zip(
        tmp_path,
        "2025Q4",
        {
            "RCCI": {
                3475083: {
                    "RCONJ464": "11441353",
                    "RCON1545": "11414640",
                    "RCONJ451": "26713",
                    "RCONJ454": "1137681",
                }
            }
        },
        forms={3475083: "041"},
    )
    frame = resolve(path, (SPEC_BY_NAME["loans_other_total"],))
    assert value(frame, 3475083, "loans_other_total") == (
        11_441_353 + 1_137_681
    ) * 1_000


def test_1563_covers_nondepository_lending_but_j464_does_not(
    tmp_path: Path,
) -> None:
    """Morgan Stanley Bank reports 1563, the whole of RC-C item 9.

    1563 is J454 + J451 + 1545 exactly -- it includes loans to nondepository
    financial institutions, which J464 does not -- so treating the two as the
    same rollup double counted $51bn and put the bank 62% over its own total.
    """
    path = make_zip(
        tmp_path,
        "2024Q3",
        {
            "RCCI": {
                1456501: {
                    "RCFD1563": "57175000",
                    "RCONJ454": "51150000",
                    "RCONJ451": "5066000",
                    "RCON1545": "959000",
                }
            }
        },
    )
    frame = resolve(path, (SPEC_BY_NAME["loans_other_total"],))
    assert value(frame, 1456501, "loans_other_total") == 57_175_000 * 1_000


def test_short_form_lines_are_read_where_the_detail_is_absent(
    tmp_path: Path,
) -> None:
    """A small filer reports 1766 and 2165 and none of the finer lines."""
    path = make_zip(
        tmp_path,
        "2025Q4",
        {"RCCI": {3407084: {"RCON1766": "293992", "RCON2165": "10133"}}},
        forms={3407084: "051"},
    )
    frame = resolve(
        path, (SPEC_BY_NAME["loans_ci"], SPEC_BY_NAME["loans_lease"])
    )
    assert value(frame, 3407084, "loans_ci") == 293_992 * 1_000
    assert value(frame, 3407084, "loans_lease") == 10_133 * 1_000


# --------------------------------------------------------------------------
# The two "total loans"
# --------------------------------------------------------------------------


def test_held_for_investment_is_not_the_rc_c_total(tmp_path: Path) -> None:
    """RC-C item 12 includes loans held for sale; RC item 4.b does not.

    The EDGAR panel's ``loans_total`` is held for investment, so using RC-C's
    total instead overstates JPMorgan by the $49bn it holds for sale.
    """
    path = make_zip(
        tmp_path,
        "2025Q4",
        {
            "RCCI": {852218: {"RCFD2122": "1497490000"}},
            "RC": {852218: {"RCFDB528": "1448038000", "RCFD5369": "49452000"}},
        },
    )
    frame = resolve(
        path,
        (
            mdrm.RCC_TOTAL,
            SPEC_BY_NAME["loans_total"],
            SPEC_BY_NAME["loans_held_for_sale"],
        ),
    )
    hfi = value(frame, 852218, "loans_total")
    hfs = value(frame, 852218, "loans_held_for_sale")
    total = value(frame, 852218, "loans_rcc_total")
    assert hfi is not None and hfs is not None and total is not None
    assert hfi != total
    assert hfi + hfs == total


def test_deposits_add_the_foreign_offices(tmp_path: Path) -> None:
    """There is no RCFD2200: total deposits are domestic plus foreign."""
    path = make_zip(
        tmp_path,
        "2025Q4",
        {"RC": {852218: {"RCON2200": "2167793000", "RCFN2200": "530049000"}}},
    )
    frame = resolve(path, (SPEC_BY_NAME["deposits"],))
    assert value(frame, 852218, "deposits") == (2_167_793_000 + 530_049_000) * 1_000


# --------------------------------------------------------------------------
# Reading the files
# --------------------------------------------------------------------------


def test_label_row_is_not_read_as_an_institution(tmp_path: Path) -> None:
    path = make_zip(tmp_path, "2025Q4", {"RCCI": {852218: {"RCFD2122": "1"}}})
    long = schedules.read_period(path)
    assert long["rssd"].unique().to_list() == [852218]


def test_confidential_and_not_required_are_null_not_zero(
    tmp_path: Path,
) -> None:
    """``CONF`` marks a line withheld, not a line that is zero.

    Reading it as zero reports a bank's confidential exposure as an absence of
    exposure, which is the worst direction for a credit benchmark to be wrong.
    """
    path = make_zip(
        tmp_path,
        "2025Q4",
        {"RCCI": {1: {"RCFD2122": "CONF"}, 2: {"RCFD2122": "NR"}}},
    )
    long = schedules.read_period(path)
    assert long.filter(pl.col("code") == "RCFD2122").is_empty()


def test_values_are_scaled_out_of_thousands(tmp_path: Path) -> None:
    path = make_zip(tmp_path, "2025Q4", {"RCCI": {1: {"RCFD2122": "1000"}}})
    frame = resolve(path, (mdrm.RCC_TOTAL,))
    assert value(frame, 1, "loans_rcc_total") == 1_000_000.0


def test_absent_items_give_null_not_zero(tmp_path: Path) -> None:
    """A category the bank does not report must not read as no exposure."""
    path = make_zip(tmp_path, "2025Q4", {"RCCI": {1: {"RCFD2122": "100"}}})
    frame = resolve(path, (SPEC_BY_NAME["loans_credit_card"],))
    assert value(frame, 1, "loans_credit_card") is None


def test_roster_carries_the_filing_type(tmp_path: Path) -> None:
    path = make_zip(
        tmp_path,
        "2025Q4",
        {"RCCI": {1: {"RCFD2122": "1"}}},
        forms={1: "051"},
    )
    roster = schedules.roster(path)
    assert roster["form"].to_list() == ["051"]


# --------------------------------------------------------------------------
# Flows
# --------------------------------------------------------------------------


def _flow_frame(values: list[tuple[dt.date, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "rssd": [1] * len(values),
            "period": [d for d, _ in values],
            "charge_offs_total": [v for _, v in values],
        }
    )


def test_year_to_date_flows_are_differenced_into_quarters() -> None:
    """JPMorgan's 2019 charge-offs run 730, 3346, 5022, 6810 and reset in Q1.

    Every RIAD item behaves this way, for every bank.  Reading them as filed
    makes Q4 charge-offs look like a full year of losses.
    """
    frame = panel.quarterize(
        _flow_frame(
            [
                (dt.date(2019, 3, 31), 730.0),
                (dt.date(2019, 6, 30), 3346.0),
                (dt.date(2019, 9, 30), 5022.0),
                (dt.date(2019, 12, 31), 6810.0),
                (dt.date(2020, 3, 31), 1902.0),
            ]
        )
    )
    assert frame["charge_offs_total"].to_list() == [
        730.0,
        2616.0,
        1676.0,
        1788.0,
        1902.0,
    ]


def test_flow_is_null_when_the_previous_quarter_is_missing() -> None:
    """A gap makes the difference meaningless; the year-to-date figure is not
    a quarter and must not be reported as one."""
    frame = panel.quarterize(
        _flow_frame(
            [
                (dt.date(2019, 3, 31), 730.0),
                (dt.date(2019, 12, 31), 6810.0),
            ]
        )
    )
    assert frame["charge_offs_total"].to_list() == [730.0, None]


def test_stocks_are_not_differenced() -> None:
    frame = pl.DataFrame(
        {
            "rssd": [1, 1],
            "period": [dt.date(2019, 3, 31), dt.date(2019, 6, 30)],
            "loans_total": [100.0, 120.0],
        }
    )
    assert panel.quarterize(frame)["loans_total"].to_list() == [100.0, 120.0]


# --------------------------------------------------------------------------
# The partition check
# --------------------------------------------------------------------------


def test_unearned_income_is_subtracted_from_the_partition() -> None:
    """RC-C item 12 is the categories *less* item 11, unearned income.

    Almost every filer reports zero there, which is why leaving it out went
    unnoticed until City National Bank -- $562m of unearned income put it 0.86%
    over its own total in every quarter, with nothing else wrong.
    """
    columns = {name: [0.0] for name in panel.MIX_LEAF_CATEGORIES}
    columns["loans_ci"] = [66_285_066.0]
    columns["loans_unearned_income"] = [562_051.0]
    columns["loans_rcc_total"] = [65_723_015.0]
    frame = panel.add_partition_check(pl.DataFrame(columns))
    assert frame["rcc_residual"][0] == pytest.approx(0.0)


def test_unallocated_is_reported_rather_than_hidden() -> None:
    """On 031 some lines are collected domestic-only while the total is
    consolidated, so a bank with foreign offices has loans in no category."""
    columns = {name: [0.0] for name in panel.MIX_LEAF_CATEGORIES}
    columns["loans_ci"] = [80.0]
    columns["loans_rcc_total"] = [100.0]
    frame = panel.add_partition_check(pl.DataFrame(columns))
    assert frame["loans_unallocated"][0] == pytest.approx(20.0)
    assert frame["rcc_residual_pct"][0] == pytest.approx(-20.0)


def test_cre_needs_no_completeness_judgement() -> None:
    """The four CRE classes are fixed lines on the form, so their sum is the
    CRE book by definition -- unlike the XBRL path, where a single tagged
    class must not be promoted to the total."""
    frame = panel.with_cre_rollup(
        pl.DataFrame(
            {
                "loans_cre_owner_occupied": [10.0],
                "loans_cre_investor": [20.0],
                "loans_construction": [5.0],
                "loans_multifamily": [1.0],
            }
        )
    )
    assert frame["loans_cre_total"][0] == pytest.approx(36.0)


def test_missing_nonaccrual_does_not_become_a_zero_npa() -> None:
    frame = panel.with_npa(
        pl.DataFrame({"nonaccrual_total": [None], "oreo": [5.0]})
    )
    assert frame["npa_total"][0] is None


# --------------------------------------------------------------------------
# Identity and structure
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("050412693", "050412693"),
        ("50412693", "050412693"),
        ("05-0412693", "050412693"),
        ("0", None),
        ("", None),
        (None, None),
    ],
)
def test_ein_normalisation_survives_a_leading_zero(
    raw: str | None, expected: str | None
) -> None:
    """NIC stores the EIN as a number and drops the leading zero.

    Joining the raw strings misses Citizens Financial Group, State Street and
    Santander USA -- and Citizens then fell through to a name match, where
    there is an unrelated company of the same name to match against.
    """
    assert nic.normalise_ein(raw) == expected


def test_period_ids_are_not_derivable() -> None:
    """2025Q3 is id 149 and 2025Q2 is 147; 148 does not exist.

    Nothing may assume the ids are contiguous -- they are read from the page.
    """
    assert cdr.quarter_end("2025Q4") == dt.date(2025, 12, 31)
    assert cdr.period_label("2025Q4") == "12/31/2025"
    assert cdr.periods_between(
        dt.date(2025, 1, 1), dt.date(2025, 12, 31)
    ) == ["2025Q1", "2025Q2", "2025Q3", "2025Q4"]


def test_a_holding_company_may_have_more_than_one_top_rssd() -> None:
    """Zions' holding company was merged into its own bank in 2018, so the
    organisation is one RSSD before that date and a different one after."""
    zion = next(h for h in config.HOLDINGS if h.ticker == "ZION")
    assert set(zion.rssds) == {1027004, 276579}


def test_every_holding_resolves_to_an_rssd() -> None:
    assert all(h.rssd for h in config.HOLDINGS)


def test_the_non_sec_ihcs_are_all_present() -> None:
    """The gap bankqtr_db documents and cannot fill is the point of this build."""
    from bankqtr_db.config import NON_SEC_IHCS

    names = {h.name for h in config.HOLDINGS}
    assert set(NON_SEC_IHCS) <= names


def test_ratio_definitions_are_shared_with_the_edgar_panel() -> None:
    """Imported, not restated, so the two panels cannot drift on what a ratio
    means."""
    from bankqtr_db.variables import RATIOS
    from callrpt_db.panel import RATIOS as CALL_RATIOS

    assert CALL_RATIOS is RATIOS


# --------------------------------------------------------------------------
# Integration -- skipped without a cached quarter
# --------------------------------------------------------------------------


def _cached() -> list[str]:
    return cdr.cached_periods()


needs_cache = pytest.mark.skipif(
    not _cached(), reason="no cached Call Report period; run scripts/fetch_call.py"
)


@needs_cache
def test_partition_ties_for_a_real_quarter() -> None:
    """The leaves must never sum to *more* than RC-C's own total.

    Under-allocation is the form -- some 031 lines are domestic-only -- but
    over-allocation is always a mapping defect, and every one found so far
    (J464 beside its parts, 1563 read as J464, unearned income not subtracted)
    showed up here first.
    """
    period = _cached()[-1]
    path = cdr.zip_path(period)
    roster = schedules.roster(path)
    long = schedules.read_period(path, codes=mdrm.wanted_codes())
    specs = (
        *mdrm.RCC_LOAN_ITEMS,
        mdrm.RCC_TOTAL,
        mdrm.RCC_UNEARNED,
    )
    frame = panel.add_partition_check(
        schedules.resolve_items(long, specs, forms=roster)
    )
    checked = frame.filter(
        pl.col("loans_rcc_total").is_not_null()
        & (pl.col("loans_rcc_total").abs() > 0)
    )
    assert checked.height > 1000
    # One reporting unit of tolerance: the form is filed in whole thousands.
    over = checked.filter(pl.col("rcc_residual") > 1_000.0)
    assert over.is_empty(), over.select(
        "rssd", "loans_rcc_total", "rcc_residual"
    ).head(10)


@needs_cache
def test_the_universe_maps_onto_real_filers() -> None:
    period = _cached()[-1]
    holdings = tuple(h for h in config.HOLDINGS if h.tier == "dfast")
    mapping = panel.universe_filers([period], holdings)
    assert mapping.height > 30
    # No charter may be claimed by two holding companies in the same quarter.
    per_charter = mapping.group_by(["rssd", "period"]).agg(
        pl.col("ticker").n_unique().alias("n")
    )
    assert per_charter.filter(pl.col("n") > 1).is_empty()
