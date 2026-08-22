"""End-to-end build: cached FFIEC bulk data -> bank-quarter panel + cross-check.

uv run python scripts/build_call_panel.py                      # 2001Q1 onward, with lineage
uv run python scripts/build_call_panel.py --since 2013-01-01 --no-lineage
uv run python scripts/build_call_panel.py --comparators
uv run python scripts/build_call_panel.py --no-crosscheck

Reads only what ``fetch_call.py`` has already cached and never touches the
network, so a rebuild is reproducible.  Outputs land in ``data/out``:

    call_panel.parquet / .csv       the holding-company panel
    call_panel_charters.parquet     one row per bank charter per quarter
    call_panel_coverage.csv         per bank and variable, how many quarters
    call_panel_coverage_delta.csv   before/after: what the lineage added, per bank
    call_panel_flags.csv            bank-quarters failing a check or carrying
                                    synthetic (predecessor) history
    call_panel_build_info.json      window, settings, commit, cell counts, and
                                    the notes on every schedule break and
                                    variable decision the 2001 window required
    rssd_lineage.csv                every predecessor of every firm, with dates
    source_diff.csv                 EDGAR against FFIEC, per bank-quarter
    source_diff_summary.csv         the same, aggregated per bank and variable
    source_coverage.csv             which firms each source reaches
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import subprocess
import sys

import polars as pl

from callrpt_db import cdr, config, crosscheck, lineage, mdrm, panel, schedules

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("build_call")

# A residual under one reporting unit is rounding, not a mapping error: the
# Call Report is filed in whole thousands, so a difference below $1,000 cannot
# be a real disagreement about a category.
PARTITION_TOLERANCE_USD = 1_000.0


def _date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=config.ROOT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _filled_cells(frame: pl.DataFrame) -> int:
    numeric = [c for c in frame.columns if frame.schema[c] in (pl.Float64, pl.Int64)]
    return int(sum(frame[c].is_not_null().sum() for c in numeric))


def coverage(frame: pl.DataFrame) -> pl.DataFrame:
    """Per bank and variable: quarters populated, and the window covered."""
    if frame.is_empty():
        return pl.DataFrame()
    value_cols = [
        c
        for c in frame.columns
        if frame.schema[c] == pl.Float64 and c not in ("rcc_residual_pct",)
    ]
    rows = []
    for ticker, group in frame.group_by("ticker"):
        name = ticker[0] if isinstance(ticker, tuple) else ticker
        total = group.height
        for column in value_cols:
            filled = int(group[column].is_not_null().sum())
            rows.append(
                {
                    "ticker": name,
                    "variable": column,
                    "quarters": total,
                    "populated": filled,
                    "pct": round(filled / total * 100, 1) if total else None,
                }
            )
    return pl.DataFrame(rows).sort(["variable", "ticker"])


BASELINE_SINCE = dt.date(2013, 1, 1)


def baseline_panel(
    cached: list[str], holdings: tuple[config.Holding, ...], until: dt.date
) -> pl.DataFrame:
    """The panel as the 2013 build defined it: each firm's own 2026 RSSD subtree.

    Rebuilt in-process rather than read from disk, so that the before/after
    table compares against the documented baseline even when the file on disk
    is already a lineage build.
    """
    periods = [p for p in cached if BASELINE_SINCE <= cdr.quarter_end(p) <= until]
    if not periods:
        return pl.DataFrame()
    _charters, frame = panel.build(periods, holdings, lineages=None, derived=False)
    return frame


def coverage_delta(before: pl.DataFrame, after: pl.DataFrame) -> pl.DataFrame:
    """Per firm: bank-quarters before and after, and how much is predecessor history.

    ``before`` is either the panel on disk when the build started or the
    2013 own-subtree baseline rebuilt by :func:`baseline_panel`, per
    ``--baseline``; the table answers the question actually asked -- what did
    the lineage extension add.
    """
    if after.is_empty():
        return pl.DataFrame()
    summary = after.group_by("ticker").agg(
        pl.col("bank").first().alias("bank"),
        pl.len().alias("quarters_after"),
        pl.col("period").min().alias("first_quarter_after"),
        pl.col("has_predecessor").sum().alias("quarters_with_predecessors")
        if "has_predecessor" in after.columns
        else pl.lit(0).alias("quarters_with_predecessors"),
        pl.col("predecessor_failed").sum().alias("quarters_with_failed_predecessor")
        if "predecessor_failed" in after.columns
        else pl.lit(0).alias("quarters_with_failed_predecessor"),
        pl.col("predecessor_count").max().alias("max_predecessors")
        if "predecessor_count" in after.columns
        else pl.lit(0).alias("max_predecessors"),
        pl.col("n_insured_not_filing").max().alias("max_insured_not_filing")
        if "n_insured_not_filing" in after.columns
        else pl.lit(0).alias("max_insured_not_filing"),
    )
    if before.is_empty() or "ticker" not in before.columns:
        prior = pl.DataFrame(
            schema={"ticker": pl.Utf8, "quarters_before": pl.UInt32, "first_quarter_before": pl.Date}
        )
    else:
        prior = before.group_by("ticker").agg(
            pl.len().alias("quarters_before"),
            pl.col("period").min().alias("first_quarter_before"),
        )
    out = summary.join(prior, on="ticker", how="left").with_columns(
        pl.col("quarters_before").fill_null(0),
        (pl.col("quarters_after") - pl.col("quarters_before").fill_null(0)).alias("quarters_added"),
    )
    return out.select(
        "ticker",
        "bank",
        "quarters_before",
        "quarters_after",
        "quarters_added",
        "first_quarter_before",
        "first_quarter_after",
        "quarters_with_predecessors",
        "quarters_with_failed_predecessor",
        "max_predecessors",
        "max_insured_not_filing",
    ).sort(["quarters_added", "quarters_with_predecessors"], descending=[True, True])


def flags(frame: pl.DataFrame) -> pl.DataFrame:
    """Bank-quarters that fail a sanity check, or that carry synthetic history.

    The arithmetic checks are the ones the Call Report makes possible and the
    XBRL panel cannot run: the form states its own totals, so a category that
    exceeds its parent or a partition that does not cross-foot is a defect
    rather than a disclosure choice.

    The predecessor flags are not failures.  They mark the bank-quarters whose
    figures were summed across organisations that were separate at the time
    -- and, within those, the ones containing an institution that later
    failed -- so a model calibrated on this panel can see which observations
    are the 2026 firm, which are its reconstructed past, and which are the
    stress cases.
    """
    if frame.is_empty():
        return pl.DataFrame()
    checks: list[tuple[str, pl.Expr, pl.Expr]] = []
    have = set(frame.columns)

    def const(text: str) -> pl.Expr:
        return pl.lit(text)

    if {"loans_total", "acl_total"} <= have:
        checks.append(
            (
                "acl_exceeds_loans",
                pl.col("acl_total") > pl.col("loans_total"),
                const("allowance is larger than the loan book"),
            )
        )
    if {"loans_total", "loans_cre_total"} <= have:
        checks.append(
            (
                "cre_exceeds_loans",
                pl.col("loans_cre_total") > pl.col("loans_total") * 1.001,
                const("CRE is larger than total loans"),
            )
        )
    if "rcc_residual" in have:
        checks.append(
            (
                "rcc_partition_over",
                pl.col("rcc_residual") > PARTITION_TOLERANCE_USD,
                const("loan categories sum to more than RC-C's own total"),
            )
        )
        checks.append(
            (
                "rcc_partition_under",
                pl.col("rcc_residual") < -PARTITION_TOLERANCE_USD,
                const("loan categories leave part of RC-C's total unallocated"),
            )
        )
    if {"nonaccrual_total", "loans_total"} <= have:
        checks.append(
            (
                "nonaccrual_exceeds_loans",
                pl.col("nonaccrual_total") > pl.col("loans_total"),
                const("nonaccrual is larger than the loan book"),
            )
        )
    if "loans_total" in have:
        checks.append(
            (
                "negative_loans",
                pl.col("loans_total") < 0,
                const("negative total loans"),
            )
        )
    # A subset that exceeds the set it is drawn from.  These schedules state no
    # total of their own, so an inequality is the whole of what the form
    # guarantees -- see ``mdrm.BoundCheck``.  Each one fails loudly under the
    # mapping the MDRM codes' names suggest, which is how RC-C Part II's
    # alternating count and balance columns were told apart.
    for bound in mdrm.bound_checks():
        if {bound.part, bound.whole} <= have:
            checks.append(
                (
                    bound.name,
                    pl.col(bound.part)
                    > pl.col(bound.whole) * (1 + bound.tolerance),
                    pl.format(
                        "{} exceeds {}",
                        const(bound.part),
                        const(bound.whole),
                    ),
                )
            )
    # --- synthetic history -------------------------------------------------
    if {"has_predecessor", "predecessor_count", "predecessors"} <= have:
        checks.append(
            (
                "predecessor_history",
                pl.col("has_predecessor"),
                pl.format(
                    "summed across {} predecessor organisation(s) not yet part of the firm: {}",
                    pl.col("predecessor_count"),
                    pl.col("predecessors"),
                ),
            )
        )
    if "predecessor_failed" in have:
        checks.append(
            (
                "predecessor_failed",
                pl.col("predecessor_failed"),
                const("includes an institution that subsequently failed (FDIC-assisted resolution)"),
            )
        )
    if {"n_insured_not_filing", "insured_not_filing"} <= have:
        checks.append(
            (
                "insured_depository_not_filing",
                pl.col("n_insured_not_filing") > 0,
                pl.format(
                    "{} insured depository(ies) in the organisation filed no Call Report this quarter (TFR filers before 2012): {}",
                    pl.col("n_insured_not_filing"),
                    pl.col("insured_not_filing"),
                ),
            )
        )

    frames = []
    for name, expr, detail in checks:
        hit = frame.filter(expr.fill_null(False))
        if hit.is_empty():
            continue
        keep = [c for c in ("ticker", "bank", "period") if c in hit.columns]
        frames.append(
            hit.select([*keep, pl.lit(name).alias("flag"), detail.alias("detail")])
        )
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical_relaxed").sort(["ticker", "period", "flag"])


# --------------------------------------------------------------------------
# What the 2001 window required, written into the build info
# --------------------------------------------------------------------------

SCHEDULE_BREAKS = [
    {
        "quarter": "2002Q1",
        "change": "Closed-end 1-4 family split into first and junior liens on RC-N and RI-B; fed funds purchased and repos split.",
        "items": "5401/5402/5403 -> C236-C239, C229/C230; 5413/5414 -> C234/C235, C217/C218; 2800 -> B993 + B995",
        "handling": "retired codes listed as alternatives; the 2001 quarters read the single line",
        "variables": ["dpd_30_89_resi_mortgage", "dpd_90_plus_resi_mortgage", "nonaccrual_resi_mortgage", "charge_offs_resi_mortgage", "recoveries_resi_mortgage", "fed_funds_repo_purchased"],
    },
    {
        "quarter": "2007Q1",
        "change": "Construction and nonfarm nonresidential real estate each split in two on RC-C, RC-N, RI-B and RC-L; leases split in two on RC-C.",
        "items": "1415 -> F158 + F159; 1480 -> F160 + F161; RC-N 2759/2769/3492 -> F172-F177, 3502/3503/3504 -> F178-F183; RI-B 3582/3583 -> C891-C894, 3590/3591 -> C895-C898; RC-L 3816 -> F164 + F165; 2165 -> F162 + F163",
        "handling": "totals are continuous (construction, nonfarm nonresidential, leases); the owner-occupied / investor split is null before 2007. Through 2007 the old codes are carried as derived totals beside the detail and tie to it exactly, so the most-complete-variant rule picks the detail.",
        "variables": ["loans_construction", "loans_cre_nonfarm_nonres", "loans_cre_owner_occupied", "loans_cre_investor", "loans_lease", "nonaccrual_*", "dpd_*", "charge_offs_*", "commitments_cre_construction"],
    },
    {
        "quarter": "2010Q1",
        "change": "RC-C item 9 'other loans' reorganised around loans to nondepository financial institutions; RC-L other commitments split three ways.",
        "items": "1563 (1545/2081/1564 detail) -> J454 + J464 (1545/2081/J451); 3818 -> J457 + J458 + J459",
        "handling": "loans_other_total continuous; loans_nondepository_fi null before 2010; commitments_other continuous",
        "variables": ["loans_other_total", "loans_other", "loans_nondepository_fi", "commitments_other"],
    },
    {
        "quarter": "2011Q1",
        "change": "'Other loans to individuals' split into automobile and other on RC-C, RC-N and RI-B; TDRs reported by loan category.",
        "items": "2011 -> K137 + K207; RC-N B578/B579/B580 -> K213-K218; RI-B B516/B517 -> K129/K133 + K205/K206; 1616 -> K158...K165",
        "handling": "loans_consumer_installment, charge_offs_consumer_noncard and the RC-N consumer_installment rows are continuous; loans_auto and loans_consumer_other null before 2011",
        "variables": ["loans_consumer_installment", "loans_auto", "loans_consumer_other", "charge_offs_consumer_noncard", "recoveries_consumer_noncard", "nonaccrual_consumer_installment", "loans_tdr_accruing"],
    },
    {
        "quarter": "2015Q1",
        "change": "Basel III Schedule RC-R: new prefixes (RCFA/RCOA on RCRI) and CET1; 2014 is a transition year on RCRIA/RCRIB.",
        "items": "8274/3792/A223 on RCR (to 2013Q4) and RCRIA/RCRIB (2014) vs RCRI (2015+); P859 from 2015",
        "handling": "cet1_capital, tier1_capital, total_capital, risk_weighted_assets stay null before 2015Q1 as before; the pre-2015 regime is carried under its own names (*_basel1); tier1_leverage_ratio splices the two numerators over RC-K average assets",
        "variables": ["cet1_capital", "tier1_capital", "total_capital", "risk_weighted_assets", "tier1_capital_basel1", "total_capital_basel1", "risk_weighted_assets_basel1", "tier1_leverage_ratio"],
    },
    {
        "quarter": "2017Q1",
        "change": "RC-N gains a total row. Before it the form states no total for past-due and nonaccrual loans.",
        "items": "1403/1406/1407 exist from 2017Q1 only",
        "handling": "before 2017Q1 the totals are the sum of fourteen category rows (agricultural and RE-to-non-US-addressees are memoranda and excluded; the foreign-office line is included for form 031); the list reproduces the form's own total for 99.6% of filers from 2017 on. rcn_total_built marks the rows where the total was built. This also corrects the previous build, which carried 0.0 for 2013-2016.",
        "variables": ["nonaccrual_total", "pd_dpd_30_89", "pd_dpd_90_plus", "noncurrent_total", "npa_total", "and every ratio on them"],
    },
    {
        "quarter": "2018Q2",
        "change": "Goodwill moves from RC to RC-M; other intangibles replaced by a total intangibles line.",
        "items": "3163 on RC to 2018Q1, on RCM from 2018Q2; 0426 -> 2143",
        "handling": "goodwill read from either schedule; intangibles_total is 2143 or 3163 + 0426; intangibles_other derived",
        "variables": ["goodwill", "intangibles_total", "intangibles_other", "tangible_common_equity", "tce_ratio", "texas_ratio"],
    },
    {
        "quarter": "2020Q1",
        "change": "CECL adoption (basis column); unchanged from the previous build.",
        "items": "-",
        "handling": "basis = incurred / cecl",
        "variables": ["acl_total", "provision_total", "reserve_coverage", "reserve_to_nonaccrual", "reserve_to_noncurrent"],
    },
    {
        "quarter": "2023Q1",
        "change": "ASU 2022-02 replaces troubled debt restructurings with 'modifications to borrowers experiencing financial difficulty'.",
        "items": "HK25 and K158... retained under the new definition",
        "handling": "loans_tdr_accruing continues; values from 2023Q1 are a broader population and not comparable with earlier ones",
        "variables": ["loans_tdr_accruing", "tdr_pct"],
    },
]

VARIABLES_ADDED = [
    {"name": "loans_cre_nonfarm_nonres", "schedule": "RC-C", "items": "F160 + F161 | 1480", "since": "2001Q1", "rationale": "the CRE line the whole window supports; owner-occupied vs investor cannot be split before 2007", "comparable_across_window": True},
    {"name": "loans_consumer_installment", "schedule": "RC-C", "items": "K137 + K207 | 2011", "since": "2001Q1", "rationale": "auto plus other consumer on both eras' terms; loans_auto is null before 2011", "comparable_across_window": True},
    {"name": "loans_consumer_revolving_other", "schedule": "RC-C", "items": "B539", "since": "2001Q1", "rationale": "partition leaf split out so the consumer book closes in every era", "comparable_across_window": True},
    {"name": "loans_consumer_noncard", "schedule": "derived", "items": "loans_consumer_revolving_other + loans_consumer_installment", "since": "2001Q1", "rationale": "denominator for the consistent non-card consumer charge-off rate", "comparable_across_window": True},
    {"name": "loans_tdr_accruing", "schedule": "RC-C memoranda", "items": "HK25 | 1616 | K158..K165 (+K256)", "since": "2001Q1", "rationale": "restructured loans ran from 0.2% to 2%+ of loans through 2009-2012 and led charge-offs; the single most under-used pre-GFC deterioration signal", "comparable_across_window": False, "comparability_note": "definition broadens at 2023Q1 (ASU 2022-02); category detail 2011-2016 vs single line before"},
    {"name": "dpd_30_89_* / dpd_90_plus_* / nonaccrual_* by category (14 rows + 4 sub-splits)", "schedule": "RC-N", "items": "see mdrm.RCN_BY_CATEGORY", "since": "2001Q1 (construction/CRE splits 2007, consumer splits 2011)", "rationale": "early-stage delinquency by loan type is the leading indicator a stress model is calibrated on; 30-89 day residential and construction delinquencies turned up in 2006, four quarters before nonaccruals", "comparable_across_window": True},
    {"name": "nonaccrual_total / pd_dpd_30_89 / pd_dpd_90_plus before 2017Q1", "schedule": "RC-N", "items": "sum of category rows", "since": "2001Q1", "rationale": "the form had no total row; previous build carried 0.0 for 2013-2016", "comparable_across_window": True, "comparability_note": "built for ~99.6% agreement with the form's own total where both exist; 031 filers can differ by under 0.5%"},
    {"name": "noncurrent_total, noncurrent_ratio, reserve_to_noncurrent", "schedule": "derived", "items": "pd_dpd_90_plus + nonaccrual_total", "since": "2001Q1", "rationale": "the FDIC's noncurrent definition; the standard asset-quality series in every published GFC study", "comparable_across_window": True},
    {"name": "texas_ratio", "schedule": "derived", "items": "(noncurrent + OREO) / (tangible equity + ACL)", "since": "2001Q1", "rationale": "the failure predictor that worked in 2009-2011; useless after 2013 and essential before", "comparable_across_window": True},
    {"name": "charge_offs_cre_nonfarm_nonres, recoveries_cre_nonfarm_nonres, nco_cre_nonfarm_nonres, nco_cre_total", "schedule": "RI-B", "items": "C895 + C897 | 3590; C896 + C898 | 3591", "since": "2001Q1", "rationale": "CRE charge-offs continuous across the 2007 split; nco_cre_total gives nco_rate_cre its numerator", "comparable_across_window": True},
    {"name": "charge_offs_consumer_noncard, recoveries_consumer_noncard, nco_consumer_noncard", "schedule": "RI-B", "items": "K129 + K205 | B516; K133 + K206 | B517", "since": "2001Q1", "rationale": "consumer losses excluding cards on both eras' terms; the 2008-2010 auto and installment loss cycle", "comparable_across_window": True},
    {"name": "nco_rate_construction, nco_rate_cre_nonfarm_nonres, nco_rate_multifamily, nco_rate_resi, nco_rate_home_equity, nco_rate_consumer_noncard, provision_rate", "schedule": "derived", "items": "annualised flow / balance", "since": "2001Q1", "rationale": "loss rates by segment are the calibration targets of a stress loss model", "comparable_across_window": True},
    {"name": "commitments_* (6 lines), standby_letters_of_credit, commitments_total, commitments_to_loans", "schedule": "RC-L", "items": "3814, 3815, F164 + F165 | 3816, 6550, 3817, J457-J459 | 3818, 3819", "since": "2001Q1", "rationale": "off-balance-sheet exposure drawn in a stress; HELOC and construction commitments in 2006 were the unfunded pipeline of 2008", "comparable_across_window": True},
    {"name": "deposits_brokered, brokered_deposits_pct", "schedule": "RC-E", "items": "2365", "since": "2001Q1", "rationale": "hot-money share of funding; the liquidity variable that identified the 2008-2010 failures", "comparable_across_window": True},
    {"name": "fed_funds_repo_purchased, borrowings_other, wholesale_funding, wholesale_funding_pct", "schedule": "RC", "items": "B993 + B995 | 2800; 3190", "since": "2001Q1", "rationale": "wholesale funding reliance, the run-prone liabilities of 2008", "comparable_across_window": True},
    {"name": "goodwill, intangibles_total, intangibles_other, preferred_stock, tangible_equity, tangible_common_equity, tangible_assets, tce_ratio, equity_to_assets", "schedule": "RC / RC-M", "items": "3163; 2143 | 3163 + 0426; 3838", "since": "2001Q1", "rationale": "the capital measure the market priced on in 2008-2009 when regulatory ratios did not move; TARP preferred is deducted", "comparable_across_window": True},
    {"name": "tier1_capital_basel1, total_capital_basel1, risk_weighted_assets_basel1, tier1_ratio_basel1, total_capital_ratio_basel1", "schedule": "RC-R (pre-2015)", "items": "8274, 3792, A223 on RCR / RCRIA / RCRIB", "since": "2001Q1 to 2014Q4", "rationale": "regulatory capital through two cycles, under the definitions then in force; kept apart from the Basel III columns, which stay null before 2015", "comparable_across_window": False, "comparability_note": "different capital and risk-weight definitions from the 2015+ columns; never spliced"},
    {"name": "tier1_leverage_ratio", "schedule": "derived", "items": "coalesce(tier1_capital, tier1_capital_basel1) / assets_average", "since": "2001Q1", "rationale": "the one capital ratio whose denominator did not change in 2015 and that still carried information in 2008", "comparable_across_window": False, "comparability_note": "numerator definition steps at 2015Q1; denominator is RC-K average assets rather than the regulatory leverage denominator"},
    {"name": "interest_income, interest_expense, ppnr, ppnr_rate, roa, nii_to_avg_assets, assets_average, loans_average", "schedule": "RI / RC-K", "items": "4107, 4073, 3368, 3360", "since": "2001Q1", "rationale": "pre-provision net revenue is the quantity DFAST projects and provisions are absorbed by; ROA and NII on average assets are its components", "comparable_across_window": True},
    {"name": "securities_htm, securities_htm_fair_value, securities_afs_amortized_cost, securities_afs_unrealized, securities_htm_unrealized", "schedule": "RC-B", "items": "1754, 1771, 1772", "since": "2001Q1", "rationale": "unrealised securities losses: 2022-2023, and the 2008 private-label MBS marks", "comparable_across_window": True},
]

CORRECTIONS = [
    {"variable": "nonaccrual_cre_owner_occupied, nonaccrual_cre_investor", "was": "read F180 / F181, the 90-days-past-due column", "now": "F182 / F183, the nonaccrual column; F180 / F181 are carried as dpd_90_plus_cre_owner_occupied / dpd_90_plus_cre_investor"},
    {"variable": "nonaccrual_total, pd_dpd_30_89, pd_dpd_90_plus, npa_total and ratios on them, 2013Q1-2016Q4", "was": "0.0 in every bank-quarter (no total row on the form, and the rollup summed all-null to zero)", "now": "built from the category rows; null only where no category is reported"},
    {"variable": "cet1_capital, tier1_capital, total_capital, risk_weighted_assets and their ratios, 2013Q1-2014Q4", "was": "0.0 (all-null charters summed to zero)", "now": "null, as documented"},
    {"variable": "every holding-company column", "was": "a column no charter reported rolled up as 0.0", "now": "null"},
    {"variable": "charge_offs_ci, recoveries_ci, nonaccrual_ci, dpd_30_89_ci", "was": "null for every form 041 filer (only the 031 by-addressee items were mapped)", "now": "the 041 single line (4638 / 4608 / 1608 / 1606) is an alternative"},
    {"variable": "loans_other_total, loans_other (form 031 before 2010)", "was": "1545 + 2081 only", "now": "includes 1564 'all other loans', so the pre-2010 detail equals 1563"},
    {"variable": "flows in the quarter of a pooling (common-control) merger", "was": "the survivor's difference carried the absorbed charter's year-to-date, already counted under its own RSSD", "now": "subtracted, using NIC ACCT_METHOD = 1 (pooling_adjustment in panel.quarterize)"},
]

NOT_STRICTLY_COMPARABLE = {
    "loans_cre_owner_occupied / loans_cre_investor and their RC-N, RI-B companions": "null before 2007Q1; use loans_cre_nonfarm_nonres across the window",
    "loans_auto / loans_consumer_other and their RC-N, RI-B companions": "null before 2011Q1; use loans_consumer_installment / charge_offs_consumer_noncard across the window",
    "loans_nondepository_fi, loans_securities_based": "null (nondepository FI) or 031-only (securities) before 2010Q1",
    "loans_tdr_accruing, tdr_pct": "definition change at 2023Q1 (ASU 2022-02); 2011-2016 is a sum of category lines",
    "tier1_leverage_ratio": "Tier 1 numerator changes definition at 2015Q1; denominator is RC-K average assets",
    "*_basel1 capital columns vs cet1_/tier1_/total_capital": "two regimes, never spliced; the Basel III columns are null before 2015Q1 and the legacy ones null after 2014Q4",
    "nonaccrual_total, pd_dpd_30_89, pd_dpd_90_plus before 2017Q1": "built from category rows (rcn_total_built = true); reproduces the form's total for 99.6% of filers after 2017, 031 filers can differ by under 0.5%",
    "acl_total, provision_total and reserve ratios across 2020Q1": "CECL (basis column), as before",
    "every variable in a bank-quarter with has_predecessor = true": "summed across organisations that were separate at the time; flows in the quarter of a purchase-accounting merger omit the absorbed charter's activity between the prior quarter end and the merger date",
    "every variable in a bank-quarter with n_insured_not_filing > 0": "the organisation owned an insured depository that filed no Call Report (a TFR-filing thrift before 2012Q1), so the sum is a floor",
}

GFC_VALUE = [
    "noncurrent_total / noncurrent_ratio / texas_ratio: the series that separated the 2009-2011 failures from the survivors",
    "dpd_30_89_resi_mortgage, dpd_30_89_construction, dpd_30_89_home_equity: turned up in 2006, four quarters ahead of nonaccruals",
    "loans_construction, construction_pct, nco_rate_construction: the concentration that sank Colonial, the FBOP banks and Silver State; all in the lineage with predecessor_failed = true",
    "loans_tdr_accruing: the 2009-2012 restructuring wave, invisible in a 2013-start window",
    "nco_rate_resi, nco_rate_home_equity, nco_rate_consumer_noncard: loss rates through a full housing cycle, the calibration targets for retail loss models",
    "provision_rate, ppnr_rate, roa: the income-statement side of the 2008-2010 stress, including the quarters PPNR did not cover the provision",
    "tier1_leverage_ratio, tce_ratio, tier1_ratio_basel1: capital as it was measured, and as the market re-measured it, going into 2008",
    "deposits_brokered, brokered_deposits_pct, wholesale_funding_pct: the funding profile of the institutions that failed",
    "commitments_total, commitments_cre_construction: the unfunded pipeline that became 2008-2009 balances",
    "predecessor_failed bank-quarters: 95 FDIC-assisted resolutions in the lineage, each with the failed institution's own Call Reports in the quarters before",
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since", type=_date, default=dt.date(2001, 1, 1))
    ap.add_argument("--until", type=_date, default=None)
    ap.add_argument(
        "--comparators",
        action="store_true",
        help="include the ten non-DFAST regional comparators",
    )
    ap.add_argument("--tickers", nargs="*", help="restrict to these tickers")
    ap.add_argument(
        "--no-crosscheck",
        action="store_true",
        help="skip the comparison against the EDGAR panel",
    )
    ap.add_argument(
        "--no-lineage",
        action="store_true",
        help="sum only each firm's own 2026 RSSD subtree, as the 2013 build did",
    )
    ap.add_argument(
        "--tfr",
        action="store_true",
        help=(
            "fold in FDIC BankFind history for the insured depositories that "
            "filed a Thrift Financial Report rather than a Call Report; reads "
            "only what fetch_call.py --tfr has cached"
        ),
    )
    ap.add_argument(
        "--baseline",
        choices=("previous", "own-2013", "none"),
        default="own-2013",
        help=(
            "what the before/after coverage table compares against: the panel on "
            "disk when the build started, the 2013 own-subtree build rebuilt "
            "in-process (default), or nothing"
        ),
    )
    args = ap.parse_args(argv)

    config.ensure_dirs()
    started = dt.datetime.now(dt.UTC)

    cached = cdr.cached_periods()
    if not cached:
        log.error("no cached periods; run scripts/fetch_call.py first")
        return 1
    until = args.until or cdr.quarter_end(cached[-1])
    periods = [
        p for p in cached if args.since <= cdr.quarter_end(p) <= until
    ]
    if not periods:
        log.error("no cached periods inside %s..%s", args.since, until)
        return 1

    tiers = {"dfast", "ihc", "inactive"}
    if args.comparators:
        tiers.add("comparator")
    holdings = tuple(h for h in config.HOLDINGS if h.tier in tiers)
    if args.tickers:
        wanted = {t.upper() for t in args.tickers}
        holdings = tuple(h for h in holdings if h.ticker.upper() in wanted)
    if not holdings:
        log.error("no holdings selected")
        return 1

    out = config.OUT
    previous_path = out / "call_panel.parquet"
    previous = pl.DataFrame()
    if args.baseline == "previous" and previous_path.exists():
        previous = pl.read_parquet(previous_path)

    log.info(
        "building %d periods (%s..%s) for %d firms",
        len(periods),
        periods[0],
        periods[-1],
        len(holdings),
    )

    # Every filer in the window, and the latest name the roster gives each:
    # the lineage uses the first to recognise a depository NIC's attribute
    # files omit, and the second to name it.
    filers: set[int] = set()
    names: dict[int, str] = {}
    for period in periods:
        roster = schedules.roster(cdr.zip_path(period))
        filers.update(roster["rssd"].to_list())
        names.update(zip(roster["rssd"].to_list(), roster["name"].to_list()))

    lineages: dict[str, lineage.Lineage] | None = None
    if not args.no_lineage:
        quarter_ends = [cdr.quarter_end(p) for p in periods]
        lineages = lineage.resolve_all(
            holdings, quarter_ends, floor=cdr.quarter_end(periods[0]).replace(month=1, day=1), filers=filers
        )

    # The thrift backfill needs to know which depositories are missing, and
    # that is a property of a built panel.  So the first pass is built without
    # it, the gap is read off that, and the panel is rebuilt with the history
    # folded in.  Only the cache is read here -- ``fetch_call.py --tfr`` is
    # what goes to the network.
    tfr_frame, tfr_info = None, None
    if args.tfr:
        from callrpt_db import tfr as tfr_mod

        # ``unfiled_depositories`` is where ``insured_not_filing`` comes from,
        # so it is asked directly rather than by building the panel twice to
        # read the column back off it.
        gap = panel.unfiled_depositories(periods, holdings, lineages)
        tfr_frame, tfr_info = tfr_mod.backfill(gap, periods, cached_only=True)
        if tfr_frame.is_empty():
            log.warning("--tfr: nothing cached; run fetch_call.py --tfr first")

    charters, holding_panel = panel.build(
        periods, holdings, lineages=lineages, tfr_frame=tfr_frame
    )
    if holding_panel.is_empty():
        log.error("the build produced no rows")
        return 1
    if args.baseline == "own-2013":
        log.info("rebuilding the 2013 own-subtree baseline for the coverage table")
        previous = baseline_panel(cached, holdings, until)

    holding_panel.write_parquet(out / "call_panel.parquet")
    holding_panel.write_csv(out / "call_panel.csv")
    charters.write_parquet(out / "call_panel_charters.parquet")
    coverage(holding_panel).write_csv(out / "call_panel_coverage.csv")

    delta = coverage_delta(previous, holding_panel)
    delta.write_csv(out / "call_panel_coverage_delta.csv")

    flagged = flags(holding_panel)
    if not flagged.is_empty():
        flagged.write_csv(out / "call_panel_flags.csv")

    lineage_frame = pl.DataFrame()
    lineage_summary: dict = {}
    if lineages is not None:
        contributed: dict[tuple[str, int], int] = {}
        if "via_rssd" in charters.columns:
            counted = (
                charters.filter(pl.col("via_rssd").is_not_null())
                .group_by(["ticker", "via_rssd"])
                .len()
            )
            contributed = {
                (t, int(r)): int(n) for t, r, n in counted.iter_rows()
            }
        own_counts = charters.filter(pl.col("via_rssd").is_null()).group_by(["ticker", "holding_rssd"]).len()
        contributed.update({(t, int(r)): int(n) for t, r, n in own_counts.iter_rows()})
        tracked = {r for h in holdings for r in h.rssds}
        lineage_frame = lineage.to_frame(
            lineages, contributed=contributed, tracked=tracked, names=names
        )
        lineage_frame.write_csv(out / "rssd_lineage.csv")
        kinds = dict(lineage_frame.group_by("succession_type").len().iter_rows())
        lineage_summary = {
            "rows": lineage_frame.height,
            "predecessors": int(lineage_frame.filter(pl.col("succession_type") != lineage.SELF).height),
            "by_type": {k: int(v) for k, v in sorted(kinds.items())},
            "contributing_predecessors": int(
                lineage_frame.filter(
                    (pl.col("succession_type") != lineage.SELF) & (pl.col("charter_quarters") > 0)
                ).height
            ),
            "tracked_separately": int(lineage_frame.filter(pl.col("tracked_separately")).height),
            "acquisitions_by_relationship": int(
                lineage_frame.filter(pl.col("transformation_code") == lineage.ACQUISITION_CODE).height
            ),
        }
        log.info("lineage: %s", lineage_summary)

    log.info(
        "panel: %d bank-quarters, %d firms, %d columns, %d populated cells",
        holding_panel.height,
        holding_panel["ticker"].n_unique(),
        len(holding_panel.columns),
        _filled_cells(holding_panel),
    )
    log.info(
        "charters: %d charter-quarters across %d charters",
        charters.height,
        charters["rssd"].n_unique(),
    )
    # Recorded, not merely logged: the documents quote these, and a figure that
    # exists only in a log line is one nothing can check a document against.
    partition_summary: dict[str, int] | None = None
    if "rcc_residual" in charters.columns:
        exact = int(
            (charters["rcc_residual"].abs() <= PARTITION_TOLERANCE_USD).sum()
        )
        checked = int(charters["rcc_residual"].is_not_null().sum())
        partition_summary = {
            "charter_quarters_checked": checked,
            "charter_quarters_tied": exact,
            # Holding-company rows left null because a TFR-backfilled charter
            # makes the RC-C cross-foot meaningless for that bank-quarter.
            "holding_rows_not_checked": (
                int(holding_panel["rcc_residual"].is_null().sum())
                if "rcc_residual" in holding_panel.columns
                else 0
            ),
        }
        log.info(
            "RC-C partition ties exactly for %d of %d charter-quarters (%.1f%%)",
            exact,
            checked,
            exact / checked * 100 if checked else 0.0,
        )
    if not delta.is_empty():
        log.info(
            "coverage: %d bank-quarters before, %d after (%+d); most added: %s",
            int(delta["quarters_before"].sum()),
            int(delta["quarters_after"].sum()),
            int(delta["quarters_added"].sum()),
            ", ".join(
                f"{t} +{n}" for t, n in delta.select("ticker", "quarters_added").head(8).iter_rows()
            ),
        )

    diff_rows = 0
    if not args.no_crosscheck:
        edgar_path = out / "panel.parquet"
        if not edgar_path.exists():
            log.warning("no EDGAR panel at %s; skipping the cross-check", edgar_path)
        else:
            edgar = pl.read_parquet(edgar_path)
            diff = crosscheck.compare(edgar, holding_panel)
            if diff.is_empty():
                log.warning("the two panels share no bank-quarters")
            else:
                diff.write_csv(out / "source_diff.csv")
                summary = crosscheck.summarise(diff)
                summary.write_csv(out / "source_diff_summary.csv")
                crosscheck.coverage_gained(edgar, holding_panel).write_csv(
                    out / "source_coverage.csv"
                )
                diff_rows = diff.height
                verdicts = dict(
                    diff.group_by("verdict").len().iter_rows()  # type: ignore[arg-type]
                )
                log.info(
                    "cross-check: %d comparisons, %d agree, %d differ, %d one-source",
                    diff.height,
                    verdicts.get("agree", 0),
                    verdicts.get("differ", 0),
                    verdicts.get("one_source_only", 0),
                )
                shaky = crosscheck.unstable(summary)
                if not shaky.is_empty():
                    log.info(
                        "%d bank-variable pairs have an unstable ratio; see "
                        "source_diff_summary.csv",
                        shaky.height,
                    )

    flag_counts = (
        {k: int(v) for k, v in sorted(dict(flagged.group_by("flag").len().iter_rows()).items())}
        if not flagged.is_empty()
        else {}
    )
    info = {
        "built_at": started.isoformat(),
        "baseline": args.baseline,
        "commit": _git_commit(),
        "source": "FFIEC CDR bulk Call Report data, FFIEC NIC structure data (attributes, relationships, transformations)",
        "since": args.since.isoformat(),
        "until": until.isoformat(),
        "periods": len(periods),
        "first_period": periods[0],
        "last_period": periods[-1],
        "firms": holding_panel["ticker"].n_unique(),
        "bank_quarters": holding_panel.height,
        "charter_quarters": charters.height,
        "columns": len(holding_panel.columns),
        "populated_cells": _filled_cells(holding_panel),
        "flags": 0 if flagged.is_empty() else flagged.height,
        "flag_counts": flag_counts,
        # Which build this is.  The thrift backfill changes the row counts and
        # costs the holding-level partition check, so a figure quoted from a
        # build is only meaningful beside the flag that produced it.
        "tfr": bool(args.tfr),
        "partition": partition_summary,
        "crosscheck_rows": diff_rows,
        "unit_scale": mdrm.UNIT_SCALE,
        "comparators": bool(args.comparators),
        "tickers": args.tickers or None,
        "lineage": lineage_summary if lineages is not None else "disabled (--no-lineage)",
        # The thrift gap, and what closing it took.  Recorded whether or not
        # ``--tfr`` ran, because "this panel does not carry the thrifts" is
        # the finding a reader of the 2001-2011 rows most needs to know.
        "thrift_gap": tfr_info
        if tfr_info is not None
        else {
            "status": "not attempted (--tfr not given)",
            "effect": (
                "insured depositories that filed a Thrift Financial Report "
                "contribute nothing; see n_insured_not_filing per bank-quarter"
            ),
        },
        "coverage_delta": {
            "bank_quarters_before": int(delta["quarters_before"].sum()) if not delta.is_empty() else None,
            "bank_quarters_after": int(delta["quarters_after"].sum()) if not delta.is_empty() else None,
            "bank_quarters_with_predecessors": int(delta["quarters_with_predecessors"].sum()) if not delta.is_empty() else None,
            "bank_quarters_with_failed_predecessor": int(delta["quarters_with_failed_predecessor"].sum()) if not delta.is_empty() else None,
            "top_gainers": delta.select("ticker", "quarters_added", "first_quarter_after").head(10).to_dicts() if not delta.is_empty() else [],
            "table": "call_panel_coverage_delta.csv",
        },
        "schedule_breaks": SCHEDULE_BREAKS,
        "variables_added": VARIABLES_ADDED,
        "corrections_to_previous_build": CORRECTIONS,
        "not_strictly_comparable": NOT_STRICTLY_COMPARABLE,
        "gains_value_from_2007_2009": GFC_VALUE,
    }
    (out / "call_panel_build_info.json").write_text(
        json.dumps(info, indent=2, default=str), encoding="utf-8"
    )
    log.info("wrote %s", out / "call_panel.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
