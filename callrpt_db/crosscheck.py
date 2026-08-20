"""Compare the FFIEC panel against the EDGAR one, bank-quarter by bank-quarter.

The point of a second source is disagreement
--------------------------------------------
Two independent readings of the same bank should mostly agree, and where they
do not, one of three things is true and it is worth knowing which:

**The entity differs.**  This is the big one and it is structural, not an
error.  ``bankqtr_db`` reads the SEC filings of a *holding company*; this
package reads the Call Reports of the *banks* it owns.  The broker-dealer, the
credit-card funding trusts, the insurance arms and the non-bank lenders file no
Call Report, so a holding company's loan book is normally larger than the sum
of its charters -- except where intercompany lending runs the other way, since
a loan from the bank to its own broker-dealer is an asset on the bank's Call
Report and eliminated in the holding company's consolidation.  Both directions
appear in this universe.

**The definition differs.**  The Call Report's categories are fixed lines on a
form; the 10-K's are whatever the bank chose to disclose.  "CRE" in Schedule
RC-C is construction, multifamily and nonfarm nonresidential, full stop, while
a 10-K may exclude owner-occupied, include or exclude construction, or not
break it out at all.

**One of them is wrong.**  Which is what this report exists to surface.  A gap
that is small and stable across quarters is structure; a gap that jumps in one
quarter and returns is a bug in one of the two paths.

The output carries the ratio, not just the difference, because the useful
question is proportional: State Street's banks holding 60% of the holding
company's loans is a fact about State Street, and the same figure jumping to
6% for one quarter is a fact about the pipeline.
"""

from __future__ import annotations

import logging

import polars as pl

log = logging.getLogger(__name__)

# Columns worth comparing: the ones both sources genuinely carry.  Ratios are
# compared as well as levels, because a ratio can agree while both its inputs
# differ -- which is the signature of an entity-scope difference rather than a
# reading error, and is exactly what a user wants to know before choosing a
# source for a peer comparison.
COMPARE_LEVELS: tuple[str, ...] = (
    "loans_total",
    "loans_cre_total",
    "loans_ci",
    "loans_construction",
    "loans_multifamily",
    "loans_credit_card",
    "loans_auto",
    "loans_resi_mortgage",
    "acl_total",
    "nonaccrual_total",
    "assets",
    "equity",
    "deposits",
    "nco_total",
    "provision_total",
    "net_income",
)

COMPARE_RATIOS: tuple[str, ...] = (
    "cre_pct",
    "ci_pct",
    "construction_pct",
    "credit_card_pct",
    "reserve_coverage",
    "nonaccrual_ratio",
    "nco_rate",
)

# A relative gap under this is treated as agreement.  Not zero: the two paths
# round differently -- the Call Report is filed in thousands, XBRL in units --
# so a $500 difference on a $500bn book is not a disagreement about anything.
AGREEMENT_TOLERANCE_PCT = 0.5


def compare(
    edgar: pl.DataFrame,
    ffiec: pl.DataFrame,
    *,
    columns: tuple[str, ...] = COMPARE_LEVELS,
    ratios: tuple[str, ...] = COMPARE_RATIOS,
) -> pl.DataFrame:
    """One row per bank-quarter-variable, with both readings and the gap.

    Joined on ``ticker`` and ``period``.  The IHCs have no EDGAR counterpart
    and simply do not appear; that absence is the whole point of the FFIEC
    build and is reported separately by :func:`coverage_gained`.
    """
    if edgar.is_empty() or ffiec.is_empty():
        return pl.DataFrame()

    wanted = [c for c in (*columns, *ratios) if c in edgar.columns and c in ffiec.columns]
    if not wanted:
        return pl.DataFrame()

    left = edgar.select(["ticker", "period", *wanted])
    right = ffiec.select(["ticker", "period", *wanted])
    joined = left.join(right, on=["ticker", "period"], how="inner", suffix="__ffiec")
    if joined.is_empty():
        return pl.DataFrame()

    frames: list[pl.DataFrame] = []
    ratio_set = set(ratios)
    for column in wanted:
        other = f"{column}__ffiec"
        piece = joined.select(
            pl.col("ticker"),
            pl.col("period"),
            pl.lit(column).alias("variable"),
            pl.lit("ratio" if column in ratio_set else "level").alias("kind"),
            pl.col(column).alias("edgar"),
            pl.col(other).alias("ffiec"),
        ).with_columns(
            (pl.col("ffiec") - pl.col("edgar")).alias("difference"),
            pl.when(pl.col("edgar").abs() > 0)
            .then((pl.col("ffiec") - pl.col("edgar")) / pl.col("edgar").abs() * 100)
            .otherwise(None)
            .alias("difference_pct"),
            pl.when(pl.col("edgar").abs() > 0)
            .then(pl.col("ffiec") / pl.col("edgar"))
            .otherwise(None)
            .alias("ffiec_over_edgar"),
        )
        frames.append(piece)

    out = pl.concat(frames, how="vertical_relaxed")
    out = out.with_columns(
        pl.when(pl.col("edgar").is_null() | pl.col("ffiec").is_null())
        .then(pl.lit("one_source_only"))
        .when(pl.col("difference_pct").abs() <= AGREEMENT_TOLERANCE_PCT)
        .then(pl.lit("agree"))
        .otherwise(pl.lit("differ"))
        .alias("verdict")
    )
    return out.sort(["ticker", "period", "variable"])


def summarise(diff: pl.DataFrame) -> pl.DataFrame:
    """Per bank and variable: how often the two agree, and by how much.

    ``median_ratio`` is the headline.  A stable ratio well below 1 is a bank
    whose lending mostly happens outside its charters; a ratio near 1 with a
    high ``pct_agree`` is a variable both sources can be trusted on.
    """
    if diff.is_empty():
        return pl.DataFrame()
    return (
        diff.group_by(["ticker", "variable", "kind"])
        .agg(
            pl.len().alias("quarters"),
            (pl.col("verdict") == "agree").sum().alias("n_agree"),
            (pl.col("verdict") == "differ").sum().alias("n_differ"),
            (pl.col("verdict") == "one_source_only").sum().alias("n_one_source"),
            pl.col("ffiec_over_edgar").median().alias("median_ratio"),
            pl.col("ffiec_over_edgar").std().alias("ratio_std"),
            pl.col("difference_pct").median().alias("median_diff_pct"),
        )
        .with_columns(
            pl.when(pl.col("quarters") > 0)
            .then(pl.col("n_agree") / pl.col("quarters") * 100)
            .otherwise(None)
            .alias("pct_agree")
        )
        .sort(["variable", "ticker"])
    )


def unstable(summary: pl.DataFrame, *, threshold: float = 0.15) -> pl.DataFrame:
    """Bank-variable pairs whose ratio moves around, which is where bugs live.

    A structural entity difference is *stable*: the same charters against the
    same holding company quarter after quarter.  A ratio whose standard
    deviation is a large fraction of its median is not explained by structure,
    and is worth reading the filings over.
    """
    if summary.is_empty():
        return pl.DataFrame()
    return (
        summary.filter(
            pl.col("median_ratio").is_not_null()
            & (pl.col("median_ratio").abs() > 0)
            & (pl.col("ratio_std") / pl.col("median_ratio").abs() > threshold)
        )
        .with_columns(
            (pl.col("ratio_std") / pl.col("median_ratio").abs()).alias("instability")
        )
        .sort("instability", descending=True)
    )


def coverage_gained(edgar: pl.DataFrame, ffiec: pl.DataFrame) -> pl.DataFrame:
    """What the FFIEC panel covers that the EDGAR one does not.

    The reason this package exists: seven of the eight US intermediate holding
    companies of foreign banks file no 10-K, so they cannot appear in an
    EDGAR-sourced panel at any level of effort.  They do file Call Reports.
    """
    if ffiec.is_empty():
        return pl.DataFrame()
    known = set(edgar["ticker"].unique().to_list()) if not edgar.is_empty() else set()
    return (
        ffiec.group_by("ticker")
        .agg(
            pl.col("bank").first().alias("bank"),
            pl.len().alias("quarters"),
            pl.col("period").min().alias("first_quarter"),
            pl.col("period").max().alias("last_quarter"),
            pl.col("loans_total").last().alias("loans_total_latest"),
            pl.col("n_charters").max().alias("max_charters"),
        )
        .with_columns(
            pl.col("ticker").is_in(list(known)).alias("in_edgar_panel"),
        )
        .sort(["in_edgar_panel", "loans_total_latest"], descending=[False, True])
    )
