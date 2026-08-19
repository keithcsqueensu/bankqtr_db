"""Build the bank-quarter panel from the long fact frame.

Three problems have to be solved between "facts" and "panel":

**Selection.**  A bank tags the same balance on several different cuts at
once -- portfolio segment, loan class, industry, vintage, and combinations of
those.  Each cut sums to the same total, so treating "the CRE rows" as one
population roughly doubles the answer.  Every summed column therefore fixes a
single *dimension signature* (the exact set of axes, not merely how many) per
bank-quarter before aggregating.

**Periodisation.**  Flow items are tagged year-to-date as well as
quarter-to-date, and a 10-K usually carries only the annual figure, so Q4 has
to be derived.  Fiscal years are not all calendar years (Raymond James ends in
September), so periods are grouped by the fiscal-year start each fact declares
rather than by calendar year.

**Restatement.**  The same quarter is re-reported as a comparative in later
filings, sometimes with different numbers.  The latest filed version wins and
the disagreement is flagged rather than hidden.
"""

from __future__ import annotations

import datetime as dt
import logging

import polars as pl

from . import taxonomy, variables
from .config import cecl_adoption
from .variables import ALL_VARS, RATIOS, VarDef

log = logging.getLogger(__name__)

# Day-count windows used to classify a duration fact.
QTD_RANGE = (80, 100)
YTD_RANGES = {
    "H1": (170, 195),
    "9M": (260, 285),
    "FY": (350, 380),
}


# --------------------------------------------------------------------------
# Period helpers
# --------------------------------------------------------------------------


def add_period_labels(df: pl.DataFrame) -> pl.DataFrame:
    """Attach calendar quarter labels and duration spans."""
    return df.with_columns(
        pl.col("period").dt.year().alias("cal_year"),
        pl.col("period").dt.quarter().alias("cal_quarter"),
        (
            pl.col("period").dt.year().cast(pl.String)
            + pl.lit("Q")
            + pl.col("period").dt.quarter().cast(pl.String)
        ).alias("quarter"),
        (pl.col("period") - pl.col("period_start")).dt.total_days().alias("span_days"),
    )


def _is_qtd(span: pl.Expr) -> pl.Expr:
    return span.is_between(*QTD_RANGE)


# --------------------------------------------------------------------------
# Flow quarterisation
# --------------------------------------------------------------------------

FLOW_KEY = ("ticker", "tag", "loan_cat", "cq_cat", "pd_cat", "scope")


def quarterize(df: pl.DataFrame) -> pl.DataFrame:
    """Reduce duration facts to one quarter-to-date observation per period.

    Grouping is by the fiscal-year start each fact declares, not by span
    length.  Several banks -- JPMorgan among them -- tag *only* cumulative
    year-to-date charge-offs, so the quarterly series exists nowhere in the
    filing and has to be recovered by differencing consecutive cumulative
    points.  The Q1 fact is simultaneously the first cumulative point and a
    quarterly one, which makes it the anchor for the whole year; excluding it
    from the cumulative series (because its span looks quarterly) silently
    destroys Q2 for every such bank.
    """
    flows = df.filter(~pl.col("instant") & pl.col("period_start").is_not_null())
    if flows.is_empty():
        return flows.with_columns(pl.lit("none", dtype=pl.String).alias("basis"))

    flows = add_period_labels(flows)
    keys = [k for k in FLOW_KEY if k in flows.columns]

    native = flows.filter(_is_qtd(pl.col("span_days"))).with_columns(
        pl.lit("reported", dtype=pl.String).alias("basis")
    )

    # One cumulative series per (key, fiscal-year start), including the Q1
    # anchor. Duplicate reports of the same point collapse to the latest.
    cumulative = flows.sort(["filed"], descending=True, nulls_last=True).unique(
        subset=[*keys, "period_start", "period"], keep="first", maintain_order=True
    )

    derived_frames: list[pl.DataFrame] = []
    if not cumulative.is_empty():
        group = [*keys, "period_start"]
        diffed = (
            cumulative.sort([*group, "period"])
            .with_columns(
                (pl.col("value") - pl.col("value").shift(1).over(group)).alias("_diff"),
                pl.col("period").shift(1).over(group).alias("_prev_end"),
                pl.col("span_days").shift(1).over(group).alias("_prev_span"),
            )
            # First cumulative point in a fiscal year IS the first quarter.
            .with_columns(
                pl.when(pl.col("_prev_end").is_null())
                .then(pl.col("value"))
                .otherwise(pl.col("_diff"))
                .alias("qtd_value"),
                pl.when(pl.col("_prev_end").is_null())
                .then(pl.col("span_days"))
                .otherwise(pl.col("span_days") - pl.col("_prev_span"))
                .alias("implied_span"),
            )
            .filter(pl.col("qtd_value").is_not_null())
            # Only keep differences that actually span one quarter.
            .filter(pl.col("implied_span").is_between(80, 100))
        )
        if not diffed.is_empty():
            derived_frames.append(
                diffed.drop("_diff", "_prev_end", "_prev_span", "implied_span")
                .with_columns(
                    pl.col("qtd_value").alias("value"),
                    pl.lit("derived", dtype=pl.String).alias("basis"),
                )
                .drop("qtd_value")
            )

    parts = [p for p in (native, *derived_frames) if not p.is_empty()]
    if not parts:
        return native.with_columns(pl.lit("none", dtype=pl.String).alias("basis"))

    combined = pl.concat(parts, how="vertical_relaxed")
    # Reported beats derived when both exist for the same quarter.
    return (
        combined.with_columns(
            pl.when(pl.col("basis") == "reported").then(0).otherwise(1).alias("_pref")
        )
        .sort(["_pref", "filed"], descending=[False, True], nulls_last=True)
        .unique(subset=[*keys, "period"], keep="first", maintain_order=True)
        .drop("_pref")
    )


# --------------------------------------------------------------------------
# Dimensional selection
# --------------------------------------------------------------------------


MEMBER_COLS = ("segment", "loan_class", "credit_quality", "past_due", "other_dims")


def _tag_priority(df: pl.DataFrame, tags: tuple[str, ...]) -> pl.DataFrame:
    """Rank rows by how preferred their tag is."""
    order = {t: i for i, t in enumerate(tags)}
    return df.with_columns(
        pl.col("tag").replace_strict(order, default=len(order)).alias("_tag_rank")
    )


def _n_axes(col: str = "dim_axes") -> pl.Expr:
    return (
        pl.when(pl.col(col) == "")
        .then(0)
        .otherwise(pl.col(col).str.count_matches(r"\|") + 1)
    )


# Axes that turn a balance into a *sub-slice* of itself.  A plain loan balance
# should never be read off the delinquency or risk-rating table.
# Matched as substrings of the signature, so a short entry covers the longer
# spellings: "CreditQualityIndicatorAxis" also catches the FinancingReceivable-
# and FinancingReceivableRecordedInvestmentBy- prefixed variants.
SUBSLICE_AXES = (
    "FinancingReceivablesPeriodPastDueAxis",
    "InternalCreditAssessmentAxis",
    "CreditQualityIndicatorAxis",
    "CreditScoreFicoAxis",
    # --- pre-2018 spellings of the same two tables -----------------------
    "DelinquencyStatusAxis",
    "AgingAnalysisOfLoansAndLeasesAxis",
)


# Their pre-2018 aliases (see ``instance.PROMOTED_AXES``) are a *fallback*, in
# exactly the sense the legacy tag names are: used where a filing carries
# nothing else, never in preference to a cut the modern taxonomy also
# describes.
#
# Without this, a 2021 filing that still carries the old receivable-type axis
# alongside the current class axis gets read off the old one, and the old one
# is often not a partition -- Wells Fargo tags residential mortgage there as
# the parent *and* its first-lien child, so summing the members counts the
# first lien twice and puts residential at 57% of a book where it is 29%.
MODERN_LOAN_AXES = taxonomy.MODERN_LOAN_AXES


def _is_legacy_signature() -> pl.Expr:
    """1 when a signature names no modern loan axis at all."""
    return (
        (
            pl.col("dim_axes")
            .str.split("|")
            .list.set_intersection(list(MODERN_LOAN_AXES))
            .list.len()
            == 0
        )
        .cast(pl.Int32)
        .alias("_legacy")
    )


def _restrict_to_best_signature(
    df: pl.DataFrame,
    key: list[str],
    *,
    allow_subslice: bool = False,
    prefer: str = "parsimony",
    prefer_modern_axes: bool = True,
) -> pl.DataFrame:
    """Keep one tag and one dimension signature per key.

    Banks tag the same balance on several different cuts simultaneously --
    portfolio segment, loan class, industry, vintage -- and each cut sums to
    the same total, so mixing two of them roughly doubles the answer.

    Candidate signatures are ranked by, in order: tag preference; whether the
    signature slices the balance by delinquency or risk rating (a plain
    balance should not be read off those tables); whether it is built purely
    from pre-2018 alias axes; how many axes it carries; and finally the
    magnitude of what it covers.  That last term matters --
    banks publish narrow supplementary tables (fair-value-option loans,
    purchased credit-deteriorated, a single geography) on the same axes as the
    main portfolio table, and an alphabetical tie-break lands on them roughly
    at random.  Bank of America's CRE balance comes out as zero that way.
    """
    if df.is_empty():
        return df

    subslice = pl.any_horizontal(
        [pl.col("dim_axes").str.contains(axis, literal=True) for axis in SUBSLICE_AXES]
    )
    ranked = df.with_columns(
        _n_axes().alias("_n_axes"),
        pl.lit(0).alias("_sub")
        if allow_subslice
        else subslice.cast(pl.Int32).alias("_sub"),
        _is_legacy_signature() if prefer_modern_axes else pl.lit(0).alias("_legacy"),
    )
    coverage = ranked.group_by([*key, "tag", "dim_axes"]).agg(
        pl.col("value").abs().sum().alias("_cover")
    )
    # prefer="coverage" takes the most complete partition instead of the
    # shallowest one. Totals need that: Wells Fargo tags a single member at
    # portfolio-segment level (consumer, $375bn) while the complete book only
    # appears one level deeper at segment x class ($909bn), so preferring
    # parsimony understates its loan book by two-thirds.
    order = (
        ["_tag_rank", "_sub", "_legacy", "_cover", "_n_axes", "dim_axes"]
        if prefer == "coverage"
        else ["_tag_rank", "_sub", "_legacy", "_n_axes", "_cover", "dim_axes"]
    )
    desc = (
        [False, False, False, True, False, False]
        if prefer == "coverage"
        else [False, False, False, False, True, False]
    )
    # sort_by inside the aggregation rather than a preceding .sort(): a
    # group_by does not carry the frame's row order into first(), so ranking
    # has to be expressed where the pick actually happens.
    pick = (
        ranked.join(coverage, on=[*key, "tag", "dim_axes"], how="left")
        .group_by(key)
        .agg(
            pl.col("tag").sort_by(order, descending=desc).first().alias("_best_tag"),
            pl.col("dim_axes")
            .sort_by(order, descending=desc)
            .first()
            .alias("_best_sig"),
        )
    )
    return (
        ranked.join(pick, on=key, how="inner")
        .filter(
            (pl.col("tag") == pl.col("_best_tag"))
            & (pl.col("dim_axes") == pl.col("_best_sig"))
        )
        .drop("_best_tag", "_best_sig", "_n_axes", "_sub", "_legacy")
    )


def _sum_members(df: pl.DataFrame, key: list[str], out_name: str) -> pl.DataFrame:
    """Sum distinct members within one signature, latest filing per member.

    Members inside one signature are supposed to partition the parent balance,
    but banks sometimes tag the *same* balance under two member names at once
    -- JPMorgan reports credit-card loans as both ``CreditCardReceivables`` and
    ``CreditCardLoanPortfolioSegment``, so a naive sum doubles the card book.
    Identical values within a category are therefore collapsed: two genuinely
    distinct sub-portfolios agreeing to the dollar is not a real scenario,
    whereas alias tagging demonstrably is.
    """
    member_cols = [c for c in MEMBER_COLS if c in df.columns]
    return (
        df.sort(["filed"], descending=True, nulls_last=True)
        .unique(subset=[*key, *member_cols], keep="first", maintain_order=True)
        .unique(subset=[*key, "value"], keep="first", maintain_order=True)
        .group_by(key)
        .agg(pl.col("value").sum().alias(out_name))
    )


def select_slice(
    df: pl.DataFrame,
    var: VarDef,
    *,
    axis_col: str,
    category: str,
    key: list[str],
    out_name: str,
) -> pl.DataFrame:
    """One value per key for a variable sliced to a single canonical category."""
    base = df.filter(
        pl.col("tag").is_in(list(var.tags))
        & (pl.col(axis_col) == category)
        & pl.col("scope").is_null()
        & (pl.col("unit") == var.unit)
    )
    if base.is_empty():
        return base.select([*key, pl.col("value").alias(out_name)])
    chosen = _restrict_to_best_signature(_tag_priority(base, var.tags), key)
    return _sum_members(chosen, key, out_name)


# How far below the largest reading of the same concept a candidate may sit
# before it is treated as a fragment rather than an alternative measure.
# Deliberately an order of magnitude: two genuine alternatives -- allowance on
# loans versus allowance including unfunded commitments, loans held for
# investment versus including held-for-sale -- differ by a few per cent.
FRAGMENT_RATIO = 10.0


def _drop_fragments(df: pl.DataFrame, key: list[str]) -> pl.DataFrame:
    """Discard undimensioned candidates that are a rounding error beside another.

    Tag priority assumes every candidate element measures the same concept, so
    the modern name wins and a legacy name is reached only where nothing else
    is there.  That assumption breaks where a filer used a modern-*named*
    element for something far narrower.  Citigroup's FY2013 instance tags
    ``FinancingReceivableAllowanceForCreditLosses`` -- first in ``ACL_TAGS`` --
    at $113m, while its actual $19.6bn allowance sits undimensioned on
    ``LoansAndLeasesReceivableAllowance``, last in the tuple.  Priority alone
    picks the $113m and reports a 0.02% reserve ratio for Citigroup.

    Reordering the tuple is not the fix -- that would rewrite the modern
    window -- and neither is rebuilding the total from its parts, which at
    Citigroup double-counts to $39bn.  The bank reported the right number; it
    is simply not the highest-ranked tag.  So the smallest candidates are
    dropped before ranking, and the choice among what remains is unchanged.
    """
    if df.height < 2:
        return df
    biggest = pl.col("value").abs().max().over(key)
    return df.filter(
        (biggest <= 0) | (pl.col("value").abs() * FRAGMENT_RATIO >= biggest)
    )


def _pick_best(df: pl.DataFrame, var: VarDef, key: list[str]) -> pl.DataFrame:
    """One row per key: best tag, then most recently filed.

    Loan-sliced *stocks* have several tag spellings that are all meant to be
    the same balance, so a candidate orders of magnitude below its peers is a
    mis-tagged fragment.  Nothing else has that property, and the guard is
    limited accordingly:

    * ``unfunded_commitments`` ranks a small reserve element above a large
      notional one on purpose, so magnitude must not decide there.
    * a *flow* is legitimately small, zero, or negative in a given quarter --
      Raymond James reports a nil provision beside a $10m release -- so its
      magnitude says nothing about which tag is the right one.
    """
    if df.is_empty():
        return df
    if var.kind == "stock" and var.axis == "loan":
        df = _drop_fragments(df, key)
    return (
        _tag_priority(df, var.tags)
        .sort(["_tag_rank", "filed"], descending=[False, True], nulls_last=True)
        .unique(subset=key, keep="first", maintain_order=True)
        .drop("_tag_rank")
    )


# How close a rollup must sit to the sum of its own disclosed parts before it
# is read as containing them rather than standing beside them.
#
# Tight on purpose.  Containment is an arithmetic identity taken from one
# table, so it holds to the precision the filing reports in -- Wells Fargo's
# consumer segment matches its five classes to the dollar.  A loose threshold
# catches coincidences instead: Raymond James tags six sibling segments, and
# three of them happen to sum to within 0.97% of a fourth, which at one per
# cent silently deleted a tenth of its loan book.  Filings report in millions,
# so rounding across a handful of components cannot reach a thousandth.
ROLLUP_TOLERANCE = 0.001


def _sum_partition(df: pl.DataFrame, key: list[str], out_name: str) -> pl.DataFrame:
    """Sum one signature's members, dropping any rollup that contains others.

    A signature is one disclosure table, and its members usually partition the
    balance -- but not always at one level.  Wells Fargo's 2022Q3 receivable-
    type table carries the consumer *segment* beside the five consumer classes
    underneath it, and the five come to 395.94bn, which is the segment to the
    dollar.  Summed flat that is 1.33tn of loans against a 931bn book, and
    ``prefer="coverage"`` actively selects such a signature because it is the
    largest.

    Which of the two a member is cannot be read off the category tree alone:
    Wells Fargo tags commercial real estate mortgage as ``RealEstateLoanMember``
    and construction as its sibling, and the tree calls construction a child of
    cre_total, so dropping every descendant of a present ancestor would lose
    the construction book.  The arithmetic settles it instead -- a category is
    dropped only where its own value equals the sum of the categories beneath
    it that this table actually discloses.
    """
    member_cols = [c for c in MEMBER_COLS if c in df.columns]
    per_cat = (
        df.sort(["filed"], descending=True, nulls_last=True)
        .unique(subset=[*key, *member_cols], keep="first", maintain_order=True)
        .unique(subset=[*key, "value"], keep="first", maintain_order=True)
        .group_by([*key, "loan_cat"])
        .agg(pl.col("value").sum())
    )
    wide = per_cat.pivot(on="loan_cat", index=key, values="value")
    cats = [c for c in wide.columns if c not in key]

    terms: list[pl.Expr] = []
    for cat in cats:
        below = [d for d in taxonomy.descendants(cat) if d in cats]
        value = pl.col(cat).fill_null(0.0)
        if not below:
            terms.append(value)
            continue
        parts = pl.sum_horizontal([pl.col(d).fill_null(0.0) for d in below])
        contains = (parts.abs() > 0) & (
            (value - parts).abs() <= value.abs() * ROLLUP_TOLERANCE
        )
        terms.append(pl.when(contains).then(0.0).otherwise(value))

    return wide.select([*key, pl.sum_horizontal(terms).alias(out_name)])


def partition_total(
    df: pl.DataFrame, var: VarDef, *, key: list[str], out_name: str
) -> pl.DataFrame:
    """Total built by summing the members of one dimension signature.

    Not every bank tags a consolidated, undimensioned figure -- Wells Fargo
    reports loans only as a breakdown -- so the total has to be rebuilt from
    the parts.  Within a single signature the members are the rows of one
    disclosure table; :func:`_sum_partition` handles the case where that table
    states a rollup and its own components together.
    """
    base = df.filter(
        pl.col("tag").is_in(list(var.tags))
        & pl.col("loan_cat").is_not_null()
        & pl.col("scope").is_null()
        & (pl.col("unit") == var.unit)
    )
    if base.is_empty():
        return base.select([*key, pl.col("value").alias(out_name)])
    chosen = _restrict_to_best_signature(
        _tag_priority(base, var.tags), key, prefer="coverage"
    )
    return _sum_partition(chosen, key, out_name)


def _with_partition_fallback(
    out: pl.DataFrame | None,
    source: pl.DataFrame,
    var: VarDef,
    colname: str,
) -> pl.DataFrame | None:
    """Fill a total column from the partition sum wherever it is still null."""
    derived = partition_total(
        source, var, key=PANEL_KEY, out_name=f"{colname}__derived"
    )
    if derived.is_empty():
        return out
    if out is None or colname not in out.columns:
        return _merge(out, derived.rename({f"{colname}__derived": colname}))

    # Full join, not left: a bank that tags no consolidated figure anywhere has
    # no row yet, and a left join would drop its derived total on the floor.
    merged = out.join(derived, on=PANEL_KEY, how="full", coalesce=True)
    return merged.with_columns(
        pl.col(colname).fill_null(pl.col(f"{colname}__derived")).alias(colname)
    ).drop(f"{colname}__derived")


# --------------------------------------------------------------------------
# Panel assembly
# --------------------------------------------------------------------------

PANEL_KEY = ["bank", "ticker", "cik", "quarter", "period"]


def _stock_columns(facts: pl.DataFrame, categories: tuple[str, ...]) -> pl.DataFrame:
    """Wide stock variables: totals plus one column per loan category."""
    stocks = add_period_labels(facts.filter(pl.col("instant")))
    out: pl.DataFrame | None = None

    for var in ALL_VARS:
        if var.kind != "stock":
            continue

        # --- consolidated total (undimensioned) ----------------------------
        total = stocks.filter(
            pl.col("tag").is_in(list(var.tags))
            & (pl.col("n_dims") == 0)
            & (pl.col("unit") == var.unit)
        )
        colname = f"{var.name}_total" if var.axis else var.name
        total = _pick_best(total, var, PANEL_KEY)
        if not total.is_empty():
            out = _merge(
                out, total.select([*PANEL_KEY, pl.col("value").alias(colname)])
            )

        if var.axis != "loan":
            continue

        # Banks that never tag a consolidated figure still disclose the parts.
        out = _with_partition_fallback(out, stocks, var, colname)

        # --- one column per loan category ----------------------------------
        # Each category picks the signature that discloses it best.  Forcing a
        # single table for all of them was tried and is measurably worse: the
        # widest-coverage table usually names only rollup members (commercial,
        # consumer), which drops CRE and C&I entirely.  Coherence is measured
        # by mix_coverage_pct instead of imposed here.
        for cat in categories:
            agg = select_slice(
                stocks,
                var,
                axis_col="loan_cat",
                category=cat,
                key=PANEL_KEY,
                out_name=f"{var.name}_{cat}",
            )
            if agg.is_empty():
                continue
            out = _merge(out, agg)

    return (
        out
        if out is not None
        else pl.DataFrame(schema={k: pl.String for k in PANEL_KEY})
    )


def _flow_columns(facts: pl.DataFrame, categories: tuple[str, ...]) -> pl.DataFrame:
    """Wide flow variables, quarterised."""
    flows = quarterize(facts)
    if flows.is_empty():
        return pl.DataFrame(schema={k: pl.String for k in PANEL_KEY})

    out: pl.DataFrame | None = None
    for var in ALL_VARS:
        if var.kind != "flow":
            continue

        total = flows.filter(
            pl.col("tag").is_in(list(var.tags))
            & (pl.col("n_dims") == 0)
            & (pl.col("unit") == var.unit)
        )
        colname = f"{var.name}_total" if var.axis else var.name
        total = _pick_best(total, var, PANEL_KEY)
        if not total.is_empty():
            out = _merge(
                out, total.select([*PANEL_KEY, pl.col("value").alias(colname)])
            )

        if var.axis != "loan":
            continue
        out = _with_partition_fallback(out, flows, var, colname)
        for cat in categories:
            agg = select_slice(
                flows,
                var,
                axis_col="loan_cat",
                category=cat,
                key=PANEL_KEY,
                out_name=f"{var.name}_{cat}",
            )
            if agg.is_empty():
                continue
            out = _merge(out, agg)

    return (
        out
        if out is not None
        else pl.DataFrame(schema={k: pl.String for k in PANEL_KEY})
    )


def _credit_quality_columns(facts: pl.DataFrame) -> pl.DataFrame:
    """Criticized / classified / rating-distribution columns.

    Most banks tag the risk-rating table at ``class x rating`` rather than at
    ``rating`` alone, so the consolidated figure usually has to be summed over
    classes.  Doing that safely means using exactly one nesting level per bank
    and quarter -- mixing levels double-counts, which is precisely how a
    criticized ratio ends up implausibly above 100%.
    """
    cq = add_period_labels(
        facts.filter(
            pl.col("instant")
            & pl.col("cq_cat").is_not_null()
            & pl.col("scope").is_null()
            & pl.col("tag").is_in(list(variables.LOAN_TAGS))
            & (pl.col("unit") == "USD")
        )
    )
    if cq.is_empty():
        return pl.DataFrame(schema={k: pl.String for k in PANEL_KEY})

    # One signature for the whole rating table, so the buckets stay mutually
    # exclusive and comparable to each other.
    cq = _restrict_to_best_signature(
        _tag_priority(cq, variables.LOAN_TAGS),
        PANEL_KEY,
        allow_subslice=True,
        prefer_modern_axes=False,
    )

    member_cols = [c for c in MEMBER_COLS if c in cq.columns]
    agg = (
        cq.sort(["filed"], descending=True, nulls_last=True)
        .unique(subset=[*PANEL_KEY, *member_cols], keep="first", maintain_order=True)
        .group_by([*PANEL_KEY, "cq_cat"])
        .agg(pl.col("value").sum())
        .pivot(on="cq_cat", index=PANEL_KEY, values="value")
    )
    agg = agg.rename({c: f"cq_{c}" for c in agg.columns if c not in PANEL_KEY})

    # Roll the regulatory ladder up: classified and criticized are usually
    # not tagged directly, only their components.
    have = set(agg.columns)

    def _sum(cols: list[str]) -> pl.Expr | None:
        present = [c for c in cols if c in have]
        if not present:
            return None
        return pl.sum_horizontal([pl.col(c).fill_null(0.0) for c in present])

    classified = _sum(["cq_substandard", "cq_doubtful", "cq_loss"])
    if classified is not None and "cq_classified" not in have:
        agg = agg.with_columns(classified.alias("cq_classified"))
        have.add("cq_classified")

    criticized = _sum(["cq_special_mention", "cq_classified"])
    if criticized is not None and "cq_criticized" not in have:
        agg = agg.with_columns(criticized.alias("cq_criticized"))
        have.add("cq_criticized")

    rated = _sum(
        [
            c
            for c in have
            if c.startswith("cq_")
            and taxonomy.CREDIT_QUALITY_FAMILY.get(c[3:]) == "internal_ig"
        ]
    )
    if rated is not None:
        agg = agg.with_columns(rated.alias("cq_rated_total"))
    return agg


def _past_due_columns(facts: pl.DataFrame) -> pl.DataFrame:
    pd_facts = add_period_labels(
        facts.filter(
            pl.col("instant")
            & pl.col("pd_cat").is_not_null()
            & pl.col("scope").is_null()
            & pl.col("tag").is_in(list(variables.LOAN_TAGS))
            & (pl.col("unit") == "USD")
        )
    )
    if pd_facts.is_empty():
        return pl.DataFrame(schema={k: pl.String for k in PANEL_KEY})

    pd_facts = _restrict_to_best_signature(
        _tag_priority(pd_facts, variables.LOAN_TAGS),
        PANEL_KEY,
        allow_subslice=True,
        prefer_modern_axes=False,
    )
    member_cols = [c for c in MEMBER_COLS if c in pd_facts.columns]
    agg = (
        pd_facts.sort(["filed"], descending=True, nulls_last=True)
        .unique(subset=[*PANEL_KEY, *member_cols], keep="first", maintain_order=True)
        .group_by([*PANEL_KEY, "pd_cat"])
        .agg(pl.col("value").sum())
        .pivot(on="pd_cat", index=PANEL_KEY, values="value")
    )
    return agg.rename({c: f"pd_{c}" for c in agg.columns if c not in PANEL_KEY})


def _legacy_past_due_columns(facts: pl.DataFrame) -> pl.DataFrame:
    """Delinquency columns for filings that carry no delinquency axis.

    Before 2018 a bank tags one *element* per bucket and dimensions that
    element by loan class, so the bucket lives in the tag rather than in a
    member.  :func:`_past_due_columns` cannot read this shape: it fixes one
    tag per bank-quarter to keep the axis members mutually exclusive, which is
    right when the buckets are members and fatal when each bucket *is* a
    different tag -- all but one would be dropped.

    Here the tag is chosen per *bucket* instead.  Within a bucket the
    candidates are alternatives rather than components (a bank tags 90+ as
    either "equal to or greater than 90 days" or "90 days and still
    accruing", and both would be the same exposure counted twice), so the
    priority order in :data:`variables.LEGACY_PAST_DUE_TAGS` still picks one.
    Across buckets nothing is shared, so every bucket survives.
    """
    tags = variables.LEGACY_PAST_DUE_ORDER
    pd_facts = add_period_labels(
        facts.filter(
            pl.col("instant")
            & pl.col("tag").is_in(list(tags))
            & pl.col("pd_cat").is_not_null()
            & pl.col("scope").is_null()
            & (pl.col("unit") == "USD")
        )
    )
    if pd_facts.is_empty():
        return pl.DataFrame(schema={k: pl.String for k in PANEL_KEY})

    key = [*PANEL_KEY, "pd_cat"]
    chosen = _restrict_to_best_signature(
        _tag_priority(pd_facts, tags),
        key,
        allow_subslice=True,
        prefer_modern_axes=False,
    )
    agg = _sum_members(chosen, key, "value").pivot(
        on="pd_cat", index=PANEL_KEY, values="value"
    )
    return agg.rename({c: f"pd_{c}" for c in agg.columns if c not in PANEL_KEY})


def _merge(left: pl.DataFrame | None, right: pl.DataFrame) -> pl.DataFrame:
    if left is None:
        return right
    return left.join(right, on=PANEL_KEY, how="full", coalesce=True)


def _fill_nulls_from(left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
    """Outer-join ``right`` into ``left``, writing only where ``left`` is null."""
    incoming = [c for c in right.columns if c not in PANEL_KEY]
    if not incoming:
        return left
    suffixed = right.rename({c: f"{c}__legacy" for c in incoming})
    out = left.join(suffixed, on=PANEL_KEY, how="full", coalesce=True)
    for col in incoming:
        if col in left.columns:
            out = out.with_columns(
                pl.col(col).fill_null(pl.col(f"{col}__legacy")).alias(col)
            )
        else:
            out = out.with_columns(pl.col(f"{col}__legacy").alias(col))
    return out.drop([f"{c}__legacy" for c in incoming])


def add_basis(df: pl.DataFrame) -> pl.DataFrame:
    """Label each bank-quarter with the accounting regime it was reported under.

    Incurred loss and CECL are not the same measurement.  ``acl`` under CECL is
    a lifetime expected-loss estimate; before 2020Q1 it is an incurred-loss
    reserve, and the level step between them shows up in ``reserve_coverage``
    and ``reserve_to_nonaccrual`` as a jump that looks like a credit event.
    The panel splices the two anyway -- a single series is what a user wants --
    but says which regime every row came from so the break can be seen,
    filtered on, or controlled for.
    """
    if df.is_empty() or "ticker" not in df.columns:
        return df
    adopted = pl.Series(
        "_adopted",
        [cecl_adoption(t) for t in df["ticker"].to_list()],
        dtype=pl.Date,
    )
    return df.with_columns(
        pl.when(pl.col("period") >= adopted)
        .then(pl.lit("cecl"))
        .otherwise(pl.lit("incurred"))
        .alias("basis")
    )


# --------------------------------------------------------------------------
# Derived quantities
# --------------------------------------------------------------------------


# The disjoint leaf categories.  Deliberately not the rollups: summing
# cre_total beside its own construction and multifamily members, or
# commercial_total beside the classes underneath it, would count the same
# balance twice and make the coherence metric read high for the wrong reason.
MIX_LEAF_CATEGORIES: tuple[str, ...] = (
    "cre_total",
    "ci",
    "small_business",
    "lease",
    "municipal",
    "fund_finance",
    "financial_institutions",
    "commercial_other",
    "resi_mortgage",
    "home_equity",
    "credit_card",
    "auto",
    "student",
    "securities_based",
    "consumer_other",
    "other",
)


# The CRE classes a bank may break out instead of tagging a CRE total.
CRE_COMPONENTS: tuple[str, ...] = (
    "cre_investor",
    "cre_owner_occupied",
    "multifamily",
    "construction",
)


def with_cre_rollup(df: pl.DataFrame) -> pl.DataFrame:
    """Fill ``loans_cre_total`` from its own classes where no total is tagged.

    JPMorgan is the case the README calls out as legitimately null, and it is
    only half true: it tags no commercial-real-estate *rollup* member, but it
    does tag multifamily, commercial lessors and construction and development,
    which at 2025Q4 come to $178bn.  The CRE book was in the panel all along,
    spread across three columns, while ``cre_total`` -- the one the ratio and
    the coherence metric read -- stayed empty.

    Only nulls are written, so a bank that tags its own total keeps it, and
    only the components are summed, never a mix of total and components.

    **Two components at least.**  One is not a partition of the CRE book, it
    is a part of it, and promoting it to the total asserts that every other
    class is zero.  Zions tags construction alone, at $18-25m against a $57bn
    book, and calling that its commercial real estate reported a CRE share of
    0.0% for one of the most CRE-concentrated banks in the panel.  Left null it
    is honestly missing instead.
    """
    parts = [f"loans_{c}" for c in CRE_COMPONENTS if f"loans_{c}" in df.columns]
    if len(parts) < 2:
        return df
    if "loans_cre_total" not in df.columns:
        # A bank that tags no CRE rollup has no such column until one is made.
        # It exists in a universe run only because some *other* bank tags it,
        # so without this a --tickers JPM build silently skips the rollup.
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("loans_cre_total"))
    summed = pl.sum_horizontal([pl.col(c).fill_null(0.0) for c in parts])
    n_parts = pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int32) for c in parts])
    return df.with_columns(
        pl.when(
            pl.col("loans_cre_total").is_null() & (n_parts >= 2) & (summed.abs() > 0)
        )
        .then(summed)
        .otherwise(pl.col("loans_cre_total"))
        .alias("loans_cre_total")
    )


def with_mix_coverage(df: pl.DataFrame) -> pl.DataFrame:
    """(Re)compute ``mix_coverage_pct``: leaf categories over reported loans.

    Always recomputed rather than gap-filled.  This is the column a consumer
    is told to filter on before ranking peers, and a *stale* coherence score is
    worse than a missing one -- it was reading a flawless 100.0 for Wells Fargo
    in quarters where the true figure was 101.7, because the fallbacks had
    filled a loan category after ``add_derived`` had already run.
    """
    leaf = [f"loans_{c}" for c in MIX_LEAF_CATEGORIES if f"loans_{c}" in df.columns]
    if not leaf or "loans_total" not in df.columns:
        return df
    return df.with_columns(
        pl.when(pl.col("loans_total").abs() > 0)
        .then(
            pl.sum_horizontal([pl.col(c).fill_null(0.0) for c in leaf])
            / pl.col("loans_total")
            * 100
        )
        .otherwise(None)
        .alias("mix_coverage_pct")
    )


def add_derived(df: pl.DataFrame) -> pl.DataFrame:
    """NPAs, ratios and growth rates, skipping anything whose inputs are absent."""
    have = set(df.columns)

    # NPA = nonaccrual + OREO.  OREO is often untagged and is treated as zero,
    # but a *missing nonaccrual* must stay missing: filling it with zero turns
    # "not disclosed" into "no problem loans", which is the worst possible
    # direction for a credit benchmark to be wrong in.
    if "nonaccrual_total" in have:
        oreo = pl.col("oreo").fill_null(0.0) if "oreo" in have else pl.lit(0.0)
        df = df.with_columns(
            pl.when(pl.col("nonaccrual_total").is_null())
            .then(None)
            .otherwise(pl.col("nonaccrual_total") + oreo)
            .alias("npa_total")
        )
        have.add("npa_total")

    # NCO fallback: gross charge-offs less recoveries.
    for suffix in ("total", *taxonomy.LOAN_CATEGORIES):
        nco, co, rec = f"nco_{suffix}", f"charge_offs_{suffix}", f"recoveries_{suffix}"
        if co in have and rec in have:
            filled = pl.col(co) - pl.col(rec)
            df = df.with_columns(
                (pl.col(nco) if nco in have else pl.lit(None, dtype=pl.Float64))
                .fill_null(filled)
                .alias(nco)
            )
            have.add(nco)

    # Each loan category picks the signature that discloses it best, so two
    # categories can come from different disclosure tables and need not be
    # mutually exclusive.  mix_coverage_pct measures how much of the reported
    # total the disjoint leaf categories actually account for: near 100 means
    # the mix is coherent, well above 100 means categories overlap, well below
    # means the breakdown is partial.  Filter on it before ranking peers.
    df = with_cre_rollup(df)
    df = with_mix_coverage(df)
    if "mix_coverage_pct" in df.columns:
        have.add("mix_coverage_pct")

    exprs = []
    for ratio in RATIOS:
        if ratio.numerator not in have or ratio.denominator not in have:
            continue
        num = pl.col(ratio.numerator).cast(pl.Float64)
        den = pl.col(ratio.denominator).cast(pl.Float64)
        value = pl.when(den.abs() > 0).then(num / den).otherwise(None)
        if ratio.annualize:
            value = value * 4.0
        exprs.append((value * 100.0).alias(ratio.name))
    if exprs:
        df = df.with_columns(exprs)

    # Growth: QoQ annualised and YoY, on the balances that matter.
    growth_targets = [
        c
        for c in (
            "loans_total",
            "loans_cre_total",
            "loans_construction",
            "loans_ci",
            "loans_multifamily",
            "deposits",
        )
        if c in have
    ]
    if growth_targets:
        df = df.sort(["ticker", "period"])
        for col in growth_targets:
            prev = pl.col(col).shift(1).over("ticker")
            prev4 = pl.col(col).shift(4).over("ticker")
            df = df.with_columns(
                pl.when(prev.abs() > 0)
                .then(((pl.col(col) / prev) ** 4 - 1) * 100)
                .otherwise(None)
                .alias(f"{col}_qoq_ann_pct"),
                pl.when(prev4.abs() > 0)
                .then((pl.col(col) / prev4 - 1) * 100)
                .otherwise(None)
                .alias(f"{col}_yoy_pct"),
            )
    return df


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def build_panel(
    facts: pl.DataFrame,
    *,
    categories: tuple[str, ...] | None = None,
    since: dt.date | None = None,
    derived: bool = True,
) -> pl.DataFrame:
    """Assemble the one-row-per-bank-quarter panel."""
    if facts.is_empty():
        return pl.DataFrame()

    cats = categories or tuple(taxonomy.LOAN_CATEGORIES)
    if since is not None:
        facts = facts.filter(pl.col("period") >= since)

    parts = [
        _stock_columns(facts, cats),
        _flow_columns(facts, cats),
        _credit_quality_columns(facts),
        _past_due_columns(facts),
    ]
    parts = [p for p in parts if not p.is_empty() and p.height > 0]

    # The legacy delinquency elements go in behind the axis path, in the same
    # spirit as the HTML and IR fallbacks: only cells the axis path left null
    # are written, so a quarter that has both shapes keeps the axis reading.
    legacy_pd = _legacy_past_due_columns(facts)
    has_legacy = not legacy_pd.is_empty() and legacy_pd.height > 0
    if not parts:
        if not has_legacy:
            return pl.DataFrame()
        panel = legacy_pd
    else:
        panel = parts[0]
        for part in parts[1:]:
            panel = panel.join(part, on=PANEL_KEY, how="full", coalesce=True)
        if has_legacy:
            panel = _fill_nulls_from(panel, legacy_pd)

    # Quarter-end filter: drop stub periods that are not quarter ends.
    panel = panel.filter(
        pl.col("period").dt.month().is_in([3, 6, 9, 12])
        & (pl.col("period").dt.day() >= 28)
    )

    panel = add_basis(panel)
    if derived:
        panel = add_derived(panel)

    front = [c for c in PANEL_KEY if c in panel.columns]
    if "basis" in panel.columns:
        front.append("basis")
    rest = sorted(c for c in panel.columns if c not in front)
    return panel.select([*front, *rest]).sort(["ticker", "period"])
