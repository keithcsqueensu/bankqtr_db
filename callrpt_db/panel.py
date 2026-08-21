"""Call Report facts -> bank-quarter panel, at charter and holding-company level.

Two levels, one build
---------------------
A Call Report is filed by a bank.  The panel this project cares about is one
row per *organisation* per quarter, so charters are summed up to the holding
company using the NIC control graph -- but the charter-level frame is kept as
an output too, because it is the level the data actually exists at and the only
level at which nothing has been assumed.

The rollup is banks only.  A holding company's broker-dealer, its card funding
trusts and its insurance subsidiaries file no Call Report and are simply not
here, so ``callrpt`` totals are smaller than the 10-K's and the two are not
expected to tie.  Every holding-company row carries ``n_charters`` and
``charters`` so a reader can see what was summed.

The organisation is what it has become
--------------------------------------
From 2001 the organisation summed is not the 2026 RSSD's own subtree but its
**lineage**: the subtree of every predecessor that was later absorbed into it,
for every quarter the predecessor still stood on its own -- Wachovia's banks
under Wells Fargo before 2009, National City's under PNC, Colonial Bank under
Truist until the FDIC closed it.  :mod:`callrpt_db.lineage` resolves that map;
:func:`universe_filers` applies it, own charters first and predecessors
second, so a firm tracked in the panel in its own right (Discover until 2025)
is never also summed into its acquirer's history.  Every holding-company row
says how much of it is synthetic: ``has_predecessor``, ``predecessor_count``,
``predecessors`` and, for the stress observations, ``predecessor_failed``.

Four hazards this handles
-------------------------
**Flows are year-to-date.**  Every ``RIAD`` item accumulates through the year
and resets in Q1, so a quarter is the difference against the previous quarter of
the *same year*.  Differencing happens **per charter, before the rollup**: a
holding company that acquires a bank mid-year would otherwise show that bank's
whole year-to-date as a single quarter's charge-offs.

**A pooling merger restates the survivor's year-to-date.**  When one charter
is merged into another under common control, the Call Report instructions have
the survivor report income *as if combined from the start of the year*.  Its
next difference therefore contains the absorbed charter's whole year to date
-- quarters that charter had already filed itself.  :func:`quarterize` takes
them back out using NIC's accounting-method flag.

**The charter set moves.**  Zions ran seven bank charters until 2018 and one
after; Truist is BB&T plus SunTrust from 2019Q4.  The set is resolved per
quarter from the dated NIC graph, so a merger enters the panel when it happened
rather than being back-applied to the whole window.

**A missing quarter is not a zero.**  If the previous quarter is absent the
difference cannot be taken, and the flow is left null rather than being
reported as its own year-to-date figure.  The same rule holds at the rollup: a
column no charter reported is null for the holding company, not 0 -- which an
earlier version got wrong, and which is why its CET1 read 0.0 rather than
null for every bank before 2015.
"""

from __future__ import annotations

import datetime as dt
import logging

import polars as pl

from bankqtr_db.variables import RATIOS, RatioDef

from . import cdr, config, mdrm, nic, schedules
from . import lineage as lineage_mod

log = logging.getLogger(__name__)

PANEL_KEY = ["ticker", "rssd", "period"]

# Leaf categories of the RC-C partition, for the mix.  Unlike the XBRL panel's
# equivalent this really is a partition -- see ``mix_coverage_pct`` below.
MIX_LEAF_CATEGORIES: tuple[str, ...] = tuple(
    spec.name for spec in mdrm.RCC_LOAN_ITEMS
)

# CRE on the form's own terms: nonfarm nonresidential (both occupancy classes)
# plus construction plus multifamily.  Before 2007 nonfarm nonresidential is a
# single line, so the total is built from that rather than from the
# owner-occupied / investor split, which is null there.
CRE_COMPONENTS: tuple[str, ...] = (
    "loans_cre_nonfarm_nonres",
    "loans_construction",
    "loans_multifamily",
)
CRE_COMPONENTS_SPLIT: tuple[str, ...] = (
    "loans_cre_owner_occupied",
    "loans_cre_investor",
    "loans_construction",
    "loans_multifamily",
)

# RC-N's total row exists from this quarter; before it the total is built.
RCN_TOTAL_ROW_SINCE = dt.date(2017, 3, 31)

# Ratios this source can compute that the EDGAR panel has no inputs for.
# Same class as the shared definitions so they are applied the same way.
CALL_RATIOS: tuple[RatioDef, ...] = (
    RatioDef("nco_rate_construction", "nco_construction", "loans_construction", annualize=True),
    RatioDef("nco_rate_cre_nonfarm_nonres", "nco_cre_nonfarm_nonres", "loans_cre_nonfarm_nonres", annualize=True),
    RatioDef("nco_rate_multifamily", "nco_multifamily", "loans_multifamily", annualize=True),
    RatioDef("nco_rate_resi", "nco_resi_mortgage", "loans_resi_mortgage", annualize=True),
    RatioDef("nco_rate_home_equity", "nco_home_equity", "loans_home_equity", annualize=True),
    RatioDef("nco_rate_consumer_noncard", "nco_consumer_noncard", "loans_consumer_noncard", annualize=True),
    RatioDef("provision_rate", "provision_total", "loans_total", annualize=True, description="Annualised provision / loans"),
    RatioDef("noncurrent_ratio", "noncurrent_total", "loans_total", description="(90+ past due accruing + nonaccrual) / loans"),
    RatioDef("reserve_to_noncurrent", "acl_total", "noncurrent_total"),
    RatioDef("nonaccrual_ratio_construction", "nonaccrual_construction", "loans_construction"),
    RatioDef("nonaccrual_ratio_cre_nonfarm_nonres", "nonaccrual_cre_nonfarm_nonres", "loans_cre_nonfarm_nonres"),
    RatioDef("nonaccrual_ratio_resi", "nonaccrual_resi_mortgage", "loans_resi_mortgage"),
    RatioDef("dpd_30_89_pct_resi", "dpd_30_89_resi_mortgage", "loans_resi_mortgage"),
    RatioDef("dpd_30_89_pct_construction", "dpd_30_89_construction", "loans_construction"),
    RatioDef("tdr_pct", "loans_tdr_accruing", "loans_total"),
    RatioDef("brokered_deposits_pct", "deposits_brokered", "deposits"),
    RatioDef("commitments_to_loans", "commitments_total", "loans_total"),
    RatioDef("roa", "net_income", "assets_average", annualize=True, description="Annualised net income / average assets"),
    RatioDef("ppnr_rate", "ppnr", "assets_average", annualize=True, description="Annualised pre-provision net revenue / average assets"),
    RatioDef("nii_to_avg_assets", "net_interest_income", "assets_average", annualize=True),
    RatioDef("wholesale_funding_pct", "wholesale_funding", "assets"),
    RatioDef("tce_ratio", "tangible_common_equity", "tangible_assets", description="Tangible common equity / tangible assets"),
    RatioDef("equity_to_assets", "equity", "assets"),
)


def quarter_label(period: dt.date) -> str:
    return f"{period.year}Q{(period.month - 1) // 3 + 1}"


# --------------------------------------------------------------------------
# Charter level
# --------------------------------------------------------------------------

MAPPING_SCHEMA: dict[str, type[pl.DataType] | pl.DataType] = {
    "ticker": pl.Utf8,
    "bank": pl.Utf8,
    "holding_rssd": pl.Int64,
    "rssd": pl.Int64,
    "period": pl.Date,
    # The predecessor through which the charter was claimed; null for the
    # organisation's own subtree.
    "via_rssd": pl.Int64,
    "via_type": pl.Utf8,
    # The charter itself, or the predecessor it came through, later failed.
    "failed_lineage": pl.Boolean,
}


def universe_filers(
    periods: list[str],
    holdings: tuple[config.Holding, ...],
    lineages: dict[str, lineage_mod.Lineage] | None = None,
) -> pl.DataFrame:
    """Map each quarter's Call Report filers to the holding company above them.

    One row per (ticker, rssd, period).  Resolved per quarter against that
    quarter's own roster and the NIC graph as it stood then, so a charter joins
    and leaves an organisation on the date it actually did.

    Two passes, and the order is the point.  First every holding claims the
    charters under its *own* RSSDs; then every holding claims the charters
    under its predecessors that were still independent that quarter.  A
    charter already claimed is never claimed again, so Discover Bank is
    Discover's for as long as Discover is a row in this panel and Capital
    One's only afterwards, and two organisations never sum the same charter.
    """
    rows: list[tuple] = []
    failed = nic.failed_rssds() if lineages else frozenset()

    def claim(
        claimed: dict[int, str],
        period: str,
        as_of: dt.date,
        holding: config.Holding,
        charter: int,
        via: lineage_mod.Predecessor | None,
    ) -> None:
        if charter in claimed:
            if claimed[charter] != holding.ticker and via is None:
                # Two tracked organisations' own subtrees overlap: a
                # transition quarter.  First claim wins and it is logged
                # rather than silently double counted.
                log.warning(
                    "%s: charter %d claimed by both %s and %s; kept %s",
                    period, charter, claimed[charter], holding.ticker, claimed[charter],
                )
            return
        claimed[charter] = holding.ticker
        rows.append(
            (
                holding.ticker,
                holding.name,
                holding.rssd,
                charter,
                as_of,
                None if via is None else via.rssd,
                None if via is None else via.succession_type,
                charter in failed or (via is not None and via.rssd in failed),
            )
        )

    for period in periods:
        path = cdr.zip_path(period)
        if not path.exists():
            continue
        roster = schedules.roster(path)
        present = set(roster["rssd"].to_list())
        as_of = cdr.quarter_end(period)
        graph = nic.hierarchy(as_of)
        claimed: dict[int, str] = {}
        live = [
            h for h in holdings if h.last_filing is None or as_of <= h.last_filing
        ]
        for holding in live:
            found: set[int] = set()
            for rssd in holding.rssds:
                found.update(nic.call_filers(rssd, present, as_of=as_of, kids=graph))
            for charter in sorted(found):
                claim(claimed, period, as_of, holding, charter, None)

        if lineages:
            for holding in live:
                lin = lineages.get(holding.ticker)
                if lin is None:
                    continue
                # A charter can be reachable through two live predecessors --
                # the holding company that was merged and, later, the charter
                # itself when it was folded into a sibling.  Attribute it to
                # the outside organisation rather than to its own later
                # reorganisation, so ``predecessors`` names what was bought.
                ordered = sorted(
                    lin.active(as_of),
                    key=lambda p: (p.succession_type == lineage_mod.REORG, p.effective_to, p.rssd),
                )
                for pred in ordered:
                    for charter in nic.call_filers(pred.rssd, present, as_of=as_of, kids=graph):
                        claim(claimed, period, as_of, holding, charter, pred)

    if not rows:
        return pl.DataFrame(schema=MAPPING_SCHEMA)
    return pl.DataFrame(rows, schema=MAPPING_SCHEMA, orient="row").unique()


def unfiled_depositories(
    periods: list[str],
    holdings: tuple[config.Holding, ...],
    lineages: dict[str, lineage_mod.Lineage] | None = None,
) -> pl.DataFrame:
    """Insured depositories in the organisation's tree that filed nothing CDR holds.

    Almost all of these are thrifts.  An OTS-supervised savings institution
    filed a Thrift Financial Report rather than a Call Report until 2012Q1, so
    Washington Mutual, Golden West, Countrywide Bank, Sovereign, ING Direct,
    Hudson City and E*TRADE Bank are in the graph, are in the lineage, and
    contribute nothing before that date.  The count is carried per
    bank-quarter so a reader knows where the history is thin rather than
    complete.

    Detection is by FDIC certificate -- the entity is insured -- and absence
    from that quarter's roster.  It cannot see an entity NIC's attribute
    files omit, and they omit several real filers (Discover Bank among them),
    so this is a floor, not a census.
    """
    entities = nic.entities()
    rows: list[tuple[str, dt.date, int, str]] = []
    for period in periods:
        path = cdr.zip_path(period)
        if not path.exists():
            continue
        present = set(schedules.roster(path)["rssd"].to_list())
        as_of = cdr.quarter_end(period)
        stamp = as_of.strftime("%Y%m%d")
        graph = nic.hierarchy(as_of)
        for holding in holdings:
            if holding.last_filing is not None and as_of > holding.last_filing:
                continue
            roots = list(holding.rssds)
            if lineages and holding.ticker in lineages:
                roots.extend(p.rssd for p in lineages[holding.ticker].active(as_of))
            members: set[int] = set()
            for root in roots:
                members.add(root)
                members |= nic.descendants(root, graph)
            missing: list[int] = []
            for rssd in members:
                if rssd in present:
                    continue
                ent = entities.get(rssd)
                if ent is None or ent.fdic_cert is None:
                    continue
                if ent.entity_type not in lineage_mod.DEPOSITORY_TYPES:
                    continue
                if ent.opened and ent.opened > stamp:
                    continue
                if ent.ended and ent.ended < stamp:
                    continue
                missing.append(rssd)
            if missing:
                rows.append(
                    (holding.ticker, as_of, len(missing), ";".join(str(m) for m in sorted(missing)))
                )
    return pl.DataFrame(
        rows,
        schema={
            "ticker": pl.Utf8,
            "period": pl.Date,
            "n_insured_not_filing": pl.Int64,
            "insured_not_filing": pl.Utf8,
        },
        orient="row",
    )


def charter_frame(
    periods: list[str],
    rssds: set[int] | None = None,
    *,
    specs: tuple[mdrm.ItemSpec, ...] = mdrm.ALL_ITEMS,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read the cached periods into one wide charter-quarter frame.

    Returns the frame and the concatenated rosters, the latter because the
    filing type is needed downstream and re-reading the zips to get it is the
    slowest thing here.
    """
    wide: list[pl.DataFrame] = []
    rosters: list[pl.DataFrame] = []
    codes = mdrm.wanted_codes()
    for period in periods:
        path = cdr.zip_path(period)
        if not path.exists():
            log.warning("%s is not cached; skipping", period)
            continue
        roster = schedules.roster(path)
        if rssds is not None:
            roster = roster.filter(pl.col("rssd").is_in(list(rssds)))
        rosters.append(roster)
        long = schedules.read_period(path, rssds=rssds, codes=codes)
        if long.is_empty():
            continue
        wide.append(schedules.resolve_items(long, specs, forms=roster))
    if not wide:
        return pl.DataFrame(), pl.DataFrame()
    frame = pl.concat(wide, how="diagonal_relaxed")
    roster_all = pl.concat(rosters, how="diagonal_relaxed")
    return frame.sort(["rssd", "period"]), roster_all


# Year-to-date flows that cannot fall between one quarter and the next of the
# same year.  A fall means the charter's income statement restarted -- see
# ``quarterize`` -- and one is enough to say so.
MONOTONE_YTD_FLOWS: tuple[str, ...] = (
    "charge_offs_total",
    "recoveries_total",
    "interest_income",
    "interest_expense",
    "noninterest_expense",
)


def quarterize(frame: pl.DataFrame, pooled: pl.DataFrame | None = None) -> pl.DataFrame:
    """Turn year-to-date flow columns into single-quarter figures.

    A flow is its year-to-date value less the previous quarter's, within the
    same calendar year; Q1 stands as reported.  The previous quarter has to be
    the *immediately* previous one -- a gap in a charter's filing history makes
    the difference meaningless, so it is left null.

    **Push-down accounting restarts the year.**  When a charter is acquired
    and the purchase price is pushed down to its books, its income statement
    begins again on the acquisition date, and its next Call Report's
    year-to-date covers only the weeks since.  Differencing that against the
    previous quarter's full year-to-date gives a large negative number:
    Fleet National Bank's charge-offs in 2004Q2, LaSalle's in 2007Q4,
    National City's and Wachovia's in 2008Q4 all came out below zero.  A
    restart is recognised when any gross flow that cannot fall
    (:data:`MONOTONE_YTD_FLOWS`) has fallen; the quarter is then taken as the
    year-to-date since the restart -- the weeks since the acquisition, which
    is what the filing contains -- and ``flow_reset`` marks the row.  What is
    lost is the acquired charter's activity between the previous quarter end
    and the acquisition date; what is avoided is a negative loss.

    ``pooled`` lists common-control mergers as ``(rssd, predecessor, period)``
    -- the survivor, the charter merged into it, and the quarter it happened
    in.  Under the Call Report's instructions for a combination of entities
    under common control the survivor restates its year-to-date income as if
    the two had been one since January 1, so its difference for that quarter
    carries the predecessor's year-to-date through the previous quarter -- a
    figure the predecessor had already filed under its own RSSD and that the
    rollup has already counted.  That amount is subtracted here.  Nothing is
    done for a Q1 merger: the restated year to date starts at the same
    January 1 as the predecessor's, which filed nothing for the new year.
    """
    flows = [c for c in frame.columns if c in mdrm.FLOW_COLUMNS]
    if not flows:
        return frame
    raw = frame.sort(["rssd", "period"]).with_columns(
        pl.col("period").dt.year().alias("_year"),
        pl.col("period").dt.quarter().alias("_q"),
    )
    prev_year = pl.col("_year").shift(1).over("rssd")
    prev_q = pl.col("_q").shift(1).over("rssd")
    contiguous = (prev_year == pl.col("_year")) & (prev_q == pl.col("_q") - 1)
    monotone = [c for c in MONOTONE_YTD_FLOWS if c in flows]
    fell = (
        pl.any_horizontal(
            [
                (pl.col(c) < pl.col(c).shift(1).over("rssd") - 0.5).fill_null(False)
                for c in monotone
            ]
        )
        if monotone
        else pl.lit(False)
    )
    reset = (contiguous & (pl.col("_q") != 1) & fell).fill_null(False)
    exprs = []
    for column in flows:
        previous = pl.col(column).shift(1).over("rssd")
        exprs.append(
            pl.when(pl.col("_q") == 1)
            .then(pl.col(column))
            .when(contiguous & reset)
            .then(pl.col(column))
            .when(contiguous)
            .then(pl.col(column) - previous)
            .otherwise(None)
            .alias(column)
        )
    out = raw.with_columns([*exprs, reset.alias("flow_reset")])

    if pooled is not None and not pooled.is_empty():
        # The predecessor's year-to-date at the quarter end before the merger,
        # summed per survivor-quarter (a survivor can absorb several).
        events = pooled.filter(pl.col("period").dt.quarter() > 1).with_columns(
            pl.col("period")
            .map_elements(lineage_mod.previous_quarter_end, return_dtype=pl.Date)
            .alias("_prev")
        )
        prior = raw.select(
            pl.col("rssd").alias("predecessor"),
            pl.col("period").alias("_prev"),
            *[pl.col(c).alias(f"_adj_{c}") for c in flows],
        )
        adjust = (
            events.join(prior, on=["predecessor", "_prev"], how="inner")
            .group_by(["rssd", "period"])
            .agg([pl.col(f"_adj_{c}").sum().alias(f"_adj_{c}") for c in flows])
        )
        if not adjust.is_empty():
            # A survivor whose own year restarted in the same quarter carries
            # no restated history to take out.
            out = out.join(adjust, on=["rssd", "period"], how="left").with_columns(
                [
                    pl.when(pl.col(c).is_null() | pl.col("flow_reset"))
                    .then(pl.col(c))
                    .otherwise(pl.col(c) - pl.col(f"_adj_{c}").fill_null(0.0))
                    .alias(c)
                    for c in flows
                ]
            ).drop([f"_adj_{c}" for c in flows])
            log.info(
                "pooling adjustment applied to %d charter-quarters", adjust.height
            )
    return out.drop("_year", "_q")


def add_rcn_totals(frame: pl.DataFrame) -> pl.DataFrame:
    """Fill RC-N's three totals from the category rows where the form has none.

    The total row (1403/1406/1407) exists from 2017Q1.  Before that the total
    is the sum of the fourteen category rows in :data:`mdrm.RCN_CATEGORIES`,
    a list checked against the form's own total from 2017 on: it ties for
    99.3% of filers, and the remainder are off by the foreign-office lines
    that only form 031 carries.  The reported total always wins where it
    exists; the built one is used only where the reported is null.
    """
    exprs = []
    for total, parts in mdrm.RCN_TOTAL_COMPONENTS.items():
        have = [p for p in parts if p in frame.columns]
        if total not in frame.columns or not have:
            continue
        present = pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int32) for c in have])
        summed = pl.sum_horizontal([pl.col(c).fill_null(0.0) for c in have])
        exprs.append(
            pl.when(pl.col(total).is_not_null())
            .then(pl.col(total))
            .when(present > 0)
            .then(summed)
            .otherwise(None)
            .alias(total)
        )
        exprs.append(pl.col(total).is_null().alias(f"_{total}_built"))
    if not exprs:
        return frame
    out = frame.with_columns(exprs)
    built = [c for c in out.columns if c.endswith("_built")]
    return out.with_columns(
        pl.any_horizontal([pl.col(c) for c in built]).alias("rcn_total_built")
    ).drop(built)


def add_partition_check(frame: pl.DataFrame) -> pl.DataFrame:
    """Measure the RC-C leaves against RC-C's own stated total.

    This is the check the XBRL panel cannot run.  Schedule RC-C states the
    total its categories partition, so the mapping can be cross-footed rather
    than merely believed: ``rcc_residual`` is what the leaves fail to account
    for, in dollars, and ``rcc_residual_pct`` the same as a share of the total.

    Item 12 is the categories **less item 11, unearned income**, so that is
    subtracted here.  Almost every filer reports it as zero, which is why
    leaving it out went unnoticed until City National Bank -- whose $562m of
    unearned income put it 0.86% over its own total, in every quarter, with
    nothing else wrong.

    A non-zero residual is not necessarily an error, and the sign says which
    kind it is.  **Positive is a bug**: the leaves overlap, and something is
    being counted twice.  **Negative is usually the form**: on 031 the items
    for securities lending, leases and other loans are collected in the
    domestic column only, while the total is consolidated, so a bank with
    foreign offices has loans that genuinely have no category -- BNY Mellon's
    2017Q3 consolidated book is $29.5bn against $15.6bn domestic, and $4.9bn
    of the difference belongs to lines the form never breaks out abroad.  The
    residual is reported rather than hidden so that a consumer ranking on
    portfolio mix can see how much of the book the mix actually describes.
    """
    leaves = [c for c in MIX_LEAF_CATEGORIES if c in frame.columns]
    if not leaves or "loans_rcc_total" not in frame.columns:
        return frame
    summed = pl.sum_horizontal([pl.col(c).fill_null(0.0) for c in leaves])
    if "loans_unearned_income" in frame.columns:
        summed = summed - pl.col("loans_unearned_income").fill_null(0.0)
    return frame.with_columns(
        (summed - pl.col("loans_rcc_total")).alias("rcc_residual"),
        pl.when(pl.col("loans_rcc_total").abs() > 0)
        .then((summed - pl.col("loans_rcc_total")) / pl.col("loans_rcc_total") * 100)
        .otherwise(None)
        .alias("rcc_residual_pct"),
        # What the categories leave unexplained, as a balance rather than a
        # diagnostic: the part of the book the mix does not describe.
        (pl.col("loans_rcc_total") - summed).alias("loans_unallocated"),
    )


# --------------------------------------------------------------------------
# Holding-company rollup
# --------------------------------------------------------------------------

# Columns that are keys or metadata and are never summed across charters.
NOT_SUMMED: frozenset[str] = frozenset(
    {
        "rssd", "period", "ticker", "bank", "holding_rssd", "form", "name",
        "via_rssd", "via_type", "failed_lineage", "rcn_total_built", "flow_reset",
    }
)


def _summable(frame: pl.DataFrame) -> list[str]:
    return [
        c
        for c in frame.columns
        if c not in NOT_SUMMED and frame.schema[c] in (pl.Float64, pl.Int64)
    ]


def roll_up(charters: pl.DataFrame, mapping: pl.DataFrame) -> pl.DataFrame:
    """Sum charter rows into one row per holding company per quarter.

    Summing is right for every column here because they are all dollar amounts
    on the same basis, and the charters are disjoint by construction -- each is
    claimed by exactly one holding company per quarter.  Ratios are **not**
    summed; they are recomputed from the summed inputs afterwards, since the
    average of two ratios is not the ratio of the sums.

    A column none of the charters reported stays **null**.  Polars' ``sum``
    of all-null is 0, and that 0 is a lie the panel used to tell: CET1 of 0.0
    for every bank before 2015, nonaccrual of 0.0 for every bank before 2017.
    """
    if charters.is_empty() or mapping.is_empty():
        return pl.DataFrame()
    joined = mapping.join(charters, on=["rssd", "period"], how="inner")
    if joined.is_empty():
        return pl.DataFrame()
    value_cols = _summable(joined)
    aggs = [
        pl.when(pl.col(c).is_not_null().any())
        .then(pl.col(c).sum())
        .otherwise(None)
        .alias(c)
        for c in value_cols
    ]
    aggs.append(pl.col("rssd").n_unique().alias("n_charters"))
    aggs.append(
        pl.col("rssd").sort().cast(pl.Utf8).str.join(";").alias("charters")
    )
    if "via_rssd" in joined.columns:
        via = pl.col("via_rssd")
        aggs.extend(
            [
                via.is_not_null().any().alias("has_predecessor"),
                via.drop_nulls().n_unique().alias("predecessor_count"),
                via.is_not_null().sum().alias("n_predecessor_charters"),
                via.drop_nulls().unique().sort().cast(pl.Utf8).str.join(";").alias("predecessors"),
                pl.col("failed_lineage").fill_null(False).any().alias("predecessor_failed"),
            ]
        )
    if "rcn_total_built" in joined.columns:
        aggs.append(pl.col("rcn_total_built").fill_null(False).any().alias("rcn_total_built"))
    if "flow_reset" in joined.columns:
        aggs.append(pl.col("flow_reset").fill_null(False).sum().alias("n_flow_resets"))
    out = joined.group_by(["ticker", "bank", "holding_rssd", "period"]).agg(aggs)
    return out.rename({"holding_rssd": "rssd"}).sort(["ticker", "period"])


# --------------------------------------------------------------------------
# Derived columns
# --------------------------------------------------------------------------


def _null_aware_sum(frame: pl.DataFrame, parts: list[str], name: str) -> pl.DataFrame:
    present = pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int32) for c in parts])
    summed = pl.sum_horizontal([pl.col(c).fill_null(0.0) for c in parts])
    return frame.with_columns(pl.when(present > 0).then(summed).otherwise(None).alias(name))


def with_cre_rollup(frame: pl.DataFrame) -> pl.DataFrame:
    """``loans_cre_total`` from its RC-C classes.

    Unlike the XBRL panel this needs no judgement about whether a bank's
    disclosure is complete: the classes are fixed lines on the form and every
    filer reports all of them, so the sum is the CRE book by definition rather
    than by inference.  Nonfarm nonresidential is taken as one line where the
    frame carries it that way, which is what makes the total continuous
    across the 2007 split into owner-occupied and investor.
    """
    if all(c in frame.columns for c in CRE_COMPONENTS):
        parts = list(CRE_COMPONENTS)
    elif all(c in frame.columns for c in CRE_COMPONENTS_SPLIT):
        parts = list(CRE_COMPONENTS_SPLIT)
    else:
        return frame
    return _null_aware_sum(frame, parts, "loans_cre_total")


def with_groups(frame: pl.DataFrame) -> pl.DataFrame:
    """The commercial / consumer rollups the EDGAR panel also carries."""
    for name, parts in mdrm.DERIVED_LOAN_GROUPS.items():
        if name == "loans_cre_total":
            continue
        have = [c for c in parts if c in frame.columns]
        if not have:
            continue
        frame = _null_aware_sum(frame, have, name)
    return frame


def with_nco(frame: pl.DataFrame) -> pl.DataFrame:
    """Net charge-offs, from gross charge-offs less recoveries.

    The Call Report reports both sides separately and never a net figure, so
    unlike the XBRL panel there is no tag to prefer and no fallback to arrange:
    NCO is always the subtraction.  ``nco_cre_total`` is then the sum over the
    CRE classes, which is what gives ``nco_rate_cre`` its numerator.
    """
    exprs = []
    for column in frame.columns:
        if not column.startswith("charge_offs_"):
            continue
        suffix = column.removeprefix("charge_offs_")
        recoveries = f"recoveries_{suffix}"
        if recoveries not in frame.columns:
            continue
        exprs.append((pl.col(column) - pl.col(recoveries)).alias(f"nco_{suffix}"))
    if exprs:
        frame = frame.with_columns(exprs)
    cre = [c for c in ("nco_cre_nonfarm_nonres", "nco_construction", "nco_multifamily") if c in frame.columns]
    if len(cre) == 3:
        frame = _null_aware_sum(frame, cre, "nco_cre_total")
    return frame


def with_npa(frame: pl.DataFrame) -> pl.DataFrame:
    """NPAs = nonaccrual + OREO, with a missing nonaccrual staying missing."""
    if "nonaccrual_total" not in frame.columns:
        return frame
    oreo = pl.col("oreo").fill_null(0.0) if "oreo" in frame.columns else pl.lit(0.0)
    return frame.with_columns(
        pl.when(pl.col("nonaccrual_total").is_null())
        .then(None)
        .otherwise(pl.col("nonaccrual_total") + oreo)
        .alias("npa_total")
    )


def with_stress_inputs(frame: pl.DataFrame) -> pl.DataFrame:
    """The composite balances a stress model is calibrated on.

    Each is a plain sum or difference of form lines, null where any required
    input is null, and each exists because the 2001-2012 window is where it
    earns its keep:

    ``noncurrent_total``
        90+ days past due and still accruing, plus nonaccrual -- the FDIC's
        "noncurrent" definition, and the series that led nonaccrual by two to
        three quarters through 2007.
    ``ppnr``
        Net interest income plus noninterest income less noninterest expense:
        pre-provision net revenue, the quantity the supervisory stress test
        projects and the one the provision is absorbed by.
    ``tangible_common_equity`` / ``tangible_assets``
        Equity less preferred stock and intangibles, over assets less
        intangibles.  In 2008-2009 the regulatory ratios said the large banks
        were well capitalised and the market priced on this instead; TARP
        preferred is deducted because it was not loss-absorbing in the way
        common was.
    ``wholesale_funding``
        Fed funds purchased, repos and other borrowed money -- the funding
        that ran in 2008.
    ``commitments_total``
        Every unused commitment line on RC-L: the exposure that can be drawn
        into a stressed balance sheet.
    ``securities_afs_unrealized`` / ``securities_htm_unrealized``
        Fair value less amortised cost.  The 2023 failures, but also 2008.
    ``texas_ratio``
        Noncurrent loans plus OREO over tangible equity plus the allowance,
        in percent: the 1980s Texas rule of thumb that a bank above 100 is
        likely to fail, which held up remarkably well in 2009-2011.
    """
    have = set(frame.columns)
    cols: list[pl.Expr] = []

    def both(*names: str) -> bool:
        return all(n in have for n in names)

    if both("pd_dpd_90_plus", "nonaccrual_total"):
        cols.append(
            pl.when(pl.col("pd_dpd_90_plus").is_null() | pl.col("nonaccrual_total").is_null())
            .then(None)
            .otherwise(pl.col("pd_dpd_90_plus") + pl.col("nonaccrual_total"))
            .alias("noncurrent_total")
        )
    if both("net_interest_income", "noninterest_income", "noninterest_expense"):
        cols.append(
            (
                pl.col("net_interest_income")
                + pl.col("noninterest_income")
                - pl.col("noninterest_expense")
            ).alias("ppnr")
        )
    if both("intangibles_total", "goodwill"):
        cols.append((pl.col("intangibles_total") - pl.col("goodwill")).alias("intangibles_other"))
    if both("equity", "intangibles_total"):
        preferred = pl.col("preferred_stock").fill_null(0.0) if "preferred_stock" in have else pl.lit(0.0)
        cols.append((pl.col("equity") - pl.col("intangibles_total")).alias("tangible_equity"))
        cols.append((pl.col("equity") - pl.col("intangibles_total") - preferred).alias("tangible_common_equity"))
    if both("assets", "intangibles_total"):
        cols.append((pl.col("assets") - pl.col("intangibles_total")).alias("tangible_assets"))
    if both("fed_funds_repo_purchased", "borrowings_other"):
        cols.append(
            pl.when(pl.col("fed_funds_repo_purchased").is_null() & pl.col("borrowings_other").is_null())
            .then(None)
            .otherwise(pl.col("fed_funds_repo_purchased").fill_null(0.0) + pl.col("borrowings_other").fill_null(0.0))
            .alias("wholesale_funding")
        )
    if both("securities_afs", "securities_afs_amortized_cost"):
        cols.append((pl.col("securities_afs") - pl.col("securities_afs_amortized_cost")).alias("securities_afs_unrealized"))
    if both("securities_htm", "securities_htm_fair_value"):
        cols.append((pl.col("securities_htm_fair_value") - pl.col("securities_htm")).alias("securities_htm_unrealized"))
    if cols:
        frame = frame.with_columns(cols)
    commitments = [c for c in mdrm.COMMITMENT_COMPONENTS if c in frame.columns]
    if commitments:
        frame = _null_aware_sum(frame, commitments, "commitments_total")
    if all(c in frame.columns for c in ("noncurrent_total", "tangible_equity", "acl_total")):
        oreo = pl.col("oreo").fill_null(0.0) if "oreo" in frame.columns else pl.lit(0.0)
        denominator = pl.col("tangible_equity") + pl.col("acl_total").fill_null(0.0)
        frame = frame.with_columns(
            pl.when(pl.col("noncurrent_total").is_null() | (denominator <= 0))
            .then(None)
            .otherwise((pl.col("noncurrent_total") + oreo) / denominator * 100)
            .alias("texas_ratio")
        )
    return frame


def with_mix_coverage(frame: pl.DataFrame) -> pl.DataFrame:
    """Leaf categories over total loans, on the RC-C basis.

    Kept for comparability with the EDGAR panel, where it is the main quality
    signal.  Here it should sit at 100 by construction and a departure means
    something specific -- see ``rcc_residual_pct``, which measures the same
    thing in the units that make the cause visible.
    """
    leaves = [c for c in MIX_LEAF_CATEGORIES if c in frame.columns]
    if not leaves or "loans_rcc_total" not in frame.columns:
        return frame
    summed = pl.sum_horizontal([pl.col(c).fill_null(0.0) for c in leaves])
    if "loans_unearned_income" in frame.columns:
        summed = summed - pl.col("loans_unearned_income").fill_null(0.0)
    return frame.with_columns(
        pl.when(pl.col("loans_rcc_total").abs() > 0)
        .then(summed / pl.col("loans_rcc_total") * 100)
        .otherwise(None)
        .alias("mix_coverage_pct")
    )


def add_basis(frame: pl.DataFrame) -> pl.DataFrame:
    """``incurred`` before CECL adoption, ``cecl`` from it."""
    if "period" not in frame.columns:
        return frame
    return frame.with_columns(
        pl.when(pl.col("period") >= pl.lit(config.CECL_ADOPTION_DEFAULT))
        .then(pl.lit("cecl"))
        .otherwise(pl.lit("incurred"))
        .alias("basis")
    )


def _apply_ratios(frame: pl.DataFrame, ratios: tuple[RatioDef, ...]) -> pl.DataFrame:
    have = set(frame.columns)
    exprs = []
    for ratio in ratios:
        if ratio.numerator not in have or ratio.denominator not in have:
            continue
        num = pl.col(ratio.numerator).cast(pl.Float64)
        den = pl.col(ratio.denominator).cast(pl.Float64)
        value = pl.when(den.abs() > 0).then(num / den).otherwise(None)
        if ratio.annualize:
            value = value * 4.0
        exprs.append((value * 100.0).alias(ratio.name))
    return frame.with_columns(exprs) if exprs else frame


def add_ratios(frame: pl.DataFrame) -> pl.DataFrame:
    """The EDGAR panel's ratio definitions, applied to Call Report inputs.

    Imported from :mod:`bankqtr_db.variables` rather than restated, so the two
    panels cannot drift apart on what ``nco_rate`` means.  A ratio whose inputs
    this source does not carry is simply not produced.  :data:`CALL_RATIOS`
    adds the ones only this source has the inputs for.
    """
    frame = _apply_ratios(frame, RATIOS)
    return _apply_ratios(frame, CALL_RATIOS)


def add_capital_ratios(frame: pl.DataFrame) -> pl.DataFrame:
    """Capital over risk-weighted assets, under each regime's own definitions.

    Not in the EDGAR panel at all: the XBRL path has no reliable capital tags,
    while the Call Report states every component on Schedule RC-R.  The Basel
    III columns are null before 2015Q1 and the ``_basel1`` columns null from
    it; they are not spliced, because a Tier 1 ratio that silently changes
    definition at 2015 is worse than two honestly labelled series.

    ``tier1_leverage_ratio`` *is* spliced -- Tier 1 capital under whichever
    regime applies, over RC-K average assets -- because leverage is the one
    ratio whose denominator did not change, and because it is the capital
    ratio that still carried information in 2008.  The 2015 step in the
    numerator's definition is documented in the build info.
    """
    exprs = []
    if "risk_weighted_assets" in frame.columns:
        rwa = pl.col("risk_weighted_assets")
        for name, column in (
            ("cet1_ratio", "cet1_capital"),
            ("tier1_ratio", "tier1_capital"),
            ("total_capital_ratio", "total_capital"),
        ):
            if column in frame.columns:
                exprs.append(
                    pl.when(rwa.abs() > 0).then(pl.col(column) / rwa * 100).otherwise(None).alias(name)
                )
    if "risk_weighted_assets_basel1" in frame.columns:
        rwa = pl.col("risk_weighted_assets_basel1")
        for name, column in (
            ("tier1_ratio_basel1", "tier1_capital_basel1"),
            ("total_capital_ratio_basel1", "total_capital_basel1"),
        ):
            if column in frame.columns:
                exprs.append(
                    pl.when(rwa.abs() > 0).then(pl.col(column) / rwa * 100).otherwise(None).alias(name)
                )
    if "assets_average" in frame.columns:
        tier1 = [c for c in ("tier1_capital", "tier1_capital_basel1") if c in frame.columns]
        if tier1:
            numerator = pl.coalesce([pl.col(c) for c in tier1])
            avg = pl.col("assets_average")
            exprs.append(
                pl.when(avg.abs() > 0).then(numerator / avg * 100).otherwise(None).alias("tier1_leverage_ratio")
            )
    return frame.with_columns(exprs) if exprs else frame


def add_growth(frame: pl.DataFrame) -> pl.DataFrame:
    """QoQ annualised and YoY growth on the balances that matter."""
    targets = [
        c
        for c in (
            "loans_total",
            "loans_cre_total",
            "loans_construction",
            "loans_ci",
            "loans_multifamily",
            "deposits",
        )
        if c in frame.columns
    ]
    if not targets:
        return frame
    frame = frame.sort(["ticker", "period"])
    for column in targets:
        prev = pl.col(column).shift(1).over("ticker")
        prev4 = pl.col(column).shift(4).over("ticker")
        frame = frame.with_columns(
            pl.when(prev.abs() > 0)
            .then(((pl.col(column) / prev) ** 4 - 1) * 100)
            .otherwise(None)
            .alias(f"{column}_qoq_ann_pct"),
            pl.when(prev4.abs() > 0)
            .then((pl.col(column) / prev4 - 1) * 100)
            .otherwise(None)
            .alias(f"{column}_yoy_pct"),
        )
    return frame


def add_derived(frame: pl.DataFrame) -> pl.DataFrame:
    frame = with_cre_rollup(frame)
    frame = with_groups(frame)
    frame = with_nco(frame)
    frame = with_npa(frame)
    frame = with_stress_inputs(frame)
    frame = add_partition_check(frame)
    frame = with_mix_coverage(frame)
    frame = add_basis(frame)
    frame = add_ratios(frame)
    frame = add_capital_ratios(frame)
    return add_growth(frame)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def build(
    periods: list[str],
    holdings: tuple[config.Holding, ...],
    *,
    lineages: dict[str, lineage_mod.Lineage] | None = None,
    derived: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build the charter-level and holding-company panels.

    Returns ``(charters, holdings)``.  Flows are quarterized on the charter
    frame before the rollup, and every derived column is computed after it, so
    a ratio is the ratio of the organisation's summed inputs rather than a sum
    of its charters' ratios.  ``lineages`` (from :func:`lineage.resolve_all`)
    extends each organisation back through its predecessors; without it the
    panel is the 2026 RSSDs' own subtrees, as before.
    """
    mapping = universe_filers(periods, holdings, lineages)
    if mapping.is_empty():
        return pl.DataFrame(), pl.DataFrame()

    wanted = set(mapping["rssd"].to_list())
    charters, roster = charter_frame(periods, wanted)
    if charters.is_empty():
        return pl.DataFrame(), pl.DataFrame()

    pooled = lineage_mod.pooled_events(wanted) if lineages else None
    charters = quarterize(charters, pooled)
    charters = add_rcn_totals(charters)
    if not roster.is_empty():
        charters = charters.join(
            roster.select("rssd", "period", "name", "form"), on=["rssd", "period"], how="left"
        )
    charter_panel = mapping.join(charters, on=["rssd", "period"], how="inner")
    if derived:
        charter_panel = add_partition_check(charter_panel)

    holding_panel = roll_up(charters.drop("name", "form", strict=False), mapping)
    if holding_panel.is_empty():
        return charter_panel.sort(["ticker", "rssd", "period"]), holding_panel

    gaps = unfiled_depositories(periods, holdings, lineages)
    holding_panel = holding_panel.join(gaps, on=["ticker", "period"], how="left").with_columns(
        pl.col("n_insured_not_filing").fill_null(0)
    )
    if derived:
        holding_panel = add_derived(holding_panel)
        holding_panel = _order(holding_panel)
    return charter_panel.sort(["ticker", "rssd", "period"]), holding_panel


def _order(frame: pl.DataFrame) -> pl.DataFrame:
    front = [
        c
        for c in (
            "bank", "ticker", "rssd", "period", "basis", "n_charters", "charters",
            "has_predecessor", "predecessor_count", "n_predecessor_charters",
            "predecessors", "predecessor_failed", "n_insured_not_filing",
            "insured_not_filing", "rcn_total_built", "n_flow_resets",
        )
        if c in frame.columns
    ]
    rest = sorted(c for c in frame.columns if c not in front)
    return frame.select([*front, *rest]).with_columns(
        pl.col("period")
        .map_elements(quarter_label, return_dtype=pl.Utf8)
        .alias("quarter")
    )
