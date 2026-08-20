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

Three hazards this handles
--------------------------
**Flows are year-to-date.**  Every ``RIAD`` item accumulates through the year
and resets in Q1, so a quarter is the difference against the previous quarter of
the *same year*.  Differencing happens **per charter, before the rollup**: a
holding company that acquires a bank mid-year would otherwise show that bank's
whole year-to-date as a single quarter's charge-offs.

**The charter set moves.**  Zions ran seven bank charters until 2018 and one
after; Truist is BB&T plus SunTrust from 2019Q4.  The set is resolved per
quarter from the dated NIC graph, so a merger enters the panel when it happened
rather than being back-applied to the whole window.

**A missing quarter is not a zero.**  If the previous quarter is absent the
difference cannot be taken, and the flow is left null rather than being
reported as its own year-to-date figure.
"""

from __future__ import annotations

import datetime as dt
import logging

import polars as pl

from bankqtr_db.variables import RATIOS

from . import cdr, config, mdrm, nic, schedules

log = logging.getLogger(__name__)

PANEL_KEY = ["ticker", "rssd", "period"]

# Leaf categories of the RC-C partition, for the mix.  Unlike the XBRL panel's
# equivalent this really is a partition -- see ``mix_coverage_pct`` below.
MIX_LEAF_CATEGORIES: tuple[str, ...] = tuple(
    spec.name for spec in mdrm.RCC_LOAN_ITEMS
)

CRE_COMPONENTS: tuple[str, ...] = (
    "loans_cre_owner_occupied",
    "loans_cre_investor",
    "loans_construction",
    "loans_multifamily",
)


def quarter_label(period: dt.date) -> str:
    return f"{period.year}Q{(period.month - 1) // 3 + 1}"


# --------------------------------------------------------------------------
# Charter level
# --------------------------------------------------------------------------


def universe_filers(
    periods: list[str],
    holdings: tuple[config.Holding, ...],
) -> pl.DataFrame:
    """Map each quarter's Call Report filers to the holding company above them.

    One row per (ticker, rssd, period).  Resolved per quarter against that
    quarter's own roster and the NIC graph as it stood then, so a charter joins
    and leaves an organisation on the date it actually did.
    """
    rows: list[tuple[str, str, int, int, dt.date]] = []
    for period in periods:
        path = cdr.zip_path(period)
        if not path.exists():
            continue
        roster = schedules.roster(path)
        present = set(roster["rssd"].to_list())
        as_of = cdr.quarter_end(period)
        graph = nic.hierarchy(as_of)
        claimed: dict[int, str] = {}
        for holding in holdings:
            if holding.last_filing is not None and as_of > holding.last_filing:
                continue
            found: set[int] = set()
            for rssd in holding.rssds:
                found.update(
                    nic.call_filers(rssd, present, as_of=as_of, kids=graph)
                )
            for charter in sorted(found):
                # A charter can sit under two tracked parents during a
                # transition.  First claim wins and the collision is logged
                # rather than silently double counting it into both.
                if charter in claimed and claimed[charter] != holding.ticker:
                    log.warning(
                        "%s: charter %d claimed by both %s and %s; kept %s",
                        period,
                        charter,
                        claimed[charter],
                        holding.ticker,
                        claimed[charter],
                    )
                    continue
                claimed[charter] = holding.ticker
                rows.append(
                    (holding.ticker, holding.name, holding.rssd, charter, as_of)
                )
    if not rows:
        return pl.DataFrame(
            schema={
                "ticker": pl.Utf8,
                "bank": pl.Utf8,
                "holding_rssd": pl.Int64,
                "rssd": pl.Int64,
                "period": pl.Date,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "ticker": pl.Utf8,
            "bank": pl.Utf8,
            "holding_rssd": pl.Int64,
            "rssd": pl.Int64,
            "period": pl.Date,
        },
        orient="row",
    ).unique()


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


def quarterize(frame: pl.DataFrame) -> pl.DataFrame:
    """Turn year-to-date flow columns into single-quarter figures.

    A flow is its year-to-date value less the previous quarter's, within the
    same calendar year; Q1 stands as reported.  The previous quarter has to be
    the *immediately* previous one -- a gap in a charter's filing history makes
    the difference meaningless, so it is left null.
    """
    flows = [c for c in frame.columns if c in mdrm.FLOW_COLUMNS]
    if not flows:
        return frame
    frame = frame.sort(["rssd", "period"]).with_columns(
        pl.col("period").dt.year().alias("_year"),
        pl.col("period").dt.quarter().alias("_q"),
    )
    prev_year = pl.col("_year").shift(1).over("rssd")
    prev_q = pl.col("_q").shift(1).over("rssd")
    contiguous = (prev_year == pl.col("_year")) & (prev_q == pl.col("_q") - 1)
    exprs = []
    for column in flows:
        previous = pl.col(column).shift(1).over("rssd")
        exprs.append(
            pl.when(pl.col("_q") == 1)
            .then(pl.col(column))
            .when(contiguous)
            .then(pl.col(column) - previous)
            .otherwise(None)
            .alias(column)
        )
    return frame.with_columns(exprs).drop("_year", "_q")


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

# Columns that are summed across charters.  Everything else is derived after.
def _summable(frame: pl.DataFrame) -> list[str]:
    skip = {"rssd", "period", "ticker", "bank", "holding_rssd", "form"}
    return [
        c
        for c in frame.columns
        if c not in skip and frame.schema[c] in (pl.Float64, pl.Int64)
    ]


def roll_up(charters: pl.DataFrame, mapping: pl.DataFrame) -> pl.DataFrame:
    """Sum charter rows into one row per holding company per quarter.

    Summing is right for every column here because they are all dollar amounts
    on the same basis, and the charters are disjoint by construction -- each is
    claimed by exactly one holding company per quarter.  Ratios are **not**
    summed; they are recomputed from the summed inputs afterwards, since the
    average of two ratios is not the ratio of the sums.
    """
    if charters.is_empty() or mapping.is_empty():
        return pl.DataFrame()
    joined = mapping.join(charters, on=["rssd", "period"], how="inner")
    if joined.is_empty():
        return pl.DataFrame()
    value_cols = _summable(joined)
    aggs = [pl.col(c).sum().alias(c) for c in value_cols]
    aggs.append(pl.col("rssd").n_unique().alias("n_charters"))
    aggs.append(
        pl.col("rssd").sort().cast(pl.Utf8).str.join(";").alias("charters")
    )
    out = joined.group_by(["ticker", "bank", "holding_rssd", "period"]).agg(aggs)
    return out.rename({"holding_rssd": "rssd"}).sort(["ticker", "period"])


# --------------------------------------------------------------------------
# Derived columns
# --------------------------------------------------------------------------


def with_cre_rollup(frame: pl.DataFrame) -> pl.DataFrame:
    """``loans_cre_total`` from its four RC-C classes.

    Unlike the XBRL panel this needs no judgement about whether a bank's
    disclosure is complete: the four classes are fixed lines on the form and
    every filer reports all of them, so the sum is the CRE book by definition
    rather than by inference.
    """
    parts = [c for c in CRE_COMPONENTS if c in frame.columns]
    if len(parts) < len(CRE_COMPONENTS):
        return frame
    present = pl.sum_horizontal(
        [pl.col(c).is_not_null().cast(pl.Int32) for c in parts]
    )
    summed = pl.sum_horizontal([pl.col(c).fill_null(0.0) for c in parts])
    return frame.with_columns(
        pl.when(present > 0).then(summed).otherwise(None).alias("loans_cre_total")
    )


def with_groups(frame: pl.DataFrame) -> pl.DataFrame:
    """The commercial / consumer rollups the EDGAR panel also carries."""
    for name, parts in mdrm.DERIVED_LOAN_GROUPS.items():
        if name == "loans_cre_total":
            continue
        have = [c for c in parts if c in frame.columns]
        if not have:
            continue
        present = pl.sum_horizontal(
            [pl.col(c).is_not_null().cast(pl.Int32) for c in have]
        )
        summed = pl.sum_horizontal([pl.col(c).fill_null(0.0) for c in have])
        frame = frame.with_columns(
            pl.when(present > 0).then(summed).otherwise(None).alias(name)
        )
    return frame


def with_nco(frame: pl.DataFrame) -> pl.DataFrame:
    """Net charge-offs, from gross charge-offs less recoveries.

    The Call Report reports both sides separately and never a net figure, so
    unlike the XBRL panel there is no tag to prefer and no fallback to arrange:
    NCO is always the subtraction.
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
    return frame.with_columns(exprs) if exprs else frame


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


def add_ratios(frame: pl.DataFrame) -> pl.DataFrame:
    """The EDGAR panel's ratio definitions, applied to Call Report inputs.

    Imported from :mod:`bankqtr_db.variables` rather than restated, so the two
    panels cannot drift apart on what ``nco_rate`` means.  A ratio whose inputs
    this source does not carry is simply not produced.
    """
    have = set(frame.columns)
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
    return frame.with_columns(exprs) if exprs else frame


def add_capital_ratios(frame: pl.DataFrame) -> pl.DataFrame:
    """CET1 and total capital over risk-weighted assets.

    Not in the EDGAR panel at all: the XBRL path has no reliable capital tags,
    while the Call Report states every component on Schedule RC-R.
    """
    if "risk_weighted_assets" not in frame.columns:
        return frame
    rwa = pl.col("risk_weighted_assets")
    exprs = []
    for name, column in (
        ("cet1_ratio", "cet1_capital"),
        ("tier1_ratio", "tier1_capital"),
        ("total_capital_ratio", "total_capital"),
    ):
        if column in frame.columns:
            exprs.append(
                pl.when(rwa.abs() > 0)
                .then(pl.col(column) / rwa * 100)
                .otherwise(None)
                .alias(name)
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
    derived: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build the charter-level and holding-company panels.

    Returns ``(charters, holdings)``.  Flows are quarterized on the charter
    frame before the rollup, and every derived column is computed after it, so
    a ratio is the ratio of the organisation's summed inputs rather than a sum
    of its charters' ratios.
    """
    mapping = universe_filers(periods, holdings)
    if mapping.is_empty():
        return pl.DataFrame(), pl.DataFrame()

    wanted = set(mapping["rssd"].to_list())
    charters, roster = charter_frame(periods, wanted)
    if charters.is_empty():
        return pl.DataFrame(), pl.DataFrame()

    charters = quarterize(charters)
    if not roster.is_empty():
        charters = charters.join(
            roster.select("rssd", "period", "name", "form"), on=["rssd", "period"], how="left"
        )
    charter_panel = mapping.join(charters, on=["rssd", "period"], how="inner")
    if derived:
        charter_panel = add_partition_check(charter_panel)

    holding_panel = roll_up(charters.drop("name", "form", strict=False), mapping)
    if derived and not holding_panel.is_empty():
        holding_panel = add_derived(holding_panel)
        holding_panel = _order(holding_panel)
    return charter_panel.sort(["ticker", "rssd", "period"]), holding_panel


def _order(frame: pl.DataFrame) -> pl.DataFrame:
    front = [
        c
        for c in ("bank", "ticker", "rssd", "period", "basis", "n_charters", "charters")
        if c in frame.columns
    ]
    rest = sorted(c for c in frame.columns if c not in front)
    return frame.select([*front, *rest]).with_columns(
        pl.col("period")
        .map_elements(quarter_label, return_dtype=pl.Utf8)
        .alias("quarter")
    )
