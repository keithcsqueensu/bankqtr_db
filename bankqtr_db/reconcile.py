"""Reconciliation: prefer XBRL, fall back to HTML, and make every gap visible.

The reconciliation layer exists to answer three questions a peer benchmark has
to answer honestly:

* Where did each number come from?  Every filled column gets a ``*_source``
  companion, so a chart can be traced back to a tagged fact or a parsed table.
* What scale is the HTML in?  ``read_html`` cannot see the "(in millions)"
  header, so a parsed 61,468 might be dollars, thousands or millions.  Scale is
  inferred by comparing against the XBRL value for the same bank-quarter and
  applied per filing, and any value whose scale could not be established stays
  out of the panel rather than entering it a thousand times too small.
* What is still missing, and is the gap real?  A custody bank with no CRE book
  is not a coverage failure; a regional bank with no CRE column is.  The
  coverage report separates the two.
"""

from __future__ import annotations

import logging

import polars as pl

from .config import THIN_LOAN_BOOK, Bank, universe
from .variables import RATIOS

log = logging.getLogger(__name__)

PANEL_KEY = ["bank", "ticker", "cik", "quarter", "period"]

# Plausible reporting scales in a filing table.
SCALES = (1.0, 1e3, 1e6, 1e9)


# --------------------------------------------------------------------------
# Scale inference
# --------------------------------------------------------------------------


def infer_html_scale(html: pl.DataFrame, panel: pl.DataFrame) -> pl.DataFrame:
    """Infer the reporting multiplier for each parsed filing.

    Uses every variable where the same bank-quarter also has an XBRL value:
    the ratio of the two clusters tightly around one of 1, 1e3, 1e6, so the
    modal ratio across a filing's rows is a robust estimate even when a few
    rows were mismatched.
    """
    if html.is_empty() or panel.is_empty():
        return html.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("scale"),
            pl.lit(0, dtype=pl.UInt32).alias("scale_evidence"),
        )

    value_cols = [c for c in panel.columns if c not in PANEL_KEY]
    reference = panel.unpivot(
        index=["ticker", "period"],
        on=[c for c in value_cols if panel.schema[c] in (pl.Float64, pl.Int64)],
        variable_name="variable",
        value_name="xbrl_value",
    ).filter(pl.col("xbrl_value").is_not_null() & (pl.col("xbrl_value").abs() > 0))

    joined = html.join(
        reference, on=["ticker", "period", "variable"], how="inner"
    ).filter(pl.col("value").abs() > 0)
    if joined.is_empty():
        return html.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("scale"),
            pl.lit(0, dtype=pl.UInt32).alias("scale_evidence"),
        )

    # Snap each observed ratio to the nearest plausible scale.
    ratio = (pl.col("xbrl_value").abs() / pl.col("value").abs()).alias("_ratio")
    snapped = joined.with_columns(ratio).with_columns(
        pl.coalesce(
            [
                pl.when((pl.col("_ratio") / s).is_between(0.98, 1.02))
                .then(pl.lit(s))
                .otherwise(None)
                for s in SCALES
            ]
        ).alias("_snap")
    )

    per_filing = (
        snapped.filter(pl.col("_snap").is_not_null())
        .group_by(["ticker", "accn", "_snap"])
        .agg(pl.len().alias("_n"))
        .sort(["ticker", "accn", "_n"], descending=[False, False, True])
        .group_by(["ticker", "accn"], maintain_order=True)
        .agg(
            pl.col("_snap").first().alias("scale"),
            pl.col("_n").first().alias("scale_evidence"),
        )
    )
    return html.join(per_filing, on=["ticker", "accn"], how="left").with_columns(
        pl.col("scale_evidence").fill_null(0).cast(pl.UInt32)
    )


def scaled_html(html: pl.DataFrame, *, min_evidence: int = 2) -> pl.DataFrame:
    """Apply the inferred scale, dropping rows whose scale is unproven.

    Guessing a default of "millions" would be wrong for the banks that report
    in thousands, and a silently mis-scaled peer is worse than a missing one.
    """
    if html.is_empty() or "scale" not in html.columns:
        return html
    return (
        html.filter(
            pl.col("scale").is_not_null() & (pl.col("scale_evidence") >= min_evidence)
        )
        .with_columns((pl.col("value") * pl.col("scale")).alias("value"))
        .with_columns(pl.lit(True).alias("scale_resolved"))
    )


# --------------------------------------------------------------------------
# Merge
# --------------------------------------------------------------------------


def reconcile(
    panel: pl.DataFrame,
    html: pl.DataFrame | None = None,
    *,
    min_evidence: int = 2,
) -> pl.DataFrame:
    """Fill XBRL gaps from HTML and record provenance per column."""
    if panel.is_empty():
        return panel

    numeric = [
        c
        for c in panel.columns
        if c not in PANEL_KEY and panel.schema[c] in (pl.Float64, pl.Int64)
    ]
    out = panel.with_columns(
        [
            pl.when(pl.col(c).is_not_null())
            .then(pl.lit("xbrl"))
            .otherwise(None)
            .alias(f"{c}__source")
            for c in numeric
        ]
    )
    if html is None or html.is_empty():
        return out

    resolved = scaled_html(infer_html_scale(html, panel), min_evidence=min_evidence)
    if resolved.is_empty():
        log.info("no HTML rows survived scale resolution")
        return out

    wide = _widen(resolved)
    incoming = [c for c in wide.columns if c not in ("ticker", "period")]
    wide = _reject_implausible(panel, wide, incoming)
    return fill_gaps(out, wide, source="html")


# --------------------------------------------------------------------------
# Filling
# --------------------------------------------------------------------------


def _widen(resolved: pl.DataFrame) -> pl.DataFrame:
    """One row per bank-quarter, one column per variable, best reading kept."""
    return (
        resolved.sort(["confidence", "filed"], descending=True, nulls_last=True)
        .unique(
            subset=["ticker", "period", "variable"], keep="first", maintain_order=True
        )
        .pivot(on="variable", index=["ticker", "period"], values="value")
    )


def fill_gaps(
    panel: pl.DataFrame,
    wide: pl.DataFrame,
    *,
    source: str,
    suffix: str = "__fill",
) -> pl.DataFrame:
    """Fill null cells from ``wide`` and tag each filled cell with ``source``.

    Adds columns the panel has never had.  Both fallbacks need that and for the
    same reason: the disclosures they exist to recover are the ones no filing
    tags, so there is no XBRL column waiting for them.  Office CRE is the case
    in point -- Bank of America's instance document carries no office member
    and no property-type axis at all, so the value has nowhere to land unless
    the column is created here.

    Only nulls are written, so an earlier and more authoritative source is
    never overwritten -- XBRL first, then HTML, then the IR supplements.
    """
    incoming = [c for c in wide.columns if c not in ("ticker", "period")]
    if not incoming:
        return panel

    out = panel
    missing = [c for c in incoming if c not in out.columns]
    if missing:
        out = out.with_columns(
            [pl.lit(None, dtype=pl.Float64).alias(c) for c in missing]
            + [
                pl.lit(None, dtype=pl.String).alias(f"{c}__source")
                for c in missing
                if f"{c}__source" not in out.columns
            ]
        )

    out = out.join(
        wide.select(["ticker", "period", *incoming]).rename(
            {c: f"{c}{suffix}" for c in incoming}
        ),
        on=["ticker", "period"],
        how="left",
    )
    for col in incoming:
        source_col = f"{col}__source"
        if source_col not in out.columns:
            out = out.with_columns(pl.lit(None, dtype=pl.String).alias(source_col))
        out = out.with_columns(
            pl.when(pl.col(col).is_null() & pl.col(f"{col}{suffix}").is_not_null())
            .then(pl.lit(source))
            .otherwise(pl.col(source_col))
            .alias(source_col),
            pl.col(col).fill_null(pl.col(f"{col}{suffix}")).alias(col),
        )
    return out.drop([f"{c}{suffix}" for c in incoming])


# --------------------------------------------------------------------------
# IR supplements
# --------------------------------------------------------------------------


def resolve_ir_scale(
    ir_rows: pl.DataFrame, panel: pl.DataFrame, *, min_evidence: int = 2
) -> pl.DataFrame:
    """Settle the reporting multiplier for IR rows, then apply it.

    Same idea as :func:`infer_html_scale`, with one addition that matters: an
    investor supplement usually *states* its unit inside the table it applies
    to ("In billions", "$ amounts in millions"), and Regions' office-CRE slide
    states it in its own caption.  A declared unit is direct evidence and beats
    a ratio inferred against XBRL, so it wins where present -- and it is the
    only thing that can scale office CRE or leveraged lending at all, since
    those have no XBRL counterpart to compare against.

    Rows left with no declared unit and no usable inference are dropped, on the
    same reasoning as the HTML path: a peer that is wrong by a factor of a
    thousand is worse than a peer that is missing.
    """
    if ir_rows.is_empty():
        return ir_rows

    rows = ir_rows
    if "declared_scale" not in rows.columns:
        rows = rows.with_columns(pl.lit(None, dtype=pl.Float64).alias("declared_scale"))

    inferred = infer_html_scale(rows, panel)
    # Read the declared unit back off the frame as a *column*, not as a Series
    # captured beforehand: ``infer_html_scale`` ends in a join, and a join gives
    # no row-order guarantee, so a captured Series could line a bank's declared
    # unit up against another bank's row.
    declared = pl.col("declared_scale")
    resolved = inferred.with_columns(
        pl.when(declared.is_not_null())
        .then(declared)
        .otherwise(pl.col("scale"))
        .alias("scale"),
        pl.when(declared.is_not_null())
        .then(pl.lit(min_evidence, dtype=pl.UInt32))
        .otherwise(pl.col("scale_evidence"))
        .alias("scale_evidence"),
        pl.when(declared.is_not_null())
        .then(pl.lit("declared"))
        .otherwise(pl.lit("inferred"))
        .alias("scale_basis"),
    )
    return scaled_html(resolved, min_evidence=min_evidence)


def merge_ir(
    panel: pl.DataFrame,
    ir_rows: pl.DataFrame | None,
    *,
    min_evidence: int = 2,
) -> pl.DataFrame:
    """Fill remaining gaps from IR supplements and tag their provenance.

    Runs after :func:`reconcile`, so XBRL wins, then the HTML fallback, then
    this.  Unlike those two it may also *add* columns: office CRE and leveraged
    lending exist in no filing, so there is no XBRL column for them to fill.
    """
    if panel.is_empty() or ir_rows is None or ir_rows.is_empty():
        return panel

    resolved = resolve_ir_scale(ir_rows, panel, min_evidence=min_evidence)
    if resolved.is_empty():
        log.info("no IR rows survived scale resolution")
        return panel

    wide = _widen(resolved)
    incoming = [c for c in wide.columns if c not in ("ticker", "period")]
    if not incoming:
        return panel

    # Refuse the impossible before it is written, not after: a slice of the
    # loan book cannot exceed the loan book, and a phrase pattern once read US
    # Bancorp's leveraged lending as $1.18tn against a $381bn book.
    wide = _reject_implausible(panel, wide, incoming)
    return fill_gaps(panel, wide, source="ir_supplement")


# Slice of the loan book, so it cannot exceed the loan book.  ``loans_total``
# is excluded because it *is* the book.
_SUBSET_OF_LOANS = ("loans_", "acl_", "nonaccrual_", "cq_", "pd_")

# A little headroom: a supplement rounds to a different precision than XBRL,
# and unfunded commitments occasionally ride along in a disclosed total.
_SUBSET_TOLERANCE = 1.05


def _reject_implausible(
    panel: pl.DataFrame, wide: pl.DataFrame, incoming: list[str]
) -> pl.DataFrame:
    """Blank incoming values that cannot be true beside the known loan book.

    The last line of defence, and it earns its place: a loose phrase pattern
    read US Bancorp's leveraged lending as $1.18 *trillion* against a $381bn
    loan book, and nothing upstream of here would have caught it -- the value
    scaled cleanly, it was the only reading of that slide, and the panel
    rendered it without complaint.

    Only the incoming value is discarded; the column keeps whatever XBRL or
    HTML already put there, and the gap stays visible in the coverage report.
    """
    if "loans_total" not in panel.columns:
        return wide

    checked = [
        c for c in incoming if c != "loans_total" and c.startswith(_SUBSET_OF_LOANS)
    ]
    if not checked:
        return wide

    book = panel.select(
        "ticker", "period", pl.col("loans_total").alias("_book")
    ).unique(subset=["ticker", "period"], keep="first")
    ceiling = pl.col("_book").cast(pl.Float64) * _SUBSET_TOLERANCE
    return (
        wide.join(book, on=["ticker", "period"], how="left")
        .with_columns(
            [
                pl.when(
                    pl.col("_book").is_not_null()
                    & (pl.col("_book") > 0)
                    & (pl.col(c).abs() > ceiling)
                )
                .then(None)
                .otherwise(pl.col(c))
                .alias(c)
                for c in checked
            ]
        )
        .drop("_book")
    )


def refresh_ratios(panel: pl.DataFrame) -> pl.DataFrame:
    """Compute ratios whose inputs only arrived after the panel was built.

    ``panel.add_derived`` runs on the XBRL frame, before the HTML and IR
    fallbacks have filled anything, so a CRE balance recovered from a
    supplement would leave ``cre_pct`` null.  Only null ratio cells are
    written, so an already-computed ratio is never disturbed.
    """
    if panel.is_empty():
        return panel

    have = set(panel.columns)
    out = panel
    for ratio in RATIOS:
        if ratio.numerator not in have or ratio.denominator not in have:
            continue
        num = pl.col(ratio.numerator).cast(pl.Float64)
        den = pl.col(ratio.denominator).cast(pl.Float64)
        value = pl.when(den.abs() > 0).then(num / den).otherwise(None)
        if ratio.annualize:
            value = value * 4.0
        value = value * 100.0

        if ratio.name not in out.columns:
            out = out.with_columns(
                value.alias(ratio.name),
                pl.when(value.is_not_null())
                .then(pl.lit("derived_ir"))
                .otherwise(None)
                .alias(f"{ratio.name}__source"),
            )
            continue
        source = f"{ratio.name}__source"
        if source not in out.columns:
            out = out.with_columns(pl.lit(None, dtype=pl.String).alias(source))
        out = out.with_columns(
            pl.when(pl.col(ratio.name).is_null() & value.is_not_null())
            .then(pl.lit("derived_ir"))
            .otherwise(pl.col(source))
            .alias(source),
            pl.col(ratio.name).fill_null(value).alias(ratio.name),
        )
    return out


# --------------------------------------------------------------------------
# Coverage and quality
# --------------------------------------------------------------------------

# Variables that are legitimately absent for a given business model, so their
# absence is reported as "not applicable" rather than as a gap.
NOT_APPLICABLE: dict[str, tuple[str, ...]] = {
    "custody": (
        "loans_cre_total",
        "loans_construction",
        "loans_multifamily",
        "loans_credit_card",
        "cre_pct",
        "construction_pct",
    ),
    "broker": (
        "loans_cre_total",
        "loans_construction",
        "loans_multifamily",
        "cre_pct",
        "construction_pct",
    ),
    "card_consumer": ("loans_construction", "loans_multifamily", "construction_pct"),
    "ibank": ("loans_construction", "loans_multifamily", "construction_pct"),
}

# Office CRE and leveraged lending are only meaningful where there is a CRE or
# corporate lending book to disclose.  A custody bank, a broker and a card
# issuer have none, so an empty office column there is structure rather than a
# gap and must not inflate the work queue in ``panel_gaps.csv``.
_IR_EXPOSURE_COLUMNS = (
    "loans_office_cre",
    "nonaccrual_office_cre",
    "acl_office_cre",
    "nco_office_cre",
    "office_cre_pct",
    "office_cre_share_of_cre",
    "office_nonaccrual_ratio",
    "office_reserve_coverage",
)
for _category in ("custody", "broker", "card_consumer"):
    NOT_APPLICABLE[_category] = NOT_APPLICABLE.get(_category, ()) + _IR_EXPOSURE_COLUMNS
del _category


def coverage_report(
    panel: pl.DataFrame, *, banks: tuple[Bank, ...] | None = None
) -> pl.DataFrame:
    """Per bank and variable: how many quarters are populated, and by what."""
    if panel.is_empty():
        return pl.DataFrame()

    by_ticker = {b.ticker: b for b in (banks or universe())}
    numeric = [
        c
        for c in panel.columns
        if c not in PANEL_KEY
        and not c.endswith("__source")
        and panel.schema[c] in (pl.Float64, pl.Int64)
    ]

    long = panel.unpivot(
        index=["bank", "ticker", "quarter"],
        on=numeric,
        variable_name="variable",
        value_name="value",
    )
    agg = (
        long.group_by(["bank", "ticker", "variable"])
        .agg(
            pl.len().alias("n_quarters"),
            pl.col("value").is_not_null().sum().alias("n_present"),
        )
        .with_columns(
            (pl.col("n_present") / pl.col("n_quarters") * 100)
            .round(1)
            .alias("pct_present")
        )
    )

    categories = [
        by_ticker[t].category if t in by_ticker else "" for t in agg["ticker"]
    ]
    agg = agg.with_columns(pl.Series("category", categories, dtype=pl.String))

    na_flag = [
        var in NOT_APPLICABLE.get(cat, ())
        for var, cat in zip(agg["variable"], agg["category"], strict=True)
    ]
    return (
        agg.with_columns(pl.Series("not_applicable", na_flag, dtype=pl.Boolean))
        .with_columns(
            pl.when(pl.col("not_applicable"))
            .then(pl.lit("n/a"))
            .when(pl.col("pct_present") == 0)
            .then(pl.lit("missing"))
            .when(pl.col("pct_present") < 60)
            .then(pl.lit("sparse"))
            .otherwise(pl.lit("ok"))
            .alias("status")
        )
        .sort(["variable", "ticker"])
    )


def quality_flags(panel: pl.DataFrame) -> pl.DataFrame:
    """Row-level sanity checks -- the things that should never be true."""
    if panel.is_empty():
        return pl.DataFrame()

    have = set(panel.columns)
    checks: list[pl.Expr] = []

    def add(name: str, expr: pl.Expr, needs: tuple[str, ...]) -> None:
        if all(c in have for c in needs):
            checks.append(expr.alias(name))

    add("neg_loans", pl.col("loans_total") < 0, ("loans_total",))
    add("neg_acl", pl.col("acl_total") < 0, ("acl_total",))
    add(
        "components_exceed_total",
        pl.sum_horizontal(
            [
                pl.col(c).fill_null(0.0)
                for c in ("loans_commercial_total", "loans_consumer_total")
                if c in have
            ]
        )
        > pl.col("loans_total") * 1.02,
        ("loans_total", "loans_commercial_total"),
    )
    add(
        "construction_exceeds_cre",
        pl.col("loans_construction") > pl.col("loans_cre_total") * 1.02,
        ("loans_construction", "loans_cre_total"),
    )
    add(
        "cre_exceeds_commercial",
        pl.col("loans_cre_total") > pl.col("loans_commercial_total") * 1.02,
        ("loans_cre_total", "loans_commercial_total"),
    )
    add(
        "criticized_exceeds_loans",
        pl.col("cq_criticized") > pl.col("loans_total"),
        ("cq_criticized", "loans_total"),
    )
    # Bounds are wide on purpose: Schwab's pledged-asset lines carry a genuine
    # ~0.08% reserve and Synchrony's card book a genuine ~10.5%, so a narrow
    # band flags real disclosures rather than errors.
    add(
        "reserve_coverage_implausible",
        (pl.col("reserve_coverage") < 0.02) | (pl.col("reserve_coverage") > 20),
        ("reserve_coverage",),
    )
    add(
        "mix_incoherent",
        (pl.col("mix_coverage_pct") > 110) | (pl.col("mix_coverage_pct") < 50),
        ("mix_coverage_pct",),
    )
    add(
        "nco_rate_implausible",
        (pl.col("nco_rate") < -2) | (pl.col("nco_rate") > 25),
        ("nco_rate",),
    )
    # Every percentage-of-loans ratio should sit inside a sane band.
    for ratio in RATIOS:
        if ratio.name.endswith("_pct") and ratio.name in have:
            add(
                f"{ratio.name}_out_of_range",
                (pl.col(ratio.name) < 0) | (pl.col(ratio.name) > 100),
                (ratio.name,),
            )

    if not checks:
        return pl.DataFrame()

    flagged = panel.select([*PANEL_KEY, *checks])
    flag_cols = [c for c in flagged.columns if c not in PANEL_KEY]
    return (
        flagged.with_columns(
            pl.sum_horizontal(
                [pl.col(c).cast(pl.Int32).fill_null(0) for c in flag_cols]
            ).alias("n_flags")
        )
        .filter(pl.col("n_flags") > 0)
        .sort(["n_flags", "ticker", "period"], descending=[True, False, False])
    )


def gap_summary(coverage: pl.DataFrame) -> pl.DataFrame:
    """Variables ranked by how many banks lack them -- the work queue."""
    if coverage.is_empty():
        return pl.DataFrame()
    return (
        coverage.filter(~pl.col("not_applicable"))
        .group_by("variable")
        .agg(
            pl.len().alias("n_banks"),
            (pl.col("status") == "ok").sum().alias("n_ok"),
            (pl.col("status") == "sparse").sum().alias("n_sparse"),
            (pl.col("status") == "missing").sum().alias("n_missing"),
            pl.col("pct_present").mean().round(1).alias("mean_pct"),
        )
        .with_columns(
            (pl.col("n_ok") / pl.col("n_banks") * 100).round(1).alias("pct_banks_ok")
        )
        .sort(["pct_banks_ok", "variable"])
    )


def thin_book_tickers(banks: tuple[Bank, ...] | None = None) -> set[str]:
    return {b.ticker for b in (banks or universe()) if b.category in THIN_LOAN_BOOK}
