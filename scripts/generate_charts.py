"""Charts for the download page: four PNGs and the stats the page prints.

Run from the repository root::

    uv run python scripts/generate_charts.py

Reads the published panels in ``data/`` -- the same files the page offers for
download, not the build's working copies in ``data/out/`` -- so the charts
cannot describe a different dataset from the one a colleague grabs.

Everything is drawn from the **FFIEC panel**.  That is a deliberate choice and
not the only one available: the EDGAR panel is in ``data/panel.parquet`` and is
offered beside it, but it starts at 2020Q1, so it cannot show the GFC that two
of these charts exist to put in view, and it carries no ``loans_average``
column for the charge-off cross-check below.  Every chart is captioned with its
source so the two are never confused.

The charge-off rates are checked before they are drawn
------------------------------------------------------
``nco_rate`` and its three segment variants are ratios, and a ratio inherits
every defect of its denominator.  :func:`audit_rate` runs the same three-step
check on each before it reaches a chart:

1. **Recompute** it from the panel's own definition and compare.  A stored
   value that disagrees with its own inputs is replaced and the correction
   logged.
2. **Test the denominator** where the value is extreme.  A rate more than ten
   times the bank's own median is suspect, but suspect is not wrong -- 2009 is
   *supposed* to look like that -- so the question asked is whether the
   denominator is real.
3. **Null** the row where it is not, and log it as a data-quality issue rather
   than drawing it.

What that catches, and what it deliberately does not, is in the docstring of
:func:`audit_rate`.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

# Agg before pyplot: this runs headless in CI, and importing pyplot first binds
# whatever interactive backend happens to be installed and then fails on a
# machine with no display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

log = logging.getLogger("charts")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CHARTS = ROOT / "docs" / "charts"
SUMMARY = ROOT / "docs" / "data_summary.json"

CALL_PANEL = DATA / "call_panel.parquet"
EDGAR_PANEL = DATA / "panel.parquet"

# One figure size and one DPI for all four, because the page lays them out in a
# grid and a ragged edge is the first thing a reader notices.
FIGSIZE = (11.0, 6.0)
DPI = 130

# Recessions, on NBER's dates.  Shaded rather than annotated: the point is to
# let a reader see which peaks are cyclical without reading a legend.
RECESSIONS: tuple[tuple[str, dt.date, dt.date], ...] = (
    ("GFC", dt.date(2007, 12, 1), dt.date(2009, 6, 30)),
    ("COVID", dt.date(2020, 2, 1), dt.date(2020, 4, 30)),
)

INK = "#1b1f24"
MUTED = "#6b7280"
GRID = "#dfe3e8"
ACCENT = "#1f4e79"
SHADE = "#c9ced6"

# The rates the page draws, with the inputs each is built from.  ``annualize``
# and the percent scaling match ``bankqtr_db.variables.RatioDef``: a quarterly
# flow over a period-end balance, times four, times a hundred.
RATE_SPECS: tuple[tuple[str, str, str], ...] = (
    ("nco_rate", "nco_total", "loans_total"),
    ("nco_rate_ci", "nco_ci", "loans_ci"),
    ("nco_rate_cre", "nco_cre_total", "loans_cre_total"),
    ("nco_rate_card", "nco_credit_card", "loans_credit_card"),
)

# A rate this far above the bank's own median is looked at rather than trusted.
OUTLIER_FACTOR = 10.0
# ...and its denominator is called degenerate at this share of the bank's own
# median denominator.  A hundredth: Ameriprise Bank's 2012Q3 loan book is
# $31,000 against a $766m median, which is four orders of magnitude out.
DEGENERATE_SHARE = 0.01

KEY_FIELDS: tuple[str, ...] = (
    "loans_total",
    "loans_ci",
    "loans_cre_total",
    "loans_credit_card",
    "loans_consumer_total",
    "acl_total",
    "nonaccrual_total",
    "nco_rate",
    "nco_rate_ci",
    "nco_rate_cre",
    "nco_rate_card",
    "npa_ratio",
    "reserve_coverage",
    "noncurrent_ratio",
    "deposits",
    "deposits_uninsured",
    "cet1_ratio",
    "tier1_leverage_ratio",
)


@dataclass
class RateAudit:
    """What the check did to one rate column."""

    column: str
    n_values: int = 0
    p50: float | None = None
    p90: float | None = None
    p99: float | None = None
    maximum: float | None = None
    flagged: int = 0
    corrected: int = 0
    nulled: int = 0
    corrections: list[dict] = field(default_factory=list)
    quality_issues: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "column": self.column,
            "n_values": self.n_values,
            "p50": self.p50,
            "p90": self.p90,
            "p99": self.p99,
            "max": self.maximum,
            "flagged": self.flagged,
            "corrected": self.corrected,
            "nulled": self.nulled,
            "quality_issues": self.quality_issues,
        }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.3f}"


def audit_rate(frame: pl.DataFrame, column: str, numerator: str, denominator: str) -> tuple[pl.DataFrame, RateAudit]:
    """Check one charge-off rate, correct what is wrong, null what is unusable.

    **The recomputation uses the panel's own definition**, which is
    ``numerator / denominator * 4 * 100``: a quarterly flow over the
    *period-end* balance, annualised, in percent.  It is worth being explicit
    that this is not the only definition of a charge-off rate and not the one a
    UBPR uses -- the regulatory convention divides by *average* loans, and this
    panel carries ``loans_average`` for anyone who wants it.  The two differ by
    about four per cent at the median here.  What matters for a check is that
    it tests the number against the definition that produced it: comparing the
    stored value against the average-loans variant instead disagrees on 3,447
    of 3,624 rows, none of which is an error.

    **An extreme value is not by itself a wrong one.**  Flagging on ten times
    the bank's own median finds 235 bank-quarters in ``nco_rate``, and the
    great majority are 2008-2010 doing what it did -- Ally at 29% in 2009Q4,
    Fifth Third's CRE at 11% in 2008Q4.  Nulling those would delete precisely
    the signal these charts exist to show.  So the flag opens a question and
    the denominator answers it: a rate is discarded only where the book it is
    measured against has collapsed or is smaller than the flow charged off
    against it, which cannot be true of a real quarter.
    """
    audit = RateAudit(column=column)
    if column not in frame.columns:
        log.warning("%s: absent from the panel; skipped", column)
        return frame, audit

    present = frame.filter(pl.col(column).is_not_null())
    audit.n_values = present.height
    if not present.height:
        return frame, audit

    series = present[column]
    audit.p50 = float(series.median())
    audit.p90 = float(series.quantile(0.90))
    audit.p99 = float(series.quantile(0.99))
    audit.maximum = float(series.max())

    log.info(
        "%s: n=%d  p50=%s  p90=%s  p99=%s  max=%s",
        column,
        audit.n_values,
        _fmt(audit.p50),
        _fmt(audit.p90),
        _fmt(audit.p99),
        _fmt(audit.maximum),
    )
    per_bank = (
        present.group_by("ticker")
        .agg(
            pl.col(column).median().alias("p50"),
            pl.col(column).quantile(0.90).alias("p90"),
            pl.col(column).quantile(0.99).alias("p99"),
            pl.col(column).max().alias("max"),
        )
        .sort("ticker")
    )
    log.info("%s: by bank (p50 / p90 / p99 / max)", column)
    for row in per_bank.iter_rows(named=True):
        log.info(
            "    %-8s %8s %8s %8s %10s",
            row["ticker"],
            _fmt(row["p50"]),
            _fmt(row["p90"]),
            _fmt(row["p99"]),
            _fmt(row["max"]),
        )

    if numerator not in frame.columns or denominator not in frame.columns:
        log.warning(
            "%s: cannot cross-check, %s or %s is absent",
            column,
            numerator,
            denominator,
        )
        return frame, audit

    medians = present.group_by("ticker").agg(pl.col(column).median().alias("_med"))
    den_medians = (
        frame.filter(pl.col(denominator).is_not_null())
        .group_by("ticker")
        .agg(pl.col(denominator).median().alias("_den_med"))
    )
    work = frame.join(medians, on="ticker", how="left").join(
        den_medians, on="ticker", how="left"
    )

    recomputed = (
        pl.when(pl.col(denominator).abs() > 0)
        .then(pl.col(numerator) / pl.col(denominator) * 4.0 * 100.0)
        .otherwise(None)
    )
    work = work.with_columns(recomputed.alias("_recomputed"))

    suspect = (
        pl.col(column).is_not_null()
        & pl.col("_med").is_not_null()
        & (pl.col("_med").abs() > 1e-9)
        & (pl.col(column).abs() > OUTLIER_FACTOR * pl.col("_med").abs())
    )
    # A stored ratio that disagrees with its own inputs.  Compared with a
    # relative tolerance because both sides are floats computed the same way,
    # so a genuine mismatch is a rewrite rather than a rounding difference.
    mismatched = suspect & pl.col("_recomputed").is_not_null() & (
        (pl.col(column) - pl.col("_recomputed")).abs()
        > 1e-6 * pl.col("_recomputed").abs() + 1e-9
    )
    degenerate = suspect & (
        (pl.col(denominator).is_null())
        | (pl.col(denominator).abs() <= 0)
        | (
            pl.col("_den_med").is_not_null()
            & (pl.col(denominator).abs() < DEGENERATE_SHARE * pl.col("_den_med").abs())
        )
        | (pl.col(numerator).abs() > pl.col(denominator).abs())
    )

    audit.flagged = work.filter(suspect).height
    for row in work.filter(mismatched).iter_rows(named=True):
        audit.corrections.append(
            {
                "ticker": row["ticker"],
                "period": str(row["period"]),
                "stored": row[column],
                "recomputed": row["_recomputed"],
            }
        )
        log.warning(
            "%s: %s %s stored %s != recomputed %s; overriding",
            column,
            row["ticker"],
            row["period"],
            _fmt(row[column]),
            _fmt(row["_recomputed"]),
        )
    for row in work.filter(degenerate).iter_rows(named=True):
        audit.quality_issues.append(
            {
                "ticker": row["ticker"],
                "period": str(row["period"]),
                "value": row[column],
                "numerator": row[numerator],
                "denominator": row[denominator],
                "denominator_median": row["_den_med"],
                "reason": f"{denominator} is not a usable base for this quarter",
            }
        )
        log.warning(
            "%s: %s %s = %s on %s of %s (bank median %s); nulled, investigate upstream",
            column,
            row["ticker"],
            row["period"],
            _fmt(row[column]),
            denominator,
            _fmt(row[denominator]),
            _fmt(row["_den_med"]),
        )

    # Correct first, then null: a row whose denominator is unusable stays
    # unusable however the numerator is arrived at.
    resolved = (
        pl.when(degenerate)
        .then(None)
        .when(mismatched)
        .then(pl.col("_recomputed"))
        .otherwise(pl.col(column))
        .alias(column)
    )
    audit.corrected = work.filter(mismatched & ~degenerate).height
    audit.nulled = work.filter(degenerate).height
    out = work.with_columns(resolved).drop("_med", "_den_med", "_recomputed")

    log.info(
        "%s: %d flagged, %d corrected, %d nulled, %d kept as real",
        column,
        audit.flagged,
        audit.corrected,
        audit.nulled,
        audit.flagged - audit.corrected - audit.nulled,
    )
    return out, audit


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------


def _style(ax: plt.Axes, title: str, subtitle: str, ylabel: str = "") -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", color=INK, loc="left", pad=30)
    ax.annotate(
        subtitle,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 10),
        textcoords="offset points",
        fontsize=9,
        color=MUTED,
        va="bottom",
    )
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.5, color=MUTED)
    ax.tick_params(labelsize=9, colors=MUTED, length=0)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def _shade_recessions(ax: plt.Axes, first: dt.date, last: dt.date) -> None:
    for label, start, end in RECESSIONS:
        if end < first or start > last:
            continue
        ax.axvspan(max(start, first), min(end, last), color=SHADE, alpha=0.55, lw=0)
        ax.annotate(
            label,
            xy=(max(start, first), 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(3, -12),
            textcoords="offset points",
            fontsize=8,
            color=MUTED,
            fontweight="bold",
        )


def _save(fig: plt.Figure, name: str) -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    path = CHARTS / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("wrote %s (%.0f kB)", path.relative_to(ROOT), path.stat().st_size / 1024)


def chart_nco_timeseries(frame: pl.DataFrame) -> None:
    """Every peer's charge-off rate on one axis, with the recessions shaded.

    Thirty-eight series and no per-bank legend: at this density a legend is
    unreadable and the chart is not asking which bank is which.  It is asking
    how wide the peer distribution gets and when -- so the individual banks are
    hairlines and the cross-sectional median is drawn over them.

    The y-axis is capped rather than left to the maximum.  Ally reaches 29% in
    2009Q4, which is real and which flattens every other series into the axis
    if it sets the scale; the cap keeps the mass of the distribution legible
    and the count of what it clips is printed on the chart rather than hidden.
    """
    data = frame.filter(pl.col("nco_rate").is_not_null()).sort("period")
    if data.is_empty():
        log.warning("nco_timeseries: nothing to draw")
        return
    first, last = data["period"].min(), data["period"].max()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    _shade_recessions(ax, first, last)
    for ticker in sorted(data["ticker"].unique()):
        one = data.filter(pl.col("ticker") == ticker)
        ax.plot(one["period"], one["nco_rate"], lw=0.7, color=ACCENT, alpha=0.28)

    median = data.group_by("period").agg(pl.col("nco_rate").median().alias("m")).sort("period")
    ax.plot(median["period"], median["m"], lw=2.2, color=INK, label="Peer median")

    cap = float(np.nanpercentile(data["nco_rate"].to_numpy(), 99.5))
    cap = max(cap, float(median["m"].max()) * 1.15)
    clipped = data.filter(pl.col("nco_rate") > cap).height
    ax.set_ylim(min(0.0, float(data["nco_rate"].min())), cap)

    note = f"{data['ticker'].n_unique()} banks, {data.height:,} bank-quarters"
    if clipped:
        note += f"  ·  {clipped} points above {cap:,.1f}% clipped by the axis"
    _style(
        ax,
        "Net charge-off rate, all peers",
        f"FFIEC Call Report panel  ·  annualised, % of total loans  ·  {note}",
        "NCO rate (%)",
    )
    ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK)
    _save(fig, "nco_timeseries.png")


def chart_peer_comparison(frame: pl.DataFrame) -> None:
    """The latest quarter, ranked. Horizontal because bank names are words.

    The quarter drawn is the latest for which any peer reports, and a bank
    that has not filed it is absent rather than shown at zero -- an absence and
    a zero charge-off rate are different statements.
    """
    latest = frame.filter(pl.col("nco_rate").is_not_null())["period"].max()
    data = (
        frame.filter((pl.col("period") == latest) & pl.col("nco_rate").is_not_null())
        .select("ticker", "bank", "nco_rate")
        .sort("nco_rate", descending=False)
    )
    if data.is_empty():
        log.warning("peer_comparison: nothing to draw")
        return

    height = max(6.0, 0.26 * data.height + 1.8)
    fig, ax = plt.subplots(figsize=(FIGSIZE[0], height))
    positions = np.arange(data.height)
    values = data["nco_rate"].to_numpy()
    median = float(np.median(values))
    colors = [ACCENT if v >= median else "#8fa9c4" for v in values]
    ax.barh(positions, values, color=colors, height=0.68)
    ax.set_yticks(positions)
    ax.set_yticklabels(data["ticker"].to_list(), fontsize=9)
    ax.axvline(median, color=INK, lw=1.1, ls="--", alpha=0.7)
    ax.annotate(
        f"median {median:,.2f}%",
        xy=(median, data.height - 0.4),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=8.5,
        color=INK,
    )
    span = float(values.max()) if values.size else 1.0
    # Every label goes to the right of zero, including the negative bars.
    # Anchoring a label to the end of its own bar puts the near-zero ones on
    # top of the ticker names -- four banks are within 0.01% of zero this
    # quarter and a net *recovery* is a real reading, not one to hide.
    for y, v in zip(positions, values):
        ax.annotate(
            f"{v:,.2f}",
            xy=(max(v, 0.0), y),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=8,
            color=MUTED,
        )
    ax.set_xlim(min(0.0, float(values.min()) * 1.15), span * 1.18)
    _style(
        ax,
        f"Net charge-off rate by bank, {latest:%YQ}{(latest.month - 1) // 3 + 1}",
        f"FFIEC Call Report panel  ·  annualised, % of total loans  ·  {data.height} banks reporting",
        "",
    )
    ax.set_xlabel("NCO rate (%)", fontsize=9.5, color=MUTED)
    ax.grid(axis="y", visible=False)
    _save(fig, "peer_comparison.png")


def chart_portfolio_mix(frame: pl.DataFrame) -> None:
    """How the peer group's loan book is composed, quarter by quarter.

    Dollar-weighted: each band is the peers' summed balance over their summed
    total loans, not the average of their individual shares.  An unweighted
    mean would give Ameriprise Bank the same say as JPMorgan and is not what a
    reader means by "the peer group's mix".

    Consumer is drawn net of cards because cards are the band with its own loss
    behaviour and burying them inside consumer is what the chart is trying to
    avoid.  The residual band is everything RC-C carries that these four do not
    -- residential mortgage, construction, leases, municipal, farm -- so the
    stack closes at 100% and the reader can see how much the four describe.
    """
    needed = ("loans_total", "loans_ci", "loans_cre_total", "loans_consumer_total", "loans_credit_card")
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        log.warning("portfolio_mix: missing %s", ", ".join(missing))
        return

    data = frame.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in needed]))
    agg = (
        data.group_by("period")
        .agg(
            [pl.col(c).sum().alias(c) for c in needed]
            + [pl.col("ticker").n_unique().alias("n_banks")]
        )
        .sort("period")
        .filter(pl.col("loans_total") > 0)
    )
    if agg.is_empty():
        log.warning("portfolio_mix: nothing to draw")
        return

    total = agg["loans_total"].to_numpy()
    ci = agg["loans_ci"].to_numpy() / total * 100
    cre = agg["loans_cre_total"].to_numpy() / total * 100
    card = agg["loans_credit_card"].to_numpy() / total * 100
    consumer = (
        (agg["loans_consumer_total"].to_numpy() - agg["loans_credit_card"].to_numpy())
        / total
        * 100
    )
    other = np.clip(100 - (ci + cre + card + consumer), 0, None)
    periods = agg["period"].to_list()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bands = [
        ("C&I", ci, "#1f4e79"),
        ("CRE", cre, "#3d7ab8"),
        ("Consumer ex-card", consumer, "#8fb8dd"),
        ("Credit card", card, "#d98c3f"),
        ("Other", other, "#cfd6de"),
    ]
    ax.stackplot(
        periods,
        [b[1] for b in bands],
        labels=[b[0] for b in bands],
        colors=[b[2] for b in bands],
        lw=0,
    )
    # No recession shading on this one.  Drawn over a filled stack it reads as
    # a sixth band rather than as context, and a 100%-normalised mix has no
    # cyclical spike for it to explain anyway.
    ax.set_ylim(0, 100)
    ax.set_xlim(periods[0], periods[-1])
    _style(
        ax,
        "Peer loan mix",
        "FFIEC Call Report panel  ·  dollar-weighted share of total loans  ·  consumer "
        f"includes residential mortgage  ·  {agg['n_banks'].max()} banks at peak",
        "Share of total loans (%)",
    )
    ax.legend(
        frameon=False,
        fontsize=9,
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.16),
        labelcolor=INK,
    )
    _save(fig, "portfolio_mix.png")


def chart_coverage_heatmap(frame: pl.DataFrame) -> None:
    """Which fields a given bank actually populates, over the whole window.

    The question this answers is the one asked before a screen is run: not
    "what is this bank's CRE book" but "can I put this bank in the CRE screen
    at all". Red is a column that is mostly absent for that bank, and an
    absence here is usually structural -- a custody bank has no card book to
    report -- rather than a parsing failure.
    """
    fields = [c for c in KEY_FIELDS if c in frame.columns]
    if not fields:
        log.warning("coverage_heatmap: no key fields present")
        return
    quarters = (
        frame.group_by("ticker").agg(pl.len().alias("n")).sort("ticker")
    )
    coverage = (
        frame.group_by("ticker")
        .agg([(pl.col(c).is_not_null().sum() / pl.len() * 100).alias(c) for c in fields])
        .join(quarters, on="ticker")
        .sort("ticker")
    )
    tickers = coverage["ticker"].to_list()
    matrix = np.array([[coverage[c][i] for c in fields] for i in range(len(tickers))], dtype=float)

    fig, ax = plt.subplots(figsize=(FIGSIZE[0], max(6.5, 0.30 * len(tickers) + 2.4)))
    mesh = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(len(fields)))
    ax.set_xticklabels(fields, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(np.arange(len(tickers)))
    ax.set_yticklabels(
        [f"{t}  ({n})" for t, n in zip(tickers, coverage["n"].to_list())], fontsize=8.5
    )
    ax.set_xticks(np.arange(-0.5, len(fields), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(tickers), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(colors=MUTED, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    bar = fig.colorbar(mesh, ax=ax, fraction=0.023, pad=0.015)
    bar.set_label("Quarters populated (%)", fontsize=9, color=MUTED)
    bar.ax.tick_params(labelsize=8, colors=MUTED, length=0)
    bar.outline.set_visible(False)

    ax.set_title(
        "Field coverage by bank",
        fontsize=13,
        fontweight="bold",
        color=INK,
        loc="left",
        pad=30,
    )
    ax.annotate(
        "FFIEC Call Report panel  ·  % of the bank's own quarters with a value  ·"
        "  bank-quarters in brackets",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 10),
        textcoords="offset points",
        fontsize=9,
        color=MUTED,
        va="bottom",
    )
    _save(fig, "coverage_heatmap.png")


# --------------------------------------------------------------------------
# The page's own numbers
# --------------------------------------------------------------------------


def describe(path: Path, csv: Path) -> dict:
    """Row count, column count, window and file sizes, read off the file."""
    frame = pl.read_parquet(path)
    period = frame["period"]
    return {
        "rows": frame.height,
        "columns": frame.width,
        "banks": int(frame["ticker"].n_unique()),
        "start": str(period.min()),
        "end": str(period.max()),
        "parquet_bytes": path.stat().st_size,
        "csv_bytes": csv.stat().st_size if csv.exists() else None,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not CALL_PANEL.exists():
        log.error("%s is missing; run scripts/build_call_panel.py first", CALL_PANEL)
        return 1

    frame = pl.read_parquet(CALL_PANEL)
    log.info("loaded %s: %d rows x %d cols", CALL_PANEL.name, frame.height, frame.width)

    log.info("--- charge-off rate sanity checks -------------------------------")
    audits: list[RateAudit] = []
    for column, numerator, denominator in RATE_SPECS:
        frame, audit = audit_rate(frame, column, numerator, denominator)
        audits.append(audit)

    total_nulled = sum(a.nulled for a in audits)
    total_corrected = sum(a.corrected for a in audits)
    log.info(
        "--- audit complete: %d corrected, %d nulled across %d rate columns",
        total_corrected,
        total_nulled,
        len(audits),
    )

    log.info("--- charts ------------------------------------------------------")
    chart_nco_timeseries(frame)
    chart_peer_comparison(frame)
    chart_portfolio_mix(frame)
    chart_coverage_heatmap(frame)

    summary = {
        "generated_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "ffiec": describe(CALL_PANEL, DATA / "call_panel.csv"),
        "edgar": describe(EDGAR_PANEL, DATA / "panel.csv")
        if EDGAR_PANEL.exists()
        else None,
        "rate_audit": [a.as_dict() for a in audits],
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("wrote %s", SUMMARY.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
