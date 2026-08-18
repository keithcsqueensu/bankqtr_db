"""XBRL extraction: companyfacts (consolidated) and instance documents (dimensional).

Two tiers, because they answer different questions:

``companyfacts_long``
    One call per bank, every tag the bank has ever reported, already
    period-aligned by the SEC.  Cheap and complete -- but **undimensioned**.
    It can tell you total loans and total net charge-offs; it cannot tell you
    the CRE share, because every bank tags CRE with the *same* element as
    total loans and separates them only by dimension member.

``dimensional_long``
    Parses each filing's own instance document, keeping the dimension members
    that companyfacts discards.  This is where portfolio mix, criticized and
    classified balances, and delinquency buckets come from.

Both return the same long-format schema so downstream code treats them
uniformly, with ``source`` marking provenance.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Iterable
from typing import Any

import polars as pl

from . import edgar, filings, instance, taxonomy
from .config import Bank, universe
from .filings import Filing

log = logging.getLogger(__name__)

# Long-format schema shared by every extraction path.
LONG_SCHEMA: dict[str, Any] = {
    "bank": pl.String,
    "ticker": pl.String,
    "cik": pl.String,
    "namespace": pl.String,
    "tag": pl.String,
    "period": pl.Date,  # period end (the panel key)
    "period_start": pl.Date,  # null for instants
    "instant": pl.Boolean,
    "value": pl.Float64,
    "unit": pl.String,
    "form": pl.String,
    "fy": pl.Int32,
    "fp": pl.String,
    "accn": pl.String,
    "filed": pl.Date,
    "source": pl.String,
    # Dimension columns (null on the companyfacts path).
    "segment": pl.String,
    "loan_class": pl.String,
    "credit_quality": pl.String,
    "past_due": pl.String,
    "other_dims": pl.String,
    "dim_axes": pl.String,
    "n_dims": pl.Int32,
}


def empty_long() -> pl.DataFrame:
    return pl.DataFrame(schema=LONG_SCHEMA)


# --------------------------------------------------------------------------
# Tier 1: company facts
# --------------------------------------------------------------------------


def companyfacts_long(
    bank: Bank,
    *,
    tags: Iterable[str] | None = None,
    since: dt.date | None = None,
    forms: tuple[str, ...] = ("10-K", "10-Q"),
) -> pl.DataFrame:
    """Flatten one bank's companyfacts JSON into the long schema.

    ``tags=None`` keeps every tag the bank reports, which is the right default
    for a coverage survey; pass an explicit set for a production pull.
    """
    try:
        payload = edgar.company_facts(bank.cik)
    except Exception as exc:  # noqa: BLE001 - a missing filer must not abort a run
        log.warning("companyfacts unavailable for %s: %s", bank.ticker, exc)
        return empty_long()

    wanted = set(tags) if tags is not None else None
    rows: list[dict[str, Any]] = []

    for namespace, tag_map in payload.get("facts", {}).items():
        for tag, detail in tag_map.items():
            if wanted is not None and tag not in wanted:
                continue
            for unit, items in detail.get("units", {}).items():
                for item in items:
                    form = item.get("form", "")
                    if forms and form.removesuffix("/A") not in forms:
                        continue
                    end = _date(item.get("end"))
                    if end is None or (since and end < since):
                        continue
                    start = _date(item.get("start"))
                    rows.append(
                        {
                            "bank": bank.name,
                            "ticker": bank.ticker,
                            "cik": bank.cik,
                            "namespace": namespace,
                            "tag": tag,
                            "period": end,
                            "period_start": start,
                            "instant": start is None,
                            "value": float(item["val"]),
                            "unit": unit,
                            "form": form,
                            "fy": item.get("fy"),
                            "fp": item.get("fp"),
                            "accn": item.get("accn"),
                            "filed": _date(item.get("filed")),
                            "source": "companyfacts",
                            "segment": None,
                            "loan_class": None,
                            "credit_quality": None,
                            "past_due": None,
                            "other_dims": "",
                            "dim_axes": "",
                            "n_dims": 0,
                        }
                    )

    if not rows:
        return empty_long()
    return pl.DataFrame(rows, schema_overrides=LONG_SCHEMA, infer_schema_length=None)


def _date(s: str | None) -> dt.date | None:
    if not s:
        return None
    try:
        return dt.date.fromisoformat(s[:10])
    except ValueError, TypeError:
        return None


# --------------------------------------------------------------------------
# Tier 2: instance documents
# --------------------------------------------------------------------------


def dimensional_long(
    bank: Bank,
    *,
    since: dt.date | None = None,
    forms: tuple[str, ...] = ("10-K", "10-Q"),
    tags: Iterable[str] | None = None,
    filing_list: list[Filing] | None = None,
) -> pl.DataFrame:
    """Parse every cached filing instance for a bank into the long schema."""
    fs = filing_list or filings.list_filings(bank, since=since, forms=forms)
    wanted = set(tags) if tags is not None else None
    frames: list[pl.DataFrame] = []

    for f in fs:
        try:
            df = instance.parse_instance(f)
        except Exception as exc:  # noqa: BLE001
            log.warning("instance parse failed %s %s: %s", f.ticker, f.accn, exc)
            continue
        if df.is_empty():
            continue
        if wanted is not None:
            df = df.filter(pl.col("tag").is_in(list(wanted)))
            if df.is_empty():
                continue
        frames.append(
            df.select(
                pl.col("bank"),
                pl.col("ticker"),
                pl.col("cik"),
                pl.col("namespace"),
                pl.col("tag"),
                pl.col("period_end").alias("period"),
                pl.col("period_start"),
                pl.col("instant"),
                pl.col("value"),
                pl.col("unit"),
                pl.col("form"),
                pl.lit(None, dtype=pl.Int32).alias("fy"),
                pl.lit(None, dtype=pl.String).alias("fp"),
                pl.col("accn"),
                pl.col("filing_date").alias("filed"),
                pl.lit("instance", dtype=pl.String).alias("source"),
                pl.col("segment"),
                pl.col("loan_class"),
                pl.col("credit_quality"),
                pl.col("past_due"),
                pl.col("other_dims"),
                pl.col("dim_axes"),
                pl.col("n_dims"),
            )
        )

    if not frames:
        return empty_long()
    return pl.concat(frames, how="vertical_relaxed").cast(LONG_SCHEMA)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Canonical category attachment
# --------------------------------------------------------------------------


def attach_categories(df: pl.DataFrame) -> pl.DataFrame:
    """Add canonical ``loan_cat`` / ``cq_cat`` / ``pd_cat`` / ``scope`` columns.

    A fact can carry a loan class *and* a portfolio segment; the class is more
    specific, so it wins.  Facts whose member is a measurement-basis qualifier
    (held-for-sale, unfunded) get ``scope`` set and no loan category, which
    keeps them out of the held-for-investment portfolio mix.
    """
    if df.is_empty():
        return df.with_columns(
            pl.lit(None, dtype=pl.String).alias(c)
            for c in ("loan_cat", "cq_cat", "pd_cat", "scope")
        )

    tickers = df["ticker"].to_list()
    classes = df["loan_class"].to_list()
    segments = df["segment"].to_list()
    cqs = df["credit_quality"].to_list()
    pds = df["past_due"].to_list()
    others = df["other_dims"].to_list()

    loan_cat: list[str | None] = []
    scope: list[str | None] = []
    for t, cls, seg, other in zip(tickers, classes, segments, others, strict=True):
        member = cls or seg
        sc = taxonomy.scope_of(cls) or taxonomy.scope_of(seg)
        if sc is None and other:
            for part in other.split(";"):
                _, _, val = part.partition("=")
                sc = taxonomy.scope_of(val)
                if sc:
                    break
        scope.append(sc)
        loan_cat.append(taxonomy.loan_category(member, t) if member else None)

    cq_cat = [taxonomy.credit_quality(m, t) for m, t in zip(cqs, tickers, strict=True)]
    pd_cat = [taxonomy.past_due(m, t) for m, t in zip(pds, tickers, strict=True)]

    return df.with_columns(
        pl.Series("loan_cat", loan_cat, dtype=pl.String),
        pl.Series("cq_cat", cq_cat, dtype=pl.String),
        pl.Series("pd_cat", pd_cat, dtype=pl.String),
        pl.Series("scope", scope, dtype=pl.String),
    )


# --------------------------------------------------------------------------
# Deduplication across filings
# --------------------------------------------------------------------------

FACT_KEY = (
    "ticker",
    "tag",
    "namespace",
    "period",
    "period_start",
    "segment",
    "loan_class",
    "credit_quality",
    "past_due",
    "other_dims",
    "dim_axes",
)


def deduplicate(df: pl.DataFrame, *, flag_restatements: bool = True) -> pl.DataFrame:
    """Collapse the same fact reported in several filings to one row.

    A quarter appears as a comparative in later filings, so most facts are
    reported three to five times.  The most recently filed version wins.
    Where versions disagree by more than rounding, ``restated`` is set -- that
    is a signal worth keeping, not noise to suppress.
    """
    if df.is_empty():
        return df.with_columns(
            pl.lit(False).alias("restated"),
            pl.lit(1, dtype=pl.UInt32).alias("n_versions"),
        )

    keys = [k for k in FACT_KEY if k in df.columns]
    stats = df.group_by(keys).agg(
        pl.len().alias("n_versions"),
        pl.col("value").min().alias("_vmin"),
        pl.col("value").max().alias("_vmax"),
    )
    latest = (
        df.sort(["filed", "accn"], descending=[True, True], nulls_last=True)
        .group_by(keys, maintain_order=True)
        .first()
    )
    out = latest.join(stats, on=keys, how="left")
    if flag_restatements:
        span = (pl.col("_vmax") - pl.col("_vmin")).abs()
        scale = pl.max_horizontal(pl.col("_vmax").abs(), pl.lit(1.0))
        out = out.with_columns(((span / scale) > 0.005).alias("restated"))
    else:
        out = out.with_columns(pl.lit(False).alias("restated"))
    return out.drop("_vmin", "_vmax")


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def extract_bank(
    bank: Bank,
    *,
    since: dt.date,
    tags: Iterable[str] | None = None,
    use_instances: bool = True,
) -> pl.DataFrame:
    """Full XBRL extraction for one bank: companyfacts + instance documents."""
    parts = [companyfacts_long(bank, tags=tags, since=since)]
    if use_instances:
        parts.append(dimensional_long(bank, since=since, tags=tags))
    frames = [p for p in parts if not p.is_empty()]
    if not frames:
        return attach_categories(empty_long())
    return attach_categories(pl.concat(frames, how="vertical_relaxed"))


def extract_universe(
    *,
    since: dt.date,
    banks: tuple[Bank, ...] | None = None,
    tags: Iterable[str] | None = None,
    use_instances: bool = True,
) -> pl.DataFrame:
    frames = []
    for bank in banks or universe():
        df = extract_bank(bank, since=since, tags=tags, use_instances=use_instances)
        log.info("%s: %d facts", bank.ticker, df.height)
        frames.append(df)
    frames = [f for f in frames if not f.is_empty()]
    if not frames:
        return attach_categories(empty_long())
    return pl.concat(frames, how="vertical_relaxed")


def relevant_tags() -> set[str]:
    """Every tag named by a variable definition."""
    from .variables import ALL_VARS

    return {t for v in ALL_VARS for t in v.tags}
