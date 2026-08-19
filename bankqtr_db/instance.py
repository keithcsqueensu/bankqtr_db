"""Parse XBRL instance documents into a dimensional long-format Polars frame.

The companyfacts API returns only *undimensioned* facts.  Every portfolio-mix
and credit-quality variable a bank discloses (CRE vs C&I, criticized/classified,
past-due buckets) is tagged on the same element and separated purely by
dimension members, so those series are invisible to companyfacts.  This module
reads the filing's own instance document, which carries the full context set.
"""

from __future__ import annotations

import datetime as dt
import gzip
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import polars as pl
from lxml import etree

from . import edgar
from .config import DATA
from .filings import Filing

log = logging.getLogger(__name__)

XBRLI = "http://www.xbrl.org/2003/instance"
XBRLDI = "http://xbrl.org/2006/xbrldi"
XSI = "http://www.w3.org/2001/XMLSchema-instance"

Q_CONTEXT = f"{{{XBRLI}}}context"
Q_UNIT = f"{{{XBRLI}}}unit"
Q_PERIOD = f"{{{XBRLI}}}period"
Q_INSTANT = f"{{{XBRLI}}}instant"
Q_START = f"{{{XBRLI}}}startDate"
Q_END = f"{{{XBRLI}}}endDate"
Q_MEASURE = f"{{{XBRLI}}}measure"
Q_EXPLICIT = f"{{{XBRLDI}}}explicitMember"
Q_TYPED = f"{{{XBRLDI}}}typedMember"

# Namespaces whose facts we keep.  Everything else (link, xsi, ix) is metadata.
FACT_NS_PREFIXES = ("http://fasb.org/us-gaap/", "http://fasb.org/srt/")

CACHE = DATA / "raw" / "instances"


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------


def _instance_cache_path(filing: Filing) -> Path:
    return CACHE / f"{filing.ticker}_{filing.accn_nodash}.xml.gz"


def _find_instance_name(filing: Filing) -> str | None:
    """Pick the ``*_htm.xml`` inline-XBRL instance from the filing folder."""
    index = edgar.get(f"{filing.folder_url}/index.json").json()
    names = [item["name"] for item in index["directory"]["item"]]
    candidates = [n for n in names if n.endswith("_htm.xml")]
    if candidates:
        return candidates[0]
    # Pre-2019 filings ship a standalone instance instead of inline XBRL.
    skip = ("_cal.xml", "_def.xml", "_lab.xml", "_pre.xml", "FilingSummary.xml")
    plain = [
        n for n in names if n.endswith(".xml") and not any(n.endswith(s) for s in skip)
    ]
    return plain[0] if plain else None


def ensure_cached(filing: Filing) -> bool:
    """Put a filing's instance on disk if it is not already there.

    Callers that go on to parse in a worker pool use this first, serially: the
    rate limiter in :mod:`edgar` spaces requests within one process, so letting
    workers fetch would multiply the request rate by the worker count.  It
    reads nothing when the document is already cached, which is what keeps a
    warm rebuild cheap.
    """
    if _instance_cache_path(filing).exists():
        return True
    return fetch_instance(filing) is not None


def fetch_instance(filing: Filing, *, refresh: bool = False) -> bytes | None:
    """Return raw instance XML for a filing, caching it gzipped on disk."""
    path = _instance_cache_path(filing)
    if path.exists() and not refresh:
        with gzip.open(path, "rb") as fh:
            return fh.read()

    name = _find_instance_name(filing)
    if name is None:
        log.warning("no instance document for %s %s", filing.ticker, filing.accn)
        return None

    content = edgar.get(f"{filing.folder_url}/{name}").content
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wb") as fh:
        fh.write(content)
    tmp.replace(path)
    return content


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def _local(name: str | None) -> str:
    """Strip a namespace URI or prefix, returning the local name."""
    if not name:
        return ""
    if "}" in name:
        name = name.rsplit("}", 1)[1]
    if ":" in name:
        name = name.rsplit(":", 1)[1]
    return name


def _parse_contexts(root: etree._Element) -> dict[str, dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
    for ctx in root.iter(Q_CONTEXT):
        period = ctx.find(Q_PERIOD)
        start = end = instant = None
        if period is not None:
            inst_el = period.find(Q_INSTANT)
            if inst_el is not None:
                instant = inst_el.text
            else:
                s, e = period.find(Q_START), period.find(Q_END)
                start = s.text if s is not None else None
                end = e.text if e is not None else None

        dims: dict[str, str] = {}
        for member in ctx.iter(Q_EXPLICIT):
            dims[_local(member.get("dimension"))] = _local(
                member.text.strip() if member.text else ""
            )
        for member in ctx.iter(Q_TYPED):
            child = next(iter(member), None)
            if child is not None:
                dims[_local(member.get("dimension"))] = (child.text or "").strip()

        contexts[ctx.get("id")] = {
            "start": start,
            "end": end or instant,
            "instant": instant is not None,
            "dims": dims,
        }
    return contexts


def _parse_units(root: etree._Element) -> dict[str, str]:
    units: dict[str, str] = {}
    for unit in root.iter(Q_UNIT):
        measures = [_local(m.text) for m in unit.iter(Q_MEASURE) if m.text]
        # Ratios and per-share units come through as numerator/denominator.
        units[unit.get("id")] = "/".join(measures) if measures else ""
    return units


def _as_date(s: str | None) -> dt.date | None:
    if not s:
        return None
    try:
        return dt.date.fromisoformat(s.strip()[:10])
    except ValueError:
        return None


def _iter_facts(
    root: etree._Element,
    contexts: dict[str, dict[str, Any]],
    units: dict[str, str],
    filing: Filing,
    keep_namespaces: tuple[str, ...],
) -> Iterator[dict[str, Any]]:
    for el in root.iter():
        if not isinstance(el.tag, str) or "}" not in el.tag:
            continue
        uri, local = el.tag[1:].split("}", 1)
        ctx_ref = el.get("contextRef")
        if ctx_ref is None:
            continue

        if uri.startswith(FACT_NS_PREFIXES):
            namespace = "us-gaap" if "us-gaap" in uri else "srt"
        elif keep_namespaces and uri in keep_namespaces:
            namespace = "custom"
        else:
            continue

        ctx = contexts.get(ctx_ref)
        if ctx is None:
            continue
        if el.get(f"{{{XSI}}}nil") == "true":
            continue

        raw = (el.text or "").strip()
        if not raw:
            continue
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue  # textual fact (policy blocks, member labels)

        sign = -1.0 if el.get("sign") == "-" else 1.0
        scale = el.get("scale")
        if scale is not None:
            try:
                value *= 10 ** int(scale)
            except ValueError:
                pass

        dims = ctx["dims"]
        yield {
            "bank": filing.bank,
            "ticker": filing.ticker,
            "cik": filing.cik,
            "accn": filing.accn,
            "form": filing.form,
            "filing_date": filing.filing_date,
            "report_date": filing.report_date,
            "namespace": namespace,
            "tag": local,
            "period_start": _as_date(ctx["start"]),
            "period_end": _as_date(ctx["end"]),
            "instant": ctx["instant"],
            "value": value * sign,
            "unit": units.get(el.get("unitRef") or "", ""),
            "decimals": el.get("decimals"),
            "n_dims": len(dims),
            "dims": dims,
        }


_SCHEMA = {
    "bank": pl.String,
    "ticker": pl.String,
    "cik": pl.String,
    "accn": pl.String,
    "form": pl.String,
    "filing_date": pl.Date,
    "report_date": pl.Date,
    "namespace": pl.String,
    "tag": pl.String,
    "period_start": pl.Date,
    "period_end": pl.Date,
    "instant": pl.Boolean,
    "value": pl.Float64,
    "unit": pl.String,
    "decimals": pl.String,
    "n_dims": pl.Int32,
}

# Axes lifted into their own columns because the panel keys on them.
#
# The axis names changed in the 2018 taxonomy.  A pre-2018 filing spells the
# same cut differently, and an unaliased axis lands in ``other_dims``, which
# leaves the mix, risk-rating and delinquency columns null for exactly the
# banks and years that matter -- Wells Fargo's FY2013 instance carries no
# class-of-financing-receivable axis at all, so without
# ``AccountsNotesLoansAndFinancingReceivableByReceivableTypeAxis`` its entire
# loan breakdown is invisible.  Every legacy name below was read off a FY2013
# 10-K instance rather than from the taxonomy documentation; the bank and fact
# count in each comment is what that sweep measured.
#
# Order inside a tuple decides which axis wins when one fact carries several,
# so legacy names go after the modern ones except where the legacy axis is the
# more specific cut (receivable-type beats the catch-all FinancialInstrument).
PROMOTED_AXES = {
    "segment": (
        "FinancingReceivablePortfolioSegmentAxis",
        "ProductOrServiceAxis",
        "PortfolioSegmentAxis",  # JPM, FITB
        "FinancingReceivableInformationByPortfolioSegmentAxis",  # KEY
    ),
    "loan_class": (
        "FinancingReceivableRecordedInvestmentByClassOfFinancingReceivableAxis",
        "ClassOfFinancingReceivableAxis",
        # 13 banks, 4,500 facts in FY2013 -- and the only loan axis WFC has.
        "AccountsNotesLoansAndFinancingReceivableByReceivableTypeAxis",
        "FinancialInstrumentAxis",
    ),
    "credit_quality": (
        "InternalCreditAssessmentAxis",
        "FinancingReceivableCreditQualityIndicatorAxis",
        "CreditRatingMoodysAxis",
        "CreditScoreFicoAxis",
        "CreditQualityIndicatorAxis",  # RF, HBAN
        "FinancingReceivableInformationByCreditQualityIndicatorAxis",  # KEY
        "FinancingReceivableRecordedInvestmentByCreditQualityIndicatorAxis",  # TFC
    ),
    "past_due": (
        "FinancingReceivablesPeriodPastDueAxis",
        "FinancingReceivableByDelinquencyStatusAxis",  # JPM
        "FinancingReceivableInformationByDelinquencyStatusAxis",  # NTRS
        "AgingAnalysisOfLoansAndLeasesAxis",  # HBAN
    ),
    "consolidation": ("ConsolidationItemsAxis", "StatementBusinessSegmentsAxis"),
}


def _explode_axes(rows: list[dict[str, Any]]) -> dict[str, list[str | None]]:
    """Promote the axes the panel keys on into columns; keep the rest as text.

    This works on the raw row dicts, never on a DataFrame column.  Handing
    per-fact ``{axis: member}`` dicts to Polars unifies them into a single
    Struct whose fields are the union of every axis seen in the filing, so
    every fact then appears to carry every axis -- which both destroys the
    dimension signature and shadows the alternative axes in
    :data:`PROMOTED_AXES`.

    ``dim_axes`` records the *identity* of the axis combination, not just how
    many there are.  Two facts can both carry one dimension yet be measured on
    entirely different cuts of the book (portfolio segment vs loan class vs
    industry), and adding them together silently double-counts.
    """
    promoted: dict[str, list[str | None]] = {k: [] for k in PROMOTED_AXES}
    other: list[str | None] = []
    signature: list[str | None] = []

    for row in rows:
        d: dict[str, str] = row["dims"]
        used: set[str] = set()
        for col, axes in PROMOTED_AXES.items():
            value = None
            for axis in axes:
                if axis in d:
                    value = d[axis]
                    used.add(axis)
                    break
            promoted[col].append(value)
        rest = {k: v for k, v in d.items() if k not in used}
        other.append(
            ";".join(f"{k}={v}" for k, v in sorted(rest.items())) if rest else ""
        )
        signature.append("|".join(sorted(d)))

    promoted["other_dims"] = other
    promoted["dim_axes"] = signature
    return promoted


def empty_frame() -> pl.DataFrame:
    cols: dict[str, Any] = dict(_SCHEMA)
    for k in PROMOTED_AXES:
        cols[k] = pl.String
    cols["other_dims"] = pl.String
    cols["dim_axes"] = pl.String
    return pl.DataFrame(schema=cols)


def parse_instance(
    filing: Filing, content: bytes | None = None, *, custom_ns: bool = True
) -> pl.DataFrame:
    """Parse one filing's instance document into a long dimensional frame."""
    if content is None:
        content = fetch_instance(filing)
    if content is None:
        return empty_frame()

    parser = etree.XMLParser(huge_tree=True, recover=True)
    root = etree.fromstring(content, parser=parser)

    keep: tuple[str, ...] = ()
    if custom_ns:
        # A bank's own extension namespace carries bespoke disclosures such as
        # criticized balances, which have no us-gaap element at all.
        keep = tuple(
            uri
            for uri in (root.nsmap or {}).values()
            if uri and re.match(r"^http://(www\.)?[^/]+/\d{8}$", uri)
        )

    contexts = _parse_contexts(root)
    units = _parse_units(root)
    rows = list(_iter_facts(root, contexts, units, filing, keep))
    if not rows:
        return empty_frame()

    extra = _explode_axes(rows)
    for row in rows:
        del row["dims"]

    df = pl.DataFrame(rows, schema_overrides=_SCHEMA, infer_schema_length=None)
    return df.with_columns(
        pl.Series(name, values, dtype=pl.String) for name, values in extra.items()
    )
