"""HTML table fallback for disclosures that never reach XBRL.

Two distinct gaps this covers:

1. **Pre-XBRL and thinly tagged filings.**  Older 10-Ks, and small filers who
   tag only the face financial statements, leave the ACL rollforward and the
   loan-portfolio table untagged.

2. **Narrative-only disclosures.**  Criticized and classified balances,
   office-CRE exposure and leveraged-lending exposure are frequently disclosed
   only in MD&A prose and tables that carry no XBRL element at all.  No amount
   of tag hunting recovers those -- they have to be read out of the HTML.

Tables are located by matching row *labels*, not by position, because table
ordering is not stable across banks or across years for the same bank.  Parsed
values keep a ``confidence`` and the matched label so that a reconciliation
step can prefer XBRL and show what the fallback actually matched.
"""

from __future__ import annotations

import gzip
import logging
import re
import warnings
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from . import edgar, parallel, taxonomy
from .config import RAW_HTML
from .filings import Filing

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Table identification
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TableSpec:
    """How to recognise a table and what to pull out of it."""

    name: str
    # A table qualifies when at least ``min_hits`` of these appear in its text.
    signals: tuple[str, ...]
    min_hits: int = 2
    # Row-label patterns mapped to the canonical variable they populate.
    row_map: dict[str, str] = None  # type: ignore[assignment]


ACL_ROLLFORWARD = TableSpec(
    name="acl_rollforward",
    signals=(
        "allowance for credit losses",
        "beginning balance",
        "charge-offs",
        "recoveries",
        "provision",
        "ending balance",
    ),
    min_hits=3,
    # Order is load-bearing: the first pattern to match a row label wins, and
    # "net charge-offs" contains "charge-offs".  Listed the other way round --
    # as it was -- the net rule is unreachable and every net figure is read as
    # a gross one, which understates nothing and overstates charge_offs by the
    # whole recovery rate.  Same trap as the net/gross element split in
    # ``variables``, on the other side of the pipeline.
    row_map={
        r"beginning balance|balance,? (at )?beginning": "acl_beginning",
        # 'net' is not always adjacent: Zions writes 'net loan and lease
        # charge-offs', which the adjacent form leaves in the gross column.
        r"net\b.{0,30}charge.?offs": "nco",
        r"gross charge.?offs|charge.?offs": "charge_offs",
        r"recoveries": "recoveries",
        r"provision": "provision",
        r"ending balance|balance,? (at )?end": "acl_ending",
    },
)

LOAN_PORTFOLIO = TableSpec(
    name="loan_portfolio",
    signals=(
        "commercial and industrial",
        "commercial real estate",
        "residential",
        "total loans",
        "construction",
        "home equity",
        "credit card",
    ),
    min_hits=3,
    row_map={
        r"commercial real estate|commercial mortgage": "loans_cre_total",
        r"construction|land development": "loans_construction",
        r"commercial and industrial|c&i\b": "loans_ci",
        r"multifamily|multi-family": "loans_multifamily",
        r"owner.?occupied": "loans_cre_owner_occupied",
        r"non.?owner.?occupied|investor (commercial )?real estate": "loans_cre_investor",
        r"small business": "loans_small_business",
        r"residential mortgage|residential real estate|1-4 family": "loans_resi_mortgage",
        r"home equity": "loans_home_equity",
        r"credit card": "loans_credit_card",
        r"automobile|auto\b|indirect auto": "loans_auto",
        r"lease financing|leases": "loans_lease",
        r"total loans( and leases)?": "loans_total",
    },
)

CREDIT_QUALITY = TableSpec(
    name="credit_quality",
    signals=(
        "pass",
        "special mention",
        "substandard",
        "criticized",
        "classified",
        "doubtful",
        "risk rating",
        "internal risk",
    ),
    min_hits=2,
    row_map={
        r"special mention": "cq_special_mention",
        r"substandard": "cq_substandard",
        r"doubtful": "cq_doubtful",
        r"^loss$": "cq_loss",
        r"criticized": "cq_criticized",
        r"classified": "cq_classified",
        r"^pass": "cq_pass",
        r"total (criticized|classified)": "cq_criticized",
    },
)

NONPERFORMING = TableSpec(
    name="nonperforming",
    signals=(
        "nonaccrual",
        "nonperforming",
        "other real estate owned",
        "90 days",
        "past due",
    ),
    min_hits=2,
    row_map={
        r"total nonaccrual|nonaccrual loans": "nonaccrual_total",
        r"other real estate owned|oreo|foreclosed": "oreo",
        r"total non.?performing assets": "npa_total",
        r"90 days.*(past due|accruing)": "pd_dpd_90_plus",
        r"30.{0,4}89 days": "pd_dpd_30_89",
    },
)

# Labels that look like a target row but describe a different population.
# "Total loans to borrowers experiencing financial difficulty" is a
# modification disclosure, not the loan book, and matches a naive
# "total loans" pattern.
# Labels whose row must never be read at all.  Two kinds.
#
# The first is a different population, or a different unit: held-for-sale,
# unfunded, modified -- and anything stating a *ratio*, which belongs beside
# 'percent' and '%'.  Zions prints 'Ratio of net charge-offs to average loans
# and leases' in the same table as the dollar rollforward, and a percentage
# read as a dollar amount is the quietest error this file can make.
#
# The second is a label that *negates* the concept it names, which is the same
# trap ``taxonomy.LOAN_RULES`` guards three times over on the member side and
# which this matcher had no notion of.  JPMorgan's 10-K cost all three:
# "total consumer, excluding credit card loans" was read as the card book,
# "noncriticized" as criticized, and "pre-provision profit" as the provision.
EXCLUDE_LABEL = re.compile(
    r"experiencing financial difficulty|troubled debt|modification"
    r"|held for sale|unfunded|commitment|guarantee|servicing"
    r"|fair value|weighted average|per share|percent|%"
    r"|\bexcluding\b|\bexcept\b|\bnon-?criticized\b|\bpre-?provision\b"
    r"|\bratio\b",
    re.IGNORECASE,
)

# The CRE book split by what the property is used for, which is where office
# exposure lives when a bank discloses it at all in a filing.  It is tagged in
# no XBRL instance -- Bank of America's carries no office member and no
# property-type axis anywhere -- and it is absent from the 10-Q, so this table
# in the 10-K is the only place the number exists.
#
# Bank of America's Table 33, "Outstanding Commercial Real Estate Loans by
# Geographic Region and Property Type", gives Office at 12,447 for 2025, which
# reconciles two independent ways against the MD&A prose beside it: "Office
# loans decreased $2.6 billion, or 17 percent, during 2025" (15.0 -> 12.4) and
# "represented approximately one percent of total loans" (12.4 / 1,090).
#
# The signals are the *other* property types on purpose.  "Office" alone is a
# word that appears in every filing ("principal executive offices", "deposits
# in U.S. offices"); a table that also names industrial/warehouse, multi-family
# rental and shopping centres is unambiguously this schedule.
CRE_PROPERTY_TYPE = TableSpec(
    name="cre_property_type",
    signals=(
        "by property type",
        "industrial / warehouse",
        "industrial/warehouse",
        "multi-family rental",
        "shopping centers",
        "hotel",
        "non-residential",
        "multi-use",
    ),
    min_hits=3,
    row_map={
        # Anchored: the same table has an "Other" row and a residential block,
        # and the geographic half above it must not contribute anything.
        r"^office$|^office loans?$|^office buildings?$": "loans_office_cre",
        r"^multi.?family rental": "loans_multifamily",
        r"^total outstanding commercial real estate loans?$": "loans_cre_total",
    },
)

# Two banks publish this table and they lay it out differently, which is worth
# knowing before extending the spec.  Bank of America's columns are years, so
# the left-most number is the current one -- the rule this module already uses.
# M&T's are maturity buckets followed by a total ("Office 408 809 872 1,024 310
# $3,423 14%"), so the useful number is the total.  Both come out right, and
# both were checked row by row against the filings: BAC 2019-2025 and M&T
# 2023-2025, whose buckets sum to the total taken.  A third layout should be
# verified the same way rather than assumed.
#
# M&T's population is its "permanent finance" CRE, which is a narrower book
# than the consolidated CRE line, so its office share of loans is not strictly
# comparable with Bank of America's.

SPECS: tuple[TableSpec, ...] = (
    ACL_ROLLFORWARD,
    LOAN_PORTFOLIO,
    CREDIT_QUALITY,
    NONPERFORMING,
    CRE_PROPERTY_TYPE,
)


# --------------------------------------------------------------------------
# Fetching and parsing
# --------------------------------------------------------------------------


def _html_cache_path(filing: Filing) -> Path:
    return RAW_HTML / f"{filing.ticker}_{filing.accn_nodash}.html.gz"


def ensure_cached(filing: Filing) -> bool:
    """Put a filing's primary document on disk if it is not already there.

    Same contract as ``instance.ensure_cached`` and for the same two reasons:
    it keeps fetching serial and behind the one rate limiter while parsing runs
    in a pool, and it makes a rebuild cost nothing.  Re-parsing 210 filings was
    dominated by re-downloading them, because this cache was declared in
    ``config.RAW_HTML`` and never written.
    """
    path = _html_cache_path(filing)
    if path.exists():
        return True
    content = _download_primary_document(filing)
    if content is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wb") as fh:
        fh.write(content)
    tmp.replace(path)
    return True


def fetch_primary_document(filing: Filing) -> bytes | None:
    """Primary 10-K/10-Q HTML as **bytes**, from cache where one exists."""
    path = _html_cache_path(filing)
    if path.exists():
        with gzip.open(path, "rb") as fh:
            return fh.read()
    return _download_primary_document(filing)


def _download_primary_document(filing: Filing) -> bytes | None:
    """Fetch the primary document, resolving it via the index when needed.

    Bytes rather than text on purpose: inline-XBRL filings begin with an XML
    declaration, and lxml refuses to parse a ``str`` that carries an encoding
    declaration.  Handing it the raw bytes keeps the declaration meaningful
    and avoids a double decode.
    """
    if filing.primary_doc:
        try:
            return edgar.get(filing.primary_doc_url).content
        except Exception as exc:  # noqa: BLE001
            log.warning("primary doc fetch failed %s: %s", filing.accn, exc)

    try:
        index = edgar.get(f"{filing.folder_url}/index.json").json()
    except Exception as exc:  # noqa: BLE001
        log.warning("index fetch failed %s: %s", filing.accn, exc)
        return None
    names = [
        i["name"] for i in index["directory"]["item"] if i["name"].endswith(".htm")
    ]
    main = [n for n in names if not re.search(r"ex(hibit)?[-_]?\d", n, re.IGNORECASE)]
    if not main:
        return None
    try:
        return edgar.get(f"{filing.folder_url}/{main[0]}").content
    except Exception as exc:  # noqa: BLE001
        log.warning("fallback doc fetch failed %s: %s", filing.accn, exc)
        return None


def read_tables(html: bytes | str) -> list[pd.DataFrame]:
    """All tables in a filing, trying lxml first and bs4 as a backstop.

    Failures are logged rather than swallowed: a filing that silently yields
    zero tables looks identical to a filing with no relevant disclosure, and
    that is exactly the kind of gap this module exists to close.
    """
    payload: Any = BytesIO(html) if isinstance(html, bytes) else StringIO(html)
    errors: list[str] = []
    for flavor in ("lxml", "bs4"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tables = pd.read_html(payload, flavor=flavor)
        except Exception as exc:  # noqa: BLE001 - malformed markup is the norm
            errors.append(f"{flavor}: {type(exc).__name__} {exc}")
            payload = BytesIO(html) if isinstance(html, bytes) else StringIO(html)
            continue
        if tables:
            return tables
        payload = BytesIO(html) if isinstance(html, bytes) else StringIO(html)
    if errors:
        log.warning("read_html found no tables (%s)", "; ".join(errors)[:300])
    return []


def _table_text(df: pd.DataFrame) -> str:
    return " ".join(str(v).lower() for v in df.head(60).to_numpy().ravel())


def classify_table(df: pd.DataFrame) -> TableSpec | None:
    """Match a table against the specs, best hit wins."""
    text = _table_text(df)
    best: tuple[int, TableSpec] | None = None
    for spec in SPECS:
        hits = sum(1 for s in spec.signals if s in text)
        if hits >= spec.min_hits and (best is None or hits > best[0]):
            best = (hits, spec)
    return best[1] if best else None


_NUM = re.compile(r"^\(?\$?\s*-?[\d,]+(\.\d+)?\)?$")


def _to_number(value: Any) -> float | None:
    """Parse a financial-statement cell, honouring parenthesised negatives."""
    if value is None:
        return None
    s = str(value).strip().replace("$", "").replace(",", "").replace("—", "")
    if not s or s in {"-", "--", "nan", "None"}:
        return None
    negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        num = float(s)
    except ValueError:
        return None
    return -num if negative else num


def _row_label(row: pd.Series) -> str:
    for value in row:
        s = str(value).strip()
        if s and s.lower() != "nan" and not _NUM.match(s):
            return s.lower()
    return ""


def _first_numeric(row: pd.Series) -> float | None:
    """Left-most numeric cell -- the current period in a comparative table."""
    for value in row:
        num = _to_number(value)
        if num is not None:
            return num
    return None


def extract_from_table(
    df: pd.DataFrame, spec: TableSpec, filing: Filing
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not spec.row_map:
        return rows
    compiled = [(re.compile(p, re.IGNORECASE), var) for p, var in spec.row_map.items()]

    for _, row in df.iterrows():
        label = _row_label(row)
        if not label or EXCLUDE_LABEL.search(label):
            continue
        for pattern, variable in compiled:
            if not pattern.search(label):
                continue
            value = _first_numeric(row)
            if value is None:
                continue
            rows.append(
                {
                    "bank": filing.bank,
                    "ticker": filing.ticker,
                    "cik": filing.cik,
                    "accn": filing.accn,
                    "form": filing.form,
                    "period": filing.report_date,
                    "filed": filing.filing_date,
                    "table": spec.name,
                    "variable": variable,
                    "label": label[:120],
                    "value": float(value),
                    "source": "html",
                    # read_html cannot recover the table's scale header, so the
                    # magnitude is unresolved until reconciliation sees it
                    # alongside the XBRL value.
                    "scale_resolved": False,
                    "confidence": 0.6,
                }
            )
            break
    return rows


HTML_SCHEMA: dict[str, Any] = {
    "bank": pl.String,
    "ticker": pl.String,
    "cik": pl.String,
    "accn": pl.String,
    "form": pl.String,
    "period": pl.Date,
    "filed": pl.Date,
    "table": pl.String,
    "variable": pl.String,
    "label": pl.String,
    "value": pl.Float64,
    "source": pl.String,
    "scale_resolved": pl.Boolean,
    "confidence": pl.Float64,
}


def extract_filing(filing: Filing) -> pl.DataFrame:
    """Parse one filing's HTML tables into a long frame."""
    html = fetch_primary_document(filing)
    if not html:
        return pl.DataFrame(schema=HTML_SCHEMA)

    rows: list[dict[str, Any]] = []
    for table in read_tables(html):
        if table.empty or table.shape[1] < 2:
            continue
        spec = classify_table(table)
        if spec is None:
            continue
        rows.extend(extract_from_table(table, spec, filing))

    if not rows:
        return pl.DataFrame(schema=HTML_SCHEMA)

    df = pl.DataFrame(rows, schema_overrides=HTML_SCHEMA, infer_schema_length=None)
    # A label can match in several tables; keep the highest-confidence hit.
    return df.sort(["confidence"], descending=True).unique(
        subset=["ticker", "period", "variable", "table"], keep="first"
    )


def extract_filings(filing_list: list[Filing]) -> pl.DataFrame:
    """Parse many filings, downloading serially and parsing across cores."""
    cached = [f for f in filing_list if ensure_cached(f)]
    return parallel.map_frames(
        extract_filing,
        cached,
        schema=HTML_SCHEMA,
        label="html extract",
        describe=lambda f: f"{f.ticker} {f.accn}",
    )


def canonical_loan_columns() -> set[str]:
    """Loan columns the HTML specs can populate, for coverage reporting."""
    return {f"loans_{c}" for c in taxonomy.LOAN_CATEGORIES}
