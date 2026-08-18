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

import logging
import re
import warnings
from dataclasses import dataclass
from io import BytesIO, StringIO
from typing import Any

import pandas as pd
import polars as pl

from . import edgar, taxonomy
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
    row_map={
        r"beginning balance|balance,? (at )?beginning": "acl_beginning",
        r"gross charge.?offs|charge.?offs": "charge_offs",
        r"recoveries": "recoveries",
        r"net charge.?offs": "nco",
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
EXCLUDE_LABEL = re.compile(
    r"experiencing financial difficulty|troubled debt|modification"
    r"|held for sale|unfunded|commitment|guarantee|servicing"
    r"|fair value|weighted average|per share|percent|%",
    re.IGNORECASE,
)

SPECS: tuple[TableSpec, ...] = (
    ACL_ROLLFORWARD,
    LOAN_PORTFOLIO,
    CREDIT_QUALITY,
    NONPERFORMING,
)


# --------------------------------------------------------------------------
# Fetching and parsing
# --------------------------------------------------------------------------


def fetch_primary_document(filing: Filing) -> bytes | None:
    """Primary 10-K/10-Q HTML as **bytes**, resolved via the index when needed.

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
    frames = []
    for f in filing_list:
        try:
            df = extract_filing(f)
        except Exception as exc:  # noqa: BLE001
            log.warning("html extract failed %s %s: %s", f.ticker, f.accn, exc)
            continue
        if not df.is_empty():
            frames.append(df)
    if not frames:
        return pl.DataFrame(schema=HTML_SCHEMA)
    return pl.concat(frames, how="vertical_relaxed")


def canonical_loan_columns() -> set[str]:
    """Loan columns the HTML specs can populate, for coverage reporting."""
    return {f"loans_{c}" for c in taxonomy.LOAN_CATEGORIES}
