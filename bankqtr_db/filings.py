"""Index of 10-K / 10-Q filings per bank, derived from the submissions API."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import polars as pl

from . import edgar
from .config import BANKS, Bank

PERIODIC_FORMS = ("10-K", "10-Q")


@dataclass(frozen=True)
class Filing:
    bank: str
    ticker: str
    cik: str
    accn: str
    form: str
    filing_date: dt.date
    report_date: dt.date | None
    primary_doc: str

    @property
    def accn_nodash(self) -> str:
        return self.accn.replace("-", "")

    @property
    def folder_url(self) -> str:
        return (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(self.cik)}/{self.accn_nodash}"
        )

    @property
    def primary_doc_url(self) -> str:
        return f"{self.folder_url}/{self.primary_doc}"


def _parse_date(s: str | None) -> dt.date | None:
    if not s:
        return None
    try:
        return dt.date.fromisoformat(s)
    except ValueError:
        return None


def _rows_from_block(block: dict) -> list[dict]:
    """The submissions API stores filings column-wise; transpose to rows."""
    keys = list(block)
    if not keys:
        return []
    n = len(block[keys[0]])
    return [{k: block[k][i] for k in keys} for i in range(n)]


def list_filings(
    bank: Bank,
    *,
    forms: tuple[str, ...] = PERIODIC_FORMS,
    since: dt.date | None = None,
    include_amendments: bool = False,
) -> list[Filing]:
    """All periodic filings for a bank, newest first.

    Walks the ``files`` overflow shards as well as ``recent`` so that history
    beyond the most recent ~1000 filings is included.
    """
    subs = edgar.submissions(bank.cik)
    rows = _rows_from_block(subs["filings"]["recent"])
    for extra in subs["filings"].get("files", []):
        shard = edgar.get(f"https://data.sec.gov/submissions/{extra['name']}").json()
        rows.extend(_rows_from_block(shard))

    out: list[Filing] = []
    for r in rows:
        form = r.get("form", "")
        base = form.removesuffix("/A")
        if base not in forms:
            continue
        if not include_amendments and form.endswith("/A"):
            continue
        filed = _parse_date(r.get("filingDate"))
        if filed is None or (since and filed < since):
            continue
        out.append(
            Filing(
                bank=bank.name,
                ticker=bank.ticker,
                cik=bank.cik,
                accn=r["accessionNumber"],
                form=form,
                filing_date=filed,
                report_date=_parse_date(r.get("reportDate")),
                primary_doc=r.get("primaryDocument", ""),
            )
        )
    out.sort(key=lambda f: f.filing_date, reverse=True)
    return out


def all_filings(
    *, since: dt.date | None = None, banks: tuple[Bank, ...] = BANKS
) -> list[Filing]:
    result: list[Filing] = []
    for b in banks:
        result.extend(list_filings(b, since=since))
    return result


def to_frame(filings: list[Filing]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "bank": f.bank,
                "ticker": f.ticker,
                "cik": f.cik,
                "accn": f.accn,
                "form": f.form,
                "filing_date": f.filing_date,
                "report_date": f.report_date,
                "primary_doc": f.primary_doc,
            }
            for f in filings
        ],
        schema_overrides={"report_date": pl.Date, "filing_date": pl.Date},
    )
