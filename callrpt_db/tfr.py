"""The thrift gap: insured depositories that filed no Call Report.

The hole this fills
-------------------
An OTS-supervised savings institution filed a **Thrift Financial Report**, not
a Call Report, until the OTS was folded into the OCC and the TFR was retired
after 2011Q4.  CDR holds nothing for those institutions, so Washington Mutual,
Golden West, Countrywide Bank, Sovereign, IndyMac, ING Direct, Hudson City and
E\\*TRADE Bank are in the NIC graph, are in every lineage that leads to a 2026
firm, and contribute **zero** to the panel for every quarter they existed.

The consequence is visible rather than subtle.  Washington Mutual was a $307bn
bank when the FDIC sold it to JPMorgan Chase Bank on 2008-09-25, so JPMorgan's
loan book in this panel jumps by WaMu's entirety at 2008Q3 -- the quarter the
Call Report first consolidates it -- and reads as growth.  ``panel`` already
counts the affected institutions per bank-quarter as ``n_insured_not_filing``,
which says *where* the history is a floor without being able to raise it.

Where the numbers come from
---------------------------
The FDIC compiles its own quarterly financial series from whichever report an
insured institution filed -- Call Report or TFR -- and publishes it on one
schema through the BankFind API.  That is the whole reason this works where
CDR does not: the TFR data exists, mapped into common fields, and only the
FFIEC bulk files are Call-Report-only.  Measured over the 69 non-filing
depositories in this universe's lineages, BankFind returns 1,897 quarterly
rows for 2001-2012 and every field used below is populated on 100% of them.

Two things had to be checked rather than assumed, and both were:

**The join is on RSSD, not on name.**  BankFind's institution record carries
``FED_RSSD`` alongside its own ``CERT``, and NIC's entity record carries
``fdic_cert`` alongside its RSSD -- so the two registries can be bridged on an
identifier each collects independently, which is the same standard
``resolve_rssd`` holds the EIN bridge to.  Every one of the 69 has an FDIC
certificate in NIC.

**The field names do not mean what they look like.**  ``NAASSET`` and
``NALNLS`` are both "nonaccrual" and they are not interchangeable, which is
the trap that put nonaccrual owner-occupied CRE at $1m on the RC-N side.  They
were told apart by arithmetic: BankFind states ``NCLNLS`` (noncurrent loans
and leases) as well, and for Washington Mutual at 2007Q4
``NALNLS 6,121,610 + P9ASSET 310,251 == NCLNLS 6,431,861`` to the dollar.
That identity holds only if ``NALNLS`` is nonaccrual *loans* and ``P9ASSET``
is the 90-days-past-due bucket, which is what fixes both mappings.

``LNLSGR`` against ``LNLSNET`` was settled the same way:
``LNLSGR - LNATRES == LNLSNET`` exactly, so ``LNLSGR`` is gross of the
allowance and is the analogue of the panel's ``loans_total``.

What it does not do
-------------------
**It is not a Call Report and its categories are coarser.**  The TFR's loan
breakdown does not partition the way RC-C does, and BankFind publishes a
handful of loan categories where RC-C has twenty.  Rows produced here carry
``source = "tfr"`` and populate the columns below and no others, so a
consumer can tell a reconstructed quarter from a filed one -- and
``rcc_residual`` is deliberately not computed for them, because there is no
RC-C total to cross-foot against.

**Flows are year-to-date, exactly like ``RIAD``.**  WaMu's net charge-offs run
2,167,305 at 2007Q4 and restart at 1,367,840 in 2008Q1, so the quarterly
figure needs the same differencing :func:`panel.quarterize` applies to the
Call Report flows, per charter and before the rollup.

**The last quarter is the last one filed.**  WaMu's series ends at 2008Q2; it
failed on 2008-09-25 and filed nothing for 2008Q3.  The gap this closes is
therefore 2001Q1-2008Q2, which is exactly the span in which the panel carried
nothing for it.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import time
from pathlib import Path

import httpx
import polars as pl

from . import nic
from .config import RAW_CALL, USER_AGENT

log = logging.getLogger(__name__)

# The documented host redirects to this one; using it directly avoids a
# redirect on every request and a silently double-encoded query string.
BANKFIND_URL = "https://api.fdic.gov/banks/financials"
BANKFIND_INSTITUTIONS = "https://api.fdic.gov/banks/institutions"

# Certificates per request.  The filter is an OR list and the URL is a GET, so
# this is bounded by URL length rather than by any documented page size.
CERT_CHUNK = 10
MAX_ROWS = 10_000

# BankFind is a public API with no published rate limit; the same courtesy the
# rest of this package extends to FFIEC applies.
_MIN_INTERVAL = 0.5

# Amounts are in thousands of dollars, as the Call Report's are, so the panel's
# own ``mdrm.UNIT_SCALE`` applies unchanged.  Verified against Washington
# Mutual at 2008Q2: ASSET 307,021,614 is a $307bn bank.
FIELDS: dict[str, str] = {
    # BankFind field -> panel column
    "ASSET": "assets",
    "DEP": "deposits",
    "EQ": "equity",
    "LNLSGR": "loans_total",
    "LNATRES": "acl_total",
    "LNRENRES": "loans_cre_nonfarm_nonres",
    "LNRECONS": "loans_construction",
    "LNREMULT": "loans_multifamily",
    "LNCI": "loans_ci",
    "NALNLS": "nonaccrual_total",
    "P3ASSET": "pd_dpd_30_89",
    "P9ASSET": "pd_dpd_90_plus",
    # The two sides of the charge-off, not the net figure.  ``NTLNLS`` is
    # published and is *not* what the panel wants: ``panel.with_nco`` builds
    # ``nco_total`` as gross charge-offs less recoveries, so handing it a net
    # number would either be overwritten or would misstate the gross line.
    # The two sides tie to the net one exactly -- Washington Mutual at 2007Q4
    # is 2,311,660 - 144,355 = 2,167,305 -- which is the third identity
    # :func:`check_identities` measures.
    #
    # Both are year-to-date, exactly like every ``RIAD`` item, and both are in
    # ``mdrm.FLOW_COLUMNS``, so ``panel.quarterize`` differences them per
    # charter along with the Call Report's own flows.
    "DRLNLS": "charge_offs_total",
    "CRLNLS": "recoveries_total",
}

# Read but not mapped: they exist to check the ones that are.  See the module
# docstring for the identities they settle.
CHECK_FIELDS = ("LNLSNET", "NCLNLS", "NAASSET", "NTLNLS")

_REQUEST = ["CERT", "REPDTE", *FIELDS, *CHECK_FIELDS]

SCHEMA: dict[str, type[pl.DataType] | pl.DataType] = {
    "rssd": pl.Int64,
    "cert": pl.Int64,
    "period": pl.Date,
    "source": pl.Utf8,
    **{column: pl.Float64 for column in FIELDS.values()},
}


def empty_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=SCHEMA)


def _cache_path(start: str, end: str) -> Path:
    """One file per window, holding a map of certificate -> its rows."""
    return RAW_CALL.parent / "tfr" / f"bankfind_{start}_{end}.json"


def _client() -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(120.0, connect=30.0),
        follow_redirects=True,
        headers={"User-Agent": f"callrpt_db/0.1 (+{USER_AGENT})"},
    )


def missing_depositories(panel: pl.DataFrame) -> dict[int, int]:
    """RSSD -> FDIC certificate, for every depository that filed nothing.

    Read straight off the panel's own ``insured_not_filing`` column rather
    than recomputed, so this covers exactly the institutions the build already
    reports as missing and cannot drift away from them.
    """
    if panel.is_empty() or "insured_not_filing" not in panel.columns:
        return {}
    wanted: set[int] = set()
    for (value,) in panel.select("insured_not_filing").iter_rows():
        if value:
            wanted.update(int(r) for r in value.split(";") if r)
    entities = nic.entities()
    out: dict[int, int] = {}
    for rssd in sorted(wanted):
        entity = entities.get(rssd)
        if entity is not None and entity.fdic_cert is not None:
            out[rssd] = int(entity.fdic_cert)
    if len(out) < len(wanted):
        log.info(
            "tfr: %d of %d non-filing depositories have an FDIC certificate in NIC",
            len(out),
            len(wanted),
        )
    return out


def fetch(
    certs: list[int],
    start: dt.date,
    end: dt.date,
    *,
    refresh: bool = False,
    cached_only: bool = False,
) -> list[dict]:
    """Quarterly financials for these certificates, cached on disk.

    Fetching and building are separate steps, on the same terms as the CDR
    zips: ``build_call_panel`` reads only what is already there, never touches
    the network, and stays reproducible.  ``cached_only`` is what enforces
    that -- a build asked for TFR history nobody fetched gets nothing and says
    so, rather than quietly going online in the middle of a panel build.

    The cache is keyed **per certificate**, not per request.  Which
    certificates are wanted depends on the lineages, so the set the fetch step
    resolves and the set the build resolves need not be identical, and a cache
    keyed on the request as a whole would then miss entirely -- leaving a
    build to carry no thrifts at all with a full cache sitting on disk.  Per
    certificate, a superset satisfies a subset, and only what is genuinely
    absent is fetched.
    """
    if not certs:
        return []
    ordered = sorted(set(certs))
    stamp_start, stamp_end = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
    path = _cache_path(stamp_start, stamp_end)
    cached: dict[str, list[dict]] = {}
    if path.exists() and not refresh:
        cached = json.loads(path.read_text(encoding="utf-8"))

    missing = [c for c in ordered if str(c) not in cached]
    if missing and cached_only:
        log.warning(
            "tfr: %d of %d certificates are not cached for %s..%s; "
            "run fetch_call.py --tfr",
            len(missing),
            len(ordered),
            stamp_start,
            stamp_end,
        )
        missing = []

    if missing:
        fetched: dict[str, list[dict]] = {str(c): [] for c in missing}
        with _client() as client:
            for i in range(0, len(missing), CERT_CHUNK):
                chunk = missing[i : i + CERT_CHUNK]
                params = {
                    "filters": (
                        "CERT:(" + " OR ".join(str(c) for c in chunk) + ")"
                        f" AND REPDTE:[{stamp_start} TO {stamp_end}]"
                    ),
                    "fields": ",".join(_REQUEST),
                    "limit": MAX_ROWS,
                    "sort_by": "REPDTE",
                    "sort_order": "ASC",
                    "format": "json",
                }
                time.sleep(_MIN_INTERVAL)
                response = client.get(BANKFIND_URL, params=params)
                response.raise_for_status()
                for item in response.json().get("data", []):
                    row = item["data"]
                    fetched.setdefault(str(row.get("CERT")), []).append(row)
        # A certificate BankFind has no financials for is cached as an empty
        # list, so it is not asked for again on every build.
        cached.update(fetched)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cached), encoding="utf-8")
        log.info(
            "tfr: fetched %d certificates (%d rows); cache holds %d",
            len(missing),
            sum(len(v) for v in fetched.values()),
            len(cached),
        )

    return [row for c in ordered for row in cached.get(str(c), [])]


def _period(repdte: str | int) -> dt.date | None:
    text = str(repdte)
    if len(text) != 8 or not text.isdigit():
        return None
    return dt.date(int(text[:4]), int(text[4:6]), int(text[6:]))


def frame(rows: list[dict], cert_to_rssd: dict[int, int]) -> pl.DataFrame:
    """BankFind rows -> the panel's charter schema, in dollars.

    A certificate BankFind knows but this universe does not is dropped rather
    than kept unattributed: the point of the join is that every row belongs to
    a named organisation's lineage.
    """
    from .mdrm import UNIT_SCALE

    out: list[dict] = []
    for row in rows:
        raw_cert = row.get("CERT")
        period = _period(row.get("REPDTE", ""))
        if raw_cert is None or period is None:
            continue
        cert = int(raw_cert)
        rssd = cert_to_rssd.get(cert)
        if rssd is None:
            continue
        record: dict = {
            "rssd": rssd,
            "cert": cert,
            "period": period,
            "source": "tfr",
        }
        for field, column in FIELDS.items():
            value = row.get(field)
            record[column] = None if value is None else float(value) * UNIT_SCALE
        out.append(record)
    if not out:
        return empty_frame()
    return pl.DataFrame(out, schema_overrides=SCHEMA).sort(["rssd", "period"])


def check_identities(rows: list[dict]) -> dict[str, tuple[int, int]]:
    """Do the two field identities hold? ``{name: (held, checked)}``.

    This is the evidence for the field mapping, and it is computed rather than
    asserted so that a change in what BankFind publishes shows up as a falling
    ratio instead of as silently different numbers.  Both held on 100% of the
    rows measured.
    """
    results = {
        "gross_less_allowance_is_net": [0, 0],
        "noncurrent_is_nonaccrual_plus_90": [0, 0],
        "charge_offs_less_recoveries_is_net": [0, 0],
    }

    def close(a: float, b: float) -> bool:
        return abs(a - b) <= max(1.0, abs(b) * 1e-6)

    def trio(row: dict, *names: str) -> tuple[float, float, float] | None:
        values = [row.get(n) for n in names]
        if any(v is None for v in values):
            return None
        return tuple(float(v) for v in values)  # type: ignore[return-value]

    for row in rows:
        loans = trio(row, "LNLSGR", "LNATRES", "LNLSNET")
        if loans is not None:
            gross, allowance, net = loans
            results["gross_less_allowance_is_net"][1] += 1
            results["gross_less_allowance_is_net"][0] += close(gross - allowance, net)
        credit = trio(row, "NALNLS", "P9ASSET", "NCLNLS")
        if credit is not None:
            nonaccrual, past_due, noncurrent = credit
            results["noncurrent_is_nonaccrual_plus_90"][1] += 1
            results["noncurrent_is_nonaccrual_plus_90"][0] += close(
                nonaccrual + past_due, noncurrent
            )
        flows = trio(row, "DRLNLS", "CRLNLS", "NTLNLS")
        if flows is not None:
            charge_offs, recoveries, net = flows
            results["charge_offs_less_recoveries_is_net"][1] += 1
            results["charge_offs_less_recoveries_is_net"][0] += close(
                charge_offs - recoveries, net
            )
    return {k: (v[0], v[1]) for k, v in results.items()}


def backfill(
    panel: pl.DataFrame,
    periods: list[str],
    *,
    refresh: bool = False,
    cached_only: bool = False,
) -> tuple[pl.DataFrame, dict]:
    """Charter-level TFR rows for the panel's own non-filing depositories.

    Returns ``(frame, info)``.  ``info`` is the decision log the build writes
    into ``call_panel_build_info.json``: how many institutions were sought,
    how many BankFind knew, and whether the field identities held.
    """
    cert_of = missing_depositories(panel)
    if not cert_of:
        return empty_frame(), {"sought": 0, "resolved": 0, "rows": 0}

    quarters = sorted(periods)
    if not quarters:
        return empty_frame(), {"sought": len(cert_of), "resolved": 0, "rows": 0}
    from . import cdr

    start, end = cdr.quarter_end(quarters[0]), cdr.quarter_end(quarters[-1])
    rows = fetch(
        list(cert_of.values()), start, end, refresh=refresh, cached_only=cached_only
    )
    cert_to_rssd = {cert: rssd for rssd, cert in cert_of.items()}
    built = frame(rows, cert_to_rssd)
    info = {
        "sought": len(cert_of),
        "resolved": int(built["cert"].n_unique()) if not built.is_empty() else 0,
        "rows": built.height,
        "window": [start.isoformat(), end.isoformat()],
        "columns": sorted(FIELDS.values()),
        "identities": check_identities(rows),
        "source": "FDIC BankFind (api.fdic.gov/banks/financials)",
    }
    log.info(
        "tfr: %d rows for %d of %d non-filing depositories",
        info["rows"],
        info["resolved"],
        info["sought"],
    )
    return built, info
