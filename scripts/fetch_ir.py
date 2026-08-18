"""Discover and download quarterly IR supplements for the bank universe.

Analogous to ``fetch_facts.py`` / ``fetch_instances.py``: it only fills the
local cache under ``data/ir/``.  Extraction happens in ``build_panel.py --ir``.

    uv run python scripts/fetch_ir.py --since 2020-01-01
    uv run python scripts/fetch_ir.py --tickers USB,PNC --doc-types supplement
    uv run python scripts/fetch_ir.py --discover-only        # audit the routes
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
import time

import polars as pl

from bankqtr_db import ir
from bankqtr_db.config import DATA, OUT, ensure_dirs, universe

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("fetch_ir")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2020-01-01")
    ap.add_argument("--tickers", default="", help="comma-separated subset")
    ap.add_argument("--comparators", action="store_true")
    ap.add_argument(
        "--doc-types",
        default="supplement,presentation",
        help=f"any of {','.join(ir.DOC_PRIORITY)}",
    )
    ap.add_argument(
        "--discover-only",
        action="store_true",
        help="list what would be fetched, download nothing",
    )
    ap.add_argument("--refresh", action="store_true", help="re-download cached files")
    args = ap.parse_args()

    ensure_dirs()
    ir.IR_DIR.mkdir(parents=True, exist_ok=True)
    since = dt.date.fromisoformat(args.since)
    doc_types = tuple(t.strip() for t in args.doc_types.split(",") if t.strip())

    banks = universe(include_comparators=args.comparators)
    if args.tickers:
        wanted = {t.strip().upper() for t in args.tickers.split(",")}
        banks = tuple(b for b in banks if b.ticker in wanted)
    if not banks:
        print("no banks selected", file=sys.stderr)
        return 2

    print(
        f"universe: {len(banks)} banks, since {since}, "
        f"doc types {', '.join(doc_types)}",
        flush=True,
    )

    t0 = time.monotonic()
    found: dict[str, list[ir.IRDoc]] = {}
    pairs: list[tuple[ir.IRDoc, object]] = []
    for bank in banks:
        docs = ir.discover(bank, since=since)
        found[bank.ticker] = docs
        wanted_docs = [d for d in docs if d.doc_type in doc_types]
        if args.discover_only:
            print(
                f"  {bank.ticker:7s} {len(docs):3d} found, "
                f"{len(wanted_docs):3d} in scope",
                flush=True,
            )
            continue
        got = ir.download_all(wanted_docs, refresh=args.refresh, doc_types=doc_types)
        pairs.extend(got)
        missed = len(wanted_docs) - len(got)
        print(
            f"  {bank.ticker:7s} {len(got):3d} downloaded"
            + (f", {missed} failed" if missed else ""),
            flush=True,
        )

    print(f"\n{ir.discovery_report(found, banks=banks)}")

    if args.discover_only:
        return 0

    rows = ir.manifest_rows(pairs)
    if rows:
        manifest = pl.DataFrame(rows, schema_overrides={"period": pl.Date})
        manifest = manifest.sort(["ticker", "period", "doc_type"])
        manifest.write_csv(DATA / "ir" / "manifest.csv")
        manifest.write_csv(OUT / "ir_manifest.csv")
        total_mb = manifest["bytes"].sum() / 1e6
        print(
            f"\n{manifest.height} documents, {total_mb:,.0f} MB "
            f"-> {DATA / 'ir' / 'manifest.csv'}"
        )
        print(
            manifest.group_by("origin")
            .agg(pl.len().alias("docs"), pl.col("ticker").n_unique().alias("banks"))
            .sort("docs", descending=True)
        )
    print(f"done in {time.monotonic() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
