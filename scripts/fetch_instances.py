"""Download and cache XBRL instance documents for the bank universe.

Usage:
    uv run python scripts/fetch_instances.py --since 2020-01-01
    uv run python scripts/fetch_instances.py --latest-10k     # survey mode
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging

from bankqtr_db import filings, instance
from bankqtr_db.config import BANKS, ensure_dirs

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2020-01-01")
    ap.add_argument(
        "--latest-10k",
        action="store_true",
        help="only the most recent 10-K per bank (fast vocabulary survey)",
    )
    ap.add_argument("--forms", default="10-K,10-Q")
    args = ap.parse_args()

    ensure_dirs()
    since = dt.date.fromisoformat(args.since)
    forms = tuple(args.forms.split(","))

    total = ok = fail = 0
    for bank in BANKS:
        fs = filings.list_filings(bank, since=since, forms=forms)
        if args.latest_10k:
            tens = [f for f in fs if f.form == "10-K"]
            fs = tens[:1]
        for f in fs:
            total += 1
            try:
                content = instance.fetch_instance(f)
            except Exception as exc:  # noqa: BLE001 - keep going across the universe
                fail += 1
                print(
                    f"FAIL {f.ticker:5s} {f.form:5s} {f.report_date} {exc}", flush=True
                )
                continue
            if content is None:
                fail += 1
                print(f"MISS {f.ticker:5s} {f.form:5s} {f.report_date}", flush=True)
            else:
                ok += 1
                print(
                    f"ok   {f.ticker:5s} {f.form:5s} {f.report_date} "
                    f"{len(content) / 1e6:5.1f}MB",
                    flush=True,
                )
    print(f"\n{ok} cached, {fail} failed, {total} total")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
