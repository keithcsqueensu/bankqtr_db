"""Download FFIEC bulk data into the local cache.

uv run python scripts/fetch_call.py --since 2013-01-01
uv run python scripts/fetch_call.py --since 2013-01-01 --structure-only
uv run python scripts/fetch_call.py --list-periods

Fetching and building are separate on purpose, exactly as they are for the
EDGAR side: ``build_call_panel.py`` reads only what is already cached and never
touches the network, so a rebuild is reproducible.

One quarter is a ~6 MB zip holding every schedule for every filer -- around
4,400 institutions -- so a 2013-start window is ~350 MB and a few minutes of
serial downloading behind the one rate limiter in ``cdr.py``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys

from callrpt_db import cdr, nic
from callrpt_db.config import ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("fetch_call")


def _date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since", type=_date, default=dt.date(2013, 1, 1))
    ap.add_argument("--until", type=_date, default=None)
    ap.add_argument("--refresh", action="store_true", help="re-download cached files")
    ap.add_argument(
        "--structure-only",
        action="store_true",
        help="fetch only the NIC entity/relationship files",
    )
    ap.add_argument(
        "--list-periods",
        action="store_true",
        help="print the periods CDR offers and exit",
    )
    args = ap.parse_args(argv)

    ensure_dirs()

    if args.list_periods:
        periods = cdr.available_periods()
        for period in sorted(periods):
            print(f"{period}\t{cdr.period_label(period)}\tid={periods[period]}")
        print(f"\n{len(periods)} periods offered")
        return 0

    log.info("fetching NIC structure files")
    nic.fetch_structure(refresh=args.refresh)
    if args.structure_only:
        return 0

    until = args.until or dt.datetime.now(dt.UTC).date()
    wanted = cdr.periods_between(args.since, until)

    with cdr.new_client() as client:
        offered = cdr.available_periods(client)
        missing = [p for p in wanted if p not in offered]
        if missing:
            log.warning(
                "CDR does not offer %d of the requested periods: %s",
                len(missing),
                ", ".join(missing),
            )
        todo = [p for p in wanted if p in offered]
        log.info("%d periods to consider (%s..%s)", len(todo), todo[0], todo[-1])

        failed: list[tuple[str, str]] = []
        for i, period in enumerate(todo, 1):
            try:
                path = cdr.fetch_period(
                    period, refresh=args.refresh, client=client, periods=offered
                )
            except Exception as exc:  # noqa: BLE001 - one bad quarter must not lose the rest
                log.error("%s failed: %s", period, exc)
                failed.append((period, str(exc)))
                continue
            log.info(
                "[%d/%d] %s -> %s (%.1f MB)",
                i,
                len(todo),
                period,
                path.name,
                path.stat().st_size / 1e6,
            )

    have = cdr.cached_periods()
    log.info("cache now holds %d periods (%s..%s)", len(have), have[0], have[-1])
    if failed:
        log.error("%d periods failed: %s", len(failed), ", ".join(p for p, _ in failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
