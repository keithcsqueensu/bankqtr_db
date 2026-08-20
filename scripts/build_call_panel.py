"""End-to-end build: cached FFIEC bulk data -> bank-quarter panel + cross-check.

uv run python scripts/build_call_panel.py --since 2013-01-01
uv run python scripts/build_call_panel.py --since 2013-01-01 --comparators
uv run python scripts/build_call_panel.py --since 2013-01-01 --no-crosscheck

Reads only what ``fetch_call.py`` has already cached and never touches the
network, so a rebuild is reproducible.  Outputs land in ``data/out``:

    call_panel.parquet / .csv     the holding-company panel
    call_panel_charters.parquet   one row per bank charter per quarter
    call_panel_coverage.csv       per bank and variable, how many quarters
    call_panel_flags.csv          bank-quarters failing a sanity check
    call_panel_build_info.json    window, settings, commit, cell counts
    source_diff.csv               EDGAR against FFIEC, per bank-quarter
    source_diff_summary.csv       the same, aggregated per bank and variable
    source_coverage.csv           which firms each source reaches
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import subprocess
import sys

import polars as pl

from callrpt_db import cdr, config, crosscheck, mdrm, panel

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("build_call")

# A residual under one reporting unit is rounding, not a mapping error: the
# Call Report is filed in whole thousands, so a difference below $1,000 cannot
# be a real disagreement about a category.
PARTITION_TOLERANCE_USD = 1_000.0


def _date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=config.ROOT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _filled_cells(frame: pl.DataFrame) -> int:
    numeric = [c for c in frame.columns if frame.schema[c] in (pl.Float64, pl.Int64)]
    return int(sum(frame[c].is_not_null().sum() for c in numeric))


def coverage(frame: pl.DataFrame) -> pl.DataFrame:
    """Per bank and variable: quarters populated, and the window covered."""
    if frame.is_empty():
        return pl.DataFrame()
    value_cols = [
        c
        for c in frame.columns
        if frame.schema[c] == pl.Float64 and c not in ("rcc_residual_pct",)
    ]
    rows = []
    for ticker, group in frame.group_by("ticker"):
        name = ticker[0] if isinstance(ticker, tuple) else ticker
        total = group.height
        for column in value_cols:
            filled = int(group[column].is_not_null().sum())
            rows.append(
                {
                    "ticker": name,
                    "variable": column,
                    "quarters": total,
                    "populated": filled,
                    "pct": round(filled / total * 100, 1) if total else None,
                }
            )
    return pl.DataFrame(rows).sort(["variable", "ticker"])


def flags(frame: pl.DataFrame) -> pl.DataFrame:
    """Bank-quarters that fail a sanity check, one row per failure.

    These are checks the Call Report makes possible and the XBRL panel cannot
    run: the form states its own totals, so a category that exceeds its parent
    or a partition that does not cross-foot is a defect rather than a
    disclosure choice.
    """
    if frame.is_empty():
        return pl.DataFrame()
    checks: list[tuple[str, pl.Expr, str]] = []
    have = set(frame.columns)

    if {"loans_total", "acl_total"} <= have:
        checks.append(
            (
                "acl_exceeds_loans",
                pl.col("acl_total") > pl.col("loans_total"),
                "allowance is larger than the loan book",
            )
        )
    if {"loans_total", "loans_cre_total"} <= have:
        checks.append(
            (
                "cre_exceeds_loans",
                pl.col("loans_cre_total") > pl.col("loans_total") * 1.001,
                "CRE is larger than total loans",
            )
        )
    if "rcc_residual" in have:
        checks.append(
            (
                "rcc_partition_over",
                pl.col("rcc_residual") > PARTITION_TOLERANCE_USD,
                "loan categories sum to more than RC-C's own total",
            )
        )
        checks.append(
            (
                "rcc_partition_under",
                pl.col("rcc_residual") < -PARTITION_TOLERANCE_USD,
                "loan categories leave part of RC-C's total unallocated",
            )
        )
    if {"nonaccrual_total", "loans_total"} <= have:
        checks.append(
            (
                "nonaccrual_exceeds_loans",
                pl.col("nonaccrual_total") > pl.col("loans_total"),
                "nonaccrual is larger than the loan book",
            )
        )
    if "loans_total" in have:
        checks.append(
            (
                "negative_loans",
                pl.col("loans_total") < 0,
                "negative total loans",
            )
        )

    frames = []
    for name, expr, description in checks:
        hit = frame.filter(expr.fill_null(False))
        if hit.is_empty():
            continue
        keep = [c for c in ("ticker", "bank", "period") if c in hit.columns]
        frames.append(
            hit.select(keep).with_columns(
                pl.lit(name).alias("flag"), pl.lit(description).alias("detail")
            )
        )
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical_relaxed").sort(["ticker", "period", "flag"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since", type=_date, default=dt.date(2013, 1, 1))
    ap.add_argument("--until", type=_date, default=None)
    ap.add_argument(
        "--comparators",
        action="store_true",
        help="include the ten non-DFAST regional comparators",
    )
    ap.add_argument("--tickers", nargs="*", help="restrict to these tickers")
    ap.add_argument(
        "--no-crosscheck",
        action="store_true",
        help="skip the comparison against the EDGAR panel",
    )
    args = ap.parse_args(argv)

    config.ensure_dirs()
    started = dt.datetime.now(dt.UTC)

    cached = cdr.cached_periods()
    if not cached:
        log.error("no cached periods; run scripts/fetch_call.py first")
        return 1
    until = args.until or cdr.quarter_end(cached[-1])
    periods = [
        p for p in cached if args.since <= cdr.quarter_end(p) <= until
    ]
    if not periods:
        log.error("no cached periods inside %s..%s", args.since, until)
        return 1

    tiers = {"dfast", "ihc", "inactive"}
    if args.comparators:
        tiers.add("comparator")
    holdings = tuple(h for h in config.HOLDINGS if h.tier in tiers)
    if args.tickers:
        wanted = {t.upper() for t in args.tickers}
        holdings = tuple(h for h in holdings if h.ticker.upper() in wanted)
    if not holdings:
        log.error("no holdings selected")
        return 1

    log.info(
        "building %d periods (%s..%s) for %d firms",
        len(periods),
        periods[0],
        periods[-1],
        len(holdings),
    )
    charters, holding_panel = panel.build(periods, holdings)
    if holding_panel.is_empty():
        log.error("the build produced no rows")
        return 1

    out = config.OUT
    holding_panel.write_parquet(out / "call_panel.parquet")
    holding_panel.write_csv(out / "call_panel.csv")
    charters.write_parquet(out / "call_panel_charters.parquet")
    coverage(holding_panel).write_csv(out / "call_panel_coverage.csv")

    flagged = flags(holding_panel)
    if not flagged.is_empty():
        flagged.write_csv(out / "call_panel_flags.csv")

    log.info(
        "panel: %d bank-quarters, %d firms, %d columns, %d populated cells",
        holding_panel.height,
        holding_panel["ticker"].n_unique(),
        len(holding_panel.columns),
        _filled_cells(holding_panel),
    )
    log.info(
        "charters: %d charter-quarters across %d charters",
        charters.height,
        charters["rssd"].n_unique(),
    )
    if "rcc_residual" in charters.columns:
        exact = int(
            (charters["rcc_residual"].abs() <= PARTITION_TOLERANCE_USD).sum()
        )
        checked = int(charters["rcc_residual"].is_not_null().sum())
        log.info(
            "RC-C partition ties exactly for %d of %d charter-quarters (%.1f%%)",
            exact,
            checked,
            exact / checked * 100 if checked else 0.0,
        )

    diff_rows = 0
    if not args.no_crosscheck:
        edgar_path = out / "panel.parquet"
        if not edgar_path.exists():
            log.warning("no EDGAR panel at %s; skipping the cross-check", edgar_path)
        else:
            edgar = pl.read_parquet(edgar_path)
            diff = crosscheck.compare(edgar, holding_panel)
            if diff.is_empty():
                log.warning("the two panels share no bank-quarters")
            else:
                diff.write_csv(out / "source_diff.csv")
                summary = crosscheck.summarise(diff)
                summary.write_csv(out / "source_diff_summary.csv")
                crosscheck.coverage_gained(edgar, holding_panel).write_csv(
                    out / "source_coverage.csv"
                )
                diff_rows = diff.height
                verdicts = dict(
                    diff.group_by("verdict").len().iter_rows()  # type: ignore[arg-type]
                )
                log.info(
                    "cross-check: %d comparisons, %d agree, %d differ, %d one-source",
                    diff.height,
                    verdicts.get("agree", 0),
                    verdicts.get("differ", 0),
                    verdicts.get("one_source_only", 0),
                )
                shaky = crosscheck.unstable(summary)
                if not shaky.is_empty():
                    log.info(
                        "%d bank-variable pairs have an unstable ratio; see "
                        "source_diff_summary.csv",
                        shaky.height,
                    )

    info = {
        "built_at": started.isoformat(),
        "commit": _git_commit(),
        "source": "FFIEC CDR bulk Call Report data, FFIEC NIC structure data",
        "since": args.since.isoformat(),
        "until": until.isoformat(),
        "periods": len(periods),
        "first_period": periods[0],
        "last_period": periods[-1],
        "firms": holding_panel["ticker"].n_unique(),
        "bank_quarters": holding_panel.height,
        "charter_quarters": charters.height,
        "columns": len(holding_panel.columns),
        "populated_cells": _filled_cells(holding_panel),
        "flags": 0 if flagged.is_empty() else flagged.height,
        "crosscheck_rows": diff_rows,
        "unit_scale": mdrm.UNIT_SCALE,
        "comparators": bool(args.comparators),
        "tickers": args.tickers or None,
    }
    (out / "call_panel_build_info.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )
    log.info("wrote %s", out / "call_panel.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
