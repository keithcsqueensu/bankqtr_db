"""End-to-end build: EDGAR facts -> reconciled bank-quarter panel.

uv run python scripts/build_panel.py --since 2020-01-01
uv run python scripts/build_panel.py --since 2020-01-01 --html-fallback
uv run python scripts/build_panel.py --since 2020-01-01 --html-fallback --ir
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import subprocess
import sys
import time
from collections import Counter

import polars as pl

from bankqtr_db import (
    filings,
    html_fallback,
    ir,
    ir_extract,
    panel,
    reconcile,
    taxonomy,
    xbrl,
)
from bankqtr_db.config import OUT, ensure_dirs, universe

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("build")


def _filled_cells(df: pl.DataFrame) -> int:
    """Populated numeric cells, for reporting what each fallback added."""
    numeric = [c for c in df.columns if df.schema[c] in (pl.Float64, pl.Int64)]
    if not numeric:
        return 0
    return int(sum(df[c].is_not_null().sum() for c in numeric))


def _git_commit() -> str | None:
    """The commit the panel was built from, when there is one."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=OUT.parent.parent,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def build_info(
    panel_df: pl.DataFrame,
    facts: pl.DataFrame,
    html: pl.DataFrame | None,
    ir_rows: pl.DataFrame | None,
    *,
    banks: tuple[object, ...],
    args: argparse.Namespace,
) -> dict[str, object]:
    """Everything needed to tell one published panel from another.

    A panel carries no date, no window and no build settings of its own, so a
    consumer pinning a release has no way to know what changed between two of
    them -- or whether a difference in their own results came from their code
    or from a data refresh.  This is that record, written beside the panel and
    published with it.

    ``provenance`` is the useful part for anyone testing against the data: it
    counts populated cells by where the number came from, so a drop in
    ``ir_supplement`` between releases is visible without diffing the panel.
    """
    provenance: Counter[str] = Counter()
    for column in (c for c in panel_df.columns if c.endswith("__source")):
        provenance.update(panel_df[column].drop_nulls().to_list())

    period = panel_df["period"] if "period" in panel_df.columns else None
    return {
        "built_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "settings": {
            "since": args.since,
            "tickers": args.tickers or None,
            "comparators": bool(args.comparators),
            "html_fallback": bool(args.html_fallback),
            "html_forms": args.html_forms if args.html_fallback else None,
            "ir": bool(args.ir),
            "ir_doc_types": args.ir_doc_types if args.ir else None,
        },
        "panel": {
            "banks": len(banks),
            "bank_quarters": panel_df.height,
            "columns": panel_df.width,
            "period_min": str(period.min()) if period is not None else None,
            "period_max": str(period.max()) if period is not None else None,
        },
        "inputs": {
            "xbrl_facts": facts.height,
            "html_rows": 0 if html is None else html.height,
            "ir_rows": 0 if ir_rows is None else ir_rows.height,
        },
        "provenance": dict(provenance.most_common()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2020-01-01")
    ap.add_argument("--tickers", default="", help="comma-separated subset")
    ap.add_argument(
        "--comparators",
        action="store_true",
        help="include non-DFAST regional comparators",
    )
    ap.add_argument(
        "--html-fallback",
        action="store_true",
        help="parse filing HTML for variables XBRL does not carry",
    )
    ap.add_argument(
        "--html-forms", default="10-K", help="forms to run the HTML fallback over"
    )
    ap.add_argument(
        "--all-tags",
        action="store_true",
        help="keep every tag, not just those named by variables",
    )
    ap.add_argument(
        "--ir",
        action="store_true",
        help="fold in IR supplements already cached by scripts/fetch_ir.py",
    )
    ap.add_argument(
        "--ir-doc-types",
        default="supplement,presentation",
        help="which cached IR document types to parse",
    )
    ap.add_argument("--out-prefix", default="panel")
    args = ap.parse_args()

    ensure_dirs()
    since = dt.date.fromisoformat(args.since)
    banks = universe(include_comparators=args.comparators)
    if args.tickers:
        wanted = {t.strip().upper() for t in args.tickers.split(",")}
        banks = tuple(b for b in banks if b.ticker in wanted)
    if not banks:
        print("no banks selected", file=sys.stderr)
        return 2

    tags = None if args.all_tags else xbrl.relevant_tags()
    print(f"universe: {len(banks)} banks, since {since}", flush=True)

    # ---- 1. extract -------------------------------------------------------
    t0 = time.monotonic()
    frames = []
    for bank in banks:
        df = xbrl.extract_bank(bank, since=since, tags=tags)
        frames.append(df)
        print(f"  {bank.ticker:6s} {df.height:7d} facts", flush=True)
    facts = pl.concat([f for f in frames if not f.is_empty()], how="vertical_relaxed")
    facts = xbrl.deduplicate(facts)
    print(
        f"extracted {facts.height:,} deduplicated facts in {time.monotonic() - t0:.0f}s",
        flush=True,
    )

    # An unmapped member means a bank's disclosure is silently absent from the
    # panel, so write the list out as a work queue rather than just a count.
    unmapped = taxonomy.unmapped()
    if unmapped:
        pl.DataFrame(
            unmapped, schema=["kind", "member", "n_seen"], orient="row"
        ).write_csv(OUT / f"{args.out_prefix}_unmapped_members.csv")
        print(
            f"  {len(unmapped)} unmapped dimension members "
            f"-> {args.out_prefix}_unmapped_members.csv",
            flush=True,
        )

    # ---- 2. panel ---------------------------------------------------------
    raw_panel = panel.build_panel(facts, since=since)
    print(
        f"panel: {raw_panel.height} bank-quarters x {raw_panel.width} columns",
        flush=True,
    )

    # ---- 3. HTML fallback -------------------------------------------------
    html = None
    if args.html_fallback:
        forms = tuple(f.strip() for f in args.html_forms.split(","))
        targets = []
        for bank in banks:
            targets.extend(filings.list_filings(bank, since=since, forms=forms))
        print(f"html fallback over {len(targets)} filings...", flush=True)
        html = html_fallback.extract_filings(targets)
        print(f"  parsed {html.height} rows", flush=True)
        if not html.is_empty():
            html.write_parquet(OUT / f"{args.out_prefix}_html_raw.parquet")

    # ---- 3b. IR supplements -----------------------------------------------
    # Reads whatever ``scripts/fetch_ir.py`` has already cached under data/ir/;
    # this step never goes to the network, so a build stays reproducible.
    ir_rows = None
    if args.ir:
        pairs = []
        for bank in banks:
            for doc in ir.discover_cached(bank, since=since):
                pairs.append((doc, doc.path))
        wanted = tuple(t.strip() for t in args.ir_doc_types.split(",") if t.strip())
        pairs = [(d, p) for d, p in pairs if d.doc_type in wanted]
        print(f"ir supplements over {len(pairs)} cached documents...", flush=True)
        ir_rows = ir_extract.extract_documents(pairs)
        print(f"  parsed {ir_rows.height} rows", flush=True)
        if not ir_rows.is_empty():
            ir_rows.write_parquet(OUT / f"{args.out_prefix}_ir_raw.parquet")

    # ---- 4. reconcile -----------------------------------------------------
    final = reconcile.reconcile(raw_panel, html)
    if ir_rows is not None and not ir_rows.is_empty():
        before = _filled_cells(final)
        final = reconcile.merge_ir(final, ir_rows)
        # Ratios are computed on the XBRL frame, so anything the supplements
        # just filled has no ratio yet.
        final = reconcile.refresh_ratios(final)
        print(f"  ir filled {_filled_cells(final) - before:,} panel cells", flush=True)

    coverage = reconcile.coverage_report(final, banks=banks)
    flags = reconcile.quality_flags(final)
    gaps = reconcile.gap_summary(coverage)

    # ---- 5. write ---------------------------------------------------------
    facts.write_parquet(OUT / f"{args.out_prefix}_facts.parquet")
    final.write_parquet(OUT / f"{args.out_prefix}.parquet")
    final.write_csv(OUT / f"{args.out_prefix}.csv")
    if not coverage.is_empty():
        coverage.write_csv(OUT / f"{args.out_prefix}_coverage.csv")
    if not gaps.is_empty():
        gaps.write_csv(OUT / f"{args.out_prefix}_gaps.csv")
    if not flags.is_empty():
        flags.write_csv(OUT / f"{args.out_prefix}_flags.csv")

    info = build_info(final, facts, html, ir_rows, banks=banks, args=args)
    (OUT / f"{args.out_prefix}_build_info.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\nwrote {OUT / args.out_prefix}.parquet", flush=True)
    if not gaps.is_empty():
        print("\nweakest coverage:")
        print(gaps.head(12))
    if not flags.is_empty():
        print(f"\n{flags.height} bank-quarters carry quality flags")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
