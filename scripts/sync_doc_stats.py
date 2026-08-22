"""Keep the counts quoted in prose in step with the panels they describe.

The README and the wiki quote figures -- row counts, column counts, the window,
the number of tests -- that are properties of a *build*, not of the text.  They
drift the moment a panel is rebuilt, and they drift silently: nothing renders
differently when a document claims 3,741 bank-quarters and the parquet beside it
holds 3,763.  That happened, which is why this exists.

The fix is to stop writing them by hand.  Every such figure is wrapped in a
sentinel::

    <!--stats:ffiec-rows-->3,763<!--/stats-->

and this script rewrites what sits between the markers.  Prose is untouched, so
a sentence keeps reading like a sentence; only the number moves.

What can be automated
---------------------
The test is whether the artefacts carry *both halves* of a figure.  "72 of the
72 non-filing depositories resolve, for 2,749 quarterly rows" is safe:
``sought``, ``resolved`` and ``rows`` are all recorded by the same build, so the
sentence stays internally consistent however it moves.

The 2013 build record fails that test and is left alone.  Its figures are paired
with counterparts nothing records -- ``1,648 x 456``, ``29 of 31 banks``, quality
flags ``363 -> 330``.  Refreshing one side of such a pair would state something
that was never true of any build, which is worse than a stale number.

Where the numbers come from
---------------------------
``data/*.parquet`` are committed, so CI can read them directly and they are the
authority for anything the panel itself carries: rows, columns, distinct banks,
first and last quarter.

Figures only a full build knows -- charter-quarters, lineage sizes, the thrift
backfill -- live in ``data/out/call_panel_build_info.json``, which is gitignored
because ``data/out/`` is.  ``--refresh-stats`` copies the handful that documents
quote into ``data/panel_stats.json``, which *is* committed.  CI then cross-checks
that file against the committed panels on every fact they share, so a stale
``panel_stats.json`` is caught rather than propagated.

Usage
-----
    python scripts/sync_doc_stats.py --check              # CI: fail if stale
    python scripts/sync_doc_stats.py --write              # rewrite in place
    python scripts/sync_doc_stats.py --write --wiki ../wiki
    python scripts/sync_doc_stats.py --refresh-stats      # after a full build
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
STATS_PATH = ROOT / "data" / "panel_stats.json"

# ``<!--stats:key-->value<!--/stats-->`` on a single line.  Single-line on
# purpose: these sit inside markdown table cells, which cannot hold block
# content, and a greedy multi-line match would swallow whole tables.
SENTINEL = re.compile(r"(<!--stats:([a-z0-9-]+)-->)(.*?)(<!--/stats-->)")

# Facts carried in panel_stats.json that are also derivable from the committed
# panels.  Disagreement means the stats file is stale.
CROSS_CHECKED = ("rows", "columns", "banks", "first_quarter", "last_quarter")


def _panel_facts(path: Path, entity_col: str = "ticker") -> dict[str, Any]:
    """Everything about a panel that the panel itself can answer."""
    lf = pl.scan_parquet(path)
    quarters = (
        lf.select(pl.col("quarter")).unique().collect().get_column("quarter").sort()
    )
    return {
        "rows": lf.select(pl.len()).collect().item(),
        "columns": len(lf.collect_schema().names()),
        "banks": lf.select(pl.col(entity_col).n_unique()).collect().item(),
        "first_quarter": quarters[0],
        "last_quarter": quarters[-1],
    }


def _provenance_columns(path: Path) -> int:
    """How many of the EDGAR panel's columns are ``__source`` provenance."""
    names = pl.scan_parquet(path).collect_schema().names()
    return sum(1 for n in names if n.endswith("__source"))


def _test_count() -> int | None:
    """Collected test count, or None if collection is unavailable.

    Best-effort by design: a documentation sync must not fail because the test
    environment is not installed.
    """
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", "tests/"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    m = re.search(r"^(\d+) tests? collected", out.stdout, re.MULTILINE)
    return int(m.group(1)) if m else None


def _build_record(info: dict) -> dict[str, Any]:
    """The figures documents quote from one ``call_panel_build_info.json``."""
    lineage = info.get("lineage")
    lineage = lineage if isinstance(lineage, dict) else {}
    partition = info.get("partition") or {}
    thrift = info.get("thrift_gap")
    thrift = thrift if isinstance(thrift, dict) else {}

    delta = info.get("coverage_delta") or {}

    record: dict[str, Any] = {
        "bank_quarters": info["bank_quarters"],
        "charter_quarters": info["charter_quarters"],
        "columns": info["columns"],
        "firms": info["firms"],
        "periods": info["periods"],
        "flag_counts": info.get("flag_counts") or {},
        # What the lineage extension added, against the own-subtree baseline.
        "coverage_delta": {
            "before": delta.get("bank_quarters_before"),
            "after": delta.get("bank_quarters_after"),
            "with_predecessors": delta.get("bank_quarters_with_predecessors"),
        },
        "partition": {
            "checked": partition.get("charter_quarters_checked"),
            "tied": partition.get("charter_quarters_tied"),
            "holding_not_checked": partition.get("holding_rows_not_checked"),
        },
        "lineage": {
            "predecessors": lineage.get("predecessors"),
            "fdic_assisted": (lineage.get("by_type") or {}).get("fdic_assisted"),
            "acquisitions": lineage.get("acquisitions_by_relationship"),
        },
    }
    if thrift:
        record["thrift_gap"] = {
            "sought": thrift.get("sought"),
            "resolved": thrift.get("resolved"),
            "rows": thrift.get("rows"),
            "identities": thrift.get("identities") or {},
        }
    return record


def refresh_stats() -> dict:
    """Rebuild ``panel_stats.json`` from a completed build's artefacts.

    Reads the gitignored build info for the facts the committed panels cannot
    answer, and re-derives everything else from the panels so the file is never
    the only witness to a number.

    ``data/out/`` holds one build at a time, and the documents compare two --
    the shipped ``--tfr`` panel against the plain one, because the thrift
    backfill changes the row counts and costs the holding-level partition
    check.  Each refresh files the build it finds under the variant its own
    ``tfr`` flag names and carries the other forward, so::

        build_call_panel.py        && sync_doc_stats.py --refresh-stats
        build_call_panel.py --tfr  && sync_doc_stats.py --refresh-stats

    records both.
    """
    prior: dict[str, Any] = (
        json.loads(STATS_PATH.read_text(encoding="utf-8"))
        if STATS_PATH.exists()
        else {}
    )

    stats: dict[str, Any] = {
        "edgar": _panel_facts(ROOT / "data" / "panel.parquet"),
        "ffiec": _panel_facts(ROOT / "data" / "call_panel.parquet"),
    }
    stats["edgar"]["provenance_columns"] = _provenance_columns(
        ROOT / "data" / "panel.parquet"
    )

    # Carry both build records forward, then overwrite whichever one the
    # artefacts on disk describe.
    for variant in ("build_tfr", "build_plain"):
        if variant in prior:
            stats[variant] = prior[variant]

    info_path = ROOT / "data" / "out" / "call_panel_build_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        variant = "build_tfr" if info.get("tfr") else "build_plain"
        stats[variant] = _build_record(info)

    tests = _test_count()
    stats["tests"] = tests if tests is not None else prior.get("tests")
    if stats["tests"] is None:
        del stats["tests"]

    STATS_PATH.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def load_stats(verify: bool = True) -> dict:
    """Read ``panel_stats.json``, checking it still describes the shipped panels."""
    if not STATS_PATH.exists():
        raise SystemExit(
            f"{STATS_PATH.relative_to(ROOT)} is missing -- run with --refresh-stats"
        )
    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))

    if not verify:
        return stats

    # The stats file must agree with the panels on every fact both carry.  This
    # is what stops a stale file from being copied into the documents as if it
    # were current.
    live = {
        "edgar": _panel_facts(ROOT / "data" / "panel.parquet"),
        "ffiec": _panel_facts(ROOT / "data" / "call_panel.parquet"),
    }
    for panel, facts in live.items():
        for key in CROSS_CHECKED:
            recorded, actual = stats.get(panel, {}).get(key), facts[key]
            if recorded != actual:
                raise SystemExit(
                    f"panel_stats.json is stale: {panel}.{key} says {recorded!r}, "
                    f"data/{'call_panel' if panel == 'ffiec' else 'panel'}.parquet "
                    f"says {actual!r}.  Run --refresh-stats."
                )

    # The shipped panel is the --tfr build, so that build record must describe
    # the committed parquet.  Without this a stale build record would be copied
    # into the documents as confidently as a current one.
    shipped = (stats.get("build_tfr") or {}).get("bank_quarters")
    if shipped is not None and shipped != live["ffiec"]["rows"]:
        raise SystemExit(
            f"panel_stats.json is stale: build_tfr.bank_quarters says {shipped!r}, "
            f"data/call_panel.parquet has {live['ffiec']['rows']!r} rows.  "
            f"Rebuild with --tfr and run --refresh-stats."
        )
    return stats


def values(stats: dict) -> dict[str, str]:
    """The substitution table: sentinel key -> rendered text."""
    e, f = stats["edgar"], stats["ffiec"]

    def window(p: dict) -> str:
        return f"{p['first_quarter']} – {p['last_quarter']}"

    out = {
        "edgar-rows": f"{e['rows']:,}",
        "edgar-cols": f"{e['columns']:,}",
        "edgar-banks": f"{e['banks']}",
        "edgar-window": window(e),
        "edgar-extent": (
            f"{e['rows']:,} rows, {e['banks']} banks, {e['columns']} columns"
        ),
        "edgar-provenance-cols": f"{e.get('provenance_columns', 0):,}",
        "ffiec-rows": f"{f['rows']:,}",
        "ffiec-cols": f"{f['columns']:,}",
        "ffiec-banks": f"{f['banks']}",
        "ffiec-window": window(f),
        "ffiec-extent": (
            f"{f['rows']:,} rows, {f['banks']} firms, {f['columns']} columns"
        ),
    }
    if "tests" in stats:
        out["tests"] = f"{stats['tests']}"

    # The shipped panel is the --tfr build, so its figures are the unqualified
    # ones.  The plain build is the comparison the thrift-gap tables are built
    # on, and is prefixed.
    tfr = stats.get("build_tfr") or {}
    plain = stats.get("build_plain") or {}

    def emit(prefix: str, rec: dict) -> None:
        if not rec:
            return
        if rec.get("bank_quarters") is not None:
            out[f"{prefix}bank-quarters"] = f"{rec['bank_quarters']:,}"
        if rec.get("charter_quarters") is not None:
            out[f"{prefix}charter-quarters"] = f"{rec['charter_quarters']:,}"
        if rec.get("periods") is not None:
            out[f"{prefix}periods"] = f"{rec['periods']}"
        if rec.get("columns") is not None:
            out[f"{prefix}build-columns"] = f"{rec['columns']:,}"
        if rec.get("firms") is not None:
            out[f"{prefix}firms"] = f"{rec['firms']}"
        delta = rec.get("coverage_delta") or {}
        if delta.get("before") is not None:
            out[f"{prefix}delta-before"] = f"{delta['before']:,}"
        if delta.get("after") is not None:
            out[f"{prefix}delta-after"] = f"{delta['after']:,}"
        if delta.get("before") is not None and delta.get("after") is not None:
            out[f"{prefix}delta-added"] = f"{delta['after'] - delta['before']:,}"
        if delta.get("with_predecessors") is not None:
            out[f"{prefix}delta-with-predecessors"] = (
                f"{delta['with_predecessors']:,}"
            )
        part = rec.get("partition") or {}
        if part.get("checked") is not None:
            out[f"{prefix}partition-checked"] = f"{part['checked']:,}"
        if part.get("tied") is not None:
            out[f"{prefix}partition-tied"] = f"{part['tied']:,}"
            if part.get("checked"):
                out[f"{prefix}partition-tied-pct"] = (
                    f"{part['tied'] / part['checked'] * 100:.1f}"
                )
        if part.get("holding_not_checked") is not None:
            out[f"{prefix}partition-unchecked"] = f"{part['holding_not_checked']:,}"
        for flag, n in (rec.get("flag_counts") or {}).items():
            out[f"{prefix}flag-{flag.replace('_', '-')}"] = f"{n:,}"

    emit("ffiec-", tfr)
    emit("plain-", plain)

    # Lineage and the thrift backfill are properties of the universe and the
    # registries, not of which build ran -- take them from whichever record has
    # them, preferring the shipped one.
    for rec in (tfr, plain):
        for key, value in (rec.get("lineage") or {}).items():
            if value is not None:
                out.setdefault(f"lineage-{key.replace('_', '-')}", f"{value:,}")
        thrift = rec.get("thrift_gap") or {}
        for key in ("sought", "resolved", "rows"):
            if thrift.get(key) is not None:
                out.setdefault(f"thrift-{key}", f"{thrift[key]:,}")
        for name, pair in (thrift.get("identities") or {}).items():
            if isinstance(pair, list) and len(pair) == 2:
                out.setdefault(
                    f"thrift-identity-{name.replace('_', '-')}",
                    f"{pair[0]:,} / {pair[1]:,}",
                )
    return out


def sync_file(path: Path, table: dict[str, str], write: bool) -> list[str]:
    """Rewrite one file's sentinels.  Returns a description of each change."""
    original = path.read_text(encoding="utf-8")
    changes: list[str] = []

    def replace(m: re.Match[str]) -> str:
        open_tag, key, current, close_tag = m.groups()
        if key not in table:
            changes.append(f"{path.name}: unknown sentinel key {key!r}")
            return m.group(0)
        wanted = table[key]
        if current != wanted:
            changes.append(f"{path.name}: {key}: {current!r} -> {wanted!r}")
        return f"{open_tag}{wanted}{close_tag}"

    updated = SENTINEL.sub(replace, original)
    if write and updated != original:
        path.write_text(updated, encoding="utf-8")
    return changes


def targets(wiki: Path | None) -> list[Path]:
    paths = [ROOT / "README.md"]
    if wiki:
        paths.extend(sorted(wiki.glob("*.md")))
    return [p for p in paths if p.exists()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="fail if anything is stale")
    mode.add_argument("--write", action="store_true", help="rewrite sentinels in place")
    mode.add_argument(
        "--refresh-stats",
        action="store_true",
        help="regenerate data/panel_stats.json from a completed build",
    )
    ap.add_argument("--wiki", type=Path, help="a checkout of the wiki to sync too")
    args = ap.parse_args()

    if args.refresh_stats:
        stats = refresh_stats()
        print(f"wrote {STATS_PATH.relative_to(ROOT)}")
        print(json.dumps(stats, indent=2))
        return 0

    table = values(load_stats())
    changes: list[str] = []
    for path in targets(args.wiki):
        changes.extend(sync_file(path, table, write=args.write))

    unknown = [c for c in changes if "unknown sentinel" in c]

    if not changes:
        print("documentation counts are current")
        return 0

    for c in changes:
        print(("stale: " if "unknown" not in c else "") + c)

    if unknown:
        print("\nunknown sentinel keys are always an error -- fix the key or the table")
        return 1
    if args.check:
        print("\nrun: python scripts/sync_doc_stats.py --write")
        return 1
    print(f"\nupdated {len(changes)} value(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
