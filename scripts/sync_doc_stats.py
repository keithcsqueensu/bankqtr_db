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

Two facts are deliberately *not* automated:

* Figures inside a narrative build record -- what the 2001 extension measured at
  the time it was written.  Those are observations with a date attached, not
  descriptions of the current panel, and rewriting them would be falsifying a
  record rather than updating a count.
* Anything derived from a full build's ``data/out/`` artefacts that is not
  reachable from the committed panels.  Those come via ``panel_stats.json``
  instead -- see below.

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


def refresh_stats() -> dict:
    """Rebuild ``panel_stats.json`` from a completed build's artefacts.

    Run after ``build_call_panel.py``.  Reads the gitignored build info for the
    facts the committed panels cannot answer, and re-derives everything else
    from the panels so the file is never the only witness to a number.
    """
    stats: dict[str, Any] = {
        "edgar": _panel_facts(ROOT / "data" / "panel.parquet"),
        "ffiec": _panel_facts(ROOT / "data" / "call_panel.parquet"),
    }
    stats["edgar"]["provenance_columns"] = _provenance_columns(
        ROOT / "data" / "panel.parquet"
    )

    info_path = ROOT / "data" / "out" / "call_panel_build_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        stats["ffiec"]["charter_quarters"] = info["charter_quarters"]
        stats["ffiec"]["periods"] = info["periods"]
        lineage = info.get("lineage") or {}
        stats["lineage"] = {
            "predecessors": lineage.get("predecessors"),
            "fdic_assisted": (lineage.get("by_type") or {}).get("fdic_assisted"),
        }
    elif STATS_PATH.exists():
        # No build artefacts here -- keep what is already recorded rather than
        # dropping facts the documents depend on.
        prior = json.loads(STATS_PATH.read_text(encoding="utf-8"))
        for key in ("charter_quarters", "periods"):
            if key in prior.get("ffiec", {}):
                stats["ffiec"][key] = prior["ffiec"][key]
        if "lineage" in prior:
            stats["lineage"] = prior["lineage"]

    tests = _test_count()
    if tests is not None:
        stats["tests"] = tests
    elif STATS_PATH.exists():
        prior = json.loads(STATS_PATH.read_text(encoding="utf-8"))
        if "tests" in prior:
            stats["tests"] = prior["tests"]

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
    if "charter_quarters" in f:
        out["ffiec-charter-quarters"] = f"{f['charter_quarters']:,}"
    if "periods" in f:
        out["ffiec-periods"] = f"{f['periods']}"
    if "tests" in stats:
        out["tests"] = f"{stats['tests']}"
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
