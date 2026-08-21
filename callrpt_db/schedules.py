"""Bulk Call Report TSV -> long frame.

File shape
----------
Each schedule is one tab-delimited text file inside the period's zip.  Row 1 is
MDRM codes, **row 2 is human labels**, and the data starts at row 3 -- so a
reader that treats row 2 as data silently acquires one junk institution whose
every field is a caption.  Large schedules are split across several files
(``RCO ... (1 of 2).txt``), and the split is by *columns*, not by rows, so the
parts have to be merged on ``IDRSSD`` rather than concatenated.

The label row is not junk to be skipped, though: it is the only place the codes
are described, and :func:`labels` returns it so a mapping can be checked
against what the form actually calls an item.

Values are integers in **thousands of dollars**, with an empty field meaning
"not reported".  Two non-numeric values appear: ``CONF`` marks an item withheld
as confidential, and ``NR`` marks one not required of that filer.  Both are
read as null -- treating ``CONF`` as zero would report a bank's confidential
line as an absence of exposure.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import logging
import re
import zipfile
from pathlib import Path

import polars as pl

from . import cdr, mdrm

log = logging.getLogger(__name__)

# The narrative schedule in some mid-2000s files carries a single field longer
# than the csv module's default limit, and the reader raises rather than
# truncating.  Those files are never wanted, but the limit is lifted so that a
# caller asking for every schedule still gets an answer.
csv.field_size_limit(2**31 - 1)

# The roster file: every institution that filed for the period.
POR_MARKER = "Bulk POR"
_SCHEDULE_RE = re.compile(r"Call Schedule ([A-Z0-9]+) \d{8}")

# Values that are present but are not numbers.
NON_NUMERIC = frozenset({"CONF", "NR", "N/A", "NA", ""})

LONG_SCHEMA: dict[str, type[pl.DataType] | pl.DataType] = {
    "rssd": pl.Int64,
    "period": pl.Date,
    "schedule": pl.Utf8,
    "code": pl.Utf8,
    "prefix": pl.Utf8,
    "item": pl.Utf8,
    "value": pl.Float64,
}


def empty_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=LONG_SCHEMA)


def _open_text(z: zipfile.ZipFile, name: str) -> io.TextIOWrapper:
    return io.TextIOWrapper(z.open(name), encoding="utf-8-sig", errors="replace")


def _to_float(raw: str) -> float | None:
    v = raw.strip()
    if v in NON_NUMERIC:
        return None
    try:
        return float(v)
    except ValueError:
        # A few text-valued items (audit indicator, filing type) live in the
        # same files.  They are never mapped to a panel column, so dropping
        # them here is cheaper than filtering upstream.
        return None


def roster(path: Path) -> pl.DataFrame:
    """Everyone who filed in this period: ``rssd``, ``name``, ``form``.

    This is the authority on who filed.  ``nic.call_filers`` intersects the
    holding-company descendant walk against it, so a charter that NIC still
    lists but that stopped filing simply drops out of the quarter instead of
    contributing nulls.

    ``form`` is the 031/041/051 filing type, which decides how finely RC-C is
    broken out -- see :class:`mdrm.ItemSpec`.  It is read from the file rather
    than inferred from which items are populated, because a 041 filer can carry
    a 031 item as an explicit zero.
    """
    period = _period_of(path)
    rows: list[tuple[int, str, str]] = []
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if POR_MARKER in n]
        if not names:
            raise RuntimeError(f"{path.name} has no POR roster file")
        with _open_text(z, names[0]) as fh:
            reader = csv.reader(fh, delimiter="\t")
            header = [c.strip().strip('"') for c in next(reader)]
            name_col = _column(header, "Financial Institution Name", 5)
            form_col = _column(header, "Financial Institution Filing Type", 10)
            for row in reader:
                if not row or not row[0].strip().isdigit():
                    continue
                rows.append(
                    (
                        int(row[0]),
                        row[name_col].strip() if len(row) > name_col else "",
                        row[form_col].strip() if len(row) > form_col else "",
                    )
                )
    frame = pl.DataFrame(
        rows, schema={"rssd": pl.Int64, "name": pl.Utf8, "form": pl.Utf8}, orient="row"
    )
    return frame.with_columns(pl.lit(period).cast(pl.Date).alias("period"))


def _column(header: list[str], name: str, fallback: int) -> int:
    try:
        return header.index(name)
    except ValueError:
        return fallback


def labels(path: Path) -> dict[str, tuple[str, str]]:
    """MDRM code -> (schedule, label), read from each file's second row."""
    out: dict[str, tuple[str, str]] = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            m = _SCHEDULE_RE.search(name)
            if not m:
                continue
            schedule = m.group(1)
            with _open_text(z, name) as fh:
                reader = csv.reader(fh, delimiter="\t")
                try:
                    codes = [c.strip().strip('"') for c in next(reader)]
                    text = next(reader)
                except StopIteration:
                    continue
            for code, label in zip(codes, text):
                if code and code != "IDRSSD":
                    out.setdefault(code, (schedule, label.strip()))
    return out


def read_period(
    path: Path,
    *,
    rssds: set[int] | None = None,
    codes: dict[str, set[str]] | None = None,
) -> pl.DataFrame:
    """One period's zip -> long frame, filtered to the wanted banks and codes.

    ``rssds`` of ``None`` keeps every filer, which is ~4,400 institutions and
    is only useful for industry-wide work; a universe build passes the set it
    resolved.  ``codes`` defaults to :func:`mdrm.wanted_codes`.
    """
    period = _period_of(path)
    wanted = codes if codes is not None else mdrm.wanted_codes()
    rows: list[tuple[int, str, str, str, str, float]] = []

    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            m = _SCHEDULE_RE.search(name)
            if not m:
                continue
            schedule = m.group(1)
            keep = wanted.get(schedule)
            # A schedule the mapping does not mention is skipped outright.  It
            # used to be read in full, which cost a minute per quarter and, in
            # 2006Q4, a crash on the narrative schedule's text fields.
            if keep is None or not keep:
                continue
            with _open_text(z, name) as fh:
                reader = csv.reader(fh, delimiter="\t")
                try:
                    header = [c.strip().strip('"') for c in next(reader)]
                    next(reader)  # the label row is not data
                except StopIteration:
                    continue
                # Column indices worth reading at all, resolved once per file.
                cols = [
                    (i, code)
                    for i, code in enumerate(header)
                    if code and code != "IDRSSD" and (keep is None or code in keep)
                ]
                if not cols:
                    continue
                for row in reader:
                    if not row or not row[0].strip().isdigit():
                        continue
                    rssd = int(row[0])
                    if rssds is not None and rssd not in rssds:
                        continue
                    for i, code in cols:
                        if i >= len(row):
                            continue
                        value = _to_float(row[i])
                        if value is None:
                            continue
                        rows.append(
                            (rssd, schedule, code, code[:4], code[4:], value)
                        )

    if not rows:
        return empty_frame()
    frame = pl.DataFrame(
        rows,
        schema=["rssd", "schedule", "code", "prefix", "item", "value"],
        orient="row",
    )
    return frame.with_columns(pl.lit(period).cast(pl.Date).alias("period")).select(
        list(LONG_SCHEMA)
    )


def _period_of(path: Path) -> dt.date:
    stem = path.stem
    if stem.startswith("call_"):
        return cdr.quarter_end(stem.removeprefix("call_"))
    raise ValueError(f"cannot infer a period from {path.name}")


def resolve_items(
    long: pl.DataFrame,
    specs: tuple[mdrm.ItemSpec, ...],
    *,
    forms: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Long frame -> one column per :class:`mdrm.ItemSpec`, in dollars.

    Two independent resolutions happen per bank-quarter.

    **Prefix.**  The spec's prefixes are tried in order and the first one any
    component appears under wins.  That per-row choice is what handles a filing
    reporting its totals on ``RCFD`` and its real-estate detail on ``RCON``, and
    a bank that has no ``RCFD`` at all.

    **Form.**  Where a spec lists per-form items, the bank's own declared
    filing type selects between them -- see :class:`mdrm.ItemSpec` for why this
    cannot be decided by which items happen to be populated.  ``forms`` is the
    roster frame; without it every row takes the default item list, which is
    correct for 031 and 041 and wrong only for the short form.

    Within the winning item list, present components are summed and absent ones
    count as zero; if none is present the column is null, not zero.  Values are
    multiplied out of thousands into dollars, matching the EDGAR panel's units.
    """
    if long.is_empty():
        return pl.DataFrame(schema={"rssd": pl.Int64, "period": pl.Date})

    key = ["rssd", "period"]
    base = long.select(key).unique().sort(key)
    if forms is not None and not forms.is_empty():
        base = base.join(forms.select([*key, "form"]).unique(subset=key), on=key, how="left")
    else:
        base = base.with_columns(pl.lit(None, dtype=pl.Utf8).alias("form"))

    def summed(schedules: tuple[str, ...], full_codes: list[str], out: str) -> pl.DataFrame:
        """Sum ``full_codes`` per bank-quarter, reading the listed schedules.

        Where a spec names several schedules, a code that a filing carries in
        more than one of them is taken from the first listed and never added
        twice: item 2170 sits on RC and is repeated on the old RC-R, and
        goodwill 3163 moved from RC to RC-M in 2018Q2.
        """
        if not full_codes:
            return pl.DataFrame(
                schema={"rssd": pl.Int64, "period": pl.Date, out: pl.Float64}
            )
        hit = long.filter(
            pl.col("schedule").is_in(list(schedules)) & pl.col("code").is_in(full_codes)
        )
        if len(schedules) > 1:
            rank = {s: i for i, s in enumerate(schedules)}
            hit = (
                hit.with_columns(
                    pl.col("schedule").replace_strict(rank, default=len(rank)).alias("_rank")
                )
                .sort("_rank")
                .unique(subset=[*key, "code"], keep="first")
                .drop("_rank")
            )
        return hit.group_by(key).agg(pl.col("value").sum().alias(out))

    def by_prefix(
        spec: mdrm.ItemSpec, items: tuple[str, ...], column: str
    ) -> pl.DataFrame | None:
        """Sum ``items`` per bank-quarter, resolving the prefix **per item**.

        Per item, not per group.  On form 031 schedule RC-C has two columns --
        consolidated and domestic-office -- and an item can appear in only one
        of them: a bank at 2020Q2 reports item 2081 consolidated but items 1545
        and J451 domestic-only.  Choosing one prefix for the whole group takes
        whichever the first item happened to use and silently drops the rest,
        which read that bank's other loans as $0 against a real $16.7bn.

        Returns the sum alongside ``{column}__n``, how many of the items the
        bank actually reported, which is what picks between variants.
        """
        pieces: list[tuple[str, pl.DataFrame]] = []
        for item in items:
            per_item: pl.DataFrame | None = None
            name = f"__item_{item}"
            for prefix in spec.prefixes:
                hit = summed(spec.schedules, [prefix + item], name)
                if hit.is_empty():
                    continue
                per_item = hit if per_item is None else _fill_missing(per_item, hit, name)
            if per_item is not None:
                pieces.append((name, per_item))
        if not pieces:
            return None

        out = base.select(key)
        for name, frame in pieces:
            out = out.join(frame, on=key, how="left")
        names = [name for name, _ in pieces]
        present = pl.sum_horizontal(
            [pl.col(c).is_not_null().cast(pl.Int32) for c in names]
        )
        total = pl.sum_horizontal([pl.col(c).fill_null(0.0) for c in names])
        # Null, not zero, when the bank reports none of the items: "not
        # reported" and "no exposure" must not collapse into each other.
        return out.with_columns(
            pl.when(present > 0).then(total).otherwise(None).alias(column),
            present.alias(f"{column}__n"),
        ).select([*key, column, f"{column}__n"])

    for spec in specs:
        variants = spec.variants()
        parts: list[tuple[str, str]] = []
        holder = base.select(key)
        for i, items in enumerate(variants):
            column = f"__v{i}_{spec.name}"
            piece = by_prefix(spec, items, column)
            if piece is None:
                continue
            holder = holder.join(piece, on=key, how="left")
            parts.append((column, f"{column}__n"))

        if not parts:
            base = base.with_columns(pl.lit(None, dtype=pl.Float64).alias(spec.name))
            continue

        # Pick the variant the filer reported most completely, ties going to
        # the earlier -- coarser -- one.  "Any component present" is not enough
        # to identify a variant: BNY Mellon reports J454, one of the two items
        # in the "both halves of item 9" variant, and also all four items of
        # the finer one, so matching on presence alone took the pair, kept
        # J454's $7bn and dropped the $16bn beside it.  Ties favour the coarse
        # variant because a rollup is the total of its own detail, which is why
        # Zions' 1288 beats the five zeros it sits next to.
        #
        # A spec may instead ask for the *first* variant with anything present
        # (``prefer="first"``): its variants are successive definitions of the
        # same total rather than levels of detail, and the older definition's
        # lines can outlive the newer total on the form.
        value = pl.lit(None, dtype=pl.Float64)
        if spec.prefer == "first":
            for column, count in reversed(parts):
                value = pl.when(pl.col(count) > 0).then(pl.col(column)).otherwise(value)
        else:
            best = pl.max_horizontal([pl.col(n) for _, n in parts])
            for column, count in reversed(parts):
                value = (
                    pl.when((pl.col(count) == best) & (pl.col(count) > 0))
                    .then(pl.col(column))
                    .otherwise(value)
                )
        resolved = holder.with_columns(value.alias(spec.name)).select([*key, spec.name])

        # ``plus`` adds a second basis rather than choosing between bases:
        # deposits are domestic (RCON2200) *plus* foreign (RCFN2200), and a
        # bank with no foreign offices simply contributes nothing here.
        for prefix, items in spec.plus:
            extra = summed(spec.schedules, [prefix + item for item in items], "_extra")
            if extra.is_empty():
                continue
            resolved = (
                resolved.join(extra, on=key, how="left")
                .with_columns(
                    (
                        pl.col(spec.name).fill_null(0.0)
                        + pl.col("_extra").fill_null(0.0)
                    ).alias(spec.name)
                )
                .drop("_extra")
            )

        base = base.join(resolved, on=key, how="left")

    scale = [
        pl.col(spec.name) * mdrm.UNIT_SCALE
        for spec in specs
        if spec.name in base.columns
    ]
    if scale:
        base = base.with_columns(scale)
    return base


def _fill_missing(left: pl.DataFrame, right: pl.DataFrame, column: str) -> pl.DataFrame:
    """Take ``right``'s value only where ``left`` has none, for one column.

    This is what makes prefix order a *preference* rather than an override: a
    bank reporting on ``RCFD`` keeps its consolidated figure, and a bank with
    only ``RCON`` gets the domestic one, in the same pass.
    """
    key = ["rssd", "period"]
    merged = left.join(right, on=key, how="full", coalesce=True, suffix="_alt")
    alt = f"{column}_alt"
    if alt not in merged.columns:
        return merged
    return merged.with_columns(
        pl.col(column).fill_null(pl.col(alt)).alias(column)
    ).drop(alt)
