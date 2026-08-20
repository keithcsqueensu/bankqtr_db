"""Resolve each panel bank to its FFIEC RSSD, and write the universe.

uv run python scripts/resolve_rssd.py
uv run python scripts/resolve_rssd.py --write-config

Bridging on EIN, not on name
----------------------------
NIC carries no CIK, so there is no direct key between the two registries.  It
does carry ``ID_TAX`` -- the EIN -- and EDGAR's submissions payload carries
``ein``, and both registries collect that number independently from the filer.
Matching on it is therefore a *check* rather than a guess: JPMorgan Chase & Co
is CIK 0000019617 with EIN 132624428 at EDGAR and RSSD 1039502 with ID_TAX
132624428 at NIC, and nothing about the spelling of the name had to be trusted.

Names are used only for the seven IHCs that file nothing with the SEC and so
have no EIN to bridge on.  Each is matched against NIC's legal name with the
``IHC_IND`` flag as corroboration, and every match is written to
``data/out/rssd_resolution.csv`` with the evidence that produced it.

The output records, per firm: the RSSD, how it was matched, the NIC legal name,
the entity type, and the depository subsidiaries whose Call Reports roll up to
it in the most recent quarter on disk.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from bankqtr_db import config as edgar_config
from callrpt_db import cdr, config, nic, schedules

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("resolve_rssd")

# NIC entity types that can sit at the top of a banking organisation.  A firm
# whose EIN matches several entities is resolved to the one of these; Zions is
# the reminder that the list has to include plain bank charters, since it
# merged its holding company into the bank in 2018 and the top entity of the
# organisation is now ``ZIONS BANCORPORATION, NATIONAL ASSOCIATION`` itself.
HOLDING_ENTITY_TYPES = frozenset(
    {"FHD", "BHC", "IHC", "SLHC", "FBH", "DEO", "NAT", "SMB", "NMB", "SSB"}
)

# The IHCs with no SEC registration, from bankqtr_db.config.NON_SEC_IHCS, with
# the ticker each is given in this panel.  A synthetic ticker keeps the join
# key non-null so the two panels can be concatenated.
IHC_TICKERS: dict[str, str] = {
    "BMO Financial Corp": "BMOUS",
    "TD Group US Holdings LLC": "TDUS",
    "UBS Americas Holding LLC": "UBSUS",
    "DB USA Corporation": "DBUS",
    "Barclays US LLC": "BCSUS",
    "RBC US Group Holdings LLC": "RBCUS",
    "BNP Paribas USA, Inc.": "BNPUS",
    "Credit Suisse Holdings (USA), Inc.": "CSUS",
}


@dataclass
class Resolution:
    ticker: str
    name: str
    cik: str | None
    ein: str | None
    rssd: int | None
    matched_on: str
    nic_name: str
    entity_type: str
    category: str
    tier: str
    also_rssd: tuple[int, ...] = ()
    last_filing: dt.date | None = None
    reference_period: str = ""
    filers: tuple[int, ...] = ()
    filer_names: tuple[str, ...] = ()
    note: str = ""

    @property
    def rssds(self) -> tuple[int, ...]:
        return () if self.rssd is None else (self.rssd, *self.also_rssd)


def _ein_for(cik: str) -> str | None:
    """EDGAR's EIN for a CIK, from the submissions cache.

    Read from the cache rather than fetched: ``fetch_facts.py`` already put it
    there, and a resolver that silently hits the network makes the universe
    depend on when it was last run.
    """
    path = edgar_config.RAW_FACTS / f"SUB{cik}.json.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    return nic.normalise_ein(payload.get("ein"))


def _pick(candidates: list[nic.Entity]) -> tuple[nic.Entity | None, str]:
    """Choose the top-tier entity among several sharing an EIN.

    Activity is ranked **above** the holding-company flag.  Ranking the flag
    first picks a dissolved holdco over the live entity that replaced it, which
    is exactly what happened to Zions: RSSD 1027004 is flagged BHC and closed,
    and its successor 276579 is a plain national bank charter that is very much
    open and filing.
    """
    if not candidates:
        return None, "no NIC entity carries this EIN"
    if len(candidates) == 1:
        return candidates[0], "ein"
    pool = sorted(
        candidates,
        key=lambda e: (
            not e.active,
            not (e.is_bhc or e.is_ihc),
            e.entity_type not in HOLDING_ENTITY_TYPES,
            e.rssd,
        ),
    )
    return pool[0], "ein+rank"


def _normalise(name: str) -> str:
    """Upper-case, strip punctuation and drop the corporate-form suffix.

    Punctuation becomes a space rather than nothing.  Deleting it glued
    "ZIONS BANCORPORATION, NATIONAL ASSOCIATION" into one word that matched
    nothing, which is how the resolver kept choosing Zions' dissolved holding
    company over the bank that replaced it.
    """
    out = name.upper()
    for junk in (",", ".", "/"):
        out = out.replace(junk, " ")
    out = " ".join(out.split())
    for suffix in (
        " NATIONAL ASSOCIATION",
        " CORPORATION",
        " INCORPORATED",
        " HOLDINGS",
        " HOLDING",
        " COMPANY",
        " GROUP",
        " LLC",
        " INC",
        " CORP",
        " CO",
        " NA",
        " &",
    ):
        out = out.removesuffix(suffix)
    return " ".join(out.split())


def _name_matches(
    target: str, entities: dict[int, nic.Entity], *, require_ihc: bool = False
) -> list[tuple[nic.Entity, str]]:
    """Every entity whose legal name matches, with how it matched.

    All of them, not the best one.  Several distinct NIC entities normalise to
    the same name -- ``COMERICA INCORPORATED`` and ``COMERICA HOLDINGS
    INCORPORATED`` both reduce to ``COMERICA`` -- and returning only the first
    hands the ranking a pool of one, so the tie-break that knows which of them
    owns a bank never gets to run.

    Used for the IHCs, which have no SEC registration and so no EIN to bridge
    on, and as a fallback for the SEC filers whose EDGAR ``ein`` is absent from
    NIC entirely (Morgan Stanley, Citizens, State Street, Santander USA,
    Comerica).  A name match is weaker evidence than a tax id, and the
    resolution report says which one produced each row.
    """
    want = _normalise(target)
    out: list[tuple[nic.Entity, str]] = []
    for ent in entities.values():
        name = _normalise(ent.name)
        if name == want:
            how = "name"
        elif name.startswith(want) and (ent.is_bhc or ent.is_ihc):
            how = "name_prefix"
        else:
            continue
        if require_ihc and not ent.is_ihc:
            continue
        if ent.is_ihc:
            how += "+ihc_flag"
        elif ent.is_bhc:
            how += "+bhc_flag"
        out.append((ent, how))
    if require_ihc and not out:
        return _name_matches(target, entities, require_ihc=False)
    return out


@dataclass
class Candidate:
    entity: nic.Entity
    how: str
    filers: tuple[int, ...] = ()


def _candidates(
    bank_name: str,
    ein: str | None,
    entities: dict[int, nic.Entity],
    ein_index: dict[str, list[nic.Entity]],
    *,
    require_ihc: bool = False,
) -> list[Candidate]:
    """Every entity that could be this firm, with how it was found."""
    found: dict[int, Candidate] = {}
    if ein:
        hit, how = _pick(list(ein_index.get(ein, [])))
        if hit is not None:
            found[hit.rssd] = Candidate(hit, how)
        for other in ein_index.get(ein, []):
            found.setdefault(other.rssd, Candidate(other, "ein"))
    for ent, how in _name_matches(bank_name, entities, require_ihc=require_ihc):
        if ent.rssd not in found:
            found[ent.rssd] = Candidate(ent, how)
    return list(found.values())


def _rank(candidate: Candidate) -> tuple:
    """Sort key: filers first, then activity, then holding-company standing.

    A tax-id match outranks everything, because a name is not unique across
    NIC: three distinct active bank holding companies are called some form of
    "Citizens Financial", and one of them is a $220bn DFAST participant while
    another owns a single bank in West Virginia.

    Below that, having a Call Report filer underneath is the strongest evidence
    available, because it is the only criterion that tests what the panel
    actually needs.  Every mis-resolution found while building
    this -- Comerica landing on ``COMERICA HOLDINGS INCORPORATED``, Valley
    National on ``VALLEY NATIONAL CORPORATION``, Zions on its dissolved
    holdco -- was a candidate with a plausible name, a plausible flag, and no
    bank under it.
    """
    e = candidate.entity
    return (
        not candidate.how.startswith("ein"),
        not candidate.filers,
        not e.active,
        not (e.is_bhc or e.is_ihc),
        e.entity_type not in HOLDING_ENTITY_TYPES,
        e.rssd,
    )


def resolve(reference: str | None = None) -> list[Resolution]:
    entities = nic.entities()
    ein_index = nic.by_ein()

    cached = cdr.cached_periods()
    if reference and reference in cached:
        cached = [p for p in cached if p <= reference]

    # Candidates are scored at three points in the window, not one.  A single
    # probe cannot see an organisation that was restructured inside it: Zions
    # ran seven bank charters under RSSD 1027004 until 2018 and one charter
    # with no holding company after, so scoring only at the end of the window
    # keeps the survivor and loses Amegy, California Bank & Trust, Nevada State
    # Bank and the rest for every quarter before the merger.
    probes: list[str] = []
    if cached:
        probes = list(
            dict.fromkeys([cached[0], cached[len(cached) // 2], cached[-1]])
        )
    context_cache: dict[str, tuple[set[int], dict[int, list[int]]]] = {}

    def context(period: str) -> tuple[set[int], dict[int, list[int]]]:
        if period not in context_cache:
            frame = schedules.roster(cdr.zip_path(period))
            context_cache[period] = (
                set(frame["rssd"].to_list()),
                nic.hierarchy(cdr.quarter_end(period)),
            )
        return context_cache[period]

    def probes_for(last_filing: dt.date | None) -> list[str]:
        """Quarters to score against, ending at the firm's own last quarter."""
        if not cached:
            return []
        if last_filing is None:
            return probes
        usable = [p for p in cached if cdr.quarter_end(p) <= last_filing] or [cached[0]]
        return list(dict.fromkeys([usable[0], usable[len(usable) // 2], usable[-1]]))

    def score(cands: list[Candidate], quarters: list[str]) -> dict[int, set[int]]:
        """Filers per candidate per quarter; also fills ``Candidate.filers``."""
        per_quarter: dict[int, set[int]] = {}
        for period in quarters:
            present, graph = context(period)
            as_of = cdr.quarter_end(period)
            for c in cands:
                found = set(
                    nic.call_filers(c.entity.rssd, present, as_of=as_of, kids=graph)
                )
                per_quarter.setdefault(c.entity.rssd, set()).update(found)
        for c in cands:
            c.filers = tuple(sorted(per_quarter.get(c.entity.rssd, ())))
        return per_quarter

    def related(a: int, b: int, quarters: list[str]) -> bool:
        """Whether two entities were ever in the same control tree."""
        for period in quarters:
            _present, graph = context(period)
            if b in nic.descendants(a, graph) or a in nic.descendants(b, graph):
                return True
        return False

    def extras(
        best: Candidate, cands: list[Candidate], quarters: list[str]
    ) -> tuple[int, ...]:
        """Other top entities of the *same* organisation, in other eras.

        Three conditions, all necessary.  It must add filers the chosen entity
        does not already reach -- otherwise Bank of America NA, which shares its
        parent's tax id, is recorded as a second top entity for Bank of America
        Corp.  It must be the same organisation, evidenced either by the same
        tax id or by having shared a control tree with the chosen entity at some
        point in the window -- otherwise Citizens Financial Group picks up the
        unrelated Citizens Financial Group of Kentucky, Citizens Financial
        Services, Citizens Financial Corp of West Virginia and a Wisconsin bank
        holding company, purely on the strength of the name.  Zions is what the
        graph clause is for: RSSD 276579 carries a different tax id from the
        holding company it replaced, and the only thing tying them together is
        that it was that holding company's subsidiary until 2018.
        """
        out: list[int] = []
        for c in cands:
            if c.entity.rssd == best.entity.rssd or not c.filers:
                continue
            same_org = (
                c.how.startswith("ein")
                and best.how.startswith("ein")
                or related(best.entity.rssd, c.entity.rssd, quarters)
            )
            if not same_org:
                continue
            adds = False
            for period in quarters:
                present, graph = context(period)
                as_of = cdr.quarter_end(period)
                mine = set(
                    nic.call_filers(best.entity.rssd, present, as_of=as_of, kids=graph)
                )
                theirs = set(
                    nic.call_filers(c.entity.rssd, present, as_of=as_of, kids=graph)
                )
                if theirs - mine:
                    adds = True
                    break
            if adds:
                out.append(c.entity.rssd)
        return tuple(out)

    out: list[Resolution] = []
    for bank in edgar_config.BANKS + edgar_config.COMPARATORS:
        ein = _ein_for(bank.cik)
        cands = _candidates(bank.name, ein, entities, ein_index)
        quarters = probes_for(bank.last_filing)
        score(cands, quarters)
        cands.sort(key=_rank)
        best = cands[0] if cands else None
        out.append(
            Resolution(
                ticker=bank.ticker,
                name=bank.name,
                cik=bank.cik,
                ein=ein,
                rssd=best.entity.rssd if best else None,
                matched_on=best.how if best else "unresolved",
                nic_name=best.entity.name if best else "",
                entity_type=best.entity.entity_type if best else "",
                category=bank.category,
                tier=bank.tier,
                also_rssd=extras(best, cands, quarters) if best else (),
                last_filing=bank.last_filing,
                reference_period=";".join(quarters),
            )
        )

    for name in edgar_config.NON_SEC_IHCS:
        cands = _candidates(name, None, entities, ein_index, require_ihc=True)
        score(cands, probes)
        cands.sort(key=_rank)
        best = cands[0] if cands else None
        out.append(
            Resolution(
                ticker=IHC_TICKERS.get(name, name[:8].upper()),
                name=name,
                cik=None,
                ein=best.entity.ein if best else None,
                rssd=best.entity.rssd if best else None,
                matched_on=best.how if best else "unresolved",
                nic_name=best.entity.name if best else "",
                entity_type=best.entity.entity_type if best else "",
                category="regional",
                tier="ihc",
                also_rssd=extras(best, cands, probes) if best else (),
                last_filing=None,
                reference_period=";".join(probes),
                note="no SEC registration; matched on name",
            )
        )
    return out


def attach_filers(resolutions: list[Resolution], period: str) -> None:
    """Fill in the Call Report filers under each holding company."""
    path = cdr.zip_path(period)
    if not path.exists():
        log.warning("no cached period to resolve subsidiaries against")
        return
    frame = schedules.roster(path)
    names = dict(zip(frame["rssd"].to_list(), frame["name"].to_list()))
    present = set(names)
    as_of = cdr.quarter_end(period)
    graph = nic.hierarchy(as_of)
    for res in resolutions:
        found: set[int] = set()
        for rssd in res.rssds:
            found.update(nic.call_filers(rssd, present, as_of=as_of, kids=graph))
        res.filers = tuple(sorted(found))
        res.filer_names = tuple(names.get(f, "") for f in res.filers)


def write_report(resolutions: list[Resolution], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "ticker",
                "name",
                "cik",
                "ein",
                "rssd",
                "also_rssd",
                "reference_period",
                "matched_on",
                "nic_name",
                "entity_type",
                "tier",
                "category",
                "n_call_filers",
                "call_filers",
                "call_filer_names",
                "note",
            ]
        )
        for r in resolutions:
            writer.writerow(
                [
                    r.ticker,
                    r.name,
                    r.cik or "",
                    r.ein or "",
                    r.rssd or "",
                    ";".join(str(x) for x in r.also_rssd),
                    r.reference_period,
                    r.matched_on,
                    r.nic_name,
                    r.entity_type,
                    r.tier,
                    r.category,
                    len(r.filers),
                    ";".join(str(f) for f in r.filers),
                    ";".join(r.filer_names),
                    r.note,
                ]
            )


def render_holdings(resolutions: list[Resolution]) -> str:
    lines = ["HOLDINGS: tuple[Holding, ...] = ("]
    order = {"dfast": 0, "ihc": 1, "inactive": 2, "comparator": 3}
    for r in sorted(resolutions, key=lambda x: (order.get(x.tier, 9), x.ticker)):
        if r.rssd is None:
            continue
        parts = [f'"{r.name}"', str(r.rssd), f'"{r.ticker}"']
        parts.append(f'cik="{r.cik}"' if r.cik else "cik=None")
        parts.append(f'category="{r.category}"')
        parts.append(f'tier="{r.tier}"')
        if r.last_filing is not None:
            d = r.last_filing
            parts.append(f"last_filing=dt.date({d.year}, {d.month}, {d.day})")
        if r.also_rssd:
            inner = ", ".join(str(x) for x in r.also_rssd)
            parts.append(f"also_rssd=({inner},)" if len(r.also_rssd) == 1 else f"also_rssd=({inner})")
        lines.append("    Holding(" + ", ".join(parts) + "),")
    lines.append(")")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--period", default=None, help="quarter to resolve filers against")
    ap.add_argument(
        "--write-config",
        action="store_true",
        help="rewrite the HOLDINGS tuple in callrpt_db/config.py",
    )
    args = ap.parse_args(argv)

    config.ensure_dirs()
    cached = cdr.cached_periods()
    period = args.period or (cached[-1] if cached else None)
    resolutions = resolve(period)
    if period:
        attach_filers(resolutions, period)

    out = config.OUT / "rssd_resolution.csv"
    write_report(resolutions, out)

    unresolved = [r for r in resolutions if r.rssd is None]

    # "No filers in the latest quarter" is the normal state for an acquired
    # firm, not a resolution failure, so the check is whether the firm has a
    # filer in *any* cached quarter.  Discover, MUFG Americas, Comerica,
    # Synovus, BNP Paribas USA and Credit Suisse USA all report nothing in
    # 2026Q2 because they no longer exist, and every one of them is present
    # earlier in the window.
    never: list[Resolution] = []
    if cached:
        probe = [cached[0], cached[len(cached) // 2], cached[-1]]
        for res in resolutions:
            if res.rssd is None:
                continue
            seen = False
            for quarter in dict.fromkeys(probe):
                path = cdr.zip_path(quarter)
                if not path.exists():
                    continue
                frame = schedules.roster(path)
                present = set(frame["rssd"].to_list())
                as_of = cdr.quarter_end(quarter)
                graph = nic.hierarchy(as_of)
                if any(
                    nic.call_filers(r, present, as_of=as_of, kids=graph)
                    for r in res.rssds
                ):
                    seen = True
                    break
            if not seen:
                never.append(res)

    log.info(
        "resolved %d of %d firms; %d have no Call Report filer in any sampled quarter",
        len(resolutions) - len(unresolved),
        len(resolutions),
        len(never),
    )
    for r in unresolved:
        log.warning("unresolved: %s (%s) -- %s", r.name, r.ticker, r.matched_on)
    for r in never:
        log.warning(
            "no filers anywhere: %s (%s) rssd=%s %s",
            r.name,
            r.ticker,
            r.rssd,
            r.nic_name,
        )
    log.info("wrote %s", out)

    if args.write_config:
        path = Path("callrpt_db/config.py")
        text = path.read_text(encoding="utf-8")
        start = text.index("HOLDINGS: tuple[Holding, ...] = (")
        end = text.index(")", text.index("\n", start)) + 1
        # An empty placeholder ends on the same line; a rendered one does not.
        if text[start:].startswith("HOLDINGS: tuple[Holding, ...] = ()"):
            end = start + len("HOLDINGS: tuple[Holding, ...] = ()")
        else:
            end = text.index("\n)\n", start) + len("\n)")
        path.write_text(
            text[:start] + render_holdings(resolutions) + text[end:], encoding="utf-8"
        )
        log.info("rewrote HOLDINGS in %s", path)
    else:
        print(render_holdings(resolutions))
    return 0


if __name__ == "__main__":
    sys.exit(main())
