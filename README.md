# bankqtr_db — DFAST bank peer benchmarking

Bank-quarter panel databases for peer benchmarking, built **twice from two
independent sources**. One row per bank-quarter, columns per variable, with
provenance and coverage reporting attached.

| Package | Source | Entity | Output | Window |
|---|---|---|---|---|
| `bankqtr_db` | SEC EDGAR — XBRL, filing HTML, IR supplements | the holding company, as it reports itself | `panel.parquet` | <!--stats:edgar-window-->2013Q1 – 2026Q2<!--/stats--> |
| `callrpt_db` | FFIEC — CDR bulk Call Reports, NIC structure data | its bank charters, summed | `call_panel.parquet` | <!--stats:ffiec-window-->2001Q1 – 2026Q2<!--/stats--> |

They are not redundant. The EDGAR panel is what the market sees and covers the
whole consolidated firm; the FFIEC panel is a regulatory filing on a fixed form,
so its categories cross-foot, it reaches back to 2001, and it covers the seven US
intermediate holding companies of foreign banks that file no 10-K at all. Where
the two disagree, `source_diff.csv` says by how much and whether the gap is
stable.

📖 **[Full documentation is in the wiki.](https://github.com/keithcsqueensu/bankqtr_db/wiki)**
📊 **[Panel downloads and charts.](https://keithcsqueensu.github.io/bankqtr_db/)**

## Quick start

```bash
uv sync
export BANKQTR_UA="you@yourdomain.com"          # SEC requires a real contact

uv run python scripts/fetch_facts.py            # companyfacts + submissions
uv run python scripts/fetch_instances.py --since 2013-01-01
uv run python scripts/fetch_ir.py --since 2020-01-01       # IR supplements
uv run python scripts/build_panel.py --since 2013-01-01 --html-fallback --html-since 2013-01-01 --ir
```

The FFIEC build is independent and can be run on its own. It starts at **2001Q1**
and follows each firm back through its predecessors:

```bash
uv run python scripts/fetch_call.py                      # 102 quarters, ~850 MB
uv run python scripts/resolve_rssd.py --write-config     # bank -> RSSD, once
uv run python scripts/build_call_panel.py                # 2001Q1 onward, with lineage
uv run python scripts/fetch_call.py --tfr                # the thrifts CDR has nothing for
uv run python scripts/build_call_panel.py --tfr          # ...folded in
```

Fetching and building are separate on purpose: every `build_*` script reads only
what is already cached and never touches the network, so a rebuild is
reproducible. A full 2013-start EDGAR build takes a few minutes.

See **[Quick Start](https://github.com/keithcsqueensu/bankqtr_db/wiki/Quick-Start)**
for the narrower windows and what each flag costs.

## Outputs

Both builds land in `data/out/`:

| File | Contents |
|---|---|
| `panel.parquet` / `.csv` | the EDGAR bank-quarter panel — <!--stats:edgar-extent-->1,648 rows, 31 banks, 548 columns<!--/stats--> |
| `call_panel.parquet` / `.csv` | the FFIEC panel — <!--stats:ffiec-extent-->3,763 rows, 38 firms, 279 columns<!--/stats--> |
| `call_panel_charters.parquet` | one row per bank charter per quarter, with its predecessor |
| `panel_coverage.csv` / `call_panel_coverage.csv` | per bank and variable: how many quarters are populated |
| `panel_flags.csv` / `call_panel_flags.csv` | bank-quarters failing a sanity check |
| `source_diff.csv` | EDGAR against FFIEC, per bank-quarter per variable |
| `rssd_lineage.csv` | every predecessor of every firm: who was absorbed into whom, when, how |

Full list and column reference: **[Outputs](https://github.com/keithcsqueensu/bankqtr_db/wiki/Outputs)**
and **[Data Dictionary](https://github.com/keithcsqueensu/bankqtr_db/wiki/Data-Dictionary)**.

## Read this before using the data

Four things will produce a wrong answer if you skip them. None is a bug; each is
a property of the underlying disclosure that the panel reports rather than hides.

1. **The CECL seam at 2020Q1.** Every row carries a `basis` column — `incurred`
   or `cecl`. The allowance changes meaning at that line, and the step shows up
   in `reserve_coverage` as a jump that looks exactly like a credit event.
2. **Two discontinuities that read as growth.** `TFC`'s CIK is BB&T's, and the
   SunTrust merger roughly doubles the balance sheet in 2019Q4. `FCNCA` excludes
   CIT before 2022 and SVB before 2023.
3. **The 2013 start is survivors-only.** Any cross-sectional statistic over the
   early years is conditioned on surviving to 2026.
4. **Check `mix_coverage_pct` before ranking peers.** On the EDGAR panel it is an
   estimate of coherence, not a guarantee — and ≈100 is necessary, not sufficient.

The full account, with the other two: **[Reading the Panel](https://github.com/keithcsqueensu/bankqtr_db/wiki/Reading-the-Panel)**.

## The one thing to know first

**The SEC's companyfacts API cannot produce a portfolio mix.** It returns only
*undimensioned* facts. Banks tag CRE loans, C&I loans and total loans with the
*same* XBRL element and distinguish them purely by dimension member — and
companyfacts drops every dimensioned fact. Verified on JPMorgan's FY2024 10-K:
companyfacts exposes 1 fact for the loans element at 2024-12-31; the filing's own
instance document carries **146**.

So extraction runs in two tiers — `xbrl.companyfacts_long` for consolidated
totals, and `xbrl.dimensional_long`, which parses each filing's XBRL **instance
document** and keeps the dimensions. Portfolio mix, criticized balances and
delinquency buckets all come from the second.

## Testing

```bash
uv run pytest tests/ -q
uv run ruff check .
```

<!--stats:tests-->308<!--/stats--> tests across both builds. Every one corresponds to a defect that produced
**plausible but wrong** numbers during development — the dangerous kind, since
nothing raises and the panel still renders. The catalogue of what they guard is
in the wiki: **[Traps](https://github.com/keithcsqueensu/bankqtr_db/wiki/Traps)**.

## Documentation

| Page | What it covers |
|---|---|
| [Quick Start](https://github.com/keithcsqueensu/bankqtr_db/wiki/Quick-Start) | Clean checkout to built panel, both sources |
| [Choosing a Source](https://github.com/keithcsqueensu/bankqtr_db/wiki/Choosing-a-Source) | Which panel answers your question, and what to do when they disagree |
| [Reading the Panel](https://github.com/keithcsqueensu/bankqtr_db/wiki/Reading-the-Panel) | The six caveats that matter before analysis |
| [Outputs](https://github.com/keithcsqueensu/bankqtr_db/wiki/Outputs) | What each file in `data/out/` contains |
| [Data Dictionary](https://github.com/keithcsqueensu/bankqtr_db/wiki/Data-Dictionary) | Column naming scheme, provenance, and the families |
| [Universe](https://github.com/keithcsqueensu/bankqtr_db/wiki/Universe) | Who is in, who is deliberately out |
| [Glossary](https://github.com/keithcsqueensu/bankqtr_db/wiki/Glossary) | RSSD, MDRM, RCFD/RCON, CECL, EX-13 |
| [FAQ](https://github.com/keithcsqueensu/bankqtr_db/wiki/FAQ) | Short answers with links |
| [EDGAR Panel](https://github.com/keithcsqueensu/bankqtr_db/wiki/EDGAR-Panel) | Two-tier extraction, and why the panel logic is not a groupby |
| [IR Supplements](https://github.com/keithcsqueensu/bankqtr_db/wiki/IR-Supplements) | Office CRE and the disclosures XBRL has no tag for |
| [HTML Fallback](https://github.com/keithcsqueensu/bankqtr_db/wiki/HTML-Fallback) | Parsing the filing's own tables |
| [Extending to 2013](https://github.com/keithcsqueensu/bankqtr_db/wiki/Extending-to-2013) | Build record for the XBRL back-extension |
| [Call Reports](https://github.com/keithcsqueensu/bankqtr_db/wiki/Call-Reports) | What the second source buys, and what it does not |
| [Going Back to 2001](https://github.com/keithcsqueensu/bankqtr_db/wiki/Going-Back-to-2001) | Lineage, predecessors, and the thrift gap |
| [Traps](https://github.com/keithcsqueensu/bankqtr_db/wiki/Traps) | Every catalogued failure, indexed by failure mode |
| [Testing](https://github.com/keithcsqueensu/bankqtr_db/wiki/Testing) | The two suites and what they pin |
| [Extending](https://github.com/keithcsqueensu/bankqtr_db/wiki/Extending) | Adding a variable, a bank, a disclosure or an era |

## A note on the counts in this file

Figures that describe a build rather than the text — row counts, columns, the
window, the test count — are wrapped in sentinels and rewritten from the
committed panels by `scripts/sync_doc_stats.py`, which CI runs on every push to
`main`. Do not edit them by hand; run `--write`. See
[Extending](https://github.com/keithcsqueensu/bankqtr_db/wiki/Extending#counts-quoted-in-prose).

## Layout

```
bankqtr_db/     the EDGAR build   (filings, edgar, instance, xbrl, taxonomy,
                                   variables, panel, html_fallback, ir,
                                   ir_extract, reconcile)
callrpt_db/     the FFIEC build   (cdr, nic, lineage, mdrm, schedules, panel,
                                   crosscheck, tfr)
scripts/        fetch_* and build_* entry points
tests/          308 tests, one per defect
docs/           the GitHub Pages site: downloads and charts
```
