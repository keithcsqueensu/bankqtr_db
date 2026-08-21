# bankqtr_db — DFAST bank peer benchmarking

Bank-quarter panel databases for peer benchmarking, built **twice from two
independent sources**. One row per bank-quarter, columns per variable, with
provenance and coverage reporting attached.

| Package | Source | Entity | Output |
|---|---|---|---|
| `bankqtr_db` | SEC EDGAR — XBRL, filing HTML, IR supplements | the holding company, as it reports itself | `panel.parquet` |
| `callrpt_db` | FFIEC — CDR bulk Call Reports, NIC structure data | its bank charters, summed | `call_panel.parquet` |

They are not redundant. The EDGAR panel is what the market sees and covers the
whole consolidated firm; the FFIEC panel is a regulatory filing on a fixed
form, so its categories cross-foot, it reaches back to 2001, and it covers the
seven US intermediate holding companies of foreign banks that file no 10-K at
all. Where the two disagree, `source_diff.csv` says by how much and whether the
gap is stable — see [FFIEC Call Reports](#second-source-ffiec-call-reports).

## Quick start

```bash
uv sync
export BANKQTR_UA="you@yourdomain.com"          # SEC requires a real contact

uv run python scripts/fetch_facts.py            # companyfacts + submissions
uv run python scripts/fetch_instances.py --since 2020-01-01
uv run python scripts/fetch_ir.py --since 2020-01-01       # IR supplements
uv run python scripts/build_panel.py --since 2020-01-01 --html-fallback --ir
```

The FFIEC build is independent and can be run on its own; it needs no EDGAR
data except the cached submissions the RSSD resolver reads EINs from. It
starts at **2001Q1** and follows each firm back through its predecessors:

```bash
uv run python scripts/fetch_call.py                      # 102 quarters, ~850 MB
uv run python scripts/build_call_panel.py                # 2001Q1 onward, with lineage
```

See [Going back to 2001](#going-back-to-2001) and
[docs/extending-to-2001.md](docs/extending-to-2001.md) before using the rows
before 2013.

The XBRL half of the panel reaches back to **2013**. The HTML and IR fallbacks
do not — both were written against post-2020 layouts — so hold them at 2020
even when the XBRL window is wider:

```bash
uv run python scripts/fetch_instances.py --since 2013-01-01     # ~1,670 docs, 670 MB
uv run python scripts/build_panel.py --since 2013-01-01     --html-fallback --html-since 2020-01-01 --ir
```

See [docs/extending-to-2013.md](docs/extending-to-2013.md) for what the pre-CECL
dialect required and what it changed, and *Reading across 2020Q1* below before
using the longer window.

A full 2013-start build takes a few minutes: instance, IR and filing-HTML
parsing all run across cores, while every download stays serial behind the one
rate limiter in `edgar.py`. Output is byte-identical either way — set
`BANKQTR_WORKERS=1` to force the serial path if you ever need to check that.

Outputs land in `data/out/`:

| File | Contents |
|---|---|
| `panel.parquet` / `.csv` | the bank-quarter panel |
| `panel_facts.parquet` | every extracted fact, long format, with dimensions |
| `panel_coverage.csv` | per bank and variable: how many quarters are populated |
| `panel_gaps.csv` | variables ranked by how many banks lack them |
| `panel_flags.csv` | bank-quarters failing a sanity check |
| `panel_build_info.json` | the window, settings, commit and cell-provenance counts of the build |

## The one thing to know first

**The companyfacts API cannot produce a portfolio mix.** It returns only
*undimensioned* facts. Banks tag CRE loans, C&I loans and total loans with the
*same* XBRL element and distinguish them purely by dimension member — and
companyfacts drops every dimensioned fact. Verified on JPMorgan's FY2024 10-K:
companyfacts exposes 1 fact for the loans element at 2024-12-31; the filing's
own instance document carries **146**, including the segment and class
breakdowns.

So extraction runs in two tiers:

- `xbrl.companyfacts_long` — consolidated totals, one cheap call per bank.
- `xbrl.dimensional_long` — parses each filing's XBRL **instance document**
  (`*_htm.xml`), keeping the dimensions. This is where portfolio mix,
  criticized/classified balances and delinquency buckets come from.

Both return the same long schema; `source` records which path a fact came from.

## Pipeline

The EDGAR side; the FFIEC side has [its own](#pipeline-1).

```
filings.py     10-K/10-Q index per bank (submissions API, incl. overflow shards)
edgar.py       rate-limited, gzip-cached HTTP
instance.py    XBRL instance -> dimensional long frame
xbrl.py        companyfacts + instance -> unified long frame, dedup, restatement flags
taxonomy.py    heterogeneous dimension members -> canonical categories
variables.py   declarative variable and ratio definitions
panel.py       long frame -> bank-quarter panel (selection, quarterisation, ratios)
html_fallback.py  pandas.read_html table parsing for what XBRL never carries
ir.py          IR supplement discovery and download (Q4 feed / page / 8-K)
ir_extract.py  supplement + presentation -> long frame (tables, lines, phrases)
reconcile.py   prefer XBRL, then HTML, then IR; report coverage and quality
```

## Universe

31 banks: supervisory stress-test (DFAST) participants that file 10-K/10-Q,
plus the two US IHCs of foreign banks that file with the SEC. CIKs were
resolved against EDGAR, not from memory, and every entry was confirmed to have
periodic filings.

Ten additional non-DFAST regional comparators are available behind
`--comparators`.

**Not reachable from EDGAR.** Most US IHCs of foreign banking organisations —
BMO Financial Corp, TD Group US Holdings, UBS Americas, DB USA, Barclays US,
RBC US Group, BNP Paribas USA — file no 10-K at all; they report only on
FR Y-9C. They are listed in `config.NON_SEC_IHCS` so the absence is explicit
rather than an unexplained hole. **That gap is now closed by `callrpt_db`**,
which reads their bank subsidiaries' Call Reports — see
[FFIEC Call Reports](#second-source-ffiec-call-reports). Seven of the eight are
in that panel, with 305 bank-quarters between them; Credit Suisse Holdings
(USA) is the exception, and genuinely so, since its US intermediate holding
company owned a broker-dealer rather than a Call-Report-filing bank.

Firms that stopped filing mid-window (Discover, acquired 2025; MUFG Americas,
2021) carry a `last_filing` date so "no disclosure" stays distinguishable from
"no longer exists".

**On the 2013 start, the universe is survivors-only.** It is today's DFAST
list, and the 2013-2019 peer group also held SunTrust, BBVA USA, CIT, SVB,
First Republic, Signature, People's United, E\*TRADE, MB Financial and TCF.
Adding them is a separate decision — new `inactive` entries with a
`last_filing` — not part of the back-extension. Any cross-sectional statistic
computed over the early years is conditioned on surviving to 2026.

Two discontinuities *inside* the existing universe matter more than they look,
because the ticker does not change when the company does:

- **`TFC`'s CIK is BB&T's.** Everything before 2019Q4 is BB&T standalone; the
  SunTrust merger lands in 2019Q4 and roughly doubles the balance sheet.
- **`FCNCA`** excludes CIT before 2022 and SVB before 2023.

Both read as growth in a time series and are not.

## Reading across 2020Q1

The panel spans two accounting regimes, and every row says which one it is on
in the **`basis`** column: `incurred` before CECL adoption, `cecl` from it.

This is not cosmetic. Under CECL the allowance is a lifetime expected-loss
estimate; before 2020Q1 it is an incurred-loss reserve, and the level step
between them shows up in `reserve_coverage` and `reserve_to_nonaccrual` as a
jump that looks exactly like a credit event. `acl`, `provision` and the two
reserve ratios all change meaning at that line; the move from recorded
investment to amortised cost shifts loan and nonaccrual levels slightly too.

The panel splices the two anyway — one continuous series is what a user wants —
but says where the seam is so it can be seen, filtered on, or controlled for.
Adoption dates live in `config.CECL_ADOPTION`: 2020-01-01 for every filer here
except Raymond James, whose September fiscal year puts its first CECL quarter
at 2020Q4. That is the standard large-filer effective date rather than a
per-filer confirmation from each 10-K.

## Why the panel logic is not a groupby

Four hazards, each of which silently produces plausible wrong numbers. All four
are covered by tests in `tests/test_pipeline.py`.

**1. Overlapping dimension cuts.** A bank tags the same balance on several
different cuts simultaneously — portfolio segment, loan class, industry,
vintage. Each cut sums to the same total, so treating "the CRE rows" as one
population roughly doubles it. Every summed column fixes a single *dimension
signature* (the exact axis set) per bank-quarter before aggregating.

**2. Incomplete shallow cuts.** The coarsest signature is not always the
complete one. Wells Fargo tags a single member at portfolio-segment level
(consumer, $375bn) while the full book only appears at segment × class
($909bn). Totals therefore prefer the most complete partition; individual
categories prefer the shallowest.

**3. Alias members.** JPMorgan tags credit-card loans as both
`CreditCardReceivablesMember` and `CreditCardLoanPortfolioSegmentMember` on the
same axis. Summing distinct members doubles the card book.

**4. Cumulative-only flows.** JPMorgan tags net charge-offs *only* year-to-date;
no quarterly figure exists in the filing. Quarters are recovered by
differencing consecutive cumulative points grouped by the fiscal-year start
each fact declares — which also recovers Q4 from a 10-K, and handles the
off-calendar fiscal years (Raymond James ends in September).

Restatements are kept, not hidden: the latest filed version of a fact wins and
`restated` marks where versions disagreed by more than rounding.

### The limitation this design accepts

Each loan category independently picks the signature that discloses it best. That
maximises how many categories get populated, but it means two categories can come
from different disclosure tables and are not guaranteed mutually exclusive — a
custody bank can show CRE larger than its commercial total.

Rather than hide this, the panel measures it. **`mix_coverage_pct`** is the sum of
the disjoint leaf categories over reported total loans:

- ≈100 — the mix is coherent and safe to rank on
- ≫100 — categories overlap; the breakdown is double-counting
- ≪100 — the breakdown is partial

Filter on it before ranking peers. It is also a quality flag (`mix_incoherent`).

**≈100 is necessary, not sufficient.** The metric is gameable by a single
mis-mapped rollup, and was being gamed: `TotalLoansandLeasesMember` — the grand
total — matched the lease rule, so Citizens' entire $93bn book was counted as
its lease line and `mix_coverage_pct` read a flawless 101.3%. M&T's commercial
rollup did the same at $27bn against a real lease book near $2bn, for a
flawless 100.6%. Both are fixed, and both now read honestly low (0 and 72)
because the rest of those breakdowns genuinely is not mapped. When one member
absorbs the whole book, the leaves sum to the total by construction — so treat
a suspiciously perfect score on a bank with few populated categories as a
reason to look at `panel_unmapped_members.csv`, not as a clean bill of health.

It could also go **stale**. It is computed on the XBRL frame, and until it was
recomputed after the fallbacks, any HTML or IR value that filled a loan
category left the earlier score standing — 97 bank-quarters across twelve banks
were carrying one, 14 of them reading exactly 100.0.
In practice the incoherent rows concentrate in the custody, broker and specialty
names (BNY Mellon, Morgan Stanley, Schwab, Santander USA), whose loan disclosures
are small and irregularly structured; the regional and universal banks that
matter most for CRE benchmarking come out coherent.

Forcing every category to share one signature was implemented and measured, and
it is worse on both axes: median `mix_coverage_pct` falls from 89 to 69 and the
flag count rises from 363 to 476, because the widest-coverage table usually names
only rollup members (commercial, consumer) and so drops CRE and C&I outright —
Bank of America loses its CRE column entirely. Per-category selection with a
coherence metric is the better trade for a database whose point is breadth of
comparable exposure metrics.

## Variable coverage, honestly

Which variables are obtainable differs sharply by category:

| Group | Source | Coverage |
|---|---|---|
| Total loans, ACL, NCO, provision, nonaccrual | XBRL, mostly consolidated | strong |
| Portfolio mix (CRE, C&I, construction, card, auto, resi) | XBRL dimensional | good for regionals, weaker for GSIBs |
| Criticized / classified / special mention | XBRL dimensional | good where banks tag the regulatory ladder |
| Investment-grade share, risk-rating distribution | XBRL dimensional | GSIBs mostly; regionals use the pass/criticized ladder instead |
| Delinquency buckets | XBRL dimensional | good |
| Owner-occupied vs investor CRE | XBRL where broken out, else HTML | partial |
| Office CRE exposure | IR decks (4 banks) + 10-K property-type table (2) | 6 banks — see below |
| Leveraged lending | — | **not extracted**; every pattern read something else |
| Small business | IR supplements + XBRL | 2 banks |

Two structural limits worth stating plainly:

- **Office CRE and leveraged lending exposure are not in 10-K/10-Q XBRL.**
  They appear in investor supplements and earnings presentations only. No
  amount of tag hunting recovers them; see *IR supplements* below for the
  path that does, and for how far it actually gets.
- **JPMorgan and Citigroup do not tag CRE as a loan class.** They disclose
  wholesale exposure by *industry* instead, on a different axis. Their `cre_pct`
  is legitimately null rather than wrong; filling it requires the industry-axis
  table or the IR supplement.

`panel_coverage.csv` distinguishes `missing` from `n/a` — a custody bank with no
CRE book is not a coverage failure, and `reconcile.NOT_APPLICABLE` encodes which
variables are meaningless for which business model.

## IR supplements

Office CRE, leveraged lending, small business and the criticized/classified
ladder are disclosed in quarterly *investor supplements* and earnings
presentations, not in XBRL. `scripts/fetch_ir.py` collects those documents into
`data/ir/`, `ir_extract.py` reads them, and `reconcile.merge_ir` folds them in
behind XBRL and HTML.

```bash
uv run python scripts/fetch_ir.py --since 2020-01-01
uv run python scripts/fetch_ir.py --discover-only        # audit the routes
uv run python scripts/build_panel.py --since 2020-01-01 --html-fallback --ir
```

Fetching and building are separate on purpose: `build_panel --ir` reads only
what is already cached and never touches the network, so a rebuild is
reproducible.

### Getting the documents

There is no standard for how a bank publishes a supplement, so `ir.py` carries a
registry of three routes and records which applies to whom. `--discover-only`
prints the table.

| Route | How | Banks |
|---|---|---|
| `q4` | Q4 Inc hosts most large-cap IR sites. The pages are client-rendered and return **zero** document links to a scripted fetch, but the widget behind them reads a stable JSON feed at `/feed/FinancialReport.svc/GetFinancialReportList` carrying every quarter's documents with titles and types. | USB, RF, FITB, KEY, FCNCA, AXP, AMP, STT |
| `page` | A server-rendered index of quarterly PDFs, scraped by link pattern. | BAC, WFC, TFC, ALLY, BNY |
| `edgar8k` | Everyone else: the document list is JavaScript with no feed, the host returns 403 to a scripted client, or documents sit under opaque per-document UUID paths that cannot be enumerated. | the rest |

The `edgar8k` route is not a shortcut. A bank furnishes its quarterly supplement
and earnings deck as EX-99 exhibits to the Item 2.02 earnings 8-K — the same
documents, reachable through the rate limiter and cache already in `edgar.py`,
and frequently as **HTML where the IR site serves a PDF**, which extracts far
more reliably. Every bank gets `edgar8k` appended to its own route, because an
IR index is usually pruned: Truist's page lists earnings decks back six years
but no Quarterly Performance Summary at all, and the 8-K supplies it.

Two things make that route work rather than merely plausible:

- **Exhibits are classified by content, not by name.** A modern EDGAR filing
  index puts only `EX-99.2` in its description column and the filename says no
  more (`ex992-qpsx2q26.htm`), so each exhibit is read and identified from its
  own opening text. Verdicts are cached per CIK, and the cache is versioned so
  changing the classifier re-derives them.
- **Competing 8-Ks are ranked by earnings lag.** More than one 8-K can fall in a
  quarter's window. JPMorgan's February investor-day deck lands in the same
  window as its January earnings release and claimed the 2025Q4 supplement slot
  with a file that is nothing but slide images; filings are now visited in order
  of how close they sit to a typical two-to-three-week earnings lag, and a
  document with no extractable text is refused a slot outright.

Documents land at `data/ir/<TICKER>/<TICKER>_<YYYYQn>_<doctype>.<ext>` with a
manifest at `data/ir/manifest.csv` and `data/out/ir_manifest.csv`.

### Reading them

Three shapes, three readers — see the `ir_extract` docstring for the full
reasoning:

- **HTML tables** — `read_html`, one table at a time through lxml so each keeps
  the caption printed above it. The caption is load-bearing, not decoration: a
  bank's allowance rollforward, its charge-off table and its loan schedule all
  list the loan classes down the side and score identically on body text alone.
  Wells Fargo's supplement produced five "loan schedules" per quarter, four of
  them charge-offs, until the caption was allowed to decide.
- **PDF lines** — PDF supplements are typeset without cell borders, so table
  detection has to guess column boundaries from whitespace and guesses
  differently for two blocks on the same page. Lines are parsed directly
  instead.
- **Slide phrases** — office CRE exists in no table anywhere. Regions publishes
  it as a caption reading `Balances $858 % of Total Loans 0.9% NPL $121 ... ACL
  $35`, read by a section-scoped label/value scan.

### The traps

Each of these produced plausible, wrong, non-raising numbers, and each has a
test in `tests/test_ir.py`:

- **Average vs period-end.** Supplements print both on facing pages with
  identical row labels. Regions' end-of-period owner-occupied CRE is 4,890 and
  its six-month average is 4,882.
- **Sub-schedules.** The same variable is stated for the whole bank and again
  per segment, per property type, per risk grade. Wells Fargo's Commercial
  Banking segment reports C&I of 181,739 against a consolidated 487,630. Segment
  tables are skipped, and where readings still compete the largest wins — every
  sub-schedule is a subset of the consolidated figure.
- **Percentage twins.** The portfolio mix is published twice, in dollars and as
  a share of loans, with the same row labels.
- **Footnote markers.** `Home equity—lines of credit (1) 3,184` — `(1)` is a
  footnote, and reading it as a parenthesised negative turned a $3.2bn book
  into -$1.
- **Contents pages.** A supplement's table of contents pairs section titles with
  page numbers, and those titles contain every phrase the loan schedule matches
  on. Wells Fargo's loaded a page number into C&I.
- **Presentation-only signs.** An allowance rollforward writes the closing
  allowance as a deduction. Balances are taken as magnitudes; flows are left
  alone, because a provision release and a net recovery are both real.

### Scale

`reconcile.resolve_ir_scale` follows `infer_html_scale` — compare against the
XBRL value for the same bank-quarter, snap the ratio to 1/1e3/1e6/1e9, take the
modal snap per document — with one addition. Unlike a 10-K table, an IR document
usually *declares* its unit where a parser can see it, and a declared unit beats
an inferred one. It is also the only thing that can scale office CRE at all,
since there is no XBRL counterpart to compare against.

Declared units are read at three levels, most specific first, because PNC needs
all three: its supplement is headed "In millions" and then overrides individual
rows with "Average loans **In billions**". Reading only the table header puts
those rows out by a factor of a thousand.

Rows whose scale cannot be established are **dropped**, not guessed — the same
rule the HTML fallback uses.

### What it actually recovered

Over the 2020Q1–2026Q2 window, 976 documents (1.0 GB) across 30 of the 31
banks — HSBC USA publishes no quarterly earnings materials at all. From those,
2,168 extracted rows filled **532 panel cells directly and 576 more through the
ratios they enable**, against 110 cells from the HTML fallback. 50 variables
improved, none regressed.

The gains are concentrated where they were expected: `provision_to_nco` +9.9pp,
`ci_pct` +8.5, `nco_rate` +8.1, `loans_ci` +6.6, `cre_pct` +4.3, and the
owner-occupied/investor CRE split that XBRL only partly carries.

Three honest limits:

- **Office CRE reaches four banks, and each publishes something different.**
  An early version reached five and four of them were wrong: it read Bank of
  America's *share count* (7.87 billion shares booked as $7.9bn of offices),
  Citizens' *allowance* in place of the balance it is held against, First
  Citizens' *venture* book, US Bancorp's *NPA table*, and Zions' *chart axis
  labels*. Every value was plausible, nothing raised, and the column looked
  well populated.

  The cause is structural. A statistical schedule has a caption, row labels and
  aligned columns to lean on; a slide has none, so the only evidence that a
  number belongs to a heading is that it is printed nearby — and "nearby" is
  worthless when the neighbouring caption is a share count. Phrase specs are
  therefore an **allowlist**: one spec per bank whose deck has been read and
  checked, not one pattern stretched over the universe.

  | Bank | What it publishes | Recovered |
  |---|---|---|
  | ZION | the balance inside the slide heading, `Office ($1.6B)`, twice per deck | 16 quarters; 2.3 → 1.6 ($bn), plus nonaccrual rate |
  | RF | a fixed "Key Portfolio Metrics" block — balance, NPL, ACL, charge-offs | 9 quarters; 1,504 → 858 ($m) |
  | CFG | a sentence: "CRE General Office portfolio of $3.4 billion" | 6 quarters; 3.7 → 2.52 ($bn) |
  | USB | no balance since 2Q23, only the CRE split by property class | 6 quarters of `office_cre_share_of_cre`; 13% → 9% |

  **Bank of America and M&T come from the 10-K instead.** Neither publishes
  office exposure in its earnings deck — Bank of America names it in every
  quarter's materials but only ever as the *explanation* of a number ("net
  charge-offs ... driven by commercial real estate office"), never as one. It
  is not in the 10-Q either, and its XBRL instance carries no office member and
  no property-type axis at all.

  It is in the **10-K**, as a table: "Outstanding Commercial Real Estate Loans
  by Geographic Region and Property Type". `html_fallback.CRE_PROPERTY_TYPE`
  reads it, which is squarely that module's remit — its docstring already named
  office CRE as a target. Annual rather than quarterly, so the values land at
  Q4:

  | Bank | Recovered | Cross-check |
  |---|---|---|
  | BAC | 5 of 6 year-ends; 17.7 → 12.4 ($bn), 1.9% → 1.05% of loans | 15,061 − 12,447 = 2,614 vs the prose "decreased $2.6 billion, or 17 percent" |
  | MTB | 2023–2025; 4.73 → 3.42 ($bn) | maturity buckets sum to the total taken, all three years |

  The two tables are laid out differently — BAC's columns are years, M&T's are
  maturity buckets followed by a total — so both were checked row by row
  against the filings rather than assumed. M&T's population is its "permanent
  finance" CRE, a narrower book than the consolidated CRE line, so its office
  share of loans is not strictly comparable with Bank of America's.

  BAC's 2022 year-end is absent: too few rows in that filing snapped to a
  plausible multiplier, so the scale could not be established and the row was
  dropped rather than guessed — the same rule the rest of the HTML path uses.

  Six other large banks were checked for the same table and have none, so the
  spec does not contaminate anyone it was not verified against.

- **Leveraged lending is not extracted at all.** Every pattern tried against it
  read something else: Goldman's *net revenues* ($58.28bn), Citizens' *average
  hold position* ($12m), PNC's *CLO securitizations*, First Citizens' *rail*
  business, and, at its worst, US Bancorp's leveraged book at $1.18 **trillion**
  against a $381bn loan book. The disclosure is qualitative at almost every
  bank. The column exists and stays null, which is the correct answer.

- **Extraction is far better for regionals than for GSIBs and specialty
  lenders.** Wells Fargo, Zions, Regions, KeyCorp and Truist each yield 250–350
  rows; JPMorgan, Capital One and Ally yield a handful, because their
  supplements are laid out differently and their loan disclosures are thinner.

### What this stage does not do

`mix_coverage_pct` and the `mix_incoherent` flag are computed in
`panel.add_derived`, which runs on the XBRL frame before either fallback has
filled anything. They therefore describe the coherence of the *XBRL-sourced*
mix, not of the final panel. Ratios are refreshed after the merge
(`reconcile.refresh_ratios`, provenance `derived_ir`), but only where they were
null — an already-computed ratio is never disturbed — and the coherence metric
is deliberately left alone rather than recomputed from a duplicated copy of the
leaf-category list.

## Second source: FFIEC Call Reports

`callrpt_db` builds the same panel from the **Call Report** (FFIEC 031/041/051),
the quarterly regulatory filing every insured US bank makes. It is a different
source in every respect that matters: a different regulator, a different entity,
a fixed form rather than a disclosure choice, and — because the form states its
own totals — numbers that can be *checked* rather than merely believed.

```bash
export BANKQTR_UA="you@yourdomain.com"

uv run python scripts/fetch_call.py                         # ~850 MB, 102 quarters + NIC files
uv run python scripts/resolve_rssd.py --write-config        # bank -> RSSD, once
uv run python scripts/build_call_panel.py                   # 2001Q1 onward, with lineage
uv run python scripts/build_call_panel.py --since 2013-01-01 --no-lineage   # the 2026 RSSDs' own subtrees only
```

Fetching and building are separate on the same terms as the EDGAR side:
`build_call_panel.py` reads only what is cached and never touches the network.

Outputs land in `data/out/`:

| File | Contents |
|---|---|
| `call_panel.parquet` / `.csv` | the holding-company panel — 3,676 bank-quarters, 38 firms, 2001Q1–2026Q2 |
| `call_panel_charters.parquet` | one row per **bank charter** per quarter — 30,802 rows, 1,123 charters, each with the predecessor it came through |
| `call_panel_coverage.csv` | per bank and variable: how many quarters are populated |
| `call_panel_coverage_delta.csv` | before/after the lineage extension: quarters added per firm, and how many carry predecessor history |
| `call_panel_flags.csv` | bank-quarters failing a sanity check, or carrying synthetic (predecessor) history |
| `call_panel_build_info.json` | the window, settings, commit, cell counts — and the decision log for every schedule break and variable the 2001 window required |
| `rssd_lineage.csv` | every predecessor of every firm: who was absorbed into whom, when, and how (merger / fdic_assisted / reorg) |
| `rssd_resolution.csv` | how each firm was matched to its RSSD, with the evidence |
| `source_diff.csv` | EDGAR against FFIEC, per bank-quarter per variable |
| `source_diff_summary.csv` | the same, aggregated per bank and variable |
| `source_coverage.csv` | which firms each source reaches |

### What it buys

**The IHCs that EDGAR cannot reach.** This is the headline. Seven of the eight
firms in `config.NON_SEC_IHCS` are now in a panel, on the same schema as their
SEC-filing peers:

| | Loans, 2026Q2 | Quarters | From |
|---|---:|---:|---|
| TD Group US Holdings | $173.7bn | 54 | 2013Q1 |
| BMO Financial Corp | $141.6bn | 54 | 2013Q1 |
| UBS Americas Holding | $93.1bn | 44 | 2015Q3 |
| RBC US Group Holdings | $73.5bn | 33 | 2018Q2 |
| BNP Paribas USA | $60.1bn | 26 | 2016Q3 (last 2022Q4) |
| Barclays US | $29.2bn | 40 | 2016Q3 |
| DB USA Corporation | $15.5bn | 54 | 2013Q1 |

The windows differ because the IHC structure itself did: the Federal Reserve's
IHC requirement bit in mid-2016, and RBC's US bank arrived with the City
National acquisition. Credit Suisse Holdings (USA) is absent and correctly so —
its US intermediate holding company owned a broker-dealer, not a bank that files
a Call Report.

**A portfolio mix that cross-foots.** Schedule RC-C is a partition of a total
the schedule itself states, so `mix_coverage_pct` is not an estimate of
coherence — it is an arithmetic identity, and a departure from 100 is a
measurable quantity of loans in no category rather than a warning that
categories may overlap.

| | median `mix_coverage_pct` | 5th percentile |
|---|---:|---:|
| `bankqtr_db` (XBRL) | 90.0 | — |
| `callrpt_db` (Call Report) | **100.0** | 93.0 |

Across 30,802 charter-quarters the leaves tie to RC-C's own total **exactly**
in 29,249 of them (95.0%), on the 2001 form as on the 2026 one. None of the
remainder is an over-count beyond a handful of 2001 filers' own arithmetic
(a few thousand dollars on books of tens of millions); the rest is
under-allocation with a single cause described under *What it does not buy*
below.

**Owner-occupied against investor CRE, for everyone.** The EDGAR panel calls
this split "partial" because it depends on the bank breaking it out. Items F160
and F161 are fixed lines on the form, so every filer reports both, every
quarter.

**Capital.** CET1, tier 1, total capital and risk-weighted assets come off
Schedule RC-R. The XBRL path has no dependable tags for these and the EDGAR
panel carries none. Populated from 2015 onward — Basel III restructured RC-R
Part I that year — and **null** before it rather than approximated from the
old items. The regime before 2015 is carried under its own names
(`tier1_capital_basel1`, `total_capital_basel1`, `risk_weighted_assets_basel1`
and their ratios) and never spliced into the Basel III columns;
`tier1_leverage_ratio` is the one series that spans both, and the build info
says where its numerator's definition steps.

**Depth.** 102 quarters back to 2001Q1 — two full credit cycles, including
2007–2009, which the 2013-start EDGAR panel does not reach — and, because
each firm is followed through its predecessors, the 2007 rows are Wells Fargo
*and* Wachovia, PNC *and* National City, Truist as BB&T *and* SunTrust *and*
Colonial. See [Going back to 2001](#going-back-to-2001).

### What it does not buy

**It is not the same company.** A Call Report is filed by a *bank*. The panel
sums the depository charters under each holding company using the NIC control
graph, and that rollup excludes the broker-dealer, the credit-card funding
trusts, the insurance arms and every non-bank lender. Both directions of
divergence follow, and both are real here:

| | `loans_total`, FFIEC ÷ EDGAR |
|---|---:|
| Ameriprise | 0.19 |
| Santander USA | 0.65 |
| BNY Mellon | 0.76 |
| Goldman Sachs | 0.85 |
| median across 31 firms | **1.00** |
| Bank of America | 1.00 (87% of quarters agree within 0.5%) |

Ameriprise at 0.19 is not a defect: almost all of its lending happens outside
Ameriprise Bank. Ratios above 1.00 are intercompany — a loan from the bank to
its own broker-dealer is an asset on the Call Report and is eliminated in the
holding company's consolidation. Every holding-company row carries `n_charters`
and `charters` so a reader can see exactly what was summed.

**The categories are the form's, not the filing's.** `loans_ci` runs about 0.80
of the 10-K's C&I line and `loans_cre_total` about 1.11 of its CRE line, and
neither is wrong. RC-C puts loans to nondepository financial institutions and
municipal obligations on their own lines, where a 10-K usually folds them into
C&I; and RC-C's CRE includes owner-occupied, which many 10-Ks exclude. Use one
source or the other for a mix comparison — not a mixture of the two.

**Some loans are in no category, and only on form 031.** Items 1545 (securities
lending), 2165 (leases), J451 and J454 are collected in the *domestic-office*
column while the total is consolidated, so a bank with foreign offices has
exposure the form never breaks out abroad. BNY Mellon at 2017Q3 is the clearest
case: $29.5bn consolidated against $15.6bn domestic, with $4.9bn of the
difference belonging to lines that have no foreign column to sit in. This is
reported, not hidden — as `loans_unallocated` in dollars, as `rcc_residual_pct`
as a share, and as the `rcc_partition_under` flag, which fires on 1,308 of
3,676 bank-quarters. The only other arithmetic flags the build raises are one
`rcc_partition_over` — $8,000 on M&T's $70bn book in 2003Q1, a filer's own
rounding — and one `nonaccrual_exceeds_loans`, at Ameriprise Bank in 2012Q3,
when nearly its whole book was held for sale and so outside `loans_total`.
Everything else in `call_panel_flags.csv` is the predecessor bookkeeping
described under [Going back to 2001](#going-back-to-2001).

**Office CRE and leveraged lending are not here either.** The Call Report has no
property-type breakdown. Those remain IR-supplement territory, and the EDGAR
build is the one that reaches them.

### Pipeline

```
cdr.py         CDR bulk download (an ASP.NET form, not an API), cached per quarter
nic.py         NIC entity graph: who owns whom, dated; who became whom; EIN bridge to EDGAR
lineage.py     each 2026 firm's predecessors, quarter by quarter, from the NIC graph
mdrm.py        MDRM item codes -> panel variables across four redesigns of the form
schedules.py   bulk TSV -> long frame -> one column per variable
panel.py       long -> charter panel -> holding-company panel, ratios, checks
crosscheck.py  EDGAR against FFIEC: agreement, stability, coverage gained
```

### The traps

Each of these produced plausible, non-raising, wrong numbers, and each has a
test in `tests/test_callrpt.py`.

- **The prefix is a reporting basis.** `RCFD` is the consolidated bank, `RCON`
  is domestic offices only. Preferring `RCON` reads JPMorgan's loan book as
  $1,340bn instead of $1,497bn — an 11% understatement that looks entirely
  plausible next to its peers. Insisting on `RCFD` instead drops Zions, which
  files no `RCFD` at all. And the preference is **per item, not per schedule**:
  Schedule RC-N reports its totals on `RCFD` and its entire real-estate
  breakdown on `RCON`, in the same filing, for the same bank.
- **There are two "total loans".** RC-C item 12 includes loans held for sale;
  RC item 4.b does not. The EDGAR panel's `loans_total` is held-for-investment,
  so `B528` is the comparable one — and `B528 + 5369 == 2122` holds to the
  dollar on every filer checked, which is how the two were told apart.
- **Rollups sit beside their own components.** A filer can report the coarse
  line *and* the detail it totals. Morgan Stanley Bank reports item 1563 next to
  the J454, J451 and 1545 that sum to it exactly; counting both put its loan
  book 62% over its own reported total. Another filer does the same with J464.
  And `1563` is not `J464` — 1563 covers the whole of RC-C item 9 including
  loans to nondepository financial institutions, while J464 covers only item
  9.b.
- **…and the detail can be zeros.** The mirror-image trap. Zions reports the
  five-way interbank split as explicit zeros next to the single line 1288 that
  carries the real $59m, so "prefer the detail where present" reads its
  interbank book as nothing. Variants are therefore chosen by **how completely
  the filer reported them**, with ties going to the coarser line.
- **Which shape a filer uses is not a function of the form.** Keying the choice
  on 031/041/051 got 5,368 of 6,974 filers wrong in 2013Q1, because a small 041
  filer before 2017 collapses C&I into 1766 and leases into 2165 exactly as the
  short form does — the short form did not exist yet.
- **Item 12 subtracts item 11.** Unearned income is deducted from the loan
  total. Almost every filer reports zero there, which is why omitting it went
  unnoticed until City National Bank, whose $562m put it 0.86% over its own
  total in every quarter with nothing else wrong.
- **Every flow is year-to-date.** JPMorgan's charge-offs run 730 → 3,346 →
  5,022 → 6,810 through 2019 and reset at 1,902 in 2020Q1. Unlike the XBRL
  panel, where cumulative-only tagging is one bank's quirk, this is how *every*
  `RIAD` item behaves for *every* bank. Quarters are recovered by differencing,
  **per charter and before the rollup** — otherwise a holding company that
  acquires a bank mid-year books that bank's whole year to date as one quarter.
- **The second row is not data.** Each bulk file has MDRM codes on row 1 and
  human labels on row 2. A reader that starts at row 2 gains an institution
  whose every field is a caption.
- **`CONF` is not zero.** A withheld line read as zero reports a bank's
  confidential exposure as an absence of exposure.
- **The relationship graph is dated.** `DT_END = 99991231` means open; anything
  else closed. Reading the graph as current puts SunTrust Bank under Truist in
  2014. Every quarter resolves its charter set against the graph as it stood
  then.
- **The EIN join has a leading zero.** NIC stores `ID_TAX` as a number and drops
  it; EDGAR keeps it. Joining the raw strings misses Citizens Financial Group
  (`050412693` against `50412693`), State Street and Santander USA — and
  Citizens then fell through to a name match, where there is an *unrelated*
  active bank holding company also called Citizens Financial Group, Inc.
- **The sum of nothing is zero.** Polars sums an all-null column to 0, so a
  column no charter reported rolled up as 0.0 rather than null: CET1 of 0.0
  for every bank before 2015, nonaccrual of 0.0 for every bank before 2017.
  The rollup now keeps null.
- **RC-N had no total row before 2017.** Items 1403/1406/1407 exist from
  2017Q1, so a build that reads only them carries nothing for 2013–2016. The
  total is built from fourteen category rows, checked against the form's own
  total from 2017 on (99.6% tie) — which is how it emerged that agricultural
  loans and real-estate loans to non-US addressees are memoranda inside other
  rows, and that form 031's foreign-office loans sit on their own `RCFN` line.
- **F180 is not nonaccrual.** RC-N lays out owner-occupied CRE as F178
  (30–89), F180 (90+), F182 (nonaccrual). The nonaccrual columns read the
  90-days-past-due one; JPMorgan's nonaccrual owner-occupied CRE came out at
  $1m.
- **Push-down accounting restarts the year.** An acquired charter's
  year-to-date begins again on the acquisition date, so differencing Fleet
  National Bank's 2004Q2 against its Q1 gave charge-offs of −$115m. A fall in
  a gross flow that cannot fall marks the restart, and the quarter is taken
  as the year-to-date since it.
- **Pooling restates the survivor.** A common-control merger has the survivor
  report income as if combined from January 1, so its next difference carries
  the absorbed charter's whole year to date — already counted under that
  charter's own RSSD. NIC's accounting-method flag says which mergers these
  are and the amount is taken back out.
- **Acquisitions are not transformations.** NIC's transformations table
  records mergers and failures; a company bought and kept as a subsidiary —
  Countrywide, Merrill Lynch, Bear Stearns, MUFG Americas — appears only as a
  control relationship that begins on the closing date. Both tables are read.
- **NIC's attribute files omit real filers.** Discover Bank, MUFG Union Bank,
  FirstMerit Bank and National City Bank have no attributes row. The Call
  Report roster is the authority on who is a depository, and supplies the
  names NIC lacks.

### Identity: which RSSD is which bank

NIC carries no CIK, so `scripts/resolve_rssd.py` bridges on **EIN** — a federal
tax identifier both registries collect independently from the filer — rather
than on company name. Names are used only for the IHCs, which have no SEC
registration to carry an EIN. Every match is written to `rssd_resolution.csv`
with the evidence that produced it, so a name match is visibly weaker than a tax
match rather than indistinguishable from one.

Two refinements were needed and both came from the data:

- **Candidates are ranked by whether they own a bank.** Three distinct *active*
  holding companies are called some form of "Citizens Financial", one of them a
  $220bn DFAST participant and another the owner of a single bank in West
  Virginia. Comerica, Valley National and Synovus each have a plausibly-named
  sibling entity with no charter under it. Having a Call Report filer underneath
  is the only criterion that tests what the panel actually needs.
- **A firm is not one RSSD for all time.** Zions Bancorporation merged its
  holding company into its own bank in 2018: RSSD 1027004 until then, 276579
  after, with a different EIN and no parent. `Holding.also_rssd` carries the
  other era, and the quarter's filers are the union — seven charters before the
  merger, one after.

### Going back to 2001

The panel's floor is 2001Q1, and before 2013 — and in a good many quarters
after — the organisation summed is not the 2026 RSSD's own subtree but its
**lineage**: every predecessor later merged, acquired or failed into it, for
every quarter that predecessor still stood on its own. `rssd_lineage.csv` is
the map (1,654 predecessors of 38 firms; 95 of them FDIC-assisted failures,
125 acquisitions that NIC records only as a change of control),
and every bank-quarter says how much of it is reconstructed:

| Column | Meaning |
|---|---|
| `has_predecessor`, `predecessor_count`, `predecessors` | summed across organisations that were separate at the time, and which |
| `predecessor_failed` | includes an institution that subsequently failed — the stress observations |
| `n_insured_not_filing` | insured depositories in the tree that filed no Call Report (TFR-filing thrifts before 2012Q1): the sum is a floor |
| `rcn_total_built` | RC-N total built from category rows (the form had none before 2017) |
| `n_flow_resets` | charters whose income statement restarted that quarter (push-down accounting) |

Three consequences to read before using the long window:

- **Thrifts are invisible before 2012Q1.** Washington Mutual, Golden West,
  Countrywide Bank, Sovereign, ING Direct and IndyMac filed a Thrift Financial
  Report, which CDR does not hold. They are in the lineage and contribute
  nothing; JPMorgan's loan book jumps by WaMu's entirety at 2008Q3, and
  `n_insured_not_filing` says why.
- **A firm tracked in its own right is never also its acquirer's history.**
  Discover Bank is Discover's row until Discover's last filing and Capital
  One's only afterwards.
- **The 2013–2026 rows changed too.** Truist carries SunTrust from 2013Q1;
  First Citizens carries CIT and SVB; the EDGAR cross-check compares such rows
  against the standalone SEC filer and they legitimately differ — read
  `source_diff_summary.csv` on the rows with `has_predecessor = false`. And
  four defects in the previous build are corrected: the all-null-to-zero
  rollup, the missing RC-N totals for 2013–2016, the CRE nonaccrual columns
  reading the wrong RC-N column, and null C&I charge-offs for every 041 filer.

Twenty-odd variables were added for the window — noncurrent loans and the
Texas ratio, delinquency and nonaccrual by loan category, TDRs, segment
charge-off rates, unused commitments, brokered deposits and wholesale funding,
tangible common equity, the pre-Basel III capital items, PPNR — and two
coarser partition leaves (`loans_cre_nonfarm_nonres`,
`loans_consumer_installment`) carry CRE and consumer lending continuously
across the 2007 and 2011 form changes that the finer columns cannot cross.
`call_panel_build_info.json` records every one with its items, its rationale
and its comparability caveats; [docs/extending-to-2001.md](docs/extending-to-2001.md)
records how the lineage is derived and what the form changes required.

### Testing

```bash
uv run pytest tests/test_callrpt.py tests/test_lineage.py -q
```

62 tests. The unit tests build a bulk file in memory with the real row shape,
and a small NIC by monkeypatching the structure loaders, so they run without
`data/raw`; the integration tests are skipped when no quarter or no NIC file is
cached. The strongest are `test_partition_ties_for_a_real_quarter` (and its
2001 twin), which assert across every filer in a quarter that the loan
categories never sum to *more* than RC-C's own total, and
`test_rcn_categories_reproduce_the_forms_total`, which holds the pre-2017
RC-N total construction to the form's own total where both exist.
`test_lineage.py` pins lineage completeness over the universe, that a
predecessor and its successor are never both summed in a quarter, and that
Colonial, Washington Mutual, First Republic and Park National are flagged as
FDIC-assisted.

### Extending

- **New variable**: add an `ItemSpec` in `mdrm.py`. Give it `alternatives` if
  some filers report a coarser line instead, and `prefixes` if it is not an
  ordinary balance-sheet item.
- **New partition check**: add a `PartitionCheck`. Any schedule stating its own
  total can be cross-footed the way RC-C is.
- **New bank**: add it to `bankqtr_db/config.py` and rerun
  `scripts/resolve_rssd.py --write-config`; the RSSD, the charter set and the
  evidence are all derived.
- **An older code for an existing item**: append it as an `alternatives`
  group, coarsest first, never as a new item; the most-complete-variant rule
  then prefers the modern detail wherever a filer reports it. The partition
  checks and `test_rcn_categories_reproduce_the_forms_total` say exactly what
  a change did across every filer.
- **A new predecessor rule**: `lineage.resolve` has two — transformations into
  a member, and relationships that bring an outsider in. Anything added there
  must keep `test_predecessor_and_successor_in_the_same_quarter_are_not_both_summed`
  green.

## HTML fallback

`pandas.read_html` over the primary document, with tables located by matching
row *labels* rather than position (table ordering is not stable across banks or
years). Two notes:

- Filings are fetched as **bytes**: inline-XBRL documents open with an XML
  declaration, and lxml refuses a `str` carrying one.
- `read_html` cannot see the "(in millions)" header, so parsed magnitudes are
  unresolved. `reconcile.infer_html_scale` recovers the multiplier by comparing
  against XBRL values for the same bank-quarter and applies it per filing; rows
  whose scale cannot be established are **dropped**, not guessed. A silently
  mis-scaled peer is worse than a missing one.

## Testing

```bash
uv run pytest tests/ -q
uv run ruff check .
```

245 tests across both builds. Every one corresponds to a defect that produced
plausible but wrong numbers during development — the dangerous kind, since
nothing raises and the panel still renders. `tests/test_callrpt.py` and
`tests/test_lineage.py` cover the FFIEC path and need no cached data for all
but their integration cases.

## Extending

- **New variable**: add a `VarDef` in `variables.py`. The XBRL path, the HTML
  path and the coverage report all read from that one definition.
- **New ratio**: add a `RatioDef`; it is computed automatically wherever its
  inputs exist.
- **New bank**: add a `Bank` to `config.py`, and an `IRSource` to `ir.py` if
  its IR site is reachable — without one it still works, falling back to the
  8-K exhibits.
- **New IR disclosure**: add an `IRTableSpec` (for a schedule) or a
  `PhraseSpec` (for a slide caption) in `ir_extract.py`. A spec that populates
  a column no filing carries also needs its `RatioDef`s in `variables.py` and
  an entry in `reconcile.NOT_APPLICABLE` for the business models it cannot
  apply to.
- **Unmapped members**: `taxonomy.unmapped()` reports dimension members seen but
  not mapped. An unmapped member means a bank's disclosure is silently missing
  from the panel, so the build prints the count on every run.
- **An older filing era**: element *and* axis names both changed in the 2018
  taxonomy, and both are handled the same way — append the legacy name to the
  tuple in `variables.py` or `instance.PROMOTED_AXES`, never prepend. Ranking
  is by position for tags, and by `panel.MODERN_LOAN_AXES` for axes, so a
  legacy name is reached only where a filing carries nothing else. Getting that
  backwards rewrites the modern panel rather than extending it;
  `docs/extending-to-2013.md` records what happened when the axis half of that
  rule was missing.
