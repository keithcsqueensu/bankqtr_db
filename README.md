# bankqtr_db — DFAST bank peer benchmarking from SEC EDGAR

A bank-quarter panel database for peer benchmarking, built from SEC EDGAR
filings. One row per bank-quarter, columns per variable, with provenance and
coverage reporting attached.

## Quick start

```bash
uv sync
export BANKQTR_UA="you@yourdomain.com"          # SEC requires a real contact

uv run python scripts/fetch_facts.py            # companyfacts + submissions
uv run python scripts/fetch_instances.py --since 2020-01-01
uv run python scripts/fetch_ir.py --since 2020-01-01       # IR supplements
uv run python scripts/build_panel.py --since 2020-01-01 --html-fallback --ir
```

Outputs land in `data/out/`:

| File | Contents |
|---|---|
| `panel.parquet` / `.csv` | the bank-quarter panel |
| `panel_facts.parquet` | every extracted fact, long format, with dimensions |
| `panel_coverage.csv` | per bank and variable: how many quarters are populated |
| `panel_gaps.csv` | variables ranked by how many banks lack them |
| `panel_flags.csv` | bank-quarters failing a sanity check |

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
rather than an unexplained hole. Closing that gap needs FFIEC/NIC call-report
data, which is a different source and a separate build.

Firms that stopped filing mid-window (Discover, acquired 2025; MUFG Americas,
2021) carry a `last_filing` date so "no disclosure" stays distinguishable from
"no longer exists".

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
| Office CRE exposure | IR presentations + supplements | 3 banks, 3 different disclosures — see below |
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

- **Office CRE reaches three banks, and each publishes something different.**
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
  | RF | a fixed "Key Portfolio Metrics" block — balance, NPL, ACL, charge-offs | 9 quarters; balance 1,504 → 858 ($m) |
  | CFG | a sentence: "CRE General Office portfolio of $3.4 billion" | 6 quarters; 3.7 → 2.52 ($bn) |
  | USB | no balance since 2Q23, only the CRE split by property class | 6 quarters of `office_cre_share_of_cre`; 13% → 9% |

  Three details that each cost a wrong number to find. Citizens' magnitude word
  belongs to the figure, not the page — consuming "billion" inside the match
  left a $3.4bn book scaled by the document's "in millions" caption. Regions'
  block always leads with its balance, so a quarter yielding only a charge-off
  has matched something else. And US Bancorp's later decks are PDFs whose text
  extraction interleaves characters from adjacent boxes, out of which the
  office share read 21% and 19% against a real trend of 13, 13, 13, 12, 10, 9 —
  so the legend's slices must still sum to 100 or nothing is taken from it.

  Citizens' allowance is deliberately left alone: its memo line puts the
  **prior** quarter first, the reverse of every other schedule here. And
  `office_cre_share_of_cre` is never *derived*, only taken where a bank states
  it — deriving it needs a complete CRE denominator, and Regions has one in two
  quarters out of nine, understated enough to report a ~15% share as 31%.
  `office_cre_pct`, against total loans, carries the cross-bank comparison.

  Adding a bank means reading its deck and confirming the block — not
  loosening a pattern until something matches.

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

Every test corresponds to a defect that produced plausible but wrong numbers
during development — the dangerous kind, since nothing raises and the panel
still renders.

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
