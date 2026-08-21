# Extending the Call Report panel to 2001Q1

The FFIEC panel now starts at **2001Q1**, the first quarter CDR offers, and
every firm in it is carried back through its **predecessors**: the
organisations that were later merged, acquired or failed into what is now
each 2026 DFAST institution. This note records what that required, what it
changed in the 2013–2026 rows that already existed, and what a user of the
longer window has to know. The shorter, operational version is in
[README](../README.md#going-back-to-2001); the decision log for every
variable is in `data/out/call_panel_build_info.json`.

```bash
uv run python scripts/fetch_call.py                  # 102 quarters, ~850 MB, NIC files
uv run python scripts/build_call_panel.py            # 2001Q1 onward, with lineage
uv run python scripts/build_call_panel.py --no-lineage --since 2013-01-01   # the old build
```

## Why the survivor's RSSD is the wrong entity before 2013

The universe is today's DFAST list, keyed on each firm's 2026 RSSD, and the
2013 build resolved that RSSD's charters quarter by quarter against the dated
NIC graph. That is exact for the entity that exists today and wrong for most
of the window. Wells Fargo in 2007 was Wells Fargo *and Wachovia*, two
organisations of similar size; PNC was PNC and National City; Truist was BB&T
and SunTrust and Colonial. A panel that carries only the survivor's own RSSD
back to 2001 shows Wells Fargo's loan book doubling at 2008Q4, reads it as
growth, and loses exactly the institutions whose 2007–2009 deterioration is
the reason to go back that far.

So the organisation summed in each quarter is its **lineage**: its own
subtree plus the subtree of every predecessor still standing on its own that
quarter. `callrpt_db/lineage.py` resolves the map; `panel.universe_filers`
applies it; `rssd_lineage.csv` is the map itself.

## Where the lineage comes from

NIC publishes a fourth structure file the package did not previously read,
**transformations**, recording every event in which one entity's balance
sheet passed to another. Its codes were read off the NIC data dictionary,
not guessed:

| `TRNSFM_CD` | Meaning | Used |
|---|---|---|
| 1 | Charter discontinued — merger or purchase and assumption | yes |
| 9 | Charter retained under a new RSSD, 95%+ of assets transferred | yes |
| 50 | Failure, government assistance provided | yes → `fdic_assisted` |
| 5, 7 | Split / sale of 40–94% of assets; predecessor continues | no |

The attributes table carries an independent record of failure
(`REASON_TERM_CD` 4 or 5), and the two agree on every case checked —
Washington Mutual, Colonial, First Republic, Park National. The union is
taken.

### The walk runs over members, not over the top entity

Washington Mutual Bank was sold by the FDIC to JPMorgan Chase *Bank*, a
charter, not to JPMorgan Chase & Co. Countrywide was merged into a Bank of
America subsidiary. A walk over the holding company's own transformations
finds Bank One and Fleet and misses both. So, for every quarter end from
2026Q2 back to 2001Q1, the organisation's *members* — own roots, every
predecessor still standing, and everything the dated graph puts beneath them
— are the set whose absorptions are examined, and an event counts only when
its successor was a member at the quarter end it falls in. A predecessor
found in 2008 is already a root when 2007 is examined, so SouthTrust (into
Wachovia, 2004) is Wells Fargo history and the FBOP banks (into U.S. Bank,
2009) are US Bancorp's.

### Acquisitions are not transformations

Countrywide Financial, Merrill Lynch & Co., Bear Stearns, MUFG Americas and
Golden West were **bought and kept as subsidiaries**. NIC records no
transformation until the shell is dissolved — 2013 for Merrill, never for
Countrywide — by which time the banks beneath it have long since been moved.
What NIC does record is a control *relationship* beginning on the closing
date. The second discovery rule therefore treats as a predecessor any entity
that was outside the organisation at the previous quarter end, is inside it
at this one, joined through a relationship that began in between, existed
before the quarter, and is or owns a depository. Such rows carry
`transformation_code = ACQ` and `succession_type = merger`. 125 were found,
and where such an entity's shell is later dissolved the acquisition date is
the one recorded, not the dissolution (MUFG Union Bank: 2022-12-01, not its
2023 merger into U.S. Bank).

### Three kinds of succession

| `succession_type` | What it means | Adds history? |
|---|---|---|
| `merger` | an outside organisation absorbed whole (or acquired and kept) | yes |
| `fdic_assisted` | the predecessor failed and the FDIC arranged the disposition | yes — the stress observations |
| `reorg` | the predecessor was already a member when absorbed (Chase Bank USA into JPMorgan Chase Bank; Zions' holding company into its own bank) | no — its reports were already summed |
| `self` | the firm's own top entity (and Zions' dissolved one) | — |

Reorgs are listed because the file should be complete and because they are
the events that restate year-to-date flows (below).

### What is deliberately not done

- **A firm tracked in its own right keeps its own row.** Discover Bank is
  Discover's until Discover's last filing and Capital One's only afterwards.
  `universe_filers` claims every firm's own charters first and predecessor
  charters second; a charter is never claimed twice in a quarter, and the
  lineage file marks the case `tracked_separately`.
- **A predecessor dated on a quarter end is claimed from both sides.** The
  merger date is inclusive at both ends of `effective_from..effective_to`,
  because the dated graph can put the charter under either parent on that
  day; the per-holding set union dedupes it. A charter reached through two
  live predecessors — the holding company that was merged, and the charter
  itself when later folded into a sibling — is attributed to the outside
  organisation, so `predecessors` names what was bought.
- **Thrifts are the gap the Call Report cannot close.** Washington Mutual,
  Golden West, Countrywide Bank, Sovereign, ING Direct, Hudson City, IndyMac
  and E\*TRADE Bank filed a Thrift Financial Report, not a Call Report, until
  2012Q1, and CDR has nothing for them. Their lineage rows are written — the
  succession is real — and `n_insured_not_filing` counts, per bank-quarter,
  the insured depositories in the organisation's tree that filed nothing CDR
  holds, so a reader can see where the synthetic history is a floor.
  JPMorgan's 2008Q3 loan book jumps by WaMu's entirety for this reason; it is
  flagged, not hidden. NIC's coverage of thrift holding companies before the
  Fed took them over in 2011 is thin (Golden West's holding company record
  ends in 2001), so the gap count is itself a floor.
- **NIC's attribute files omit real filers.** Discover Bank, MUFG Union
  Bank, FirstMerit Bank, National City Bank and PNC Bank Delaware have no row
  in either attributes file. The walk therefore takes the set of every RSSD
  that ever filed a Call Report as its definition of "is a depository", and
  the lineage file names such entities from the Call Report roster.

## What the panel carries for it

Per bank-quarter: `has_predecessor`, `predecessor_count` (distinct
predecessor organisations summed), `n_predecessor_charters`, `predecessors`
(their RSSDs), `predecessor_failed` (a summed charter, or the predecessor it
came through, later failed), `n_insured_not_filing` / `insured_not_filing`
(the thrift gap), `rcn_total_built` (RC-N total built from categories, see
below) and `n_flow_resets` (charters whose income statement restarted that
quarter). `call_panel_flags.csv` repeats the first three as rows
(`predecessor_history`, `predecessor_failed`,
`insured_depository_not_filing`) so a filter on the flags file finds them.
The charter file carries `via_rssd` and `via_type` per charter-quarter.

## Flows across a merger: two more traps

The existing rule — difference year-to-date flows **per charter, before the
rollup** — is necessary and not sufficient. Two accounting conventions break
it, and both produced large wrong numbers before they were handled:

- **Pooling restates the survivor.** When one charter is merged into another
  under common control, the survivor reports income *as if combined from
  January 1*. Its next difference then contains the absorbed charter's whole
  year to date — quarters that charter had already filed itself. NIC's
  `ACCT_METHOD = 1` marks these; `quarterize` subtracts the predecessor's
  previous-quarter year-to-date from the survivor's quarter. 219
  charter-quarters were adjusted.
- **Push-down restarts the acquired.** When a charter is acquired and the
  purchase price is pushed down to its books, its income statement begins
  again on the acquisition date, and its next year-to-date covers only the
  weeks since. Differenced against the previous quarter's full year-to-date
  that is a large negative: Fleet National Bank's 2004Q2 charge-offs,
  LaSalle's 2007Q4, National City's and Wachovia's 2008Q4. A restart is
  recognised when any gross flow that cannot fall (charge-offs, recoveries,
  interest income and expense, noninterest expense) has fallen; the quarter is
  then the year-to-date since the restart and `flow_reset` marks the row. What
  is lost is the charter's activity between the previous quarter end and the
  acquisition date; what is avoided is a negative loss.

Neither correction is exact, and both are documented as such in
`not_strictly_comparable` in the build info.

## The form changes under you

A 2001 start crosses four redesigns of the schedules. Every break below was
*measured* — the first and last quarter each code carries a value, counted
over all ~8,900 filers in every one of the 102 bulk files — and the rule for
all of them is the same: the modern codes are `items`, the retired ones are
`alternatives`, and a column's meaning never changes across the break. Where
the old form cannot support the modern meaning, the modern column stays null
and a coarser column that is consistent across the whole window is added
beside it.

| Quarter | Change | Handling |
|---|---|---|
| 2002Q1 | closed-end 1-4 family split into first/junior liens (RC-N, RI-B); fed funds & repos split | alternatives |
| 2007Q1 | construction and nonfarm nonresidential each split in two on RC-C, RC-N, RI-B, RC-L; leases split | `loans_construction`, **`loans_cre_nonfarm_nonres`** and `loans_lease` continuous; owner-occupied / investor null before 2007. Through 2007 the old codes are carried as derived totals and tie to the detail exactly (1415 = F158 + F159 for all 4,722 filers reporting both), so the most-complete-variant rule picks the detail |
| 2010Q1 | RC-C item 9 reorganised around nondepository FIs; RC-L other commitments split | `loans_other_total` continuous (the pre-2010 `1564` line added); `loans_nondepository_fi` null before |
| 2011Q1 | other loans to individuals split into auto and other (RC-C, RC-N, RI-B); TDRs by category | **`loans_consumer_installment`**, `charge_offs_consumer_noncard` continuous; `loans_auto`, `loans_consumer_other` null before |
| 2015Q1 | Basel III RC-R | as before: `cet1_capital`, `tier1_capital`, `total_capital`, `risk_weighted_assets` null before 2015Q1. The prior regime is carried under its own names, `*_basel1`, never spliced; `tier1_leverage_ratio` splices the numerators over RC-K average assets and says so |
| **2017Q1** | **RC-N gains a total row** | before it the totals are the sum of fourteen category rows — see below |
| 2018Q2 | goodwill moves RC → RC-M; other intangibles replaced by total intangibles | `goodwill` read from either schedule, never both; `intangibles_total` is 2143 or 3163 + 0426 |
| 2023Q1 | ASU 2022-02 replaces TDRs | `loans_tdr_accruing` continues under the broader definition; flagged |

### RC-N had no total before 2017

Items 1403/1406/1407 exist from 2017Q1 and not before. The 2013 build read
only them and — because the rollup summed all-null charters to zero — carried
**0.0** for nonaccrual and past-due totals in every one of its 2013–2016
bank-quarters, and for every ratio on them. Before 2017 the total is now the
sum of the category rows, and the category grid is checked against the form's
own total from 2017 on: it ties for 99.6% of filers. Getting there took three
findings, all from the data:

- **Agricultural loans are a memorandum.** They look like a row and the same
  dollars sit inside "all other loans"; for 877 of the 1,017 filers reporting
  agricultural past-dues in 2001Q1 the two lines are identical. Counting
  both put 523 filers over the 2017Q1 total.
- **Real-estate loans to non-US addressees are a memorandum too.** Mercantil
  Bank's sum came out 60% over by exactly that line.
- **Form 031's real-estate rows are domestic only.** The loans booked abroad
  are one line on the `RCFN` prefix (B572–B574); with it, 49 of the 70 031
  filers that were short tie exactly.

`rcn_total_built` marks the rows where the total was built. The same grid
fixes a second defect: `nonaccrual_cre_owner_occupied` and
`nonaccrual_cre_investor` were reading F180/F181, the **90-days-past-due**
column; nonaccrual is F182/F183.

### The null-versus-zero rollup

`roll_up` summed each column over a holding company's charters with Polars'
`sum`, whose sum of all-null is 0. Every column no charter reported was 0.0
rather than null at holding level — CET1 before 2015, the RC-N totals before
2017, and everything pre-2007 or pre-2011 would have joined them. It is now
null. This is the change most likely to be visible in a 2013–2026 comparison
against the previous build, and it is a correction.

## Variables added

Each is on the form since before 2001 under one code or another, and each
exists because the 2001–2012 window is where it earns its keep. The full list
with items and rationale is `variables_added` in the build info; the ones that
matter most for a stress model:

| Variable | Why |
|---|---|
| `noncurrent_total`, `noncurrent_ratio`, `reserve_to_noncurrent` | the FDIC's noncurrent definition; the series in every published GFC study |
| `texas_ratio` | the failure predictor that worked in 2009–2011 |
| `dpd_30_89_*`, `dpd_90_plus_*`, `nonaccrual_*` by category | early-stage delinquency by loan type; residential and construction 30–89 turned up in 2006, four quarters ahead of nonaccruals |
| `loans_tdr_accruing`, `tdr_pct` | the 2009–2012 restructuring wave |
| `nco_rate_construction`, `_cre_nonfarm_nonres`, `_multifamily`, `_resi`, `_home_equity`, `_consumer_noncard`; `provision_rate` | segment loss rates: the calibration targets |
| `commitments_*` (six RC-L lines), `standby_letters_of_credit`, `commitments_total`, `commitments_to_loans` | the unfunded pipeline drawn in a stress |
| `deposits_brokered`, `brokered_deposits_pct`; `fed_funds_repo_purchased`, `borrowings_other`, `wholesale_funding_pct` | the funding profile of the institutions that failed |
| `goodwill`, `intangibles_total`, `preferred_stock`, `tangible_common_equity`, `tce_ratio`, `equity_to_assets` | capital as the market re-measured it in 2008–2009; TARP preferred deducted |
| `tier1_capital_basel1`, `total_capital_basel1`, `risk_weighted_assets_basel1`, the two ratios, `tier1_leverage_ratio` | regulatory capital through two cycles under the definitions then in force |
| `interest_income`, `interest_expense`, `ppnr`, `ppnr_rate`, `roa`, `nii_to_avg_assets`, `assets_average`, `loans_average` | pre-provision net revenue, the quantity DFAST projects |
| `securities_htm`, `securities_htm_fair_value`, `securities_afs_amortized_cost`, the two unrealised columns | 2022–2023, and the 2008 private-label MBS marks |

Every new column with a structural break carries it in
`not_strictly_comparable` in the build info.

## What it did to the 2013–2026 rows

The existing rows change in four ways, all intended:

1. **Predecessors inside the window.** Truist now carries SunTrust from
   2013Q1, not 2019Q4; First Citizens carries CIT and SVB; JPMorgan carries
   First Republic; Huntington carries FirstMerit and TCF; Fifth Third carries
   MB Financial; KeyCorp carries First Niagara; M&T carries Hudson City and
   People's United. These rows are the 2026 firm's pro-forma past and say so
   (`has_predecessor`). The EDGAR cross-check (`source_diff.csv`) compares
   them against the standalone SEC filer and they legitimately differ; the
   cross-check summary should be read on the rows with `has_predecessor =
   false`, where agreement is what it was (Bank of America's loan book agrees
   within 0.5% in 87% of quarters).
2. **Nulls where there were zeros** (above).
3. **RC-N totals for 2013–2016**, and the CRE nonaccrual columns, corrected.
4. **C&I charge-offs, recoveries, nonaccrual and 30–89 past due for every
   form 041 filer**, which were null because only the 031 by-addressee items
   were mapped.

## Before and after

`call_panel_coverage_delta.csv` has the full table; the build log prints its
head. 1,929 bank-quarters became 3,676 (+1,747); 38 firms; 271 columns. The
firms that gained most are the ones whose 2026 entity is young and whose
history is all predecessors — RBC US (City National, from 2001Q1), Barclays
US, BNP Paribas USA (BancWest), UBS Americas — followed by the serial
acquirers (Huntington, PNC, Fifth Third, M&T, +48 each). The 95 FDIC-assisted
predecessors put `predecessor_failed = true` on the quarters before each
failure: Colonial in Truist's 2001–2009Q2, the FBOP banks in US Bancorp's,
Silver State, Alliance and Vineyard in Zions'.
