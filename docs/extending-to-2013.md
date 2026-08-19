# Extending the panel back to 2013

Findings from a feasibility pass, recorded before any code changed. Every
number below was measured against live EDGAR data (FY2013 10-K instance
documents, companyfacts, and the XBRL frames API), not recalled from the
taxonomy documentation.

**Verdict.** The pipeline is already date-parameterised; what stops a 2013
start is that `variables.py` and `instance.py` speak only the post-2018,
post-CECL dialect. Teach them the legacy names and the panel fills in.

## What already works unchanged

* Every script takes `--since`; `2020-01-01` is a default and a doc string,
  nothing more. `panel.py` -- quarterisation, fiscal-year differencing,
  restatement flagging -- carries no start-year assumption.
* `filings.list_filings` walks the submissions overflow shards, so 2013
  filings are reachable.
* `instance._find_instance_name` already falls back to the standalone
  (non-inline) instance that pre-2019 filings ship. Replaying that logic
  against five FY2013 10-Ks picked the right file every time
  (`usb-20131231.xml`, `jpm-20131231.xml`, `wfc-20131231.xml`,
  `rf-20131231.xml`, `mtb-20131231.xml`).
* Universe coverage is not the constraint: **29 of 31 banks have all 28
  quarters of 2013-2019**. Citizens starts 2014Q3 and Synchrony 2014Q2 --
  their IPOs, not a gap in the crawler.
* IR is reachable further back than the current window. The Q4 feeds return
  documents from 2007 (RF), 2010 (KEY), 2012 (FITB) and 2014 (USB), and the
  `edgar8k` route is year-agnostic.

## 1. Element names: the pre-CECL dialect

The CECL elements do not exist in a 2013 filing. Counts below are distinct
filers in the universe reporting the tag *undimensioned* per the frames API
(CY2015Q4I for stocks, CY2015 for flows), so they are lower bounds -- the
dimensional path sees more.

| Variable | Add to the tag tuple | Filers | Note |
|---|---|---|---|
| `LOAN_TAGS` | `LoansAndLeasesReceivableNetOfDeferredIncome` | 27 | the dominant element on the loan axes in every 2013 instance opened: JPM 52, USB 64, WFC 60, RF 123, MTB 30 dimensional facts |
| `LOAN_TAGS` | `LoansAndLeasesReceivableGrossCarryingAmount` | 17 | |
| `ACL_TAGS` | `LoansAndLeasesReceivableAllowance` | 38 | M&T's only ACL element in 2013 (76 facts) |
| `NONACCRUAL_TAGS` | `LoansAndLeasesReceivableImpairedNonperformingNonaccrualOfInterest` | 8 | PNC, FITB, MTB, CMA, NTRS, AXP, CFR, SNV |
| `nco` | `AllowanceForLoanAndLeaseLossesWriteoffsNet` | 13 | **net** of recoveries |
| `charge_offs` | `AllowanceForLoanAndLeaseLossesWriteOffs` | 22 | **gross** |
| `charge_offs` | `FinancingReceivableAllowanceForCreditLossesWriteOffs` | 17 | **gross** |

Net versus gross is the one assignment here that fails silently: put a net
write-off element in `charge_offs` and `nco_rate` comes out understated by
the recovery rate with nothing raising.

Recoveries and provision already resolve in the legacy era through tags the
module carries (`FinancingReceivableAllowanceForCreditLossesRecovery`,
`AllowanceForLoanAndLeaseLossesRecoveriesOfBadDebts`,
`ProvisionForLoanAndLeaseLosses`, `ProvisionForLoanLeaseAndOtherLosses`).
Nonaccrual works too: `FinancingReceivableRecordedInvestmentNonaccrualStatus`
is the pre-CECL name and is already first in the list.

**Append, never prepend.** `_restrict_to_best_signature` ranks by tag order
within each bank-quarter, so a legacy name added at the end of a tuple cannot
displace a modern one in a quarter where both exist; the 2013 quarters, which
have only the legacy name, fall through to it. Prepending would rewrite the
2020-2026 panel.

Confirmation that the history is really there: RF's
`LoansAndLeasesReceivableNetOfDeferredIncome` runs 2008-06-30 to 2020 in
companyfacts, and `AllowanceForLoanAndLeaseLossesWriteoffsNet` 2012 to 2024.

## 2. Axis names changed in 2018

`instance.PROMOTED_AXES` misses every pre-2018 spelling. Unaliased, those
facts land in `other_dims`, and the mix, risk-rating and delinquency columns
stay null for exactly the banks that matter.

| Promoted axis | Legacy alias to add | Seen at |
|---|---|---|
| `segment` | `PortfolioSegmentAxis` | JPM FY2013, 1,210 facts |
| `credit_quality` | `CreditQualityIndicatorAxis` | RF FY2013 |
| `past_due` | `FinancingReceivableByDelinquencyStatusAxis` | JPM FY2013 |
| `loan_class` | `AccountsNotesLoansAndFinancingReceivableByReceivableTypeAxis` | WFC FY2013, 318 facts |

Wells Fargo is the reason this is not optional: its FY2013 instance carries no
class-of-financing-receivable axis at all, so without the alias its entire
loan breakdown is invisible.

## 3. Legacy delinquency is element-shaped, not axis-shaped

A 2013 filing does not put delinquency on an axis; it uses a separate element
per bucket, then dimensions *that* by loan class:

| Element | Bucket |
|---|---|
| `FinancingReceivableRecordedInvestment30To59DaysPastDue` | `dpd_30_59` |
| `FinancingReceivableRecordedInvestment60To89DaysPastDue` | `dpd_60_89` |
| `FinancingReceivableRecordedInvestment90DaysPastDueAndStillAccruing` | `dpd_90_plus` |
| `FinancingReceivableRecordedInvestmentEqualToGreaterThan90DaysPastDue` | `dpd_90_plus` |
| `FinancingReceivableRecordedInvestmentCurrent` | `current` |
| `FinancingReceivableRecordedInvestmentPastDue` | `past_due_total` |

`_past_due_columns` keeps **one tag per key** so the buckets stay mutually
exclusive -- correct for the axis shape, fatal for this one, where each bucket
*is* a different tag and all but one would be dropped. This wants a separate
builder merged in behind the axis path, in the same spirit as the HTML and IR
fallbacks, rather than a change to logic that is already verified against the
modern window. Deriving `pd_cat` from the tag in `xbrl.attach_categories` is
safe: the legacy elements are not in `LOAN_TAGS`, so they cannot leak into the
axis-based builder's filter.

## 4. Era-specific scope members

Mapping the loan-axis members of the FY2013 filings through
`taxonomy.loan_category`:

| Bank | Mapped | Unmapped |
|---|---|---|
| JPM | 36 | 9 |
| WFC | 18 | 7 |
| RF | 19 | 4 |
| USB | 20 | 30 |
| MTB | 10 | 10 |

The residue is almost entirely 2013-vintage *scope* qualifiers, not loan
classes: purchased-credit-impaired, FDIC-covered (loss-share), individually
versus collectively evaluated for impairment, acquired versus non-acquired,
allocated versus unallocated. They belong in `SCOPE_MEMBERS` alongside
held-for-sale. Left unmapped they are dropped and `partition_total`
understates; mapped naively as classes they double-count against the members
they sit beside.

`TotalCommercialMember` and `TotalConsumerMember` (RF) are safe rollups to
`commercial_total` / `consumer_total`. `IndirectLoansMember` is better left
unmapped and reported than guessed at -- the module's own rule.

USB's large unmapped count is a different thing again and mostly harmless: it
tags an *industry* cut (agriculture, arts and entertainment, health care,
hotel/motel) that is not a loan class in this schema.

## 5. The CECL break is real and belongs in the data

Incurred loss to CECL at 2020Q1 changes what `acl`, `provision`,
`reserve_coverage` and `reserve_to_nonaccrual` mean, and recorded investment
to amortised cost shifts loan and nonaccrual levels slightly. Splicing the two
regimes into one column without saying so produces a discontinuity that reads
as a credit event.

Add a `basis` column (`incurred` / `cecl`) keyed on a per-bank adoption date:
2020-01-01 for every filer in the universe except Raymond James, whose
September fiscal year puts adoption at 2020-10-01. That date is the standard
large-filer effective date rather than a per-filer confirmation, and should be
documented as such.

## 6. Survivorship, deliberately accepted

The universe is today's DFAST list, so a 2013 start is a survivors-only panel.
The 2013-2019 peer group also held SunTrust, BBVA USA, CIT, SVB, First
Republic, Signature, People's United, E*TRADE, MB Financial and TCF. Adding
them is a separate decision (new `inactive` entries with `last_filing`), not
part of this work.

Two discontinuities inside the *existing* universe are worth stating in the
README rather than discovering in a chart:

* `TFC`'s CIK is BB&T's. Pre-2019 rows are BB&T standalone; the SunTrust
  merger lands in 2019Q4.
* `FCNCA`'s pre-2022 rows exclude CIT, and pre-2023 rows exclude SVB.

## 7. Cost of the run

Roughly 870 additional instance documents -- JPMorgan's FY2013 instance alone
is 27.8 MB -- so budget several GB of cache and hours of fetching at the 6
requests/second the rate limiter allows. IR was deliberately left at 2020:
office CRE is a post-2020 disclosure, the pre-2020 gain is mostly
criticized/classified and small business, and every extraction spec was
verified against modern deck layouts only. Extending IR means re-verifying
those specs against older layouts, which the allowlist principle in
`ir_extract` requires and which is a separate piece of work.

## Suggested order of work

1. `variables.py` -- legacy tags appended, net/gross assigned deliberately.
2. `instance.py` -- the four axis aliases.
3. `taxonomy.py` -- legacy scope members, the two RF rollups.
4. `panel.py` -- legacy past-due builder behind the axis path; `basis` column.
5. Tests in `tests/test_pipeline.py` for each of the four traps above: legacy
   tag ranked behind modern, legacy axis promoted, per-bucket elements
   surviving the one-tag-per-key rule, scope members excluded from a
   partition sum.
6. A crawl from 2013-01-01 and a coverage diff against the current panel;
   nothing pre-2020 should change a single 2020-2026 cell.

---

# What the build actually needed

Written after the work, against the same live data. The plan above held in
outline; six things it did not anticipate are recorded here because each one
would have produced a wrong panel rather than a missing one.

## 1. More legacy axis spellings than four

Sweeping all 29 FY2013 10-K instances rather than five found the pre-2018 cut
under nine names, not four. The extra five are single-bank but not optional --
KeyCorp's entire portfolio-segment breakdown and Huntington's whole delinquency
table hang off them.

| Promoted axis | Legacy alias | Seen at |
|---|---|---|
| `segment` | `FinancingReceivableInformationByPortfolioSegmentAxis` | KEY, 299 facts |
| `credit_quality` | `FinancingReceivableInformationByCreditQualityIndicatorAxis` | KEY, 184 |
| `credit_quality` | `FinancingReceivableRecordedInvestmentByCreditQualityIndicatorAxis` | TFC, 52 |
| `past_due` | `AgingAnalysisOfLoansAndLeasesAxis` | HBAN, 140 |
| `past_due` | `FinancingReceivableInformationByDelinquencyStatusAxis` | NTRS, 100 |

`panel.SUBSLICE_AXES` needed the same additions. It is matched as a substring
of the signature, so the risk was the reverse of the usual one: without the
legacy delinquency and rating axes listed there, a plain loan balance could be
read off a 2013 aging table.

## 2. The recoveries element name in the plan was wrong

The plan asserted recoveries already resolve through
`AllowanceForLoanAndLeaseLossesRecoveriesOfBadDebts`. That element is in the
tag tuple and **no bank in the universe uses it in 2013**. The element they do
use is `AllowanceForLoanAndLeaseLossRecoveryOfBadDebts` -- singular "Loss",
singular "Recovery" -- at 9 of 29 banks. Worth stating because the failure mode
is invisible: `nco` derives from gross charge-offs *minus* recoveries, so a
missing recoveries element silently overstates net charge-offs by the whole
recovery rate.

`FinancingReceivableAllowanceForCreditLossesNetChargeOffs` (2 banks) was added
to `nco` alongside it -- another **net** element, on the same reasoning as
`AllowanceForLoanAndLeaseLossesWriteoffsNet`.

## 3. Which 90+ element to prefer

Both spellings are common (19 banks each) and they do not mean the same thing.
`...EqualToGreaterThan90DaysPastDue` is all 90+ exposure;
`...90DaysPastDueAndStillAccruing` excludes the nonaccrual part. The first is
preferred, because it means what the modern
`FinancialAssetEqualToOrGreaterThan90DaysPastDueMember` means, and a
delinquency series whose definition changes at 2020Q1 is worse than one that
is slightly conservative throughout.

## 4. A legacy axis must not outrank a modern one

The plan's rule for tags -- append, never prepend -- turned out to need an
equivalent for axes, and this was the one omission that actually broke the
2020-2026 window.

Wells Fargo still tags `AccountsNotesLoansAndFinancingReceivableByReceivableTypeAxis`
into the 2020s, and **does not partition its book on it**: residential mortgage
appears there as the parent *and* its first-lien child at once. Promoting the
axis therefore made a new signature available, `select_slice` preferred it on
parsimony (one axis rather than two), and residential mortgage came out at 57%
of a book where it is 29%.

`panel.MODERN_LOAN_AXES` and the `_legacy` ranking term fix this: a signature
built purely from pre-2018 alias axes sorts behind one the modern taxonomy also
describes. It is deliberately *not* applied to the rating and delinquency
builders, whose own axis is not a loan-class axis -- there it would demote the
consolidated delinquency table below the `class x past-due` one.

## 5. Wells Fargo's receivable-type axis is a two-level hierarchy

Even with the ranking fixed, the axis is the only loan cut WFC's 10-Qs carry,
and its members are two rollups plus ten leaves. Three of the leaves are named
with members that read as rollups anywhere else, so they were being summed
*beside* the rollup they belong to and inflating commercial loans by half.

The FY2013 arithmetic settles all three, exactly:

| | |
|---|---|
| `CommercialLoanMember` 197.2 + CRE mortgage 107.1 + construction 16.7 + lease 12.0 + `ForeignLoansMember` 47.7 | = **380.8bn**, the commercial rollup |
| first lien 258.5 + junior lien 65.9 + card 26.9 + auto 50.8 + `ConsumerLoanMember` 43.0 | = **445.0bn**, the consumer rollup |
| 380.8 + 445.0 | = **825.8bn**, Wells Fargo's reported FY2013 total loans |

So `CommercialLoanMember` is C&I, `ConsumerLoanMember` is other revolving
credit and installment, and `ForeignLoansMember` is a commercial class -- three
`taxonomy.OVERRIDES` entries.

The same axis also carries `ResidentialMortgageMember` (the parent) beside
`FirstMortgageMember` (its first-lien child), and both map to `resi_mortgage`,
so the 10-Q quarters double-counted the first lien. A plain `(ticker, member)`
override cannot express the fix, because Wells Fargo's *modern* class axis uses
the same member for the *correct* figure. See section 9.

## 6. The fair-value note outranks the loan note at Wells Fargo

Promoting the receivable-type axis also exposed a latent tag-ranking hazard.
WFC tags its portfolio table with `NotesReceivableGross` (third in
`LOAN_TAGS`) and its fair-value-of-financial-instruments note with
`LoansAndLeasesReceivableNetReportedAmount` (second). Tag rank is the first
sort term, so the note won and commercial loans came out at $15bn against a
$500bn book.

The fix belongs where the concept does: `EstimateOfFairValueFairValueDisclosureMember`
and `CarryingReportedAmountFairValueDisclosureMember` are measurement bases,
not loan classes, and are now `SCOPE_MEMBERS`. They appear at 18 banks.

## What it cost the 2020-2026 window

The plan asked for zero change there. That was not achievable and, on
inspection, not desirable: every legacy name added is a name some bank still
uses. Measured on a like-for-like rebuild (XBRL only, no fallbacks):

| | cells |
|---|---|
| newly populated | 6,162 |
| changed | 516 |
| no longer populated | 147 |

Quality flags fell from 363 bank-quarters to 330. Of the 167 changed
`mix_coverage_pct` values, 121 moved *towards* 100 and 46 away from it -- but
that ratio is the wrong summary, for the reason section 7 gives: some of the
46 moved away because a fabricated 100 was removed. The changes were traced
individually instead:

* **`acl_total` at AXP, CFG and TFC** now comes from a directly reported
  `LoansAndLeasesReceivableAllowance` instead of a partition sum that was
  missing a category. Truist Q2 2024 goes from 4.401bn to 4.808bn -- and
  4.401 + 0.407 (the credit-card allowance the sum omitted) = 4.808.
* **Goldman Sachs** loses `loans_consumer_total` and gains `loans_credit_card`:
  its facts carry a segment *and* a receivable-type member, and the class-level
  member is the more specific reading. `mix_coverage_pct` rises 82 -> 97.
* **Wells Fargo** gains a mix in the 10-Q quarters it previously had none for,
  and `cre_pct`/`ci_pct` for its 10-Ks. See section 5 for what is still wrong
  there.
* **The 147 lost cells** are `*_commercial_total`, `*_consumer_total` and
  `*_lease` at WFC, GS, CFG and MTB, in every case because a rollup column was
  replaced by the leaf columns underneath it, or because a member that was
  never the lease line stopped being counted as one.

## What the 2013 half looks like

A FY2013-only build over all 29 filers, before the full crawl: `loans_total`
and `acl_total` populated for 108 and 113 of 116 bank-quarters, `nonaccrual`
82, `nco` 76, and the legacy delinquency builder filling `pd_dpd_90_plus` for
78. The receivable-type axis is load-bearing exactly as the plan said -- without
it Wells Fargo and Huntington have **no** 2013 loan mix at all, and Discover
and PNC collapse to under 7% coverage.

## 7. The lease rule, and a metric that was lying

Promoting the receivable-type axis put a new population of members in front of
`taxonomy.loan_category` for the first time — members that had previously sat
in `other_dims` and were never mapped at all. That exposed the lease rule,
which required `Member` to follow the lease word directly and so missed every
`...LeaseFinancingReceivableMember` spelling: 707 sightings across WFC, FCNCA,
ZION, NTRS, STT and KEY went unmapped, and Wells Fargo's
`CommercialLeaseFinancingReceivableMember` fell through to the `^Commercial`
rollup and was summed into `commercial_total` beside it.

Widening the rule needed one guard, and getting it wrong was instructive.
**"Loans *and* leases" names a rollup; "lease financing" alone names the
class.** Without that distinction the rule swallowed US Bancorp's
`CommercialLoanAndLeaseFinancingLoanMember` — its entire $139bn commercial
book — and reported it as a lease line that is really about $4bn. It showed up
as USB's mix coverage moving from 116% to 154%, which is the only reason it was
caught before the rebuild.

The same guard turned up two **pre-existing** faults that had nothing to do
with 2013:

| Bank | Member read as `lease` | Value | Real lease book | `mix_coverage_pct` |
|---|---|---|---|---|
| CFG | `TotalLoansandLeasesMember` | 93.4bn | — | **101.3** |
| MTB | `CommercialLoansAndLeasesMember` | 27.4bn | ~2bn | **100.6** |

Citizens' entire loan book was being counted as its lease line. Both banks
scored near-perfect coherence *because* of the error: when one member absorbs
the whole book, the leaves sum to the total by construction. Corrected, they
read 0 and 72 — which is worse-looking and correct, since the rest of those
breakdowns genuinely is not mapped.

The lesson is about the metric, not the rule. `mix_coverage_pct` near 100 is
necessary but not sufficient, and a distance-from-100 score is the wrong way to
judge a taxonomy change: it rewards exactly the failure that fabricates
coherence. Both banks now sit in `panel_unmapped_members.csv` instead, which is
where an unread disclosure belongs.

## 8. A modern element name used for a pre-modern concept

The last defect the back-extension exposed, and the one that most nearly
shipped. Tag priority rests on an assumption that is never stated: that every
candidate element in a tuple measures *the same concept*, so ranking picks the
best spelling of one number. Four filers break it before 2020 by tagging a
modern-*named* element with something far narrower.

| Bank | Top-ranked tag holds | Actual allowance, on the legacy tag |
|---|---|---|
| Citigroup FY2013 | $113m | **$19,648m** |
| Capital One FY2013 | $38m | **$4,315m** |
| Zions FY2016 | ~nil | **$567m** |

The reported reserve ratio for Citigroup came out at 0.02%.

Three fixes were tried; the order matters because the first two look right.

1. **Reorder `ACL_TAGS`.** Rejected without testing: it rewrites the 2020-2026
   window, which is the one thing the whole design of this change avoids.
2. **Rebuild the total from its parts** when the reported total is an order of
   magnitude below the partition sum. Implemented, measured, and *backed out*:
   at Citigroup the partition double-counts, so it replaced a 200x-too-small
   number with a 2x-too-large one ($39.3bn), and at Zions it produced 0.0.
   Partition sums are not trustworthy enough to overrule a reported figure --
   see section 5 for why.
3. **Drop the fragment before ranking.** The bank reported the right number; it
   simply is not the highest-ranked tag. Discarding undimensioned candidates an
   order of magnitude below the largest one leaves priority to choose among
   what remains.

The third is what shipped, scoped to loan-sliced **stocks**. Both halves of
that scope are load-bearing and were found by measurement, not by reasoning:

* `unfunded_commitments` ranks a small reserve element above a large notional
  one *deliberately*, so magnitude must not decide there.
* a **flow** is legitimately small, zero, or negative in a quarter. The
  unscoped guard changed exactly one cell in 2020-2026 -- Raymond James'
  nil provision, dropped in favour of a $10m release -- which is how the flow
  half of the scope was found.

Scoped, it changes **97 cells before 2020 and none after it**. Implausible
reserve ratios across the panel fall from 71 bank-quarters to 48, and all 48
that remain are real: Schwab's pledged-asset lines (37) and State Street's
institutional book (7) genuinely carry reserves near a basis point.

### Why not the HTML fallback

Recovering these from the 10-K table was considered and does not work, for
three independent reasons. `reconcile.fill_gaps` writes only into **null**
cells, and $113m is not null, so the value would never land; the HTML path is
allowlisted to `loans_office_cre` alone; and its row-label patterns and scale
inference were verified against post-2020 layouts only, which is the same
reason `--html-since` exists. More basically, it is the wrong direction: using
a parsed table to overrule a tagged fact inverts the source precedence the
reconciliation layer is built on. The fix that shipped stays inside the same
instance document and changes only which of the filer's own tagged totals is
read.

## 9. A member that is a class on one axis and a rollup on another

The residual from section 5. Wells Fargo tags residential mortgage three ways
at once on the pre-2018 receivable-type axis -- the parent and both liens --
and the arithmetic is exact at 2021Q4: 258.89 = 242.27 + 16.62. Reading the
parent as a class *beside its own child* put residential at 57% of a book where
it is 29%.

Overrides key on `(ticker, member)`, and that is not enough here: the **same
member, at the same bank, in the same filing** is the correct reading where it
is tagged alone. The first attempt made the override axis-conditional, which
fixed the 10-Q quarters and left a 3% residual in the 10-K ones, where the
parent comes off the modern axis and the junior lien off the legacy one, so the
junior lien sat inside both `resi_mortgage` and `home_equity`.

The general rule is simpler than the axis one and closes both: **a rollup is
only a rollup when its parts are there.** `taxonomy.SUPERSEDED_BY` names the
components that supersede a member, and `attach_categories` decides per
bank-quarter from the members actually present -- which is the only place the
question can be answered, since the same member is a double count in one
quarter and the only figure published in the next.

Two scoping details, both found by measurement:

* **Balances only.** Wells Fargo reports residential charge-offs and recoveries
  on the parent alone, so suppressing it for flows dropped three quarters of
  `nco_resi_mortgage` for nothing. The overlap being prevented is a
  balance-sheet one -- the liens partition the parent's *balance*.
* **Components must be present.** In the 15 recent quarters Wells Fargo stopped
  tagging the split, and there the parent is the only residential figure there
  is.

The safety condition was checked across all 58 quarters rather than assumed:

| | quarters |
|---|---|
| rollup on the receivable-type axis, children also tagged | 15 |
| rollup on the modern axis only, no children | 15 |
| **rollup on the receivable-type axis with no children** | **0** |

That last row is what makes suppression safe: Wells Fargo never puts the parent
on that axis alone, so removing it can never leave the bank with no residential
figure at all. Measured across the three regimes:

| | before | after |
|---|---|---|
| 2021 10-Q quarters, `resi_pct` | 57% | **29%** |
| 2021 10-Q quarters, `mix_coverage_pct` | 128-130 | **99** |
| 2021 10-K quarter, `resi_mortgage` | 258.89 (incl. junior lien) | **242.27** |
| 2021 10-K quarter, `mix_coverage_pct` | 103.3 | **101.4** |
| 2013 quarters | 100.0 | unchanged |
| 2025-26, rollup-only quarters | parent kept | unchanged |

`resi_pct` across 2021Q1-2022Q2 now runs 30.1, 29.2, 28.6, 27.4, 27.2, 27.1 --
a series rather than a saw-tooth, because every quarter is now the first-lien
figure rather than alternating between that and the parent.

## 11. The coherence metric could go stale

Found while checking the above. `mix_coverage_pct` read a flawless **100.0**
for Wells Fargo in 2025-26 where the true figure was 101.7.

It is computed in `panel.add_derived`, which runs on the XBRL frame -- before
the HTML and IR fallbacks have filled anything. `reconcile.refresh_ratios`
recomputes the `RatioDef`s afterwards, but `mix_coverage_pct` is not a
`RatioDef`, so nothing recomputed it. Any fallback that filled a loan category
changed the true coverage and left the old number in place.

97 bank-quarters across twelve banks were carrying a stale score, 14 of them
reading exactly 100.0. That is the third time in this work that the number
meant to signal "safe to rank on" was the one that lied, so it is now
recomputed outright rather than gap-filled -- a stale coherence score is worse
than a missing one -- and `refresh_ratios` runs after every fallback rather
than only after the IR one, because the HTML path fills columns too.

Purchased-credit-impaired needed one more spelling in the same pass: Truist
writes `PCIMember` in capitals, which the `Pci` alternation missed. The
`(?!.*Excluding)` guard from section 4 already covers the two members that must
*not* be caught -- `ConsumerPortfolioSegmentExcludingPCIMember` is the main
consumer book, not the PCI cut.

## 10. Where the build time goes

Measured rather than guessed, because the answer was not where it looked:

| Stage | Before | Bound by |
|---|---|---|
| XBRL extraction, 1,666 instances | 3.7 min | lxml parsing, one core |
| HTML fallback, 210 filings | 4.6 min | **re-downloading**, not parsing |
| IR extraction, 1,004 cached documents | 3.3 min | PDF/HTML parsing, one core |

The HTML row was the surprise. `config.RAW_HTML` is declared, `ensure_dirs`
creates it, and **nothing ever wrote to it** -- so every build re-fetched all
210 primary documents through the rate limiter. Wiring that cache up, on the
pattern `instance.fetch_instance` already used, is most of that stage.

The other two are pure CPU over independent documents on a 24-core machine, so
they go to a pool (`parallel.map_frames`). Three things make that safe rather
than merely fast:

* **Fetching stays in the parent.** The rate limiter in `edgar` spaces requests
  within one process; a pool of workers each holding their own would multiply
  the request rate by the worker count, straight through the SEC's fair-access
  limit. Callers warm the cache serially through `ensure_cached`, and workers
  only ever read from disk.
* **One pool, not one per call.** `dimensional_long` runs per bank, so a pool
  per call is 31 pools in a universe run -- and Windows *spawns* workers rather
  than forking, so each one re-imports polars and lxml. Sharing a single pool
  took extraction from 128s to 91s on its own.
* **Order is preserved.** `Executor.map` yields results in *input* order, so the
  concatenated frame is identical to the serial one rather than merely
  equivalent. A panel that reshuffled between builds would make every diff
  unreadable.

That last point was verified rather than argued: building the whole panel
serially (`BANKQTR_WORKERS=1`) and in parallel gives byte-identical output --
the same 339,180 facts and the same 1,648 x 456 panel, `DataFrame.equals` true
for both.

| | before | after |
|---|---|---|
| XBRL extraction | 216s | **91s** |
| Full build, warm cache | ~15 min | **222s** |

`BANKQTR_WORKERS=1` restores the serial path, which is the first thing to reach
for if a parallel build is ever suspected of disagreeing with a serial one.
