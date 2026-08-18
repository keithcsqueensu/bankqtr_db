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
