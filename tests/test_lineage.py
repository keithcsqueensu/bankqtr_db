"""Tests for the predecessor lineage and the 2001 window.

The unit tests build a small NIC -- a handful of entities, relationships and
successions -- by monkeypatching the module-level loaders in ``nic``, so they
need neither the bulk structure files nor a cached quarter.  Where a test
needs a Call Report roster it writes one with ``make_zip`` from
``test_callrpt`` and points ``cdr.zip_path`` at it.

The integration tests at the end run against the real NIC files and cached
quarters, and are skipped without them.  They pin the cases that motivated
the lineage: Colonial Bank failing into Truist, Washington Mutual into
JPMorgan Chase Bank, Wachovia into Wells Fargo, Countrywide into Bank of
America by acquisition rather than merger.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from callrpt_db import cdr, config, lineage, mdrm, nic, panel, schedules
from tests.test_callrpt import make_zip, resolve, value

SPEC = mdrm.BY_NAME

# --------------------------------------------------------------------------
# A small NIC
# --------------------------------------------------------------------------
#
#   H  (100)  tracked holding, owns bank B (10) throughout
#   QH (300)  holding with bank Q (30); QH merged into H on 2004-07-01 (code 1),
#             Q then stays a charter under H and is merged into B on
#             2006-01-01 under common control (code 1, pooled)
#   P  (20)   a standalone bank, failed 2008-09-26 and sold to B (code 50)
#   DH (400)  a second *tracked* holding with bank D (40); DH merged into H on
#             2025-05-18 (code 1) -- the Discover case
#   A  (500)  an independent holding with bank AB (50), acquired by H on
#             2008-07-01 by relationship only; no transformation ever
#   N  (600)  a nonbank subsidiary H formed in 2010 (never a predecessor)


def _entity(rssd: int, name: str, kind: str, *, fdic: int | None = None, opened: str = "19900101",
            ended: str | None = None, term: str = "0", bhc: bool = False) -> nic.Entity:
    return nic.Entity(
        rssd=rssd, name=name, entity_type=kind, ein=None, lei=None, fdic_cert=fdic,
        is_bhc=bhc, is_ihc=False, active=ended is None, opened=opened, ended=ended,
        reason_term=term,
    )


ENTITIES = {
    100: _entity(100, "H CORP", "FHD", bhc=True),
    10: _entity(10, "H BANK", "NAT", fdic=1),
    300: _entity(300, "QH CORP", "BHC", bhc=True, ended="20040701", term="2"),
    30: _entity(30, "Q BANK", "SMB", fdic=3, ended="20060101", term="2"),
    20: _entity(20, "P BANK", "NMB", fdic=2, ended="20080926", term="5"),
    400: _entity(400, "DH CORP", "FHD", bhc=True, ended="20250518", term="2"),
    40: _entity(40, "D BANK", "NMB", fdic=4),
    500: _entity(500, "A CORP", "BHC", bhc=True),
    50: _entity(50, "A BANK", "NMB", fdic=5),
    600: _entity(600, "N LLC", "DEO", opened="20100101"),
}

# (parent, child, start, end)
RELATIONSHIPS = [
    (100, 10, "19900101", nic.OPEN_END),
    (300, 30, "19900101", "20040701"),
    (100, 30, "20040701", "20060101"),
    (400, 40, "19900101", "20250518"),
    (100, 40, "20250518", nic.OPEN_END),
    (500, 50, "19900101", nic.OPEN_END),
    (100, 500, "20080701", nic.OPEN_END),
    (100, 600, "20100101", nic.OPEN_END),
]

SUCCESSIONS = (
    nic.Succession(300, 100, "20040701", "1", False),
    nic.Succession(30, 10, "20060101", "1", True),
    nic.Succession(20, 10, "20080926", "50", False),
    nic.Succession(400, 100, "20250518", "1", False),
)

H = config.Holding("H Corp", 100, "HHH")
DH = config.Holding("DH Corp", 400, "DDD", last_filing=dt.date(2025, 5, 18))


@pytest.fixture
def small_nic(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(nic, "entities", lambda: ENTITIES)
    monkeypatch.setattr(nic, "_relationships", lambda: RELATIONSHIPS)
    monkeypatch.setattr(nic, "successions", lambda: SUCCESSIONS)
    by_succ: dict[int, list[nic.Succession]] = {}
    for s in SUCCESSIONS:
        by_succ.setdefault(s.successor, []).append(s)
    monkeypatch.setattr(nic, "by_successor", lambda: {k: tuple(v) for k, v in by_succ.items()})
    monkeypatch.setattr(nic, "failed_rssds", lambda: frozenset({20}))
    parents: dict[int, list[tuple[int, str, str]]] = {}
    for p, c, s, e in RELATIONSHIPS:
        parents.setdefault(c, []).append((p, s, e))
    monkeypatch.setattr(nic, "control_parents", lambda: {k: tuple(v) for k, v in parents.items()})
    nic._hierarchy_at.cache_clear()
    yield
    nic._hierarchy_at.cache_clear()


QUARTERS = [cdr.quarter_end(f"{y}Q{q}") for y in range(2001, 2027) for q in (1, 2, 3, 4)]


# --------------------------------------------------------------------------
# Discovery and classification
# --------------------------------------------------------------------------


def test_predecessor_absorbed_by_a_subsidiary_is_found(small_nic) -> None:
    """Washington Mutual was sold to JPMorgan Chase *Bank*, not to the holding
    company.  A walk over the top entity's own transformations never sees it;
    the walk over every member does."""
    lin = lineage.resolve(H, QUARTERS)
    assert 20 in lin.predecessors
    assert lin.predecessors[20].successor == 10


def test_fdic_assisted_is_flagged_and_dated(small_nic) -> None:
    lin = lineage.resolve(H, QUARTERS)
    p = lin.predecessors[20]
    assert p.succession_type == lineage.FDIC_ASSISTED
    assert p.effective_to == dt.date(2008, 9, 26)
    assert p.active_on(dt.date(2008, 6, 30))
    assert not p.active_on(dt.date(2008, 9, 30))


def test_merger_and_reorg_are_told_apart(small_nic) -> None:
    """QH was an outside organisation when it merged into H; Q was already
    inside when it was folded into B two years later."""
    lin = lineage.resolve(H, QUARTERS)
    assert lin.predecessors[300].succession_type == lineage.MERGER
    assert lin.predecessors[30].succession_type == lineage.REORG
    assert lin.predecessors[30].pooled


def test_acquisition_by_relationship_is_a_predecessor(small_nic) -> None:
    """Countrywide, Merrill Lynch and Bear Stearns were bought and kept as
    subsidiaries.  NIC records no transformation, only a relationship that
    starts on the closing date."""
    lin = lineage.resolve(H, QUARTERS)
    assert 500 in lin.predecessors
    p = lin.predecessors[500]
    assert p.code == lineage.ACQUISITION_CODE
    assert p.succession_type == lineage.MERGER
    assert p.effective_to == dt.date(2008, 7, 1)


def test_a_subsidiary_formed_inside_is_not_a_predecessor(small_nic) -> None:
    lin = lineage.resolve(H, QUARTERS)
    assert 600 not in lin.predecessors


def test_the_lineage_file_is_complete_over_the_universe(small_nic) -> None:
    """Every firm gets a row, predecessors or not."""
    lins = lineage.resolve_all((H, DH), QUARTERS)
    frame = lineage.to_frame(lins, tracked={100, 400})
    assert set(frame["bhc_rssd_2026"].to_list()) == {100, 400}
    selves = frame.filter(pl.col("succession_type") == lineage.SELF)
    assert set(selves["predecessor_rssd"].to_list()) == {100, 400}
    # DH is tracked in its own right, and the file says so on H's row for it.
    row = frame.filter((pl.col("ticker") == "HHH") & (pl.col("predecessor_rssd") == 400))
    assert row["tracked_separately"][0]
    assert set(frame.columns) >= {
        "bhc_name", "bhc_rssd_2026", "predecessor_rssd", "predecessor_name",
        "effective_from", "effective_to", "succession_type",
    }


# --------------------------------------------------------------------------
# Claiming charters: no over-counting
# --------------------------------------------------------------------------


def _mapping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, periods: list[str], filers: list[int]):
    for period in periods:
        make_zip(tmp_path, period, {}, forms={r: "041" for r in filers})
    monkeypatch.setattr(cdr, "zip_path", lambda period: tmp_path / f"call_{period}.zip")
    lins = lineage.resolve_all((H, DH), QUARTERS)
    return panel.universe_filers(periods, (H, DH), lins)


def test_predecessor_and_successor_in_the_same_quarter_are_not_both_summed(
    small_nic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In 2004Q3 QH has just merged into H and Q sits under H in the dated
    graph; QH is still a live root for the quarter end (inclusive).  Both
    paths reach Q and it must be claimed once."""
    mapping = _mapping(tmp_path, monkeypatch, ["2004Q2", "2004Q3"], [10, 30])
    per = mapping.group_by(["rssd", "period"]).len()
    assert per.filter(pl.col("len") > 1).is_empty()
    q2 = mapping.filter(pl.col("period") == dt.date(2004, 6, 30))
    assert set(q2["rssd"].to_list()) == {10, 30}
    # Before the merger Q is predecessor history; after it, H's own.
    assert q2.filter(pl.col("rssd") == 30)["via_rssd"][0] == 300
    q3 = mapping.filter((pl.col("period") == dt.date(2004, 9, 30)) & (pl.col("rssd") == 30))
    assert q3.height == 1


def test_a_firm_tracked_in_its_own_right_is_not_also_its_acquirers_history(
    small_nic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Discover Bank belongs to Discover's row until Discover stops being a
    row, and to Capital One's only after."""
    mapping = _mapping(tmp_path, monkeypatch, ["2024Q4", "2025Q2"], [10, 40])
    before = mapping.filter((pl.col("period") == dt.date(2024, 12, 31)) & (pl.col("rssd") == 40))
    assert before["ticker"].to_list() == ["DDD"]
    after = mapping.filter((pl.col("period") == dt.date(2025, 6, 30)) & (pl.col("rssd") == 40))
    assert after["ticker"].to_list() == ["HHH"]
    assert after["via_rssd"][0] is None


def test_failed_predecessor_flags_the_quarters_it_is_summed_into(
    small_nic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapping = _mapping(tmp_path, monkeypatch, ["2008Q2", "2008Q4"], [10, 20])
    charters = pl.DataFrame(
        {
            "rssd": [10, 20, 10],
            "period": [dt.date(2008, 6, 30), dt.date(2008, 6, 30), dt.date(2008, 12, 31)],
            "loans_total": [100.0, 50.0, 150.0],
        }
    )
    rolled = panel.roll_up(charters, mapping).sort("period")
    assert rolled["predecessor_failed"].to_list() == [True, False]
    assert rolled["has_predecessor"].to_list() == [True, False]
    assert rolled["predecessor_count"].to_list() == [1, 0]
    assert rolled["predecessors"].to_list() == ["20", ""]
    assert rolled["loans_total"].to_list() == [150.0, 150.0]


# --------------------------------------------------------------------------
# The rollup and the flows
# --------------------------------------------------------------------------


def test_all_null_charters_roll_up_to_null_not_zero() -> None:
    """CET1 read 0.0 for every bank before 2015 because the sum of nothing
    is zero.  It is null."""
    mapping = pl.DataFrame(
        {"ticker": ["X", "X"], "bank": ["X", "X"], "holding_rssd": [1, 1], "rssd": [10, 11],
         "period": [dt.date(2013, 3, 31)] * 2}
    )
    charters = pl.DataFrame(
        {"rssd": [10, 11], "period": [dt.date(2013, 3, 31)] * 2,
         "cet1_capital": [None, None], "loans_total": [1.0, None]},
        schema={"rssd": pl.Int64, "period": pl.Date, "cet1_capital": pl.Float64, "loans_total": pl.Float64},
    )
    rolled = panel.roll_up(charters, mapping)
    assert rolled["cet1_capital"][0] is None
    assert rolled["loans_total"][0] == 1.0


def test_pooling_merger_does_not_double_count_the_absorbed_year_to_date() -> None:
    """Q was merged into B on 2006-01-01... make it a Q3 merger: B's Q3 YTD
    is restated to include Q's January-June, which Q already filed."""
    frame = pl.DataFrame(
        {
            "rssd": [10, 10, 10, 30, 30],
            "period": [dt.date(2006, 3, 31), dt.date(2006, 6, 30), dt.date(2006, 9, 30),
                       dt.date(2006, 3, 31), dt.date(2006, 6, 30)],
            # B alone: 10 a quarter.  Q alone: 5 a quarter.  B's Q3 YTD of 40
            # = 30 (own, three quarters) + 10 (Q's first half, restated in).
            "charge_offs_total": [10.0, 20.0, 40.0, 5.0, 10.0],
        }
    )
    pooled = pl.DataFrame(
        {"rssd": [10], "predecessor": [30], "period": [dt.date(2006, 9, 30)]},
        schema={"rssd": pl.Int64, "predecessor": pl.Int64, "period": pl.Date},
    )
    out = panel.quarterize(frame, pooled).sort(["rssd", "period"])
    b = out.filter(pl.col("rssd") == 10)["charge_offs_total"].to_list()
    assert b == [10.0, 10.0, 10.0]
    q = out.filter(pl.col("rssd") == 30)["charge_offs_total"].to_list()
    assert q == [5.0, 5.0]


def test_rcn_totals_are_built_before_2017_and_reported_after() -> None:
    cats = mdrm.RCN_TOTAL_COMPONENTS["nonaccrual_total"]
    data = {c: [1.0, 1.0] for c in cats}
    data["nonaccrual_total"] = [None, 99.0]
    data["pd_dpd_30_89"] = [None, None]
    data["pd_dpd_90_plus"] = [None, None]
    frame = pl.DataFrame(data, schema={k: pl.Float64 for k in data})
    out = panel.add_rcn_totals(frame)
    assert out["nonaccrual_total"].to_list() == [float(len(cats)), 99.0]
    # Built only where the form reported nothing; the second row is the form's own.
    assert out["rcn_total_built"].to_list() == [True, False]


def test_nonaccrual_cre_reads_the_nonaccrual_column(tmp_path: Path) -> None:
    """F180/F181 are 90 days past due; F182/F183 are nonaccrual."""
    path = make_zip(
        tmp_path, "2025Q4",
        {"RCN": {1: {"RCONF180": "5", "RCONF182": "700", "RCONF181": "6", "RCONF183": "800"}}},
    )
    frame = resolve(path, (SPEC["nonaccrual_cre_owner_occupied"], SPEC["nonaccrual_cre_investor"],
                           SPEC["dpd_90_plus_cre_owner_occupied"]))
    assert value(frame, 1, "nonaccrual_cre_owner_occupied") == 700_000
    assert value(frame, 1, "nonaccrual_cre_investor") == 800_000
    assert value(frame, 1, "dpd_90_plus_cre_owner_occupied") == 5_000


# --------------------------------------------------------------------------
# The old forms
# --------------------------------------------------------------------------


def test_pre_2007_construction_and_nonfarm_nonres_are_read(tmp_path: Path) -> None:
    path = make_zip(
        tmp_path, "2005Q4",
        {"RCCI": {1: {"RCON1415": "100", "RCON1480": "200"}}},
        forms={1: "041"},
    )
    frame = resolve(path, (SPEC["loans_construction"], SPEC["loans_cre_nonfarm_nonres"],
                           SPEC["loans_cre_owner_occupied"]))
    assert value(frame, 1, "loans_construction") == 100_000
    assert value(frame, 1, "loans_cre_nonfarm_nonres") == 200_000
    assert value(frame, 1, "loans_cre_owner_occupied") is None


def test_2007_derived_total_beside_the_detail_is_not_double_counted(tmp_path: Path) -> None:
    """Through 2007 the old code is carried as a derived total equal to the
    new detail, for 4,722 filers at once."""
    path = make_zip(
        tmp_path, "2007Q2",
        {"RCCI": {1: {"RCON1415": "300", "RCONF158": "100", "RCONF159": "200"}}},
        forms={1: "041"},
    )
    frame = resolve(path, (SPEC["loans_construction"],))
    assert value(frame, 1, "loans_construction") == 300_000


def test_pre_2011_consumer_installment_covers_auto(tmp_path: Path) -> None:
    path = make_zip(
        tmp_path, "2009Q4",
        {"RCCI": {1: {"RCON2011": "500", "RCONB539": "20"}}},
        forms={1: "041"},
    )
    frame = resolve(path, (SPEC["loans_consumer_installment"], SPEC["loans_auto"],
                           SPEC["loans_consumer_revolving_other"]))
    assert value(frame, 1, "loans_consumer_installment") == 500_000
    assert value(frame, 1, "loans_auto") is None
    assert value(frame, 1, "loans_consumer_revolving_other") == 20_000


def test_form_041_ci_charge_offs_are_read_from_the_single_line(tmp_path: Path) -> None:
    path = make_zip(tmp_path, "2009Q4", {"RIBI": {1: {"RIAD4638": "9"}}}, forms={1: "041"})
    frame = resolve(path, (SPEC["charge_offs_ci"],))
    assert value(frame, 1, "charge_offs_ci") == 9_000


def test_tdr_total_prefers_the_form_total_over_the_lines_it_sums(tmp_path: Path) -> None:
    """HK25 is the total from 2017; the category lines it totals still exist,
    and completeness would pick them over it."""
    path = make_zip(
        tmp_path, "2020Q1",
        {"RCCI": {1: {"RCONHK25": "100", "RCONK158": "10", "RCONK165": "20", "RCONK256": "70"}}},
        forms={1: "041"},
    )
    frame = resolve(path, (SPEC["loans_tdr_accruing"],))
    assert value(frame, 1, "loans_tdr_accruing") == 100_000


def test_goodwill_is_read_from_either_schedule_but_never_both(tmp_path: Path) -> None:
    path = make_zip(
        tmp_path, "2018Q1",
        {"RC": {1: {"RCON3163": "5"}}, "RCM": {1: {"RCON3163": "5"}}},
        forms={1: "041"},
    )
    frame = resolve(path, (SPEC["goodwill"],))
    assert value(frame, 1, "goodwill") == 5_000


def test_legacy_capital_stays_apart_from_basel_iii(tmp_path: Path) -> None:
    path = make_zip(
        tmp_path, "2008Q4",
        {"RCR": {1: {"RCON8274": "80", "RCONA223": "1000"}}},
        forms={1: "041"},
    )
    frame = resolve(path, (SPEC["tier1_capital"], SPEC["tier1_capital_basel1"], SPEC["risk_weighted_assets_basel1"]))
    assert value(frame, 1, "tier1_capital") is None
    assert value(frame, 1, "tier1_capital_basel1") == 80_000
    rolled = panel.add_capital_ratios(frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("risk_weighted_assets")))
    assert rolled["tier1_ratio_basel1"][0] == pytest.approx(8.0)
    assert "tier1_ratio" not in rolled.columns or rolled["tier1_ratio"][0] is None


# --------------------------------------------------------------------------
# Integration -- skipped without the NIC files and cached quarters
# --------------------------------------------------------------------------


def _have_nic() -> bool:
    return all((config.RAW_NIC / f"{k}.zip").exists() for k in config.NPW_FILES)


needs_nic = pytest.mark.skipif(not _have_nic(), reason="NIC structure files not cached")
needs_cache = pytest.mark.skipif(not cdr.cached_periods(), reason="no cached Call Report period")


@pytest.fixture(scope="module")
def real_lineages():
    if not _have_nic():
        pytest.skip("NIC structure files not cached")
    periods = cdr.cached_periods()
    quarters = [cdr.quarter_end(p) for p in periods] or QUARTERS
    # Every filer in the window: how the walk recognises a depository that
    # NIC's attribute files omit (Discover Bank, Countrywide's banks).
    filers: set[int] = set()
    for period in periods:
        filers.update(schedules.roster(cdr.zip_path(period))["rssd"].to_list())
    holdings = tuple(h for h in config.HOLDINGS if h.tier in ("dfast", "ihc"))
    return holdings, lineage.resolve_all(holdings, quarters, filers=filers)


@needs_nic
def test_every_dfast_bank_has_a_lineage_entry(real_lineages) -> None:
    holdings, lins = real_lineages
    frame = lineage.to_frame(lins, tracked={r for h in holdings for r in h.rssds})
    assert set(frame["bhc_rssd_2026"].to_list()) == {h.rssd for h in holdings}
    assert set(frame["ticker"].to_list()) == {h.ticker for h in holdings}


@needs_nic
def test_fdic_assisted_acquisitions_are_flagged(real_lineages) -> None:
    _, lins = real_lineages
    assert lins["TFC"].predecessors[570231].succession_type == lineage.FDIC_ASSISTED  # Colonial Bank
    assert lins["JPM"].predecessors[1222108].succession_type == lineage.FDIC_ASSISTED  # Washington Mutual Bank
    assert lins["JPM"].predecessors[4114567].succession_type == lineage.FDIC_ASSISTED  # First Republic Bank
    assert lins["USB"].predecessors[15536].succession_type == lineage.FDIC_ASSISTED  # Park National Bank
    assert lins["RF"].predecessors[2922339].succession_type == lineage.FDIC_ASSISTED  # Integrity Bank


@needs_nic
def test_the_famous_mergers_are_in_the_lineage(real_lineages) -> None:
    _, lins = real_lineages
    assert lins["WFC"].predecessors[1073551].succession_type == lineage.MERGER  # Wachovia Corporation
    assert lins["WFC"].predecessors[1079441].successor == 1073551  # SouthTrust, into Wachovia
    assert lins["PNC"].predecessors[1069125].effective_to == dt.date(2008, 12, 31)  # National City
    assert lins["TFC"].predecessors[1131787].succession_type == lineage.MERGER  # SunTrust Banks
    assert lins["TFC"].predecessors[675332].succession_type == lineage.MERGER  # SunTrust Bank, same day
    assert lins["JPM"].predecessors[1068294].succession_type == lineage.MERGER  # Bank One
    assert lins["COF"].predecessors[3846375].succession_type == lineage.MERGER  # Discover


@needs_nic
@needs_cache
def test_acquisitions_kept_as_subsidiaries_are_found(real_lineages) -> None:
    _, lins = real_lineages
    bac = lins["BAC"].predecessors
    assert 2549857 in bac and bac[2549857].code == lineage.ACQUISITION_CODE  # Countrywide Financial
    assert 1246140 in bac  # Merrill Lynch & Co.
    assert 1573257 in lins["JPM"].predecessors  # Bear Stearns Companies


@needs_cache
def test_rcn_categories_reproduce_the_forms_total() -> None:
    """From 2017Q1 RC-N states its total; the category grid that builds the
    total before 2017 must reproduce it."""
    period = cdr.cached_periods()[-1]
    path = cdr.zip_path(period)
    roster = schedules.roster(path)
    long = schedules.read_period(path, codes=mdrm.wanted_codes())
    frame = schedules.resolve_items(long, (*mdrm.RCN_TOTALS, *mdrm.RCN_BY_CATEGORY), forms=roster)
    for total, parts in mdrm.RCN_TOTAL_COMPONENTS.items():
        summed = pl.sum_horizontal([pl.col(c).fill_null(0.0) for c in parts])
        rep = frame.with_columns((summed - pl.col(total)).alias("_d")).filter(pl.col(total).is_not_null())
        assert rep.height > 1000
        tie = rep.filter(pl.col("_d").abs() <= 1_000.0).height / rep.height
        assert tie > 0.99, (total, tie)
        over = rep.filter(pl.col("_d") > 1_000.0).height / rep.height
        assert over < 0.005, (total, over)


@needs_cache
def test_partition_ties_for_an_old_quarter() -> None:
    """The RC-C partition must close on the 2001 form as on the 2026 one."""
    oldest = cdr.cached_periods()[0]
    path = cdr.zip_path(oldest)
    roster = schedules.roster(path)
    long = schedules.read_period(path, codes=mdrm.wanted_codes())
    frame = panel.add_partition_check(
        schedules.resolve_items(long, (*mdrm.RCC_LOAN_ITEMS, mdrm.RCC_TOTAL, mdrm.RCC_UNEARNED), forms=roster)
    )
    checked = frame.filter(pl.col("loans_rcc_total").abs() > 0)
    assert checked.height > 1000
    # A handful of 2001 filers report categories a few thousand dollars
    # over their own total -- data entry, not mapping -- so the bar is a
    # twentieth of a percent rather than zero.  A mapping defect is never
    # that small: the ones found so far were 0.86% (unearned income) and
    # 62% (item 9 counted twice).
    over = checked.filter(pl.col("rcc_residual_pct") > 0.05)
    assert over.is_empty(), over.select("rssd", "loans_rcc_total", "rcc_residual").head(5)
    exact = checked.filter(pl.col("rcc_residual").abs() <= 1_000.0).height
    assert exact / checked.height > 0.98


def test_push_down_restart_does_not_produce_a_negative_quarter() -> None:
    """Fleet National Bank's 2004Q2: acquired on April 1, books restarted,
    year-to-date charge-offs of 40 against a Q1 year-to-date of 155."""
    frame = pl.DataFrame(
        {
            "rssd": [76201, 76201, 76201],
            "period": [dt.date(2004, 3, 31), dt.date(2004, 6, 30), dt.date(2004, 9, 30)],
            "charge_offs_total": [155.0, 40.0, 95.0],
            "net_income": [300.0, 120.0, 250.0],
        }
    )
    out = panel.quarterize(frame)
    assert out["charge_offs_total"].to_list() == [155.0, 40.0, 55.0]
    # Every flow follows the restart, including ones that could fall anyway.
    assert out["net_income"].to_list() == [300.0, 120.0, 130.0]
    assert out["flow_reset"].to_list() == [False, True, False]
