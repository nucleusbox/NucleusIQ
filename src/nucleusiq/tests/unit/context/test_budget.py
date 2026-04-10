"""Tests for Region, ContextBudget, and ContextLedger."""

import pytest
from nucleusiq.agents.context.budget import ContextBudget, ContextLedger, Region


class TestRegion:
    def test_region_values(self):
        assert Region.SYSTEM == "system"
        assert Region.TOOL_RESULT == "tool_result"
        assert len(Region) == 6


class TestContextBudget:
    def test_effective_limit(self):
        b = ContextBudget(
            max_tokens=100_000, response_reserve=8192, allocated=0, by_region={}
        )
        assert b.effective_limit == 91_808

    def test_available_empty(self):
        b = ContextBudget(
            max_tokens=100_000, response_reserve=8192, allocated=0, by_region={}
        )
        assert b.available == 91_808

    def test_available_partial(self):
        b = ContextBudget(
            max_tokens=100_000, response_reserve=8192, allocated=50_000, by_region={}
        )
        assert b.available == 41_808

    def test_available_full(self):
        b = ContextBudget(
            max_tokens=100_000, response_reserve=8192, allocated=91_808, by_region={}
        )
        assert b.available == 0

    def test_available_over(self):
        b = ContextBudget(
            max_tokens=100_000, response_reserve=8192, allocated=100_000, by_region={}
        )
        assert b.available == 0

    def test_utilization_zero(self):
        b = ContextBudget(
            max_tokens=100_000, response_reserve=8192, allocated=0, by_region={}
        )
        assert b.utilization == 0.0

    def test_utilization_half(self):
        b = ContextBudget(
            max_tokens=100_000, response_reserve=8192, allocated=45_904, by_region={}
        )
        assert abs(b.utilization - 0.5) < 0.001

    def test_utilization_capped(self):
        b = ContextBudget(
            max_tokens=100_000, response_reserve=8192, allocated=200_000, by_region={}
        )
        assert b.utilization == 1.0

    def test_can_fit_true(self):
        b = ContextBudget(
            max_tokens=100_000, response_reserve=8192, allocated=50_000, by_region={}
        )
        assert b.can_fit(10_000)

    def test_can_fit_false(self):
        b = ContextBudget(
            max_tokens=100_000, response_reserve=8192, allocated=90_000, by_region={}
        )
        assert not b.can_fit(10_000)

    def test_frozen(self):
        b = ContextBudget(
            max_tokens=100_000, response_reserve=8192, allocated=0, by_region={}
        )
        with pytest.raises(AttributeError):
            b.allocated = 5000  # type: ignore[misc]


class TestContextLedger:
    def test_allocate_and_snapshot(self):
        ledger = ContextLedger(100_000, 8192)
        ledger.allocate("msg1", 500, Region.USER)
        snap = ledger.snapshot()
        assert snap.allocated == 500
        assert snap.by_region["user"] == 500

    def test_allocate_multiple_regions(self):
        ledger = ContextLedger(100_000, 8192)
        ledger.allocate("sys", 200, Region.SYSTEM)
        ledger.allocate("tool1", 1000, Region.TOOL_RESULT)
        ledger.allocate("user1", 300, Region.USER)
        snap = ledger.snapshot()
        assert snap.allocated == 1500
        assert snap.by_region["system"] == 200
        assert snap.by_region["tool_result"] == 1000
        assert snap.by_region["user"] == 300

    def test_deallocate(self):
        ledger = ContextLedger(100_000, 8192)
        ledger.allocate("msg1", 500, Region.USER)
        freed = ledger.deallocate("msg1")
        assert freed == 500
        assert ledger.total_allocated == 0

    def test_deallocate_nonexistent(self):
        ledger = ContextLedger(100_000, 8192)
        freed = ledger.deallocate("nonexistent")
        assert freed == 0

    def test_reallocate_replaces(self):
        ledger = ContextLedger(100_000, 8192)
        ledger.allocate("msg1", 500, Region.USER)
        ledger.allocate("msg1", 200, Region.ASSISTANT)
        assert ledger.total_allocated == 200
        assert ledger.snapshot().by_region["assistant"] == 200
        assert ledger.snapshot().by_region["user"] == 0

    def test_entries_by_region(self):
        ledger = ContextLedger(100_000, 8192)
        ledger.allocate("t1", 100, Region.TOOL_RESULT)
        ledger.allocate("t2", 200, Region.TOOL_RESULT)
        ledger.allocate("u1", 50, Region.USER)
        entries = ledger.entries_by_region(Region.TOOL_RESULT)
        assert len(entries) == 2
        assert sum(e.tokens for e in entries) == 300

    def test_metadata_on_entry(self):
        ledger = ContextLedger(100_000, 8192)
        ledger.allocate(
            "msg1",
            500,
            Region.TOOL_RESULT,
            importance=0.9,
            source_type="tool_result",
            restorable=True,
        )
        entry = ledger.get_entry("msg1")
        assert entry is not None
        assert entry.importance == 0.9
        assert entry.source_type == "tool_result"
        assert entry.restorable is True

    def test_reset(self):
        ledger = ContextLedger(100_000, 8192)
        ledger.allocate("msg1", 500, Region.USER)
        ledger.reset()
        assert ledger.total_allocated == 0
        assert ledger.entry_count == 0

    def test_reserved_in_snapshot(self):
        ledger = ContextLedger(100_000, 8192)
        snap = ledger.snapshot()
        assert snap.by_region["reserved"] == 8192
        assert snap.response_reserve == 8192
