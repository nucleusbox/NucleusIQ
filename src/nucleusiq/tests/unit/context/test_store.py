"""Tests for ContentStore and ContentRef."""

from nucleusiq.agents.context.store import ContentRef, ContentStore


class TestContentRef:
    def test_to_marker_contains_key(self):
        ref = ContentRef(
            key="search:abc123", original_tokens=5000, preview="first lines..."
        )
        marker = ref.to_marker()
        assert "[context_ref: search:abc123]" in marker
        assert "~5000 tokens" in marker
        assert "first lines..." in marker

    def test_to_marker_contains_preview_boundaries(self):
        ref = ContentRef(key="tool:xyz", original_tokens=100, preview="data")
        marker = ref.to_marker()
        assert "--- preview ---" in marker
        assert "--- end preview ---" in marker

    def test_frozen(self):
        ref = ContentRef(key="k", original_tokens=0, preview="")
        assert ref.key == "k"


class TestContentStore:
    def test_store_and_retrieve(self):
        store = ContentStore()
        content = "line1\nline2\nline3\nline4\nline5"
        ref = store.store("key1", content, 100)
        assert store.size == 1
        assert store.retrieve("key1") == content
        assert ref.key == "key1"
        assert ref.original_tokens == 100

    def test_preview(self):
        store = ContentStore()
        lines = "\n".join(f"line {i}" for i in range(20))
        ref = store.store("key1", lines, 500, preview_lines=5)
        preview = store.preview("key1")
        assert preview is not None
        assert "line 0" in preview
        assert "line 4" in preview
        assert "15 more lines" in preview

    def test_retrieve_nonexistent(self):
        store = ContentStore()
        assert store.retrieve("no") is None

    def test_contains(self):
        store = ContentStore()
        store.store("x", "data", 10)
        assert store.contains("x")
        assert not store.contains("y")

    def test_remove(self):
        store = ContentStore()
        store.store("x", "data", 10)
        assert store.remove("x")
        assert not store.contains("x")
        assert not store.remove("x")

    def test_clear(self):
        store = ContentStore()
        store.store("a", "data", 10)
        store.store("b", "data", 10)
        store.clear()
        assert store.size == 0

    def test_keys(self):
        store = ContentStore()
        store.store("a", "data", 10)
        store.store("b", "data", 10)
        assert sorted(store.keys()) == ["a", "b"]

    def test_preview_max_chars_truncates_dense_content(self):
        """Dense content (few newlines) must be capped by character budget."""
        store = ContentStore()
        dense = "x" * 10_000
        ref = store.store("k", dense, 2500, preview_max_chars=500)
        assert len(ref.preview) < 600
        assert "remaining" in ref.preview

    def test_preview_max_chars_no_op_for_short_content(self):
        """Short content should not be affected by preview_max_chars."""
        store = ContentStore()
        short = "hello world"
        ref = store.store("k", short, 3, preview_max_chars=500)
        assert ref.preview == short

    def test_preview_max_chars_respects_line_based_when_smaller(self):
        """When line-based preview is smaller than char budget, keep it."""
        store = ContentStore()
        content = "\n".join(f"line{i}" for i in range(50))
        ref = store.store("k", content, 500, preview_lines=3, preview_max_chars=5000)
        assert "line0" in ref.preview
        assert "line2" in ref.preview
        assert "47 more lines" in ref.preview
