# test_search.py
"""
Tests for search.py — retrieval, source formatting, and the query pipeline.

All external API calls are mocked (see conftest.py). These tests verify:
  1. API key validation
  2. Source document formatting with relevance scores
  3. Guardrails (missing collection, empty database)
  4. Verbose mode toggling
"""
from unittest.mock import MagicMock, patch

import pytest

from search import (
    _validate_api_key,
    format_docs_with_sources,
    print_sources,
    _verbose,
)


class TestValidateApiKey:
    """Same validation tests as ingest, ensuring both modules are consistent."""

    def test_valid_key_passes(self):
        key = _validate_api_key("sk-proj-Abcdefghijklmnopqrstuvwxyz123456")
        assert key == "sk-proj-Abcdefghijklmnopqrstuvwxyz123456"

    def test_no_key_raises(self):
        with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
            _validate_api_key(None)

    def test_placeholder_raises(self):
        with pytest.raises(ValueError, match="placeholder"):
            _validate_api_key("sk-your-actual-key-here")


class TestFormatDocsWithSources:
    def test_basic_formatting(self):
        """Each doc should be tagged with source and relevance (3 decimal places)."""
        docs_and_scores = [
            (MagicMock(page_content="Content A", metadata={"source": "data/policy.txt"}), 0.85),
            (MagicMock(page_content="Content B", metadata={"source": "data/manual.md"}), 0.72),
        ]
        result = format_docs_with_sources(docs_and_scores)

        assert "[Source: policy.txt (Chunk 1, Relevance: 0.850)]" in result
        assert "[Source: manual.md (Chunk 2, Relevance: 0.720)]" in result
        assert "Content A" in result
        assert "Content B" in result

    def test_unknown_source_handling(self):
        """Docs without source metadata should show 'unknown'."""
        docs_and_scores = [
            (MagicMock(page_content="Test", metadata={}), 0.5),
        ]
        result = format_docs_with_sources(docs_and_scores)
        assert "unknown" in result

    def test_empty_list_returns_empty_string(self):
        """An empty list of docs should return an empty string."""
        result = format_docs_with_sources([])
        assert result == ""

    def test_score_precision(self):
        """Scores should be formatted to 3 decimal places."""
        docs_and_scores = [
            (MagicMock(page_content="X", metadata={"source": "a.txt"}), 0.87654),
        ]
        result = format_docs_with_sources(docs_and_scores)
        assert "Relevance: 0.877" in result


class TestPrintSources:
    def test_prints_source_summary(self, capsys):
        """print_sources should output source file names and scores."""
        docs_and_scores = [
            (MagicMock(page_content="A", metadata={"source": "data/policy.txt"}), 0.85),
        ]
        print_sources(docs_and_scores)
        captured = capsys.readouterr()
        assert "policy.txt" in captured.out
        assert "0.85" in captured.out

    def test_verbose_mode_dumps_chunks(self, capsys, monkeypatch):
        """With _verbose=True, raw chunk content should be printed."""
        import search as s
        monkeypatch.setattr(s, "_verbose", True)
        docs_and_scores = [
            (MagicMock(page_content="Raw chunk text", metadata={"source": "a.txt"}), 0.5),
        ]
        print_sources(docs_and_scores)
        captured = capsys.readouterr()
        assert "Raw chunk text" in captured.out
        monkeypatch.setattr(s, "_verbose", False)  # reset

    def test_no_verbose_no_dump(self, capsys, monkeypatch):
        """Without verbose mode, raw chunk content should not be printed."""
        import search as s
        monkeypatch.setattr(s, "_verbose", False)
        docs_and_scores = [
            (MagicMock(page_content="Secret chunk", metadata={"source": "a.txt"}), 0.5),
        ]
        print_sources(docs_and_scores)
        captured = capsys.readouterr()
        assert "Secret chunk" not in captured.out


class TestRunQueryGuardrails:
    def test_missing_env_file_raises(self, tmp_path, monkeypatch):
        """run_query should fail fast if .env doesn't exist."""
        import config as cfg
        missing = tmp_path / ".env.missing"
        # Use object.__setattr__ to bypass frozen dataclass restriction
        object.__setattr__(cfg.config.paths, "env_file", missing)
        try:
            from search import run_query
            with pytest.raises(FileNotFoundError, match="not found"):
                run_query("test question")
        finally:
            # Restore original path
            object.__setattr__(
                cfg.config.paths,
                "env_file",
                cfg._BASE_DIR / ".env",
            )

    @patch("search.Chroma")
    def test_empty_collection_prints_error(self, mock_chroma, capsys):
        """An empty database should print an error and return None."""
        mock_store = MagicMock()
        mock_store._collection.count.return_value = 0
        mock_chroma.return_value = mock_store

        from search import run_query
        result = run_query("test question")
        captured = capsys.readouterr()
        assert "empty" in captured.out
        assert result is None
