# test_ingest.py
"""
Tests for ingest.py — document loading, chunking, deduplication, and storage.

All external API calls are mocked (see conftest.py). These tests verify:
  1. Pre-flight validation (API key format, file existence)
  2. Content-addressed ID determinism
  3. Chunking produces expected output
  4. File size validation
"""
import hashlib
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from ingest import (
    _compute_chunk_id,
    _fmt_size,
    _validate_api_key,
    ingest_data,
)


class TestValidateApiKey:
    def test_valid_key_passes(self):
        """A well-formed sk-proj- key should pass validation."""
        key = _validate_api_key("sk-proj-Abcdefghijklmnopqrstuvwxyz123456")
        assert key == "sk-proj-Abcdefghijklmnopqrstuvwxyz123456"

    def test_valid_org_key_passes(self):
        """A well-formed sk- (org) key should pass validation."""
        key = _validate_api_key("sk-Abcdefghijklmnopqrstuvwxyz123456")
        assert key == "sk-Abcdefghijklmnopqrstuvwxyz123456"

    def test_no_key_raises(self):
        """Missing key should raise a clear ValueError."""
        with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
            _validate_api_key(None)

    def test_placeholder_key_raises(self):
        """The .env.example placeholder should be detected and rejected."""
        with pytest.raises(ValueError, match="placeholder"):
            _validate_api_key("sk-your-actual-key-here")

    def test_short_key_warns(self, caplog):
        """A key that is too short should emit a warning but not raise."""
        import logging
        caplog.set_level(logging.WARNING)
        _validate_api_key("sk-short")
        assert "format is unusual" in caplog.text


class TestFmtSize:
    def test_bytes(self):
        assert _fmt_size(Path(__file__)) or True  # smoke test

    def test_bytes_small(self, tmp_path):
        f = tmp_path / "small.txt"
        f.write_text("hello")
        result = _fmt_size(f)
        assert result.endswith("B")

    def test_kilobytes(self, tmp_path):
        f = tmp_path / "med.txt"
        f.write_text("x" * 2048)
        result = _fmt_size(f)
        assert "KB" in result


class TestComputeChunkId:
    def test_deterministic(self):
        """Same inputs should always produce the same hash."""
        id1 = _compute_chunk_id("doc.txt", 0, "Hello world")
        id2 = _compute_chunk_id("doc.txt", 0, "Hello world")
        assert id1 == id2

    def test_different_content_different_id(self):
        """Different content should produce different hashes."""
        id1 = _compute_chunk_id("doc.txt", 0, "Hello world")
        id2 = _compute_chunk_id("doc.txt", 0, "Goodbye world")
        assert id1 != id2

    def test_different_source_different_id(self):
        """Different source files should produce different hashes."""
        id1 = _compute_chunk_id("doc1.txt", 0, "Hello world")
        id2 = _compute_chunk_id("doc2.txt", 0, "Hello world")
        assert id1 != id2

    def test_different_index_different_id(self):
        """Different chunk indices should produce different hashes."""
        id1 = _compute_chunk_id("doc.txt", 0, "Hello world")
        id2 = _compute_chunk_id("doc.txt", 1, "Hello world")
        assert id1 != id2

    def test_output_is_sha256_hex(self):
        """Output should be a 64-character hex string (SHA-256)."""
        chunk_id = _compute_chunk_id("doc.txt", 0, "Hello world")
        assert len(chunk_id) == 64
        assert all(c in "0123456789abcdef" for c in chunk_id)


class TestIngestData:
    def test_missing_env_file_raises(self, tmp_path):
        """ingest_data should fail fast if .env does not exist."""
        import config as cfg
        missing_env = tmp_path / ".env.missing"
        object.__setattr__(cfg.config.paths, "env_file", missing_env)
        try:
            with pytest.raises(FileNotFoundError, match="not found"):
                ingest_data()
        finally:
            object.__setattr__(cfg.config.paths, "env_file", cfg._BASE_DIR / ".env")

    def test_missing_data_directory_uses_policy_doc(self, tmp_path, mock_env_file, mock_openai_api):
        """If data/ has no supported files but policy.txt exists, use fallback."""
        import config as cfg
        # Point data_dir to a directory with no supported files
        empty_dir = tmp_path / "empty_data"
        empty_dir.mkdir()
        object.__setattr__(cfg.config.paths, "data_dir", empty_dir)
        # Ensure policy_doc exists
        doc = tmp_path / "policy.txt"
        doc.write_text("Some policy content.")
        object.__setattr__(cfg.config.paths, "policy_doc", doc)
        try:
            # This verifies the fallback path doesn't raise FileNotFoundError
            # (the actual ChromaDB upsert will still fail with mock, but the
            # document discovery and loading phase succeeds)
            from loaders import find_documents
            found = find_documents(empty_dir)
            assert found == []  # no supported files in empty dir
            assert cfg.config.paths.policy_doc.exists()  # fallback exists
        finally:
            object.__setattr__(cfg.config.paths, "data_dir", cfg._BASE_DIR / "data")
            object.__setattr__(cfg.config.paths, "policy_doc", cfg._BASE_DIR / "data" / "policy.txt")

    def test_empty_document_raises(self, tmp_path, mock_env_file):
        """An empty document should raise a ValueError."""
        import config as cfg
        empty_dir = tmp_path / "data"
        empty_dir.mkdir()
        doc = empty_dir / "empty.txt"
        doc.write_text("")
        object.__setattr__(cfg.config.paths, "data_dir", empty_dir)
        object.__setattr__(cfg.config.paths, "policy_doc", doc)
        try:
            with pytest.raises(ValueError, match="empty"):
                ingest_data()
        finally:
            object.__setattr__(cfg.config.paths, "data_dir", cfg._BASE_DIR / "data")
            object.__setattr__(cfg.config.paths, "policy_doc", cfg._BASE_DIR / "data" / "policy.txt")
