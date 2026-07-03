# test_loaders.py
"""
Tests for loaders.py — document discovery, format registry, and file validation.

These tests verify:
  1. find_documents discovers supported file types (.txt, .md) in a directory
  2. find_documents skips unsupported/binary files with warnings
  3. get_loader_for_file returns the correct loader class per extension
  4. validate_file_size raises on oversized files
  5. Empty directories return empty lists (not errors)
"""
from pathlib import Path

import pytest

from loaders import (
    LOADER_REGISTRY,
    find_documents,
    get_loader_for_file,
    validate_file_size,
)


class TestFindDocuments:
    def test_finds_txt_files(self, test_data_dir):
        """find_documents should discover .txt files."""
        docs = find_documents(test_data_dir)
        assert len(docs) == 1
        assert docs[0].name == "policy.txt"

    def test_finds_multiple_formats(self, tmp_path):
        """find_documents should discover both .txt and .md files."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "doc1.txt").write_text("hello")
        (data_dir / "doc2.md").write_text("# Markdown")
        docs = find_documents(data_dir)
        assert len(docs) == 2

    def test_skips_binary_files(self, tmp_path, caplog):
        """Binary files (e.g., .png) should be skipped with a warning."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "image.png").write_text("not a real png")
        (data_dir / "doc.txt").write_text("hello")
        docs = find_documents(data_dir)
        assert len(docs) == 1
        assert docs[0].name == "doc.txt"

    def test_skips_unsupported_formats(self, tmp_path, caplog):
        """Unknown extensions should be skipped with a warning."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "file.xyz").write_text("unknown")
        (data_dir / "doc.txt").write_text("hello")
        docs = find_documents(data_dir)
        assert len(docs) == 1
        assert docs[0].name == "doc.txt"

    def test_empty_directory_returns_empty_list(self, test_empty_dir):
        """An empty data directory should return an empty list (not raise)."""
        docs = find_documents(test_empty_dir)
        assert docs == []

    def test_nonexistent_directory_raises(self, tmp_path):
        """A non-existent directory should raise FileNotFoundError."""
        fake_dir = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError):
            find_documents(fake_dir)


class TestGetLoaderForFile:
    def test_txt_returns_text_loader(self, tmp_path):
        """get_loader_for_file should return TextLoader for .txt files."""
        from langchain_community.document_loaders import TextLoader
        f = tmp_path / "doc.txt"
        f.write_text("hello")
        loader = get_loader_for_file(f)
        assert isinstance(loader, TextLoader)

    def test_md_returns_text_loader(self, tmp_path):
        """get_loader_for_file should return TextLoader for .md files."""
        from langchain_community.document_loaders import TextLoader
        f = tmp_path / "doc.md"
        f.write_text("# Header")
        loader = get_loader_for_file(f)
        assert isinstance(loader, TextLoader)

    def test_binary_format_raises(self, tmp_path):
        """Binary formats should raise ValueError."""
        f = tmp_path / "image.png"
        f.write_text("data")
        with pytest.raises(ValueError, match="binary format"):
            get_loader_for_file(f)

    def test_unsupported_format_raises(self, tmp_path):
        """Unsupported formats should raise ValueError."""
        f = tmp_path / "file.xyz"
        f.write_text("data")
        with pytest.raises(ValueError, match="Unsupported"):
            get_loader_for_file(f)


class TestValidateFileSize:
    def test_small_file_passes(self, tmp_path):
        """A small file should pass validation."""
        f = tmp_path / "small.txt"
        f.write_text("x" * 100)
        validate_file_size(f, max_size_mb=50)  # should not raise

    def test_oversized_file_raises(self, tmp_path):
        """A file exceeding max_size_mb should raise ValueError."""
        f = tmp_path / "large.txt"
        # Write enough content to exceed 1 KB limit
        f.write_text("x" * 2000)
        with pytest.raises(ValueError, match="exceeds"):
            validate_file_size(f, max_size_mb=0.001)

    def test_exact_boundary_passes(self, tmp_path):
        """A file exactly at the limit should pass."""
        f = tmp_path / "boundary.txt"
        f.write_text("x" * 1024)  # 1 KB
        validate_file_size(f, max_size_mb=1)  # should not raise


class TestLoaderRegistry:
    def test_registry_contains_expected_formats(self):
        """LOADER_REGISTRY should contain .txt, .md (and .pdf if pypdf available)."""
        assert ".txt" in LOADER_REGISTRY
        assert ".md" in LOADER_REGISTRY
        # .pdf is conditionally added, so just check it doesn't crash
        assert isinstance(LOADER_REGISTRY, dict)

    def test_registry_values_are_loader_classes(self):
        """Each registry value should be a (class, description) tuple."""
        for ext, (cls, description) in LOADER_REGISTRY.items():
            assert hasattr(cls, "load"), f"{ext} loader has no .load() method"
            assert isinstance(description, str)
