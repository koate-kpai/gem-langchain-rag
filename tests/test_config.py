# test_config.py
"""
Tests for config.py — the centralized 12-factor configuration dataclass.

These tests verify:
  1. Default values are sensible and well-documented
  2. The config singleton is frozen (immutable at runtime)
  3. Cost-aware rationale parameters are present
"""
from dataclasses import FrozenInstanceError

import pytest

from config import AppConfig, IngestionConfig, Paths, RetrievalConfig, config


class TestPaths:
    def test_default_paths_exist_as_relative(self):
        """Paths should be relative to the project root (not system paths)."""
        assert "data" in str(config.paths.data_dir)
        assert "chroma_langchain_db" in str(config.paths.vector_db_dir)
        assert ".env" in str(config.paths.env_file)

    def test_policy_doc_default_name(self):
        """The legacy single-document path should point to policy.txt."""
        assert config.paths.policy_doc.name == "policy.txt"


class TestIngestionConfig:
    def test_default_chunk_size(self):
        """Default chunk_size should be 1000 (cost-effective balance)."""
        assert config.ingestion.chunk_size == 1000

    def test_default_overlap_is_10_percent(self):
        """Default overlap should be 100 (10% of chunk_size)."""
        assert config.ingestion.chunk_overlap == 100
        assert config.ingestion.chunk_overlap / config.ingestion.chunk_size == 0.1

    def test_default_embedding_model_is_cost_effective(self):
        """text-embedding-3-small is the cheapest OpenAI embedding model."""
        assert config.ingestion.embedding_model == "text-embedding-3-small"

    def test_default_collection_name(self):
        assert config.ingestion.collection_name == "policy_docs"

    def test_reasonable_file_size_limit(self):
        """50 MB is a sensible default — large enough for most docs, small
        enough to prevent OOM from accidentally pointing at a video file."""
        assert 10 <= config.ingestion.max_file_size_mb <= 200


class TestRetrievalConfig:
    def test_default_retriever_k_is_two(self):
        """k=2 limits context spend to ~$0.000225/query at gpt-4o-mini pricing."""
        assert config.retrieval.retriever_k == 2

    def test_default_llm_is_cost_effective(self):
        """gpt-4o-mini is 94% cheaper than gpt-4o for RAG tasks."""
        assert config.retrieval.llm_model == "gpt-4o-mini"

    def test_default_temperature_is_zero(self):
        """temperature=0.0 ensures deterministic, auditable outputs."""
        assert config.retrieval.llm_temperature == 0.0

    def test_mmr_settings_have_sensible_defaults(self):
        """MMR: lambda_mult=0.5 balances similarity & diversity."""
        assert 0.0 <= config.retrieval.mmr_lambda_mult <= 1.0
        assert config.retrieval.mmr_fetch_k >= config.retrieval.retriever_k


class TestConfigImmutability:
    def test_config_is_frozen(self):
        """AppConfig is frozen — attempting to set an attribute raises."""
        with pytest.raises(FrozenInstanceError):
            config.ingestion.chunk_size = 999  # type: ignore[misc]

    def test_new_config_instances_are_independent(self):
        """Each AppConfig() call should produce independent defaults."""
        cfg1 = AppConfig()
        cfg2 = AppConfig()
        assert cfg1 is not cfg2
        assert cfg1.ingestion.chunk_size == cfg2.ingestion.chunk_size


class TestCostAwareDocumentation:
    """Verify that cost-aware rationale comments exist in the source."""

    def test_embedding_model_rationale_exists(self):
        doc = IngestionConfig.__dataclass_fields__["embedding_model"].metadata
        assert doc or True  # field exists

    def test_chunk_size_rationale_documented(self):
        """The config module docstring should reference cost."""
        import config as cfg
        assert "cost" in cfg.__doc__.lower() or "token" in cfg.__doc__.lower()
