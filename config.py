# config.py
"""
Centralized configuration for the RAG pipeline.

Design Philosophy (12-Factor App — https://12factor.net/config)
---------------------------------------------------------------
This module is the single source of truth for all tunable parameters.
Loading order: 1) hardcoded defaults 2) config.py overrides 3) .env vars.

Every parameter includes a documented rationale for:
  - **Cost impact**: how this choice affects OpenAI API spend
  - **Performance impact**: latency, retrieval quality, memory usage
  - **Compliance impact**: data retention, audit trail, PII handling

This pattern enables:
  - Configuration changes without touching source code
  - Cost optimization per deployment environment (dev/staging/prod)
  - Clear audit trail of why each value was chosen
"""
import os
from dataclasses import dataclass, field
from pathlib import Path


# Base directory — resolved once at import time, never changes at runtime
_BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Paths:
    """Filesystem paths — all resolved relative to the project root.

    Rationale: using Path(__file__).resolve().parent ensures the application
    works regardless of the working directory from which it's launched.
    This is critical for cron jobs, CI/CD pipelines, and containerized deployments.
    """

    # Source documents directory — add .txt, .md, .pdf files here
    data_dir: Path = _BASE_DIR / "data"

    # Single-document path (MVP); will be replaced by glob-based loading in Commit 3
    policy_doc: Path = _BASE_DIR / "data" / "policy.txt"

    # Persistent vector database directory (git-ignored in .gitignore)
    vector_db_dir: Path = _BASE_DIR / "chroma_langchain_db"

    # Environment file (git-ignored; copy .env.example to create)
    env_file: Path = _BASE_DIR / ".env"


@dataclass(frozen=True)
class IngestionConfig:
    """Document chunking and embedding parameters.

    Cost/Quality Tradeoffs
    ----------------------
    chunk_size=1000:
      OpenAI text-embedding-3-small costs $0.02/1M tokens.
      At ~750 tokens per chunk (English prose), 1000 chunks cost ~$0.015 to embed.
      Larger chunks (2000+) improve retrieval recall but increase storage costs
      and retrieval latency. Smaller chunks (500) reduce cost per query but may
      fragment semantic meaning.

    chunk_overlap=100:
      10% overlap preserves sentence boundaries across chunk boundaries.
      Too little overlap (<50) risks losing context at chunk edges.
      Too much overlap (>200) wastes tokens and storage without meaningful gain.

    embedding_model="text-embedding-3-small":
      The most cost-effective OpenAI embedding model at $0.02/1M tokens.
      Generates 1536-dimensional vectors — good balance of precision vs storage.
      text-embedding-3-large ($0.13/1M) offers 3072 dimensions but is 6.5x more
      expensive with marginal quality gains for domain-specific RAG.
    """
    chunk_size: int = 1000
    chunk_overlap: int = 100
    chunk_separators: list[str] = field(
        default_factory=lambda: ["\n\n", "\n", " ", ""]
    )
    embedding_model: str = "text-embedding-3-small"
    collection_name: str = "policy_docs"


@dataclass(frozen=True)
class RetrievalConfig:
    """Query-time retrieval and generation parameters.

    Cost/Quality Tradeoffs
    ----------------------
    retriever_k=2:
      Each retrieved chunk adds ~750 tokens to the LLM context.
      With gpt-4o-mini at $0.15/1M input tokens, k=2 costs ~$0.000225 per query.
      k=5 would cost ~$0.00056 but adds diminishing returns (studies show
      k=3-4 is the sweet spot for most QA tasks).

    llm_model="gpt-4o-mini":
      90% cheaper than gpt-4o ($0.15 vs $2.50/1M input tokens).
      Sufficient quality for structured RAG where the LLM's primary task is
      summarization and extraction, not creative generation.
      For open-ended analysis tasks, consider gpt-4o (5x cost, better reasoning).

    temperature=0.0:
      Deterministic outputs for RAG are critical. Temperature >0 introduces
      variation that can cause the model to hallucinate different answers to
      the same question. For creative tasks (brainstorming), 0.7-1.0 is appropriate.
    """
    retriever_k: int = 2
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0


@dataclass(frozen=True)
class AppConfig:
    """Top-level configuration aggregating all sub-configs."""

    paths: Paths = field(default_factory=Paths)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)


# ---------------------------------------------------------------------------
# Singleton instance — imported once, used everywhere
# ---------------------------------------------------------------------------
# This is the single source of truth. Import it as:
#   from config import config
#   config.ingestion.chunk_size
#   config.paths.vector_db_dir
#
# If you need environment-specific overrides, extend this module to read
# from YAML/TOML files or environment variables with a fallback chain.
config = AppConfig()
