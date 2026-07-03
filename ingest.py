# ingest.py
"""
Ingestion pipeline: loads documents, chunks them, embeds, and stores in ChromaDB.

Security & Operations Philosophy
--------------------------------
- Defense-in-depth: layered guardrails (gitignore → env validation → API key format check)
- Fail fast: pre-flight checks catch misconfiguration before any billable API call
- Idempotency: each ingest run is safe to re-execute (destructive by design in this MVP;
  see Commit 4 for incremental dedup upgrade)
- Cost awareness: validation occurs before any OpenAI API call to avoid wasted spend
- Extensibility: multi-format document loading via a registry pattern (loaders.py)
"""
import os
import re
import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import chromadb
from chromadb.errors import NotFoundError

from config import config  # centralized 12-factor config (see config.py)
from loaders import find_documents, get_loader_for_file, validate_file_size

# Load environment variables from .env (must exist before any API calls)
load_dotenv(dotenv_path=config.paths.env_file)

# Set up basic logging (you can adjust the level and format)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ---------------------------------------------------------------------------
# Security: API key validation
# ---------------------------------------------------------------------------
def _validate_api_key(key: str | None) -> str:
    """
    Pre-flight API key validation.

    Security rationale: OpenAI returns a generic 401 for any auth failure.
    By validating format and detecting placeholder values here, we give the
    user a clear, actionable message instead of a cryptic error — and we
    avoid making a billable API call that would fail anyway.

    DevSecOps pattern: fail fast + fail safe.
    """
    if not key:
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Create a .env file in the project root with:\n\n"
            '    OPENAI_API_KEY="sk-your-actual-key-here"\n\n'
            "See .env.example for instructions."
        )
    # Detect placeholder keys from .env.example (sk-your-actual-key-here)
    if "your-actual-key" in key:
        raise ValueError(
            "OPENAI_API_KEY is set to the placeholder value from .env.example.\n"
            "Replace it with a real key from https://platform.openai.com/api-keys"
        )
    # Validate key format: OpenAI project keys start with sk-proj-,
    # org keys start with sk-. Minimum reasonable length is 20+ chars.
    if not re.match(r"^sk-(proj-)?[A-Za-z0-9]{20,}$", key):
        logging.warning(
            "OPENAI_API_KEY format is unusual. Expected 'sk-...' or 'sk-proj-...'. "
            "Double-check the key at https://platform.openai.com/api-keys"
        )
    return key


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _fmt_size(path: Path) -> str:
    """Human-readable file size."""
    size_bytes = path.stat().st_size
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 ** 2):.1f} MB"


def ingest_data():
    # --- Pre-flight checks ---
    # Order matters: check local state before making any API calls
    if not config.paths.env_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config.paths.env_file}\n"
            "Copy .env.example to .env and add your OpenAI API key:\n\n"
            "    copy .env.example .env"
        )
    # Validate the API key before any OpenAI calls — fail fast to avoid wasted cost
    _validate_api_key(os.getenv("OPENAI_API_KEY"))

    # 1. Load documents — supports .txt, .md, .pdf via extensible registry
    #    First, try to find supported documents in the data/ directory.
    #    If none found, fall back to the legacy single-document path.
    doc_files = find_documents(config.paths.data_dir, config.ingestion.data_glob_pattern)
    if not doc_files:
        if config.paths.policy_doc.exists():
            doc_files = [config.paths.policy_doc]
            logging.info("No multi-format documents found; falling back to '%s'", config.paths.policy_doc.name)
        else:
            raise FileNotFoundError(
                f"No documents found in '{config.paths.data_dir}' and fallback "
                f"'{config.paths.policy_doc}' does not exist. "
                "Add .txt, .md, or .pdf files to the data/ directory."
            )

    raw_docs = []
    for file_path in doc_files:
        validate_file_size(file_path, config.ingestion.max_file_size_mb)
        logging.info("Loading '%s' (%s)", file_path.name, _fmt_size(file_path))
        loader = get_loader_for_file(file_path)
        file_docs = loader.load()
        logging.info("  → %d page(s) loaded", len(file_docs))
        raw_docs.extend(file_docs)

    if not raw_docs:
        raise ValueError("All source documents are empty. Nothing to ingest.")

    # 2. Chunk intelligently (preserving paragraphs and sentences)
    logging.info("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.ingestion.chunk_size,
        chunk_overlap=config.ingestion.chunk_overlap,
        separators=config.ingestion.chunk_separators,
    )
    chunks = text_splitter.split_documents(raw_docs)
    logging.info("Created %d chunks.", len(chunks))

    # 3. Prevent duplicate documents by deleting the existing collection
    client = chromadb.PersistentClient(path=str(config.paths.vector_db_dir))
    try:
        client.delete_collection(config.ingestion.collection_name)
        logging.info("Deleted existing collection '%s'.", config.ingestion.collection_name)
    except NotFoundError:
        logging.info("No existing collection '%s' to delete.", config.ingestion.collection_name)

    # 4. Embed and store in ChromaDB
    logging.info("Generating embeddings and storing vectors...")
    embeddings = OpenAIEmbeddings(model=config.ingestion.embedding_model)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(config.paths.vector_db_dir),
        collection_name=config.ingestion.collection_name,
    )

    logging.info("Ingestion complete! The database is ready.")


if __name__ == "__main__":
    ingest_data()
