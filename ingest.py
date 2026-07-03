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
"""
import os
import re
import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import chromadb
from chromadb.errors import NotFoundError

# Load environment variables from .env (must exist before any API calls)
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

# Set up basic logging (you can adjust the level and format)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Resolve paths relative to the location of this script
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "policy.txt"
DB_DIR = BASE_DIR / "chroma_langchain_db"
COLLECTION_NAME = "policy_docs"


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
            '    OPENAI_API_KEY="sk-your-actual-key-here"\n\n"
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


def ingest_data():
    # --- Pre-flight checks ---
    # Order matters: check local state before making any API calls
    if not env_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {env_path}\n"
            "Copy .env.example to .env and add your OpenAI API key:\n\n"
            "    copy .env.example .env"
        )
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Document not found: {DATA_PATH}")
    # Validate the API key before any OpenAI calls — fail fast to avoid wasted cost
    _validate_api_key(os.getenv("OPENAI_API_KEY"))

    # 1. Load document
    logging.info("Loading document from %s", DATA_PATH)
    loader = TextLoader(str(DATA_PATH))
    raw_docs = loader.load()
    if not raw_docs:
        raise ValueError("The document is empty. Nothing to ingest.")

    # 2. Chunk intelligently (preserving paragraphs and sentences)
    logging.info("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # meaningful chunk size
        chunk_overlap=100,  # sufficient overlap for context
        separators=["\n\n", "\n", " ", ""],  # split at natural boundaries
    )
    chunks = text_splitter.split_documents(raw_docs)
    logging.info("Created %d chunks.", len(chunks))

    # 3. Prevent duplicate documents by deleting the existing collection
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
        logging.info("Deleted existing collection '%s'.", COLLECTION_NAME)
    except NotFoundError:
        logging.info("No existing collection '%s' to delete.", COLLECTION_NAME)

    # 4. Embed and store in ChromaDB
    logging.info("Generating embeddings and storing vectors...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
        collection_name=COLLECTION_NAME,
    )

    logging.info("Ingestion complete! The database is ready.")


if __name__ == "__main__":
    ingest_data()
