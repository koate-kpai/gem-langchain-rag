# ingest.py
import os
import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import chromadb
from chromadb.errors import NotFoundError

# Load environment variables
load_dotenv()

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


def ingest_data():
    # --- Pre-flight checks ---
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Document not found: {DATA_PATH}")
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

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
