# search.py
"""
Semantic retrieval pipeline: embeds a user question, retrieves relevant chunks,
and generates a grounded answer using an LLM.

Security & Operations Philosophy
--------------------------------
- Defense-in-depth: API key validation before any external call
- Cost-aware retrieval: config.retrieval.retriever_k limits context token spend
- Fail gracefully: guardrails for missing database, empty collection, invalid config
- Source-grounded: every answer includes citations showing which policy document
  and chunk were used, with relevance scores for transparency

Responsible AI Note
-------------------
Source citation is a critical guardrail against hallucination. By showing the
user exactly which documents informed the answer, we enable:
  1. Trust verification — users can read the source directly
  2. Error detection — low-relevance sources alert users to weak retrieval
  3. Auditability — every answer is traceable to its source material
"""
import os
import re
import sys
import logging
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from chromadb.errors import NotFoundError  # to catch missing DB/collection

from config import config  # centralized 12-factor config (see config.py)

# Load .env — fail immediately if missing (don't let OpenAI return a cryptic 401)
load_dotenv(dotenv_path=config.paths.env_file)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ---------------------------------------------------------------------------
# Security: API key validation (shared pattern with ingest.py)
# ---------------------------------------------------------------------------
def _validate_api_key(key: str | None) -> str:
    """Validate the OpenAI API key before making any billable calls.

    Pattern: fail fast + fail safe. Catches placeholder keys and malformed
    formats early, saving the user both time and money.
    """
    if not key:
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Create a .env file in the project root with:\n\n"
            '    OPENAI_API_KEY="sk-your-actual-key-here"\n\n'
            "See .env.example for instructions."
        )
    if "your-actual-key" in key:
        raise ValueError(
            "OPENAI_API_KEY is set to the placeholder value from .env.example.\n"
            "Replace it with a real key from https://platform.openai.com/api-keys"
        )
    if not re.match(r"^sk-(proj-)?[A-Za-z0-9]{20,}$", key):
        logging.warning(
            "OPENAI_API_KEY format is unusual. Expected 'sk-...' or 'sk-proj-...'. "
            "Double-check the key at https://platform.openai.com/api-keys"
        )
    return key


# ---------------------------------------------------------------------------
# Source-aware document formatting
# ---------------------------------------------------------------------------
def format_docs_with_sources(docs_and_scores: list[tuple]) -> str:
    """Format retrieved documents with source citations and relevance scores.

    Each chunk is tagged with:
      [Source: filename (Relevance: 0.XX)]

    This gives the LLM explicit context boundaries AND makes the source
    provenance visible to the end user. The relevance score (cosine similarity
    to the query embedding) helps users gauge confidence.

    Args:
        docs_and_scores: List of (Document, score) tuples from
            similarity_search_with_relevance_scores.

    Returns:
        Formatted context string with source tags.
    """
    formatted_parts = []
    for idx, (doc, score) in enumerate(docs_and_scores, start=1):
        # Extract the source filename from metadata (LangChain adds 'source' automatically)
        source_file = doc.metadata.get("source", "unknown")
        source_name = source_file.split("\\")[-1].split("/")[-1]  # cross-platform basename

        # Build a tagged chunk with relevance score
        tagged = (
            f"[Source: {source_name} (Chunk {idx}, Relevance: {score:.3f})]\n"
            f"{doc.page_content}"
        )
        formatted_parts.append(tagged)

    return "\n\n---\n\n".join(formatted_parts)


def print_sources(docs_and_scores: list[tuple]) -> None:
    """Print a human-readable source summary for the user.

    Shows:
      - Which files were used
      - Individual chunk relevance scores
      - Total number of chunks retrieved

    This is printed AFTER the LLM answer so the user can verify the response
    against its source material.
    """
    print("\n" + "-" * 40)
    print("📚 SOURCES CONSULTED:")
    print("-" * 40)
    for idx, (doc, score) in enumerate(docs_and_scores, start=1):
        source_file = doc.metadata.get("source", "unknown")
        source_name = source_file.split("\\")[-1].split("/")[-1]
        print(f"  {idx}. {source_name} — Relevance: {score:.3f}")
    print(f"\n  Total chunks retrieved: {len(docs_and_scores)}")

    # Verbose mode: dump full chunk text for debugging
    if "--verbose" in sys.argv or "-v" in sys.argv:
        print("\n" + "-" * 40)
        print("🔍 RAW CHUNKS (verbose):")
        print("-" * 40)
        for idx, (doc, _score) in enumerate(docs_and_scores, start=1):
            source_file = doc.metadata.get("source", "unknown")
            source_name = source_file.split("\\")[-1].split("/")[-1]
            print(f"\n--- Chunk {idx} ({source_name}) ---")
            print(doc.page_content)
    print("-" * 40)


# ---------------------------------------------------------------------------
# Main query pipeline
# ---------------------------------------------------------------------------
def run_query(user_question):
    logging.info("Initializing vector store retriever...")

    # --- Pre-flight checks ---
    # Fail fast: validate environment before any external API call
    if not config.paths.env_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config.paths.env_file}\n"
            "Copy .env.example to .env and add your OpenAI API key:\n\n"
            "    copy .env.example .env"
        )
    _validate_api_key(os.getenv("OPENAI_API_KEY"))

    embeddings = OpenAIEmbeddings(model=config.ingestion.embedding_model)
    llm = ChatOpenAI(
        model=config.retrieval.llm_model,
        temperature=config.retrieval.llm_temperature,
    )

    # 1. Connect to ChromaDB – handle missing database/collection
    try:
        vectorstore = Chroma(
            persist_directory=str(config.paths.vector_db_dir),
            embedding_function=embeddings,
            collection_name=config.ingestion.collection_name,
        )
    except NotFoundError:
        print("\n❌ Error: The ChromaDB collection was not found.")
        print(
            "Please run the ingestion script first to create and populate the database.\n"
        )
        return

    # 2. Guardrail: Ensure the collection actually contains data
    if vectorstore._collection.count() == 0:
        print("\n❌ Error: The ChromaDB database is empty.")
        print("Please run the ingestion script to populate it with documents.\n")
        return

    # 3. Retrieve documents WITH relevance scores
    #    similarity_search_with_relevance_scores returns (Document, score) tuples
    #    where score is the cosine similarity (0 to 1, higher = more relevant).
    logging.info(
        "Retrieving top-%d chunks for query: '%s'",
        config.retrieval.retriever_k,
        user_question,
    )
    docs_and_scores = vectorstore.similarity_search_with_relevance_scores(
        user_question,
        k=config.retrieval.retriever_k,
    )

    if not docs_and_scores:
        print("\n❌ No relevant documents found for your query.")
        return

    # 4. Build prompt with source-tagged context
    system_prompt = (
        "You are a helpful HR assistant. Answer the user's question using ONLY "
        "the provided context below. Each context chunk is tagged with its source "
        "file and relevance score. If the context does not contain the answer, "
        "say 'I cannot find that in the company policy manual.'\n\n"
        "When you use information from a specific source, cite it inline:\n"
        '  "According to the [Source Name], ..."\n\n'
        "CONTEXT:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    # Extract just the documents (drop scores) for the retriever chain
    retrieved_docs = [doc for doc, _ in docs_and_scores]

    # 5. LCEL pipeline
    logging.info("Building LCEL Chain...")
    rag_chain = (
        {
            "context": RunnablePassthrough() | (lambda _: format_docs_with_sources(docs_and_scores)),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # 6. Execute
    logging.info("Generating answer...")
    response = rag_chain.invoke(user_question)

    # 7. Print results
    print("\n" + "=" * 40)
    print(f"🤖 AI ANSWER:\n{response}")
    print("=" * 40)

    # 8. Print source citations for transparency
    print_sources(docs_and_scores)


if __name__ == "__main__":
    query = "How much can I spend setting up my computer setup at home, and when do I need to submit it?"
    run_query(query)
