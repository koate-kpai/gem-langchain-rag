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
import time
import logging
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from chromadb.errors import NotFoundError  # to catch missing DB/collection

from config import config  # centralized 12-factor config (see config.py)
from retry import call_with_retry  # exponential backoff for API resilience

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
    if _verbose:
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
# Runtime state (for REPL commands)
# ---------------------------------------------------------------------------
_verbose: bool = "--verbose" in sys.argv or "-v" in sys.argv


# ---------------------------------------------------------------------------
# Main query pipeline
# ---------------------------------------------------------------------------
def run_query(user_question: str) -> float | None:
    """Execute a single RAG query and return total elapsed time in seconds.

    This function is called both from the REPL loop and from batch scripts.
    It handles all pre-flight checks, retrieval, generation, and display.

    The streaming output (llm.stream) shows tokens incrementally — a critical
    UX pattern for production RAG systems. Without streaming, users wait
    silently for 2-10 seconds before seeing any response. With streaming,
    the first token appears in ~200ms, providing immediate feedback.
    """
    t_start = time.perf_counter()
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
        streaming=True,  # enables token-by-token streaming
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
        return None

    # 2. Guardrail: Ensure the collection actually contains data
    if vectorstore._collection.count() == 0:
        print("\n❌ Error: The ChromaDB database is empty.")
        print("Please run the ingestion script to populate it with documents.\n")
        return None

    # 3. Retrieve documents WITH relevance scores
    t_retrieval_start = time.perf_counter()

    # Wrap the retrieval in retry logic — transient API failures during
    # embedding should not crash the pipeline. The retry module handles
    # exponential backoff for 429 (rate limit) and 5xx (server error).
    if config.retrieval.search_type == "mmr":
        _search_func = lambda q: vectorstore.max_marginal_relevance_search(
            q,
            k=config.retrieval.retriever_k,
            fetch_k=config.retrieval.mmr_fetch_k,
            lambda_mult=config.retrieval.mmr_lambda_mult,
        )
    else:
        _search_func = lambda q: vectorstore.similarity_search_with_relevance_scores(
            q,
            k=config.retrieval.retriever_k,
        )
    docs_and_scores = call_with_retry(_search_func, user_question)

    # For MMR, max_marginal_relevance_search returns docs only (no scores).
    # We assign a neutral score of 0.5 for display consistency.
    if config.retrieval.search_type == "mmr":
        docs_and_scores = [(doc, 0.5) for doc in docs_and_scores]
        logging.info(
            "MMR retrieval: k=%d, fetch_k=%d, lambda_mult=%.1f",
            config.retrieval.retriever_k,
            config.retrieval.mmr_fetch_k,
            config.retrieval.mmr_lambda_mult,
        )

    t_retrieval_elapsed = time.perf_counter() - t_retrieval_start

    if not docs_and_scores:
        print("\n❌ No relevant documents found for your query.")
        return None

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

    # 5. LCEL pipeline with streaming
    logging.info("Building LCEL Chain...")
    rag_chain = (
        {
            "context": RunnablePassthrough()
            | (lambda _: format_docs_with_sources(docs_and_scores)),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # 6. Execute with streaming
    t_gen_start = time.perf_counter()
    logging.info("Generating answer via streaming...")

    print("\n" + "=" * 40)
    print("🤖 AI ANSWER:")
    print("-" * 40)

    response_parts: list[str] = []
    # Wrap the streaming invocation in retry logic
    stream_iter = call_with_retry(lambda: rag_chain.stream(user_question))
    for chunk in stream_iter:
        print(chunk, end="", flush=True)
        response_parts.append(chunk)

    t_gen_elapsed = time.perf_counter() - t_gen_start
    t_total = time.perf_counter() - t_start

    full_response = "".join(response_parts)
    tokens = len(full_response.split())
    tokens_per_sec = tokens / t_gen_elapsed if t_gen_elapsed > 0 else 0

    print()  # newline after streaming
    print("=" * 40)

    # 7. Print timing summary
    print(f"\n⚡ Timing: retrieval={t_retrieval_elapsed:.2f}s | "
          f"generation={t_gen_elapsed:.2f}s ({tokens_per_sec:.0f} tok/s) | "
          f"total={t_total:.2f}s")

    # 8. Print source citations for transparency
    print_sources(docs_and_scores)

    return t_total


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------
def repl() -> None:
    """Interactive read-eval-print loop for querying the RAG pipeline.

    Commands:
      /quit, /exit, /q  — exit the REPL
      /verbose, /v      — toggle verbose chunk dump mode
      /help, /?         — show this help message

    The REPL persists the vector store connection across queries (warm start),
    avoiding the overhead of re-initializing embeddings and ChromaDB on every
    question. This is a common production pattern for chat interfaces.
    """
    global _verbose

    print("\n" + "=" * 50)
    print("🔍 Enterprise RAG Pipeline — Interactive Mode")
    print("=" * 50)
    print("Type your question or a command:")
    print("  /quit  — exit")
    print("  /v     — toggle verbose mode")
    print("  /help  — show commands")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n❓ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            cmd = user_input.lower().split()[0]
            if cmd in ("/quit", "/exit", "/q"):
                print("Goodbye!")
                break
            elif cmd in ("/verbose", "/v"):
                _verbose = not _verbose
                print(f"Verbose mode: {'ON' if _verbose else 'OFF'}")
                continue
            elif cmd in ("/help", "/?"):
                print("Commands: /quit, /verbose, /help")
                continue
            else:
                print(f"Unknown command: {cmd}. Type /help for commands.")
                continue

        # Run the query
        run_query(user_input)


if __name__ == "__main__":
    # If a query is passed as a CLI argument, run it once and exit (batch mode)
    # Otherwise, launch the interactive REPL
    cli_args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if cli_args:
        run_query(" ".join(cli_args))
    else:
        repl()
