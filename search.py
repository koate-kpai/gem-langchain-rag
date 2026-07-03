# search.py
"""
Semantic retrieval pipeline: embeds a user question, retrieves relevant chunks,
and generates a grounded answer using an LLM.

Security & Operations Philosophy
--------------------------------
- Defense-in-depth: API key validation before any external call
- Cost-aware retrieval: k=2 limits context token usage ($0.0015/query at gpt-4o-mini pricing)
- Fail gracefully: guardrails for missing database, empty collection, invalid config
- Source-grounded: future commits add relevance scoring and citation metadata
"""
import os
import re
import logging
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from chromadb.errors import NotFoundError  # to catch missing DB/collection

# Load .env — fail immediately if missing (don't let OpenAI return a cryptic 401)
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma_langchain_db"
COLLECTION_NAME = "policy_docs"


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


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def run_query(user_question):
    logging.info("Initializing vector store retriever...")

    # --- Pre-flight checks ---
    # Fail fast: validate environment before any external API call
    if not env_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {env_path}\n"
            "Copy .env.example to .env and add your OpenAI API key:\n\n"
            "    copy .env.example .env"
        )
    _validate_api_key(os.getenv("OPENAI_API_KEY"))

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    # 1. Connect to ChromaDB – handle missing database/collection
    try:
        vectorstore = Chroma(
            persist_directory=str(DB_DIR),
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
    except NotFoundError:
        print("\n❌ Error: The ChromaDB collection was not found.")
        print(
            "Please run the ingestion script first to create and populate the database.\n"
        )
        return

    # 2. Guardrail: Ensure the collection actually contains data
    #    (counts documents, works even if collection exists but is empty)
    if vectorstore._collection.count() == 0:
        print("\n❌ Error: The ChromaDB database is empty.")
        print("Please run the ingestion script to populate it with documents.\n")
        return

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 3. Build prompt
    system_prompt = (
        "You are a helpful HR assistant.\n"
        "Answer the user's question using ONLY the provided context below. "
        "If you do not know the answer based strictly on the context, say "
        "'I cannot find that in the company policy manual.'\n\n"
        "CONTEXT:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    # 4. LCEL pipeline
    logging.info("Building LCEL Chain...")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 5. Execute
    logging.info("Executing pipeline for query: '%s'", user_question)
    response = rag_chain.invoke(user_question)

    print("\n" + "=" * 40)
    print(f"🤖 AI ANSWER:\n{response}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    query = "How much can I spend setting up my computer setup at home, and when do I need to submit it?"
    run_query(query)
