# search.py

import logging
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from chromadb.errors import NotFoundError  # to catch missing DB/collection

load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma_langchain_db"
COLLECTION_NAME = "policy_docs"


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def run_query(user_question):
    logging.info("Initializing vector store retriever...")

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
