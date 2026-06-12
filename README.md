# Enterprise RAG Pipeline: Corporate Policy Engine

A robust, local Retrieval-Augmented Generation (RAG) pipeline built to securely ingest, chunk, and query corporate documents using the modern LangChain Expression Language (LCEL) architecture.

## 🧠 Architecture Overview

This system allows users to ask complex, natural language questions about internal company policies. It mitigates hallucination by enforcing strict context boundaries—if the answer is not in the provided documents, the model will gracefully decline to answer.

### Core Stack
* **Orchestration:** LangChain (v0.2+ LCEL Framework)
* **Vector Database:** ChromaDB (Persistent Local Storage)
* **Embeddings:** OpenAI `text-embedding-3-small` (1,536 dimensions)
* **LLM:** OpenAI `gpt-4o-mini`
* **Environment:** Python 3.11+

## ✨ Key Features

* **Automated Ingestion (`ingest.py`):** Intelligently chunks text documents using `RecursiveCharacterTextSplitter` to preserve paragraph and sentence context boundaries.
* **Semantic Retrieval (`search.py`):** Bypasses keyword-matching in favor of geometric cosine-similarity search to find the underlying meaning of user queries.
* **Future-Proof LCEL Routing:** Utilizes modern LangChain pipe routing (`|`) for maximum readability, stability, and speed, completely avoiding legacy "black box" chain modules.
* **Bulletproof Guardrails:** Features comprehensive error handling for missing databases, empty collections, and missing environment variables to prevent runtime crashes.

## 🚀 Getting Started

### 1. Clone & Environment Setup

```powershell
git clone https://github.com/koate-kpai/gem-langchain-rag.git
cd gem-langchain-rag

# Create and activate a virtual environment

python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install langchain langchain-openai langchain-chroma chromadb python-dotenv
```

### 2. Environment Variables
Create a .env file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY="sk-your-api-key-here"
```
### 3. Usage
#### Step A: Populate the Vector Database
Place your source documents in the data/ folder (a sample policy.txt is provided). Run the ingestion script to chunk the text and generate embeddings:

```powerShell
python ingest.py
```
#### Step B: Query the Engine
Once the database is populated, execute the search script to run the LCEL pipeline:

```powerShell
python search.py
```
### 📂 Project Structure
```
📦 gem-langchain-rag
 ┣ 📂 data/
 ┃ ┗ 📜 policy.txt              # Raw source documents
 ┣ 📂 chroma_langchain_db/      # Local vector database (Git-ignored)
 ┣ 📜 ingest.py                 # Document loading & embedding script
 ┣ 📜 search.py                 # LCEL retrieval & generation script
 ┣ 📜 .env                      # API keys (Git-ignored)
 ┣ 📜 .gitignore                # Security guardrails
 ┗ 📜 README.md                 # Project documentation
```
