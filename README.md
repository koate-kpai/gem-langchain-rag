# Enterprise RAG Pipeline: Corporate Policy Engine

A robust, local Retrieval-Augmented Generation (RAG) pipeline built to securely ingest, chunk, and query corporate documents using the modern LangChain Expression Language (LCEL) architecture.

---

## 📈 Evolution: From MVP to Production-Ready

This repository documents the **progressive enhancement** of a basic RAG proof-of-concept into a production-grade system. Each commit addresses a specific architectural gap in the original implementation, demonstrating senior-level engineering practices across **security**, **cost optimization**, **resilience**, **testability**, and **responsible AI**.

| # | Commit | Senior Engineering Signal |
|---|--------|--------------------------|
| 1 | `security: audit secrets, harden gitignore, and add defense-in-depth guardrails` | **DevSecOps** — 4-layer defense (gitignore → `.env.example` → `SECURITY.md` → runtime validation). API key format verification before any billable call. OWASP fail-securely pattern. |
| 2 | `refactor: extract configuration into centralized dataclass with cost-aware defaults` | **12-Factor App** — every parameter documented with cost/performance/compliance tradeoffs. Frozen dataclasses prevent runtime mutation. Single source of truth for all tuning. |
| 3 | `feat: support multi-format ingestion with extensible loader registry` | **SOLID Design** — Open/Closed Principle via loader registry. Adding `.docx` is a 2-line change, zero edits to `ingest.py`. File validation before any API call (cost awareness). |
| 4 | `feat: idempotent incremental ingestion with content-addressed deduplication` | **Data Engineering** — SHA-256 content addressing (same pattern as Git/Docker/IPFS). Zero re-embedding cost on re-runs. `--force-reindex` flag for intentional rebuilds. |
| 5 | `feat: source-grounded responses with relevance scoring and citation metadata` | **Responsible AI** — every answer cites its source with cosine similarity scores. Users can verify against original documents. Transparent vs black-box AI. |
| 6 | `feat: interactive REPL with streaming output and MMR diversity retrieval` | **UX Maturity** — token-by-token streaming (200ms first token vs 10s silence). MMR retrieval prevents redundant context. Timing instrumentation for cost monitoring. |
| 7 | `feat: resilient API calls with exponential backoff, rate-limit handling, and structured logging` | **Resilience Engineering** — exponential backoff with jitter thwarts thundering herd. Discriminates between retryable (429/5xx) and fatal (401/400/404) errors. |
| 8 | `test: comprehensive test suite mocking external APIs with 68 tests across 5 modules` | **Testing Strategy** — 68 tests, zero API cost (all mocked), 13s runtime. Guardrails, retry logic, chunk ID determinism, and format validation all verified. |

### Original MVP Limitations

The initial implementation (`5709b2c` — `1a664b8`) was a functional proof-of-concept with significant technical debt:

- ❌ Hardcoded API keys and file paths
- ❌ Single `.txt` file support only
- ❌ Destructive re-ingestion (delete + rebuild every run)
- ❌ No error handling for API failures
- ❌ Black-box answers with no source attribution
- ❌ Batch-only output (no streaming)
- ❌ Zero test coverage

### Production-Ready Result

After 8 incremental commits:

- ✅ Defense-in-depth security with 4 control layers
- ✅ 12-factor configuration with documented cost tradeoffs
- ✅ Multi-format ingestion (.txt, .md, .pdf) via extensible registry
- ✅ Idempotent re-ingestion with content-addressed dedup (zero wasted spend)
- ✅ Source-cited answers with relevance scores and verbose debug mode
- ✅ Interactive REPL with streaming, MMR diversity, and timing instrumentation
- ✅ Exponential backoff retry with error classification
- ✅ 68 passing tests with zero external API dependencies

---



## 🧠 Architecture Overview

This system allows users to ask complex, natural language questions about internal company policies. It mitigates hallucination by enforcing strict context boundaries—if the answer is not in the provided documents, the model will gracefully decline to answer.

### Core Stack
* **Orchestration:** LangChain (v0.2+ LCEL Framework)
* **Vector Database:** ChromaDB (Persistent Local Storage)
* **Embeddings:** OpenAI `text-embedding-3-small` (1,536 dimensions)
* **LLM:** OpenAI `gpt-4o-mini` (94% cheaper than gpt-4o, sufficient for RAG)
* **Resilience:** `tenacity` (exponential backoff with jitter)
* **Testing:** `pytest` with mocked OpenAI (zero-cost CI)
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
 ┣ 📂 data/                          # Source documents (.txt, .md, .pdf)
 ┃ ┗ 📜 policy.txt                   # Sample corporate policy
 ┣ 📂 chroma_langchain_db/           # Local vector database (Git-ignored)
 ┣ 📂 tests/                         # 68 tests, zero API cost (all mocked)
 ┃ ┣ 📜 conftest.py                  # Shared fixtures & OpenAI mocks
 ┃ ┣ 📜 test_config.py               # Configuration validation (15 tests)
 ┃ ┣ 📜 test_ingest.py               # Ingestion pipeline (11 tests)
 ┃ ┣ 📜 test_loaders.py              # Loader registry (17 tests)
 ┃ ┣ 📜 test_retry.py                # Resilience patterns (13 tests)
 ┃ ┗ 📜 test_search.py               # Retrieval pipeline (12 tests)
 ┣ 📜 config.py                      # 12-factor config dataclass
 ┣ 📜 .env.example                   # Environment template with security notes
 ┣ 📜 SECURITY.md                    # Security policy & incident response
 ┣ 📜 ingest.py                      # Document loading & embedding pipeline
 ┣ 📜 loaders.py                     # Extensible loader registry (.txt, .md, .pdf)
 ┣ 📜 search.py                      # LCEL retrieval & generation pipeline
 ┣ 📜 retry.py                       # Exponential backoff & error classification
 ┣ 📜 .env                           # API keys (Git-ignored)
 ┣ 📜 .gitignore                     # 28-entry hardened ignore patterns
 ┗ 📜 README.md                      # Project documentation
```
