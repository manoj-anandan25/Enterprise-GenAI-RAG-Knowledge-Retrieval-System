# Enterprise GenAI & RAG Knowledge Retrieval System

## 1. Business Problem

Modern enterprises generate massive volumes of internal knowledge—HR policies, IT troubleshooting guides, compliance directives, and expense regulations—spread across unstructured text files, email threads, and shared drives. Employees routinely lose productivity searching for authoritative answers to routine questions such as:

- "Can I work remotely and for how many days?"
- "I'm locked out of my account—how do I reset my password?"
- "What are the travel expense reimbursement limits?"
- "What must I do if I suspect a data breach?"

Without an intelligent retrieval layer, staff rely on manual keyword searches, outdated intranet pages, or ad-hoc messages, increasing operational overhead, compliance risk, and resolution time. This project addresses the gap by deploying a **Retrieval-Augmented Generation (RAG)** pipeline that indexes enterprise policy documents into a vector database and serves precise, citation-backed answers via a REST API.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Enterprise RAG Pipeline                      │
├─────────────────────────────────────────────────────────────────────┤
│  DATA SOURCE                                                         │
│  data/enterprise_policies.txt  ──►  TextLoader + Chunk Splitter     │
│                                     (chunk_size=500, overlap=50)      │
│                                                                     │
│  EMBEDDING MODEL                                                    │
│  all-MiniLM-L6-v2 (HuggingFaceEmbeddings)  ──►  Vector Encodings    │
│                                                                     │
│  VECTOR STORE                                                       │
│  ChromaDB (persist_directory=chroma_db)  ──►  Collection:         │
│  "enterprise_policies"                                               │
│                                                                     │
│  RETRIEVAL LAYER                                                    │
│  FastAPI  ──►  /query endpoint  ──►  similarity_search_with_score │
│  (k=3 top chunks)                                                   │
│                                                                     │
│  RESPONSE                                                           │
│  { query, answer, sources, similarity_scores }                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| Document Ingest | `langchain` `TextLoader` + `RecursiveCharacterTextSplitter` | Load and segment policy text |
| Embedding Engine | `all-MiniLM-L6-v2` via `langchain-huggingface` | Convert text chunks into dense vectors |
| Vector Database | `ChromaDB` (`chroma`) | Persistent vector storage and similarity search |
| API Server | `FastAPI` + `uvicorn` | REST endpoint for query and retrieval |
| Testing / Validation | `unittest` (`tests/test_rag.py`) | Query simulation, precision measurement, >92% accuracy verification |

---

## 3. Installation Steps

### Prerequisites
- Python 3.9+
- `pip` package manager

### 1. Clone / Navigate to Project
```bash
cd "Enterprise GenAI & RAG Knowledge Retrieval System"
```

### 2. Create Virtual Environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Linux / macOS
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages (from `requirements.txt`):
- `langchain`
- `langchain-community`
- `fastapi`
- `uvicorn`
- `chromadb`
- `sentence-transformers`
- `pydantic`

### 4. Build the Vector Index
```bash
python src/ingest.py
```
Expected output:
```
Ingested 4 chunks into ../chroma_db
```

This creates the `chroma_db` directory with the embedded `enterprise_policies` collection.

---

## 4. API Usage Instructions

### Start the Server
```bash
uvicorn src.main:app --reload --port 8000
```

### `/query` Endpoint Example

**Request:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How many days per week can I work remotely?"}'
```

**Response:**
```json
{
  "query": "How many days per week can I work remotely?",
  "answer": "Employees may work remotely up to 3 days per week with manager approval. All remote workers must maintain a secure, private workspace and use company-provided VPN access for all internal systems. Remote work requests must be submitted at least 5 business days in advance.",
  "sources": ["enterprise_policies"],
  "similarity_scores": [0.2345, 0.4567, 0.6789]
}
```

**Fields Explained:**
- `answer`: Concatenated text of the top 3 retrieved document chunks.
- `sources`: Metadata source identifiers for citation tracking.
- `similarity_scores`: Distance scores from ChromaDB (lower = more similar); useful for precision auditing.

---

## 5. Testing / Validation Results

### Test Suite: `tests/test_rag.py`
The evaluation module covers **7 distinct validation scenarios** (5+ specific query cases plus precision and aggregate checks):

| Test Case | Query Focus | Expected Keywords | Accuracy Threshold |
|-----------|-------------|-------------------|-------------------|
| `test_hr_remote_work_policy` | Remote work rules | `remote`, `3 days`, `manager approval`, `vpn`, `5 business days` | >92% |
| `test_it_password_reset_lockout` | IT troubleshooting / lockout | `password`, `reset`, `90 days`, `12 characters`, `helpdesk` | >92% |
| `test_expense_reimbursement_limits` | Travel claim limits | `$500`, `lodging`, `$150`, `meals`, `$200`, `transportation` | >92% |
| `test_data_security_compliance` | Breach reporting & compliance | `annual data security training`, `CISO`, `1 hour`, `disciplinary action` | >92% |
| `test_it_self_service_portal` | Portal / MFA reset | `self-service`, `portal`, `multi-factor authentication` | >92% |
| `test_retrieval_precision_scores` | Score ordering & positivity | Positive scores, best match = lowest score | — |
| `test_overall_keyword_coverage_above_92_percent` | Aggregate across 4 queries | Multi-keyword coverage | >92% average |

### Execution
```bash
python -m unittest tests/test_rag.py -v
```

### Validation Criteria Met
- **Retrieval Pipeline Integrity:** All 5 specific evaluation queries return at least 1 relevant chunk.
- **Keyword Match Accuracy:** Each test asserts that >92% of expected keywords appear in retrieved content.
- **Precision Metrics:** Similarity scores are positive and ordered correctly (lowest = best match).
- **Aggregate Coverage:** Average keyword match accuracy across the combined query set exceeds 92%.

### Example Output Snippet
```
[HR Remote Work] Match accuracy: 100.00%
[IT Password Reset] Match accuracy: 100.00%
[Expense Limits] Match accuracy: 100.00%
[Data Security] Match accuracy: 100.00%
[Self-Service Portal] Match accuracy: 100.00%
[Overall Coverage] Average keyword match accuracy: 100.00%
```

---

## 6. Project Structure

```
Enterprise GenAI & RAG Knowledge Retrieval System/
├── data/
│   └── enterprise_policies.txt          # Source policy document
├── src/
│   ├── ingest.py                        # Chunking + embedding + ChromaDB ingestion
│   └── main.py                          # FastAPI server with /query endpoint
├── tests/
│   └── test_rag.py                      # 7-scenario evaluation suite
├── chroma_db/                           # Persisted vector store (generated)
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

---

## 7. Maintenance & Next Steps

- **Scale:** Replace `all-MiniLM-L6-v2` with a larger domain-tuned embedding model for higher retrieval fidelity.
- **Security:** Add authentication middleware (`fastapi` `OAuth2`, `API-Key`) before production deployment.
- **Observability:** Integrate logging of similarity scores and query latency for continuous monitoring.
- **Expansion:** Add additional policy files (`data/*.txt`) and re-run `python src/ingest.py` to grow the knowledge base.

---

---

## Actual Execution & Test Results

### Ingestion Script (`src/ingest.py`)
Executed with the local virtual environment (`venv`):

```bash
$ venv\Scripts\python src\ingest.py
```

**Exact terminal output:**
```
D:\deveploment\Enterprise GenAI & RAG Knowledge Retrieval System\src\ingest.py:2: DeprecationWarning: `langchain-community` is being sunset and is no longer actively maintained. See https://github.com/langchain-ai/langchain-community/issues/674 for details and migration guidance toward standalone integration packages.
  from langchain_community.document_loaders import TextLoader
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
D:\deveploment\Enterprise GenAI & RAG Knowledge Retrieval System\venv\lib\site-packages\huggingface_hub\file_download.py:149: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\MANOJ PILLAI\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
  warnings.warn(message)
D:\deveploment\Enterprise GenAI & RAG Knowledge Retrieval System\venv\lib\site-packages\huggingface_hub\file_download.py:149: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\MANOJ PILLAI\.cache\huggingface\hub. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
  warnings.warn(message)
Loading weights: 100%|##########| 103/103 [00:00<00:00, 1782.24it/s]
D:\deveploment\Enterprise GenAI & RAG Knowledge Retrieval System\src\ingest.py:28: LangChainDeprecationWarning: Since Chroma 0.4.x the manual persistence method is no longer supported as docs are automatically persisted.
  db.persist()
Ingested 4 chunks into D:\deveploment\Enterprise GenAI & RAG Knowledge Retrieval System\chroma_db
```

**Evaluation metrics:**
- Chunks ingested: **4**
- Embedding model: `all-MiniLM-L6-v2` (103 weights loaded)
- Persist directory: `chroma_db`
- Collection: `enterprise_policies`

---

### Test Suite (`tests/test_rag.py`)
Executed with `python -m unittest`:

```bash
$ venv\Scripts\python -m unittest tests\test_rag.py -v
```

**Exact terminal output:**
```
test_data_security_compliance (tests.test_rag.TestRAGPipeline) ... ok
test_expense_reimbursement_limits (tests.test_rag.TestRAGPipeline) ... ok
test_hr_remote_work_policy (tests.test_rag.TestRAGPipeline) ... ok
test_it_password_reset_lockout (tests.test_rag.TestRAGPipeline) ... ok
test_it_self_service_portal (tests.test_rag.TestRAGPipeline) ... ok
test_overall_keyword_coverage_above_92_percent (tests.test_rag.TestRAGPipeline)
Aggregate check across all test queries ensuring >92% coverage. ... ok
test_retrieval_precision_scores (tests.test_rag.TestRAGPipeline) ... ok
----------------------------------------------------------------------
Ran 7 tests in 40.694s
OK
[Data Security] Match accuracy: 100.00%
[Expense Limits] Match accuracy: 100.00%
[HR Remote Work] Match accuracy: 100.00%
[IT Password Reset] Match accuracy: 100.00%
[Self-Service Portal] Match accuracy: 100.00%
[Overall Coverage] Average keyword match accuracy: 100.00%
[Precision] Similarity scores: [0.8504396677017212, 1.5217065811157227, 1.7247037887573242]
```

**Evaluation metrics captured:**
- Total test cases executed: **7**
- Results: **7 passed, 0 failed** (`OK` in 40.694s)
- Retrieval precision scores (top-3 similarity): `[0.8504, 1.5217, 1.7247]`
- Keyword match accuracy per query:
  - HR Remote Work: **100.00%**
  - IT Password Reset: **100.00%**
  - Expense Limits: **100.00%**
  - Data Security: **100.00%**
  - Self-Service Portal: **100.00%**
- Overall aggregate keyword coverage: **100.00%** (threshold >92%)
- Retrieval integrity: All 5 targeted queries return ≥1 relevant chunk; scores are positive and ordered (lowest = best match).

**Notes before GitHub push:**
- `chroma_db/` is generated (not committed — add to `.gitignore` if needed).
- `venv/` is excluded from source control.
- Minor code updates applied to `src/ingest.py`: switched import to `langchain_text_splitters` and resolved relative paths for robust local execution.

---

*Built with FastAPI, LangChain, ChromaDB, and HuggingFace Embeddings.*
