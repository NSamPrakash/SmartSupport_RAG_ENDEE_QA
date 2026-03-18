# SmartSupport — RAG-Powered Support Q&A System

> **Retrieval-Augmented Generation (RAG) using the [Endee](https://github.com/endee-io/endee) vector database and Google Gemini LLM**

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![Endee](https://img.shields.io/badge/Vector%20DB-Endee-orange)](https://endee.io)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-blue?logo=google)](https://ai.google/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

##  Table of Contents

- [Problem Statement](#-problem-statement)
- [System Design](#-system-design)
- [How Endee Is Used](#-how-endee-is-used)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Usage & Workflow](#-usage--workflow)
- [Web Interface](#-web-interface)
- [API Reference](#-api-reference)
- [Configuration](#-configuration)
- [Example Walkthrough](#-example-walkthrough)
- [Documentation](#-documentation)

---

##  Problem Statement

Knowledge workers spend enormous time searching through large document collections for specific answers.  
Traditional keyword search fails to capture *semantic meaning* — a query for "climate solutions" won't match a passage about "carbon-reduction strategies" even though they mean the same thing.

**SmartSupport solves this with a RAG pipeline:**

1. Documents are chunked, embedded into dense vector representations, and stored in **Endee** for sub-millisecond similarity search.
2. When a user asks a question, the most semantically relevant chunks are retrieved from Endee.
3. An LLM uses only those retrieved chunks to generate a **grounded, cited answer** — eliminating hallucination.

---

## System Design

```
┌───────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                     │
│                                                               │
│  Raw Text / File                                              │
│       │                                                       │
│       ▼                                                       │
│  ┌─────────────┐    ┌───────────────────┐    ┌────────────┐  │
│  │  Text       │    │  Sentence         │    │   Endee    │  │
│  │  Chunker    │───▶│  Transformer      │───▶│  Vector DB │  │
│  │  (512 char, │    │  (all-MiniLM-L6)  │    │  (cosine,  │  │
│  │   64 ovlap) │    │  384-dim vectors  │    │   INT8)    │  │
│  └─────────────┘    └───────────────────┘    └────────────┘  │
└───────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────┐
│                        QUERY PIPELINE                         │
│                                                               │
│  User Question                                                │
│       │                                                       │
│       ▼                                                       │
│  ┌──────────────┐    ┌────────────┐    ┌──────────────────┐  │
│  │  Embedding   │    │   Endee    │    │   LLM (GPT-4o)   │  │
│  │  Model       │───▶│  ANN Search│───▶│  + Context       │  │
│  │  (same model │    │  top-k=5   │    │  → Grounded      │  │
│  │   as ingest) │    │  chunks    │    │    Answer        │  │
│  └──────────────┘    └────────────┘    └──────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | Fast, accurate, 384-dim — ideal balance for local inference |
| Similarity metric | Cosine | Normalised embeddings → cosine = dot product; direction > magnitude |
| Precision | INT8 | Endee INT8 quantisation gives ~4× memory savings with negligible accuracy loss |
| Chunk size | 512 chars / 64 overlap | Preserves sentence context; overlap prevents boundary misses |
| LLM | Google Gemini (optional) | Cost-effective, powerful instruction following; fallback to extractive mode |

---

##  How Endee Is Used

Endee is the **core vector storage and retrieval engine** of SmartSupport. Every interaction with the knowledge base goes through Endee.

### 1. Index Creation

```python
from endee import Endee, Precision

client = Endee()          # connects to localhost:8080
client.create_index(
    name="documind_chunks",
    dimension=384,          # matches all-MiniLM-L6-v2 output
    space_type="cosine",    # semantic similarity metric
    precision=Precision.INT8,  # memory-efficient quantisation
)
```

### 2. Upserting Document Vectors

Each chunk is stored with its embedding and full metadata (text, filename, doc_id):

```python
index = client.get_index("documind_chunks")
index.upsert([
    {
        "id":     "abc123_c0001",
        "vector": [0.12, -0.45, ...],   # 384-dim float list
        "meta":   {
            "doc_id":   "abc123",
            "filename": "climate_report.txt",
            "text":     "The Paris Agreement commits nations to…",
            "chunk_idx": 1,
        },
    }
])
```

### 3. Semantic Search

The query is embedded with the same model and sent to Endee's ANN index:

```python
query_vector = embedder.encode("What is the Paris Agreement?").tolist()
results = index.query(vector=query_vector, top_k=5)
# → returns list of {id, similarity, meta} sorted by cosine similarity
```

### Why Endee?

- **High-performance HNSW indexing** — sub-millisecond ANN queries at scale
- **Simple REST / SDK API** — no complex configuration required
- **Docker-ready** — single `docker compose up` to start
- **Up to 1B vectors on a single node** — production-grade scalability
- **INT8 precision support** — 4× memory reduction with minimal quality loss

---

## Project Structure

```
SmartSupport_RAG_ENDEE_QA/
├── app.py                           # FastAPI REST API server
├── chat.html                        # Web interface for Q&A
├── build_knowledge_base.py          # CLI tool for uploading custom documents
├── demo.py                          # Demo with sample ML documents
├── requirements.txt
├── Dockerfile
├── docker-compose.yml               # Runs Endee + API
├── SCREEN_RECORDING_SPEECH.md       # Complete presentation script
├── DOCUMENT_UPLOAD_GUIDE.md         # User guide for document uploads
├── README.md                        # This file
├── src/
│   ├── __init__.py
│   ├── rag_engine.py                # Core: chunking, embedding, Endee search
│   └── qa_pipeline.py               # QA: Gemini LLM + context building
├── tests/
│   └── test_documind.py             # Unit tests
├── endee/                           # Endee vector database (git submodule)
├── __pycache__/
└── .env                             # Configuration (API keys, hosts)
```

---

##  Quick Start

### Prerequisites

- Python 3.11+
- Docker + Docker Compose
- Google Gemini API key (optional, for generative mode)

### 1. Clone & Setup

```bash
git clone https://github.com/NSamPrakash/SmartSupport_RAG_ENDEE_QA.git
cd SmartSupport_RAG_ENDEE_QA

python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start Endee Vector Database

```bash
docker-compose up -d
# Endee runs on http://localhost:8080
```

### 3. (Optional) Configure Gemini API

```bash
# Create .env file with your Gemini API key
echo GEMINI_API_KEY=your_api_key_here >> .env
echo GEMINI_MODEL=gemini-2.5-flash >> .env
```

Without an API key, the system runs in **extractive mode** (still works great!).

### 4. Choose Your Workflow

**Option A: Upload Sample Documents (Demo)**
```bash
python demo.py
# Ingests 3 sample ML documents and shows Q&A examples
```

**Option B: Upload Your Own Documents**
```bash
python build_knowledge_base.py
# Interactive menu to upload your custom documents
# See DOCUMENT_UPLOAD_GUIDE.md for detailed instructions
```

**Option C: Start the Web Server**
```bash
python app.py
# Open http://localhost:8000 in your browser
# Use chat.html interface to upload documents and ask questions
```

---

##  Usage & Workflow

### Workflow 1: Web Interface (Easiest for Non-Technical Users)

```
1. Open http://localhost:8000 in browser
2. Upload .txt documents through the web form
3. Ask questions in the chat interface
4. Get instant answers with source citations
```

### Workflow 2: Command-Line (Batch Processing)

```bash
# Upload multiple documents from a folder
python build_knowledge_base.py --folder ./my_documents

# Ask questions programmatically
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?", "top_k": 5}'
```

### Workflow 3: Python API (Programmatic)

```python
from src.rag_engine import RAGEngine
from src.qa_pipeline import QAPipeline

# Initialize
rag = RAGEngine()
qa = QAPipeline(rag_engine=rag)

# Ingest documents
with open("document.txt") as f:
    rag.ingest_text(f.read(), filename="document.txt")

# Ask questions
result = qa.ask("What is this document about?")
print(result["answer"])
print(result["sources"])
print(result["mode"])  # 'generative' or 'extractive'
```

---

##  Web Interface

The `chat.html` file provides a beautiful, user-friendly interface:

**Features:**
- 📤 Drag-and-drop document upload
- 💬 Chat-like Q&A interface
- 🔗 Source citations with similarity scores
- 🎯 Document management
- 🌙 Dark/light theme support

**Access:**
- Run the API: `python app.py`
- Open: `http://localhost:8000` in your browser

---

##  API Reference

### `POST /ingest/text`

Ingest raw text into the knowledge base.

```json
{
  "text": "The Great Wall of China is one of the greatest wonders…",
  "filename": "great_wall.txt"
}
```

Response:
```json
{
  "status": "ingested",
  "doc_id": "a1b2c3d4e5f6",
  "filename": "great_wall.txt",
  "num_chunks": 3
}
```

### `POST /ingest/file`

Upload a `.txt` file:

```bash
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@my_document.txt"
```

### `POST /ask`

Ask a question and get a grounded answer:

```json
{
  "question": "What is the Great Wall of China made of?",
  "top_k": 5
}
```

Response:
```json
{
  "question": "What is the Great Wall of China made of?",
  "answer": "The Great Wall was built from stone, brick, tamped earth…",
  "sources": [
    {
      "filename": "great_wall.txt",
      "chunk_id": "a1b2c3_c0001",
      "similarity": 0.9412,
      "excerpt": "The Great Wall of China is one of…"
    }
  ],
  "mode": "generative"
}
```

### `POST /search`

Raw semantic search (no LLM generation):

```json
{
  "query": "renewable energy sources",
  "top_k": 3
}
```

### `GET /health`

Service health check.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

The test suite covers:
- Text chunking (edge cases, overlap, whitespace)
- Document ingestion (chunk count, Endee upsert called)
- Semantic search (result parsing, empty results)
- QA pipeline (extractive mode, no-results graceful handling)

All tests use mocks for Endee and the embedding model — no live server required.

---

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `ENDEE_HOST` | `http://localhost:8080` | Endee server URL |
| `ENDEE_AUTH_TOKEN` | `""` | Endee auth token (leave blank for no auth) |
| `GEMINI_API_KEY` | `""` | Google Gemini API key (enables generative mode) |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model to use |

**Example .env file:**
```
ENDEE_HOST=http://localhost:8080
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

---

---

## 🔄 Example Walkthrough

### Using the Python API

```python
from src.rag_engine import RAGEngine
from src.qa_pipeline import QAPipeline

# 1. Start Endee: docker-compose up -d

# 2. Initialize RAG engine and QA pipeline
rag = RAGEngine()
qa = QAPipeline(rag_engine=rag)

# 3. Ingest a document
result = rag.ingest_text(
    text="Machine Learning is a subset of AI that enables systems to learn from data...",
    filename="ml_guide.txt"
)
print(f"Ingested: {result['num_chunks']} chunks")
# Output: Ingested: 2 chunks

# 4. Ask questions
answer = qa.ask("What is Machine Learning?")
print(answer["answer"])
# Output: Machine Learning is a subset of AI that enables systems to learn...
print(answer["mode"])
# Output: generative (if Gemini API key is set) or extractive
print(answer["sources"][0])
# Output: {filename: ml_guide.txt, similarity: 0.98, ...}
```

### Using the Command-Line Tool

```bash
# Build knowledge base from custom documents
python build_knowledge_base.py

# OR upload all docs from a folder
python build_knowledge_base.py --folder ./my_documents

# OR upload specific files
python build_knowledge_base.py --file doc1.txt --file doc2.txt
```

### Using the Web Interface

```bash
# Start the API server
python app.py
# Open http://localhost:8000 in your browser
# Upload documents and ask questions through chat.html
```

---

##  Documentation

- **[DOCUMENT_UPLOAD_GUIDE.md](DOCUMENT_UPLOAD_GUIDE.md)** — Complete guide for uploading custom documents
   
## 📄 License

MIT — see [LICENSE](LICENSE)
