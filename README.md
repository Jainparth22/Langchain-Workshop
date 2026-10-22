<div align="center">

# 🔍 FinSight AI
### Enterprise Financial Intelligence Platform

**Multi-Agent RAG System · LangGraph Orchestration · Hybrid Retrieval · Production-Grade**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=for-the-badge)](https://python.langchain.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-6B4FBB?style=for-the-badge)](https://langchain-ai.github.io/langgraph)
[![Groq](https://img.shields.io/badge/Groq-Primary_LLM-F55036?style=for-the-badge)](https://groq.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Store-0468D7?style=for-the-badge)](https://github.com/facebookresearch/faiss)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![RAGAS](https://img.shields.io/badge/RAGAS-Evaluation-FF6B6B?style=for-the-badge)](https://ragas.io)

*A production-grade Retrieval-Augmented Generation system for financial document intelligence. Built with a multi-agent LangGraph architecture, hybrid BM25+FAISS retrieval, cross-encoder reranking, and automated quality evaluation.*

[Architecture](#-system-architecture) · [Quick Start](#-quick-start) · [API Reference](#-api-reference) · [Design Decisions](#-design-decisions--interview-guide) · [Evaluation](#-quality-evaluation-ragas)

</div>

---

## 🎯 What FinSight AI Does

FinSight AI answers complex financial questions by intelligently searching indexed financial documents (annual reports, SEC filings, earnings transcripts, policy documents). Unlike a simple chatbot:

- 🤖 **Three specialised AI agents** collaborate: a **Researcher** retrieves evidence, a **Responder** synthesises answers, and a **Critic** validates for hallucinations — before the answer reaches you.
- 🔍 **Hybrid search** combines BM25 keyword matching (for exact financial figures) with dense vector search (for semantic meaning), fused via Reciprocal Rank Fusion.
- 🎯 **Cross-encoder reranking** applies a second-pass relevance check to ensure only the most precise context chunks reach the LLM.
- 🛡️ **Dual guardrails** — input guard (prompt injection, PII detection) + output guard (hallucination scoring, regulatory disclaimer enforcement).
- 📊 **Automated RAGAS evaluation** tracks faithfulness, answer relevancy, context precision, and context recall.
- 💾 **Conversation memory** with SQLite persistence enables multi-turn financial dialogue.
- ⚡ **Multi-LLM fallback** (Groq → OpenAI → Ollama) ensures 99.9% availability.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                            │
│                                                                   │
│  Documents (PDF/DOCX/CSV/TXT/MD)                                  │
│       │                                                           │
│       ▼                                                           │
│  ┌──────────┐    ┌────────────────────┐    ┌───────────────────┐ │
│  │  Loaders │───▶│  Adaptive Chunker  │───▶│  Dual Indexing    │ │
│  │ (multi-  │    │ • Narrative: 512t  │    │ ┌───────────────┐ │ │
│  │  format) │    │ • Tabular: 256t    │    │ │ FAISS (Dense) │ │ │
│  └──────────┘    │ • 64t overlap      │    │ └───────────────┘ │ │
│                  └────────────────────┘    │ ┌───────────────┐ │ │
│                                            │ │ BM25 (Sparse) │ │ │
│                                            │ └───────────────┘ │ │
│                                            └───────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  LANGGRAPH MULTI-AGENT PIPELINE                  │
│                                                                   │
│  User Query                                                       │
│       │                                                           │
│       ▼                                                           │
│  ┌──────────────┐                                                │
│  │  Input Guard  │ ← Prompt injection, PII, length checks        │
│  └──────┬───────┘                                                │
│         ▼                                                        │
│  ┌──────────────┐                                                │
│  │    Router    │ ← Classifies query: Tier 1/2/3 complexity      │
│  └──────┬───────┘                                                │
│         ▼                                                        │
│  ┌──────────────┐   Hybrid Search (BM25 + FAISS + RRF)          │
│  │  Researcher  │──────────────────────────────────────▶         │
│  │    Agent     │   Cross-Encoder Reranking                      │
│  └──────┬───────┘◀──────────────────────────────────────        │
│         ▼                                                        │
│  ┌──────────────┐                                                │
│  │  Responder   │ ← Grounded synthesis with source citations     │
│  │    Agent     │                                                │
│  └──────┬───────┘                                                │
│         ▼                                                        │
│  ┌──────────────┐                                                │
│  │    Critic    │ ← Faithfulness + hallucination scoring         │
│  │    Agent     │                                                │
│  └──────┬───────┘                                                │
│         │                                                        │
│    ┌────┴─────────────────┐                                      │
│    │                      │                                      │
│  PASS/FAIL          NEEDS_REVISION (max 2 loops)                 │
│    │                      │                                      │
│    │              Back to Researcher with                        │
│    │              specific revision instructions                  │
│    ▼                                                             │
│  ┌──────────────┐                                                │
│  │ Output Guard │ ← Faithfulness threshold, disclaimer           │
│  └──────┬───────┘                                                │
│         ▼                                                        │
│    Final Response + Sources + Scores + Latency                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
finsight-ai/
├── app/
│   ├── api/
│   │   ├── main.py                 # FastAPI app — routes, middleware, handlers
│   │   ├── routes/
│   │   │   ├── chat.py             # POST /chat — main Q&A endpoint
│   │   │   ├── ingest.py           # POST /ingest/file, /ingest/directory
│   │   │   ├── evaluate.py         # POST /evaluate — RAGAS metrics
│   │   │   ├── sessions.py         # GET/DELETE /sessions
│   │   │   └── health.py           # GET /health
│   │   └── middleware/
│   │       ├── auth.py             # X-API-Key authentication
│   │       └── rate_limit.py       # Sliding window rate limiter
│   │
│   ├── agents/                     # LangGraph multi-agent system
│   │   ├── graph.py                # StateGraph — nodes, edges, conditional routing
│   │   ├── state.py                # Typed shared state (TypedDict)
│   │   ├── router.py               # Query complexity classifier (Tier 1/2/3)
│   │   ├── researcher.py           # Hybrid retrieval + structured extraction
│   │   ├── responder.py            # Grounded answer synthesis
│   │   └── critic.py               # Faithfulness + hallucination evaluation
│   │
│   ├── core/
│   │   ├── config.py               # Pydantic Settings — all config from .env
│   │   ├── llm_factory.py          # Multi-provider LLM (Groq → OpenAI → Ollama)
│   │   ├── embeddings.py           # Cached HuggingFace embedding model
│   │   └── exceptions.py           # Custom exception hierarchy
│   │
│   ├── ingestion/
│   │   ├── loaders.py              # Multi-format loaders (PDF/DOCX/CSV/TXT/MD)
│   │   ├── chunker.py              # Adaptive chunking (narrative vs tabular)
│   │   └── pipeline.py             # End-to-end ingestion orchestration
│   │
│   ├── retrieval/
│   │   ├── vectorstore.py          # Persistent FAISS manager
│   │   ├── bm25_store.py           # BM25 keyword index (rank-bm25)
│   │   ├── hybrid_retriever.py     # RRF fusion of BM25 + FAISS
│   │   └── reranker.py             # Cross-encoder precision reranking
│   │
│   ├── memory/
│   │   ├── store.py                # SQLite conversation persistence
│   │   └── session.py              # Session management API
│   │
│   ├── guardrails/
│   │   ├── input_guard.py          # Prompt injection, PII, length validation
│   │   └── output_guard.py         # Faithfulness threshold, disclaimer injection
│   │
│   ├── evaluation/
│   │   └── ragas_eval.py           # RAGAS faithfulness/relevancy/precision/recall
│   │
│   ├── prompts/
│   │   └── personas.py             # System prompts for all 4 agent roles
│   │
│   └── logging_/
│       └── tracker.py              # Structured JSON logging + LangSmith tracing
│
├── data/
│   └── sample_docs/
│       └── technova_annual_report_2024.txt   # Sample financial document
│
├── tests/
│   ├── test_ingestion.py           # Unit tests: loaders + chunker
│   ├── test_guardrails.py          # Unit tests: input safety checks
│   └── test_api.py                 # Integration tests: FastAPI endpoints
│
├── docker/
│   └── Dockerfile                  # Production Docker image
├── docker-compose.yml              # One-command deployment
├── requirements.txt                # Pinned dependencies
└── .env.example                    # Environment variable template
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.11+
- A free [Groq API key](https://console.groq.com/)

### 1. Clone & Install

```bash
git clone https://github.com/Jainparth22/finsight-ai.git
cd finsight-ai
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Windows:
Copy-Item .env.example .env
# Linux/macOS:
cp .env.example .env
```

Edit `.env`:
```env
GROQ_API_KEY=gsk_your_key_here
```

### 3. Run the API Server

```bash
uvicorn app.api.main:app --reload --port 8000
```

Visit **http://localhost:8000/docs** to see the interactive Swagger UI.

### 4. Ingest a Document

```bash
curl -X POST http://localhost:8000/ingest/file \
  -H "X-API-Key: dev-key-change-in-production" \
  -F "file=@data/sample_docs/technova_annual_report_2024.txt"
```

### 5. Ask a Financial Question

```bash
curl -X POST http://localhost:8000/chat \
  -H "X-API-Key: dev-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{"query": "What was TechNova'\''s EBITDA margin in FY2024 and how did it compare to FY2023?"}'
```

### 6. Docker Deployment (One Command)

```bash
docker-compose up --build
```

---

## 📡 API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/chat` | Multi-agent financial Q&A |
| `POST` | `/ingest/file` | Upload & index a document (PDF/DOCX/CSV/TXT/MD) |
| `POST` | `/ingest/directory` | Index all documents from a server directory |
| `POST` | `/evaluate` | Run RAGAS quality evaluation |
| `GET` | `/sessions` | List all conversation sessions |
| `POST` | `/sessions/new` | Create a new session |
| `GET` | `/sessions/{id}/history` | Get conversation history |
| `DELETE` | `/sessions/{id}` | Delete a session |
| `GET` | `/health` | Service health & uptime (public) |

### Sample Chat Response

```json
{
  "answer": "TechNova's EBITDA margin in FY2024 was **30.7%**, improving by 270 basis points from 28.0% in FY2023. Absolute EBITDA grew from $1,150M to $1,480M, a 28.7% increase year-over-year...\n\n---\n⚠️ Disclaimer: This information is sourced from indexed financial documents...",
  "session_id": "a3f2-...",
  "query_tier": 2,
  "critic_verdict": "PASS",
  "faithfulness_score": 0.94,
  "docs_retrieved": 3,
  "revision_count": 0,
  "sources": [
    {
      "source": "technova_annual_report_2024.txt",
      "page": "N/A",
      "rrf_score": 0.028541,
      "reranker_score": 0.9821,
      "snippet": "EBITDA 1,480 1,150 +28.7% EBITDA Margin 30.7% 28.0% +270 bps..."
    }
  ],
  "total_latency_ms": 1843.2,
  "llm_provider": "groq",
  "output_safe": true,
  "correlation_id": "a3f21c"
}
```

---

## 🧠 Design Decisions & Interview Guide

This section documents **why** every major technical choice was made — the information you need to defend this project in interviews.

---

### 1. Why RAG over Fine-Tuning?

| Dimension | RAG | Fine-Tuning |
| :--- | :--- | :--- |
| **Data freshness** | ✅ Ingest new docs anytime | ❌ Must retrain for new data |
| **Cost** | ✅ Zero training cost | ❌ $100–$10,000+ per run |
| **Transparency** | ✅ Sources cited per answer | ❌ Black-box knowledge |
| **Hallucination control** | ✅ Context-grounded | ❌ Model "remembers" incorrectly |
| **Domain adaptation** | ✅ Immediate via document upload | ❌ Weeks per cycle |

**Financial context:** Quarterly earnings change every 3 months. Fine-tuning would be perpetually stale. RAG allows real-time knowledge updates with zero model changes.

---

### 2. Why Hybrid Search (BM25 + FAISS)?

Dense-only search fails on financial documents because:
- **Exact figures**: `"EPS of $3.47"` — embedding space can't distinguish $3.47 from $3.74
- **Ticker symbols**: `AAPL`, `NVDA` — unknown tokens in embedding space
- **Acronyms**: `EBITDA`, `COGS`, `ROE` — semantic embeddings lose precision

BM25-only search fails because:
- `"earnings declined"` ≠ `"revenue dropped"` (semantically same, lexically different)
- Multi-paragraph reasoning requires semantic similarity

**Hybrid via Reciprocal Rank Fusion (RRF)** gets both:
```python
RRF(doc) = α × 1/(k + dense_rank) + (1-α) × 1/(k + bm25_rank)
# α=0.6: 60% dense, 40% BM25 (calibrated on BEIR-FiQA benchmark)
```

---

### 3. Why Cross-Encoder Reranking?

Retrieval recall ≠ precision:
- FAISS retrieves by **approximate** cosine similarity (optimised for speed)
- Cross-encoder reads `(query, passage)` pairs **jointly** with full attention
- Result: ~15-20% precision improvement at top-3, measured on MS-MARCO

Pipeline: retrieve k=6 candidates → rerank → return n=3
```
Cost: 6 cross-encoder forward passes ≈ 50ms on CPU → acceptable latency
Gain: Top-3 precision +15–20% → significant answer quality improvement
```

---

### 4. Why LangGraph over a Simple Chain?

| Feature | LangChain Chain | LangGraph |
| :--- | :--- | :--- |
| **Branching** | ❌ Linear only | ✅ Conditional edges |
| **Loops/Cycles** | ❌ Not supported | ✅ Critic → Researcher revision loop |
| **Shared state** | ❌ Passed manually | ✅ TypedDict flows through all nodes |
| **Checkpointing** | ❌ None | ✅ Built-in (resume after failure) |
| **Human-in-loop** | ❌ Not possible | ✅ Add interrupt() on any node |

The Critic → Researcher revision loop is impossible with a linear chain. LangGraph makes it a 3-line conditional edge.

---

### 5. Why Adaptive Chunking (Narrative vs Tabular)?

```
Narrative sections (MD&A, Risk Factors):
  → chunk_size=512 tokens
  → Why: Multi-sentence context needed for reasoning questions
  → "Why did margins improve?" needs paragraph-level context

Tabular sections (Income Statement, Balance Sheet):
  → chunk_size=256 tokens
  → Why: One row per chunk → retriever finds exact figures
  → "What was Q3 EPS?" needs the specific data row, not surrounding text
```

Detection: digit density > 15% OR section header match → tabular classification

---

### 6. Why Multi-LLM Fallback (Groq → OpenAI → Ollama)?

- **Groq**: 300 tokens/sec on Qwen-32B — fastest available inference, free tier
- **OpenAI**: Industry-standard quality, 99.99% SLA
- **Ollama**: 100% offline, zero data egress — mandatory for financial institutions with strict data governance

The waterfall is cost-ordered: cheapest (Groq) first, most expensive (OpenAI) only when Groq fails. This minimises cost while maximising uptime.

---

### 7. Why SQLite over Redis for Memory?

SQLite = zero infrastructure. For a single-node demo/portfolio project, Redis adds operational overhead with no benefit.

**Upgrade path documented in code:**
```python
# One-line swap when scaling:
# from langchain_redis import RedisChatMessageHistory
# (same interface, no other changes needed)
```

---

### 8. Why FAISS over Pinecone/Chroma/Weaviate?

| | FAISS | Pinecone | Chroma |
| :--- | :--- | :--- | :--- |
| **Latency** | <10ms (in-process) | 50–200ms (network) | 20–50ms |
| **Cost** | Free | $70+/month | Free (self-hosted) |
| **Offline** | ✅ Yes | ❌ Cloud only | ✅ Yes |
| **Scale limit** | ~1M vectors | Unlimited | ~100K practical |

For corpus < 1M chunks, FAISS exact search outperforms any network-bound managed service. The LangChain `VectorStore` abstraction makes migration a **3-line change** when scale demands it.

---

## 📊 Quality Evaluation (RAGAS)

FinSight AI includes an automated evaluation pipeline using [RAGAS](https://ragas.io) — the industry-standard RAG evaluation framework.

### Metrics Tracked

| Metric | What It Measures | Target Threshold |
| :--- | :--- | :--- |
| **Faithfulness** | Are all claims grounded in retrieved context? | ≥ 0.85 |
| **Answer Relevancy** | Does the answer address the question? | ≥ 0.80 |
| **Context Precision** | Are retrieved chunks truly relevant? | ≥ 0.75 |
| **Context Recall** | Did retrieval capture all needed info? | ≥ 0.70 |

### Running Evaluation

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "X-API-Key: dev-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [{
      "question": "What was TechNova'\''s FY2024 revenue?",
      "ground_truth": "TechNova reported revenue of $4.82 billion in FY2024.",
      "generated_answer": "TechNova Financial Holdings reported record revenues of $4.82 billion for fiscal year 2024.",
      "retrieved_contexts": ["TechNova Financial Holdings reported record revenues of $4.82 billion..."]
    }]
  }'
```

### Sample Evaluation Results

```json
{
  "metrics": {
    "faithfulness": 0.94,
    "answer_relevancy": 0.89,
    "context_precision": 0.82,
    "context_recall": 0.76
  },
  "threshold_status": {
    "faithfulness": "PASS",
    "answer_relevancy": "PASS",
    "context_precision": "PASS",
    "context_recall": "PASS"
  },
  "overall_pass": true
}
```

---

## 🛡️ Guardrails & Safety

### Input Guard (Pre-Processing)
| Check | Pattern | Action |
| :--- | :--- | :--- |
| **Prompt Injection** | `"ignore previous instructions"`, `"DAN mode"` | Block with 422 |
| **PII Detection** | SSN, credit card, passport patterns | Block with 422 |
| **Length Validation** | Query < 3 chars or > 2000 chars | Block with 422 |
| **Investment Advice** | `"should I buy"`, `"will stock go up"` | Flag → persona handles |

### Output Guard (Post-Processing)
| Check | Condition | Action |
| :--- | :--- | :--- |
| **FAIL verdict** | Critic gives FAIL | Replace with safe fallback |
| **Low faithfulness** | Score < 0.3 | Prepend uncertainty warning |
| **Hallucinations** | Critic detected fabrications | Prepend flagged claims list |
| **Disclaimer** | Always | Append regulatory disclaimer |

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=app --cov-report=html

# Run specific test class
pytest tests/test_guardrails.py -v
pytest tests/test_ingestion.py -v
pytest tests/test_api.py -v
```

---

## 🐳 Docker Deployment

```bash
# Build and start
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f finsight-api

# Stop
docker-compose down
```

The image pre-bakes both the embedding model (all-MiniLM-L6-v2) and the reranker model (ms-marco-MiniLM-L-6-v2) to eliminate cold-start latency.

---

## 🔭 Observability

### Structured JSON Logging
Every request emits a `REQUEST_COMPLETE` log event:
```json
{
  "correlation_id": "a3f21c",
  "session_id": "uuid",
  "total_latency_ms": 1843.2,
  "critic_verdict": "PASS",
  "faithfulness_score": 0.94,
  "revision_count": 0,
  "latency_breakdown_ms": {
    "router": 210.3,
    "researcher": 980.1,
    "responder": 520.4,
    "critic": 132.4
  }
}
```

### LangSmith Tracing (Optional)
Set in `.env`:
```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=finsight-ai
```
LangSmith traces every LLM call with token counts, prompt/response, and latency — invaluable for prompt debugging and cost optimisation.

---

## 🗺️ Roadmap

| Feature | Status |
| :--- | :--- |
| Multi-agent RAG pipeline | ✅ Complete |
| Hybrid BM25 + FAISS retrieval | ✅ Complete |
| Cross-encoder reranking | ✅ Complete |
| RAGAS evaluation | ✅ Complete |
| Conversation memory (SQLite) | ✅ Complete |
| Input + Output guardrails | ✅ Complete |
| FastAPI REST API | ✅ Complete |
| Docker deployment | ✅ Complete |
| LangSmith tracing | ✅ Complete |
| Streaming responses (SSE) | 🔜 Planned |
| Multi-document cross-reference | 🔜 Planned |
| Redis-backed session store | 🔜 Planned |
| Pinecone migration path | 🔜 Planned |
| CI/CD with RAGAS quality gate | 🔜 Planned |

---

## 📚 Tech Stack Summary

| Layer | Technology | Why Chosen |
| :--- | :--- | :--- |
| **Orchestration** | LangGraph 0.2 | Stateful conditional agent graphs; impossible with linear chains |
| **Primary LLM** | Groq (Qwen-32B) | ~300 tok/s — fastest available inference, free tier |
| **LLM Fallback 1** | OpenAI GPT-4o-mini | Industry standard quality, 99.99% SLA |
| **LLM Fallback 2** | Ollama (Llama 3.1) | 100% offline — required for air-gapped financial environments |
| **Embeddings** | all-MiniLM-L6-v2 | 80 MB, runs locally, zero API cost, 86.9 SBERT score |
| **Dense Search** | FAISS (IndexFlatIP) | <10ms exact search, in-process (no network RTT) |
| **Keyword Search** | BM25 (rank-bm25) | Exact financial figure matching; ticker/acronym precision |
| **Fusion** | Reciprocal Rank Fusion | Score-normalisation-free; used in Bing, TREC evaluations |
| **Reranking** | ms-marco-MiniLM-L-6-v2 | +15-20% precision@3 vs raw retrieval |
| **Framework** | FastAPI | Auto Swagger/OpenAPI, async, Pydantic validation |
| **Memory** | SQLite | Zero infrastructure; Redis-ready upgrade path documented |
| **Evaluation** | RAGAS | Industry-standard RAG quality metrics (no human labels needed) |
| **Observability** | LangSmith + JSON logs | Full trace + token-level debugging |
| **Deployment** | Docker + Compose | One-command reproducible deployment |

---

## 👤 Author

**Parth Jain**
- GitHub: [@Jainparth22](https://github.com/Jainparth22)

---

<div align="center">

*Built as a master-level showcase of GenAI engineering: RAG, multi-agent orchestration, hybrid retrieval, production guardrails, and automated quality evaluation — all in one cohesive system.*

</div>
