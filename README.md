# 🧠 RAG System — Retrieval-Augmented Generation

A full-stack RAG system that retrieves relevant documents from a vector database and generates grounded responses using Google Gemini. Features **LangGraph orchestration**, **strict structured routing/planning/validation**, **SSE streaming**, a **React chat interface**, and a **built-in evaluation framework**.

> Built with FastAPI, LangGraph, Neon PostgreSQL (`pgvector`), Google Gemini, and React.

## Architecture

```mermaid
flowchart TB
    subgraph Frontend["Frontend (React + Vite)"]
        UI["Chat UI"]
        SSE["SSE Stream Reader"]
    end

    subgraph Backend["Backend (FastAPI)"]
        API["API Endpoints"]
        EMB["Embedding Service<br/>Gemini Embedding"]
        VS["Vector Store<br/>Neon pgvector"]
        LLM["LLM Service<br/>Gemini 2.0 Flash"]
        RAG["LangGraph Orchestrator"]
    end

    subgraph Data["Data Layer"]
        DOCS["Source Documents<br/>.md files"]
        NEON[("Neon PostgreSQL<br/>Relational + pgvector")]
    end

    UI -->|"User Query"| API
    API --> RAG
    RAG --> EMB -->|"Query Embedding"| VS
    VS -->|"Top-K Docs"| RAG
    RAG -->|"Context + Query"| LLM
    LLM -->|"SSE Chunks"| API
    API -->|"Streaming Response"| SSE --> UI

    DOCS -->|"Ingest & Chunk"| EMB
    EMB -->|"Store Embeddings"| NEON
    VS <-->|"Search"| NEON

    style Frontend fill:#1a1a2e,stroke:#8b5cf6,color:#fff
    style Backend fill:#16213e,stroke:#3b82f6,color:#fff
    style Data fill:#0f3460,stroke:#10b981,color:#fff
```

### Key Features

- 🔍 **Semantic search** — Gemini embeddings with Neon PostgreSQL (`pgvector`)
- 🧭 **LangGraph state machine** — Routed execution with conditional edges and retry paths
- ✅ **Strict structured control nodes** — Schema-enforced Router/Planner/Validator outputs
- ⚡ **Streaming** — Token-by-token SSE streaming for real-time responses
- 📊 **Evaluation** — Built-in metrics framework (precision, recall, faithfulness)
- 🎨 **Chat UI** — Dark-mode React frontend with source citations
- 🐳 **Docker Compose** — One-command full-stack deployment
- 🚀 **CI/CD** — Automated deploy to Cloud Run (backend) and Firebase Hosting (frontend)

## RAG Plan Status

- [x] Phase 1 MVP checklist complete (QueryState, router, planner, tool_orchestrator, writer, relevance/groundedness validator, single retry loop)
- [x] LangGraph runtime integrated for orchestration
- [x] Strict schema-based Router/Planner/Validator output wired through Gemini JSON mode
- [x] Legacy SQLModel SQuAD ingestion path retained as an optional pipeline (not in main API runtime)
- [ ] Phase 2 (agent layer, completeness and citation checks, richer failure policies)
- [ ] Phase 3 (observability dashboards, caching/rate limiting, latency/cost tuning)

---

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# 1. Clone and configure
git clone <repo-url> && cd coding-exercise
cp backend/.env.example backend/.env
# Edit backend/.env — set GEMINI_API_KEY and DATABASE_URL

# 2. Start everything
docker-compose up --build

# 3. Ingest documents
curl -X POST http://localhost:8000/ingest

# 4. Open the UI
open http://localhost:5173
```

### Option 2: Local Development

#### Backend (Terminal 1)

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure your Gemini API key
cp .env.example .env
# Edit .env and set GEMINI_API_KEY + DATABASE_URL

# Ingest the source documents into Neon PostgreSQL (pgvector)
python -m data.ingest

# Start the backend server
uvicorn main:app --reload
# ✅ Backend running at http://localhost:8000
```

#### Frontend (Terminal 2)

```bash
cd frontend
npm install
npm run dev
# ✅ Frontend running at http://localhost:5173
```

> **Note:** The frontend proxies API requests to `http://localhost:8000` automatically via Vite config.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check + model status |
| `POST` | `/query` | RAG query (JSON response) |
| `POST` | `/query/stream` | RAG query (SSE streaming) |
| `GET` | `/documents` | List indexed documents |
| `POST` | `/ingest` | Ingest documents from disk |

### Example Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Why is centering a div so hard?", "top_k": 3}'
```

---

## GitHub Actions Deploy Secrets

For backend auto-deploy (`.github/workflows/deploy-backend.yml`), set these repository secrets:

- `GCP_PROJECT_ID`
- `GCP_SA_KEY`
- `GEMINI_API_KEY`
- `DATABASE_URL` (Neon/Postgres connection string)

Without `DATABASE_URL`, Cloud Run startup will fail when `VectorStoreService` initializes.

---

## Evaluation

Run the built-in evaluation framework:

```bash
cd backend
python -m evaluation.evaluate
```

This runs 12 curated queries and measures:

| Metric | Description |
|--------|-------------|
| Context Precision | Was the correct source document retrieved? |
| Context Recall@K | Position of the correct source in results |
| Keyword Coverage | Do expected keywords appear in the answer? |
| Answer Length | Is the answer appropriately sized? |
| Faithfulness | Is the answer grounded in retrieved context? |

Results are printed as a table and saved to `evaluation_results.json`.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python, FastAPI, Uvicorn, LangGraph |
| Vector DB | Neon PostgreSQL + `pgvector` |
| Embeddings | Gemini Embedding (`gemini-embedding-001`) |
| LLM | Google Gemini 2.0 Flash |
| Frontend | React 18, Vite |
| Infra | Docker Compose, Nginx, Cloud Run, Firebase Hosting |

## Design Decisions

1. **Neon PostgreSQL + pgvector as a unified store** — One managed database for both relational data and vector search, simplifying deployment and operations.

2. **Gemini Embeddings over sentence-transformers** — Eliminates the ~2 GB PyTorch dependency, keeping Docker images small. Uses `gemini-embedding-001` with `output_dimensionality=768` via the same API key already needed for generation.

3. **SSE over WebSockets** — Simpler to implement, works through proxies, and is the standard for LLM streaming (used by ChatGPT, Claude, etc.).

4. **Custom evaluation over RAGAS** — Lightweight, no heavy dependencies, and more transparent. Each metric is <30 lines and easy to understand.

---

## Running Tests

```bash
cd backend
pip install -r requirements.txt
python -m pytest tests/ -v
```

---

## Project Structure

```
coding-exercise/
├── backend/
│   ├── main.py                  # FastAPI endpoints (query, stream, ingest, health)
│   ├── config.py                # Pydantic settings from .env
│   ├── models.py                # Request/response schemas
│   ├── Dockerfile               # Multi-stage backend container
│   ├── deploy.sh                # Manual Cloud Run deploy script
│   ├── requirements.txt         # Python dependencies (dev)
│   ├── requirements-prod.txt    # Python dependencies (production)
│   ├── .env.example             # Environment variable template
│   ├── .dockerignore            # Docker build exclusions
│   ├── services/
│   │   ├── embedding.py         # Gemini embedding service (gemini-embedding-001)
│   │   ├── vector_store.py      # Neon PostgreSQL pgvector operations
│   │   ├── llm.py               # Gemini LLM integration + strict structured output helpers
│   │   └── rag.py               # LangGraph RAG pipeline orchestration
│   ├── data/
│   │   ├── ingest.py            # Document chunking & ingestion
│   │   └── documents/           # Source .md files (knowledge base)
│   ├── evaluation/
│   │   ├── evaluate.py          # Evaluation runner
│   │   ├── metrics.py           # Precision, recall, faithfulness metrics
│   │   └── test_queries.json    # 12 curated test queries
│   └── tests/
│       ├── test_api.py          # API integration tests
│       ├── test_rag.py          # RAG evaluation tests
│       └── test_vector_store.py # Vector store unit tests
├── frontend/
│   ├── index.html               # Entry point
│   ├── vite.config.js           # Vite config + API proxy
│   ├── package.json             # Node dependencies
│   ├── Dockerfile               # Multi-stage build (Vite → Nginx)
│   ├── nginx.conf               # Nginx config for SPA + API proxy
│   ├── firebase.json            # Firebase Hosting configuration
│   ├── .firebaserc              # Firebase project binding
│   └── src/
│       ├── main.jsx             # React entry
│       ├── App.jsx              # Main app + SSE streaming logic
│       ├── index.css            # Global styles (dark theme)
│       └── components/
│           ├── ChatInterface.jsx   # Message list + empty state
│           ├── Message.jsx         # Individual message rendering
│           ├── QueryInput.jsx      # Auto-resizing textarea input
│           └── SourceDocuments.jsx  # Retrieved sources panel
├── .github/
│   └── workflows/
│       ├── deploy-backend.yml   # CI/CD: Backend → Cloud Run
│       └── deploy-frontend.yml  # CI/CD: Frontend → Firebase Hosting
├── docker-compose.yml           # Backend + Frontend orchestration
└── README.md
```
