# Tegifa Legal — Source-Grounded Legal Intelligence

Tegifa Legal is a privacy-first, multi-agent legal AI system that runs entirely on your infrastructure. Upload legal documents, ask questions, and get cited answers verified through an adversarial self-critique pipeline.

## Architecture

```
User Browser
    │
    ▼
  Caddy (TLS Reverse Proxy)
    │
    ▼
  Streamlit App ──────► RAG Pipeline (LlamaIndex + ChromaDB)
    │                        │
    │                        ▼
    │                   Local LLM (llama.cpp / Ollama)
    │
    ├──► Multi-Agent Orchestrator (LangGraph)
    │       Reasoner → Epistemologist → Investigator → loop
    │
    ├──► Knowledge Graph (Neo4j)
    │
    └──► Persistence (PostgreSQL / SQLite)
```

**Key principle:** No data leaves your network. All LLM inference runs locally.

## Quick Start

### Prerequisites
- Docker & Docker Compose
- A GGUF model file (e.g., `Phi-4-mini-instruct.Q8_0.gguf`) in `models/`

### Setup

```bash
# 1. Clone and enter the project
git clone <repository-url>
cd AI_LawFirmProject

# 2. Configure secrets (REQUIRED — do not skip)
cp .env.example .env
# Edit .env and set real passwords + cookie key

# 3. Launch
docker-compose up --build -d

# 4. Upload legal documents to docs/ and use the UI to re-index
```

### Generate auth cookie key
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

## Multi-Agent Pipeline (SRLC)

When Self-Reflective Critique is enabled, queries go through a cyclic multi-agent process:

1. **Reasoning Agent** — Synthesizes a hypothesis from document context
2. **Epistemology Agent** — Critiques the hypothesis for hallucinations and gaps
3. **Investigator Agent** — If rejected, searches external sources for evidence
4. Loop back to Reasoning with new evidence (max 2 iterations)

## Project Structure

```
├── app.py                  # Streamlit UI (main entry point)
├── agents/
│   ├── orchestrator.py     # LangGraph multi-agent pipeline
│   ├── tools.py            # Web search & graph query tools
│   └── graph_builder.py    # Neo4j knowledge graph manager
├── rag_scripts/
│   ├── rag_setup.py        # Document ingestion engine
│   └── query_rag.py        # RAG query engine (RAG/CAG/SRLC modes)
├── db/
│   ├── models.py           # SQLAlchemy models
│   └── persistence.py      # Chat session persistence
├── tests/                  # Pytest test suite
├── docker-compose.yml      # Production deployment
├── .env.example            # Environment variable template
└── config.yaml             # Auth configuration (placeholder creds)
```

## Security

- All secrets externalized to `.env` (never committed)
- HTML output sanitized against XSS
- Non-root container users
- Database ports not exposed to host
- Cookie signing key required via environment variable
- Input length validation

## License

Private & Confidential — DigitalSvarga LLC
