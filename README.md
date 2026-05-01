# Tegifa Legal — Source-Grounded Legal Intelligence

Private, local-first AI legal intelligence platform. Upload legal documents, ask questions with pinpoint citations, and run full contract reviews with clause-by-clause risk analysis — all running on your own infrastructure.

## What Makes This Different

**1. Pinpoint Citations, Not Vague Snippets**
Every answer cites the exact page, paragraph, and section heading from your documents, with confidence scores. Lawyers can verify every claim.

**2. Contract Review Vertical**
Upload a contract and get: clause-by-clause risk assessment, obligation extraction (who, what, when), missing clause detection, and an executive summary with an overall risk score.

**3. Self-Reflective Critique (SRLC)**
A multi-agent pipeline where an AI "Skeptic" challenges every answer before it reaches you. If the answer lacks evidence, an Investigator searches for more — automatically.

**4. Nougat OCR for Scanned Documents**
Uses Meta's Nougat model for academic/scanned PDF extraction that preserves tables, formulas, and complex layouts better than standard OCR.

**5. Everything Runs Locally**
No data leaves your network. LLM inference runs on llama.cpp or Ollama. Documents stay in your Docker volumes.

## Architecture

```
User Browser
    │
    ▼
  Caddy (TLS)
    │
    ▼
  Streamlit App ─────────────────────────────────────────┐
    │                                                     │
    ├──► Legal Q&A Tab                                    │
    │     ├── RAG Pipeline (LlamaIndex + ChromaDB)        │
    │     ├── Citation Engine (page/paragraph pinpoints)  │
    │     └── SRLC Multi-Agent Cycle (LangGraph)          │
    │           Reasoner → Epistemologist → Investigator   │
    │                                                     │
    ├──► Contract Review Tab                              │
    │     ├── Document Processor (Nougat OCR + pypdf)     │
    │     ├── Clause Classifier (14 legal clause types)   │
    │     ├── Risk Scorer (per-clause + aggregate)        │
    │     ├── Obligation Extractor (party/action/deadline)│
    │     └── Missing Clause Detector                     │
    │                                                     │
    ├──► Knowledge Graph (Neo4j) ← optional               │
    └──► Persistence (PostgreSQL / SQLite)                │
```

## Quick Start

```bash
# 1. Clone and configure
git clone <repo> && cd AI_LawFirmProject
cp .env.example .env
# Edit .env — set passwords and cookie key

# 2. Add a GGUF model
mkdir models && cp /path/to/model.gguf models/

# 3. Launch
docker-compose up --build -d

# 4. Open https://localhost and log in
```

## Contract Review Demo

1. Go to the **Contract Review** tab
2. Upload a PDF contract
3. Click **Run Contract Review**
4. Get back:
   - Executive summary with overall risk score (0-10)
   - Clause-by-clause analysis with risk badges
   - Extracted obligations (party, action, deadline)
   - Missing standard clauses warning
   - Export as JSON

## Project Structure

```
├── app.py                          # Streamlit UI (Q&A + Contract Review)
├── agents/
│   ├── orchestrator.py             # LangGraph SRLC pipeline
│   ├── contract_analyzer.py        # Contract review engine
│   ├── tools.py                    # Web search & graph query
│   └── graph_builder.py            # Neo4j knowledge graph
├── rag_scripts/
│   ├── document_processor.py       # Nougat OCR + clause tagging + chunking
│   ├── citation_engine.py          # Pinpoint citation builder
│   ├── rag_setup.py                # Document ingestion pipeline
│   └── query_rag.py                # RAG/CAG/SRLC query engine
├── db/
│   ├── models.py                   # SQLAlchemy models
│   └── persistence.py              # Chat session persistence
├── tests/                          # 6 test files, 60+ test cases
├── docker-compose.yml
├── .env.example
└── config.yaml
```

## Legal Clause Types Detected (14)

Termination, Indemnification, Liability, Confidentiality, Intellectual Property,
Governing Law, Payment, Force Majeure, Non-Compete, Warranty, Insurance,
Assignment, Severability, Entire Agreement

## Security

- All secrets in `.env` (never committed)
- XSS protection via `html.escape()` on all dynamic content
- Non-root Docker containers
- Database ports closed to host
- Input length validation
- Auth cookie key required via environment

## License

Private & Confidential — DigitalSvarga LLC
