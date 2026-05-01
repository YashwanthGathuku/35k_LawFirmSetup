# ⚖️ Tegifa Legal — Autonomous Legal Operating System (v1.0 MVP)

Tegifa Legal is a private, local-first **Autonomous Legal Operating System** designed for high-stakes law firms. It transforms raw legal documents into an intelligent, event-driven knowledge base capable of cross-document conflict detection, autonomous drafting, and automated workflow integration.

Built for privacy-conscious attorneys, Tegifa runs entirely on your own infrastructure (Ollama/Docker), ensuring zero data leakage while providing intelligence that rivals top-tier legal tech suites.

---

## 🌟 Key Breakthroughs in v1.0

### 1. 🕸️ Deal Anatomy Graph (Cross-Document Intelligence)
Unlike standard RAG tools that look at one document at a time, Tegifa builds a **Knowledge Graph** of your entire deal.
- **Definition Drift:** Automatically flags when "Confidential Information" is defined differently in an NDA versus a Master Agreement.
- **Conflict Mapping:** Visualizes the relationship between Obligations, Liabilities, and Parties across multiple documents.
- **Red Nodes:** Glowing visual alerts in the UI when a legal contradiction is detected.

### 2. ✍️ The Shadow Redliner (Autonomous Drafting)
The first AI that doesn't just "talk"—it "works."
- **One-Click Harmonization:** Found a conflict in the graph? Click "Harmonize" and Tegifa drafts the fix.
- **Native .docx Output:** Generates a real Microsoft Word document with **Visual Redlining** (Track-Changes simulation) applied.
- **Lawyer-in-the-Loop:** Redlines are presented as suggestions, allowing the attorney to maintain 100% control over the final signature-ready copy.

### 3. 🧠 Advanced Cognitive Architecture (CAG)
Powered by **LangGraph**, Tegifa uses a state-machine reasoning loop that replaces linear prompts:
- **Router Agent:** Intelligently directs queries to specialized nodes (Legal Reasoner, Conflict Resolver, or Investigator).
- **Investigator (Live Web Search):** Can live-query the internet (DuckDuckGo) to fetch the latest statutes, case law, or legislative bills in real-time.
- **Action Executor:** Can trigger external webhooks to send alerts or sync data with firm management software.

### 4. 🔗 n8n Automation & Webhooks
Tegifa is an "extensible" OS. 
- **Inbound Webhooks:** FastAPI endpoints allow **n8n**, Zapier, or Outlook to automatically ingest client email attachments into a specific Matter workspace.
- **Outbound Triggers:** The AI can autonomously signal n8n to send Slack alerts, update CRM records, or file documents when specific legal milestones are met.

### 5. 📁 Matter-Based Workspace Isolation
Full multi-tenant architecture. 
- **Workspace Partitioning:** Every case/matter has its own isolated vector database and SQLite memory.
- **Metadata Filtering:** Ensures that queries for "Matter A" never leak into or hallucinate data from "Matter B."

---

## 🛠️ Tech Stack

- **UI:** Streamlit (Custom Premium CSS Overhaul)
- **API:** FastAPI (Webhooks & Background Tasks)
- **Orchestration:** LangGraph (Stateful Multi-Agent Reasoning)
- **Intelligence:** Ollama (Llama 3 / Gemma 4)
- **Vector DB:** ChromaDB (Metadata-filtered)
- **Graph Engine:** NetworkX + Streamlit Agraph
- **Document Processing:** python-docx, PyPDF2, Meta's Nougat OCR
- **Automation:** n8n (via Docker Compose)

---

## 🏗️ Architecture Overview

```text
    [ LAWYER UI (Streamlit) ] <───────> [ AUTOMATION HUB (n8n) ]
               │                                 │
               ▼                                 ▼
    [ COGNITIVE ARCHITECTURE ] <───────> [ WEBHOOK API (FastAPI) ]
    (LangGraph State Machine)                    │
               │                                 │
    ┌──────────┴──────────┐                      │
    ▼                     ▼                      ▼
[ LEGAL REASONER ]  [ CONFLICT RESOLVER ]  [ ACTION EXECUTOR ]
(RAG/CAG Logic)     (Graph Analysis)       (n8n Integration)
               │
               ▼
    [ VECTOR STORE / GRAPH DB ]
    (Isolated by Matter ID)
```

---

## 🚀 Installation & Setup

### 1. Prerequisites
- **Python 3.10+**
- **Ollama** (Running locally)
- **Docker & Docker Compose** (For n8n and database services)

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/YashwanthGathuku/35k_LawFirmSetup.git
cd 35k_LawFirmSetup

# Create Virtual Environment
python -m venv venv
source venv/bin/activate  # Or venv\Scripts\activate on Windows

# Install Dependencies
pip install -r requirements.txt
```

### 3. Configure `.env`
Copy `.env.example` to `.env` and configure your keys:
- `OLLAMA_HOST`: Set to your local Ollama address (default: `http://localhost:11434`)
- `SECRET_KEY`: For auth cookies.

### 4. Run the Platform
You need to run two services simultaneously:

**Terminal 1: The Webhook Server**
```bash
python api/webhooks.py
```

**Terminal 2: The Streamlit UI**
```bash
streamlit run app.py
```

---

## 📖 Usage Guide: The MVP Workflow

1.  **Create a Matter:** On the sidebar, create a new case (e.g., "Project Phoenix Acquisition").
2.  **Ingest Documents:** 
    - **Manual:** Upload PDFs/Word docs in the Ingestion tab.
    - **Automated:** Copy the Webhook URL from the sidebar into your n8n workflow to ingest attachments from Gmail/Outlook.
3.  **Analyze the Deal:** Go to the **Deal Anatomy Graph** tab, upload all related deal docs, and click **Analyze**. Watch the AI map the relationships and highlight red-flag conflicts.
4.  **Auto-Draft Redlines:** Click **"Harmonize"** on any detected conflict to download a redlined `.docx` file ready for review.
5.  **Live Research:** Use the Legal Q&A tab to ask about current laws (e.g., "What is the current status of the California AI safety bill?"). The AI will search the web and provide a cited answer.

---

## 🛡️ Security & Privacy
- **100% Local Inference:** No legal data is sent to OpenAI or Anthropic.
- **Workspace Isolation:** Enforced at the database query level via `matter_id`.
- **Encrypted Persistence:** SQLite/PostgreSQL with secure environment-based authentication.

---

## 📜 License
Private & Confidential — DigitalSvarga LLC. 
Built with ⚖️ for the future of Law.
