"""
Tegifa Legal — Source-Grounded Legal Intelligence
Main Streamlit application with:
- Legal Q&A with pinpoint citations
- Contract Review vertical (clause analysis, risk scoring, obligation extraction)
- Document upload & intelligent ingestion
"""
import streamlit as st
import streamlit_authenticator as stauth
import requests
import os
import json
import html
import logging
import yaml
import tempfile
import shutil
from yaml.loader import SafeLoader

from db.persistence import save_chat_session, load_chat_session, get_matters, create_matter
from rag_scripts.query_rag import init_llm, init_index, execute_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tegifa.app")

# --- Custom CSS (Premium UI) ---
st.markdown("""
<style>
    /* Sleek Dark Mode Overrides */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Glowing Nodes Effect & Premium Typography */
    h1, h2, h3 {
        font-family: 'Inter', 'Roboto', sans-serif;
        font-weight: 600;
        background: -webkit-linear-gradient(#00C4EB, #00D26A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stButton>button {
        border-radius: 8px;
        background: linear-gradient(135deg, #00C4EB 0%, #0073E6 100%);
        color: white;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 15px rgba(0, 196, 235, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 6px 20px rgba(0, 196, 235, 0.6);
        transform: translateY(-2px);
    }
    
    /* Sidebar Polish */
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }

    :root {
        --primary-color: #00C4EB;
        --accent-gold: #00D26A;
        --risk-critical: #FF4B4B;
        --risk-high: #EA580C;
        --risk-medium: #D97706;
        --risk-low: #00D26A;
    }
    
    /* Legacy Classes */
    .thought-bubble {
        background-color: #1F242D;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid var(--primary-color);
        margin: 10px 0;
        font-family: 'Courier New', Courier, monospace;
        font-size: 0.9rem;
        color: #E2E8F0;
    }
    .citation-card {
        background-color: #161B22;
        padding: 14px 16px;
        border-radius: 8px;
        border: 1px solid #30363D;
        margin-bottom: 10px;
    }
    .citation-card .pinpoint {
        color: var(--primary-color);
        font-weight: 600;
        font-size: 0.85rem;
    }
    .citation-card .confidence-high { color: var(--risk-low); }
    .citation-card .confidence-medium { color: var(--risk-medium); }
    .citation-card .confidence-low { color: var(--risk-critical); }
    .citation-card .snippet {
        font-size: 0.88rem;
        color: #A3B1C6;
        margin-top: 6px;
        line-height: 1.5;
    }
    .clause-tag {
        display: inline-block;
        background: #1F242D;
        color: var(--primary-color);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        margin-right: 4px;
        margin-top: 4px;
    }
    .risk-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        color: white;
    }
    .risk-critical { background-color: var(--risk-critical); }
    .risk-high { background-color: var(--risk-high); }
    .risk-medium { background-color: var(--risk-medium); }
    .risk-low { background-color: var(--risk-low); }
    
    .exec-summary {
        background: linear-gradient(135deg, #161B22, #1F242D);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid var(--primary-color);
        margin: 15px 0;
        line-height: 1.6;
        color: #FAFAFA;
    }
</style>
""", unsafe_allow_html=True)


# --- Auth Setup ---
with open("config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

env_cookie_key = os.getenv("AUTH_COOKIE_KEY")
if env_cookie_key:
    config["cookie"]["key"] = env_cookie_key
elif config["cookie"]["key"] == "OVERRIDE_VIA_ENV_VAR":
    st.error("AUTH_COOKIE_KEY environment variable is not set. Cannot start.")
    st.stop()

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

authenticator.login(location="main", key="Login")

name = st.session_state.get("name")
authentication_status = st.session_state.get("authentication_status")
username = st.session_state.get("username")

if authentication_status is False:
    st.error("Username/password is incorrect")
    st.stop()
elif authentication_status is None:
    st.warning("Please enter your username and password")
    st.stop()


# --- Constants ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
LLM_API_BASE = os.getenv("LLM_API_BASE", "http://llm-server:8080/v1")
DB_PATH = os.getenv("RAG_DB_PATH", "/app/chroma_db")
STORAGE_PATH = os.getenv("RAG_STORAGE_PATH", "/app/storage")
DOCS_PATH = os.getenv("RAG_DOCS_PATH", "/app/docs")
MAX_QUESTION_LENGTH = 2000
MAX_UPLOAD_SIZE_MB = 50
TEMP_DIR = tempfile.gettempdir()


def sanitize_html(text: str) -> str:
    """Escape HTML to prevent XSS."""
    return html.escape(str(text))


def get_ollama_models() -> list[str]:
    """Fetch available Ollama models."""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if response.status_code == 200:
            return [m["name"] for m in response.json().get("models", [])]
    except requests.RequestException as e:
        logger.warning("Failed to fetch Ollama models: %s", e)
    return []


def render_citation_card(citation: dict):
    """Render a single pinpoint citation card with XSS protection."""
    pinpoint = sanitize_html(citation.get("pinpoint", "Unknown"))
    snippet = sanitize_html(citation.get("snippet", "")[:400])
    confidence = citation.get("confidence", 0)
    confidence_label = citation.get("confidence_label", "low")
    clauses = citation.get("detected_clauses", [])
    has_table = citation.get("contains_table", False)

    conf_pct = f"{confidence * 100:.0f}%"

    clause_tags = ""
    if clauses:
        clause_tags = '<div class="clause-tags">' + "".join(
            f'<span class="clause-tag">{sanitize_html(c.replace("_", " ").title())}</span>'
            for c in clauses
        ) + "</div>"

    table_indicator = ' <span class="clause-tag">📊 Contains Table</span>' if has_table else ""

    st.markdown(
        f"""<div class="citation-card">
            <div class="pinpoint">📌 {pinpoint}</div>
            <div>
                <span class="confidence-{confidence_label}">■</span>
                Confidence: <strong>{conf_pct}</strong> ({confidence_label})
                {table_indicator}
            </div>
            <div class="snippet">"{snippet}..."</div>
            {clause_tags}
        </div>""",
        unsafe_allow_html=True,
    )


def render_risk_badge(level: str) -> str:
    """Return HTML for a risk level badge."""
    return f'<span class="risk-badge risk-{sanitize_html(level)}">{sanitize_html(level.upper())}</span>'


# --- Sidebar ---
with st.sidebar:
    st.markdown("## ⚖️ Tegifa Legal")
    st.markdown(f"Welcome, **{sanitize_html(name)}**")
    authenticator.logout("Logout", "sidebar")
    st.markdown("---")

    st.subheader("📁 Case / Matter Workspace")
    matters = get_matters(username)
    matter_options = {m["name"]: m["id"] for m in matters}
    
    active_matter_id = None
    if not matters:
        st.warning("No matters found. Create one below.")
    else:
        default_idx = 0
        if "active_matter_id" in st.session_state:
            for i, m in enumerate(matters):
                if m["id"] == st.session_state["active_matter_id"]:
                    default_idx = i
                    break
        
        selected_matter_name = st.selectbox("Active Matter", list(matter_options.keys()), index=default_idx)
        active_matter_id = matter_options[selected_matter_name]
        st.session_state["active_matter_id"] = active_matter_id
        
        # Display Webhook Integration Details
        st.markdown(f"""
        <div style="background-color: #1F242D; padding: 10px; border-radius: 5px; margin-top: 10px; margin-bottom: 10px;">
            <small style="color: #A3B1C6;">🔗 <b>n8n Webhook Ingestion URL</b></small><br/>
            <code style="font-size: 0.8em; color: #00D26A;"><a href="http://127.0.0.1:8000/webhook/ingest" target="_blank" style="color:#00D26A; text-decoration:none;">http://127.0.0.1:8000/webhook/ingest</a></code><br/>
            <small style="color: #A3B1C6;">Payload: <code>{{"matter_id": {active_matter_id}, "files": [...]}}</code></small>
        </div>
        """, unsafe_allow_html=True)
    
    new_matter_name = st.text_input("New Matter Name", placeholder="e.g., Project Phoenix M&A")
    if st.button("➕ Create Matter") and new_matter_name:
        new_m = create_matter(username, new_matter_name)
        st.session_state["active_matter_id"] = new_m["id"]
        st.success(f"Created matter: {new_matter_name}")
        st.rerun()



    st.subheader("🤖 Model Hub")
    backend = st.selectbox("Backend Engine", ["llama.cpp", "ollama"])

    ollama_models = get_ollama_models()
    if backend == "ollama":
        model_name = st.selectbox("Choose Model", ollama_models) if ollama_models else "llama3"
        if not ollama_models:
            st.warning("No Ollama models found.")
    else:
        model_name = "local-llama-cpp"

    st.markdown("---")
    st.subheader("🔬 Advanced Logic")
    use_srlc = st.toggle("Self-Reflective Critique (SRLC)", value=True,
                          help="Multi-agent cycle: Reasoner → Epistemologist → Investigator")
    use_cag = st.toggle("CAG Optimization", value=False)
    use_nougat = st.toggle("Nougat OCR (scanned PDFs)", value=True,
                           help="Use Meta's Nougat for academic/scanned PDF extraction")

    st.markdown("---")
    st.subheader("📄 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload legal documents",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
        help=f"Max {MAX_UPLOAD_SIZE_MB}MB per file",
    )

    if uploaded_files:
        if st.button("📥 Ingest Uploaded Documents"):
            os.makedirs(DOCS_PATH, exist_ok=True)
            saved = []
            for uploaded_file in uploaded_files:
                if uploaded_file.size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                    st.error(f"{uploaded_file.name} exceeds {MAX_UPLOAD_SIZE_MB}MB limit.")
                    continue
                dest = os.path.join(DOCS_PATH, uploaded_file.name)
                with open(dest, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved.append(uploaded_file.name)

            if saved:
                with st.spinner(f"Ingesting {len(saved)} document(s)..."):
                    try:
                        from rag_scripts.rag_setup import run_ingestion
                        if not active_matter_id:
                            st.error("Please select or create an Active Matter first.")
                        else:
                            result = run_ingestion(
                                matter_id=active_matter_id,
                                db_path=DB_PATH,
                                docs_path=DOCS_PATH,
                                storage_path=STORAGE_PATH,
                                use_nougat=use_nougat,
                            )
                            st.success(result)
                            # Clear cached pipeline to pick up new docs
                            load_rag_pipeline.clear()
                    except Exception as e:
                        logger.error("Ingestion failed: %s", e)
                        st.error(f"Ingestion failed: {e}")

    st.markdown("---")
    if st.button("🔄 Re-index All Documents"):
        with st.spinner("Re-indexing knowledge base..."):
            try:
                from rag_scripts.rag_setup import run_ingestion
                result = run_ingestion(
                    db_path=DB_PATH, docs_path=DOCS_PATH,
                    storage_path=STORAGE_PATH, use_nougat=use_nougat,
                )
                st.success(result)
                load_rag_pipeline.clear()
            except Exception as e:
                logger.error("Re-indexing failed: %s", e)
                st.error(f"Re-indexing failed: {e}")


# --- Initialize RAG pipeline (cached) ---
@st.cache_resource(show_spinner="Loading intelligence engine...")
def load_rag_pipeline(backend_type: str, model: str):
    """Initialize LLM and vector index once per session."""
    from llama_index.core import Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    llm = init_llm(
        model_type=backend_type, model_name=model,
        ollama_host=OLLAMA_HOST, llm_api_base=LLM_API_BASE,
    )
    Settings.llm = llm
    try:
        index = init_index(db_path=DB_PATH, storage_path=STORAGE_PATH)
    except Exception as e:
        logger.warning("Could not load existing index: %s. Creating empty.", e)
        from llama_index.core import VectorStoreIndex
        index = VectorStoreIndex.from_documents([])
    return llm, index


# =========================================================================
# MAIN UI — Tabs: Legal Q&A | Contract Review
# =========================================================================

st.title("⚖️ Tegifa Legal Intelligence")
st.caption(f"Engine: {backend} | Model: {model_name} | "
           f"Mode: {'SRLC' if use_srlc else 'Standard'} | "
           f"OCR: {'Nougat' if use_nougat else 'Standard'}")

tab_qa, tab_contract, tab_deal_room = st.tabs(["💬 Legal Q&A", "📋 Contract Review", "🕸️ Deal Anatomy Graph"])


# ─────────────────────────────────────────────────────────────────
# Tab 1: Legal Q&A with Pinpoint Citations
# ─────────────────────────────────────────────────────────────────
with tab_qa:
    # Chat history
    if "messages" not in st.session_state or st.session_state.get("current_matter_view") != active_matter_id:
        try:
            if active_matter_id:
                st.session_state.messages = load_chat_session(username, active_matter_id)
            else:
                st.session_state.messages = []
            st.session_state.current_matter_view = active_matter_id
        except Exception as e:
            logger.error("Failed to load chat history: %s", e)
            st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("thought_stream"):
                with st.status("Algorithm Thought Process", expanded=False, state="complete"):
                    for step in message["thought_stream"]:
                        st.markdown(f"**Step: {sanitize_html(step.get('step', ''))}**")
                        st.markdown(
                            f'<div class="thought-bubble">{sanitize_html(step.get("content", ""))}</div>',
                            unsafe_allow_html=True,
                        )
            st.markdown(message["content"])

            # Render citation report if present
            report = message.get("citation_report", {})
            citations = report.get("citations", [])
            if citations:
                with st.expander(f"📚 Pinpoint Citations ({report.get('summary', '')})"):
                    for cit in citations:
                        render_citation_card(cit)

    # Chat input
    query = st.chat_input("Enter legal inquiry...")

    if query:
        query = query.strip()
        if not query:
            st.warning("Please enter a non-empty question.")
        elif len(query) > MAX_QUESTION_LENGTH:
            st.error(f"Question exceeds {MAX_QUESTION_LENGTH}-character limit.")
        else:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                spinner_msg = (
                    "Tegifa is analyzing (SRLC active)..."
                    if use_srlc else "Tegifa is thinking..."
                )
                with st.spinner(spinner_msg):
                    try:
                        llm, index = load_rag_pipeline(backend, model_name)
                        result = execute_query(
                            question=query, index=index, llm=llm,
                            matter_id=active_matter_id,
                            top_k=3, use_cag=use_cag, use_srlc=use_srlc,
                            model_name=model_name,
                        )

                        answer_text = result.get("answer", "No answer generated.")
                        citation_report = result.get("citation_report", {})
                        thought_stream = result.get("thought_stream", [])

                        if thought_stream:
                            with st.status("Algorithm Thought Process", expanded=False) as status:
                                for step in thought_stream:
                                    st.markdown(f"**Step: {sanitize_html(step.get('step', ''))}**")
                                    st.markdown(
                                        f'<div class="thought-bubble">{sanitize_html(step.get("content", ""))}</div>',
                                        unsafe_allow_html=True,
                                    )
                                status.update(label="Critique Complete", state="complete")

                        st.markdown(answer_text)

                        citations = citation_report.get("citations", [])
                        if citations:
                            with st.expander(f"📚 Pinpoint Citations ({citation_report.get('summary', '')})"):
                                for cit in citations:
                                    render_citation_card(cit)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer_text,
                            "thought_stream": thought_stream,
                            "citation_report": citation_report,
                        })
                        try:
                            if active_matter_id:
                                save_chat_session(username, active_matter_id, st.session_state.messages)
                        except Exception as e:
                            logger.error("Failed to save chat: %s", e)

                    except Exception as e:
                        logger.error("Query failed: %s", e, exc_info=True)
                        st.error(f"Error: {e}")


# ─────────────────────────────────────────────────────────────────
# Tab 2: Contract Review
# ─────────────────────────────────────────────────────────────────
with tab_contract:
    st.subheader("📋 AI Contract Review")
    st.markdown(
        "Upload a contract to get a full clause-by-clause analysis with "
        "risk scoring, obligation extraction, and missing clause detection."
    )

    contract_file = st.file_uploader(
        "Upload contract for review",
        type=["pdf", "txt", "docx"],
        key="contract_upload",
        help="Supported: PDF (with Nougat OCR), DOCX, TXT",
    )

    if contract_file:
        st.info(f"📄 **{contract_file.name}** ({contract_file.size / 1024:.1f} KB)")

        if st.button("🔍 Run Contract Review", type="primary"):
            if contract_file.size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                st.error(f"File exceeds {MAX_UPLOAD_SIZE_MB}MB limit.")
            else:
                with st.spinner("Analyzing contract — this may take 1-3 minutes..."):
                    try:
                        # Save to temp file for processing
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=os.path.splitext(contract_file.name)[1]
                        ) as tmp:
                            tmp.write(contract_file.getbuffer())
                            tmp_path = tmp.name

                        # Process document
                        from rag_scripts.document_processor import process_file
                        from agents.contract_analyzer import run_contract_review, report_to_dict

                        processed_doc = process_file(
                            filepath=tmp_path, use_nougat=use_nougat
                        )
                        # Override filename from temp path
                        processed_doc.file_name = contract_file.name

                        llm, _ = load_rag_pipeline(backend, model_name)
                        report = run_contract_review(llm=llm, processed_doc=processed_doc)
                        report_data = report_to_dict(report)

                        # Clean up temp file
                        os.unlink(tmp_path)

                        # Store in session state
                        st.session_state["contract_report"] = report_data

                    except Exception as e:
                        logger.error("Contract review failed: %s", e, exc_info=True)
                        st.error(f"Contract review failed: {e}")

    # --- Display Contract Report ---
    report_data = st.session_state.get("contract_report")
    if report_data:
        st.markdown("---")

        # Executive Summary
        risk_level = report_data["overall_risk_level"]
        risk_score = report_data["overall_risk_score"]

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"### {sanitize_html(report_data['document_name'])}")
            st.caption(f"Type: {report_data['document_type'].title()} | Pages: {report_data['total_pages']}")
        with col2:
            st.markdown(f"**Overall Risk:** {render_risk_badge(risk_level)}", unsafe_allow_html=True)
        with col3:
            st.metric("Risk Score", f"{risk_score * 10:.1f}/10")

        st.markdown(
            f'<div class="exec-summary">{sanitize_html(report_data["executive_summary"])}</div>',
            unsafe_allow_html=True,
        )

        # Key Findings
        if report_data["key_findings"]:
            st.markdown("#### ⚡ Key Findings")
            for finding in report_data["key_findings"]:
                st.markdown(f"- {finding}")

        # Missing Clauses
        if report_data["missing_clauses"]:
            st.markdown("#### ⚠️ Missing Standard Clauses")
            missing_str = ", ".join(
                f"**{c.replace('_', ' ').title()}**"
                for c in report_data["missing_clauses"]
            )
            st.warning(f"The following standard clauses were not found: {missing_str}")

        # Clause Analysis
        clause_analyses = report_data.get("clause_analyses", [])
        if clause_analyses:
            st.markdown("#### 🔍 Clause-by-Clause Analysis")
            for ca in clause_analyses:
                clause_title = ca["clause_type"].replace("_", " ").title()
                ca_risk = ca["risk_level"]
                with st.expander(
                    f"{clause_title} — {ca_risk.upper()} risk"
                    + (f" (p. {ca['page_number']})" if ca.get("page_number") else "")
                ):
                    st.markdown(f"**Risk:** {render_risk_badge(ca_risk)} Score: {ca['risk_score']:.2f}", unsafe_allow_html=True)
                    st.markdown(f"**Summary:** {ca['summary']}")

                    if ca.get("concerns"):
                        st.markdown("**Concerns:**")
                        for concern in ca["concerns"]:
                            st.markdown(f"- ⚠️ {concern}")

                    if ca.get("recommendation"):
                        st.markdown(f"**Recommendation:** {ca['recommendation']}")

                    if ca.get("source_text"):
                        st.markdown("**Source Text:**")
                        st.code(ca["source_text"][:500], language=None)

        # Obligations
        obligations = report_data.get("obligations", [])
        if obligations:
            st.markdown("#### 📜 Extracted Obligations")
            for i, ob in enumerate(obligations, 1):
                deadline_str = f" | **Deadline:** {ob['deadline']}" if ob.get("deadline") else ""
                condition_str = f" | **Condition:** {ob['condition']}" if ob.get("condition") else ""
                page_str = f" *(p. {ob['page_number']})*" if ob.get("page_number") else ""
                st.markdown(
                    f"{i}. **{sanitize_html(ob['party'])}:** "
                    f"{sanitize_html(ob['action'])}{deadline_str}{condition_str}{page_str}"
                )

        # Recommendation
        st.markdown("---")
        st.markdown(f"### 📌 Recommendation: {report_data['recommendation']}")

        # Export
        if st.button("📥 Export Report as JSON"):
            st.download_button(
                label="Download Report",
                data=json.dumps(report_data, indent=2),
                file_name=f"tegifa_review_{report_data['document_name']}.json",
                mime="application/json",
            )

# ─────────────────────────────────────────────────────────────────
# Tab 3: Deal Anatomy Graph (Cross-Document Analyzer)
# ─────────────────────────────────────────────────────────────────
with tab_deal_room:
    st.header("🕸️ Deal Anatomy Graph")
    st.markdown("Upload multiple related deal documents (e.g., MSA, NDA, SOW) to map definitions and find cross-document conflicts/loopholes.")

    deal_files = st.file_uploader("Upload Deal Documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    
    if st.button("Analyze Deal Structure") and deal_files:
        st.session_state["deal_analysis_running"] = True
        
    if st.session_state.get("deal_analysis_running") and deal_files:
        with st.spinner("Processing documents and building Knowledge Graph... This may take a minute."):
            try:
                from rag_scripts.document_processor import process_file
                from agents.deal_graph import build_deal_graph, get_conflicts
                from streamlit_agraph import agraph, Node, Edge, Config

                processed_docs = []
                # 1. Process all documents
                for f in deal_files:
                    temp_path = os.path.join(TEMP_DIR, f.name)
                    with open(temp_path, "wb") as out_f:
                        out_f.write(f.read())
                    
                    doc = process_file(temp_path, use_nougat=False)
                    processed_docs.append(doc)

                # 2. Build Graph
                st.info(f"Extracting Definitions & Dependencies from {len(processed_docs)} documents...")
                G = build_deal_graph(llm, processed_docs)
                
                # 3. Detect Conflicts
                if "conflicts" not in st.session_state:
                    st.session_state["conflicts"] = get_conflicts(G)
                
                conflicts = st.session_state["conflicts"]
                if conflicts:
                    st.error(f"🚨 Found {len(conflicts)} Definition Conflicts (Drift) across documents!")
                    for c in conflicts:
                        with st.expander(f"Conflict: {c['term']}"):
                            st.write(c['details'])
                            
                            # Shadow Redliner Integration
                            docx_files = [f for f in deal_files if f.name.endswith('.docx')]
                            if docx_files:
                                target_f = docx_files[0]
                                if st.button(f"Harmonize in {target_f.name}", key=f"redline_{c['term']}"):
                                    with st.spinner("Shadow Redliner is drafting..."):
                                        from agents.shadow_redliner import find_and_redline_document
                                        temp_path = os.path.join(TEMP_DIR, target_f.name)
                                        out_path = find_and_redline_document(
                                            file_path=temp_path,
                                            target_conflict_term=c['term'],
                                            harmonization_instructions=c['details'],
                                            llm=llm
                                        )
                                        with open(out_path, "rb") as file:
                                            btn = st.download_button(
                                                label="📥 Download Redlined Document",
                                                data=file,
                                                file_name=os.path.basename(out_path),
                                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                                            )
                else:
                    st.success("✅ No definition drift detected across documents.")

                # 4. Render Graph
                st.subheader("Semantic Graph View")
                nodes = []
                edges = []

                # Convert NetworkX to agraph
                for n, data in G.nodes(data=True):
                    node_type = data.get("type")
                    if data.get("conflict"):
                        color = "#FF4B4B" # Red
                    elif node_type == "Document":
                        color = "#00C4EB" # Blue
                    elif node_type == "Obligation":
                        color = "#D97706" # Orange
                    elif node_type == "Liability":
                        color = "#9333EA" # Purple
                    else:
                        color = "#00D26A" # Green
                        
                    shape = "hexagon" if node_type == "Document" else ("triangle" if node_type == "Liability" else ("square" if node_type == "Obligation" else "dot"))
                    title = data.get("definition", data.get("details", n))
                    nodes.append(Node(id=n, label=data.get("label", n), size=25, color=color, shape=shape, title=title))

                for src, dst, data in G.edges(data=True):
                    edges.append(Edge(source=src, target=dst, label=data.get("relation", "")))

                config = Config(width=800, height=500, directed=True, nodeHighlightBehavior=True, highlightColor="#F7A7A6", collapsible=True)
                
                agraph(nodes=nodes, edges=edges, config=config)

            except Exception as e:
                st.error(f"Failed to build Deal Graph: {e}")
                logger.error(f"Deal Graph Error: {e}", exc_info=True)
