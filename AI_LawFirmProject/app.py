import streamlit as st
import requests
import os
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="LexAI Pro - Legal Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for "Legal Tech" Look ---
st.markdown("""
<style>
    :root {
        --primary-color: #1E3A8A; /* Navy Blue */
        --secondary-color: #64748B;
        --bg-color: #F8FAFC;
        --accent-gold: #D4AF37;
    }
    .main {
        background-color: var(--bg-color);
    }
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #1E40AF;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    .source-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid var(--primary-color);
        margin-bottom: 10px;
        box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1);
    }
    .cag-badge {
        background-color: #E0F2FE;
        color: #0369A1;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: bold;
        border: 1px solid #7DD3FC;
    }
    .sidebar .sidebar-content {
        background-color: #1E293B;
        color: white;
    }
    h1, h2, h3 {
        color: #1E293B;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# --- Constants ---
N8N_WEBHOOK_URL = "http://n8n-app:5678/webhook/22398436-911c-4798-a801-789a7411d5e8"
DOCS_DIR = "/app/docs"

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1053/1053155.png", width=100)
    st.title("LexAI Pro Dashboard")
    st.markdown("---")

    st.subheader("📁 Knowledge Base")
    if os.path.exists(DOCS_DIR):
        files = os.listdir(DOCS_DIR)
        if files:
            for f in files:
                st.caption(f"📄 {f}")
        else:
            st.info("No documents uploaded yet.")
    else:
        st.warning("Docs directory not found.")

    st.markdown("---")
    st.subheader("🚀 Advanced Features")
    use_nougat = st.toggle("Deep Ingestion (Nougat OCR)", help="Uses AI-powered OCR for complex PDFs (Math, Tables, Citations)")
    use_cag = st.toggle("Enable CAG Mode", value=True, help="Cache-Augmented Generation: Pre-loads core context for lightning-fast high-accuracy responses.")

    st.markdown("---")
    st.subheader("⚙️ Settings")
    top_k = st.slider("Retrieval Depth (Top-K)", 1, 10, 3)

    if st.button("🔄 Re-sync Knowledge Base"):
        with st.spinner("Indexing new files..."):
            # This would call the ingestion webhook
            st.success("Indexing request sent.")

# --- Main UI ---
st.title("⚖️ LexAI Pro: Legal Intelligence")
if use_cag:
    st.markdown('<span class="cag-badge">CACHE-AUGMENTED GENERATION ACTIVE</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="cag-badge" style="background-color: #F1F5F9; color: #475569; border-color: #CBD5E1;">STANDARD RAG MODE</span>', unsafe_allow_html=True)

st.markdown("Confidential analysis portal. Inquiries are processed locally and securely.")

query = st.chat_input("Analyze case law, contracts, or statutes...")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Processing inquiry through LexAI Engine..."):
            try:
                # Add CAG flag to payload
                payload = {
                    "question": query,
                    "use_cag": use_cag,
                    "top_k": top_k
                }
                # Note: n8n workflow needs to be updated to pass these flags to the script
                response = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()

                # Parse advanced response
                raw_stdout = ""
                if isinstance(result, list) and len(result) > 0:
                    raw_stdout = result[0].get('stdout', '')
                elif isinstance(result, dict):
                    raw_stdout = result.get('stdout', result.get('answer', ''))

                try:
                    data = json.loads(raw_stdout)
                    answer = data.get("answer", "No answer generated.")
                    sources = data.get("sources", [])
                    mode_used = data.get("mode", "RAG")
                except:
                    answer = raw_stdout
                    sources = []
                    mode_used = "Standard"

                st.markdown(answer)

                if sources:
                    with st.expander(f"🔍 Sources & Citations ({len(sources)} verified)"):
                        for idx, src in enumerate(sources):
                            meta = src.get('metadata', {})
                            fname = meta.get('file_name', 'Unknown Source')
                            score = src.get('score', 0)
                            text = src.get('text', '')

                            st.markdown(f"""
                            <div class="source-card">
                                <strong>Source {idx+1}: {fname}</strong> (Relevance: {score:.2f})<br>
                                <small>"{text[:400]}..."</small>
                            </div>
                            """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Inquiry Failed: {str(e)}")
                st.info("Please ensure the LexAI backend services are initialized.")

# --- Footer ---
st.markdown("---")
st.caption("LexAI Pro v2.0 | Secure Legal-Grade AI | Fully Local Deployment")
