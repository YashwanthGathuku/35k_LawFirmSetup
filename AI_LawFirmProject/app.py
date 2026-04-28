import streamlit as st
import requests
import os
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="LexAI - Premium Legal Assistant",
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
    st.title("LexAI Dashboard")
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
    st.subheader("⚙️ Settings")
    temperature = st.slider("Precision vs Creativity", 0.0, 1.0, 0.1)
    st.info("High precision is recommended for legal analysis.")

# --- Main UI ---
st.title("⚖️ LexAI: Premium Legal Assistant")
st.markdown("Welcome back. Please enter your legal inquiry below for confidential RAG-powered analysis.")

query = st.chat_input("Ask a question about your documents...")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing case law and documents..."):
            try:
                payload = {"question": query}
                response = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=90)
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
                except:
                    answer = raw_stdout
                    sources = []

                st.markdown(answer)

                if sources:
                    with st.expander("🔍 Verified Sources & Citations"):
                        for idx, src in enumerate(sources):
                            meta = src.get('metadata', {})
                            fname = meta.get('file_name', 'Unknown Source')
                            score = src.get('score', 0)
                            text = src.get('text', '')

                            st.markdown(f"""
                            <div class="source-card">
                                <strong>Source {idx+1}: {fname}</strong> (Relevance: {score:.2f})<br>
                                <small>"{text[:300]}..."</small>
                            </div>
                            """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Inquiry Failed: {str(e)}")
                st.info("Please ensure the n8n workflow and LLM server are online.")

# --- Footer ---
st.markdown("---")
st.caption("Confidential & Secure | powered by Private LLM & LexAI RAG Engine")
