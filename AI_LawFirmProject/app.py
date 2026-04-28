import streamlit as st
import requests
import os
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="LexAI Pro v3.0 - breakthrough Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    :root {
        --primary-color: #1E3A8A;
        --accent-gold: #D4AF37;
    }
    .thought-bubble {
        background-color: #F1F5F9;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid var(--accent-gold);
        margin: 10px 0;
        font-family: 'Courier New', Courier, monospace;
        font-size: 0.9rem;
    }
    .source-card {
        background-color: white;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- Constants & Helpers ---
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "http://n8n-app:5678/webhook/22398436-911c-4798-a801-789a7411d5e8")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
DOCS_DIR = os.getenv("DOCS_DIR", "/app/docs")
SIDEBAR_ICON_PATH = os.getenv("SIDEBAR_ICON_PATH", "/app/assets/sidebar-icon.png")

def get_ollama_models():
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if response.status_code == 200:
            return [m['name'] for m in response.json().get('models', [])]
    except:
        pass
    return []

# --- Sidebar ---
with st.sidebar:
    if os.path.exists(SIDEBAR_ICON_PATH):
        st.image(SIDEBAR_ICON_PATH, width=100)
    else:
        st.markdown("## ⚖️")

    st.title("LexAI Pro v3.0")
    st.markdown("---")

    st.subheader("🤖 Model Hub")
    backend = st.selectbox("Backend Engine", ["llama.cpp", "ollama"])

    ollama_models = get_ollama_models()
    if backend == "ollama":
        if ollama_models:
            model_name = st.selectbox("Choose Local Model", ollama_models)
        else:
            st.warning("No Ollama models found. Pull some via CLI!")
            model_name = "llama3"
    else:
        model_name = "local-llama-cpp"

    st.markdown("---")
    st.subheader("🔬 Advanced Logic")
    use_srlc = st.toggle("Self-Reflective Critique (SRLC)", value=True, help="Breakthrough algorithm that makes the AI review and correct its own answers.")
    use_cag = st.toggle("CAG Optimization", value=False)
    use_nougat = st.toggle("Deep Ingestion (Nougat)", value=False)

    st.markdown("---")
    if st.button("🔄 Sync & Re-index Docs"):
        with st.spinner("Requesting knowledge base sync..."):
            try:
                # Trigger ingestion by passing an 'ingest' flag
                requests.post(N8N_WEBHOOK_URL, json={"ingest": True, "use_nougat": use_nougat}, timeout=10)
                st.success("Re-indexing started.")
            except Exception as e:
                st.error(f"Sync request failed: {e}")

# --- Main UI ---
st.title("⚖️ Legal Intelligence Portal")
st.caption(f"Engine: {backend} | Model: {model_name} | Mode: {'SRLC' if use_srlc else 'Standard'}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Show thought stream if it exists
        if message.get("thought_stream"):
            with st.status("Algorithm Thought Process", expanded=False, state="complete"):
                for step in message["thought_stream"]:
                    st.markdown(f"**Step: {step['step']}**")
                    st.markdown(f'<div class="thought-bubble">{step["content"]}</div>', unsafe_allow_html=True)

        st.markdown(message["content"])

        # Show sources if they exist
        if message.get("sources"):
            with st.expander(f"📚 Sources & Citations ({len(message['sources'])} verified)"):
                for src in message["sources"]:
                    meta = src.get('metadata', {})
                    file_name = meta.get('file_name')
                    file_path = meta.get('file_path')
                    fname = file_name or (os.path.basename(file_path) if file_path else 'Unknown Source')
                    st.markdown(f'<div class="source-card"><strong>{fname}</strong>: {src["text"][:300]}...</div>', unsafe_allow_html=True)

query = st.chat_input("Enter complex legal inquiry...")

if query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("LexAI is thinking (Self-Reflective Algorithm Active)..."):
            try:
                payload = {
                    "question": query,
                    "use_cag": use_cag,
                    "use_srlc": use_srlc,
                    "model_type": backend,
                    "model_name": model_name
                }
                response = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=240)
                response.raise_for_status()
                result = response.json()

                # Check for direct answer from webhook (backward compatibility)
                answer_text = ""
                sources = []
                thought_stream = []

                if isinstance(result, list) and len(result) > 0:
                    raw_stdout = result[0].get('stdout', '')
                    answer_text = result[0].get('answer', '')
                else:
                    raw_stdout = result.get('stdout', '')
                    answer_text = result.get('answer', '')

                # If raw_stdout contains JSON, parse it for advanced data
                if raw_stdout:
                    try:
                        data = json.loads(raw_stdout)
                        answer_text = data.get("answer", answer_text)
                        sources = data.get("sources", [])
                        thought_stream = data.get("thought_stream", [])
                    except json.JSONDecodeError:
                        # Fallback to treat raw_stdout as plain text answer if not JSON
                        if not answer_text:
                            answer_text = raw_stdout

                # Show Thought Stream
                if thought_stream:
                    with st.status("Algorithm Thought Process", expanded=False) as status:
                        for step in thought_stream:
                            st.markdown(f"**Step: {step['step']}**")
                            st.markdown(f'<div class="thought-bubble">{step["content"]}</div>', unsafe_allow_html=True)
                        status.update(label="Critique & Refinement Complete", state="complete")

                st.markdown(answer_text if answer_text else "No answer generated.")

                if sources:
                    with st.expander(f"📚 Sources & Citations ({len(sources)} verified)"):
                        for src in sources:
                            meta = src.get('metadata', {})
                            file_name = meta.get('file_name')
                            file_path = meta.get('file_path')
                            fname = file_name or (os.path.basename(file_path) if file_path else 'Unknown Source')
                            st.markdown(f'<div class="source-card"><strong>{fname}</strong>: {src["text"][:300]}...</div>', unsafe_allow_html=True)

                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text if answer_text else "No answer generated.",
                    "thought_stream": thought_stream,
                    "sources": sources
                })

            except Exception as e:
                st.error(f"Error: {e}")
