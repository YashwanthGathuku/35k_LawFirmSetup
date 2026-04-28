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
        requests.post(N8N_WEBHOOK_URL, json={"use_nougat": use_nougat})
        st.info("Re-indexing started.")

# --- Main UI ---
st.title("⚖️ Legal Intelligence Portal")
st.caption(f"Engine: {backend} | Model: {model_name} | Mode: {'SRLC' if use_srlc else 'Standard'}")

query = st.chat_input("Enter complex legal inquiry...")

if query:
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
                result = response.json()

                raw_stdout = result[0].get('stdout', '') if isinstance(result, list) else result.get('stdout', '')
                data = json.loads(raw_stdout)

                # Show Thought Stream
                if data.get("thought_stream"):
                    with st.status("Algorithm Thought Process", expanded=False) as status:
                        for step in data["thought_stream"]:
                            st.markdown(f"**Step: {step['step']}**")
                            st.markdown(f'<div class="thought-bubble">{step["content"]}</div>', unsafe_allow_html=True)
                        status.update(label="Critique & Refinement Complete", state="complete")

                st.markdown(data.get("answer", "Inquiry failed."))

                if data.get("sources"):
                    with st.expander("📚 Sources & Citations"):
                        for src in data["sources"]:
                            st.markdown(f'<div class="source-card"><strong>{src["metadata"].get("file_name")}</strong>: {src["text"][:300]}...</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {e}")
