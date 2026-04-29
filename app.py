"""
Tegifa Legal — Source-Grounded Legal Intelligence
Main Streamlit application with direct RAG pipeline integration.
"""
import streamlit as st
import streamlit_authenticator as stauth
import requests
import os
import json
import html
import logging
import yaml
from yaml.loader import SafeLoader

from db.persistence import save_chat_session, load_chat_session, list_case_sessions
from rag_scripts.query_rag import init_llm, init_index, execute_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tegifa.app")

# --- Page Configuration ---
st.set_page_config(
    page_title="Tegifa Legal — Source-Grounded Intelligence",
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


# --- Auth Setup ---
with open("config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

# Override cookie key from environment variable (mandatory in production)
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
    config.get("pre-authorized", {}),
)

name, authentication_status, username = authenticator.login("Login", "main")

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
SIDEBAR_ICON_PATH = os.getenv("SIDEBAR_ICON_PATH", "/app/assets/sidebar-icon.png")
MAX_QUESTION_LENGTH = 1000
MAX_TOP_K = 10
PRIVACY_MODE_STRICT = os.getenv("PRIVACY_MODE_STRICT", "false").lower() in {"1", "true", "yes", "on"}



def sanitize_html(text: str) -> str:
    """Escape HTML to prevent XSS in unsafe_allow_html contexts."""
    return html.escape(str(text))


def get_ollama_models() -> list[str]:
    """Fetch available Ollama models with proper error handling."""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if response.status_code == 200:
            return [m["name"] for m in response.json().get("models", [])]
    except requests.RequestException as e:
        logger.warning("Failed to fetch Ollama models: %s", e)
    return []


# --- Sidebar ---
with st.sidebar:
    if os.path.exists(SIDEBAR_ICON_PATH):
        st.image(SIDEBAR_ICON_PATH, width=100)
    else:
        st.markdown("## ⚖️")

    st.title("Tegifa Legal")
    st.markdown(f"Welcome, **{sanitize_html(name)}**")
    authenticator.logout("Logout", "sidebar")
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
    try:
        existing_cases = list_case_sessions(username)
    except Exception as e:
        logger.error("Failed to list case sessions: %s", e)
        existing_cases = []
    new_case = st.text_input("New case session", value="")
    case_options = existing_cases or ["Default Case"]
    if new_case.strip() and new_case.strip() not in case_options:
        case_options = [new_case.strip()] + case_options
    selected_case = st.selectbox("Case session", case_options, key="selected_case")

    st.markdown("---")
    st.subheader("🔬 Advanced Logic")
    use_srlc = st.toggle(
        "Self-Reflective Critique (SRLC)",
        value=True,
        help="Multi-agent algorithm that reviews and critiques its own answers.",
    )
    use_cag = st.toggle("CAG Optimization", value=False)

    st.markdown("---")
    if st.button("🔄 Sync & Re-index Docs"):
        with st.spinner("Re-indexing knowledge base..."):
            try:
                from rag_scripts.rag_setup import run_ingestion

                result = run_ingestion(db_path=DB_PATH, docs_path="/app/docs", storage_path=STORAGE_PATH)
                st.success(f"Re-indexing complete. {result}")
            except Exception as e:
                logger.error("Re-indexing failed: %s", e)
                st.error(f"Re-indexing failed: {e}")


# --- Initialize RAG pipeline (cached per session) ---
@st.cache_resource(show_spinner="Loading intelligence engine...")
def load_rag_pipeline(backend_type: str, model: str):
    """Initialize LLM and vector index once per session."""
    llm = init_llm(
        model_type=backend_type,
        model_name=model,
        ollama_host=OLLAMA_HOST,
        llm_api_base=LLM_API_BASE,
    )
    try:
        index = init_index(db_path=DB_PATH, storage_path=STORAGE_PATH)
    except Exception as e:
        logger.warning("Could not load existing index: %s. Creating empty index.", e)
        from llama_index.core import VectorStoreIndex
        index = VectorStoreIndex.from_documents([])
    return llm, index


# --- Main UI ---
st.title("⚖️ Legal Intelligence Portal")
st.caption(f"Engine: {backend} | Model: {model_name} | Mode: {'SRLC' if use_srlc else 'Standard'}")
if PRIVACY_MODE_STRICT:
    st.warning('🔒 External search disabled (PRIVACY_MODE_STRICT=true).')

# Reload chat history when the selected case changes or on first load
_current_case = st.session_state.get("selected_case", "Default Case")
if "messages" not in st.session_state or st.session_state.get("_last_loaded_case") != _current_case:
    try:
        st.session_state.messages = load_chat_session(username, _current_case)
        st.session_state._last_loaded_case = _current_case
    except Exception as e:
        logger.error("Failed to load chat history: %s", e)
        st.session_state.messages = []

# Display chat history
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

        if message.get("sources"):
            with st.expander(f"📚 Sources & Citations ({len(message['sources'])} verified)"):
                for src in message["sources"]:
                    meta = src.get("metadata", {})
                    fname = sanitize_html(
                        meta.get("file_name")
                        or (os.path.basename(meta.get("file_path", "")) if meta.get("file_path") else "Unknown Source")
                    )
                    snippet = sanitize_html(src.get("text", "")[:300])
                    st.markdown(
                        f'<div class="source-card"><strong>{fname}</strong>: {snippet}...</div>',
                        unsafe_allow_html=True,
                    )


# --- Chat Input ---
query = st.chat_input("Enter legal inquiry...")

if query:
    query = query.strip()

    if not query:
        st.warning("Please enter a non-empty question.")
    elif len(query) > MAX_QUESTION_LENGTH:
        st.error(f"Your question exceeds the {MAX_QUESTION_LENGTH}-character limit. Please shorten it.")
    else:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Tegifa is analyzing (Self-Reflective Critique active)..." if use_srlc else "Tegifa is thinking..."):
                try:
                    llm, index = load_rag_pipeline(backend, model_name)

                    result = execute_query(
                        question=query,
                        index=index,
                        llm=llm,
                        top_k=3,
                        use_cag=use_cag,
                        use_srlc=use_srlc,
                        model_name=model_name,
                    )

                    answer_text = result.get("answer", "No answer generated.")
                    sources = result.get("sources", [])
                    thought_stream = result.get("thought_stream", [])

                    # Show Thought Stream
                    if thought_stream:
                        with st.status("Algorithm Thought Process", expanded=False) as status:
                            for step in thought_stream:
                                st.markdown(f"**Step: {sanitize_html(step.get('step', ''))}**")
                                st.markdown(
                                    f'<div class="thought-bubble">{sanitize_html(step.get("content", ""))}</div>',
                                    unsafe_allow_html=True,
                                )
                            status.update(label="Critique & Refinement Complete", state="complete")

                    st.markdown(answer_text)

                    if sources:
                        with st.expander(f"📚 Sources & Citations ({len(sources)} verified)"):
                            for src in sources:
                                meta = src.get("metadata", {})
                                fname = sanitize_html(
                                    meta.get("file_name")
                                    or (os.path.basename(meta.get("file_path", "")) if meta.get("file_path") else "Unknown Source")
                                )
                                snippet = sanitize_html(src.get("text", "")[:300])
                                st.markdown(
                                    f'<div class="source-card"><strong>{fname}</strong>: {snippet}...</div>',
                                    unsafe_allow_html=True,
                                )

                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "thought_stream": thought_stream,
                        "sources": sources,
                    })

                    try:
                        save_chat_session(username, st.session_state.get("selected_case", "Default Case"), st.session_state.messages)
                    except Exception as e:
                        logger.error("Failed to save chat: %s", e)
                        st.error(f"Failed to save to database: {e}")

                except Exception as e:
                    logger.error("Query failed: %s", e, exc_info=True)
                    st.error(f"Error: {e}")
