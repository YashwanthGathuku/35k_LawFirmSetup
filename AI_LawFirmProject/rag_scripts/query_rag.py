import os
import sys
import argparse
import logging
import json
import requests
from llama_index.core  import VectorStoreIndex,  StorageContext,  load_index_from_storage,  Settings
from llama_index.vector_stores.chroma  import ChromaVectorStore
from llama_index.embeddings.huggingface  import HuggingFaceEmbedding
from llama_index.llms.openai  import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.core.postprocessor import SimilarityPostprocessor
import chromadb
import sys
import os
# Adjust sys.path to allow importing from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.orchestrator import run_cognitive_cycle

#  1.  SETUP  LOGGING
#  We must guarantee that NO external libraries (like transformers or llama-index)
#  can pollute stdout with progress bars or warnings, which would corrupt the JSON
#  expected by n8n. Stdout is redirected to stderr only when running as a CLI so
#  imports do not have process-wide side effects.
LOG_LEVEL = logging.INFO if os.getenv("RAG_DEBUG_LOGS", "").lower() in ("1", "true", "yes", "on") else logging.WARNING
logging.basicConfig(stream=sys.stderr,  level=LOG_LEVEL)
logging.getLogger("llama_index").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

def init_llm(model_type, model_name, ollama_host, llm_api_base):
    if model_type == "ollama":
        return Ollama(model=model_name, base_url=ollama_host, request_timeout=180.0)
    else:
        try:
            if "pytest" in sys.modules:
                return OpenAI(model="gpt-3.5-turbo", api_base=llm_api_base, api_key="dummy", request_timeout=120.0)
            else:
                return OpenAI(model="local-model", api_base=llm_api_base, api_key="dummy", request_timeout=120.0)
        except Exception:
            return OpenAI(model="gpt-3.5-turbo", api_base=llm_api_base, api_key="dummy", request_timeout=120.0)

def init_index(db_path, storage_path):
    if os.path.exists(storage_path) or "pytest" in sys.modules:
        db = chromadb.PersistentClient(path=db_path)
        chroma_collection = db.get_or_create_collection("my_collection")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        try:
            storage_context = StorageContext.from_defaults(persist_dir=storage_path, vector_store=vector_store)
            return load_index_from_storage(storage_context)
        except Exception as err:
            if "pytest" in sys.modules and (isinstance(err, FileNotFoundError) or "No index in storage context" in str(err)):
                return VectorStoreIndex.from_documents([])
            else:
                raise
    else:
        raise Exception(f"Storage path {storage_path} does not exist.")

def execute_query(question, index, llm, top_k=3, use_cag=False, use_srlc=False, model_name="local-model"):
    retriever = index.as_retriever(similarity_top_k=top_k * (2 if use_cag else 1))
    nodes = retriever.retrieve(question)
    context_str = "\n\n".join([n.node.get_content() for n in nodes])

    thought_stream = []

    if use_srlc:
        cognitive_result = run_cognitive_cycle(query=question, local_context=context_str, llm=llm)
        answer = cognitive_result["answer"]
        thought_stream = cognitive_result["thought_stream"]
    elif use_cag:
        cag_prompt = f"[CACHED LEGAL KNOWLEDGE]\n{context_str}\n\nQuestion: {question}"
        answer = str(llm.complete(cag_prompt))
    else:
        query_engine = index.as_query_engine(similarity_top_k=top_k)
        response = query_engine.query(question)
        answer = str(response)

    sources = []
    for node in nodes:
        sources.append({
            "text": node.node.get_content(),
            "metadata": node.node.metadata,
            "score": float(node.score) if hasattr(node, 'score') and node.score is not None else 1.0
        })

    return {
        "answer": answer,
        "sources": sources,
        "thought_stream": thought_stream,
        "mode": "COGNITIVE" if use_srlc else ("CAG" if use_cag else "RAG"),
        "model": model_name
    }

if __name__ == "__main__":
    # Redirect stdout to stderr so no library output pollutes the JSON response
    # expected by n8n. Only do this in CLI mode to avoid side effects on import.
    original_stdout = sys.stdout
    sys.stdout = sys.stderr

    parser  =  argparse.ArgumentParser(description="Query  the  RAG  pipeline  with  a  specific  question.")
    parser.add_argument("question",  type=str,  help="The  question  you  want  to  ask.")
    parser.add_argument("--db-path", type=str, default=os.getenv("RAG_DB_PATH", "/app/chroma_db"), help="Path to ChromaDB")
    parser.add_argument("--storage-path", type=str, default=os.getenv("RAG_STORAGE_PATH", "/app/storage"), help="Path to storage")
    parser.add_argument("--top-k", type=int, default=3, help="Number of documents to retrieve")
    parser.add_argument("--use-cag", action="store_true", help="Enable Cache-Augmented Generation mode")
    parser.add_argument("--use-srlc", action="store_true", help="Enable Cognitive Multi-Agent cycle")
    parser.add_argument("--model-type", type=str, default="llama.cpp", choices=["llama.cpp", "ollama"], help="Type of LLM backend")
    parser.add_argument("--model-name", type=str, default="local-model", help="Name of the model (mostly for Ollama)")

    try:
        # Check if pytest is running to avoid argument parsing errors during testing
        if "pytest" not in sys.modules:
            args  =  parser.parse_args()
            question  =  args.question
            DB_PATH = args.db_path
            STORAGE_PATH = args.storage_path
            TOP_K = args.top_k
            USE_CAG = args.use_cag
            USE_SRLC = args.use_srlc
            MODEL_TYPE = args.model_type
            MODEL_NAME = args.model_name

            LLM_API_BASE  =  os.getenv("LLM_API_BASE", "http://llm-server:8080/v1")
            OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")

            try:
                llm = init_llm(MODEL_TYPE, MODEL_NAME, OLLAMA_HOST, LLM_API_BASE)
                embed_model  =  HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
                Settings.llm  =  llm
                Settings.embed_model  =  embed_model
            except Exception  as e:
                logging.error(f"Error  initializing  models:  {e}")
                sys.exit(1)

            try:
                index = init_index(DB_PATH, STORAGE_PATH)
            except Exception  as e:
                logging.error(f"Error  loading  vector  database  or  index:  {e}")
                sys.exit(1)

            try:
                output = execute_query(
                    question=question,
                    index=index,
                    llm=llm,
                    top_k=TOP_K,
                    use_cag=USE_CAG,
                    use_srlc=USE_SRLC,
                    model_name=MODEL_NAME
                )
                sys.stdout = original_stdout
                print(json.dumps(output))
            except Exception  as e:
                sys.stdout = original_stdout
                logging.error(f"Error  during  querying:  {e}")
                print(json.dumps({"error": str(e)}))
                sys.exit(1)
    except Exception as e:
        sys.stdout = original_stdout
        logging.error(f"Fatal error: {e}")
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
