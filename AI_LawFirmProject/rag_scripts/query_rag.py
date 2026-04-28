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
from srlc_engine import SRLCEngine

#  1.  SETUP  LOGGING & STDOUT REDIRECTION
#  We must guarantee that NO external libraries (like transformers or llama-index)
#  can pollute stdout with progress bars or warnings, which would corrupt the JSON
#  expected by n8n. We redirect stdout to stderr for the duration of execution.
original_stdout = sys.stdout
sys.stdout = sys.stderr

LOG_LEVEL = logging.INFO if os.getenv("RAG_DEBUG_LOGS", "").lower() in ("1", "true", "yes", "on") else logging.WARNING
logging.basicConfig(stream=sys.stderr,  level=LOG_LEVEL)
logging.getLogger("llama_index").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

#  2.  SETUP  COMMAND-LINE  ARGUMENT  PARSER
parser  =  argparse.ArgumentParser(description="Query  the  RAG  pipeline  with  a  specific  question.")
parser.add_argument("question",  type=str,  help="The  question  you  want  to  ask.")
parser.add_argument("--db-path", type=str, default=os.getenv("RAG_DB_PATH", "/app/chroma_db"), help="Path to ChromaDB")
parser.add_argument("--storage-path", type=str, default=os.getenv("RAG_STORAGE_PATH", "/app/storage"), help="Path to storage")
parser.add_argument("--top-k", type=int, default=3, help="Number of documents to retrieve")
parser.add_argument("--use-cag", action="store_true", help="Enable Cache-Augmented Generation mode")
parser.add_argument("--use-srlc", action="store_true", help="Enable Self-Reflective Legal Critique algorithm")
parser.add_argument("--model-type", type=str, default="llama.cpp", choices=["llama.cpp", "ollama"], help="Type of LLM backend")
parser.add_argument("--model-name", type=str, default="local-model", help="Name of the model (mostly for Ollama)")

args  =  parser.parse_args()
question  =  args.question
DB_PATH = args.db_path
STORAGE_PATH = args.storage_path
TOP_K = args.top_k
USE_CAG = args.use_cag
USE_SRLC = args.use_srlc
MODEL_TYPE = args.model_type
MODEL_NAME = args.model_name

#  3.  GET  LLM  SERVER  ADDRESSES
LLM_API_BASE  =  os.getenv("LLM_API_BASE", "http://llm-server:8080/v1")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")

#  4.  INITIALIZE  MODELS  AND  SETTINGS
try:
    if MODEL_TYPE == "ollama":
        llm = Ollama(model=MODEL_NAME, base_url=OLLAMA_HOST, request_timeout=180.0)
    else:
        llm  =  OpenAI(model="local-model",  api_base=LLM_API_BASE,  api_key="dummy", request_timeout=120.0)

    embed_model  =  HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.llm  =  llm
    Settings.embed_model  =  embed_model
except Exception  as e:
    logging.error(f"Error  initializing  models:  {e}")
    sys.exit(1)

#  5.  LOAD  VECTOR  DATABASE  AND  INDEX
try:
    db  =  chromadb.PersistentClient(path=DB_PATH)
    chroma_collection  =  db.get_or_create_collection("my_collection")
    vector_store  =  ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context  =  StorageContext.from_defaults(persist_dir=STORAGE_PATH, vector_store=vector_store)

    index  =  load_index_from_storage(storage_context)
except Exception  as e:
    logging.error(f"Error  loading  vector  database  or  index:  {e}")
    sys.exit(1)

#  6.  QUERY  THE  PIPELINE
try:
    retriever = index.as_retriever(similarity_top_k=TOP_K * (2 if USE_CAG else 1))
    nodes = retriever.retrieve(question)
    context_str = "\n\n".join([n.node.get_content() for n in nodes])

    thought_stream = []

    if USE_SRLC:
        srlc = SRLCEngine(llm)
        srlc_result = srlc.run(question, context_str)
        answer = srlc_result["final"]
        thought_stream = [
            {"step": "Drafting", "content": srlc_result["draft"]},
            {"step": "Self-Critique", "content": srlc_result["critique"]},
            {"step": "Refining", "content": srlc_result["final"]}
        ]
    elif USE_CAG:
        cag_prompt = f"[CACHED LEGAL KNOWLEDGE]\n{context_str}\n\nQuestion: {question}"
        answer = str(llm.complete(cag_prompt))
    else:
        query_engine = index.as_query_engine(similarity_top_k=TOP_K)
        response = query_engine.query(question)
        answer = str(response)

    # Extract sources
    sources = []
    for node in nodes:
        sources.append({
            "text": node.node.get_content(),
            "metadata": node.node.metadata,
            "score": float(node.score) if hasattr(node, 'score') and node.score is not None else 1.0
        })

    output = {
        "answer": answer,
        "sources": sources,
        "thought_stream": thought_stream,
        "mode": "SRLC" if USE_SRLC else ("CAG" if USE_CAG else "RAG"),
        "model": MODEL_NAME
    }

    # Restore stdout strictly for the final JSON payload
    sys.stdout = original_stdout
    print(json.dumps(output))

except Exception  as e:
    # Ensure stdout is restored so n8n can receive the error message if needed
    sys.stdout = original_stdout
    logging.error(f"Error  during  querying:  {e}")
    sys.exit(1)
