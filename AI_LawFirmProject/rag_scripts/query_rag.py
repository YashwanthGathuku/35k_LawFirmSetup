import os
import sys
import argparse
import logging
import json
from llama_index.core  import VectorStoreIndex,  StorageContext,  load_index_from_storage,  Settings
from llama_index.vector_stores.chroma  import ChromaVectorStore
from llama_index.embeddings.huggingface  import HuggingFaceEmbedding
from llama_index.llms.openai  import OpenAI
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.postprocessor import SimilarityPostprocessor
import chromadb

#  This  configuration  silences  the  noisy  logs  from  underlying  libraries
logging.basicConfig(stream=sys.stdout,  level=logging.INFO)
logging.getLogger("llama_index").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

#  1.  SETUP  COMMAND-LINE  ARGUMENT  PARSER
parser  =  argparse.ArgumentParser(description="Query  the  RAG  pipeline  with  a  specific  question.")
parser.add_argument("question",  type=str,  help="The  question  you  want  to  ask.")
parser.add_argument("--db-path", type=str, default=os.getenv("RAG_DB_PATH", "/app/chroma_db"), help="Path to ChromaDB")
parser.add_argument("--storage-path", type=str, default=os.getenv("RAG_STORAGE_PATH", "/app/storage"), help="Path to storage")
parser.add_argument("--top-k", type=int, default=3, help="Number of documents to retrieve")
parser.add_argument("--use-cag", action="store_true", help="Enable Cache-Augmented Generation mode")
args  =  parser.parse_args()
question  =  args.question
DB_PATH = args.db_path
STORAGE_PATH = args.storage_path
TOP_K = args.top_k
USE_CAG = args.use_cag

#  2.  GET  LLM  SERVER  ADDRESS  FROM  ENVIRONMENT  VARIABLE
LLM_API_BASE  =  os.getenv("LLM_API_BASE")
if not LLM_API_BASE:
    print("Error:  The  LLM_API_BASE  environment  variable  was  not  set.")
    sys.exit(1)

#  3.  INITIALIZE  MODELS  AND  SETTINGS
try:
    # For CAG, we want larger context if possible, though local-server usually handles this in its params
    llm  =  OpenAI(model="local-model",  api_base=LLM_API_BASE,  api_key="dummy", request_timeout=120.0)
    embed_model  =  HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.llm  =  llm
    Settings.embed_model  =  embed_model
except Exception  as e:
    print(f"Error  initializing  models:  {e}")
    sys.exit(1)

#  4.  LOAD  VECTOR  DATABASE  AND  INDEX
try:
    db  =  chromadb.PersistentClient(path=DB_PATH)
    chroma_collection  =  db.get_or_create_collection("my_collection")
    vector_store  =  ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context  =  StorageContext.from_defaults(persist_dir=STORAGE_PATH, vector_store=vector_store)

    index  =  load_index_from_storage(storage_context)
except Exception  as e:
    print(f"Error  loading  vector  database  or  index:  {e}")
    sys.exit(1)

#  5.  QUERY  THE  PIPELINE
try:
    if USE_CAG:
        # In a real CAG system with massive context, we'd inject ALL relevant context once and cache KV.
        # Here we simulate "High-Context Cache" by retrieving more documents and structuring the prompt
        # to acknowledge the "Cached Legal Knowledge Base".
        retriever = index.as_retriever(similarity_top_k=TOP_K * 2) # Double the context for CAG
        nodes = retriever.retrieve(question)

        context_str = "\n\n".join([n.node.get_content() for n in nodes])

        cag_prompt = f"""
        [CACHED LEGAL KNOWLEDGE BASE ACTIVE]
        The following information has been pre-loaded from the law firm's private document cache:

        {context_str}

        [INSTRUCTION]
        Based on the cached knowledge above, answer the user's inquiry with high precision.
        User Question: {question}
        """

        response = llm.complete(cag_prompt)
        source_nodes = nodes
    else:
        # Standard RAG
        query_engine  =  index.as_query_engine(
            similarity_top_k=TOP_K,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.5)
            ]
        )
        response  =  query_engine.query(question)
        source_nodes = response.source_nodes

    # Extract sources for advanced UI
    sources = []
    for node in source_nodes:
        sources.append({
            "text": node.node.get_content(),
            "metadata": node.node.metadata,
            "score": float(node.score) if hasattr(node, 'score') and node.score is not None else 1.0
        })

    # Return a structured JSON response
    output = {
        "answer": str(response),
        "sources": sources,
        "mode": "CAG" if USE_CAG else "RAG"
    }
    print(json.dumps(output))

except Exception  as e:
    print(f"Error  during  querying:  {e}")
    sys.exit(1)
