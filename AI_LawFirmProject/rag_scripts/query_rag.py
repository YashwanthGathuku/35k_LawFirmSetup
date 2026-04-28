import os
import sys
import argparse
import logging
from llama_index.core  import VectorStoreIndex,  StorageContext,  load_index_from_storage,  Settings
from llama_index.vector_stores.chroma  import ChromaVectorStore
from llama_index.embeddings.huggingface  import HuggingFaceEmbedding
from llama_index.llms.openai  import OpenAI
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
args  =  parser.parse_args()
question  =  args.question

#  2.  GET  LLM  SERVER  ADDRESS  FROM  ENVIRONMENT  VARIABLE
LLM_API_BASE  =  os.getenv("LLM_API_BASE")
if not LLM_API_BASE:
    print("Error:  The  LLM_API_BASE  environment  variable  was  not  set.")
    sys.exit(1)

#  3.  INITIALIZE  MODELS  AND  SETTINGS
try:
    llm  =  OpenAI(model="local-model",  api_base=LLM_API_BASE,  api_key="dummy")
    embed_model  =  HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.llm  =  llm
    Settings.embed_model  =  embed_model
except Exception  as e:
    print(f"Error  initializing  models:  {e}")
    sys.exit(1)

#  4.  LOAD  VECTOR  DATABASE  AND  INDEX
try:
    db_path  =  "/app/chroma_db"
    storage_path  =  "/app/storage"
    db  =  chromadb.PersistentClient(path=db_path)
    chroma_collection  =  db.get_or_create_collection("my_collection")
    vector_store  =  ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context  =  StorageContext.from_defaults(persist_dir=storage_path, vector_store=vector_store)

    index  =  load_index_from_storage(storage_context)
except Exception  as e:
    print(f"Error  loading  vector  database  or  index:  {e}")
    sys.exit(1)

#  5.  QUERY  THE  PIPELINE  AND  PRINT  THE  ANSWER
try:
    query_engine  =  index.as_query_engine()
    response  =  query_engine.query(question)
    print(str(response))
except Exception  as e:
    print(f"Error  during  querying:  {e}")
    sys.exit(1)
