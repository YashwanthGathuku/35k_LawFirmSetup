import os
import sys
from llama_index.core  import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.vector_stores.chroma  import ChromaVectorStore
from llama_index.embeddings.huggingface  import HuggingFaceEmbedding
import chromadb

#  Define  absolute  paths  for  use  inside  the  Docker  container
DB_PATH  =  "/app/chroma_db"
DOCS_PATH  =  "/app/docs"
STORAGE_PATH  =  "/app/storage"

print("---  Smart  Ingestion  Script  Started  ---")

try:
    db  =  chromadb.PersistentClient(path=DB_PATH)
    chroma_collection  =  db.get_or_create_collection("my_collection")
    vector_store  =  ChromaVectorStore(chroma_collection=chroma_collection)
    embed_model  =  HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.embed_model  =  embed_model
except Exception  as e:
    print(f"Error  initializing  database  or  models:  {e}")
    sys.exit(1)

print("Checking  for  already  indexed  documents...")
try:
    existing_items  =  chroma_collection.get(include=["metadatas"])
    indexed_files  =  set(meta['file_path']  for meta  in existing_items['metadatas'])
    print(f"Found  {len(indexed_files)} source  files  already  in  the  index.")
except Exception:
    indexed_files  =  set()

all_files_on_disk  =  set()
if os.path.exists(DOCS_PATH):
    for filename  in os.listdir(DOCS_PATH):
        all_files_on_disk.add(os.path.join(DOCS_PATH,  filename))

new_files_to_process  =  all_files_on_disk  -  indexed_files

if not new_files_to_process:
    print("No  new  documents  found  to  process.  Exiting.")
    sys.exit(0)

print(f"\nFound  {len(new_files_to_process)} new  document(s)  to  ingest:")
for file  in new_files_to_process:
    print(f"   -  {os.path.basename(file)}")

try:
    storage_context  =  StorageContext.from_defaults(vector_store=vector_store, persist_dir=STORAGE_PATH)

    try:
        index  =  load_index_from_storage(storage_context)
        print("Loaded  existing  index  from  storage.")
    except:
        print("No  existing  index  found.  Creating  a  new  one.")
        index  =  VectorStoreIndex.from_documents([],  storage_context=storage_context)

    for filepath  in new_files_to_process:
        print(f"\nProcessing  '{os.path.basename(filepath)}'...")
        new_document  =  SimpleDirectoryReader(input_files=[filepath]).load_data()
        index.insert_nodes(new_document)
        print(f"Successfully  inserted  '{os.path.basename(filepath)}'  into  the  index.")

    print("\nPersisting  updated  index...")
    index.storage_context.persist(persist_dir=STORAGE_PATH)
    print("---  Smart  Ingestion  Script  Finished  Successfully  ---")

except Exception  as e:
    print(f"\nAn  error  occurred  during  indexing:  {e}")
    sys.exit(1)
