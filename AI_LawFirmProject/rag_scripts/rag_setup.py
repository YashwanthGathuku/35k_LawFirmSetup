import os
import sys
import argparse
from llama_index.core  import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.vector_stores.chroma  import ChromaVectorStore
from llama_index.embeddings.huggingface  import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
import chromadb

#  Define default absolute paths for use inside the Docker container,
#  while allowing overrides via CLI args or environment variables.
parser = argparse.ArgumentParser(description="Ingest documents into the RAG pipeline.")
parser.add_argument("--db-path", type=str, default=os.getenv("RAG_DB_PATH", "/app/chroma_db"), help="Path to ChromaDB")
parser.add_argument("--docs-path", type=str, default=os.getenv("RAG_DOCS_PATH", "/app/docs"), help="Path to documents")
parser.add_argument("--storage-path", type=str, default=os.getenv("RAG_STORAGE_PATH", "/app/storage"), help="Path to storage")
parser.add_argument("--nougat", action="store_true", help="Use Nougat OCR for PDF ingestion")
args = parser.parse_args()

DB_PATH = args.db_path
DOCS_PATH = args.docs_path
STORAGE_PATH = args.storage_path
USE_NOUGAT = args.nougat

print(f"---  LexAI Smart Ingestion Engine Started {'(Deep Ingestion Mode: Nougat)' if USE_NOUGAT else ''} ---")

try:
    db  =  chromadb.PersistentClient(path=DB_PATH)
    chroma_collection  =  db.get_or_create_collection("my_collection")
    vector_store  =  ChromaVectorStore(chroma_collection=chroma_collection)

    # Advanced: Custom node parser for legal documents
    Settings.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)

    embed_model  =  HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.embed_model  =  embed_model
except Exception  as e:
    print(f"Error  initializing  database  or  models:  {e}")
    sys.exit(1)

print("Checking knowledge base status...")
try:
    existing_items  =  chroma_collection.get(include=["metadatas"])
    indexed_files  =  set(meta['file_path']  for meta  in existing_items['metadatas'])
    print(f"Found  {len(indexed_files)} files in index.")
except Exception as e:
    print(f"Starting fresh knowledge base. (Info: {e})")
    indexed_files  =  set()

all_files_on_disk  =  set()
if os.path.exists(DOCS_PATH):
    for filename  in os.listdir(DOCS_PATH):
        all_files_on_disk.add(os.path.join(DOCS_PATH,  filename))

new_files_to_process  =  all_files_on_disk  -  indexed_files

if not new_files_to_process:
    print("Knowledge base is up to date.")
    sys.exit(0)

print(f"Ingesting {len(new_files_to_process)} new document(s)...")

try:
    storage_context  =  StorageContext.from_defaults(vector_store=vector_store, persist_dir=STORAGE_PATH)

    try:
        index  =  load_index_from_storage(storage_context)
        print("Connected to existing index.")
    except Exception:
        print("Initializing new vector index.")
        index  =  VectorStoreIndex.from_documents([],  storage_context=storage_context)

    # Optional Nougat Reader
    nougat_reader = None
    if USE_NOUGAT:
        try:
            from llama_index.readers.nougat_ocr import PDFNougatOCR
            nougat_reader = PDFNougatOCR()
            print("Nougat OCR initialized for high-accuracy PDF parsing.")
        except ImportError:
            print("Warning: llama-index-readers-nougat-ocr not found. Falling back to default reader.")

    for filepath  in new_files_to_process:
        print(f"Processing: {os.path.basename(filepath)}")

        if nougat_reader and filepath.lower().endswith(".pdf"):
            try:
                new_document = nougat_reader.load_data(filepath)
                print(f"Used Nougat OCR for '{os.path.basename(filepath)}'.")
            except Exception as e:
                print(f"Nougat failed for '{os.path.basename(filepath)}': {e}. Falling back to default.")
                new_document = SimpleDirectoryReader(input_files=[filepath]).load_data()
        else:
            new_document  =  SimpleDirectoryReader(input_files=[filepath]).load_data()

        index.insert_nodes(Settings.text_splitter.get_nodes_from_documents(new_document))
        print(f"Successfully indexed '{os.path.basename(filepath)}'.")

    print("Persisting changes...")
    index.storage_context.persist(persist_dir=STORAGE_PATH)
    print("---  Ingestion Complete  ---")

except Exception  as e:
    print(f"Ingestion Failed: {e}")
    sys.exit(1)
