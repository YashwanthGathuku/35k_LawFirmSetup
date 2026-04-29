"""
Tegifa Legal — Document Ingestion Engine
Indexes legal documents into ChromaDB with optional Nougat OCR.
Refactored as a callable function (not just CLI script).
"""
import os
import sys
import argparse
import logging
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
import chromadb

logger = logging.getLogger("tegifa.ingestion")


def run_ingestion(
    db_path: str = "/app/chroma_db",
    docs_path: str = "/app/docs",
    storage_path: str = "/app/storage",
    use_nougat: bool = False,
) -> str:
    """
    Ingest new documents into the vector store.

    Returns a summary string of what was processed.
    Can be called programmatically from Streamlit or via CLI.
    """
    logger.info(
        "Ingestion started (Nougat=%s) db=%s docs=%s storage=%s",
        use_nougat, db_path, docs_path, storage_path,
    )

    # Initialize ChromaDB
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection("tegifa_legal")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Configure text splitting for legal documents
    Settings.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Check what's already indexed
    try:
        existing_items = chroma_collection.get(include=["metadatas"])
        indexed_files = {
            meta.get("file_path")
            for meta in existing_items.get("metadatas", [])
            if meta.get("file_path")
        }
        logger.info("Found %d files already indexed.", len(indexed_files))
    except Exception as e:
        logger.info("Starting fresh knowledge base. (%s)", e)
        indexed_files = set()

    # Find new files
    all_files_on_disk = set()
    if os.path.exists(docs_path):
        for filename in os.listdir(docs_path):
            filepath = os.path.join(docs_path, filename)
            if os.path.isfile(filepath):
                all_files_on_disk.add(filepath)

    new_files = all_files_on_disk - indexed_files

    if not new_files:
        return "Knowledge base is up to date. No new documents found."

    logger.info("Ingesting %d new document(s)...", len(new_files))

    # Load or create index
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=storage_path
    )
    try:
        index = load_index_from_storage(storage_context)
        logger.info("Connected to existing index.")
    except Exception:
        logger.info("Initializing new vector index.")
        index = VectorStoreIndex.from_documents([], storage_context=storage_context)

    # Optional Nougat OCR reader
    nougat_reader = None
    if use_nougat:
        try:
            from llama_index.readers.nougat_ocr import PDFNougatOCR

            nougat_reader = PDFNougatOCR()
            logger.info("Nougat OCR initialized.")
        except ImportError:
            logger.warning("Nougat OCR not available. Using default reader.")

    # Process each new file
    processed = []
    for filepath in new_files:
        basename = os.path.basename(filepath)
        logger.info("Processing: %s", basename)

        try:
            if nougat_reader and filepath.lower().endswith(".pdf"):
                try:
                    documents = nougat_reader.load_data(filepath)
                except Exception as e:
                    logger.warning("Nougat failed for %s: %s. Falling back.", basename, e)
                    documents = SimpleDirectoryReader(input_files=[filepath]).load_data()
            else:
                documents = SimpleDirectoryReader(input_files=[filepath]).load_data()

            # Ensure file_name metadata for citations
            for doc in documents:
                if "file_name" not in doc.metadata:
                    doc.metadata["file_name"] = basename
                doc.metadata["file_path"] = filepath

            nodes = Settings.text_splitter.get_nodes_from_documents(documents)
            index.insert_nodes(nodes)
            processed.append(basename)
            logger.info("Indexed: %s (%d chunks)", basename, len(nodes))

        except Exception as e:
            logger.error("Failed to process %s: %s", basename, e)

    # Persist
    index.storage_context.persist(persist_dir=storage_path)
    return f"Ingested {len(processed)} document(s): {', '.join(processed)}"


# --- CLI entrypoint ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tegifa Document Ingestion Engine")
    parser.add_argument(
        "--db-path",
        type=str,
        default=os.getenv("RAG_DB_PATH", "/app/chroma_db"),
    )
    parser.add_argument(
        "--docs-path",
        type=str,
        default=os.getenv("RAG_DOCS_PATH", "/app/docs"),
    )
    parser.add_argument(
        "--storage-path",
        type=str,
        default=os.getenv("RAG_STORAGE_PATH", "/app/storage"),
    )
    parser.add_argument("--nougat", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    result = run_ingestion(
        db_path=args.db_path,
        docs_path=args.docs_path,
        storage_path=args.storage_path,
        use_nougat=args.nougat,
    )
    print(result)
