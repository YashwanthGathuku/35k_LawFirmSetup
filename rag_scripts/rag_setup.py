"""
Tegifa Legal — Document Ingestion Engine
Indexes legal documents using the advanced document processor:
- Nougat OCR for scanned/complex PDFs
- Page-level metadata preserved through chunking
- Clause pre-tagging for contract review
- Table detection
"""
import os
import sys
import argparse
import logging
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
    Document,
)
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

from rag_scripts.document_processor import process_file, chunk_to_metadata

logger = logging.getLogger("tegifa.ingestion")


def run_ingestion(
    matter_id: int,
    db_path: str = "/app/chroma_db",
    docs_path: str = "/app/docs",
    storage_path: str = "/app/storage",
    use_nougat: bool = True,
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
) -> str:
    """
    Ingest new documents using the advanced document processor.

    Each chunk is stored with full metadata:
    - page_number, paragraph_index, total_pages
    - detected_clauses, contains_table, section_heading
    - char_offset_start, char_offset_end

    This metadata powers pinpoint citations and contract review.
    """
    logger.info(
        "Ingestion started (Nougat=%s) db=%s docs=%s",
        use_nougat, db_path, docs_path,
    )

    # Initialize ChromaDB
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection("tegifa_legal")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Configure embedding model
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

    logger.info("Ingesting %d new document(s) for matter %s...", len(new_files), matter_id)

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

    # Process each file with the advanced processor
    processed_files = []
    total_chunks = 0

    for filepath in sorted(new_files):
        basename = os.path.basename(filepath)
        logger.info("Processing: %s", basename)

        try:
            doc = process_file(
                filepath=filepath,
                use_nougat=use_nougat,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            if not doc.chunks:
                logger.warning("No content extracted from %s", basename)
                continue

            # Convert ProcessedChunks to LlamaIndex TextNodes with full metadata
            nodes = []
            for chunk in doc.chunks:
                metadata = chunk_to_metadata(chunk)
                metadata["document_type"] = doc.document_type
                metadata["matter_id"] = matter_id

                node = TextNode(
                    text=chunk.text,
                    metadata=metadata,
                    excluded_embed_metadata_keys=[
                        "char_offset_start",
                        "char_offset_end",
                        "chunk_index",
                        "total_chunks",
                    ],
                )
                nodes.append(node)

            index.insert_nodes(nodes)
            total_chunks += len(nodes)
            processed_files.append(
                f"{basename} ({doc.total_pages}p, {len(nodes)} chunks, "
                f"type={doc.document_type}, tables={doc.detected_tables})"
            )

            logger.info(
                "Indexed: %s — %d pages, %d chunks, type=%s",
                basename, doc.total_pages, len(nodes), doc.document_type,
            )

        except Exception as e:
            logger.error("Failed to process %s: %s", basename, e)

    # Persist
    index.storage_context.persist(persist_dir=storage_path)

    summary = (
        f"Ingested {len(processed_files)} document(s), {total_chunks} total chunks:\n"
        + "\n".join(f"  • {f}" for f in processed_files)
    )
    logger.info(summary)
    return summary


# --- CLI entrypoint ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tegifa Document Ingestion Engine")
    parser.add_argument("--db-path", type=str, default=os.getenv("RAG_DB_PATH", "/app/chroma_db"))
    parser.add_argument("--docs-path", type=str, default=os.getenv("RAG_DOCS_PATH", "/app/docs"))
    parser.add_argument("--storage-path", type=str, default=os.getenv("RAG_STORAGE_PATH", "/app/storage"))
    parser.add_argument("--nougat", action="store_true", default=True)
    parser.add_argument("--no-nougat", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--chunk-overlap", type=int, default=128)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    result = run_ingestion(
        db_path=args.db_path,
        docs_path=args.docs_path,
        storage_path=args.storage_path,
        use_nougat=not args.no_nougat,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(result)
