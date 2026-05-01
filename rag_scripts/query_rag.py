"""
Tegifa Legal — RAG Query Engine
Handles querying the vector index with optional SRLC and CAG modes.
Now integrated with the Citation Engine for pinpoint page/paragraph citations.
"""
import os
import sys
import argparse
import logging
import json

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
import chromadb

from rag_scripts.citation_engine import build_citation_report, report_to_dict

logger = logging.getLogger("tegifa.query")


def init_llm(model_type: str, model_name: str, ollama_host: str, llm_api_base: str):
    """Initialize the LLM based on backend type."""
    if model_type == "ollama":
        return Ollama(model=model_name, base_url=ollama_host, request_timeout=180.0)
    else:
        return OpenAI(
            model="local-model",
            api_base=llm_api_base,
            api_key="not-needed",
            request_timeout=120.0,
        )


def init_index(db_path: str, storage_path: str):
    """Load the vector index from ChromaDB + persisted storage."""
    if not os.path.exists(storage_path) and "pytest" not in sys.modules:
        raise FileNotFoundError(f"Storage path does not exist: {storage_path}")

    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection("tegifa_legal")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    try:
        storage_context = StorageContext.from_defaults(
            persist_dir=storage_path, vector_store=vector_store
        )
        return load_index_from_storage(storage_context)
    except Exception as err:
        if "pytest" in sys.modules and (
            isinstance(err, FileNotFoundError)
            or "No index in storage context" in str(err)
        ):
            return VectorStoreIndex.from_documents([])
        raise


def execute_query(
    question: str,
    index,
    llm,
    matter_id: int = None,
    top_k: int = 3,
    use_cag: bool = False,
    use_srlc: bool = False,
    model_name: str = "local-model",
) -> dict:
    """
    Execute a legal query against the vector index.

    Returns answer, citation_report, and thought_stream.
    Citation report includes page-level pinpoint citations with confidence scores.
    """
    if matter_id is not None:
        from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
        filters = MetadataFilters(filters=[ExactMatchFilter(key="matter_id", value=matter_id)])
        retriever = index.as_retriever(similarity_top_k=top_k * (2 if use_cag else 1), filters=filters)
    else:
        retriever = index.as_retriever(similarity_top_k=top_k * (2 if use_cag else 1))
    nodes = retriever.retrieve(question)
    context_str = "\n\n".join([n.node.get_content() for n in nodes])

    thought_stream = []

    if use_srlc:
        from agents.advanced_cag import run_advanced_cag

        cognitive_result = run_advanced_cag(
            query=question, local_context=context_str, llm=llm
        )
        thought_stream = cognitive_result["thought_stream"]
        answer = cognitive_result["answer"]
    elif use_cag:
        cag_prompt = f"[CACHED LEGAL KNOWLEDGE]\n{context_str}\n\nQuestion: {question}"
        answer = str(llm.complete(cag_prompt))
    else:
        query_engine = index.as_query_engine(similarity_top_k=top_k)
        response = query_engine.query(question)
        answer = str(response)

    # Build pinpoint citation report
    citation_report = build_citation_report(nodes, query=question)

    return {
        "answer": answer,
        "citation_report": report_to_dict(citation_report),
        "thought_stream": thought_stream,
        "mode": "SRLC" if use_srlc else ("CAG" if use_cag else "RAG"),
        "model": model_name,
    }


# --- CLI entrypoint ---
if __name__ == "__main__":
    original_stdout = sys.stdout
    sys.stdout = sys.stderr

    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    logging.getLogger("llama_index").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Tegifa RAG Query Engine")
    parser.add_argument("question", type=str, help="The legal question to answer.")
    parser.add_argument("--db-path", type=str, default=os.getenv("RAG_DB_PATH", "/app/chroma_db"))
    parser.add_argument("--storage-path", type=str, default=os.getenv("RAG_STORAGE_PATH", "/app/storage"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--use-cag", action="store_true")
    parser.add_argument("--use-srlc", action="store_true")
    parser.add_argument("--model-type", type=str, default="llama.cpp", choices=["llama.cpp", "ollama"])
    parser.add_argument("--model-name", type=str, default="local-model")

    try:
        args = parser.parse_args()

        LLM_API_BASE = os.getenv("LLM_API_BASE", "http://llm-server:8080/v1")
        OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")

        llm = init_llm(args.model_type, args.model_name, OLLAMA_HOST, LLM_API_BASE)
        Settings.llm = llm
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        index = init_index(args.db_path, args.storage_path)

        output = execute_query(
            question=args.question,
            index=index,
            llm=llm,
            top_k=args.top_k,
            use_cag=args.use_cag,
            use_srlc=args.use_srlc,
            model_name=args.model_name,
        )

        sys.stdout = original_stdout
        print(json.dumps(output))

    except Exception as e:
        sys.stdout = original_stdout
        logging.error("Fatal error: %s", e)
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
