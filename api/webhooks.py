"""
Tegifa Legal — FastAPI Webhooks for n8n Integration
Exposes endpoints for external workflow automation tools (like n8n) to trigger
document ingestion, run the deal graph, or query the legal engine.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging

from rag_scripts.rag_setup import run_ingestion
from rag_scripts.query_rag import execute_query, init_llm, init_index
from agents.deal_graph import build_deal_graph, get_conflicts
from rag_scripts.document_processor import process_file

logger = logging.getLogger("tegifa.webhooks")
app = FastAPI(title="Tegifa Legal Automation API")

# Setup Paths
DB_PATH = os.getenv("RAG_DB_PATH", "chroma_db")
STORAGE_PATH = os.getenv("RAG_STORAGE_PATH", "storage")
DOCS_PATH = os.getenv("RAG_DOCS_PATH", "docs")

# Models
class IngestRequest(BaseModel):
    matter_id: int
    files: List[str]

class QueryRequest(BaseModel):
    matter_id: int
    question: str
    model_name: str = "llama3"

@app.post("/webhook/ingest")
async def trigger_ingestion(req: IngestRequest, background_tasks: BackgroundTasks):
    """Triggered by n8n when new documents arrive via email or external system."""
    def _ingest_task(matter_id: int):
        try:
            logger.info(f"Starting background ingestion for matter {matter_id}...")
            run_ingestion(
                matter_id=matter_id,
                db_path=DB_PATH,
                docs_path=DOCS_PATH,
                storage_path=STORAGE_PATH,
                use_nougat=False
            )
            logger.info("Background ingestion completed.")
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")

    background_tasks.add_task(_ingest_task, req.matter_id)
    return {"status": "accepted", "message": f"Ingestion started for {len(req.files)} files."}


@app.post("/webhook/query")
async def trigger_query(req: QueryRequest):
    """Triggered by n8n to ask the legal engine a question."""
    try:
        # Initialize LLM and Index (in a real production app, cache these)
        llm = init_llm(backend="ollama", model_name=req.model_name)
        index = init_index(db_path=DB_PATH, storage_path=STORAGE_PATH)
        
        result = execute_query(
            question=req.question,
            index=index,
            llm=llm,
            matter_id=req.matter_id,
            top_k=3,
            use_srlc=True,
            model_name=req.model_name
        )
        return {"answer": result["answer"]}
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
