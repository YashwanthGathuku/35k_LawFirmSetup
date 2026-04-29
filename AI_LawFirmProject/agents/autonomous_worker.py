import time
import logging
import os
import sys

# Adjust sys.path so we can import orchestrator and graph_builder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.orchestrator import run_cognitive_cycle
from agents.graph_builder import kg_manager
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AutonomousWorker")

# Initialize LLM for the background worker
LLM_API_BASE = os.getenv("LLM_API_BASE", "http://llm-server:8080/v1")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
MODEL_TYPE = os.getenv("MODEL_TYPE", "llama.cpp")
MODEL_NAME = os.getenv("MODEL_NAME", "local-model")

try:
    if MODEL_TYPE == "ollama":
        llm = Ollama(model=MODEL_NAME, base_url=OLLAMA_HOST, request_timeout=180.0)
    else:
        llm = OpenAI(model="local-model", api_base=LLM_API_BASE, api_key="dummy", request_timeout=120.0)
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")
    sys.exit(1)

def run_predictive_jurisprudence_loop():
    """
    The daemon loop that continuously runs in the background.
    It periodically executes a cognitive cycle on predefined broad legal monitoring queries
    and pushes the resulting relationships to the Neo4j Knowledge Graph.
    """
    logger.info("Starting Continuous Autonomous Mode...")

    monitoring_queries = [
        "What are the most recent Supreme Court rulings affecting environmental law?",
        "Are there any new federal regulations regarding Artificial Intelligence liability?"
    ]

    while True:
        for query in monitoring_queries:
            logger.info(f"[Autonomous Trigger] Initiating cycle for query: '{query}'")

            try:
                # 1. Run the Multi-Agent Cognitive Cycle
                # It will automatically trigger the Investigator Agent (web search)
                # because we provide an empty local_context
                result = run_cognitive_cycle(query=query, local_context="", llm=llm)

                # 2. Extract Entities (Simplified for this scaffolding)
                # In a full implementation, we'd use the LLM to extract JSON {"subject": "", "predicate": "", "object": ""}
                hypothesis = result["answer"]

                # We simulate extracting a relationship based on the hypothesis
                logger.info(f"[Autonomous Success] Generated hypothesis: {hypothesis[:100]}...")

                # 3. Push to Neo4j Graph
                kg_manager.merge_epistemic_relationship(
                    subject="Autonomous Engine",
                    predicate="MONITORED_HYPOTHESIS",
                    object_node="Recent Legal Development",
                    confidence=0.85,
                    source=query
                )

            except Exception as e:
                logger.error(f"Autonomous cycle failed for query '{query}': {e}")

            # Sleep between queries to avoid rate limits
            time.sleep(30)

        # Sleep before restarting the monitoring loop
        logger.info("Cycle complete. Sleeping for 1 hour before next monitoring sweep...")
        time.sleep(3600) # Sleep 1 hour

if __name__ == "__main__":
    run_predictive_jurisprudence_loop()
