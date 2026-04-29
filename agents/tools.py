"""Tegifa Legal — Investigator Agent Tools."""
import logging
import os
from typing import Dict, Callable

logger = logging.getLogger("tegifa.tools")

STRICT_PRIVACY_MSG = "External search disabled in strict privacy mode. Using local-only reasoning."


def is_privacy_mode_strict() -> bool:
    return os.getenv("PRIVACY_MODE_STRICT", "false").lower() in {"1", "true", "yes", "on"}


class InvestigatorTools:
    @staticmethod
    def search_duckduckgo(query: str, max_results: int = 5) -> str:
        if is_privacy_mode_strict():
            return STRICT_PRIVACY_MSG
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
            logger.info("Web search for: %s", query)
            results = list(DDGS().text(query, max_results=max_results))
            if not results:
                return "No search results found."
            return "\n".join([
                f"Title: {r.get('title', 'N/A')}\nSnippet: {r.get('body', 'N/A')}\nURL: {r.get('href', 'N/A')}\n---"
                for r in results
            ])
        except Exception as e:
            logger.error("Web search failed: %s", e)
            return f"Error executing web search: {e}"

    @staticmethod
    def query_local_knowledge_graph(entity_name: str) -> str:
        logger.info("Graph query for: %s", entity_name)
        try:
            from agents.graph_builder import kg_manager
            return kg_manager.query_relationships(entity_name)
        except Exception:
            return "Knowledge Graph is currently unavailable."

    @staticmethod
    def get_tool_map() -> Dict[str, Callable]:
        return {
            "web_search": InvestigatorTools.search_duckduckgo,
            "graph_query": InvestigatorTools.query_local_knowledge_graph,
        }
