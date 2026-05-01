"""
Tegifa Legal — Investigator Agent Tools
External intelligence gathering capabilities for the multi-agent system.
"""
import logging
from typing import Dict, Any, Callable

logger = logging.getLogger("tegifa.tools")


class InvestigatorTools:
    """Tools for the Investigator Agent to autonomously gather external intelligence."""

    @staticmethod
    def search_duckduckgo(query: str, max_results: int = 5) -> str:
        """
        Perform web searches to find legal precedents, news, or statutes.
        Tries 'ddgs' first, falls back to 'duckduckgo_search'.
        """
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS

            logger.info("Web search for: %s", query)
            results = list(DDGS().text(query, max_results=max_results))

            if not results:
                return "No search results found."

            formatted = []
            for r in results:
                formatted.append(
                    f"Title: {r.get('title', 'N/A')}\n"
                    f"Snippet: {r.get('body', 'N/A')}\n"
                    f"URL: {r.get('href', 'N/A')}\n---"
                )
            return "\n".join(formatted)

        except Exception as e:
            logger.error("Web search failed: %s", e)
            return f"Error executing web search: {e}"

    @staticmethod
    def query_local_knowledge_graph(entity_name: str) -> str:
        """
        Query the relational Knowledge Graph (Neo4j) for explicit
        judicial relationships beyond simple vector similarity.
        """
        logger.info("Graph query for: %s", entity_name)
        try:
            from agents.graph_builder import kg_manager

            return kg_manager.query_relationships(entity_name)
        except Exception as e:
            logger.error("Graph query error: %s", e)
            return "Knowledge Graph is currently unavailable."

    @staticmethod
    def get_tool_map() -> Dict[str, Callable]:
        """Returns a map of tools available to the LangGraph agents."""
        return {
            "web_search": InvestigatorTools.search_duckduckgo,
            "graph_query": InvestigatorTools.query_local_knowledge_graph,
        }
