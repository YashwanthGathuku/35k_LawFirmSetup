import os
import requests
import urllib.parse
from typing import Dict, Any

class InvestigatorTools:
    """
    Tools for the Investigator Agent to autonomously gather external intelligence.
    """

    @staticmethod
    def search_duckduckgo(query: str, max_results: int = 5) -> str:
        """
        A tool to perform open web searches to find legal precedents, news, or statutes.
        Uses the `ddgs` library to scrape real-time data from the web.
        """
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS

            print(f"[Investigator Tool Executed] Live Web Search for: {query}")

            # Use list() to consume the generator
            results = list(DDGS().text(query, max_results=max_results))

            if not results:
                return "No search results found."

            formatted_results = []
            for r in results:
                formatted_results.append(f"Title: {r.get('title')}\nSnippet: {r.get('body')}\nURL: {r.get('href')}\n---")

            return "\n".join(formatted_results)
        except Exception as e:
            print(f"[Investigator Tool Error] Web Search failed: {e}")
            return f"Error executing web search: {e}"

    @staticmethod
    def query_local_knowledge_graph(entity_name: str) -> str:
        """
        A tool to query the relational Knowledge Graph (e.g., Neo4j).
        This goes beyond vector similarity to find explicit judicial relationships.
        """
        print(f"[Investigator Tool Executed] Graph Query for: {entity_name}")
        try:
            from agents.graph_builder import kg_manager
            return kg_manager.query_relationships(entity_name)
        except Exception as e:
            print(f"[Graph Query Error]: {e}")
            return "Knowledge Graph is currently unavailable."

    @staticmethod
    def get_tool_map() -> Dict[str, Any]:
        """Returns a map of tools available to the LangGraph agents."""
        return {
            "web_search": InvestigatorTools.search_duckduckgo,
            "graph_query": InvestigatorTools.query_local_knowledge_graph
        }
