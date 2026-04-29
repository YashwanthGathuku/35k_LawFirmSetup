"""
Tegifa Legal — Agent Tests
Tests for tools, graph builder, and orchestrator components.
"""
import pytest
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.tools import InvestigatorTools
from agents.graph_builder import KnowledgeGraphManager
from agents.orchestrator import run_cognitive_cycle, epistemology_node


# ---------------------------------------------------------------------------
# Test Tools
# ---------------------------------------------------------------------------

def test_search_duckduckgo_success(mocker):
    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.text.return_value = [
        {"title": "Test Case", "body": "Snippet", "href": "http://test"}
    ]
    mocker.patch("ddgs.DDGS", return_value=mock_ddgs_instance)

    result = InvestigatorTools.search_duckduckgo("test query")
    assert "Title: Test Case" in result
    assert "Snippet: Snippet" in result


def test_search_duckduckgo_no_results(mocker):
    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.text.return_value = []
    mocker.patch("ddgs.DDGS", return_value=mock_ddgs_instance)

    result = InvestigatorTools.search_duckduckgo("empty query")
    assert result == "No search results found."


def test_search_duckduckgo_exception(mocker):
    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.text.side_effect = Exception("API limit reached")
    mocker.patch("ddgs.DDGS", return_value=mock_ddgs_instance)

    result = InvestigatorTools.search_duckduckgo("fail query")
    assert "Error executing web search: API limit reached" in result


def test_query_local_knowledge_graph(mocker):
    mocker.patch(
        "agents.graph_builder.KnowledgeGraphManager.query_relationships",
        return_value="(A)-[O]->(B)",
    )
    result = InvestigatorTools.query_local_knowledge_graph("Entity")
    assert result == "(A)-[O]->(B)"


def test_query_local_knowledge_graph_exception(mocker):
    mocker.patch(
        "agents.graph_builder.KnowledgeGraphManager.query_relationships",
        side_effect=Exception("DB Down"),
    )
    result = InvestigatorTools.query_local_knowledge_graph("Entity")
    assert "Knowledge Graph is currently unavailable." in result


def test_get_tool_map():
    tools = InvestigatorTools.get_tool_map()
    assert "web_search" in tools
    assert "graph_query" in tools
    assert callable(tools["web_search"])
    assert callable(tools["graph_query"])


# ---------------------------------------------------------------------------
# Test Graph Builder (Lazy Initialization)
# ---------------------------------------------------------------------------

def test_kg_manager_lazy_init():
    """KnowledgeGraphManager should NOT connect on instantiation."""
    kg = KnowledgeGraphManager()
    assert kg._driver is None
    assert kg._initialized is False


def test_kg_manager_merge_no_driver():
    """Without Neo4j, merge should return unavailable message."""
    kg = KnowledgeGraphManager()
    kg._initialized = True  # Pretend we tried and failed
    kg._driver = None
    res = kg.merge_epistemic_relationship("A", "O", "B", 0.9, "Source")
    assert res == "Graph DB unavailable."


def test_kg_manager_query_no_driver():
    """Without Neo4j, query should return unavailable message."""
    kg = KnowledgeGraphManager()
    kg._initialized = True
    kg._driver = None
    res = kg.query_relationships("A")
    assert res == "Graph DB unavailable."


def test_kg_manager_query_empty(mocker):
    """With Neo4j connected but no results, should return 'no relationships'."""
    kg = KnowledgeGraphManager()
    mock_driver = MagicMock()
    mock_session = mock_driver.session.return_value.__enter__.return_value
    mock_session.run.return_value = []
    kg._driver = mock_driver
    kg._initialized = True

    res = kg.query_relationships("A")
    assert "No known relationships found" in res


def test_kg_manager_query_success(mocker):
    """With Neo4j connected and results, should format them correctly."""
    kg = KnowledgeGraphManager()
    mock_driver = MagicMock()
    mock_session = mock_driver.session.return_value.__enter__.return_value
    mock_session.run.return_value = [
        {"subject": "A", "predicate": "O", "object": "B", "confidence": 0.9}
    ]
    kg._driver = mock_driver
    kg._initialized = True

    res = kg.query_relationships("A")
    assert "(A) -[O (Conf: 0.9)]-> (B)" in res


def test_kg_manager_merge_success(mocker):
    """With Neo4j connected, merge should succeed."""
    kg = KnowledgeGraphManager()
    mock_driver = MagicMock()
    mock_session = mock_driver.session.return_value.__enter__.return_value
    mock_session.run.return_value = []
    kg._driver = mock_driver
    kg._initialized = True

    res = kg.merge_epistemic_relationship("A", "O", "B", 0.9, "Source")
    assert "Merged" in res
    mock_session.run.assert_called_once()


# ---------------------------------------------------------------------------
# Test Orchestrator
# ---------------------------------------------------------------------------

def test_epistemology_node_accept():
    state = {"query": "test", "current_hypothesis": "hyp", "iterations": 0}
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "ACCEPT. It is good."

    res = epistemology_node(state, mock_llm)
    assert res["critique_passed"] is True
    assert "ACCEPT" in res["messages"][0]["content"]


def test_epistemology_node_reject():
    state = {"query": "test", "current_hypothesis": "hyp", "iterations": 0}
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "REJECT. Needs info."

    res = epistemology_node(state, mock_llm)
    assert res["critique_passed"] is False


def test_epistemology_node_max_iterations():
    state = {"query": "test", "current_hypothesis": "hyp", "iterations": 2}
    res = epistemology_node(state, llm=None)
    assert res["critique_passed"] is True
    assert "Maximum iterations" in res["messages"][0]["content"]


def test_run_cognitive_cycle_with_llm():
    mock_llm = MagicMock()
    mock_llm.complete.side_effect = ["Hypothesis generated", "ACCEPT. Fine."]

    res = run_cognitive_cycle("Test query", local_context="", llm=mock_llm)
    assert res["answer"] == "Hypothesis generated"
    assert len(res["thought_stream"]) == 2


def test_run_cognitive_cycle_simulated_no_llm():
    """Without LLM, uses fallback simulation: reject → investigate → accept."""
    res = run_cognitive_cycle("Test query", local_context="", llm=None)
    assert "adjusted" in res["answer"].lower()
    assert len(res["thought_stream"]) == 5
