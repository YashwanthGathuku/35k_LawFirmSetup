import pytest
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.tools import InvestigatorTools
from agents.graph_builder import KnowledgeGraphManager
from agents.orchestrator import run_cognitive_cycle, epistemology_node

# -----------------------------------------------------------------------------
# Test Tools
# -----------------------------------------------------------------------------

def test_search_duckduckgo_success(mocker):
    # Mock the DDGS().text generator where it is actually imported
    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.text.return_value = [{"title": "Test Case", "body": "Snippet", "href": "http://test"}]

    # We patch it where it is loaded in the module hierarchy (or rather, the module itself)
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
    mock_kg = MagicMock()
    mock_kg.query_relationships.return_value = "(A)-[O]->(B)"

    # Because it is imported inside the method
    mocker.patch("agents.graph_builder.KnowledgeGraphManager.query_relationships", return_value="(A)-[O]->(B)")

    result = InvestigatorTools.query_local_knowledge_graph("Entity")
    assert result == "(A)-[O]->(B)"

def test_query_local_knowledge_graph_exception(mocker):
    mock_kg = MagicMock()
    mock_kg.query_relationships.side_effect = Exception("DB Down")

    mocker.patch("agents.graph_builder.KnowledgeGraphManager.query_relationships", side_effect=Exception("DB Down"))

    result = InvestigatorTools.query_local_knowledge_graph("Entity")
    assert "Knowledge Graph is currently unavailable." in result

def test_get_tool_map():
    tools = InvestigatorTools.get_tool_map()
    assert "web_search" in tools
    assert "graph_query" in tools

# -----------------------------------------------------------------------------
# Test Graph Builder
# -----------------------------------------------------------------------------

def test_kg_manager_init_success(mocker):
    mock_driver = mocker.patch("agents.graph_builder.GraphDatabase.driver")
    kg = KnowledgeGraphManager()
    mock_driver.assert_called_once()
    assert kg.driver is not None

def test_kg_manager_init_fail(mocker):
    mock_driver = mocker.patch("agents.graph_builder.GraphDatabase.driver", side_effect=Exception("Failed"))
    kg = KnowledgeGraphManager()
    assert kg.driver is None

def test_kg_manager_merge_success(mocker):
    # Setup mock driver and session
    mock_driver = mocker.patch("agents.graph_builder.GraphDatabase.driver")
    mock_instance = mock_driver.return_value
    mock_session = mock_instance.session.return_value.__enter__.return_value
    mock_session.run.return_value = []

    kg = KnowledgeGraphManager()
    res = kg.merge_epistemic_relationship("A", "O", "B", 0.9, "Source")

    assert "Successfully merged" in res
    mock_session.run.assert_called_once()

def test_kg_manager_merge_no_driver(mocker):
    mocker.patch("agents.graph_builder.GraphDatabase.driver", side_effect=Exception("Failed"))
    kg = KnowledgeGraphManager()
    res = kg.merge_epistemic_relationship("A", "O", "B", 0.9, "Source")
    assert res == "Graph DB unavailable."

def test_kg_manager_query_success(mocker):
    mock_driver = mocker.patch("agents.graph_builder.GraphDatabase.driver")
    mock_instance = mock_driver.return_value
    mock_session = mock_instance.session.return_value.__enter__.return_value
    # Simulate DB records
    mock_session.run.return_value = [{"subject": "A", "predicate": "O", "object": "B", "confidence": 0.9}]

    kg = KnowledgeGraphManager()
    res = kg.query_relationships("A")
    assert "(A) -[O (Conf: 0.9)]-> (B)" in res

def test_kg_manager_query_empty(mocker):
    mock_driver = mocker.patch("agents.graph_builder.GraphDatabase.driver")
    mock_instance = mock_driver.return_value
    mock_session = mock_instance.session.return_value.__enter__.return_value
    mock_session.run.return_value = []

    kg = KnowledgeGraphManager()
    res = kg.query_relationships("A")
    assert "No known relationships found" in res

def test_kg_manager_query_no_driver(mocker):
    mocker.patch("agents.graph_builder.GraphDatabase.driver", side_effect=Exception("Failed"))
    kg = KnowledgeGraphManager()
    res = kg.query_relationships("A")
    assert res == "Graph DB unavailable."

# -----------------------------------------------------------------------------
# Test Orchestrator
# -----------------------------------------------------------------------------

def test_epistemology_node_accept(mocker):
    state = {
        "query": "test",
        "current_hypothesis": "hyp",
        "iterations": 0
    }
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "ACCEPT. It is good."

    res = epistemology_node(state, mock_llm)
    assert res["critique_passed"] is True
    assert "ACCEPT" in res["messages"][0]["content"]

def test_epistemology_node_reject(mocker):
    state = {
        "query": "test",
        "current_hypothesis": "hyp",
        "iterations": 0
    }
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "REJECT. Needs info."

    res = epistemology_node(state, mock_llm)
    assert res["critique_passed"] is False

def test_run_cognitive_cycle(mocker):
    # Mock the LLM to ACCEPT immediately on the first reasoning
    mock_llm = MagicMock()
    # It will be called first by Reasoner, then by Epistemology
    mock_llm.complete.side_effect = ["Hypothesis generated", "ACCEPT. Fine."]

    res = run_cognitive_cycle("Test query", local_context="", llm=mock_llm)
    assert res["answer"] == "Hypothesis generated"
    assert len(res["thought_stream"]) == 2

def test_run_cognitive_cycle_simulated_no_llm():
    # If LLM is None, it falls back to the simulated scaffold output
    res = run_cognitive_cycle("Test query", local_context="", llm=None)
    # The default simulated skeptic rejects the first one, triggering investigation
    # and accepting the second one due to iteration limit.
    assert "adjusted" in res["answer"].lower()
    assert len(res["thought_stream"]) == 5

# -----------------------------------------------------------------------------
# Test Autonomous Worker
# -----------------------------------------------------------------------------

def test_autonomous_worker_loop(mocker):
    # We only want to run the loop ONCE instead of infinite
    # So we patch the loop to break after one iteration, or we just test the inner function.
    from agents.autonomous_worker import run_predictive_jurisprudence_loop

    # Mock the cognitive cycle
    mock_cycle = mocker.patch("agents.autonomous_worker.run_cognitive_cycle")
    mock_cycle.return_value = {"answer": "A new law passed."}

    # Mock the graph push
    mock_kg = mocker.patch("agents.autonomous_worker.kg_manager.merge_epistemic_relationship")

    # Patch time.sleep to raise an exception to break the infinite while loop
    mocker.patch("agents.autonomous_worker.time.sleep", side_effect=InterruptedError)

    with pytest.raises(InterruptedError):
        run_predictive_jurisprudence_loop()

    mock_cycle.assert_called()
    mock_kg.assert_called()
