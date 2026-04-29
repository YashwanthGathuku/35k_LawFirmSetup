"""
Tegifa Legal — RAG Pipeline Tests
Tests for query execution modes (RAG, CAG, SRLC).
"""
import pytest
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_scripts.query_rag import execute_query


def _make_mock_index():
    """Create a mock index with a mock retriever returning one fake node."""
    mock_index = MagicMock()
    mock_retriever = MagicMock()

    mock_node = MagicMock()
    mock_node.node.get_content.return_value = "Mocked legal context"
    mock_node.node.metadata = {"file_name": "test_brief.pdf"}
    mock_node.score = 0.95

    mock_retriever.retrieve.return_value = [mock_node]
    mock_index.as_retriever.return_value = mock_retriever

    return mock_index


def test_execute_query_standard_rag():
    """Standard RAG mode should use the query engine."""
    mock_index = _make_mock_index()
    mock_query_engine = MagicMock()
    mock_query_engine.query.return_value = "Standard RAG Answer"
    mock_index.as_query_engine.return_value = mock_query_engine

    mock_llm = MagicMock()

    result = execute_query(
        question="What is the statute of limitations?",
        index=mock_index,
        llm=mock_llm,
        use_cag=False,
        use_srlc=False,
        model_name="test-model",
    )

    assert result["answer"] == "Standard RAG Answer"
    assert result["mode"] == "RAG"
    assert len(result["sources"]) == 1
    assert result["sources"][0]["metadata"]["file_name"] == "test_brief.pdf"
    assert result["sources"][0]["score"] == 0.95


def test_execute_query_cag_mode():
    """CAG mode should call llm.complete with cached context."""
    mock_index = _make_mock_index()
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "CAG Answer"

    result = execute_query(
        question="What is the ruling?",
        index=mock_index,
        llm=mock_llm,
        use_cag=True,
        use_srlc=False,
        model_name="test-model",
    )

    assert result["answer"] == "CAG Answer"
    assert result["mode"] == "CAG"
    mock_llm.complete.assert_called_once()
    # Verify the prompt included cached context
    call_args = mock_llm.complete.call_args[0][0]
    assert "[CACHED LEGAL KNOWLEDGE]" in call_args


def test_execute_query_srlc_mode(mocker):
    """SRLC mode should trigger the multi-agent cognitive cycle."""
    mock_index = _make_mock_index()
    mock_llm = MagicMock()

    mocker.patch(
        "rag_scripts.query_rag.run_cognitive_cycle",
        return_value={
            "answer": "SRLC Verified Answer",
            "thought_stream": [
                {"step": "Reasoning", "content": "Initial hypothesis"},
                {"step": "Critique", "content": "ACCEPT"},
            ],
        },
    )

    # Need to re-import after patching
    from rag_scripts.query_rag import execute_query as eq

    result = eq(
        question="Analyze this case",
        index=mock_index,
        llm=mock_llm,
        use_cag=False,
        use_srlc=True,
        model_name="test-model",
    )

    assert result["answer"] == "SRLC Verified Answer"
    assert result["mode"] == "SRLC"
    assert len(result["thought_stream"]) == 2


def test_execute_query_with_no_sources():
    """Should handle case where retriever returns no nodes."""
    mock_index = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []
    mock_index.as_retriever.return_value = mock_retriever

    mock_query_engine = MagicMock()
    mock_query_engine.query.return_value = "Answer without sources"
    mock_index.as_query_engine.return_value = mock_query_engine

    mock_llm = MagicMock()

    result = execute_query(
        question="General question",
        index=mock_index,
        llm=mock_llm,
        use_cag=False,
        use_srlc=False,
    )

    assert result["answer"] == "Answer without sources"
    assert result["sources"] == []
