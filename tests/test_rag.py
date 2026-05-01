"""
Tegifa Legal — RAG Pipeline Tests
Tests for query execution modes with citation engine integration.
"""
import pytest
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_scripts.query_rag import execute_query


def _make_mock_index(metadata=None):
    """Create a mock index with a mock retriever returning one fake node."""
    mock_index = MagicMock()
    mock_retriever = MagicMock()

    mock_node = MagicMock()
    mock_node.node.get_content.return_value = "Mocked legal context"
    mock_node.node.metadata = metadata or {
        "file_name": "test_brief.pdf",
        "page_number": 3,
        "paragraph_index": 1,
        "section_heading": "Analysis",
        "detected_clauses": "liability",
        "contains_table": False,
        "total_pages": 10,
        "chunk_index": 5,
        "total_chunks": 20,
    }
    mock_node.score = 0.85

    mock_retriever.retrieve.return_value = [mock_node]
    mock_index.as_retriever.return_value = mock_retriever
    return mock_index


def test_execute_query_standard_rag():
    mock_index = _make_mock_index()
    mock_query_engine = MagicMock()
    mock_query_engine.query.return_value = "Standard RAG Answer"
    mock_index.as_query_engine.return_value = mock_query_engine
    mock_llm = MagicMock()

    result = execute_query(
        question="What is the statute of limitations?",
        index=mock_index, llm=mock_llm,
        use_cag=False, use_srlc=False, model_name="test-model",
    )

    assert result["answer"] == "Standard RAG Answer"
    assert result["mode"] == "RAG"
    # Citation report instead of flat sources
    report = result["citation_report"]
    assert report["total_sources"] == 1
    assert report["citations"][0]["file_name"] == "test_brief.pdf"
    assert report["citations"][0]["page_number"] == 3
    assert report["citations"][0]["confidence_label"] == "high"
    assert "pinpoint" in report["citations"][0]


def test_execute_query_cag_mode():
    mock_index = _make_mock_index()
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "CAG Answer"

    result = execute_query(
        question="What is the ruling?",
        index=mock_index, llm=mock_llm,
        use_cag=True, use_srlc=False,
    )

    assert result["answer"] == "CAG Answer"
    assert result["mode"] == "CAG"
    assert result["citation_report"]["total_sources"] == 1


def test_execute_query_srlc_mode(mocker):
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

    from rag_scripts.query_rag import execute_query as eq

    result = eq(
        question="Analyze this case",
        index=mock_index, llm=mock_llm,
        use_cag=False, use_srlc=True,
    )

    assert result["answer"] == "SRLC Verified Answer"
    assert result["mode"] == "SRLC"
    assert len(result["thought_stream"]) == 2
    assert result["citation_report"]["total_sources"] == 1


def test_execute_query_with_no_sources():
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
        index=mock_index, llm=mock_llm,
    )

    assert result["answer"] == "Answer without sources"
    assert result["citation_report"]["total_sources"] == 0
    assert "No source citations" in result["citation_report"]["summary"]


def test_citation_report_has_pinpoint_data():
    """Verify the citation report includes page/paragraph pinpoints."""
    mock_index = _make_mock_index({
        "file_name": "contract.pdf",
        "page_number": 7,
        "paragraph_index": 2,
        "section_heading": "Liability Cap",
        "detected_clauses": "liability,indemnification",
        "contains_table": False,
        "total_pages": 15,
        "chunk_index": 10,
        "total_chunks": 30,
    })
    mock_query_engine = MagicMock()
    mock_query_engine.query.return_value = "Liability answer"
    mock_index.as_query_engine.return_value = mock_query_engine
    mock_llm = MagicMock()

    result = execute_query("What is the liability?", mock_index, mock_llm)
    cit = result["citation_report"]["citations"][0]

    assert cit["page_number"] == 7
    assert cit["paragraph_index"] == 2
    assert cit["section_heading"] == "Liability Cap"
    assert "liability" in cit["detected_clauses"]
    assert "indemnification" in cit["detected_clauses"]
    assert "p. 7 of 15" in cit["pinpoint"]
    assert "¶ 3" in cit["pinpoint"]
