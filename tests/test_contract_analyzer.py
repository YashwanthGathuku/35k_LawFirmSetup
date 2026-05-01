"""
Tegifa Legal — Contract Analyzer Tests
"""
import pytest
import os
import sys
from unittest.mock import MagicMock
from dataclasses import dataclass, field
from typing import Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.contract_analyzer import (
    parse_llm_json,
    analyze_clause_with_llm,
    extract_obligations_with_llm,
    run_contract_review,
    report_to_dict,
    EXPECTED_CLAUSES,
    ClauseAnalysis,
    Obligation,
)
from rag_scripts.document_processor import ProcessedChunk, ProcessedDocument


# --- JSON Parsing ---

def test_parse_llm_json_clean():
    result = parse_llm_json('{"risk_level": "high", "risk_score": 0.8}')
    assert result["risk_level"] == "high"


def test_parse_llm_json_markdown_fences():
    result = parse_llm_json('```json\n{"risk_level": "low"}\n```')
    assert result["risk_level"] == "low"


def test_parse_llm_json_invalid():
    result = parse_llm_json("This is not JSON at all", fallback={"default": True})
    assert result == {"default": True}


def test_parse_llm_json_array():
    result = parse_llm_json('[{"party": "A", "action": "pay"}]')
    assert isinstance(result, list)
    assert result[0]["party"] == "A"


# --- Clause Analysis ---

def test_analyze_clause_success():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = (
        '{"risk_level": "high", "risk_score": 0.75, '
        '"summary": "Broad indemnification clause", '
        '"concerns": ["One-sided obligation"], '
        '"recommendation": "Negotiate mutual indemnification"}'
    )

    result = analyze_clause_with_llm(
        llm=mock_llm,
        text="Contractor shall indemnify and hold harmless...",
        clause_type="indemnification",
        file_name="contract.pdf",
        page_number=5,
        section_heading="Indemnification",
    )

    assert result.clause_type == "indemnification"
    assert result.risk_level == "high"
    assert result.risk_score == 0.75
    assert "One-sided" in result.concerns[0]
    assert result.page_number == 5


def test_analyze_clause_llm_failure():
    mock_llm = MagicMock()
    mock_llm.complete.side_effect = Exception("LLM timeout")

    result = analyze_clause_with_llm(
        llm=mock_llm,
        text="Some clause text",
        clause_type="termination",
        file_name="test.pdf",
        page_number=1,
        section_heading=None,
    )

    assert result.risk_level == "medium"  # safe fallback
    assert "unavailable" in result.summary.lower()


def test_analyze_clause_bad_json_from_llm():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "I think this clause is risky because..."

    result = analyze_clause_with_llm(
        llm=mock_llm,
        text="Some clause",
        clause_type="liability",
        file_name="test.pdf",
        page_number=2,
        section_heading=None,
    )

    # Should not crash, should return sensible defaults
    assert result.clause_type == "liability"
    assert result.risk_level == "medium"


# --- Obligation Extraction ---

def test_extract_obligations_success():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = (
        '[{"party": "Client", "action": "Pay invoices within 30 days", '
        '"deadline": "30 days from receipt", "condition": null}]'
    )

    obligations = extract_obligations_with_llm(mock_llm, "Payment terms text", page_number=8)
    assert len(obligations) == 1
    assert obligations[0].party == "Client"
    assert obligations[0].deadline == "30 days from receipt"
    assert obligations[0].page_number == 8


def test_extract_obligations_none_found():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "[]"

    obligations = extract_obligations_with_llm(mock_llm, "Boilerplate text")
    assert obligations == []


def test_extract_obligations_llm_failure():
    mock_llm = MagicMock()
    mock_llm.complete.side_effect = Exception("Timeout")

    obligations = extract_obligations_with_llm(mock_llm, "Some text")
    assert obligations == []


def test_extract_obligations_invalid_json():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "Here are the obligations I found..."

    obligations = extract_obligations_with_llm(mock_llm, "Some text")
    assert obligations == []


# --- Full Contract Review ---

def _make_processed_doc(chunks=None):
    """Create a mock ProcessedDocument for testing."""
    if chunks is None:
        chunks = [
            ProcessedChunk(
                text="This agreement may be terminated by either party upon 30 days notice. "
                     "The contractor shall indemnify and hold harmless the client.",
                file_name="test_contract.pdf",
                file_path="/tmp/test_contract.pdf",
                page_number=1,
                paragraph_index=0,
                total_pages=5,
                chunk_index=0,
                total_chunks=3,
                detected_clauses=["termination", "indemnification"],
            ),
            ProcessedChunk(
                text="All disputes shall be resolved by arbitration in Delaware. "
                     "This agreement constitutes the entire agreement between the parties.",
                file_name="test_contract.pdf",
                file_path="/tmp/test_contract.pdf",
                page_number=3,
                paragraph_index=0,
                total_pages=5,
                chunk_index=1,
                total_chunks=3,
                detected_clauses=["governing_law", "entire_agreement"],
            ),
            ProcessedChunk(
                text="Payment shall be due within Net 30 from invoice date.",
                file_name="test_contract.pdf",
                file_path="/tmp/test_contract.pdf",
                page_number=4,
                paragraph_index=0,
                total_pages=5,
                chunk_index=2,
                total_chunks=3,
                detected_clauses=["payment"],
            ),
        ]

    return ProcessedDocument(
        file_name="test_contract.pdf",
        file_path="/tmp/test_contract.pdf",
        total_pages=5,
        total_chunks=len(chunks),
        chunks=chunks,
        document_type="contract",
    )


def test_run_contract_review_full():
    mock_llm = MagicMock()
    # Clause analysis responses
    mock_llm.complete.side_effect = [
        # termination clause analysis
        '{"risk_level": "low", "risk_score": 0.2, "summary": "Standard mutual termination", "concerns": [], "recommendation": "Acceptable"}',
        # indemnification clause analysis
        '{"risk_level": "high", "risk_score": 0.8, "summary": "One-sided indemnification", "concerns": ["Only contractor indemnifies"], "recommendation": "Negotiate mutual"}',
        # governing_law clause analysis
        '{"risk_level": "low", "risk_score": 0.1, "summary": "Delaware arbitration", "concerns": [], "recommendation": "Standard"}',
        # entire_agreement clause analysis
        '{"risk_level": "low", "risk_score": 0.1, "summary": "Standard merger clause", "concerns": [], "recommendation": "OK"}',
        # payment clause analysis
        '{"risk_level": "low", "risk_score": 0.15, "summary": "Net 30 payment terms", "concerns": [], "recommendation": "Standard"}',
        # obligation extraction x3 chunks
        '[{"party": "Client", "action": "Pay within 30 days", "deadline": "Net 30", "condition": null}]',
        '[{"party": "Contractor", "action": "Indemnify client", "deadline": null, "condition": null}]',
        '[]',
        # executive summary
        "This is a standard services agreement with one significant risk in the indemnification clause.",
    ]

    doc = _make_processed_doc()
    report = run_contract_review(llm=mock_llm, processed_doc=doc)

    assert report.document_name == "test_contract.pdf"
    assert len(report.clause_analyses) == 5
    assert len(report.obligations) >= 1
    assert len(report.missing_clauses) > 0  # Should flag missing clauses
    assert report.overall_risk_score > 0
    assert report.executive_summary != ""


def test_run_contract_review_missing_clauses():
    """Should detect which standard clauses are missing."""
    mock_llm = MagicMock()
    mock_llm.complete.return_value = '{"risk_level": "low", "risk_score": 0.1, "summary": "OK", "concerns": [], "recommendation": "OK"}'

    # Only has termination clause
    chunks = [
        ProcessedChunk(
            text="This agreement may be terminated upon notice.",
            file_name="sparse.pdf", file_path="/tmp/sparse.pdf",
            page_number=1, paragraph_index=0, total_pages=1,
            chunk_index=0, total_chunks=1,
            detected_clauses=["termination"],
        ),
    ]
    doc = _make_processed_doc(chunks)
    report = run_contract_review(llm=mock_llm, processed_doc=doc)

    # Should flag many missing clauses
    assert len(report.missing_clauses) >= 5
    assert "liability" in report.missing_clauses
    assert "confidentiality" in report.missing_clauses


# --- Serialization ---

def test_report_to_dict_structure():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = '{"risk_level": "low", "risk_score": 0.1, "summary": "OK", "concerns": [], "recommendation": "OK"}'

    doc = _make_processed_doc()
    report = run_contract_review(llm=mock_llm, processed_doc=doc)
    data = report_to_dict(report)

    assert "document_name" in data
    assert "overall_risk_score" in data
    assert "executive_summary" in data
    assert "clause_analyses" in data
    assert "obligations" in data
    assert "missing_clauses" in data
    assert "key_findings" in data
    assert isinstance(data["clause_analyses"], list)
