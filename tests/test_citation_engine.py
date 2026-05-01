"""
Tegifa Legal — Citation Engine Tests
"""
import pytest
import os
import sys
from unittest.mock import MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_scripts.citation_engine import (
    score_to_confidence,
    build_citation,
    build_citation_report,
    citation_to_dict,
    report_to_dict,
    Citation,
    CitationReport,
)


# --- Score to Confidence ---

def test_score_high():
    conf, label = score_to_confidence(0.85)
    assert label == "high"
    assert conf == 0.85


def test_score_medium():
    conf, label = score_to_confidence(0.60)
    assert label == "medium"


def test_score_low():
    conf, label = score_to_confidence(0.30)
    assert label == "low"


def test_score_clamp_above_1():
    conf, label = score_to_confidence(1.5)
    assert conf == 1.0
    assert label == "high"


def test_score_clamp_below_0():
    conf, label = score_to_confidence(-0.2)
    assert conf == 0.0
    assert label == "low"


# --- Citation ---

def test_citation_pinpoint_full():
    c = Citation(
        file_name="contract.pdf",
        page_number=5,
        paragraph_index=2,
        section_heading="Termination Rights",
        snippet="The agreement may be terminated...",
        confidence=0.9,
        confidence_label="high",
        total_pages=20,
    )
    assert "contract.pdf" in c.pinpoint
    assert "p. 5 of 20" in c.pinpoint
    assert "Termination Rights" in c.pinpoint
    assert "¶ 3" in c.pinpoint  # paragraph_index 2 -> ¶ 3 (1-indexed)


def test_citation_pinpoint_minimal():
    c = Citation(
        file_name="brief.pdf",
        page_number=None,
        paragraph_index=None,
        section_heading=None,
        snippet="Text",
        confidence=0.5,
        confidence_label="medium",
    )
    assert c.pinpoint == "brief.pdf"


def test_citation_short_ref():
    c = Citation(
        file_name="nda.pdf",
        page_number=3,
        paragraph_index=0,
        section_heading=None,
        snippet="Text",
        confidence=0.7,
        confidence_label="medium",
    )
    assert c.short_ref == "nda.pdf, p. 3"


# --- Build Citation from Node ---

def _make_mock_node(metadata=None, text="Sample text", score=0.8):
    node_result = MagicMock()
    node_result.node.get_content.return_value = text
    node_result.node.metadata = metadata or {}
    node_result.score = score
    return node_result


def test_build_citation_full_metadata():
    node = _make_mock_node(
        metadata={
            "file_name": "contract.pdf",
            "page_number": 7,
            "paragraph_index": 3,
            "section_heading": "Liability",
            "detected_clauses": "liability,indemnification",
            "contains_table": True,
            "chunk_index": 5,
            "total_chunks": 20,
            "total_pages": 15,
        },
        text="The liability shall not exceed...",
        score=0.82,
    )
    c = build_citation(node)
    assert c.file_name == "contract.pdf"
    assert c.page_number == 7
    assert c.paragraph_index == 3
    assert c.section_heading == "Liability"
    assert "liability" in c.detected_clauses
    assert "indemnification" in c.detected_clauses
    assert c.contains_table is True
    assert c.confidence_label == "high"


def test_build_citation_missing_metadata():
    node = _make_mock_node(metadata={}, score=0.3)
    c = build_citation(node)
    assert c.file_name == "Unknown"
    assert c.page_number is None
    assert c.confidence_label == "low"


def test_build_citation_no_score():
    node = _make_mock_node(metadata={"file_name": "test.pdf"})
    node.score = None
    c = build_citation(node)
    assert c.confidence == 0.5  # default


# --- Citation Report ---

def test_build_citation_report_deduplication():
    """Same file+page+paragraph should be deduplicated."""
    nodes = [
        _make_mock_node(
            {"file_name": "a.pdf", "page_number": 1, "paragraph_index": 0},
            score=0.9,
        ),
        _make_mock_node(
            {"file_name": "a.pdf", "page_number": 1, "paragraph_index": 0},
            score=0.85,
        ),
    ]
    report = build_citation_report(nodes, query="test")
    assert report.total_sources == 1


def test_build_citation_report_sorted_by_confidence():
    nodes = [
        _make_mock_node({"file_name": "low.pdf", "page_number": 1, "paragraph_index": 0}, score=0.3),
        _make_mock_node({"file_name": "high.pdf", "page_number": 1, "paragraph_index": 0}, score=0.9),
    ]
    report = build_citation_report(nodes, query="test")
    assert report.citations[0].file_name == "high.pdf"
    assert report.citations[1].file_name == "low.pdf"


def test_build_citation_report_summary():
    nodes = [
        _make_mock_node({"file_name": "a.pdf", "page_number": 1, "paragraph_index": 0}, score=0.9),
        _make_mock_node({"file_name": "b.pdf", "page_number": 2, "paragraph_index": 1}, score=0.6),
    ]
    report = build_citation_report(nodes, query="test")
    assert report.unique_documents == 2
    assert "2 citation(s)" in report.summary
    assert "1 high confidence" in report.summary


def test_empty_report():
    report = build_citation_report([], query="test")
    assert report.total_sources == 0
    assert "No source citations" in report.summary


# --- Serialization ---

def test_citation_to_dict():
    c = Citation(
        file_name="test.pdf", page_number=1, paragraph_index=0,
        section_heading="Intro", snippet="Hello", confidence=0.85,
        confidence_label="high", detected_clauses=["termination"],
    )
    d = citation_to_dict(c)
    assert d["file_name"] == "test.pdf"
    assert d["confidence"] == 0.85
    assert d["pinpoint"] == c.pinpoint
    assert d["detected_clauses"] == ["termination"]


def test_report_to_dict():
    nodes = [
        _make_mock_node({"file_name": "a.pdf", "page_number": 1, "paragraph_index": 0}, score=0.9),
    ]
    report = build_citation_report(nodes, query="test")
    d = report_to_dict(report)
    assert d["total_sources"] == 1
    assert len(d["citations"]) == 1
    assert "summary" in d
