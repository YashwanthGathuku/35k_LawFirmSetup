"""
Tegifa Legal — Document Processor Tests
"""
import pytest
import os
import sys
import tempfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_scripts.document_processor import (
    detect_clauses,
    detect_table,
    detect_section_heading,
    classify_document_type,
    chunk_with_metadata,
    process_file,
    chunk_to_metadata,
)


# --- Clause Detection ---

def test_detect_clauses_termination():
    text = "This agreement may be terminated by either party upon 30 days written notice."
    clauses = detect_clauses(text)
    assert "termination" in clauses


def test_detect_clauses_indemnification():
    text = "Contractor shall indemnify and hold harmless the Client against all claims."
    clauses = detect_clauses(text)
    assert "indemnification" in clauses


def test_detect_clauses_multiple():
    text = (
        "The parties agree to maintain confidentiality of all proprietary information. "
        "Any disputes shall be resolved by arbitration under the governing law of Delaware."
    )
    clauses = detect_clauses(text)
    assert "confidentiality" in clauses
    assert "governing_law" in clauses


def test_detect_clauses_none():
    text = "The weather today is sunny and warm."
    clauses = detect_clauses(text)
    assert clauses == []


def test_detect_clauses_liability():
    text = "In no event shall the aggregate liability of either party exceed the total fees paid."
    clauses = detect_clauses(text)
    assert "liability" in clauses


def test_detect_clauses_force_majeure():
    text = "Neither party shall be liable for failure caused by force majeure events."
    clauses = detect_clauses(text)
    assert "force_majeure" in clauses


def test_detect_clauses_warranty():
    text = "The software is provided AS-IS without warranty of merchantability."
    clauses = detect_clauses(text)
    assert "warranty" in clauses


# --- Table Detection ---

def test_detect_table_markdown():
    text = "| Name | Amount |\n|------|--------|\n| Item | $100   |"
    assert detect_table(text) is True


def test_detect_table_tabs():
    text = "Name\tAge\tCity\tCountry\nJohn\t30\tNYC\tUSA"
    assert detect_table(text) is True


def test_detect_table_none():
    text = "This is a normal paragraph of text."
    assert detect_table(text) is False


# --- Section Heading Detection ---

def test_detect_section_heading_numbered():
    text = "Section 3.2 — Termination Rights\nThe client may terminate..."
    heading = detect_section_heading(text)
    assert heading is not None
    assert "Termination" in heading


def test_detect_section_heading_article():
    text = "ARTICLE IV\nRepresentations and Warranties"
    heading = detect_section_heading(text)
    assert heading is not None


def test_detect_section_heading_allcaps():
    text = "GOVERNING LAW AND JURISDICTION\nThis agreement shall be governed..."
    heading = detect_section_heading(text)
    assert heading is not None


def test_detect_section_heading_none():
    text = "The parties hereto agree to the following terms and conditions."
    heading = detect_section_heading(text)
    assert heading is None


# --- Document Classification ---

def test_classify_contract():
    text = "This Agreement is entered into by and between the parties hereto. WHEREAS, the Company wishes to engage..."
    doc_type = classify_document_type(text)
    assert doc_type == "contract"


def test_classify_brief():
    text = "IN THE UNITED STATES DISTRICT COURT. The Plaintiff respectfully submits this memorandum of law."
    doc_type = classify_document_type(text)
    assert doc_type == "brief"


def test_classify_statute():
    text = "Be it enacted by the Senate. Section 42 U.S.C. § 1983 provides that..."
    doc_type = classify_document_type(text)
    assert doc_type == "statute"


def test_classify_unknown():
    text = "The quick brown fox jumps over the lazy dog."
    doc_type = classify_document_type(text)
    assert doc_type == "unknown"


# --- Chunking with Metadata ---

def test_chunk_with_metadata_basic():
    pages = [
        {"text": "This agreement may be terminated upon notice.\n\nConfidential information shall not be disclosed.", "page_number": 1, "total_pages": 1}
    ]
    doc = chunk_with_metadata(pages, "test.pdf", "/tmp/test.pdf", chunk_size=2000)
    assert doc.total_chunks > 0
    assert doc.chunks[0].file_name == "test.pdf"
    assert doc.chunks[0].page_number == 1


def test_chunk_preserves_clause_detection():
    pages = [
        {"text": "The contractor shall indemnify and hold harmless the client against all claims arising from negligence.", "page_number": 3, "total_pages": 10}
    ]
    doc = chunk_with_metadata(pages, "contract.pdf", "/tmp/contract.pdf")
    assert any("indemnification" in c.detected_clauses for c in doc.chunks)


def test_chunk_long_paragraph_splits():
    long_text = "This is a sentence. " * 200  # ~4000 chars
    pages = [{"text": long_text, "page_number": 1, "total_pages": 1}]
    doc = chunk_with_metadata(pages, "long.pdf", "/tmp/long.pdf", chunk_size=500)
    assert doc.total_chunks > 1


def test_chunk_to_metadata_dict():
    pages = [
        {"text": "Test content for metadata conversion.", "page_number": 5, "total_pages": 20}
    ]
    doc = chunk_with_metadata(pages, "test.pdf", "/tmp/test.pdf")
    meta = chunk_to_metadata(doc.chunks[0])

    assert meta["file_name"] == "test.pdf"
    assert meta["page_number"] == 5
    assert meta["total_pages"] == 20
    assert isinstance(meta["detected_clauses"], str)
    assert isinstance(meta["contains_table"], bool)


# --- File Processing ---

def test_process_text_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("This Agreement is entered into between Party A and Party B.\n\n"
                "The parties agree to maintain confidentiality of trade secrets.\n\n"
                "This agreement shall be governed by the laws of Delaware.")
        tmp_path = f.name

    try:
        doc = process_file(tmp_path, use_nougat=False)
        assert doc.file_name.endswith(".txt")
        assert doc.total_chunks > 0
        assert doc.document_type == "contract"
    finally:
        os.unlink(tmp_path)


def test_process_empty_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("")
        tmp_path = f.name

    try:
        doc = process_file(tmp_path, use_nougat=False)
        assert doc.total_chunks == 0
    finally:
        os.unlink(tmp_path)
