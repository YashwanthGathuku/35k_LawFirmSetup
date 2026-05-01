"""
Tegifa Legal — Citation Engine
Produces precise, verifiable citations with:
- Page and paragraph-level pinpointing
- Confidence scores derived from vector similarity
- Legal-style citation formatting
- Source deduplication and ranking
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("tegifa.citations")


@dataclass
class Citation:
    """A single pinpoint citation to a source document."""

    file_name: str
    page_number: Optional[int]
    paragraph_index: Optional[int]
    section_heading: Optional[str]
    snippet: str  # The actual text that supports the claim
    confidence: float  # 0.0 to 1.0, derived from similarity score
    confidence_label: str  # "high", "medium", "low"
    detected_clauses: list = field(default_factory=list)
    contains_table: bool = False
    chunk_index: int = 0
    total_chunks: int = 0
    total_pages: Optional[int] = None

    @property
    def pinpoint(self) -> str:
        """Generate a legal-style pinpoint citation string."""
        parts = [self.file_name]
        if self.page_number is not None:
            parts.append(f"p. {self.page_number}")
            if self.total_pages:
                parts[-1] += f" of {self.total_pages}"
        if self.section_heading:
            parts.append(f'§ "{self.section_heading}"')
        if self.paragraph_index is not None:
            parts.append(f"¶ {self.paragraph_index + 1}")
        return ", ".join(parts)

    @property
    def short_ref(self) -> str:
        """Short reference for inline use."""
        if self.page_number is not None:
            return f"{self.file_name}, p. {self.page_number}"
        return self.file_name


@dataclass
class CitationReport:
    """A collection of citations for a single query response."""

    citations: list  # list[Citation]
    total_sources: int
    unique_documents: int
    highest_confidence: float
    query: str

    @property
    def summary(self) -> str:
        """Human-readable citation summary."""
        if not self.citations:
            return "No source citations available."
        high = sum(1 for c in self.citations if c.confidence_label == "high")
        med = sum(1 for c in self.citations if c.confidence_label == "medium")
        return (
            f"{self.total_sources} citation(s) from {self.unique_documents} document(s). "
            f"{high} high confidence, {med} medium confidence."
        )


def score_to_confidence(score: float) -> tuple[float, str]:
    """
    Convert a raw similarity score to a normalized confidence value and label.

    ChromaDB/LlamaIndex scores can vary by embedding model, so we use
    relative thresholds rather than absolute values.
    """
    # Normalize: scores from most embedding models range ~0.0 to ~1.0
    # but can sometimes exceed 1.0 or be negative (cosine distance vs similarity)
    confidence = max(0.0, min(1.0, score))

    if confidence >= 0.75:
        return confidence, "high"
    elif confidence >= 0.50:
        return confidence, "medium"
    else:
        return confidence, "low"


def build_citation(node_result) -> Citation:
    """
    Build a Citation from a LlamaIndex NodeWithScore result.

    Extracts all page/paragraph metadata injected by document_processor.
    """
    metadata = node_result.node.metadata
    text = node_result.node.get_content()

    raw_score = (
        float(node_result.score)
        if hasattr(node_result, "score") and node_result.score is not None
        else 0.5
    )
    confidence, confidence_label = score_to_confidence(raw_score)

    # Parse detected_clauses from comma-separated string back to list
    clauses_str = metadata.get("detected_clauses", "")
    detected_clauses = [c.strip() for c in clauses_str.split(",") if c.strip()]

    return Citation(
        file_name=metadata.get("file_name", "Unknown"),
        page_number=metadata.get("page_number"),
        paragraph_index=metadata.get("paragraph_index"),
        section_heading=metadata.get("section_heading") or None,
        snippet=text[:500],  # First 500 chars for citation display
        confidence=confidence,
        confidence_label=confidence_label,
        detected_clauses=detected_clauses,
        contains_table=metadata.get("contains_table", False),
        chunk_index=metadata.get("chunk_index", 0),
        total_chunks=metadata.get("total_chunks", 0),
        total_pages=metadata.get("total_pages"),
    )


def build_citation_report(nodes: list, query: str) -> CitationReport:
    """
    Build a full citation report from retrieval results.

    Handles:
    - Citation construction with page-level metadata
    - Deduplication (same page/paragraph from same file)
    - Ranking by confidence
    - Summary statistics
    """
    citations = []
    seen = set()

    for node_result in nodes:
        citation = build_citation(node_result)

        # Deduplicate by (file, page, paragraph)
        dedup_key = (
            citation.file_name,
            citation.page_number,
            citation.paragraph_index,
        )
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        citations.append(citation)

    # Sort by confidence descending
    citations.sort(key=lambda c: c.confidence, reverse=True)

    unique_docs = len({c.file_name for c in citations})
    highest = max((c.confidence for c in citations), default=0.0)

    return CitationReport(
        citations=citations,
        total_sources=len(citations),
        unique_documents=unique_docs,
        highest_confidence=highest,
        query=query,
    )


def citation_to_dict(citation: Citation) -> dict:
    """Serialize a Citation for JSON/Streamlit consumption."""
    return {
        "file_name": citation.file_name,
        "page_number": citation.page_number,
        "paragraph_index": citation.paragraph_index,
        "section_heading": citation.section_heading,
        "snippet": citation.snippet,
        "confidence": round(citation.confidence, 3),
        "confidence_label": citation.confidence_label,
        "pinpoint": citation.pinpoint,
        "short_ref": citation.short_ref,
        "detected_clauses": citation.detected_clauses,
        "contains_table": citation.contains_table,
    }


def report_to_dict(report: CitationReport) -> dict:
    """Serialize a CitationReport for JSON/Streamlit consumption."""
    return {
        "citations": [citation_to_dict(c) for c in report.citations],
        "total_sources": report.total_sources,
        "unique_documents": report.unique_documents,
        "highest_confidence": round(report.highest_confidence, 3),
        "summary": report.summary,
    }
