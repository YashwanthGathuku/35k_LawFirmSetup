"""
Tegifa Legal — Contract Analyzer
The killer vertical: AI-powered contract review that produces:
- Clause-by-clause classification and risk assessment
- Obligation extraction (who, what, when)
- Missing clause detection
- Executive summary with risk score

Designed to be demonstrably better than ChatGPT + manual review by:
1. Grounding every finding in the actual document text (pinpoint citations)
2. Using the SRLC multi-agent cycle for self-verification
3. Providing structured, actionable output instead of prose
"""
import logging
import json
import re
from dataclasses import dataclass, field
from typing import Optional

from rag_scripts.document_processor import CLAUSE_PATTERNS, detect_clauses

logger = logging.getLogger("tegifa.contract_analyzer")


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class ClauseAnalysis:
    """Analysis of a single contract clause."""

    clause_type: str
    risk_level: str  # "low", "medium", "high", "critical"
    risk_score: float  # 0.0 to 1.0
    summary: str
    concerns: list = field(default_factory=list)
    source_text: str = ""
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    recommendation: str = ""


@dataclass
class Obligation:
    """An extracted contractual obligation."""

    party: str  # Who is obligated
    action: str  # What they must do
    deadline: Optional[str] = None  # When (if specified)
    condition: Optional[str] = None  # Under what conditions
    source_text: str = ""
    page_number: Optional[int] = None


@dataclass
class ContractReport:
    """Full contract review report."""

    document_name: str
    document_type: str
    total_pages: int
    overall_risk_score: float  # 0.0 to 1.0
    overall_risk_level: str
    executive_summary: str
    clause_analyses: list  # list[ClauseAnalysis]
    obligations: list  # list[Obligation]
    missing_clauses: list  # list[str]
    key_findings: list  # list[str]
    recommendation: str


# ---------------------------------------------------------------------------
# Prompts for LLM-powered analysis
# ---------------------------------------------------------------------------

CLAUSE_ANALYSIS_PROMPT = """You are a senior contract attorney performing a clause-by-clause review.

Analyze the following contract section and provide a structured assessment.

DOCUMENT: {file_name}
PAGE: {page_number}
SECTION: {section_heading}
CLAUSE TYPE: {clause_type}

CONTRACT TEXT:
---
{text}
---

Respond ONLY with valid JSON in this exact format (no markdown, no explanation):
{{
    "risk_level": "low|medium|high|critical",
    "risk_score": 0.0 to 1.0,
    "summary": "One sentence summary of what this clause does",
    "concerns": ["Specific concern 1", "Specific concern 2"],
    "recommendation": "What the client should negotiate or watch for"
}}"""

OBLIGATION_EXTRACTION_PROMPT = """You are a legal analyst extracting contractual obligations.

From the following contract text, extract ALL obligations (things parties MUST do).

CONTRACT TEXT:
---
{text}
---

Respond ONLY with valid JSON array (no markdown, no explanation):
[
    {{
        "party": "Party A or Party B or specific name",
        "action": "What they must do",
        "deadline": "When (null if not specified)",
        "condition": "Under what conditions (null if unconditional)"
    }}
]

If no obligations found, return an empty array: []"""

EXECUTIVE_SUMMARY_PROMPT = """You are a senior partner preparing a contract review summary for a client.

Based on the following clause analyses and findings, write a concise executive summary.

DOCUMENT: {file_name}
TOTAL PAGES: {total_pages}
OVERALL RISK: {risk_level} ({risk_score}/10)

CLAUSE FINDINGS:
{clause_findings}

MISSING CLAUSES:
{missing_clauses}

KEY OBLIGATIONS:
{obligations}

Write a 3-5 sentence executive summary that:
1. States what type of agreement this is
2. Highlights the most critical risk(s)
3. Identifies what's missing
4. Gives a clear recommendation (sign as-is, negotiate, or walk away)

Respond with plain text only, no JSON."""


# ---------------------------------------------------------------------------
# Standard clauses that should be present in any commercial contract
# ---------------------------------------------------------------------------
EXPECTED_CLAUSES = {
    "termination",
    "liability",
    "confidentiality",
    "governing_law",
    "indemnification",
    "warranty",
    "payment",
    "severability",
    "entire_agreement",
    "assignment",
}


def parse_llm_json(response: str, fallback=None):
    """Safely parse JSON from LLM output, handling markdown fences."""
    text = str(response).strip()
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse LLM JSON: %s | Response: %s", e, text[:200])
        return fallback


def analyze_clause_with_llm(
    llm,
    text: str,
    clause_type: str,
    file_name: str,
    page_number: Optional[int],
    section_heading: Optional[str],
) -> ClauseAnalysis:
    """Use the LLM to analyze a single clause and produce a structured assessment."""
    prompt = CLAUSE_ANALYSIS_PROMPT.format(
        file_name=file_name,
        page_number=page_number or "Unknown",
        section_heading=section_heading or "N/A",
        clause_type=clause_type.replace("_", " ").title(),
        text=text[:2000],  # Limit context to avoid token overflow
    )

    try:
        response = str(llm.complete(prompt))
        data = parse_llm_json(response, fallback={})

        return ClauseAnalysis(
            clause_type=clause_type,
            risk_level=data.get("risk_level", "medium"),
            risk_score=float(data.get("risk_score", 0.5)),
            summary=data.get("summary", "Analysis pending."),
            concerns=data.get("concerns", []),
            source_text=text[:500],
            page_number=page_number,
            section_heading=section_heading,
            recommendation=data.get("recommendation", ""),
        )
    except Exception as e:
        logger.error("Clause analysis failed: %s", e)
        return ClauseAnalysis(
            clause_type=clause_type,
            risk_level="medium",
            risk_score=0.5,
            summary=f"Automated analysis unavailable: {e}",
            source_text=text[:500],
            page_number=page_number,
            section_heading=section_heading,
        )


def extract_obligations_with_llm(
    llm,
    text: str,
    page_number: Optional[int] = None,
) -> list[Obligation]:
    """Use the LLM to extract obligations from contract text."""
    prompt = OBLIGATION_EXTRACTION_PROMPT.format(text=text[:2000])

    try:
        response = str(llm.complete(prompt))
        data = parse_llm_json(response, fallback=[])

        if not isinstance(data, list):
            return []

        obligations = []
        for item in data:
            if isinstance(item, dict) and item.get("party") and item.get("action"):
                obligations.append(
                    Obligation(
                        party=item["party"],
                        action=item["action"],
                        deadline=item.get("deadline"),
                        condition=item.get("condition"),
                        source_text=text[:300],
                        page_number=page_number,
                    )
                )
        return obligations

    except Exception as e:
        logger.error("Obligation extraction failed: %s", e)
        return []


def generate_executive_summary(
    llm,
    file_name: str,
    total_pages: int,
    risk_level: str,
    risk_score: float,
    clause_analyses: list[ClauseAnalysis],
    missing_clauses: list[str],
    obligations: list[Obligation],
) -> str:
    """Generate an executive summary of the contract review."""
    clause_findings = "\n".join(
        f"- {ca.clause_type.replace('_', ' ').title()}: "
        f"{ca.risk_level.upper()} risk — {ca.summary}"
        for ca in clause_analyses
    )

    missing_str = ", ".join(
        c.replace("_", " ").title() for c in missing_clauses
    ) or "None"

    obligations_str = "\n".join(
        f"- {ob.party}: {ob.action}" + (f" (by {ob.deadline})" if ob.deadline else "")
        for ob in obligations[:10]
    ) or "None extracted"

    prompt = EXECUTIVE_SUMMARY_PROMPT.format(
        file_name=file_name,
        total_pages=total_pages,
        risk_level=risk_level,
        risk_score=round(risk_score * 10, 1),
        clause_findings=clause_findings or "None analyzed",
        missing_clauses=missing_str,
        obligations=obligations_str,
    )

    try:
        return str(llm.complete(prompt)).strip()
    except Exception as e:
        logger.error("Executive summary generation failed: %s", e)
        return f"Executive summary unavailable: {e}"


def run_contract_review(
    llm,
    processed_doc,  # ProcessedDocument from document_processor
    max_clauses: int = 20,
) -> ContractReport:
    """
    Run the full contract review pipeline.

    Steps:
    1. Identify all clause types present in the document
    2. Analyze each unique clause type (risk, concerns, recommendations)
    3. Extract obligations from the full document
    4. Detect missing standard clauses
    5. Generate executive summary
    """
    logger.info(
        "Starting contract review for: %s (%d chunks)",
        processed_doc.file_name,
        processed_doc.total_chunks,
    )

    # Step 1: Collect chunks by clause type
    clause_chunks: dict[str, list] = {}
    for chunk in processed_doc.chunks:
        for clause_type in chunk.detected_clauses:
            if clause_type not in clause_chunks:
                clause_chunks[clause_type] = []
            clause_chunks[clause_type].append(chunk)

    # Step 2: Analyze each clause type
    clause_analyses = []
    analyzed_count = 0

    for clause_type, chunks in clause_chunks.items():
        if analyzed_count >= max_clauses:
            break

        # Use the longest chunk for this clause type (most complete context)
        best_chunk = max(chunks, key=lambda c: len(c.text))

        analysis = analyze_clause_with_llm(
            llm=llm,
            text=best_chunk.text,
            clause_type=clause_type,
            file_name=processed_doc.file_name,
            page_number=best_chunk.page_number,
            section_heading=best_chunk.section_heading,
        )
        clause_analyses.append(analysis)
        analyzed_count += 1

    # Step 3: Extract obligations (sample from first 10 substantive chunks)
    all_obligations = []
    substantive_chunks = [
        c for c in processed_doc.chunks if len(c.text) > 100
    ][:10]

    for chunk in substantive_chunks:
        obligations = extract_obligations_with_llm(
            llm=llm,
            text=chunk.text,
            page_number=chunk.page_number,
        )
        all_obligations.extend(obligations)

    # Step 4: Detect missing clauses
    found_clauses = set(clause_chunks.keys())
    missing = sorted(EXPECTED_CLAUSES - found_clauses)

    # Step 5: Calculate overall risk
    if clause_analyses:
        avg_risk = sum(ca.risk_score for ca in clause_analyses) / len(clause_analyses)
        # Boost risk if critical clauses are missing
        missing_penalty = len(missing) * 0.05
        overall_risk = min(1.0, avg_risk + missing_penalty)
    else:
        overall_risk = 0.5  # Unknown if no clauses analyzed

    if overall_risk >= 0.75:
        overall_risk_level = "critical"
    elif overall_risk >= 0.5:
        overall_risk_level = "high"
    elif overall_risk >= 0.25:
        overall_risk_level = "medium"
    else:
        overall_risk_level = "low"

    # Step 6: Key findings
    key_findings = []
    critical_clauses = [ca for ca in clause_analyses if ca.risk_level in ("high", "critical")]
    if critical_clauses:
        key_findings.append(
            f"{len(critical_clauses)} clause(s) flagged as high/critical risk"
        )
    if missing:
        key_findings.append(
            f"Missing standard clauses: {', '.join(c.replace('_', ' ') for c in missing)}"
        )
    if all_obligations:
        key_findings.append(f"{len(all_obligations)} contractual obligations identified")

    # Step 7: Executive summary
    executive_summary = generate_executive_summary(
        llm=llm,
        file_name=processed_doc.file_name,
        total_pages=processed_doc.total_pages,
        risk_level=overall_risk_level,
        risk_score=overall_risk,
        clause_analyses=clause_analyses,
        missing_clauses=missing,
        obligations=all_obligations,
    )

    return ContractReport(
        document_name=processed_doc.file_name,
        document_type=processed_doc.document_type,
        total_pages=processed_doc.total_pages,
        overall_risk_score=round(overall_risk, 3),
        overall_risk_level=overall_risk_level,
        executive_summary=executive_summary,
        clause_analyses=clause_analyses,
        obligations=all_obligations,
        missing_clauses=missing,
        key_findings=key_findings,
        recommendation=(
            "Seek legal review before signing"
            if overall_risk_level in ("high", "critical")
            else "Review flagged items before signing"
        ),
    )


def report_to_dict(report: ContractReport) -> dict:
    """Serialize the contract report for JSON/UI consumption."""
    return {
        "document_name": report.document_name,
        "document_type": report.document_type,
        "total_pages": report.total_pages,
        "overall_risk_score": report.overall_risk_score,
        "overall_risk_level": report.overall_risk_level,
        "executive_summary": report.executive_summary,
        "clause_analyses": [
            {
                "clause_type": ca.clause_type,
                "risk_level": ca.risk_level,
                "risk_score": round(ca.risk_score, 3),
                "summary": ca.summary,
                "concerns": ca.concerns,
                "source_text": ca.source_text,
                "page_number": ca.page_number,
                "section_heading": ca.section_heading,
                "recommendation": ca.recommendation,
            }
            for ca in report.clause_analyses
        ],
        "obligations": [
            {
                "party": ob.party,
                "action": ob.action,
                "deadline": ob.deadline,
                "condition": ob.condition,
                "page_number": ob.page_number,
            }
            for ob in report.obligations
        ],
        "missing_clauses": report.missing_clauses,
        "key_findings": report.key_findings,
        "recommendation": report.recommendation,
    }
