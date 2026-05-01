"""
Tegifa Legal — Deal Anatomy Graph (NetworkX)
Extracts Defined Terms, Obligations, and Liabilities from deal documents
and maps their relationships into an in-memory graph for conflict detection.
"""
import logging
import json
import re
import networkx as nx
from typing import List, Dict, Any

logger = logging.getLogger("tegifa.deal_graph")

DEFINITION_EXTRACTION_PROMPT = """You are a legal analyst extracting capitalized Defined Terms and their definitions.

From the following contract text, extract ALL Defined Terms (usually capitalized, e.g., "Confidential Information", "Affiliate", "Services") and their exact definitions or meanings provided in the text.

CONTRACT TEXT:
---
{text}
---

Respond ONLY with valid JSON array (no markdown, no explanation):
[
    {{
        "term": "The Capitalized Term",
        "definition": "The exact definition or meaning given",
        "is_defined_here": true or false
    }}
]

If no defined terms found, return an empty array: []"""

OBLIGATION_EXTRACTION_PROMPT = """You are a legal analyst extracting obligations from contracts.

From the following contract text, extract ALL Obligations (things a party MUST do).

CONTRACT TEXT:
---
{text}
---

Respond ONLY with valid JSON array (no markdown):
[
    {{
        "party": "The obligated party",
        "action": "The required action",
        "deadline": "When it must be done",
        "conditions": "Any conditions"
    }}
]

If no obligations found, return an empty array: []"""

LIABILITY_EXTRACTION_PROMPT = """You are a legal analyst extracting liabilities and indemnities.

From the following contract text, extract ALL Liabilities, Indemnities, and Caps on Liability.

CONTRACT TEXT:
---
{text}
---

Respond ONLY with valid JSON array (no markdown):
[
    {{
        "type": "Indemnity or Liability Cap",
        "party": "The responsible party",
        "details": "Details of the liability or cap"
    }}
]

If no liabilities found, return an empty array: []"""


def parse_llm_json(response: str, fallback=None):
    """Safely parse JSON from LLM output, handling markdown fences."""
    text = str(response).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse LLM JSON: %s | Response: %s", e, text[:200])
        return fallback


def extract_definitions(llm, text: str) -> List[Dict[str, Any]]:
    """Extract defined terms from a chunk of text."""
    prompt = DEFINITION_EXTRACTION_PROMPT.format(text=text[:2500])
    try:
        response = str(llm.complete(prompt))
        data = parse_llm_json(response, fallback=[])
        if not isinstance(data, list):
            return []
        
        valid_terms = []
        for item in data:
            if isinstance(item, dict) and item.get("term") and item.get("definition"):
                valid_terms.append(item)
        return valid_terms
    except Exception as e:
        logger.error("Definition extraction failed: %s", e)
        return []

def extract_obligations(llm, text: str) -> List[Dict[str, Any]]:
    prompt = OBLIGATION_EXTRACTION_PROMPT.format(text=text[:2500])
    try:
        response = str(llm.complete(prompt))
        data = parse_llm_json(response, fallback=[])
        return [i for i in data if isinstance(i, dict) and i.get("party") and i.get("action")]
    except Exception as e:
        logger.error("Obligation extraction failed: %s", e)
        return []

def extract_liabilities(llm, text: str) -> List[Dict[str, Any]]:
    prompt = LIABILITY_EXTRACTION_PROMPT.format(text=text[:2500])
    try:
        response = str(llm.complete(prompt))
        data = parse_llm_json(response, fallback=[])
        return [i for i in data if isinstance(i, dict) and i.get("party") and i.get("details")]
    except Exception as e:
        logger.error("Liability extraction failed: %s", e)
        return []


def build_deal_graph(llm, processed_docs: List[Any]) -> nx.Graph:
    """
    Builds a NetworkX graph of the deal.
    processed_docs is a list of ProcessedDocument objects from document_processor.
    """
    G = nx.Graph()
    
    # Add Document Nodes
    for doc in processed_docs:
        doc_name = doc.file_name
        # Use simple label for node
        G.add_node(doc_name, type="Document", label=doc_name[:20] + "...")
        
        logger.info(f"Extracting definitions from {doc_name}...")
        
        # We sample the first few chunks to find definitions (usually in Article 1)
        chunks_to_process = doc.chunks[:5]
        
        for chunk in chunks_to_process:
            defs = extract_definitions(llm, chunk.text)
            for d in defs:
                term = d["term"]
                definition = d["definition"]
                is_defined_here = d.get("is_defined_here", True)
                
                # Add Term Node
                if not G.has_node(term):
                    G.add_node(term, type="DefinedTerm", label=term, definition=definition, source_doc=doc_name)
                elif is_defined_here:
                    # Check for definition drift
                    existing_def = G.nodes[term].get("definition", "")
                    if existing_def and existing_def != definition:
                        logger.warning(f"Definition Drift detected for '{term}'!")
                        # Mark the node to highlight conflict
                        G.nodes[term]["conflict"] = True
                        G.nodes[term]["definition"] = f"CONFLICT:\n1: {existing_def}\n2: {definition}"
                
                # Add Relationship
                if is_defined_here:
                    G.add_edge(doc_name, term, relation="DEFINES")
                else:
                    G.add_edge(doc_name, term, relation="USES")
            
            # Extract Obligations
            obs = extract_obligations(llm, chunk.text)
            for i, o in enumerate(obs):
                node_id = f"Obs_{doc_name}_{chunk.page_number}_{i}"
                label = f"{o['party']}: {o['action'][:15]}..."
                G.add_node(node_id, type="Obligation", label=label, details=o['action'], party=o['party'], deadline=o.get('deadline'), source_doc=doc_name)
                G.add_edge(doc_name, node_id, relation="REQUIRES")

            # Extract Liabilities
            liabs = extract_liabilities(llm, chunk.text)
            for i, l in enumerate(liabs):
                node_id = f"Liab_{doc_name}_{chunk.page_number}_{i}"
                label = f"{l['type'][:15]}..."
                G.add_node(node_id, type="Liability", label=label, details=l['details'], party=l['party'], source_doc=doc_name)
                G.add_edge(doc_name, node_id, relation="CREATES_RISK")

    return G

def get_conflicts(G: nx.Graph) -> List[Dict[str, str]]:
    """Scan the graph for terms with definition drift."""
    conflicts = []
    for node, data in G.nodes(data=True):
        if data.get("type") == "DefinedTerm" and data.get("conflict"):
            conflicts.append({
                "term": node,
                "details": data.get("definition")
            })
    return conflicts
