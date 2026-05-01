"""
Tegifa Legal — Cognitive Orchestrator
Multi-agent LangGraph pipeline: Reasoner → Epistemologist → Investigator → loop.
"""
from typing import TypedDict, Annotated, Sequence, List
import operator
import logging
from langgraph.graph import StateGraph, END
from agents.tools import InvestigatorTools

logger = logging.getLogger("tegifa.orchestrator")

MAX_ITERATIONS = 2  # Hard cap to prevent infinite loops


# ---------------------------------------------------------
# State Schema
# ---------------------------------------------------------
class CognitiveState(TypedDict):
    query: str
    local_context: str
    messages: Annotated[Sequence[dict], operator.add]
    current_hypothesis: str
    investigation_results: List[str]
    critique_passed: bool
    iterations: int


# ---------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------
def reasoning_node(state: CognitiveState, llm=None) -> dict:
    """Senior Reasoning Agent: synthesizes hypothesis from context + evidence."""
    logger.info("Reasoning Agent: synthesizing hypothesis...")
    query = state["query"]
    context = state["local_context"]
    external_data = "\n".join(state["investigation_results"])

    prompt = (
        "You are the Senior Reasoning Agent in a legal firm.\n"
        "Analyze the user's query using the provided local document context "
        "and any external investigation data.\n\n"
        f"USER QUERY: {query}\n\n"
        f"LOCAL DOCUMENT CONTEXT:\n{context}\n\n"
        f"EXTERNAL INVESTIGATION DATA:\n{external_data or 'None yet.'}\n\n"
        "Provide a clear, detailed legal hypothesis or answer based strictly "
        "on the provided evidence."
    )

    if llm:
        hypothesis = str(llm.complete(prompt))
    elif external_data:
        hypothesis = (
            f"Based on local context and external investigation: "
            f"{external_data[:200]}... The answer is adjusted."
        )
    else:
        hypothesis = (
            "Based purely on local context, this is the initial drafted hypothesis."
        )

    return {
        "current_hypothesis": hypothesis,
        "messages": [{"step": "Reasoning Formulation", "content": hypothesis}],
    }


def epistemology_node(state: CognitiveState, llm=None) -> dict:
    """Epistemology Agent (The Skeptic): critiques the hypothesis."""
    logger.info("Epistemology Agent: critiquing hypothesis...")
    query = state["query"]
    hypothesis = state["current_hypothesis"]

    if state["iterations"] >= MAX_ITERATIONS:
        return {
            "critique_passed": True,
            "messages": [
                {
                    "step": "Epistemology Critique",
                    "content": "Maximum iterations reached. Hypothesis accepted as final.",
                }
            ],
            "iterations": state["iterations"] + 1,
        }

    prompt = (
        "You are the Epistemology Agent (The Skeptic) in a legal firm.\n"
        "Review the proposed hypothesis for the given query.\n"
        "If the hypothesis lacks sufficient evidence, hallucinates, or explicitly "
        "says it doesn't know, you must REJECT it and demand external investigation.\n"
        "Otherwise, ACCEPT it.\n\n"
        f"USER QUERY: {query}\n"
        f"PROPOSED HYPOTHESIS: {hypothesis}\n\n"
        'Your response MUST start with either "ACCEPT" or "REJECT", '
        "followed by a brief explanation."
    )

    if llm:
        response = str(llm.complete(prompt))
        passed = response.strip().upper().startswith("ACCEPT")
        critique = response
    else:
        passed = False
        critique = (
            "REJECT. Hypothesis lacks external verification. "
            "I recommend investigating recent case law."
        )

    return {
        "critique_passed": passed,
        "messages": [{"step": "Epistemology Critique", "content": critique}],
        "iterations": state["iterations"] + 1,
    }


def investigator_node(state: CognitiveState) -> dict:
    """Investigator Agent: searches external databases for evidence."""
    logger.info("Investigator Agent: searching external sources...")
    search_results = InvestigatorTools.search_duckduckgo(
        state["query"], max_results=2
    )
    return {
        "investigation_results": [search_results],
        "messages": [
            {
                "step": "Investigation",
                "content": f"Found external data: {search_results[:200]}...",
            }
        ],
    }


# ---------------------------------------------------------
# Edge Routing
# ---------------------------------------------------------
def route_after_critique(state: CognitiveState) -> str:
    return END if state["critique_passed"] else "investigator"


# ---------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------
def build_cognitive_graph(llm=None):
    """Build and compile the LangGraph multi-agent pipeline."""
    workflow = StateGraph(CognitiveState)

    workflow.add_node("reasoner", lambda state: reasoning_node(state, llm))
    workflow.add_node("epistemologist", lambda state: epistemology_node(state, llm))
    workflow.add_node("investigator", investigator_node)

    workflow.set_entry_point("reasoner")
    workflow.add_edge("reasoner", "epistemologist")
    workflow.add_conditional_edges("epistemologist", route_after_critique)
    workflow.add_edge("investigator", "reasoner")

    return workflow.compile()


def run_cognitive_cycle(
    query: str, local_context: str = "", llm=None
) -> dict:
    """Execute the full multi-agent cognitive cycle and return answer + thought stream."""
    graph = build_cognitive_graph(llm)
    initial_state: CognitiveState = {
        "query": query,
        "local_context": local_context,
        "messages": [],
        "current_hypothesis": "",
        "investigation_results": [],
        "critique_passed": False,
        "iterations": 0,
    }

    final_state = graph.invoke(initial_state, {"recursion_limit": 10})
    return {
        "answer": final_state["current_hypothesis"],
        "thought_stream": final_state["messages"],
    }
