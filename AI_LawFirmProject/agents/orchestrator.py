from typing import TypedDict, Annotated, Sequence, List
import operator
import logging
from langgraph.graph import StateGraph, END
from AI_LawFirmProject.agents.tools import InvestigatorTools

logger = logging.getLogger("CognitiveOrchestrator")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------
# Define the State for the Multi-Agent Graph
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
def reasoning_node(state: CognitiveState, llm=None):
    logger.info("Reasoning Agent: Synthesizing hypothesis...")
    query = state["query"]
    context = state["local_context"]
    external_data = "\n".join(state["investigation_results"])

    prompt = f"""
    You are the Senior Reasoning Agent in a legal firm.
    Analyze the user's query using the provided local document context and any external investigation data.

    USER QUERY: {query}

    LOCAL DOCUMENT CONTEXT:
    {context}

    EXTERNAL INVESTIGATION DATA:
    {external_data if external_data else 'None yet.'}

    Provide a clear, detailed legal hypothesis or answer based strictly on the provided evidence.
    """

    if llm:
        hypothesis = str(llm.complete(prompt))
    else:
        # Fallback simulation if LLM not injected
        if external_data:
            hypothesis = f"Based on local context and external investigation: {external_data[:100]}... The answer is adjusted."
        else:
            hypothesis = "Based purely on local context, this is the initial drafted hypothesis."

    return {
        "current_hypothesis": hypothesis,
        "messages": [{"step": "Reasoning Formulation", "content": hypothesis}]
    }

def epistemology_node(state: CognitiveState, llm=None):
    logger.info("Epistemology Agent: Critiquing hypothesis...")
    query = state["query"]
    hypothesis = state["current_hypothesis"]

    if state["iterations"] >= 1:
        # Pass on the second iteration to avoid infinite loops and limit API calls
        return {
            "critique_passed": True,
            "messages": [{"step": "Epistemology Critique", "content": "Maximum iteration reached. Hypothesis accepted as final."}],
            "iterations": state["iterations"] + 1
        }

    prompt = f"""
    You are the Epistemology Agent (The Skeptic) in a legal firm.
    Review the proposed hypothesis for the given query.
    If the hypothesis seems to lack sufficient evidence, hallucinate, or if it explicitly says it doesn't know the answer, you must REJECT it and demand an external investigation.
    Otherwise, ACCEPT it.

    USER QUERY: {query}
    PROPOSED HYPOTHESIS: {hypothesis}

    Your response MUST start with either "ACCEPT" or "REJECT", followed by a brief explanation.
    """

    if llm:
        response = str(llm.complete(prompt))
        passed = response.strip().upper().startswith("ACCEPT")
        critique = response
    else:
        # Simulated critique
        passed = False
        critique = "REJECT. Hypothesis lacks external verification. I recommend investigating recent case law."

    return {
        "critique_passed": passed,
        "messages": [{"step": "Epistemology Critique", "content": critique}],
        "iterations": state["iterations"] + 1
    }

def investigator_node(state: CognitiveState):
    logger.info("Investigator Agent: Searching external databases...")
    query = state["query"]

    # Call the actual web search tool
    search_results = InvestigatorTools.search_duckduckgo(query, max_results=2)

    return {
        "investigation_results": [search_results],
        "messages": [{"step": "Investigation", "content": f"Found external data: {search_results[:100]}..."}]
    }

# ---------------------------------------------------------
# Edge Routing
# ---------------------------------------------------------
def route_after_critique(state: CognitiveState):
    if state["critique_passed"]:
        return END
    else:
        return "investigator"

# ---------------------------------------------------------
# Graph Orchestration
# ---------------------------------------------------------
def build_cognitive_graph(llm=None):
    workflow = StateGraph(CognitiveState)

    # We use functools.partial or lambdas to inject the LLM into the nodes
    workflow.add_node("reasoner", lambda state: reasoning_node(state, llm))
    workflow.add_node("epistemologist", lambda state: epistemology_node(state, llm))
    workflow.add_node("investigator", investigator_node)

    workflow.set_entry_point("reasoner")
    workflow.add_edge("reasoner", "epistemologist")
    workflow.add_conditional_edges("epistemologist", route_after_critique)
    workflow.add_edge("investigator", "reasoner")

    return workflow.compile()

def run_cognitive_cycle(query: str, local_context: str = "", llm=None) -> dict:
    graph = build_cognitive_graph(llm)
    initial_state = {
        "query": query,
        "local_context": local_context,
        "messages": [],
        "current_hypothesis": "",
        "investigation_results": [],
        "critique_passed": False,
        "iterations": 0
    }

    # Invoke the graph (using config recursion_limit to be safe)
    final_state = graph.invoke(initial_state, {"recursion_limit": 10})
    return {
        "answer": final_state["current_hypothesis"],
        "thought_stream": final_state["messages"]
    }
