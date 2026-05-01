"""
Tegifa Legal — Advanced Cognitive Architecture for Generation (CAG)
A LangGraph-based state machine capable of routing queries to specialized agents
and triggering external tools/webhooks (e.g., n8n).
"""
import logging
import operator
import json
import requests
from typing import TypedDict, Annotated, Sequence, List
from langgraph.graph import StateGraph, END

logger = logging.getLogger("tegifa.advanced_cag")

N8N_WEBHOOK_URL = "http://localhost:5678/webhook/tegifa-action"

# ---------------------------------------------------------
# State Schema
# ---------------------------------------------------------
class AdvancedState(TypedDict):
    query: str
    local_context: str
    messages: Annotated[Sequence[dict], operator.add]
    route_decision: str
    action_payload: dict
    action_result: str
    final_answer: str

# ---------------------------------------------------------
# Nodes
# ---------------------------------------------------------
def router_node(state: AdvancedState, llm) -> dict:
    """Decides if the query needs Legal Reasoning, Conflict Resolution, or an Action/Webhook."""
    logger.info("Router Agent: analyzing query...")
    query = state["query"]
    
    prompt = (
        "You are the Lead Routing Agent for a Legal AI.\n"
        "Analyze the user query and decide the best execution path.\n\n"
        "PATHS:\n"
        "1. 'ACTION' - If the user explicitly asks to DO something external (e.g., 'send an email', 'alert John', 'trigger webhook').\n"
        "2. 'CONFLICT' - If the user asks about definition drift, contradictions, or deal graph conflicts.\n"
        "3. 'INVESTIGATOR' - If the user asks about current events, recent laws, news, or outside general knowledge requiring a web search.\n"
        "4. 'REASONER' - For general legal Q&A, summarizing, or standard RAG queries on the uploaded documents.\n\n"
        f"USER QUERY: {query}\n\n"
        "Respond with ONLY ONE WORD representing the path: ACTION, CONFLICT, INVESTIGATOR, or REASONER."
    )
    
    route = str(llm.complete(prompt)).strip().upper()
    
    # Fallback and normalization
    if "ACTION" in route: route = "ACTION"
    elif "CONFLICT" in route: route = "CONFLICT"
    elif "INVESTIGATOR" in route: route = "INVESTIGATOR"
    else: route = "REASONER"
    
    return {
        "route_decision": route,
        "messages": [{"step": "Routing", "content": f"Decided path: {route}"}]
    }

def reasoner_node(state: AdvancedState, llm) -> dict:
    """Standard legal Q&A using provided context."""
    logger.info("Legal Reasoner Agent: generating answer...")
    prompt = (
        "You are a Senior Legal Reasoner. Answer the query based on the context.\n\n"
        f"Context: {state['local_context']}\n\n"
        f"Query: {state['query']}"
    )
    answer = str(llm.complete(prompt))
    return {
        "final_answer": answer,
        "messages": [{"step": "Legal Reasoning", "content": "Generated standard legal response."}]
    }

def conflict_resolver_node(state: AdvancedState, llm) -> dict:
    """Specialized node for handling contradictions and Definition Drift."""
    logger.info("Conflict Resolver Agent: analyzing contradictions...")
    prompt = (
        "You are a Conflict Resolution Legal Agent. Look closely at the context and find any contradictions, "
        "ambiguities, or definition drift. Explain the risk clearly.\n\n"
        f"Context: {state['local_context']}\n\n"
        f"Query: {state['query']}"
    )
    answer = str(llm.complete(prompt))
    return {
        "final_answer": answer,
        "messages": [{"step": "Conflict Resolution", "content": "Analyzed deep legal conflicts."}]
    }

def action_executor_node(state: AdvancedState, llm) -> dict:
    """Formats payload and executes an external tool or n8n webhook."""
    logger.info("Action Executor: formatting and triggering webhook...")
    
    prompt = (
        "Extract actionable parameters from the user query to send to an external system.\n"
        "Return ONLY a JSON object with keys: 'action_type', 'target', and 'message'.\n"
        f"Query: {state['query']}"
    )
    payload_str = str(llm.complete(prompt)).strip()
    
    # Simple JSON extraction
    try:
        import re
        json_match = re.search(r'\{.*\}', payload_str, re.DOTALL)
        payload = json.loads(json_match.group(0)) if json_match else {"raw": payload_str}
    except Exception:
        payload = {"action_type": "unknown", "message": state['query']}
        
    try:
        # Attempt to hit the n8n webhook (non-blocking if it fails)
        response = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=2)
        action_res = f"Webhook triggered successfully. Status: {response.status_code}"
    except Exception as e:
        action_res = f"Webhook simulated. (n8n not reachable at {N8N_WEBHOOK_URL})"
        
    answer = f"Action processed.\nPayload: {json.dumps(payload, indent=2)}\nResult: {action_res}"
    return {
        "action_payload": payload,
        "action_result": action_res,
        "final_answer": answer,
        "messages": [{"step": "Action Execution", "content": answer}]
    }

def investigator_node(state: AdvancedState, llm) -> dict:
    """Agent for fetching live information from the internet."""
    logger.info("Investigator Agent: executing web search...")
    from agents.tools import InvestigatorTools
    
    search_prompt = (
        "Extract the core search term from this query to look up on the web.\n"
        f"Query: {state['query']}\n"
        "Return ONLY the search term string."
    )
    search_term = str(llm.complete(search_prompt)).strip()
    
    results = InvestigatorTools.search_duckduckgo(search_term, max_results=3)
    
    answer_prompt = (
        "You are a Legal Research Assistant. Using the following web search results, answer the user's query.\n\n"
        f"Web Results:\n{results}\n\n"
        f"User Query: {state['query']}"
    )
    answer = str(llm.complete(answer_prompt))
    
    return {
        "final_answer": answer,
        "messages": [
            {"step": "Web Search Execution", "content": f"Searched for: {search_term}\nFound results."},
            {"step": "Research Synthesis", "content": "Synthesized web search results into an answer."}
        ]
    }

# ---------------------------------------------------------
# Edges
# ---------------------------------------------------------
def route_from_router(state: AdvancedState) -> str:
    route = state.get("route_decision", "REASONER")
    if route == "ACTION": return "action_executor"
    if route == "CONFLICT": return "conflict_resolver"
    if route == "INVESTIGATOR": return "investigator"
    return "reasoner"

# ---------------------------------------------------------
# Graph Compilation
# ---------------------------------------------------------
def build_advanced_cag(llm):
    workflow = StateGraph(AdvancedState)
    
    workflow.add_node("router", lambda s: router_node(s, llm))
    workflow.add_node("reasoner", lambda s: reasoner_node(s, llm))
    workflow.add_node("conflict_resolver", lambda s: conflict_resolver_node(s, llm))
    workflow.add_node("action_executor", lambda s: action_executor_node(s, llm))
    workflow.add_node("investigator", lambda s: investigator_node(s, llm))
    
    workflow.set_entry_point("router")
    
    workflow.add_conditional_edges("router", route_from_router)
    
    workflow.add_edge("reasoner", END)
    workflow.add_edge("conflict_resolver", END)
    workflow.add_edge("action_executor", END)
    workflow.add_edge("investigator", END)
    
    return workflow.compile()

def run_advanced_cag(query: str, local_context: str, llm) -> dict:
    graph = build_advanced_cag(llm)
    initial_state: AdvancedState = {
        "query": query,
        "local_context": local_context,
        "messages": [],
        "route_decision": "",
        "action_payload": {},
        "action_result": "",
        "final_answer": ""
    }
    
    final_state = graph.invoke(initial_state, {"recursion_limit": 10})
    return {
        "answer": final_state["final_answer"],
        "thought_stream": final_state["messages"]
    }
