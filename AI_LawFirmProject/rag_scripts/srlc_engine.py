import logging
import json

class SRLCEngine:
    """
    Self-Reflective Legal Critique (SRLC) Engine
    A breakthrough 3-step algorithm for high-fidelity legal RAG.
    """

    def __init__(self, llm):
        self.llm = llm
        self.logger = logging.getLogger(__name__)

    def run(self, query, context_str):
        # Step 1: Initial Generation
        self.logger.info("Step 1: Generating Initial Draft...")
        draft_prompt = f"""
        [PHASE 1: DRAFT]
        You are a highly detailed legal clerk. Based ON ONLY the provided context, answer the user question.

        CONTEXT:
        {context_str}

        USER QUESTION: {query}

        ANSWER:
        """
        initial_draft = str(self.llm.complete(draft_prompt))

        # Step 2: Self-Critique
        self.logger.info("Step 2: Performing Self-Critique...")
        critique_prompt = f"""
        [PHASE 2: CRITIQUE]
        You are a Senior Partner at a Law Firm reviewing a junior's work.
        Compare the DRAFT below with the SOURCE CONTEXT.
        Identify:
        1. Any claims made in the DRAFT NOT supported by the SOURCE CONTEXT (Hallucinations).
        2. Any critical legal details from the SOURCE CONTEXT missing in the DRAFT.
        3. Accuracy of citations.

        SOURCE CONTEXT:
        {context_str}

        DRAFT:
        {initial_draft}

        CRITIQUE:
        """
        critique = str(self.llm.complete(critique_prompt))

        # Step 3: Final Refinement
        self.logger.info("Step 3: Final Refinement...")
        refine_prompt = f"""
        [PHASE 3: REFINEMENT]
        Produce the final, verified legal answer.
        Incorporate the CRITIQUE to correct and improve the INITIAL DRAFT.
        Ensure every claim is grounded in the SOURCE CONTEXT.

        INITIAL DRAFT:
        {initial_draft}

        CRITIQUE:
        {critique}

        FINAL VERIFIED ANSWER:
        """
        final_answer = str(self.llm.complete(refine_prompt))

        return {
            "draft": initial_draft,
            "critique": critique,
            "final": final_answer
        }
