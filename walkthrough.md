# 🕸️ Deal Anatomy Graph: Feature Walkthrough

We have successfully built and integrated the most cutting-edge feature of Tegifa Legal: **The Deal Anatomy Graph**.

This solves one of the most painful, billable-hour-draining tasks for junior associates: cross-referencing multiple deal documents (like an NDA, an MSA, and an SOW) to ensure defined terms and obligations don't conflict.

## What Was Implemented

### 1. Zero-Config Local Graph Architecture
Instead of relying on heavy external database containers like Neo4j (which require complex Docker setups), we pivoted to a **100% local, in-memory graph architecture using `NetworkX`**. 
This guarantees the tool works instantly on any lawyer's machine with zero setup and ensures absolute data privacy.

### 2. The Semantic Extraction Engine
We created a new agent (`agents/deal_graph.py`) that uses the local LLM to perform Semantic Extraction. 
- It scans through every uploaded document.
- It identifies every capitalized **Defined Term** (e.g., "Confidential Information", "Affiliate").
- It records exactly *how* that term is defined.

### 3. Cross-Document Conflict Detection
As the system builds the graph, it looks for **"Definition Drift"**.
- If Document A defines "Services" as "software development", and Document B defines "Services" as "consulting and support", the graph detects the collision.
- The system automatically flags the conflicting node as high risk.

### 4. Interactive "Deal Room" UI
We added a brand new **"Deal Anatomy Graph"** tab to the main application interface.
- **Bulk Upload:** You can now upload multiple `.pdf` or `.docx` files at once.
- **Visual Mapping:** We integrated `streamlit-agraph` to render a beautiful, interactive web of your deal. 
  - **Blue Hexagons** represent Documents.
  - **Green Dots** represent correctly mapped Defined Terms.
  - **Red Dots** represent critical Definition Conflicts (loopholes).
- **Conflict Report:** A dedicated UI section expands to show the exact text of the conflicting definitions side-by-side so the lawyer can immediately draft an amendment.

> [!TIP]
> To test this, go to your browser at `http://localhost:8501`, log in, navigate to the third tab ("Deal Anatomy Graph"), and upload two dummy contracts that define the same term differently.

## Next Steps for Future Expansion
While the current version focuses on **Definitions**, the graph architecture we built is highly extensible. 
In the future, we can add:
- **Obligation Deadlocks:** Graphing deadlines to ensure Document A's 5-day SLA doesn't conflict with Document B's 10-day review period.
- **Shadow Redlining:** Automatically generating a `.docx` file with Track Changes that rewrites the conflicting clause to fix the loophole.
