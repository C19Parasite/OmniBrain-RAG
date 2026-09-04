# OmniBrain Studio Architecture

## System Overview
- **Supervisor Orchestrator**: LangGraph state machine routing sub-tasks
- **Multi-Modal Retrieval**: Dense semantic vector embeddings in ChromaDB + Vision VLM image exhibit parsing
- **Structured Financials**: Text-to-SQL agent querying SQLite database
- **Guardrails**: NeMo / LLM-as-a-Judge claim-level verification
