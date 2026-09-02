# OmniBrain Studio

**Agentic Multi-Modal RAG Orchestrator for Institutional Financial Analysis**

OmniBrain dynamically routes complex financial queries across a structured financial database (Text-to-SQL), unstructured corporate filings (Dense Semantic Vector RAG), and visual financial charts/tables (Vision-Language Models), followed by LLM-as-Judge hallucination guardrails.

---

## 🌟 Studio Features

1. **Live Agent Graph Stream**: Real-time visual state machine graph showing query decomposition, multi-agent dispatch, tool invocations, and grounding audit.
2. **Multimodal Document Center**: Drag-and-drop ingestion of financial disclosures (PDF, Markdown, plain text) and graphic balance sheet/market segment figures (PNG, JPG, SVG).
3. **SQL Sandbox & Schema Explorer**: Interactive SQLite schema browser and ad-hoc query runner protected by strict read-only validation.
4. **Memo Studio**: Institutional investment memos with clickable inline citations (`[SQL]`, `[Visual]`, `[Text]`) opening source evidence drawers and VLM chart previews.
5. **Guardrail Scorecard**: Sentence-level LLM-as-judge factual verification with inline callouts on ungrounded claims.
6. **Settings Panel**: Custom configuration for retrieval top-$k$, model temperature, LLM provider selection, and demo data resets.

---

## 🚀 Quick Start

### 1. Requirements & Setup
```bash
pip install -r requirements.txt
```

### 2. Launch OmniBrain Studio
```bash
python run.py
```
Open your browser and navigate to: **`http://localhost:8000`**

---

## 🏛️ Architecture & Modules

- **`backend/app/agents/supervisor.py`**: Explicit state-machine coordinator managing planning, routing, synthesis, and guardrail auditing.
- **`backend/app/agents/sql_agent.py`**: Safe Text-to-SQL generator executing over URI read-only SQLite connections.
- **`backend/app/agents/vision_agent.py`**: VLM chart/table interpreter transforming visual financial exhibits into searchable text embeddings.
- **`backend/app/agents/search_agent.py`**: Multimodal cosine similarity retriever backed by persistent ChromaDB.
- **`backend/app/guardrails/evaluator.py`**: LLM-as-Judge factual grounding auditor verifying claims against primary evidence.
- **`backend/app/db/database.py`**: Financial database schema manager (`companies`, `stock_history`, `quarterly_financials`, `analyst_estimates`).
