# OmniBrain Studio
### *Agentic Multi-Modal RAG Orchestrator*

OmniBrain Studio is an enterprise-grade multi-modal Retrieval-Augmented Generation (RAG) platform. It orchestrates specialized agents across **dense semantic text search (ChromaDB)**, **quantitative financial databases (Text-to-SQL)**, and **visual chart transcription (Vision VLM)** with **NeMo Guardrail factual claim auditing**.

---

## ?? Contributors
- **Udipta Ray**
- **Kalash Jain**
- **Vansh Teotia**
- **Aaryan Rathi**
- **Tushar Kumar Pradhan**
- **Rajeshwari Chetlapalli**

---

## ??? Project Structure

```text
omnibrain-multimodal-rag/
??? app/                           # Core application package & CLI
??? backend/                       # FastAPI backend server
?   ??? agents/                    # Multi-agent orchestrators (Supervisor, Search, Vision, SQL)
?   ??? rag/                       # Document parser, multi-scale embeddings, ChromaDB vector store
?   ??? guardrails/                # NeMo / LLM-as-a-Judge factual grounding auditor
?   ??? db/                        # SQLite financial database & seeders
?   ??? models/                    # Pydantic schemas & response models
??? chroma_db/                     # Persistent ChromaDB vector storage
??? config/                        # Configuration settings, config.yaml, & system prompts
??? data/                          # Uploaded documents & benchmark filings
??? docs/                          # Architecture diagrams & API documentation
??? frontend/                      # Web UI (Space Grotesk typography, landing page, chat studio)
??? notebooks/                     # Interactive Jupyter walkthroughs
??? tests/                         # Pytest test suite
??? .env.example                   # Environment configuration template
??? .gitignore                     # Git ignore rules (safeguards secrets & cache)
??? conftest.py                    # Pytest configuration
??? requirements.txt               # Dependencies
??? main.py                        # Root launcher script
```

---

## ?? Quickstart Guide

### 1. Installation
```bash
git clone <repo-url>
cd omnibrain-multimodal-rag
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python main.py
```
Open **`http://localhost:8000`** in your browser.

---

## ?? Automated Tests
```bash
pytest
```
