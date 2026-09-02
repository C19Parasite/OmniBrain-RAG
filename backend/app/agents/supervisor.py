import re
import json
import time
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional
from ..config import settings
from .state import SupervisorState, SubTask
from .sql_agent import TextToSQLAgent as SQLAgent
from .search_agent import SearchAgent
from .vision_agent import VisionAgent
from ..guardrails.evaluator import GuardrailEvaluator

class SupervisorOrchestrator:
    """
    LangGraph State Machine Supervisor Agent for OmniBrain.
    Coordinates Multi-Agent RAG execution:
    1. Evaluates incoming user query & decomposes into specialized sub-tasks.
    2. Routes sub-tasks across Text-to-SQL, Dense Semantic Search (ChromaDB), and Vision VLM agents.
    3. Synthesizes multimodal context into an institutional research memorandum with strict citations.
    4. Audits factual grounding and sentence-level hallucination claims (NeMo / LLM-as-Judge).
    """

    def __init__(
        self,
        sql_agent: Optional[SQLAgent] = None,
        search_agent: Optional[SearchAgent] = None,
        vision_agent: Optional[VisionAgent] = None,
        evaluator: Optional[GuardrailEvaluator] = None
    ):
        self.sql_agent = sql_agent or SQLAgent()
        self.search_agent = search_agent or SearchAgent()
        self.vision_agent = vision_agent or VisionAgent()
        self.evaluator = evaluator or GuardrailEvaluator()

    def process_query(self, query: str, top_k: Optional[int] = None, temperature: Optional[float] = None, document_ids: Optional[List[str]] = None, gemini_api_key: Optional[str] = None, openai_api_key: Optional[str] = None) -> SupervisorState:
        """
        Executes end-to-end multi-agent LangGraph workflow.
        """
        start_time = time.time()
        state = SupervisorState(query=query)

        # -------------------------------------------------------------
        # CHECK: NO DOCUMENT ATTACHED
        # -------------------------------------------------------------
        if document_ids is not None and len(document_ids) == 0:
            state.add_trace(
                event_type="thought",
                agent="Supervisor",
                content="No document context is attached to this chat session. Halting execution to prevent ungrounded hallucination."
            )
            state.synthesized_memo = (
                "# ⚠️ No Document Context Attached\n\n"
                "**Notice**: There is currently no document to refer from in this chat session.\n\n"
                "To ask questions and analyze filings, please upload or attach a corporate financial filing (PDF/Markdown) "
                "or visual exhibit chart (PNG/JPG) using the sidebar **Project Knowledge** drawer or the **📎 Upload Document** button."
            )
            state.execution_time_seconds = 0.01
            state.guardrail_report = {
                "overall_score": 1.0,
                "status": "PASSED",
                "total_claims": 0,
                "grounded_claims": 0,
                "ungrounded_claims": 0,
                "claim_verdicts": [],
                "citations": []
            }
            state.citations = []
            return state

        # -------------------------------------------------------------
        # STATE 1: SUPERVISOR QUERY DECOMPOSITION & PLANNING
        # -------------------------------------------------------------
        state.add_trace(
            event_type="thought",
            agent="Supervisor",
            content=f"Received query: '{query}'. Evaluating intent, decomposing sub-tasks, and determining agent routing."
        )

        sub_tasks = self._decompose_query(query)
        state.sub_tasks = sub_tasks

        task_descriptions = [f"[{t.target_agent}] {t.description}" for t in sub_tasks]
        state.add_trace(
            event_type="plan",
            agent="Supervisor",
            content=f"Decomposed query into {len(sub_tasks)} specialized sub-task(s): " + " | ".join(task_descriptions),
            metadata={"sub_tasks_count": len(sub_tasks)}
        )

        # -------------------------------------------------------------
        # STATE 2: AGENT EXECUTION & ROUTING
        # -------------------------------------------------------------
        for task in sub_tasks:
            if task.target_agent == "SQLAgent":
                state.add_trace(
                    event_type="action",
                    agent="Supervisor",
                    content=f"Routing sub-task '{task.description}' to SQLAgent.",
                    metadata={"target_agent": "SQLAgent", "task_id": task.id}
                )

                sql_response = self.sql_agent.run(task.description)
                state.sql_results.append(sql_response)
                task.status = "completed"
                task.result_summary = f"Executed SQL: {sql_response.get('executed_sql')}, returned {len(sql_response.get('rows', []))} rows"

                state.add_trace(
                    event_type="result",
                    agent="SQLAgent",
                    content=f"SQL Query executed: `{sql_response.get('executed_sql')}`. Retrieved {len(sql_response.get('rows', []))} row(s) from financial database.",
                    metadata={"sql": sql_response.get("executed_sql"), "row_count": len(sql_response.get("rows", []))}
                )

            elif task.target_agent == "SearchAgent":
                state.add_trace(
                    event_type="action",
                    agent="Supervisor",
                    content=f"Routing query to SearchAgent: '{task.description}'.",
                    metadata={"target_agent": "SearchAgent", "task_id": task.id}
                )

                state.add_trace(
                    event_type="tool",
                    agent="SearchAgent",
                    content=f"Querying ChromaDB vector store for multimodal chunks matching: '{task.description}'"
                )

                k = top_k or settings.TOP_K
                search_response = self.search_agent.search(
                    task.description,
                    top_k=k,
                    doc_ids=document_ids
                )
                state.search_results.extend(search_response)

                visual_count = sum(1 for c in search_response if c.get("chunk_type") == "visual")
                text_count = len(search_response) - visual_count
                task.status = "completed"
                task.result_summary = f"{len(search_response)} chunks ({text_count} text, {visual_count} visual)"

                state.add_trace(
                    event_type="result",
                    agent="SearchAgent",
                    content=f"Retrieved {len(search_response)} multimodal chunk(s) from knowledge base ({text_count} text, {visual_count} visual exhibits).",
                    metadata={"chunks_count": len(search_response), "visual_chunks": visual_count}
                )

        # -------------------------------------------------------------
        # STATE 3: SYNTHESIS & INLINE CITATION GROUNDING
        # -------------------------------------------------------------
        state.add_trace(
            event_type="thought",
            agent="Synthesizer",
            content="Aggregating retrieved document passages, visual exhibits, and SQL metrics. Composing research memorandum."
        )

        draft_memo = self._synthesize_memo(state, gemini_api_key=gemini_api_key, openai_api_key=openai_api_key)

        # -------------------------------------------------------------
        # STATE 4: GUARDRAIL & HALLUCINATION EVALUATION (LLM-as-Judge)
        # -------------------------------------------------------------
        state.add_trace(
            event_type="thought",
            agent="GuardrailEvaluator",
            content="Executing sentence-level factual grounding check against retrieved multimodal and SQL evidence."
        )

        eval_report = self.evaluator.evaluate_memo(
            memo_markdown=draft_memo,
            sql_results=state.sql_results,
            search_results=state.search_results
        )

        state.synthesized_memo = eval_report["annotated_memo"]
        state.guardrail_report = eval_report
        state.citations = eval_report["citations"]

        state.add_trace(
            event_type="result",
            agent="GuardrailEvaluator",
            content=f"Audit complete. Grounding Score: {int(eval_report['overall_score']*100)}% ({eval_report['status']}). Grounded Claims: {eval_report['grounded_claims']}/{eval_report['total_claims']}, Ungrounded Claims: {eval_report['ungrounded_claims']}.",
            metadata={
                "overall_score": eval_report["overall_score"],
                "status": eval_report["status"],
                "ungrounded_count": eval_report["ungrounded_claims"]
            }
        )

        total_time = round(time.time() - start_time, 2)
        state.execution_time_seconds = total_time

        state.add_trace(
            event_type="state_update",
            agent="Supervisor",
            content=f"End-to-end orchestration complete in {total_time}s with {len(state.citations)} verified citations.",
            metadata={"execution_time_s": total_time}
        )

        return state

    def _decompose_query(self, query: str) -> List[SubTask]:
        """
        Decomposes query into sub-tasks based on intent:
        1. Always creates a primary semantic search task with the exact user query.
        2. Adds SQL task if the user asks for quantitative database metrics (e.g. quarterly revenue of ticker).
        """
        q_lower = query.lower()
        sub_tasks: List[SubTask] = []
        task_idx = 1

        # Check if query is explicitly asking for SQL database records of known tickers
        detected_tickers = []
        if "nvda" in q_lower or "nvidia" in q_lower:
            detected_tickers.append("NVDA")
        if "msft" in q_lower or "microsoft" in q_lower:
            detected_tickers.append("MSFT")
        if "aapl" in q_lower or "apple" in q_lower:
            detected_tickers.append("AAPL")
        if "tsla" in q_lower or "tesla" in q_lower:
            detected_tickers.append("TSLA")

        needs_sql = len(detected_tickers) > 0 and any(term in q_lower for term in [
            "quarterly", "revenue growth", "gross margin", "net income", "eps", "database", "sql", "compare nvda"
        ])

        if needs_sql:
            for ticker in detected_tickers:
                sub_tasks.append(SubTask(
                    id=f"task_{task_idx}",
                    description=f"Query {ticker} recent quarterly financials from database",
                    target_agent="SQLAgent"
                ))
                task_idx += 1

        # Primary Search / Multimodal Task ALWAYS uses the user's actual question!
        sub_tasks.append(SubTask(
            id=f"task_{task_idx}",
            description=query,
            target_agent="SearchAgent"
        ))

        return sub_tasks

    def _synthesize_memo(self, state: SupervisorState, gemini_api_key: Optional[str] = None, openai_api_key: Optional[str] = None) -> str:
        """
        Synthesizes collected multimodal data into direct Markdown answer.
        Uses frontier LLM (Gemini / OpenAI) if keys available, else falls back to extractive synthesizer.
        """
        gkey = gemini_api_key or settings.GEMINI_API_KEY
        okey = openai_api_key or settings.OPENAI_API_KEY

        if gkey:
            try:
                return self._synthesize_llm_gemini(state, key=gkey)
            except Exception as e:
                print(f"[Synthesizer] Gemini call failed: {e}, falling back.")
        
        if okey:
            try:
                return self._synthesize_llm_openai(state, key=okey)
            except Exception as e:
                print(f"[Synthesizer] OpenAI call failed: {e}, falling back.")

        return self._extractive_synthesizer(state)

    def _synthesize_llm_gemini(self, state: SupervisorState, key: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}"
        context_prompt = self._build_context_prompt(state)
        system_prompt = """You are an intelligent multimodal research assistant.
Answer the user's question directly, clearly, comprehensively and naturally based ONLY on the provided retrieved context.
CRITICAL RULES:
- Do NOT output any boilerplate memo titles, 'Institutional Research Memorandum', 'Prepared by', 'Query:', or repetitive headers.
- Directly answer the question in natural paragraphs or bullet points with full detail.
- Include inline citations for every factual statement:
  * For text reports/documents: [Source: document_name, p.X]
  * For visual charts/tables: [Visual: image_name, p.X]
  * For database metrics: [SQL: table_name]"""

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": system_prompt},
                        {"text": f"User Question: {state.query}\n\nRetrieved Context:\n{context_prompt}\n\nDirect Comprehensive Answer:"}
                    ]
                }
            ],
            "generationConfig": {"temperature": settings.TEMPERATURE, "maxOutputTokens": 1200}
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()

    def _synthesize_llm_openai(self, state: SupervisorState, key: str) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        context_prompt = self._build_context_prompt(state)
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a multimodal research assistant. Answer the user question directly and thoroughly without any memo headers or query repeats. Include inline citations [Source: doc, p.X], [Visual: img, p.X], or [SQL: table]."},
                {"role": "user", "content": f"User Question: {state.query}\n\nContext:\n{context_prompt}"}
            ],
            "temperature": settings.TEMPERATURE
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()

    def _build_context_prompt(self, state: SupervisorState) -> str:
        ctx = ""
        if state.sql_results:
            ctx += "=== STRUCTURED FINANCIAL DATABASE RESULTS (SQL AGENT) ===\n"
            for res in state.sql_results:
                if res.get("is_valid") and res.get("rows"):
                    ctx += f"Query: {res.get('executed_sql')} [SQL: quarterly_financials]\nRows: {res.get('rows')}\n\n"

        ctx += "=== MULTIMODAL RETRIEVED CHUNKS (TEXT & VISUAL VLM) ===\n"
        for chunk in state.search_results:
            tag = "[Visual]" if chunk.get("chunk_type") == "visual" else "[Text]"
            ctx += f"{tag} [Source: {chunk.get('source_document')}, p.{chunk.get('page_number')}] ({chunk.get('section_title')}):\n{chunk.get('text')}\n\n"
        return ctx

    def _extractive_synthesizer(self, state: SupervisorState) -> str:
        """
        High-precision direct answer synthesis that directly addresses the user's query
        using the actual retrieved text passages and visual exhibits without boilerplate headers.
        """
        query = state.query
        chunks = state.search_results
        sql_res = state.sql_results

        if not chunks and not sql_res:
            return "No matching context found in the attached documents. Please ensure relevant documents or charts are attached to this chat session."

        text_chunks = [c for c in chunks if c.get("chunk_type") != "visual"]
        visual_chunks = [c for c in chunks if c.get("chunk_type") == "visual"]

        lines = []

        # Direct Answer bullet points
        if text_chunks:
            for idx, c in enumerate(text_chunks[:4]):
                doc_name = c.get("source_document", "document")
                page_num = c.get("page_number", 1)
                citation = f"[Source: {doc_name}, p.{page_num}]"
                text = c.get("text", "").strip()

                sentences = re.split(r'(?<=[.!?])\s+', text)
                summary_snippet = " ".join(sentences[:3]) if sentences else text
                lines.append(f"• {summary_snippet} {citation}")
                lines.append("")

        # Visual Exhibit Analysis (if present)
        if visual_chunks:
            for v_idx, vc in enumerate(visual_chunks[:2]):
                v_doc = vc.get("source_document", "chart.png")
                v_page = vc.get("page_number", 1)
                v_cite = f"[Visual: {v_doc}, p.{v_page}]"
                lines.append(f"**Visual Exhibit ({vc.get('section_title', 'Chart')})** {v_cite}:")
                lines.append(f"{vc.get('text', '').strip()} {v_cite}")
                lines.append("")

        # Structured Database Metrics (if SQL rows present)
        has_valid_sql = any(r.get("is_valid") and r.get("rows") for r in sql_res)
        if has_valid_sql:
            for sr in sql_res:
                if sr.get("is_valid") and sr.get("rows"):
                    cols = sr.get("columns", [])
                    rows = sr.get("rows", [])
                    lines.append(f"**Database Query**: `{sr.get('executed_sql')}` `[SQL: quarterly_financials]`")
                    lines.append("")
                    if cols and rows:
                        lines.append("| " + " | ".join(cols) + " | Citation |")
                        lines.append("| " + " | ".join([":---"] * len(cols)) + " | :--- |")
                        for row in rows[:5]:
                            row_vals = [str(x) for x in row]
                            lines.append("| " + " | ".join(row_vals) + " | `[SQL: quarterly_financials]` |")
                        lines.append("")

        # Primary Quotation / Excerpt if available
        if len(text_chunks) > 1:
            best_chunk = text_chunks[0]
            doc_name = best_chunk.get("source_document", "document")
            page_num = best_chunk.get("page_number", 1)
            citation = f"[Source: {doc_name}, p.{page_num}]"
            lines.append(f"> \"{best_chunk.get('text', '').strip()}\" ({citation})")
            lines.append("")

        return "\n".join(lines).strip()
