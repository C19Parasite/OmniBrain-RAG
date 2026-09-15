import re
import json
import time
import sqlite3
import logging
from uuid import uuid4
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional, Tuple, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:  # Allows a clear development fallback before dependencies are installed.
    SqliteSaver = None
from ..config import settings
from .state import SupervisorState, SubTask
from .sql_agent import TextToSQLAgent as SQLAgent
from .search_agent import SearchAgent
from .vision_agent import VisionAgent
from ..guardrails.evaluator import GuardrailEvaluator
from ..observability.langfuse import observe, update as update_observation, flush as flush_langfuse

logger = logging.getLogger(__name__)


class GeminiAPIError(RuntimeError):
    """Gemini failure with a safe, actionable HTTP diagnostic."""


class _ConversationGraphState(TypedDict, total=False):
    """Persisted graph channels. API keys are intentionally excluded."""
    query: str
    top_k: Optional[int]
    temperature: Optional[float]
    document_ids: Optional[List[str]]
    retrieval_mode: Optional[str]
    conversation_history: List[Dict[str, Any]]
    result: Dict[str, Any]


class _RequestContext(TypedDict, total=False):
    """Per-request values that LangGraph does not checkpoint."""
    gemini_api_key: Optional[str]
    openai_api_key: Optional[str]


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
        self.checkpointer = self._create_checkpointer()
        self.graph = self._build_graph()

    def _create_checkpointer(self):
        """Create durable thread memory, with an in-process fallback for local dev."""
        if SqliteSaver is not None:
            settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
            # Keep this connection open for the lifespan of the FastAPI process.
            # check_same_thread=False is required because sync FastAPI work may
            # execute in different worker threads.
            self._checkpoint_connection = sqlite3.connect(
                str(settings.DATA_DIR / "langgraph_checkpoints.sqlite"),
                check_same_thread=False,
            )
            return SqliteSaver(self._checkpoint_connection)
        print("[OmniBrain] langgraph-checkpoint-sqlite is not installed; using non-persistent memory.")
        return InMemorySaver()

    def _build_graph(self):
        """Compile a checkpointed LangGraph workflow for a single chat thread."""
        builder = StateGraph(_ConversationGraphState, context_schema=_RequestContext)
        builder.add_node("research", self._run_research_turn)
        builder.add_edge(START, "research")
        builder.add_edge("research", END)
        return builder.compile(checkpointer=self.checkpointer)

    def process_query(self, query: str, top_k: Optional[int] = None, temperature: Optional[float] = None, document_ids: Optional[List[str]] = None, gemini_api_key: Optional[str] = None, openai_api_key: Optional[str] = None, thread_id: Optional[str] = None, retrieval_mode: Optional[str] = "dense") -> SupervisorState:
        """Run a query in a checkpointed LangGraph conversation thread."""
        # Legacy clients that do not send a thread ID must not accidentally
        # share memory with another request.
        thread_id = thread_id or str(uuid4())
        mode = retrieval_mode or "dense"
        with observe(
            "OmniBrain Supervisor",
            "agent",
            input={"query": query, "document_ids": document_ids, "retrieval_mode": mode},
            metadata={"thread_id": thread_id, "environment": settings.LANGFUSE_ENVIRONMENT},
        ) as root_observation:
            graph_state = self.graph.invoke(
                {
                    "query": query,
                    "top_k": top_k,
                    "temperature": temperature,
                    "document_ids": document_ids,
                    "retrieval_mode": mode,
                },
                {"configurable": {"thread_id": thread_id}},
                context={"gemini_api_key": gemini_api_key, "openai_api_key": openai_api_key},
            )
            result = SupervisorState.model_validate(graph_state["result"])
            update_observation(
                root_observation,
                output={"memo_characters": len(result.synthesized_memo or "")},
                metadata={
                    "thread_id": thread_id,
                    "environment": settings.LANGFUSE_ENVIRONMENT,
                    "grounding_score": result.guardrail_report.get("overall_score"),
                    "grounding_status": result.guardrail_report.get("status"),
                    "citation_count": len(result.citations),
                    "retrieval_mode": result.retrieval_mode,
                },
            )
        flush_langfuse()
        return result

    def _run_research_turn(self, graph_state: "_ConversationGraphState", runtime: Runtime[_RequestContext]) -> Dict[str, Any]:
        """Use the previous checkpoint only as context, never as uncited evidence."""
        history = list(graph_state.get("conversation_history", []))[-6:]
        state = self._process_query_once(
            query=graph_state["query"],
            top_k=graph_state.get("top_k"),
            temperature=graph_state.get("temperature"),
            document_ids=graph_state.get("document_ids"),
            gemini_api_key=(runtime.context or {}).get("gemini_api_key"),
            openai_api_key=(runtime.context or {}).get("openai_api_key"),
            conversation_history=history,
            retrieval_mode=graph_state.get("retrieval_mode", "dense") or "dense",
        )
        turn = {
            "user_query": state.query,
            "resolved_query": state.resolved_query,
            "assistant_memo": (state.synthesized_memo or "")[:4000],
            "document_ids": graph_state.get("document_ids"),
            "citations": [
                {key: citation.get(key) for key in ("source_type", "source_name", "page_number")}
                for citation in state.citations
            ],
        }
        updated_history = (history + [turn])[-6:]
        state.conversation_history = updated_history
        return {
            "result": state.model_dump(mode="json"),
            "conversation_history": updated_history,
        }

    def _process_query_once(self, query: str, top_k: Optional[int] = None, temperature: Optional[float] = None, document_ids: Optional[List[str]] = None, gemini_api_key: Optional[str] = None, openai_api_key: Optional[str] = None, conversation_history: Optional[List[Dict[str, Any]]] = None, retrieval_mode: str = "dense") -> SupervisorState:
        """
        Executes end-to-end multi-agent LangGraph workflow.
        """
        start_time = time.time()
        state = SupervisorState(
            query=query,
            conversation_history=conversation_history or [],
            retrieval_mode=retrieval_mode
        )

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
        # STATE 0: MULTI-TURN CONTEXTUAL CO-REFERENCE RESOLUTION
        # -------------------------------------------------------------
        resolved_query, was_resolved = self._resolve_conversational_query(
            query=query,
            history=state.conversation_history,
            gemini_api_key=gemini_api_key,
            openai_api_key=openai_api_key
        )
        if was_resolved:
            state.resolved_query = resolved_query
            state.add_trace(
                event_type="thought",
                agent="Supervisor",
                content=f"Multi-Turn Memory: Resolved follow-up query to: '{resolved_query}' using conversational context.",
                metadata={"original_query": query, "resolved_query": resolved_query, "history_turns": len(state.conversation_history)}
            )
            effective_query = resolved_query
        else:
            effective_query = query

        # -------------------------------------------------------------
        # STATE 1: SUPERVISOR QUERY DECOMPOSITION & PLANNING
        # -------------------------------------------------------------
        state.add_trace(
            event_type="thought",
            agent="Supervisor",
            content=f"Received query: '{effective_query}'. Evaluating intent, decomposing sub-tasks, and determining agent routing."
        )

        sub_tasks = self._decompose_query(effective_query)
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

                with observe("Text-to-SQL", "tool", input={"task": task.description}) as sql_observation:
                    sql_response = self.sql_agent.run(task.description)
                    update_observation(sql_observation, output={
                        "is_valid": sql_response.get("is_valid"),
                        "row_count": len(sql_response.get("rows", [])),
                        "executed_sql": sql_response.get("executed_sql"),
                    })
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
                    content=f"Querying knowledge base ({retrieval_mode.upper()} Search: ChromaDB + BM25 RRF) for: '{task.description}'"
                )

                k = top_k or settings.TOP_K
                with observe("Vector retrieval", "retriever", input={"query": task.description, "top_k": k, "retrieval_mode": retrieval_mode}) as retrieval_observation:
                    search_response = self.search_agent.search(
                        task.description,
                        top_k=k,
                        doc_ids=document_ids,
                        mode=retrieval_mode
                    )
                    update_observation(retrieval_observation, output={
                        "chunk_count": len(search_response),
                        "document_ids": document_ids,
                    })
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
        # STATE 2.5: SELF-CORRECTION RETRIEVAL LOOP (Self-RAG / CRAG)
        # -------------------------------------------------------------
        has_search_task = any(t.target_agent == "SearchAgent" for t in sub_tasks)
        if has_search_task:
            needs_correction, reason = self._evaluate_retrieval_quality(effective_query, state.search_results)
            if needs_correction:
                state.add_trace(
                    event_type="thought",
                    agent="Supervisor",
                    content=f"Self-RAG Evaluator: {reason} Triggering autonomous query reformulation.",
                    metadata={"reason": reason, "initial_chunks": len(state.search_results)}
                )

                rewritten_query = self._reformulate_query(
                    query=effective_query,
                    reason=reason,
                    gemini_api_key=gemini_api_key,
                    openai_api_key=openai_api_key
                )

                state.add_trace(
                    event_type="action",
                    agent="Supervisor",
                    content=f"Self-RAG: Reformulated search query: '{rewritten_query}'. Re-querying knowledge base.",
                    metadata={"original_query": effective_query, "rewritten_query": rewritten_query}
                )

                k = top_k or settings.TOP_K
                retry_results = self.search_agent.search(
                    rewritten_query,
                    top_k=k,
                    doc_ids=document_ids,
                    mode=retrieval_mode
                )

                # Merge and deduplicate chunks by id
                existing_by_id = {c["id"]: c for c in state.search_results if "id" in c}
                initial_count = len(state.search_results)
                for c in retry_results:
                    cid = c.get("id")
                    if not cid or cid not in existing_by_id:
                        if cid:
                            existing_by_id[cid] = c
                        else:
                            state.search_results.append(c)
                    else:
                        if c.get("similarity_score", 0.0) > existing_by_id[cid].get("similarity_score", 0.0):
                            existing_by_id[cid] = c

                merged_chunks = list(existing_by_id.values())
                merged_chunks.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
                state.search_results = merged_chunks[:max(k, 6)]

                new_max_sim = max([c.get("similarity_score", 0.0) for c in state.search_results], default=0.0)
                state.self_correction = {
                    "triggered": True,
                    "original_query": query,
                    "rewritten_query": rewritten_query,
                    "initial_count": initial_count,
                    "final_count": len(state.search_results),
                    "reason": reason,
                    "max_similarity": new_max_sim
                }

                state.add_trace(
                    event_type="result",
                    agent="Supervisor",
                    content=f"Self-RAG: Re-retrieval completed with {len(retry_results)} additional chunk(s) (top similarity {new_max_sim:.2f}). Query successfully self-corrected.",
                    metadata=state.self_correction
                )
            else:
                state.self_correction = {
                    "triggered": False,
                    "reason": "Retrieval passed confidence and relevance checks."
                }

        # -------------------------------------------------------------
        # STATE 3: SYNTHESIS & INLINE CITATION GROUNDING
        # -------------------------------------------------------------
        state.add_trace(
            event_type="thought",
            agent="Synthesizer",
            content="Aggregating retrieved document passages, visual exhibits, and SQL metrics. Composing research memorandum."
        )

        with observe("Memo synthesis", "generation", input={"query": state.query, "evidence_chunks": len(state.search_results), "sql_results": len(state.sql_results)}) as synthesis_observation:
            draft_memo = self._synthesize_memo(state, gemini_api_key=gemini_api_key, openai_api_key=openai_api_key, observation=synthesis_observation)
            update_observation(synthesis_observation, output={"memo_characters": len(draft_memo)})

        # -------------------------------------------------------------
        # STATE 4: GUARDRAIL & HALLUCINATION EVALUATION (LLM-as-Judge)
        # -------------------------------------------------------------
        state.add_trace(
            event_type="thought",
            agent="GuardrailEvaluator",
            content="Executing sentence-level factual grounding check against retrieved multimodal and SQL evidence."
        )

        with observe("Grounding and hallucination guardrail", "guardrail", input={"memo_characters": len(draft_memo), "evidence_chunks": len(state.search_results)}) as guardrail_observation:
            eval_report = self.evaluator.evaluate_memo(
                memo_markdown=draft_memo,
                sql_results=state.sql_results,
                search_results=state.search_results
            )
            update_observation(guardrail_observation, output={
                "overall_score": eval_report["overall_score"],
                "status": eval_report["status"],
                "grounded_claims": eval_report["grounded_claims"],
                "ungrounded_claims": eval_report["ungrounded_claims"],
            })

        state.synthesized_memo = eval_report["annotated_memo"]
        state.guardrail_report = eval_report
        state.citations = eval_report["citations"]

        if settings.BLOCK_UNGROUNDED_MEMOS and eval_report["overall_score"] < settings.MIN_GROUNDING_SCORE:
            state.synthesized_memo = (
                "# Grounding review required\n\n"
                f"This memo was withheld because its grounding score ({eval_report['overall_score']:.0%}) "
                f"is below the configured minimum ({settings.MIN_GROUNDING_SCORE:.0%}). "
                "Review the cited evidence, refine the question, or attach the relevant filing."
            )

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

    def _resolve_conversational_query(
        self,
        query: str,
        history: List[Dict[str, Any]],
        gemini_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None
    ) -> Tuple[str, bool]:
        """
        Resolves follow-up queries using conversational memory.
        Detects pronouns, ellipsis, or missing subjects from prior turns.
        Returns (resolved_query, was_resolved).
        """
        if not history or not query.strip():
            return query, False

        q_clean = query.strip()
        q_lower = q_clean.lower()

        # Follow-up indicators
        has_pronouns = bool(re.search(r'\b(they|their|theirs|them|it|its|this|that|these|those)\b', q_lower))
        has_ellipsis = bool(re.search(r'^(what about|how about|compare with|compare to|and for|why did it|did it|what of)\b', q_lower))

        known_companies = ["nvidia", "nvda", "microsoft", "msft", "apple", "aapl", "tesla", "tsla", "amazon", "amzn", "google", "googl"]
        query_has_company = any(c in q_lower for c in known_companies)

        prior_companies = []
        for turn in reversed(history[-4:]):
            pq = turn.get("user_query", "")
            if pq:
                for comp in ["NVIDIA", "Microsoft", "Apple", "Tesla", "Amazon", "Alphabet"]:
                    if comp.lower() in pq.lower() and comp not in prior_companies:
                        prior_companies.append(comp)
                for tick in ["NVDA", "MSFT", "AAPL", "TSLA", "AMZN", "GOOGL"]:
                    if tick.lower() in pq.lower() and tick not in prior_companies:
                        prior_companies.append(tick)

        asks_metric = bool(re.search(r'\b(revenue|sales|gross margin|margin|net income|earnings|eps|guidance|capex|growth|data center|cloud)\b', q_lower))
        is_definitional_or_general = bool(re.search(r'\b(difference between|what is\b|what are\b|define\b|explain the concept|how does a\b|how do\b)', q_lower))

        is_follow_up = (
            has_pronouns or 
            has_ellipsis or 
            (not query_has_company and len(prior_companies) > 0 and asks_metric and not is_definitional_or_general)
        )
        if not is_follow_up:
            return query, False

        # Attempt frontier LLM resolution
        gkey = gemini_api_key or settings.GEMINI_API_KEY
        okey = openai_api_key or settings.OPENAI_API_KEY

        if gkey:
            try:
                resolved = self._resolve_llm_gemini(query, history[-3:], gkey)
                if resolved and resolved.lower() != q_lower:
                    return resolved, True
            except Exception as e:
                print(f"[Memory] Gemini co-reference resolution failed: {e}")

        if okey:
            try:
                resolved = self._resolve_llm_openai(query, history[-3:], okey)
                if resolved and resolved.lower() != q_lower:
                    return resolved, True
            except Exception as e:
                print(f"[Memory] OpenAI co-reference resolution failed: {e}")

        resolved = self._heuristic_coreference_resolution(query, history[-3:], prior_companies)
        return (resolved, True) if resolved != query else (query, False)

    def _resolve_llm_gemini(self, query: str, recent_turns: List[Dict[str, Any]], key: str) -> str:
        model = getattr(settings, "GEMINI_MODEL", "gemini-3.5-flash-lite")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        turns_summary = "\n".join([f"Analyst: {t.get('user_query', '')}" for t in recent_turns])
        prompt = f"""You are a financial query resolution agent.
Given the previous dialogue turns between an analyst and financial assistant:
{turns_summary}

Rewrite the following follow-up question into a single, complete, standalone financial search query:
Follow-up: "{query}"

Rules:
- Resolve all pronouns ('their', 'its', 'they', 'it') to the target company or metric.
- Resolve ellipsis ('what about...', 'compare to...') by incorporating the relevant metric from earlier turns.
- Return ONLY the rewritten query text without quotes or explanation."""
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 200}
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            rewritten = "".join([p.get("text", "") for p in parts if "text" in p]).strip().strip('"\'')
            return rewritten if rewritten else query

    def _resolve_llm_openai(self, query: str, recent_turns: List[Dict[str, Any]], key: str) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        turns_summary = "\n".join([f"Analyst: {t.get('user_query', '')}" for t in recent_turns])
        prompt = f"""Given the previous dialogue turns:
{turns_summary}

Rewrite the following follow-up question into a single, complete, standalone financial search query:
Follow-up: "{query}"

Rules:
- Resolve all pronouns ('their', 'its', 'they', 'it') to the target company or metric.
- Return ONLY the rewritten query text without quotes or explanation."""
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a financial query co-reference resolution agent. Return ONLY the rewritten standalone query."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 60
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            rewritten = data["choices"][0]["message"]["content"].strip().strip('"\'')
            return rewritten if rewritten else query

    def _heuristic_coreference_resolution(
        self,
        query: str,
        history: List[Dict[str, Any]],
        prior_companies: List[str]
    ) -> str:
        clean_q = query.strip()
        last_turn = history[-1] if history else {}
        last_query = last_turn.get("user_query", "")

        subject_company = prior_companies[0] if prior_companies else ""

        # 1. Ellipsis "What about [Entity]?"
        ellipsis_match = re.match(r'^(what about|how about|and what about)\s+([a-zA-Z0-9\s]+)\??$', clean_q, re.IGNORECASE)
        if ellipsis_match:
            target_entity = ellipsis_match.group(2).strip()
            metrics = []
            for term in ["revenue", "gross margin", "margin", "net income", "growth", "data center", "cloud", "capex", "capital expenditure"]:
                if term in last_query.lower():
                    metrics.append(term)
            metric_str = " and ".join(metrics) if metrics else "financial performance and revenue"
            return f"What was {target_entity} {metric_str}?"

        # 2. Pronoun replacement (their/its -> subject_company)
        resolved = clean_q
        if subject_company:
            resolved = re.sub(r'\b(their|theirs|its)\b', f"{subject_company}'s", resolved, flags=re.IGNORECASE)
            resolved = re.sub(r'\b(them|it)\b', subject_company, resolved, flags=re.IGNORECASE)

            if subject_company.lower() not in resolved.lower():
                resolved = f"{subject_company} {resolved}"

        # 3. Fiscal period continuity if present in last query
        fiscal_match = re.search(r'\b(Q[1-4]\s*(?:FY)?\d{2,4}|FY\d{2,4}|202[0-9])\b', last_query, re.IGNORECASE)
        if fiscal_match:
            period = fiscal_match.group(1)
            if period.lower() not in resolved.lower():
                resolved = f"{resolved.rstrip('?. ')} in {period}"

        return resolved.strip()

    @staticmethod
    def _query_with_memory(query: str, history: List[Dict[str, Any]]) -> str:
        """Give routing/retrieval only compact prior context, bounded to three turns."""
        if not history:
            return query
        prior_queries = [turn.get("user_query", "") for turn in history[-3:] if turn.get("user_query")]
        if not prior_queries:
            return query
        return "Prior analyst questions: " + " | ".join(prior_queries) + "\nCurrent question: " + query

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

    def _evaluate_retrieval_quality(self, query: str, chunks: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Self-RAG Evaluator: Evaluates retrieval quality against candidate chunks.
        Triggers autonomous self-correction if:
        1. Zero chunks retrieved from vector store.
        2. Highest similarity score is below 0.40 confidence threshold.
        3. Core query concepts/entities are absent from top results.
        """
        if not chunks:
            return True, "Zero chunks retrieved from vector store."

        scores = [c.get("similarity_score", 0.0) for c in chunks]
        max_sim = max(scores) if scores else 0.0

        if max_sim < 0.40:
            return True, f"Low retrieval confidence: maximum similarity score ({max_sim:.2f}) falls below 0.40 threshold."

        # Check for presence of key query entities & financial terms
        q_lower = query.lower()
        key_tokens = [w for w in re.findall(r'\b[a-zA-Z]{3,}\b', q_lower) if w not in {
            "what", "were", "with", "from", "that", "this", "have", "about", "explain", "tell",
            "show", "does", "report", "filing", "give", "some", "more", "then", "their", "there",
            "please", "could", "would", "should"
        }]

        if key_tokens and len(chunks) > 0:
            all_text = " ".join([c.get("text", "").lower() for c in chunks[:3]])
            matched_tokens = [t for t in key_tokens if t in all_text]
            match_ratio = len(matched_tokens) / len(key_tokens)
            if match_ratio < 0.30 and max_sim < 0.55:
                return True, f"Low concept overlap ({int(match_ratio*100)}% match for keywords: {', '.join(key_tokens[:4])})."

        return False, "Retrieval passed confidence and relevance checks."

    def _reformulate_query(
        self,
        query: str,
        reason: str,
        gemini_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None
    ) -> str:
        """
        Reformulates search query using frontier LLM (if API keys provided)
        or intelligent heuristic financial expansion.
        """
        gkey = gemini_api_key or settings.GEMINI_API_KEY
        okey = openai_api_key or settings.OPENAI_API_KEY

        if gkey:
            try:
                return self._reformulate_llm_gemini(query, key=gkey)
            except Exception as e:
                print(f"[Self-RAG] Gemini query reformulation failed: {e}")

        if okey:
            try:
                return self._reformulate_llm_openai(query, key=okey)
            except Exception as e:
                print(f"[Self-RAG] OpenAI query reformulation failed: {e}")

        return self._heuristic_query_expansion(query)

    def _reformulate_llm_gemini(self, query: str, key: str) -> str:
        model = getattr(settings, "GEMINI_MODEL", "gemini-3.5-flash-lite")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        prompt = (
            "You are a financial search query optimization specialist. "
            "Rewrite the user's natural language question into a clean, concise, keyword-rich search query "
            "optimized for corporate 10-K filings, earnings reports, and financial balance sheets. "
            "Strip all conversational filler (e.g. 'can you tell me', 'what about', 'please explain'). "
            "Expand financial abbreviations, tickers, and technical terms. "
            "Return ONLY the rewritten query string, nothing else.\n\n"
            f"User Question: {query}\n"
            "Optimized Search Query:"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 200}
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            rewritten = "".join([p.get("text", "") for p in parts if "text" in p]).strip().strip('"\'')
            return rewritten if rewritten else query

    def _reformulate_llm_openai(self, query: str, key: str) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a financial search query optimization specialist. Rewrite the user's query into a concise keyword-rich search query for financial documents. Return ONLY the rewritten query without quotes or extra text."},
                {"role": "user", "content": f"User query: {query}"}
            ],
            "temperature": 0.1,
            "max_tokens": 60
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            rewritten = data["choices"][0]["message"]["content"].strip().strip('"\'')
            return rewritten if rewritten else query

    def _heuristic_query_expansion(self, query: str) -> str:
        """
        Rule-based financial query expansion:
        1. Strips conversational prefixes.
        2. Normalizes tickers and company names.
        3. Expands core financial concepts.
        """
        clean = query.strip()
        clean = re.sub(r'^(can\s+you\s+(please\s+)?(tell\s+me|explain|give|show|find|summarize|detail)\s+(about\s+)?|what\s+(is|are|was|were|about)\s+|could\s+you\s+|please\s+|tell\s+me\s+about\s+|how\s+(does|is|did)\s+)', '', clean, flags=re.IGNORECASE).strip()
        clean = clean.rstrip("?.! ")

        expansions = []
        q_lower = clean.lower()

        if any(w in q_lower for w in ["nvidia", "nvda"]):
            expansions.append("NVIDIA NVDA Data Center Compute GPU Hopper Blackwell")
        if any(w in q_lower for w in ["microsoft", "msft"]):
            expansions.append("Microsoft MSFT Intelligent Cloud Azure commercial")
        if any(w in q_lower for w in ["apple", "aapl"]):
            expansions.append("Apple AAPL iPhone Services Gross Margin")
        if any(w in q_lower for w in ["tesla", "tsla"]):
            expansions.append("Tesla TSLA Automotive Deliveries Energy")

        if any(w in q_lower for w in ["margin", "profitability", "gross margin"]):
            expansions.append("gross margin percentage operating expenses profitability")
        if any(w in q_lower for w in ["capex", "capital expenditure", "investment"]):
            expansions.append("capital expenditures capex infrastructure investment")
        if any(w in q_lower for w in ["revenue", "sales", "growth", "growing"]):
            expansions.append("revenue growth YoY segment performance sales")
        if any(w in q_lower for w in ["risk", "threat", "uncertainty", "competition"]):
            expansions.append("risk factors supply chain regulatory export restrictions")
        if any(w in q_lower for w in ["balance sheet", "debt", "cash", "liabilities"]):
            expansions.append("balance sheet total assets liabilities cash equivalents debt")

        if expansions:
            return f"{clean} {' '.join(expansions[:2])}".strip()

        return f"{clean} financial results performance segment revenue"

    @staticmethod
    def _deduplicate_and_rank_context(chunks: List[Dict[str, Any]], max_chunks: int = 8) -> List[Dict[str, Any]]:
        """
        Deduplicates and ranks retrieved chunks before synthesis.
        1. Sorts by similarity_score descending.
        2. Removes near-duplicate chunks (>85% text overlap via character-level set similarity).
        3. Keeps top max_chunks after deduplication.
        """
        if not chunks:
            return chunks

        # Sort by relevance
        sorted_chunks = sorted(chunks, key=lambda c: c.get("similarity_score", 0.0), reverse=True)

        deduplicated = []
        seen_texts: List[set] = []

        for chunk in sorted_chunks:
            text = (chunk.get("text") or "").strip().lower()
            if not text:
                continue
            text_chars = set(text)

            # Check overlap with already-accepted chunks
            is_duplicate = False
            for seen in seen_texts:
                if not text_chars or not seen:
                    continue
                intersection = len(text_chars & seen)
                union = len(text_chars | seen)
                if union > 0 and (intersection / union) > 0.85:
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduplicated.append(chunk)
                seen_texts.append(text_chars)

            if len(deduplicated) >= max_chunks:
                break

        return deduplicated

    def _synthesize_memo(self, state: SupervisorState, gemini_api_key: Optional[str] = None, openai_api_key: Optional[str] = None, observation: Optional[Any] = None) -> str:
        """
        Synthesizes collected multimodal data into direct Markdown answer.
        Uses frontier LLM (Gemini / OpenAI) if keys available, else falls back to extractive synthesizer.
        Records synthesis provider and any failures in the execution trace.
        """
        gkey = gemini_api_key or settings.GEMINI_API_KEY
        okey = openai_api_key or settings.OPENAI_API_KEY

        # Deduplicate and compress context before synthesis
        state.search_results = self._deduplicate_and_rank_context(state.search_results)

        if gkey:
            try:
                result = self._synthesize_llm_gemini(state, key=gkey, observation=observation)
                state.add_trace(
                    event_type="result",
                    agent="Synthesizer",
                    content="Synthesis completed via Gemini LLM. Generated comprehensive answer from retrieved evidence.",
                    metadata={"provider": "gemini", "evidence_chunks": len(state.search_results)}
                )
                return result
            except Exception as e:
                logger.exception(
                    "Gemini synthesis failed (model=%s); trying configured fallback.",
                    settings.GEMINI_MODEL,
                )
                state.add_trace(
                    event_type="thought",
                    agent="Synthesizer",
                    content=f"Gemini synthesis failed: {e}. Attempting fallback provider.",
                    metadata={"provider": "gemini", "error": str(e)}
                )
                print(f"[Synthesizer] Gemini call failed: {e}, falling back.")
        
        if okey:
            try:
                result = self._synthesize_llm_openai(state, key=okey, observation=observation)
                state.add_trace(
                    event_type="result",
                    agent="Synthesizer",
                    content="Synthesis completed via OpenAI LLM. Generated comprehensive answer from retrieved evidence.",
                    metadata={"provider": "openai", "evidence_chunks": len(state.search_results)}
                )
                return result
            except Exception as e:
                state.add_trace(
                    event_type="thought",
                    agent="Synthesizer",
                    content=f"OpenAI synthesis failed: {e}. Falling back to extractive mode.",
                    metadata={"provider": "openai", "error": str(e)}
                )
                print(f"[Synthesizer] OpenAI call failed: {e}, falling back.")

        state.add_trace(
            event_type="thought",
            agent="Synthesizer",
            content="No LLM API available for synthesis. Using extractive mode (direct document excerpts).",
            metadata={"provider": "extractive", "reason": "No valid API key or all LLM calls failed"}
        )
        extractive = self._extractive_synthesizer(state)
        notice = (
            "> **Note:** AI synthesis is temporarily unavailable (no valid API key or LLM calls failed). "
            "The following is a direct extract from retrieved documents.\n\n"
        )
        return notice + extractive

    def _synthesize_llm_gemini(self, state: SupervisorState, key: str, observation: Optional[Any] = None) -> str:
        model = getattr(settings, "GEMINI_MODEL", "gemini-3.5-flash-lite")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        context_prompt = self._build_context_prompt(state)
        system_prompt = (
            "You are OmniBrain, an expert multimodal research analyst. "
            "Your task is to synthesize a comprehensive, well-reasoned answer to the user's question "
            "using ONLY the retrieved evidence provided below. Follow these rules strictly:\n\n"
            "SYNTHESIS RULES:\n"
            "1. SYNTHESIZE across all relevant evidence passages. Do NOT simply list or paraphrase individual chunks. "
            "Identify patterns, compare data points, draw connections, and present a coherent analysis.\n"
            "2. Structure your answer clearly: lead with the direct answer, then provide supporting analysis with details and reasoning.\n"
            "3. Use natural paragraphs and bullet points as appropriate. Never output boilerplate memo titles, "
            "'Institutional Research Memorandum', 'Prepared by', 'Query:', or repetitive headers.\n"
            "4. Include inline citations for every factual statement:\n"
            "   - For text reports/documents: [Source: document_name, p.X]\n"
            "   - For visual charts/tables: [Visual: image_name, p.X]\n"
            "   - For database metrics: [SQL: table_name]\n"
            "5. If evidence is conflicting, acknowledge both sides and explain the discrepancy.\n"
            "6. If evidence is insufficient to fully answer, say so explicitly: "
            "'Based on the available evidence, ...' and explain what information is missing.\n"
            "7. NEVER fabricate data, numbers, or facts not present in the retrieved context.\n"
            "8. When conversation history is provided, use it for contextual continuity but do not cite prior answers as evidence.\n"
            "9. For numerical data, present it precisely as it appears in the evidence.\n"
            "10. Be thorough. A good answer is typically 200-500 words for complex questions."
        )

        payload = {
            "system_instruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [
                {
                    "parts": [
                        {"text": f"User Question: {state.query}\n\n{context_prompt}\n\nProvide a comprehensive, well-synthesized answer:"}
                    ]
                }
            ],
            "generationConfig": {"temperature": settings.TEMPERATURE, "maxOutputTokens": 4096}
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:2_000]
            raise GeminiAPIError(
                f"Gemini generateContent failed: HTTP {exc.code}; response body: {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise GeminiAPIError(f"Gemini generateContent network error: {exc.reason}") from exc
        except TimeoutError as exc:
            raise GeminiAPIError("Gemini generateContent timed out after 30 seconds.") from exc

        if "error" in data:
            raise GeminiAPIError(f"Gemini returned an error payload: {data['error']}")
        logger.info(
            "Gemini synthesis succeeded (model=%s, response_id=%s, tokens=%s).",
            model,
            data.get("responseId"),
            data.get("usageMetadata", {}).get("totalTokenCount"),
        )

        usage = data.get("usageMetadata", {})
        update_observation(observation, metadata={"provider": "gemini", "model": model}, usage_details={
            "input_tokens": usage.get("promptTokenCount", 0),
            "output_tokens": usage.get("candidatesTokenCount", 0),
            "total_tokens": usage.get("totalTokenCount", 0),
        })
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = "".join([p.get("text", "") for p in parts if "text" in p]).strip()
        if not text:
            raise GeminiAPIError("Gemini returned no candidate text.")
        return text

    def _synthesize_llm_openai(self, state: SupervisorState, key: str, observation: Optional[Any] = None) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        context_prompt = self._build_context_prompt(state)
        system_prompt = (
            "You are OmniBrain, an expert multimodal research analyst. "
            "Synthesize a comprehensive, well-reasoned answer using ONLY the retrieved evidence provided. "
            "RULES: "
            "(1) SYNTHESIZE across all relevant evidence -- identify patterns, compare data, draw connections, present coherent analysis. "
            "Do NOT simply list or paraphrase individual chunks. "
            "(2) Lead with the direct answer, then provide supporting details and reasoning. "
            "(3) Use natural paragraphs and bullet points. Never output boilerplate headers, 'Query:', or 'Prepared by'. "
            "(4) Cite every factual statement: [Source: document_name, p.X], [Visual: image_name, p.X], or [SQL: table_name]. "
            "(5) If evidence is conflicting, acknowledge both sides. "
            "(6) If evidence is insufficient, say 'Based on the available evidence, ...' and explain what is missing. "
            "(7) NEVER fabricate data not in the context. "
            "(8) Use conversation history for continuity but do not cite prior answers as evidence. "
            "(9) Be thorough -- 200-500 words for complex questions."
        )
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User Question: {state.query}\n\n{context_prompt}\n\nProvide a comprehensive, well-synthesized answer:"}
            ],
            "temperature": settings.TEMPERATURE,
            "max_tokens": 4096
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            usage = data.get("usage", {})
            update_observation(observation, metadata={"provider": "openai", "model": "gpt-4o-mini"}, usage_details={
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            })
            return data["choices"][0]["message"]["content"].strip()

    def _build_context_prompt(self, state: SupervisorState) -> str:
        sections = []

        # Section 1: Conversation history (if multi-turn)
        if state.conversation_history:
            history_lines = ["=== CONVERSATION HISTORY (use for context continuity, do NOT cite as evidence) ==="]
            for idx, turn in enumerate(state.conversation_history[-3:], start=1):
                user_q = turn.get("user_query", "")
                memo_prev = turn.get("assistant_memo", "")[:300].replace("\n", " ").strip()
                if user_q:
                    history_lines.append(f"Turn {idx} -- User: '{user_q}' | Assistant summary: {memo_prev}")
            sections.append("\n".join(history_lines))

        # Section 2: SQL database results
        if state.sql_results:
            sql_lines = ["=== DATABASE RESULTS (SQL Agent) ==="]
            for res in state.sql_results:
                if res.get("is_valid") and res.get("rows"):
                    sql_lines.append(f"SQL Query: {res.get('executed_sql')} [SQL: quarterly_financials]")
                    sql_lines.append(f"Rows: {res.get('rows')}")
                    sql_lines.append("")
            sections.append("\n".join(sql_lines))

        # Section 3: Retrieved evidence passages (numbered, with similarity scores, truncated)
        if state.search_results:
            evidence_lines = ["=== RETRIEVED EVIDENCE PASSAGES ==="]
            for idx, chunk in enumerate(state.search_results, start=1):
                tag = "Visual" if chunk.get("chunk_type") == "visual" else "Text"
                sim = chunk.get("similarity_score", 0.0)
                source = chunk.get("source_document", "unknown")
                page = chunk.get("page_number", 1)
                section = chunk.get("section_title", "General")
                text = (chunk.get("text") or "").strip()
                # Truncate very long chunks to prevent single-chunk domination
                if len(text) > 600:
                    text = text[:597] + "..."
                evidence_lines.append(
                    f"[Evidence {idx}] ({tag}) [Source: {source}, p.{page}] "
                    f"Section: {section} | Relevance: {sim:.2f}\n{text}"
                )
                evidence_lines.append("")
            sections.append("\n".join(evidence_lines))

        return "\n\n".join(sections)

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
