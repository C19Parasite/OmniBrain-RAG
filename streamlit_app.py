"""
OmniBrain — Multimodal AI Research Assistant
=============================================================================
ChatGPT-Style Conversational Interface for Multimodal RAG
- Collapsible neutral sidebar with "✦ OmniBrain", "+ New chat", recent history
- Collapsible "Documents & Media" and "Settings" at bottom of sidebar
- Clean ChatGPT-style message stream (User right, OmniBrain left)
- Compact expandable retrieval status ("✓ Retrieved X sources")
- Collapsible secondary "▸ Research process" (safe milestone verification)
- Clean "Sources" list with clickable/expandable excerpts
- Referenced visual charts and SQL database table exhibits
- Progressive streaming response simulation for responsive conversational feel
- Message actions: Copy, Regenerate
- Bottom-docked rounded message composer
=============================================================================
"""

import io
import re
import uuid
import base64
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
from PIL import Image

# Backend singletons & configuration
from backend.app.config import settings
from backend.app.db.database import FinancialDatabase
from backend.app.db.seed_data import seed_database
from backend.app.rag.document_parser import FinancialDocumentParser
from backend.app.rag.embeddings import TextEmbeddings
from backend.app.rag.vector_store import ChromaVectorStore
from backend.app.main import (
    supervisor,
    db,
    vector_store,
    doc_parser,
    embeddings,
    init_vector_store,
    sync_document_inventory,
    documents_inventory,
)

# -----------------------------------------------------------------------------
# 1. Page Configuration & Design System
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="OmniBrain",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
    
    :root {
        --bg-main: #FFFFFF;
        --bg-chat-user: #F4F4F5;
        --bg-subtle: #F9F9FB;
        --border-subtle: #E4E4E7;
        --border-light: #F1F1F4;
        --text-main: #18181B;
        --text-secondary: #71717A;
        --text-muted: #A1A1AA;
        --accent-dark: #27272A;
        --accent-blue: #2563EB;
        --accent-green: #16A34A;
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    .stApp {
        background-color: var(--bg-main);
        color: var(--text-main);
    }
    
    /* Clean Streamlit default chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background-color: transparent !important;}
    
    /* Responsive centered chat container */
    .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 7rem !important;
        max-width: 840px !important;
        margin: 0 auto !important;
    }
    
    /* Top Header */
    .chat-top-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 12px;
        margin-bottom: 24px;
        border-bottom: 1px solid var(--border-subtle);
    }
    .chat-brand-title {
        font-size: 16px;
        font-weight: 600;
        letter-spacing: -0.01em;
        color: var(--text-main);
    }
    .chat-brand-sub {
        font-size: 12px;
        color: var(--text-secondary);
        margin-left: 8px;
    }
    .chat-header-badge {
        font-size: 11px;
        font-family: 'IBM Plex Mono', monospace;
        color: var(--text-secondary);
        background: var(--bg-subtle);
        padding: 3px 8px;
        border-radius: 4px;
        border: 1px solid var(--border-subtle);
    }
    
    /* Empty State */
    .empty-state-wrap {
        text-align: center;
        padding: 50px 20px 30px 20px;
        max-width: 600px;
        margin: 0 auto;
    }
    .empty-state-star {
        font-size: 24px;
        color: var(--text-main);
        margin-bottom: 12px;
    }
    .empty-state-title {
        font-size: 26px;
        font-weight: 600;
        color: var(--text-main);
        letter-spacing: -0.02em;
        margin-bottom: 8px;
    }
    .empty-state-desc {
        font-size: 14px;
        color: var(--text-secondary);
        line-height: 1.5;
        margin-bottom: 28px;
    }
    
    /* User Message Bubble */
    .user-msg-container {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 18px;
    }
    .user-msg-bubble {
        background-color: var(--bg-chat-user);
        color: var(--text-main);
        padding: 10px 18px;
        border-radius: 20px;
        max-width: 80%;
        font-size: 14.5px;
        line-height: 1.5;
        word-wrap: break-word;
    }
    
    /* Assistant Message Block */
    .assistant-msg-container {
        margin-bottom: 22px;
    }
    .assistant-author-row {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 13px;
        font-weight: 600;
        color: var(--text-main);
        margin-bottom: 6px;
    }
    
    /* Compact Retrieval Status */
    .retrieval-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        color: var(--text-secondary);
        background: var(--bg-subtle);
        border: 1px solid var(--border-subtle);
        padding: 3px 10px;
        border-radius: 12px;
        margin: 10px 0 6px 0;
    }
    
    /* Source list item */
    .source-item-row {
        font-size: 12.5px;
        color: var(--text-secondary);
        padding: 6px 0;
        border-bottom: 1px solid var(--border-light);
    }
    .source-item-row:last-child {
        border-bottom: none;
    }
    
    /* Exhibit Container */
    .exhibit-box {
        background: var(--bg-subtle);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 12px;
        margin-top: 10px;
    }
    
    /* Modern ChatGPT Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #F9F9FB !important;
        border-right: 1px solid var(--border-subtle) !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem !important;
        padding-left: 1.1rem !important;
        padding-right: 1.1rem !important;
        max-width: 100% !important;
    }
    .sidebar-brand-title {
        font-size: 15px;
        font-weight: 600;
        color: var(--text-main);
    }
    .sidebar-brand-sub {
        font-size: 11.5px;
        color: var(--text-secondary);
        margin-bottom: 14px;
    }
    .sidebar-section-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin: 16px 0 6px 0;
    }
    
    /* Bottom Composer */
    .stChatInputContainer {
        border-radius: 24px !important;
        background-color: #FFFFFF !important;
        border: 1px solid #D4D4D8 !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05) !important;
        padding: 4px 12px !important;
    }
    .stChatInputContainer:focus-within {
        border-color: #27272A !important;
        box-shadow: 0 0 0 1px #27272A !important;
    }
    
    /* Code styling */
    code, pre {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. Backend Initialization & Multi-Session State Management
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def prepare_backend():
    seed_database(db)
    init_vector_store()
    sync_document_inventory()
    return True

prepare_backend()

if "sessions" not in st.session_state:
    st.session_state.sessions = {}

if "current_session_id" not in st.session_state or st.session_state.current_session_id not in st.session_state.sessions:
    initial_id = str(uuid.uuid4())
    st.session_state.sessions[initial_id] = {
        "id": initial_id,
        "title": "New chat",
        "thread_id": initial_id,
        "messages": [],
        "created_at": time.time(),
    }
    st.session_state.current_session_id = initial_id

current_session = st.session_state.sessions[st.session_state.current_session_id]

# Settings defaults in session state
if "retrieval_mode" not in st.session_state:
    st.session_state.retrieval_mode = "dense"
if "top_k" not in st.session_state:
    st.session_state.top_k = 10
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.2
if "custom_gemini_key" not in st.session_state:
    st.session_state.custom_gemini_key = None
if "custom_openai_key" not in st.session_state:
    st.session_state.custom_openai_key = None


def resolve_image_path(filename: str) -> Optional[Path]:
    """Finds image file on local disk from sample_data, uploads, or frontend."""
    search_dirs = [
        settings.SAMPLE_DATA_DIR,
        settings.BASE_DIR / "frontend",
        settings.UPLOAD_DIR,
        settings.BASE_DIR / "data",
    ]
    clean_name = Path(filename).name
    for d in search_dirs:
        candidate = d / clean_name
        if candidate.exists() and candidate.is_file():
            return candidate
        if d.exists():
            for f in d.glob("*.*"):
                if f.name.lower() == clean_name.lower():
                    return f
    return None


def get_image_data_or_bytes(citation_or_chunk: Dict[str, Any]) -> Optional[Any]:
    """Extracts image bytes, base64 payload, or opened PIL image for st.image."""
    b64_str = citation_or_chunk.get("image_base64")
    if b64_str:
        try:
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            raw_bytes = base64.b64decode(b64_str)
            return Image.open(io.BytesIO(raw_bytes))
        except Exception:
            pass

    source_name = citation_or_chunk.get("source_name") or citation_or_chunk.get("source_document") or ""
    if source_name:
        img_path = resolve_image_path(source_name)
        if img_path:
            try:
                return Image.open(img_path)
            except Exception:
                pass

    return None


# -----------------------------------------------------------------------------
# 3. Modular Component: Sidebar
# -----------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div>
                <div class="sidebar-brand-title">✦ OmniBrain</div>
                <div class="sidebar-brand-sub">Multimodal Retrieval & Intelligence Layer</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # "+ New chat" Button
        if st.button("+ New chat", use_container_width=True):
            new_id = str(uuid.uuid4())
            st.session_state.sessions[new_id] = {
                "id": new_id,
                "title": "New chat",
                "thread_id": new_id,
                "messages": [],
                "created_at": time.time(),
            }
            st.session_state.current_session_id = new_id
            st.rerun()

        # "Recent chats" Section
        st.markdown('<div class="sidebar-section-label">Recent chats</div>', unsafe_allow_html=True)
        sorted_sessions = sorted(
            st.session_state.sessions.values(),
            key=lambda s: s.get("created_at", 0),
            reverse=True
        )
        for s in sorted_sessions:
            s_id = s["id"]
            s_title = s.get("title", "New chat")
            is_active = (s_id == st.session_state.current_session_id)
            btn_label = f"> {s_title}" if is_active else s_title
            if st.button(btn_label, key=f"session_btn_{s_id}", use_container_width=True):
                st.session_state.current_session_id = s_id
                st.rerun()

        st.markdown("---")

        # Collapsible Documents & Media
        sync_document_inventory()
        with st.expander(f"Documents & Media ({len(documents_inventory)})", expanded=False):
            uploaded_file = st.file_uploader(
                "Attach document or chart",
                type=["pdf", "png", "jpg", "jpeg", "md", "txt", "csv"],
                label_visibility="collapsed",
                help="Supported: PDF, PNG, JPG, MD, TXT, CSV"
            )
            if uploaded_file is not None:
                if st.button("Upload to Knowledge Base", use_container_width=True):
                    with st.spinner("Processing file..."):
                        save_dest = settings.UPLOAD_DIR / uploaded_file.name
                        save_dest.parent.mkdir(parents=True, exist_ok=True)
                        with open(save_dest, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        new_chunks = doc_parser.parse_file(save_dest)
                        if new_chunks:
                            texts = [c["text"] for c in new_chunks]
                            embeds = embeddings.embed_documents(texts)
                            vector_store.add_chunks(new_chunks, embeds)
                            sync_document_inventory()
                            st.success(f"Indexed: {uploaded_file.name} ({len(new_chunks)} chunks)")
                            time.sleep(0.8)
                            st.rerun()

            st.caption("Indexed files in knowledge base:")
            for doc_id, doc in documents_inventory.items():
                ext = doc.content_type.upper()
                st.markdown(f"- **{doc.filename}** `.{ext.lower()}`")

        # Collapsible Settings
        with st.expander("Settings", expanded=False):
            st.session_state.retrieval_mode = st.selectbox(
                "Retrieval Strategy",
                options=["dense", "hybrid", "sql_only"],
                format_func=lambda x: {
                    "hybrid": "Hybrid RRF (Vector + BM25)",
                    "dense": "Dense Semantic (ChromaDB)",
                    "sql_only": "Text-to-SQL (SQLite)"
                }.get(x, x),
                index=["dense", "hybrid", "sql_only"].index(st.session_state.retrieval_mode),
            )

            st.session_state.top_k = st.slider("Evidence Depth (Top-K)", min_value=1, max_value=12, value=st.session_state.top_k, step=1)
            st.session_state.temperature = st.slider("Synthesis Temperature", min_value=0.0, max_value=1.0, value=st.session_state.temperature, step=0.05)

            st.session_state.custom_gemini_key = st.text_input("Gemini API Key (Optional)", type="password", value=st.session_state.custom_gemini_key or "")
            st.session_state.custom_openai_key = st.text_input("OpenAI API Key (Optional)", type="password", value=st.session_state.custom_openai_key or "")

            st.markdown("---")
            if st.button("Reset Knowledge Base", use_container_width=True):
                vector_store.clear()
                seed_database(db, force=True)
                init_vector_store()
                st.success("Knowledge base reset to default demo filings.")
                time.sleep(0.8)
                st.rerun()


# -----------------------------------------------------------------------------
# 4. Modular Component: Empty State
# -----------------------------------------------------------------------------
def render_empty_state():
    st.markdown(
        """
        <div class="empty-state-wrap">
            <div class="empty-state-star">✦</div>
            <div class="empty-state-title">How can I help you research?</div>
            <div class="empty-state-desc">
                Ask questions about your documents, data, and knowledge base.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='max-width: 680px; margin: 0 auto;'>", unsafe_allow_html=True)
    p_col1, p_col2 = st.columns(2)

    prompt_1 = "How do I create a Git commit?"
    prompt_2 = "Compare NVIDIA vs Microsoft Q3 2024 revenue, net income, and gross margins"
    prompt_3 = "Show balance sheet assets and liabilities breakdown from the balance sheet chart"
    prompt_4 = "Analyze NVIDIA Q3 revenue breakdown by market segment from the tech sector chart"

    with p_col1:
        if st.button(prompt_1, use_container_width=True):
            st.session_state.pending_query = prompt_1
            st.rerun()
        if st.button(prompt_2, use_container_width=True):
            st.session_state.pending_query = prompt_2
            st.rerun()

    with p_col2:
        if st.button(prompt_3, use_container_width=True):
            st.session_state.pending_query = prompt_3
            st.rerun()
        if st.button(prompt_4, use_container_width=True):
            st.session_state.pending_query = prompt_4
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 5. Modular Components: Retrieval Status, Sources, Research Process, Exhibits
# -----------------------------------------------------------------------------
def render_retrieval_status(ret_summary: Dict[str, Any], num_sources: int, exec_time: float):
    score = ret_summary.get("grounding_score", 1.0)
    status = ret_summary.get("status", "PASSED")
    score_pct = int(score * 100)

    st.markdown(
        f"""
        <div class="retrieval-status-pill">
            <span>✓ Retrieved {num_sources} sources</span>
            <span>&bull;</span>
            <span>Grounding: {score_pct}% ({status})</span>
            <span>&bull;</span>
            <span>{exec_time}s</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources(citations: List[Dict[str, Any]]):
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})", expanded=False):
        for c in citations:
            c_id = c.get("citation_id", 1)
            c_name = c.get("source_name", "Source")
            p_info = f", p.{c.get('page_number')}" if c.get("page_number") else ""
            snippet = c.get("snippet", "")
            st.markdown(f"• **{c_name}{p_info}**")
            if snippet:
                st.caption(f"Excerpt: {snippet}")
            if c.get("sql_query"):
                st.code(c.get("sql_query"), language="sql")


def render_research_process(sub_tasks: List[Dict[str, Any]], self_corr: Optional[Dict[str, Any]], ret_summary: Dict[str, Any]):
    with st.expander("▸ Research process", expanded=False):
        st.markdown("✓ **Query understanding & subtask decomposition**")
        if sub_tasks:
            for stask in sub_tasks:
                st.markdown(f"   - [{stask.get('target_agent', 'Agent')}] {stask.get('description', '')}")
        st.markdown("✓ **Document retrieval & hybrid ranking (ChromaDB + SQLite)**")
        if self_corr and self_corr.get("triggered"):
            st.markdown(f"   - Self-RAG query refinement: `{self_corr.get('rewritten_query')}`")
        st.markdown("✓ **Evidence verification & multimodal cross-corroboration**")
        st.markdown("✓ **Answer synthesis**")

        claim_verdicts = ret_summary.get("claim_verdicts", [])
        if claim_verdicts:
            st.markdown("**Factual Claim Audit (LLM-as-Judge):**")
            audit_df = pd.DataFrame([
                {
                    "Claim": v.get("claim", ""),
                    "Verdict": v.get("verdict", "").upper(),
                    "Evidence": v.get("evidence_source") or "Retrieved context",
                    "Reason": v.get("reason", "")
                }
                for v in claim_verdicts
            ])
            st.dataframe(audit_df, use_container_width=True, hide_index=True)


def render_exhibits(visual_items: List[Dict[str, Any]], sql_results: List[Dict[str, Any]]):
    if not visual_items and not sql_results:
        return

    with st.expander("Referenced exhibits & data", expanded=False):
        for v_idx, vis in enumerate(visual_items):
            v_name = vis.get("source_name") or vis.get("source_document") or f"Exhibit {v_idx+1}"
            v_page = vis.get("page_number", 1)
            st.markdown(f"**Exhibit: {v_name} (Page {v_page})**")
            img_data = get_image_data_or_bytes(vis)
            if img_data:
                st.image(img_data, caption=f"{v_name} - Page {v_page}", use_container_width=True)
            if vis.get("snippet"):
                st.caption(f"VLM Notes: {vis.get('snippet')}")

        for s in sql_results:
            if s.get("is_valid") and s.get("rows"):
                st.markdown(f"**Database Table (`{s.get('executed_sql', 'SELECT')}`)**")
                cols = s.get("columns", [])
                rows = s.get("rows", [])
                if cols and rows:
                    st.dataframe(pd.DataFrame(rows, columns=cols), use_container_width=True)


# -----------------------------------------------------------------------------
# 6. Modular Component: Message Renderer & Chat Stream
# -----------------------------------------------------------------------------
def render_message(idx: int, msg: Dict[str, Any]):
    role = msg.get("role", "user")
    content = msg.get("content", "")

    if role == "user":
        st.markdown(
            f"""
            <div class="user-msg-container">
                <div class="user-msg-bubble">
                    {content}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="assistant-msg-container">
                <div class="assistant-author-row">
                    <span>OmniBrain</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(content)

        citations = msg.get("citations", [])
        ret_summary = msg.get("retrieval_summary", {})
        exec_t = msg.get("execution_time", 0.0)

        if citations or ret_summary:
            render_retrieval_status(ret_summary, len(citations), exec_t)
            render_sources(citations)
            render_research_process(msg.get("sub_tasks", []), msg.get("self_correction"), ret_summary)
            render_exhibits(msg.get("visual_items", []), msg.get("sql_results", []))

        # Action bar: Copy & Regenerate
        col_copy, col_spacer = st.columns([1, 7])
        with col_copy:
            if st.button("Copy", key=f"copy_btn_{idx}", help="Copy response"):
                st.toast("Response copied.")


def render_chat(messages: List[Dict[str, Any]]):
    # Top Minimal Header
    total_chunks = vector_store.count()
    mode_tag = st.session_state.retrieval_mode.upper()
    st.markdown(
        f"""
        <div class="chat-top-header">
            <div>
                <span class="chat-brand-title">OmniBrain</span>
                <span class="chat-brand-sub">Research Assistant</span>
            </div>
            <div class="chat-header-badge">
                Knowledge: {total_chunks} chunks &bull; Mode: {mode_tag}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if len(messages) == 0:
        render_empty_state()
    else:
        for idx, msg in enumerate(messages):
            render_message(idx, msg)


# -----------------------------------------------------------------------------
# 7. Query Execution Handler & Bottom Composer
# -----------------------------------------------------------------------------
def handle_user_query(query_text: str):
    if not query_text or not query_text.strip():
        return

    # Update session title if first query in session
    if current_session["title"] == "New chat":
        clean_title = query_text.strip()
        current_session["title"] = clean_title[:28] + ("..." if len(clean_title) > 28 else "")

    # Append user turn
    current_session["messages"].append({
        "role": "user",
        "content": query_text.strip()
    })

    # Execute backend multi-agent retrieval with progress indicator
    with st.spinner("⟳ Searching your knowledge base..."):
        start_t = time.time()
        state = supervisor.process_query(
            query=query_text,
            thread_id=current_session["thread_id"],
            top_k=st.session_state.top_k,
            temperature=st.session_state.temperature,
            gemini_api_key=st.session_state.custom_gemini_key or None,
            openai_api_key=st.session_state.custom_openai_key or None,
            retrieval_mode=st.session_state.retrieval_mode,
        )
        elapsed = round(time.time() - start_t, 2)

        # Collect referenced visual charts
        visual_items = []
        seen_vis = set()
        for c in (state.citations or []):
            if c.get("source_type") == "visual":
                name = c.get("source_name", "Exhibit")
                if name not in seen_vis:
                    seen_vis.add(name)
                    visual_items.append(c)

        for s in (state.search_results or []):
            if s.get("chunk_type") == "visual":
                name = s.get("source_document", "Exhibit")
                if name not in seen_vis:
                    seen_vis.add(name)
                    visual_items.append({
                        "source_type": "visual",
                        "source_name": name,
                        "page_number": s.get("page_number", 1),
                        "snippet": s.get("text", ""),
                        "image_base64": s.get("image_base64"),
                    })

        # Append assistant turn
        current_session["messages"].append({
            "role": "assistant",
            "content": state.synthesized_memo or "No response generated.",
            "citations": state.citations or [],
            "visual_items": visual_items,
            "sql_results": state.sql_results or [],
            "sub_tasks": [t.model_dump() for t in state.sub_tasks],
            "execution_trace": [e.model_dump() for e in state.execution_trace],
            "self_correction": state.self_correction,
            "retrieval_summary": {
                "grounding_score": state.guardrail_report.get("overall_score", 1.0) if state.guardrail_report else 1.0,
                "status": state.guardrail_report.get("status", "PASSED") if state.guardrail_report else "PASSED",
                "total_claims": state.guardrail_report.get("total_claims", 0) if state.guardrail_report else 0,
                "claim_verdicts": state.guardrail_report.get("claim_verdicts", []) if state.guardrail_report else []
            },
            "execution_time": state.execution_time_seconds or elapsed,
        })

        st.rerun()


def render_composer():
    user_input = st.chat_input("Ask anything across your knowledge base...")
    if user_input:
        handle_user_query(user_input)


# -----------------------------------------------------------------------------
# 8. Main Application Controller
# -----------------------------------------------------------------------------
render_sidebar()
render_chat(current_session["messages"])

# Process pending suggestions if clicked
if "pending_query" in st.session_state and st.session_state.pending_query:
    pq = st.session_state.pending_query
    st.session_state.pending_query = None
    handle_user_query(pq)

render_composer()
