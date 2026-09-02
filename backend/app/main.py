import shutil
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from .config import settings
from .models.schemas import (
    HealthCheckResponse, QueryRequest, QueryResponse,
    SQLTestRequest, SQLTestResponse,
    SearchTestRequest, SearchTestResponse, TextChunkResult,
    GuardrailReport, StructuredCitation,
    DocumentItem, TableSchemaInfo, SQLSandboxRequest, SQLSandboxResponse
)
from .db.database import FinancialDatabase
from .db.seed_data import seed_database
from .rag.embeddings import TextEmbeddings
from .rag.vector_store import ChromaVectorStore
from .rag.document_parser import FinancialDocumentParser
from .agents.sql_agent import TextToSQLAgent, SQLValidationError
from .agents.search_agent import SearchAgent
from .agents.vision_agent import VisionAgent
from .agents.supervisor import SupervisorOrchestrator
from .guardrails.evaluator import GuardrailEvaluator

# Instantiate singletons
db = FinancialDatabase()
sql_agent = TextToSQLAgent(db=db)

embeddings = TextEmbeddings()
vector_store = ChromaVectorStore()
vision_agent = VisionAgent()
doc_parser = FinancialDocumentParser(vision_agent=vision_agent)
search_agent = SearchAgent(vector_store=vector_store, embeddings=embeddings)
guardrail_evaluator = GuardrailEvaluator()

supervisor = SupervisorOrchestrator(
    sql_agent=sql_agent,
    search_agent=search_agent,
    evaluator=guardrail_evaluator
)

# Document inventory store in memory
documents_inventory: Dict[str, DocumentItem] = {}

def sync_document_inventory():
    """Scans sample_data and uploads folders to sync document inventory."""
    sample_dir = settings.BASE_DIR / "sample_data"
    upload_dir = settings.UPLOAD_DIR
    upload_dir.mkdir(parents=True, exist_ok=True)

    for folder in [sample_dir, upload_dir]:
        if folder.exists():
            for f in folder.glob("*.*"):
                if f.suffix.lower() in [".md", ".txt", ".pdf", ".png", ".jpg", ".jpeg", ".webp", ".svg"]:
                    doc_id = f.stem.lower()
                    if doc_id not in documents_inventory:
                        size_bytes = f.stat().st_size
                        ext = f.suffix.lower().replace(".", "")
                        documents_inventory[doc_id] = DocumentItem(
                            id=doc_id,
                            filename=f.name,
                            content_type=ext,
                            size_bytes=size_bytes,
                            page_count=1,
                            chunk_count=1,
                            created_at=datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                            status="indexed"
                        )

def init_vector_store():
    """Index sample financial documents and visual charts into Chroma vector store if empty."""
    sample_dir = settings.BASE_DIR / "sample_data"
    if sample_dir.exists():
        if vector_store.count() == 0:
            chunks = doc_parser.parse_directory(sample_dir)
            if chunks:
                texts = [c["text"] for c in chunks]
                embeds = embeddings.embed_documents(texts)
                vector_store.add_chunks(chunks, embeds)
                print(f"[OmniBrain] Indexed {len(chunks)} multimodal chunks into ChromaDB.")
    sync_document_inventory()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Seed synthetic DB and vector store
    seed_database(db)
    init_vector_store()
    yield

app = FastAPI(
    title=settings.APP_NAME,
    description="Agentic Multi-Modal RAG Orchestrator for Financial Analysis",
    version="0.1.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Health Check Endpoint
@app.get("/api/health", response_model=HealthCheckResponse)
async def health_check():
    db_stats = db.get_stats()
    return HealthCheckResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        version="0.1.0",
        total_documents=len(documents_inventory),
        total_chunks=vector_store.count(),
        database_records=db_stats
    )

# 2. Main Supervisor Orchestrator Query Endpoint
@app.post("/api/query", response_model=QueryResponse)
async def run_query(req: QueryRequest):
    """
    Main Supervisor Orchestration endpoint:
    Decomposes query, coordinates SQL, Search, Vision Agents, synthesizes memo,
    and runs LLM-as-Judge guardrail evaluation.
    """
    state = supervisor.process_query(req.query)
    
    # Format structured citations
    structured_cites = [
        StructuredCitation(
            citation_id=c.get("citation_id", idx + 1),
            source_type=c.get("source_type", "text"),
            source_name=c.get("source_name", "unknown"),
            page_number=c.get("page_number"),
            sql_query=c.get("sql_query"),
            snippet=c.get("snippet", ""),
            image_base64=c.get("image_base64")
        )
        for idx, c in enumerate(state.citations)
    ]

    # Format guardrail report
    gr_dict = state.guardrail_report
    gr_report = GuardrailReport(
        overall_score=gr_dict.get("overall_score", 1.0),
        status=gr_dict.get("status", "PASSED"),
        total_claims=gr_dict.get("total_claims", 0),
        grounded_claims=gr_dict.get("grounded_claims", 0),
        ungrounded_claims=gr_dict.get("ungrounded_claims", 0),
        claim_verdicts=gr_dict.get("claim_verdicts", [])
    )

    return QueryResponse(
        query=state.query,
        memo_markdown=state.synthesized_memo or "No memo generated.",
        guardrail_report=gr_report,
        citations=structured_cites,
        sub_tasks=[t.model_dump() for t in state.sub_tasks],
        execution_trace=[e.model_dump() for e in state.execution_trace],
        search_results=state.search_results,
        sql_results=state.sql_results,
        execution_time_seconds=state.execution_time_seconds
    )

# 3. Multimodal Document Center Endpoints
@app.get("/api/documents", response_model=List[DocumentItem])
async def list_documents():
    sync_document_inventory()
    return list(documents_inventory.values())

@app.post("/api/upload", response_model=DocumentItem)
async def upload_document(file: UploadFile = File(...)):
    """Uploads a report, PDF, or chart image, extracts text/visual descriptions, and indexes into ChromaDB."""
    try:
        dest_path = settings.UPLOAD_DIR / file.filename
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Parse document or visual chart
        chunks = doc_parser.parse_file(dest_path)
        if chunks:
            texts = [c["text"] for c in chunks]
            embeds = embeddings.embed_documents(texts)
            vector_store.add_chunks(chunks, embeds)

        doc_id = dest_path.stem.lower()
        doc_item = DocumentItem(
            id=doc_id,
            filename=file.filename,
            content_type=dest_path.suffix.lower().replace(".", ""),
            size_bytes=dest_path.stat().st_size,
            page_count=max(1, max([c.get("page_number", 1) for c in chunks])) if chunks else 1,
            chunk_count=len(chunks),
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            status="indexed"
        )
        documents_inventory[doc_id] = doc_item
        return doc_item
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    if doc_id in documents_inventory:
        del documents_inventory[doc_id]
    return {"status": "success", "doc_id": doc_id}

# 4. SQL Sandbox & Schema Explorer Endpoints (Strictly reusing Phase 1's safe read-only validator)
@app.get("/api/sql/schema", response_model=List[TableSchemaInfo])
async def get_sql_schema():
    """Browse financial database tables, columns, and sample rows."""
    return db.get_tables_metadata()

@app.post("/api/sql/sandbox", response_model=SQLSandboxResponse)
async def execute_sandbox_sql(req: SQLSandboxRequest):
    """
    Executes ad-hoc read-only SQL queries in the sandbox.
    Reuses the exact same allowlist validator and read-only connection as the SQL Agent.
    """
    start_time = datetime.now()
    try:
        # Validate query using SQL Agent's allowlist
        validated_sql = sql_agent.validate_sql(req.query)
        
        # Execute strictly on read-only connection
        exec_res = sql_agent._execute_read_only(validated_sql)
        duration_ms = round((datetime.now() - start_time).total_seconds() * 1000, 2)

        return SQLSandboxResponse(
            query=req.query,
            executed_sql=validated_sql,
            columns=exec_res.get("columns", []),
            rows=exec_res.get("rows", []),
            row_count=exec_res.get("row_count", 0),
            execution_time_ms=duration_ms,
            is_valid=exec_res.get("error") is None,
            error=exec_res.get("error"),
            data_source="SYNTHETIC_DEMO"
        )
    except SQLValidationError as ve:
        duration_ms = round((datetime.now() - start_time).total_seconds() * 1000, 2)
        return SQLSandboxResponse(
            query=req.query,
            executed_sql=None,
            columns=[],
            rows=[],
            row_count=0,
            execution_time_ms=duration_ms,
            is_valid=False,
            error=f"Validation Error: {str(ve)}",
            data_source="SYNTHETIC_DEMO"
        )

@app.post("/api/reset-demo")
async def reset_demo_data():
    """Re-seeds synthetic database and vector store with sample documents."""
    global documents_inventory
    vector_store.clear()
    documents_inventory.clear()
    seed_database(db, force=True)
    init_vector_store()
    return {"status": "success", "message": "Demo database and vector store reset successfully."}

# 5. Individual Agent Testing Endpoints
@app.post("/api/sql-test", response_model=SQLTestResponse)
async def test_sql_agent(req: SQLTestRequest):
    result = sql_agent.run(req.question)
    return SQLTestResponse(**result)

@app.post("/api/search-test", response_model=SearchTestResponse)
async def test_search_agent(req: SearchTestRequest):
    k = req.top_k or settings.TOP_K
    results = search_agent.search(req.query, top_k=k)
    chunk_items = [TextChunkResult(**c) for c in results]
    return SearchTestResponse(
        query=req.query,
        top_k=k,
        total_results=len(chunk_items),
        chunks=chunk_items
    )

# Static File Serving
STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(str(STATIC_DIR / "index.html"))
