"""
Pydantic schemas and base models for OmniBrain Studio.
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class HealthCheckResponse(BaseModel):
    """Schema for health check endpoint response."""
    status: str
    app_name: str
    version: str = "0.1.0"
    total_documents: int = 0
    total_chunks: int = 0
    database_records: Dict[str, int] = Field(default_factory=dict)

# --- Document Center Schemas ---

class DocumentItem(BaseModel):
    id: str
    filename: str
    content_type: str
    size_bytes: int = 0
    page_count: int = 1
    chunk_count: int = 0
    created_at: str = ""
    status: str = "indexed"

class UploadPreviewResponse(BaseModel):
    doc_id: str
    filename: str
    content_type: str
    size_bytes: int = 0
    page_count: int = 1
    chunk_count: int = 0
    is_image: bool = False
    image_base64: Optional[str] = None
    extracted_text: str

class CommitUploadRequest(BaseModel):
    doc_id: str
    filename: str
    content_type: str
    text_content: str
    image_base64: Optional[str] = None
    attach_to_chat: bool = True

# --- SQL Explorer Schemas ---

class ColumnInfo(BaseModel):
    name: str
    type: str
    is_primary_key: bool = False

class TableSchemaInfo(BaseModel):
    table_name: str
    columns: List[ColumnInfo] = Field(default_factory=list)
    row_count: int = 0
    sample_rows: List[Dict[str, Any]] = Field(default_factory=list)

class SQLSandboxRequest(BaseModel):
    query: str = Field(..., description="Read-only SQL query to execute in sandbox")

class SQLSandboxResponse(BaseModel):
    query: str
    executed_sql: Optional[str] = None
    columns: List[str] = Field(default_factory=list)
    rows: List[List[Any]] = Field(default_factory=list)
    row_count: int = 0
    execution_time_ms: float = 0.0
    is_valid: bool = True
    error: Optional[str] = None
    data_source: str = "SYNTHETIC_DEMO"

# --- Guardrails & Citations Models ---

class ClaimVerdict(BaseModel):
    claim: str
    verdict: str  # 'grounded', 'ungrounded', 'partial'
    reason: str
    evidence_source: Optional[str] = None

class StructuredCitation(BaseModel):
    citation_id: int
    source_type: str  # 'text', 'visual', 'sql'
    source_name: str
    page_number: Optional[int] = None
    sql_query: Optional[str] = None
    snippet: str = ""
    image_base64: Optional[str] = None

class GuardrailReport(BaseModel):
    overall_score: float = 1.0
    status: str = "PASSED"  # 'PASSED', 'WARNING', 'FAILED'
    total_claims: int = 0
    grounded_claims: int = 0
    ungrounded_claims: int = 0
    claim_verdicts: List[ClaimVerdict] = Field(default_factory=list)

# --- Multi-Agent Orchestrator Models ---

class QueryRequest(BaseModel):
    query: str
    thread_id: Optional[str] = None
    top_k: Optional[int] = None
    temperature: Optional[float] = None
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    document_ids: Optional[List[str]] = None  # Specific document context for this chat

class QueryResponse(BaseModel):
    query: str
    memo_markdown: str
    guardrail_report: GuardrailReport = Field(default_factory=GuardrailReport)
    citations: List[StructuredCitation] = Field(default_factory=list)
    sub_tasks: List[Dict[str, Any]] = Field(default_factory=list)
    execution_trace: List[Dict[str, Any]] = Field(default_factory=list)
    search_results: List[Dict[str, Any]] = Field(default_factory=list)
    sql_results: List[Dict[str, Any]] = Field(default_factory=list)
    execution_time_seconds: float = 0.0

# --- Individual Agent Test Models ---

class SQLTestRequest(BaseModel):
    question: str

class SQLTestResponse(BaseModel):
    question: str
    generated_sql: str
    executed_sql: Optional[str] = None
    columns: List[str] = Field(default_factory=list)
    rows: List[List[Any]] = Field(default_factory=list)
    row_count: int = 0
    execution_time_ms: float = 0.0
    is_valid: bool = True
    error: Optional[str] = None
    data_source: str = "SYNTHETIC_DEMO"

class SearchTestRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    document_ids: Optional[List[str]] = None

class TextChunkResult(BaseModel):
    id: str
    text: str
    source_document: str
    page_number: int = 1
    section_title: str = "General"
    chunk_type: str = "text"
    similarity_score: float = 0.0

class SearchTestResponse(BaseModel):
    query: str
    top_k: int
    total_results: int
    chunks: List[TextChunkResult] = Field(default_factory=list)
