"""
Pydantic Models for Request/Response Validation
Defines data schemas for the RAG API
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Tuple
from datetime import datetime


class DocumentUploadRequest(BaseModel):
    """Request model for document upload"""
    document_name: str = Field(..., description="Name of the document")
    document_type: Optional[str] = Field("pdf", description="Type of document")


class QueryRequest(BaseModel):
    """Request model for RAG query"""
    query: str = Field(..., description="User query")
    document_ids: Optional[List[str]] = Field(None, description="Specific documents to search")
    top_k: Optional[int] = Field(5, description="Number of results to retrieve")
    include_verification: Optional[bool] = Field(True, description="Include verification in response")
    session_id: Optional[str] = Field(None, description="Chat session ID for conversation history")


class ChunkMetadata(BaseModel):
    """Metadata for a text chunk"""
    page_number: int
    chunk_index: int
    document_id: str
    source_document: str
    extraction_type: str = "text"
    bbox: Optional[Tuple[float, float, float, float]] = None


class SourceEvidence(BaseModel):
    """Evidence from a source chunk"""
    page_number: int
    document: str
    exact_chunk: str
    bbox: Optional[Tuple[float, float, float, float]] = None
    chunk_id: str
    highlighted: str


class VerificationResult(BaseModel):
    """Answer verification result"""
    verified: bool
    confidence_score: float
    meets_threshold: bool
    grounding_score: Optional[float] = None
    consistency_score: Optional[float] = None
    relevance_score: Optional[float] = None
    evidence: List[SourceEvidence]
    checks: Optional[Dict[str, bool]] = None


class RAGResponse(BaseModel):
    """Response model for RAG queries"""
    success: bool
    answer: str
    query: str
    sources: List[Dict] = Field(default_factory=list, description="Source information")
    evidence: List[SourceEvidence] = Field(default_factory=list, description="Supporting evidence")
    page_numbers: List[int] = Field(default_factory=list, description="Pages referenced in answer")
    verification: Optional[VerificationResult] = None
    context_used: int = Field(default=0, description="Number of context chunks used")
    model: str = Field(default="gpt-4", description="LLM model used")
    confidence_score: float = Field(default=0.0, description="Confidence score for the answer")
    tokens_used: int = Field(default=0, description="Tokens used in generation")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DocumentUploadResponse(BaseModel):
    """Response model for document upload"""
    success: bool
    document_id: str
    document_name: str
    storage_path: str
    pdf_type: str
    total_chunks: int
    ingestion_date: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = False
    error: str
    error_code: Optional[str] = None
    details: Optional[Dict] = None


class ChatHistoryMessage(BaseModel):
    """A message in chat history"""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str
    timestamp: Optional[str] = None


class ChatSessionInfo(BaseModel):
    """Chat session information"""
    session_id: str
    total_messages: int
    user_messages: int
    assistant_messages: int
    documents_used: List[str] = []
    created_at: datetime
    last_updated: datetime
