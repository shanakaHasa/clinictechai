"""
API Endpoints for RAG System
Main API routes for document upload, querying, and retrieval
Uses services layer for business logic separation
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
import logging

from app.schemas.models import (
    QueryRequest,
    RAGResponse,
    DocumentUploadResponse,
)
from app.utils.helpers import setup_logging
from app.services.upload_service import UploadService
from app.services.query_service import QueryService
from app.embedding.embedding_service import EmbeddingService

router = APIRouter()
logger = setup_logging(__name__)

# Initialize services
upload_service = UploadService()
query_service = QueryService()
embedding_service = EmbeddingService()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document for processing
    
    Args:
        file: PDF file to upload
        
    Returns:
        DocumentUploadResponse with document metadata and chunk count
    """
    try:
        result = await upload_service.process_upload(file)
        
        return DocumentUploadResponse(
            success=result.get("success", True),
            document_id=result.get("document_id"),
            document_name=result.get("document_name"),
            storage_path=result.get("storage_path"),
            pdf_type=result.get("pdf_type"),
            total_chunks=result.get("total_chunks", 0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/query", response_model=RAGResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system
    
    Args:
        request: QueryRequest with query and optional session_id for chat history
        
    Returns:
        RAGResponse with answer and evidence
    """
    try:
        from app.config.settings import settings
        from datetime import datetime
        
        # Use session_id if provided for conversation history
        result = await query_service.process_query_with_history(
            request.query,
            session_id=request.session_id
        )
        
        return RAGResponse(
            success=True,
            answer=result.get("answer", ""),
            query=result.get("query", ""),
            sources=result.get("sources", []),
            evidence=result.get("evidence", []),
            page_numbers=result.get("page_numbers", []),
            verification=result.get("verification", None),
            context_used=result.get("context_used", 0),
            model=settings.llm_model,
            confidence_score=result.get("confidence_score", 0.0),
            tokens_used=result.get("tokens_used", 0),
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    from app.services.chat_memory import memory_manager
    
    try:
        memory = memory_manager.get_session(session_id)
        
        if not memory:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        return {
            "session_id": session_id,
            "messages": memory.get_history(),
            "stats": memory.get_summary_stats()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.delete("/chat/history/{session_id}")
async def clear_chat_history(session_id: str):
    """Clear chat history for a session"""
    from app.services.chat_memory import memory_manager
    
    try:
        deleted = memory_manager.delete_session(session_id)
        
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        return {"success": True, "message": f"Session {session_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/chat/sessions")
async def list_chat_sessions():
    """List all active chat sessions"""
    from app.services.chat_memory import memory_manager
    
    try:
        sessions_list = []
        for session_id in memory_manager.get_all_sessions():
            stats = memory_manager.get_session_stats(session_id)
            if stats:
                sessions_list.append(stats)
        
        return {
            "total_sessions": len(sessions_list),
            "sessions": sessions_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/health")
async def health_check():
    """Health check endpoint"""
    from app.embedding.embedding_service import VectorStore
    
    # Check if embedding service is initialized
    embedding_ok = embedding_service.client is not None
    
    # Check if vector store is initialized
    try:
        vector_store = VectorStore()
        vector_store_ok = vector_store.client is not None
    except:
        vector_store_ok = False
    
    return {
        "status": "healthy" if (embedding_ok and vector_store_ok) else "degraded",
        "services": {
            "embedding": embedding_ok,
            "vector_store": vector_store_ok
        }
    }


