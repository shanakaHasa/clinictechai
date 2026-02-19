"""
FastAPI RAG Application Main Entry Point
Multi-layer RAG system for medical document processing and intelligent retrieval
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import os
import sys

from app.config.settings import settings
from app.api.routes import router
from app.utils.helpers import setup_logging

# Initialize logger
logger = setup_logging(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    logger.info("FastAPI RAG Application starting...")
    logger.info(f"Environment: {settings.fastapi_env}")
    logger.info(f"LLM Provider: {settings.llm_provider}")
    logger.info(f"Vector DB: {settings.vector_db_type}")
    
    # Configure Tesseract path at startup
    if settings.tesseract_path:
        tesseract_path = settings.tesseract_path
        tesseract_dir = str(Path(tesseract_path).parent)
        
        logger.info(f"[OCR] Configuring Tesseract path: {tesseract_path}")
        
        # Add to system PATH
        if tesseract_dir not in os.environ.get('PATH', ''):
            os.environ['PATH'] = tesseract_dir + os.pathsep + os.environ.get('PATH', '')
            logger.info(f"[OCR] Added to Windows PATH: {tesseract_dir}")
        
        # Also set pytesseract_cmd
        import pytesseract
        pytesseract.pytesseract.pytesseract_cmd = tesseract_path
        logger.info(f"[OCR] Set pytesseract_cmd: {tesseract_path}")
        
        # Verify the path exists
        if Path(tesseract_path).exists():
            logger.info(f"[OCR] ✅ Tesseract found at: {tesseract_path}")
        else:
            logger.error(f"[OCR] ❌ Tesseract not found at: {tesseract_path}")
    else:
        logger.warning("[OCR] No tesseract_path configured in settings")
    
    yield
    logger.info("FastAPI RAG Application shutting down...")


# Create FastAPI application
app = FastAPI(
    title="ClinicTech RAG API",
    description="Multi-layer Retrieval-Augmented Generation system for medical documents",
    version="0.1.0",
    lifespan=lifespan
)


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=600,
)


# Include API routes
app.include_router(router, prefix="/api/v1", tags=["RAG"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ClinicTech RAG API",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": {
            "upload": "/api/v1/upload",
            "query": "/api/v1/query",
            "health": "/api/v1/health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="info"
    )
