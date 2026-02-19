#!/usr/bin/env python3
"""
Verify the setup and configuration for the RAG application
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_environment():
    """Verify environment configuration"""
    logger.info("=" * 60)
    logger.info("🔍 VERIFYING RAG APPLICATION SETUP")
    logger.info("=" * 60)
    
    # 1. Check Python version
    logger.info(f"1️⃣  Python Version: {sys.version}")
    
    # 2. Check required packages
    logger.info("2️⃣  Checking required packages...")
    required_packages = [
        'fastapi', 'uvicorn', 'pydantic', 'pydantic_settings',
        'openai', 'chromadb', 'pymupdf', 'pdf2image', 'pytesseract',
        'sentence_transformers'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"   ✅ {package}")
        except ImportError:
            logger.warning(f"   ⚠️  {package} not found")
    
    # 3. Check configuration
    logger.info("3️⃣  Checking configuration...")
    try:
        from app.config.settings import settings
        logger.info(f"   ✅ Config loaded")
        logger.info(f"      - LLM Provider: {settings.llm_provider}")
        logger.info(f"      - LLM Model: {settings.llm_model}")
        logger.info(f"      - Embedding Model: {settings.embedding_model}")
        logger.info(f"      - Embedding Dimension: {settings.embedding_dimension}")
        logger.info(f"      - Vector DB: {settings.vector_db_type}")
        logger.info(f"      - Vector DB Path: {settings.vector_db_path}")
        
        # Check for API key
        if settings.llm_api_key:
            key_preview = settings.llm_api_key[:20] + "..." if len(settings.llm_api_key) > 20 else settings.llm_api_key
            logger.info(f"      - API Key: {key_preview} (present)")
        else:
            logger.error(f"      - API Key: NOT SET")
    except Exception as e:
        logger.error(f"   ❌ Error loading config: {str(e)}")
    
    # 4. Check embedding service
    logger.info("4️⃣  Checking embedding service...")
    try:
        from app.embedding.embedding_service import EmbeddingService
        embedding_service = EmbeddingService()
        
        if embedding_service.client:
            logger.info(f"   ✅ Embedding service initialized")
            logger.info(f"      - Model: {embedding_service.model_name}")
            logger.info(f"      - Dimension: {embedding_service.embedding_dimension}")
        else:
            logger.error(f"   ❌ Embedding service client not initialized")
    except Exception as e:
        logger.error(f"   ❌ Error initializing embedding service: {str(e)}")
    
    # 5. Check vector store
    logger.info("5️⃣  Checking vector store...")
    try:
        from app.embedding.embedding_service import VectorStore
        vector_store = VectorStore()
        
        if vector_store.collection:
            logger.info(f"   ✅ Vector store initialized")
            logger.info(f"      - Backend: {vector_store.backend}")
            logger.info(f"      - Collection: rag_documents")
        else:
            logger.warning(f"   ⚠️  Vector store collection not ready")
    except Exception as e:
        logger.error(f"   ❌ Error initializing vector store: {str(e)}")
    
    # 6. Check LLM service
    logger.info("6️⃣  Checking LLM service...")
    try:
        from app.llm.llm_service import LLMService
        llm_service = LLMService()
        
        if llm_service.client:
            logger.info(f"   ✅ LLM service initialized")
            logger.info(f"      - Provider: {llm_service.provider}")
            logger.info(f"      - Model: {llm_service.model}")
        else:
            logger.error(f"   ❌ LLM service client not initialized")
    except Exception as e:
        logger.error(f"   ❌ Error initializing LLM service: {str(e)}")
    
    logger.info("=" * 60)
    logger.info("✅ VERIFICATION COMPLETE")
    logger.info("=" * 60)

if __name__ == "__main__":
    verify_environment()
