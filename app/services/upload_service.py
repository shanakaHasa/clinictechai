"""
Upload Service Layer
Handles document upload workflow with 5-step processing pipeline
"""

import tempfile
import logging
from pathlib import Path
from fastapi import HTTPException
from typing import Dict

from app.utils.helpers import generate_id
from app.data_ingest.ingester import DocumentIngester
from app.pdf_processing.processor import PDFProcessor
from app.chunking.chunker import TextChunker
from app.embedding.embedding_service import EmbeddingService, VectorStore

logger = logging.getLogger(__name__)


class UploadService:
    """Service for document upload and processing"""
    
    def __init__(self):
        """Initialize upload service with dependencies"""
        self.document_ingester = DocumentIngester()
        self.pdf_processor = PDFProcessor()
        self.text_chunker = TextChunker()
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
    
    async def process_upload(self, file) -> Dict:
        """
        Process document upload with 5-step pipeline
        
        Args:
            file: UploadFile object
            
        Returns:
            Dictionary with upload result
        """
        logger.info("=" * 60)
        logger.info("[UPLOAD] Starting document upload")
        logger.info(f"   File: {file.filename}")
        logger.info(f"   Size: {file.size} bytes")
        
        # Generate document ID
        document_id = generate_id("doc")
        logger.info(f"   Document ID: {document_id}")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name
        
        logger.debug(f"   Temp file: {tmp_path}")
        
        try:
            # Check for duplicate document
            await self._check_duplicate(document_id, file.filename)
            
            # Step 1: Ingest document
            ingest_result = await self._step_ingest(tmp_path, file.filename, document_id)
            
            # Step 2: Process PDF
            pdf_result = await self._step_process_pdf(
                ingest_result["storage_path"],
                document_id,
                file.filename
            )
            
            # Step 3: Chunk content
            chunks = await self._step_chunk(
                pdf_result["pages_content"],
                document_id,
                file.filename
            )
            
            if len(chunks) == 0:
                logger.warning("   [WARNING] No chunks created")
                return {
                    "success": True,
                    "document_id": document_id,
                    "document_name": file.filename,
                    "storage_path": ingest_result["storage_path"],
                    "pdf_type": pdf_result["pdf_type"],
                    "total_chunks": 0
                }
            
            # Step 4: Generate embeddings
            embeddings = await self._step_embed(chunks)
            
            # Step 5: Store vectors
            store_result = await self._step_store(chunks, embeddings)
            
            logger.info("=" * 60)
            logger.info(f"[SUCCESS] Document uploaded and persisted: {document_id}")
            logger.info("=" * 60)
            
            return {
                "success": True,
                "document_id": document_id,
                "document_name": file.filename,
                "storage_path": ingest_result["storage_path"],
                "pdf_type": pdf_result["pdf_type"],
                "total_chunks": len(chunks)
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("=" * 60)
            logger.error(f"[ERROR] Error uploading document: {str(e)}")
            logger.error("=" * 60)
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _check_duplicate(self, document_id: str, filename: str):
        """Check if document already exists"""
        logger.info("[DUPLICATE CHECK] Checking if document already exists...")
        if self.vector_store.check_document_exists(document_id):
            logger.warning(f"   [WARNING] Document already uploaded: {document_id}")
            raise HTTPException(
                status_code=400,
                detail=f"Document '{filename}' has already been uploaded. Cannot upload duplicate files."
            )
        logger.info("   [OK] Document is unique - proceeding with upload")
    
    async def _step_ingest(self, tmp_path: str, filename: str, document_id: str) -> Dict:
        """Step 1: Ingest document"""
        logger.info("[1/5] Ingesting document...")
        ingest_result = await self.document_ingester.ingest_document(
            tmp_path,
            filename,
            document_id
        )
        
        if not ingest_result["success"]:
            logger.error(f"   [ERROR] Ingestion failed: {ingest_result.get('error', 'Unknown error')}")
            raise HTTPException(status_code=400, detail=ingest_result["error"])
        
        logger.info(f"   [OK] Document ingested: {ingest_result['storage_path']}")
        return ingest_result
    
    async def _step_process_pdf(self, storage_path: str, document_id: str = None, filename: str = None) -> Dict:
        """Step 2: Process PDF"""
        logger.info("[2/5] Processing PDF...")
        pdf_result = await self.pdf_processor.process_pdf(
            Path(storage_path),
            document_id=document_id,
            filename=filename
        )
        
        if not pdf_result["success"]:
            logger.error(f"   [ERROR] PDF processing failed: {pdf_result.get('error', 'Unknown error')}")
            raise HTTPException(status_code=400, detail=pdf_result["error"])
        
        logger.info(f"   [OK] PDF processed: Type={pdf_result.get('pdf_type', 'N/A')}, Pages={len(pdf_result.get('pages_content', []))}")
        return pdf_result
    
    async def _step_chunk(self, pages_content, document_id: str, filename: str) -> list:
        """Step 3: Chunk content"""
        logger.info("[3/5] Chunking content...")
        chunks = await self.text_chunker.chunk_documents(
            pages_content,
            document_id,
            filename
        )
        logger.info(f"   [OK] Created {len(chunks)} chunks")
        return chunks
    
    async def _step_embed(self, chunks) -> list:
        """Step 4: Generate embeddings"""
        logger.info("[4/5] Generating embeddings...")
        chunk_texts = [chunk.text for chunk in chunks]
        logger.debug(f"   Texts to embed: {len(chunk_texts)}")
        
        embeddings = await self.embedding_service.embed_texts(chunk_texts)
        logger.info(f"   [OK] Generated {len(embeddings)} embeddings")
        
        if len(embeddings) == 0:
            logger.error("   [ERROR] Embedding generation failed - returned empty list")
            raise HTTPException(status_code=500, detail="Failed to generate embeddings")
        
        if len(embeddings) != len(chunks):
            logger.error(f"   [ERROR] Embedding count ({len(embeddings)}) != Chunk count ({len(chunks)})")
            raise HTTPException(status_code=500, detail=f"Embedding mismatch: {len(embeddings)} vs {len(chunks)}")
        
        # Validate embeddings
        for i, emb in enumerate(embeddings):
            if len(emb) != 1536:
                logger.error(f"   [ERROR] Embedding {i} has wrong dimension: {len(emb)} (expected 1536)")
                raise HTTPException(status_code=500, detail=f"Invalid embedding dimension at index {i}")
        
        logger.debug(f"   Embedding dimensions: {len(embeddings[0])} (expected 1536)")
        return embeddings
    
    async def _step_store(self, chunks, embeddings) -> Dict:
        """Step 5: Store vectors in ChromaDB"""
        logger.info("[5/5] Storing vectors in ChromaDB...")
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        metadata = [chunk.to_dict() for chunk in chunks]
        
        logger.debug(f"   Chunk IDs: {len(chunk_ids)}")
        logger.debug(f"   Metadata: {len(metadata)}")
        
        store_result = await self.vector_store.add_vectors(chunk_ids, embeddings, metadata)
        
        if not store_result.get("success"):
            logger.error(f"   [ERROR] Vector storage failed: {store_result.get('error', 'Unknown error')}")
            raise HTTPException(status_code=500, detail=f"Vector storage error: {store_result.get('error', 'Unknown')}")
        
        logger.info(f"   [OK] Vectors stored: {store_result.get('added', 0)} vectors in {store_result.get('backend', 'N/A')}")
        return store_result
