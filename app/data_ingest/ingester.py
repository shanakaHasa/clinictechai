"""
Document Ingestion Handler
Manages PDF upload, storage, and initial processing
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging

from app.config.settings import settings

logger = logging.getLogger(__name__)


class DocumentIngester:
    """Handles document ingestion and storage"""
    
    def __init__(self):
        """Initialize document ingester"""
        self.storage_path = Path(settings.storage_path)
        self.raw_docs_path = self.storage_path / "raw_documents"
        self.processed_docs_path = self.storage_path / "processed_documents"
        
        # Create directories if they don't exist
        self.raw_docs_path.mkdir(parents=True, exist_ok=True)
        self.processed_docs_path.mkdir(parents=True, exist_ok=True)
    
    async def ingest_document(
        self, 
        file_path: str, 
        document_name: str,
        document_id: str
    ) -> Dict:
        """
        Ingest a PDF document into the system
        
        Args:
            file_path: Path to uploaded PDF file
            document_name: Original document name
            document_id: Unique identifier for the document
            
        Returns:
            Dictionary with ingestion status and metadata
        """
        try:
            # Create document directory
            doc_dir = self.raw_docs_path / document_id
            doc_dir.mkdir(exist_ok=True)
            
            # Move file to storage
            destination = doc_dir / document_name
            shutil.copy2(file_path, destination)
            
            # Create metadata file
            metadata = {
                "document_id": document_id,
                "document_name": document_name,
                "ingestion_date": datetime.utcnow().isoformat(),
                "file_size": os.path.getsize(destination),
                "original_path": file_path,
                "storage_path": str(destination)
            }
            
            logger.info(f"Document {document_id} ingested successfully")
            
            return {
                "success": True,
                "document_id": document_id,
                "storage_path": str(destination),
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error ingesting document: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_document_path(self, document_id: str) -> Optional[Path]:
        """Get the stored path of a document"""
        doc_dir = self.raw_docs_path / document_id
        if doc_dir.exists():
            pdf_files = list(doc_dir.glob("*.pdf"))
            if pdf_files:
                return pdf_files[0]
        return None
