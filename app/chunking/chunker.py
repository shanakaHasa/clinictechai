"""
Text Chunking with Metadata Preservation
Creates semantic chunks with preserved page numbers, bounding boxes, and document metadata
"""

from typing import Dict, List, Tuple
import logging

from app.config.settings import settings

logger = logging.getLogger(__name__)


class Chunk:
    """Represents a single chunk with metadata"""
    
    def __init__(
        self,
        chunk_id: str,
        text: str,
        page_number: int,
        chunk_index: int,
        bbox: Tuple[float, float, float, float] = None,
        document_id: str = None,
        source_document: str = None,
        extraction_type: str = "text"
    ):
        """
        Initialize a chunk with metadata
        
        Args:
            chunk_id: Unique identifier for the chunk
            text: Chunk text content
            page_number: Page number in the source document
            chunk_index: Index of chunk on the page
            bbox: Bounding box coordinates (x0, y0, x1, y1)
            document_id: ID of source document
            source_document: Name of source document
            extraction_type: Type of extraction (text or OCR)
        """
        self.chunk_id = chunk_id
        self.text = text
        self.page_number = page_number
        self.chunk_index = chunk_index
        self.bbox = bbox
        self.document_id = document_id
        self.source_document = source_document
        self.extraction_type = extraction_type
    
    def to_dict(self) -> Dict:
        """Convert chunk to dictionary"""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "bbox": self.bbox,
            "document_id": self.document_id,
            "source_document": self.source_document,
            "extraction_type": self.extraction_type
        }


class TextChunker:
    """Handles text chunking with metadata preservation"""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        Initialize text chunker
        
        Args:
            chunk_size: Number of characters per chunk
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
    
    async def chunk_page_content(
        self,
        page_content: Dict,
        document_id: str,
        source_document: str
    ) -> List[Chunk]:
        """
        Chunk a single page with metadata preservation
        
        Args:
            page_content: Dictionary with page text and metadata
            document_id: ID of source document
            source_document: Name of source document
            
        Returns:
            List of Chunk objects
        """
        try:
            text = page_content.get("text", "")
            page_number = page_content.get("page_number", 0)
            blocks = page_content.get("blocks", [])
            extraction_type = page_content.get("pdf_type", "text")
            
            chunks = []
            chunk_index = 0
            
            # Create chunks with overlap
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunk_text = text[i:i + self.chunk_size]
                
                if len(chunk_text.strip()) < 50:  # Skip very short chunks
                    continue
                
                # Find bbox for this chunk if available
                bbox = self._get_bbox_for_chunk(chunk_text, blocks)
                
                chunk_id = f"{document_id}_p{page_number}_c{chunk_index}"
                
                chunk = Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    bbox=bbox,
                    document_id=document_id,
                    source_document=source_document,
                    extraction_type=extraction_type
                )
                
                chunks.append(chunk)
                chunk_index += 1
            
            logger.info(f"Created {len(chunks)} chunks from page {page_number}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking page content: {str(e)}")
            return []
    
    async def chunk_documents(
        self,
        pages_content: List[Dict],
        document_id: str,
        source_document: str
    ) -> List[Chunk]:
        """
        Chunk all pages of a document
        
        Args:
            pages_content: List of page content dictionaries
            document_id: ID of source document
            source_document: Name of source document
            
        Returns:
            List of Chunk objects from all pages
        """
        all_chunks = []
        
        for page_content in pages_content:
            chunks = await self.chunk_page_content(
                page_content,
                document_id,
                source_document
            )
            all_chunks.extend(chunks)
        
        logger.info(f"Created total {len(all_chunks)} chunks from document")
        return all_chunks
    
    def _get_bbox_for_chunk(
        self,
        chunk_text: str,
        blocks: List[Dict]
    ) -> Tuple[float, float, float, float]:
        """
        Extract bounding box for a chunk from PDF blocks
        
        Args:
            chunk_text: Text of the chunk
            blocks: List of PDF blocks with bbox info
            
        Returns:
            Tuple of (x0, y0, x1, y1) or None
        """
        try:
            if not blocks:
                return None
            
            # Find blocks that contain text from chunk
            chunk_words = chunk_text.split()[:5]  # First 5 words
            
            for block in blocks:
                if isinstance(block, dict) and "bbox" in block:
                    block_text = block.get("text", "")
                    if any(word in block_text for word in chunk_words):
                        bbox = block["bbox"]
                        return tuple(bbox) if len(bbox) >= 4 else None
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting bbox: {str(e)}")
            return None
