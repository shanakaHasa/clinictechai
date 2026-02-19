"""
PDF Type Detection and Processing
Detects if PDF is scanned (image-based) or text-based and applies appropriate processing
"""

from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

import pymupdf
from pdf2image import convert_from_path

from app.pdf_processing.ocr_processor import OCRProcessor

logger = logging.getLogger(__name__)


class PDFType(str, Enum):
    """PDF Type Classification"""
    TEXT = "text"
    SCANNED = "scanned"
    MIXED = "mixed"


class PDFProcessor:
    """Handles PDF type detection and content extraction"""
    
    def __init__(self):
        """Initialize PDF processor"""
        self.text_extraction_threshold = 0.3  # 30% text content threshold (increased for better scanned detection)
        self.ocr_processor = OCRProcessor()
    
    async def detect_pdf_type(self, pdf_path: Path) -> PDFType:
        """
        Detect if PDF is scanned (image-based) or text-based
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            PDFType enum value
        """
        try:
            logger.info(f"[PDF TYPE DETECTION] Analyzing {pdf_path.name}...")
            doc = pymupdf.open(pdf_path)
            
            text_pages = 0
            total_pages = len(doc)
            pages_to_check = min(5, total_pages)
            
            logger.info(f"  Total pages: {total_pages}, checking first {pages_to_check} pages...")
            
            for page_num in range(pages_to_check):
                page = doc[page_num]
                text = page.get_text()
                text_length = len(text.strip())
                
                logger.debug(f"  Page {page_num + 1}: {text_length} characters extracted")
                
                # If page has substantial text, it's text-based
                if text_length > 100:
                    text_pages += 1
                    logger.debug(f"    → Contains text content")
                else:
                    logger.debug(f"    → Likely image/scanned (text < 100 chars)")
            
            doc.close()
            
            text_ratio = text_pages / pages_to_check
            logger.info(f"  Text ratio: {text_ratio:.1%} ({text_pages}/{pages_to_check} pages with text)")
            logger.info(f"  Threshold: {self.text_extraction_threshold:.1%}")
            
            if text_ratio > self.text_extraction_threshold:
                logger.info(f"  ✅ PDF TYPE: TEXT (ratio {text_ratio:.1%} > threshold {self.text_extraction_threshold:.1%})")
                return PDFType.TEXT
            else:
                logger.info(f"  ✅ PDF TYPE: SCANNED (ratio {text_ratio:.1%} <= threshold {self.text_extraction_threshold:.1%})")
                return PDFType.SCANNED
                
        except Exception as e:
            logger.error(f"❌ Error detecting PDF type: {str(e)}", exc_info=True)
            return PDFType.MIXED
    
    async def extract_text_pdf(self, pdf_path: Path) -> List[Dict]:
        """
        Extract text and metadata from text-based PDF
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dicts with page text, page number, and bbox info
        """
        try:
            logger.info("Extracting text from text-based PDF...")
            doc = pymupdf.open(pdf_path)
            pages_content = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                blocks = page.get_text("blocks")
                
                logger.debug(f"  Page {page_num + 1}: {len(text)} characters extracted")
                
                page_data = {
                    "page_number": page_num,
                    "text": text,
                    "blocks": blocks,  # Contains bbox info
                    "height": page.rect.height,
                    "width": page.rect.width,
                    "pdf_type": PDFType.TEXT.value
                }
                pages_content.append(page_data)
            
            doc.close()
            logger.info(f"✅ Extracted text from {len(pages_content)} pages")
            
            return pages_content
            
        except Exception as e:
            logger.error(f"❌ Error extracting text from PDF: {str(e)}", exc_info=True)
            return []
    
    async def extract_scanned_pdf(self, pdf_path: Path, document_id: str = None, filename: str = None) -> List[Dict]:
        """
        Extract text from scanned PDF using OCR
        
        Args:
            pdf_path: Path to the PDF file
            document_id: Document ID (optional, for saving extracted text)
            filename: Original filename (optional, for saving extracted text)
            
        Returns:
            List of dicts with OCR text and metadata
        """
        try:
            logger.info("Scanned PDF detected - applying OCR pipeline")
            pages_content = await self.ocr_processor.process_scanned_pdf(pdf_path)
            logger.info(f"OCR extracted {len(pages_content)} pages")
            
            # Save extracted text to files if document_id and filename provided
            if document_id and filename and pages_content:
                text_file = self.ocr_processor.save_extracted_text(pages_content, document_id, filename)
                logger.info(f"Extracted text saved: {text_file}")
            
            return pages_content
            
        except Exception as e:
            logger.error(f"Error processing scanned PDF: {str(e)}")
            return []
    
    async def process_pdf(self, pdf_path: Path, document_id: str = None, filename: str = None) -> Dict:
        """
        Full PDF processing pipeline
        
        Args:
            pdf_path: Path to the PDF file
            document_id: Document ID (optional)
            filename: Original filename (optional)
            
        Returns:
            Dictionary with processed content and metadata
        """
        try:
            # Detect PDF type
            pdf_type = await self.detect_pdf_type(pdf_path)
            
            logger.info(f"[PDF PROCESSING] Processing {pdf_path.name} as {pdf_type.value.upper()}")
            
            # Extract content based on type
            if pdf_type == PDFType.TEXT:
                logger.info("  Using text extraction (PyMuPDF)...")
                pages_content = await self.extract_text_pdf(pdf_path)
            elif pdf_type == PDFType.SCANNED:
                logger.info("  Using OCR extraction (Tesseract)...")
                pages_content = await self.extract_scanned_pdf(pdf_path, document_id, filename)
            else:
                logger.info("  Mixed type detected - attempting text extraction first...")
                pages_content = await self.extract_text_pdf(pdf_path)
            
            logger.info(f"✅ Processing complete: {len(pages_content)} pages extracted")
            
            return {
                "success": True,
                "pdf_type": pdf_type.value,
                "pages_content": pages_content,
                "total_pages": len(pages_content)
            }
            
        except Exception as e:
            logger.error(f"❌ Error in PDF processing pipeline: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
