"""
OCR Pipeline for Scanned PDF Processing
Implements Tesseract-based OCR for scanned documents
"""

from pathlib import Path
from typing import Dict, List, Optional
import logging
import numpy as np
from datetime import datetime
from io import BytesIO

import pytesseract
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError
from PIL import Image, ImageEnhance
import pymupdf

from app.config.settings import settings

logger = logging.getLogger(__name__)


class OCRProcessor:
    """Handles OCR processing for scanned PDFs"""
    
    def __init__(self):
        """Initialize OCR processor"""
        # Configure tesseract path BEFORE any OCR operations
        logger.info("[OCR] Initializing OCRProcessor...")
        
        # Set tesseract path from settings
        if settings.tesseract_path:
            logger.info(f"[OCR] Configuring Tesseract path: {settings.tesseract_path}")
            pytesseract.pytesseract.pytesseract_cmd = settings.tesseract_path
            
            # Verify the path exists
            from pathlib import Path
            if Path(settings.tesseract_path).exists():
                logger.info(f"[OCR] ✅ Tesseract found at: {settings.tesseract_path}")
            else:
                logger.error(f"[OCR] ❌ Tesseract not found at: {settings.tesseract_path}")
        else:
            logger.warning("[OCR] No tesseract_path configured in settings")
        
        # Ensure extracted text directory exists
        self.extracted_text_dir = Path(settings.vector_db_path).parent / "extracted_text"
        self.extracted_text_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[OCR] Extracted text directory: {self.extracted_text_dir}")
    
    async def process_scanned_pdf(self, pdf_path: Path) -> List[Dict]:
        """
        Process scanned PDF using OCR
        
        Args:
            pdf_path: Path to the scanned PDF
            
        Returns:
            List of dicts with OCR text and metadata
        """
        try:
            logger.info(f"Converting PDF to images for OCR processing...")
            
            # Try using pdf2image first (requires poppler)
            images = None
            try:
                logger.info("Attempting with pdf2image (DPI=300)...")
                images = convert_from_path(pdf_path, dpi=300)
                logger.info(f"Converted PDF to {len(images)} images at 300 DPI")
            except PDFInfoNotInstalledError as e:
                logger.warning(f"pdf2image failed (Poppler not installed): {str(e)}")
                logger.info("Falling back to PyMuPDF image extraction...")
                images = await self._extract_images_pymupdf(pdf_path)
            
            if not images:
                logger.error("Failed to extract images from PDF using both methods")
                return []
            
            logger.info(f"Successfully extracted {len(images)} images from PDF")
            
            pages_content = []
            
            for page_num, image in enumerate(images):
                logger.info(f"Processing page {page_num + 1}/{len(images)}...")
                
                # Preprocess image for better OCR
                processed_image = self._preprocess_image(image)
                
                # Apply OCR with error handling
                try:
                    text = pytesseract.image_to_string(processed_image)
                except pytesseract.pytesseract.TesseractNotFoundError as e:
                    logger.error(f"❌ Tesseract not found: {str(e)}")
                    logger.error(f"   Configured path: {settings.tesseract_path}")
                    logger.error(f"   Path exists: {Path(settings.tesseract_path).exists() if settings.tesseract_path else 'Not configured'}")
                    logger.error(f"   Please ensure Tesseract-OCR is installed at: C:\\Program Files\\Tesseract-OCR")
                    raise
                
                # Get detailed OCR data
                ocr_data = pytesseract.image_to_data(processed_image, output_type="dict")
                
                # Calculate confidence
                confidence = self._calculate_confidence(ocr_data)
                
                logger.info(f"  Page {page_num + 1}: Extracted {len(text)} characters, confidence: {confidence:.2%}")
                
                if not text or len(text.strip()) == 0:
                    logger.warning(f"  ⚠️  Page {page_num + 1}: OCR returned empty text - may be blank or image-heavy page")
                
                page_data = {
                    "page_number": page_num,
                    "text": text,
                    "ocr_data": ocr_data,
                    "confidence": confidence,
                    "image_height": image.height,
                    "image_width": image.width,
                    "pdf_type": "scanned"
                }
                pages_content.append(page_data)
            
            logger.info(f"✅ OCR processing completed for {len(pages_content)} pages")
            return pages_content
            
        except Exception as e:
            logger.error(f"❌ Error in OCR processing: {str(e)}", exc_info=True)
            return []
    
    async def _extract_images_pymupdf(self, pdf_path: Path) -> List[Image.Image]:
        """
        Fallback method to extract images from PDF using PyMuPDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of PIL Image objects
        """
        try:
            logger.info("Extracting images from PDF using PyMuPDF...")
            images = []
            
            doc = pymupdf.open(pdf_path)
            logger.info(f"Opened PDF with {len(doc)} pages")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Render page to image (pixmap)
                logger.debug(f"Rendering page {page_num + 1} to image...")
                pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))  # 2x zoom for better quality
                
                # Convert pixmap to PIL Image
                img_data = pix.tobytes("ppm")
                img = Image.open(BytesIO(img_data))
                images.append(img)
                
                logger.debug(f"Page {page_num + 1} rendered: {img.size}")
            
            doc.close()
            logger.info(f"✅ Extracted {len(images)} images from PDF using PyMuPDF")
            return images
            
        except Exception as e:
            logger.error(f"❌ Error extracting images with PyMuPDF: {str(e)}", exc_info=True)
            return []
    
    async def save_extracted_text(self, pages_content: List[Dict], document_id: str, filename: str) -> str:
        """
        Save extracted OCR text to files
        
        Args:
            pages_content: List of page data with extracted text
            document_id: Document ID
            filename: Original filename
            
        Returns:
            Path to the saved text file
        """
        try:
            # Create document-specific directory
            doc_text_dir = self.extracted_text_dir / document_id
            doc_text_dir.mkdir(parents=True, exist_ok=True)
            
            # Save combined text
            combined_text_file = doc_text_dir / f"{Path(filename).stem}_full.txt"
            combined_text = []
            
            # Save individual page texts
            for page_data in pages_content:
                page_num = page_data.get("page_number", 0)
                text = page_data.get("text", "")
                confidence = page_data.get("confidence", 0)
                
                # Save individual page file
                page_file = doc_text_dir / f"page_{page_num + 1}.txt"
                with open(page_file, "w", encoding="utf-8") as f:
                    f.write(f"=== Page {page_num + 1} ===\n")
                    f.write(f"OCR Confidence: {confidence:.2%}\n")
                    f.write(f"Extracted at: {datetime.now().isoformat()}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(text)
                
                logger.debug(f"Saved page text: {page_file}")
                
                # Add to combined text
                combined_text.append(f"=== Page {page_num + 1} (Confidence: {confidence:.2%}) ===\n{text}")
            
            # Save combined text file
            combined_content = "\n\n".join(combined_text)
            with open(combined_text_file, "w", encoding="utf-8") as f:
                f.write(f"Document: {filename}\n")
                f.write(f"Document ID: {document_id}\n")
                f.write(f"Extracted: {datetime.now().isoformat()}\n")
                f.write(f"Total Pages: {len(pages_content)}\n")
                f.write("=" * 70 + "\n\n")
                f.write(combined_content)
            
            logger.info(f"✅ Extracted text saved to: {combined_text_file}")
            logger.info(f"   Individual pages saved in: {doc_text_dir}")
            logger.info(f"   Total pages: {len(pages_content)}")
            
            return str(combined_text_file)
            
        except Exception as e:
            logger.error(f"❌ Error saving extracted text: {str(e)}", exc_info=True)
            return ""
    
    async def ocr_image(self, image: Image.Image) -> Dict:
        """
        Apply OCR to a single image
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with OCR text and detailed data
        """
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Apply OCR with error handling for Tesseract not found
            try:
                text = pytesseract.image_to_string(processed_image)
            except pytesseract.pytesseract.TesseractNotFoundError as e:
                logger.error(f"❌ Tesseract not found: {str(e)}")
                logger.error(f"   Configured path: {settings.tesseract_path}")
                logger.error(f"   Path exists: {Path(settings.tesseract_path).exists() if settings.tesseract_path else 'Not configured'}")
                return {"text": "", "error": f"Tesseract OCR not found at {settings.tesseract_path}"}
            
            ocr_data = pytesseract.image_to_data(processed_image, output_type="dict")
            confidence = self._calculate_confidence(ocr_data)
            
            logger.info(f"OCR completed: {len(text)} characters, confidence: {confidence:.2%}")
            
            return {
                "text": text,
                "ocr_data": ocr_data,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"❌ Error in image OCR: {str(e)}", exc_info=True)
            return {"text": "", "error": str(e)}
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR accuracy
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed image
        """
        try:
            # Convert to grayscale for better text detection
            logger.debug("Converting image to grayscale...")
            gray_image = image.convert("L")
            
            # Increase contrast to make text stand out
            logger.debug("Enhancing contrast...")
            enhancer = ImageEnhance.Contrast(gray_image)
            contrasted = enhancer.enhance(2.0)  # Double the contrast
            
            # Increase sharpness
            logger.debug("Enhancing sharpness...")
            sharpness_enhancer = ImageEnhance.Sharpness(contrasted)
            sharpened = sharpness_enhancer.enhance(2.0)
            
            # Increase brightness if needed
            logger.debug("Enhancing brightness...")
            brightness_enhancer = ImageEnhance.Brightness(sharpened)
            brightened = brightness_enhancer.enhance(1.1)
            
            logger.debug("✅ Image preprocessing complete")
            return brightened
            
        except Exception as e:
            logger.warning(f"Error preprocessing image: {str(e)}, using original image")
            return image
    
    def _calculate_confidence(self, ocr_data: Dict) -> float:
        """Calculate overall OCR confidence score"""
        try:
            confidences = ocr_data.get("conf", [])
            valid_confidences = [c for c in confidences if c > 0]
            
            if valid_confidences:
                avg_confidence = sum(valid_confidences) / len(valid_confidences) / 100
                return max(0.0, min(1.0, avg_confidence))  # Clamp between 0 and 1
            
            logger.debug("No confidence data available from OCR")
            return 0.0
            
        except Exception as e:
            logger.error(f"❌ Error calculating confidence: {str(e)}")
            return 0.0
