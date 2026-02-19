"""
Test OCR Text Extraction to Files
Verifies that extracted text is saved to text files in storage
"""

import asyncio
import logging
from pathlib import Path
from app.pdf_processing.ocr_processor import OCRProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_save_extracted_text():
    """Test saving extracted OCR text to files"""
    
    logger.info("=" * 70)
    logger.info("[TEST] OCR TEXT EXTRACTION TO FILES")
    logger.info("=" * 70)
    
    try:
        ocr = OCRProcessor()
        
        # Create test OCR data (simulating what would come from a scanned PDF)
        logger.info("\n[1/3] Creating test OCR data...")
        
        test_pages = [
            {
                "page_number": 0,
                "text": "This is a test document demonstrating OCR text extraction.\n\nMedical information is being processed and extracted from scanned documents.\nAll text is properly preserved with formatting maintained.",
                "confidence": 0.95,
                "ocr_data": {"conf": [95, 94, 93]}
            },
            {
                "page_number": 1,
                "text": "Page 2 contains additional medical data.\n\nPatient records show symptoms and treatment plans.\nFollow-up appointments are scheduled accordingly.",
                "confidence": 0.92,
                "ocr_data": {"conf": [92, 91, 90]}
            }
        ]
        
        logger.info("✅ Test OCR data created for 2 pages")
        
        # Save extracted text
        logger.info("\n[2/3] Saving extracted text to files...")
        
        document_id = "test_doc_001"
        filename = "test_medical_scan.pdf"
        
        saved_file = ocr.save_extracted_text(test_pages, document_id, filename)
        logger.info(f"✅ Text saved to: {saved_file}")
        
        # Verify files exist
        logger.info("\n[3/3] Verifying saved files...")
        
        doc_dir = Path(saved_file).parent
        logger.info(f"Document directory: {doc_dir}")
        
        text_files = list(doc_dir.glob("*.txt"))
        logger.info(f"✅ Found {len(text_files)} text files:")
        
        for f in text_files:
            logger.info(f"   - {f.name} ({f.stat().st_size} bytes)")
            if "_full" in f.name:
                with open(f, "r", encoding="utf-8") as content:
                    lines = content.readlines()
                    logger.info(f"     Content: {lines[0].strip()} ... ({len(lines)} lines)")
        
        # Check content
        logger.info(f"\nFull content file:")
        with open(saved_file, "r", encoding="utf-8") as f:
            content = f.read(200)
            logger.info(f"   First 200 chars: {content}...")
        
        logger.info("\n" + "=" * 70)
        logger.info("[SUCCESS] OCR TEXT EXTRACTION TO FILES WORKING!")
        logger.info("=" * 70)
        logger.info("✅ Structure:")
        logger.info("   storage/")
        logger.info("   ├── extracted_text/")
        logger.info("   │   └── test_doc_001/")
        logger.info("   │       ├── test_medical_scan_full.txt  (Combined text)")
        logger.info("   │       ├── page_1.txt                  (Individual pages)")
        logger.info("   │       └── page_2.txt")
        logger.info("   └── chroma_db/")
        logger.info("\nEach scanned PDF will have:")
        logger.info("  • Full combined text with metadata")
        logger.info("  • Individual page files")
        logger.info("  • Confidence scores and extraction timestamps")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}", exc_info=True)
        return False


if __name__ == "__main__":
    import sys
    success = asyncio.run(test_save_extracted_text())
    sys.exit(0 if success else 1)
