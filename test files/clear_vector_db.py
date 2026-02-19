"""
Clear Vector Database Script
Removes all stored vectors and resets ChromaDB
"""

import logging
import shutil
from pathlib import Path
from app.config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_vector_db():
    """Clear the vector database completely"""
    try:
        db_path = Path(settings.vector_db_path).absolute()
        
        logger.info("=" * 60)
        logger.info("[CLEARING VECTOR DB]")
        logger.info(f"Vector DB Path: {db_path}")
        
        if db_path.exists():
            logger.info(f"Found vector DB directory at: {db_path}")
            logger.info("Removing entire directory...")
            shutil.rmtree(db_path)
            logger.info(f"✅ Deleted vector DB directory: {db_path}")
        else:
            logger.warning(f"Vector DB directory not found at: {db_path}")
        
        # Create empty directory again
        db_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Created fresh vector DB directory: {db_path}")
        
        logger.info("=" * 60)
        logger.info("[SUCCESS] Vector DB cleared successfully!")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"[ERROR] Failed to clear vector DB: {str(e)}")
        logger.error("=" * 60)
        return False

if __name__ == "__main__":
    clear_vector_db()
