"""
Verify Embedding and Vector Storage Pipeline
Tests the complete flow: chunks -> embeddings -> vector storage for scanned PDFs
"""

import asyncio
import logging
from pathlib import Path
from app.embedding.embedding_service import EmbeddingService, VectorStore
from app.chunking.chunker import Chunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def verify_embedding_pipeline():
    """Verify the complete embedding and storage pipeline"""
    logger.info("=" * 70)
    logger.info("[VERIFY EMBEDDING PIPELINE]")
    logger.info("=" * 70)
    
    try:
        # Initialize services
        logger.info("\n[1/5] Initializing services...")
        embedding_service = EmbeddingService()
        vector_store = VectorStore()
        logger.info("✅ Services initialized")
        
        # Verify embedding service
        logger.info("\n[2/5] Verifying embedding service...")
        if not embedding_service.validate_model():
            logger.error("❌ Embedding service not ready")
            return False
        logger.info("✅ Embedding service ready")
        
        # Test with sample text
        logger.info("\n[3/5] Testing single text embedding...")
        test_texts = [
            "The patient presents with symptoms of hypertension and diabetes.",
            "Treatment includes medication and lifestyle changes.",
            "Follow-up appointment scheduled for next month."
        ]
        
        embeddings = await embedding_service.embed_texts(test_texts)
        
        if not embeddings:
            logger.error("❌ Failed to generate embeddings")
            return False
        
        logger.info(f"✅ Generated {len(embeddings)} embeddings")
        logger.info(f"   Embedding dimension: {len(embeddings[0])}")
        
        # Verify embedding dimensions
        for i, emb in enumerate(embeddings):
            if len(emb) != 1536:
                logger.error(f"❌ Embedding {i} has wrong dimension: {len(emb)} (expected 1536)")
                return False
        logger.info("✅ All embeddings have correct dimension (1536)")
        
        # Test vector storage
        logger.info("\n[4/5] Testing vector storage...")
        
        # Create sample chunks (simulate scanned PDF chunks)
        chunk_ids = [f"test_chunk_{i}" for i in range(len(test_texts))]
        metadata = [
            {
                "text": text,
                "document_id": "test_scanned_doc_001",
                "page_number": i,
                "chunk_index": 0,
                "extraction_type": "ocr",
                "source_document": "test_scanned.pdf"
            }
            for i, text in enumerate(test_texts)
        ]
        
        # Store vectors
        store_result = await vector_store.add_vectors(chunk_ids, embeddings, metadata)
        
        if not store_result.get("success"):
            logger.error(f"❌ Failed to store vectors: {store_result.get('error')}")
            return False
        
        logger.info(f"✅ Vectors stored successfully")
        logger.info(f"   Added: {store_result.get('added')} vectors")
        logger.info(f"   Backend: {store_result.get('backend')}")
        logger.info(f"   Dimension: {store_result.get('dimension')}")
        
        # Test search
        logger.info("\n[5/5] Testing vector search/retrieval...")
        
        query = "What are the patient's symptoms?"
        query_embedding = await embedding_service.embed_text(query)
        
        if not query_embedding:
            logger.error("❌ Failed to generate query embedding")
            return False
        
        results = await vector_store.search(query_embedding, top_k=2)
        
        if not results:
            logger.error("❌ Search returned no results")
            return False
        
        logger.info(f"✅ Search returned {len(results)} results")
        for i, (chunk_id, similarity, metadata) in enumerate(results):
            logger.info(f"   Result {i+1}:")
            logger.info(f"      Chunk ID: {chunk_id}")
            logger.info(f"      Similarity: {similarity:.4f}")
            logger.info(f"      Text: {metadata.get('text', 'N/A')[:60]}...")
            logger.info(f"      Extraction Type: {metadata.get('extraction_type', 'N/A')}")
        
        logger.info("\n" + "=" * 70)
        logger.info("[SUCCESS] Full embedding and storage pipeline verified!")
        logger.info("=" * 70)
        
        return True
        
    except Exception as e:
        logger.error("=" * 70)
        logger.error(f"[ERROR] Verification failed: {str(e)}", exc_info=True)
        logger.error("=" * 70)
        return False


async def check_existing_vectors():
    """Check how many vectors exist in the database"""
    logger.info("\n" + "=" * 70)
    logger.info("[CHECK EXISTING VECTORS]")
    logger.info("=" * 70)
    
    try:
        vector_store = VectorStore()
        
        if not vector_store.collection:
            logger.warning("❌ Vector store collection not available")
            return 0
        
        count = vector_store.collection.count()
        logger.info(f"✅ Total vectors in database: {count}")
        
        return count
        
    except Exception as e:
        logger.error(f"❌ Error checking vectors: {str(e)}")
        return 0


if __name__ == "__main__":
    import sys
    
    # First check existing vectors
    existing_count = asyncio.run(check_existing_vectors())
    
    # Run verification
    success = asyncio.run(verify_embedding_pipeline())
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
