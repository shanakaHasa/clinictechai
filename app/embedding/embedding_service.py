"""
Embedding Service and Vector Store
Handles text embedding generation and vector storage management
Uses OpenAI's embedding API for consistency with LLM
"""

from typing import List, Dict, Tuple, Optional
import logging
import numpy as np
import os
import hashlib
import json

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    chromadb = None

from app.config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generates embeddings using OpenAI API"""
    
    def __init__(self, model_name: str = None, api_key: str = None):
        """
        Initialize embedding service with OpenAI API
        
        Args:
            model_name: Name of the embedding model to use
            api_key: OpenAI API key
        """
        self.model_name = model_name or settings.embedding_model
        self.embedding_dimension = settings.embedding_dimension
        self.api_key = api_key or settings.llm_api_key
        
        try:
            logger.info(f"Initializing OpenAI Embedding Service")
            logger.info(f"   Model: {self.model_name}")
            logger.info(f"   Dimension: {self.embedding_dimension}")
            
            if not self.api_key:
                logger.error("❌ No OpenAI API key provided")
                self.client = None
                return
            
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"✅ OpenAI client initialized for embeddings")
            
            # Test the client with a simple embedding
            self._validate_model()
            
        except Exception as e:
            logger.error(f"❌ Error initializing embedding service: {str(e)}")
            self.client = None
    
    def _validate_model(self):
        """Validate embedding model works with a test embedding"""
        try:
            logger.info("🔍 Validating embedding model...")
            test_response = self.client.embeddings.create(
                input="test",
                model=self.model_name
            )
            
            test_embedding = test_response.data[0].embedding
            logger.info(f"✅ Model validation successful")
            logger.info(f"   Test embedding dimension: {len(test_embedding)}")
            
            if len(test_embedding) != self.embedding_dimension:
                logger.warning(f"⚠️  Expected dimension {self.embedding_dimension}, got {len(test_embedding)}")
                self.embedding_dimension = len(test_embedding)
            
        except Exception as e:
            logger.error(f"❌ Error validating model: {str(e)}")
            raise
    
    def validate_model(self) -> bool:
        """Check if client is initialized and ready"""
        if not self.client:
            logger.error("❌ OpenAI client not initialized")
            return False
        logger.debug("✅ Embedding service is ready")
        return True
    
    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using OpenAI API
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            if not self.validate_model():
                logger.error("Cannot embed - client not ready")
                return []
            
            if not text or len(text.strip()) == 0:
                logger.warning("Empty text provided for embedding")
                return []
            
            logger.debug(f"Embedding text with OpenAI: {text[:100]}...")
            
            response = self.client.embeddings.create(
                input=text,
                model=self.model_name
            )
            
            embedding = response.data[0].embedding
            logger.debug(f"✅ Generated embedding of dimension: {len(embedding)}")
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Error embedding text: {str(e)}")
            return []
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using OpenAI API (batch)
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            if not self.validate_model():
                logger.error("Cannot embed - client not ready")
                return []
            
            if not texts:
                logger.warning("Empty text list provided")
                return []
            
            logger.info(f"Embedding {len(texts)} texts in batch mode with OpenAI...")
            
            # OpenAI API can handle multiple texts in one request
            response = self.client.embeddings.create(
                input=texts,
                model=self.model_name
            )
            
            # Extract embeddings in the same order as input
            embeddings = [data.embedding for data in response.data]
            
            logger.info(f"✅ Generated {len(embeddings)} embeddings from OpenAI API")
            logger.info(f"   Dimension per embedding: {len(embeddings[0]) if embeddings else 0}")
            logger.debug(f"   Usage - Tokens: {response.usage.total_tokens if hasattr(response, 'usage') else 'N/A'}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Error embedding texts batch: {str(e)}")
            return []


class VectorStore:
    """
    Manages vector storage and retrieval
    Supports ChromaDB backend (local storage)
    """
    
    def __init__(self, backend: str = None):
        """
        Initialize vector store
        
        Args:
            backend: Vector store backend (chroma)
        """
        self.backend = backend or settings.vector_db_type
        self.embedding_dimension = settings.embedding_dimension
        self.client = None
        self.collection = None
        
        logger.info(f"Initializing vector store with backend: {self.backend}")
        
        if self.backend == "chroma" or self.backend == "chroma_ephemeral" or self.backend == "chroma_persistent":
            self._init_chroma()
        else:
            logger.warning(f"⚠️  Unknown backend: {self.backend}")
    
    def _init_chroma(self):
        """Initialize ChromaDB client and collection with persistent storage"""
        try:
            if not chromadb:
                logger.error("[ERROR] chromadb not installed")
                return
            
            logger.info("[INFO] Initializing ChromaDB client with PERSISTENT storage...")
            
            # Create persistent database directory with absolute path
            import os
            from pathlib import Path
            
            # Use absolute path to avoid Windows path issues
            db_path = Path(settings.vector_db_path).absolute()
            logger.info(f"[INFO] Vector DB Path: {db_path}")
            
            # Create directory if it doesn't exist
            try:
                db_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"[INFO] Created/verified DB directory: {db_path}")
            except Exception as e:
                logger.warning(f"[WARNING] Could not create DB directory: {e}")
            
            try:
                # Try PersistentClient first
                logger.info("[INFO] Attempting PersistentClient with absolute path...")
                self.client = chromadb.PersistentClient(path=str(db_path))
                logger.info("[OK] ChromaDB PERSISTENT client initialized successfully")
                self.backend = "chroma_persistent"
            except Exception as e:
                logger.warning(f"[WARNING] PersistentClient failed: {str(e)}")
                logger.info("[FALLBACK] Falling back to EphemeralClient (RAM only)...")
                self.client = chromadb.EphemeralClient()
                logger.warning("[WARNING] Using EphemeralClient - data will NOT persist after restart!")
                self.backend = "chroma_ephemeral"
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="rag_documents",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("[OK] ChromaDB collection 'rag_documents' ready")
            
            # Log collection stats
            try:
                collection_count = self.collection.count()
                logger.info(f"[INFO] Collection contains {collection_count} vectors (Backend: {self.backend})")
            except:
                logger.info("[INFO] Collection count not available")
            
        except Exception as e:
            logger.error(f"[ERROR] Error initializing ChromaDB: {str(e)}")
            self.client = None
            self.collection = None
            self.backend = None
    
    def _compute_file_hash(self, file_path: str) -> str:
        """
        Compute SHA256 hash of a file
        
        Args:
            file_path: Path to file
            
        Returns:
            File hash string
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def check_document_exists(self, document_id: str) -> bool:
        """
        Check if document already exists in vector store
        
        Args:
            document_id: Document ID to check
            
        Returns:
            True if document exists, False otherwise
        """
        try:
            if not self.collection:
                return False
            
            # Get all documents with matching document_id prefix
            results = self.collection.get(
                where={"document_id": {"$eq": document_id}}
            )
            
            exists = len(results['ids']) > 0 if results else False
            
            if exists:
                logger.info(f"[WARNING] Document {document_id} already exists in vector store ({len(results['ids'])} chunks)")
            
            return exists
            
        except Exception as e:
            logger.error(f"[ERROR] Error checking document: {str(e)}")
            return False
    
    async def get_document_chunks(self, document_id: str) -> List[Dict]:
        """
        Retrieve all chunks for a specific document
        
        Args:
            document_id: Document ID
            
        Returns:
            List of chunk metadata dictionaries
        """
        try:
            if not self.collection:
                return []
            
            # Query for all chunks with this document_id
            results = self.collection.get(
                where={"document_id": {"$eq": document_id}}
            )
            
            chunks = []
            if results and 'metadatas' in results:
                for metadata in results['metadatas']:
                    chunks.append(metadata)
            
            logger.info(f"[OK] Retrieved {len(chunks)} chunks for document {document_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"[ERROR] Error retrieving chunks: {str(e)}")
            return []
    
    async def add_vectors(
        self,
        chunk_ids: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict]
    ) -> Dict:
        """
        Add vectors to the store
        
        Args:
            chunk_ids: List of unique chunk IDs
            embeddings: List of embedding vectors
            metadata: List of metadata dictionaries for each chunk
            
        Returns:
            Status dictionary
        """
        try:
            # Validation
            logger.info(f"[ADD_VECTORS] Adding vectors: {len(chunk_ids)} chunks, {len(embeddings)} embeddings, {len(metadata)} metadata")
            
            if len(chunk_ids) == 0:
                logger.warning("[ADD_VECTORS] No chunks to add")
                return {"success": True, "added": 0, "backend": self.backend}
            
            if len(chunk_ids) != len(embeddings):
                logger.error(f"[ADD_VECTORS ERROR] Chunk IDs ({len(chunk_ids)}) != Embeddings ({len(embeddings)})")
                raise ValueError(f"Chunk IDs count ({len(chunk_ids)}) != Embeddings count ({len(embeddings)})")
            
            if len(embeddings) != len(metadata):
                logger.error(f"[ADD_VECTORS ERROR] Embeddings ({len(embeddings)}) != Metadata ({len(metadata)})")
                raise ValueError(f"Embeddings count ({len(embeddings)}) != Metadata count ({len(metadata)})")
            
            # Validate embeddings
            for i, emb in enumerate(embeddings):
                if not isinstance(emb, list) or len(emb) == 0:
                    logger.error(f"[ADD_VECTORS ERROR] Invalid embedding at index {i}")
                    raise ValueError(f"Invalid embedding at index {i}")
            
            logger.debug(f"[ADD_VECTORS DEBUG] First embedding dimension: {len(embeddings[0])}")
            logger.debug(f"[ADD_VECTORS DEBUG] Metadata sample: {metadata[0] if metadata else 'none'}")
            logger.info(f"[ADD_VECTORS] Validation passed. Storing {len(chunk_ids)} vectors...")
            
            if (self.backend == "chroma" or self.backend == "chroma_ephemeral" or self.backend == "chroma_persistent") and self.collection:
                # Prepare documents and metadatas for ChromaDB
                documents = [meta.get("text", "") for meta in metadata]
                
                logger.debug(f"[ADD_VECTORS DEBUG] About to UPSERT to ChromaDB ({self.backend})")
                logger.debug(f"[ADD_VECTORS DEBUG] Chunk IDs: {chunk_ids[:3] + ['...'] if len(chunk_ids) > 3 else chunk_ids}")
                
                self.collection.upsert(
                    ids=chunk_ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadata
                )
                
                # Verify insertion
                try:
                    collection_count = self.collection.count()
                    logger.info(f"[ADD_VECTORS SUCCESS] Added {len(chunk_ids)} vectors. Collection now has {collection_count} total vectors")
                except Exception as e:
                    logger.debug(f"[ADD_VECTORS] Could not get final count: {e}")
                    logger.info(f"[ADD_VECTORS SUCCESS] Added {len(chunk_ids)} vectors to ChromaDB")
            else:
                logger.warning(f"[ADD_VECTORS WARNING] No backend implementation for {self.backend}")
            
            return {
                "success": True,
                "added": len(chunk_ids),
                "backend": self.backend,
                "dimension": len(embeddings[0]) if embeddings else 0
            }
            
        except Exception as e:
            logger.error(f"[ADD_VECTORS EXCEPTION] Error adding vectors: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e), "backend": self.backend}
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = None,
        filters: Dict = None
    ) -> List[Tuple[str, float, Dict]]:
        """
        Search for similar vectors in the store
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            filters: Filter criteria for metadata
            
        Returns:
            List of tuples (chunk_id, similarity_score, metadata)
        """
        try:
            top_k = top_k or settings.top_k_results
            
            logger.debug(f"[SEARCH DEBUG] Starting search with top_k={top_k}")
            logger.debug(f"[SEARCH DEBUG] Query embedding dimension: {len(query_embedding) if query_embedding else 0}")
            
            if not query_embedding or len(query_embedding) == 0:
                logger.warning("[SEARCH ERROR] Empty query embedding")
                return []
            
            logger.debug(f"[SEARCH DEBUG] Backend: {self.backend}")
            logger.debug(f"[SEARCH DEBUG] Collection available: {self.collection is not None}")
            
            if (self.backend == "chroma" or self.backend == "chroma_ephemeral" or self.backend == "chroma_persistent") and self.collection:
                # Check collection count before search
                try:
                    collection_count = self.collection.count()
                    logger.debug(f"[SEARCH DEBUG] Collection has {collection_count} vectors total")
                except Exception as e:
                    logger.debug(f"[SEARCH DEBUG] Could not get collection count: {e}")
                
                logger.debug(f"[SEARCH DEBUG] Executing ChromaDB query...")
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k
                )
                
                logger.debug(f"[SEARCH DEBUG] Query returned results structure: ids={results.get('ids')}, distances={results.get('distances')}")
                
                result_count = len(results['ids'][0]) if results['ids'] and len(results['ids']) > 0 else 0
                logger.info(f"[SEARCH RESULT] Found {result_count} results from ChromaDB")
                
                if result_count == 0:
                    logger.debug("[SEARCH DEBUG] No results found - returning empty list")
                    return []
                
                # Format results
                formatted_results = []
                for i, chunk_id in enumerate(results['ids'][0]):
                    similarity = results['distances'][0][i] if results['distances'] else 0
                    meta = results['metadatas'][0][i] if results['metadatas'] else {}
                    formatted_results.append((chunk_id, float(similarity), meta))
                    logger.debug(f"[SEARCH DEBUG] Result {i+1}: chunk_id={chunk_id}, distance={similarity}, metadata_keys={list(meta.keys())}")
                
                logger.info(f"[SEARCH SUCCESS] Returning {len(formatted_results)} formatted results")
                return formatted_results
            else:
                logger.warning(f"[SEARCH ERROR] Backend not available: {self.backend}, collection={self.collection is not None}")
                return []
            
        except Exception as e:
            logger.error(f"[SEARCH EXCEPTION] Error searching vectors: {str(e)}", exc_info=True)
            return []
    
    async def delete_vectors(self, chunk_ids: List[str]) -> Dict:
        """
        Delete vectors from the store
        
        Args:
            chunk_ids: List of chunk IDs to delete
            
        Returns:
            Status dictionary
        """
        try:
            logger.info(f"Deleting {len(chunk_ids)} vectors from {self.backend}")
            
            if (self.backend == "chroma" or self.backend == "chroma_ephemeral" or self.backend == "chroma_persistent") and self.collection:
                self.collection.delete(ids=chunk_ids)
                logger.info(f"✅ Deleted {len(chunk_ids)} vectors from ChromaDB")
            
            return {
                "success": True,
                "deleted": len(chunk_ids),
                "backend": self.backend
            }
            
        except Exception as e:
            logger.error(f"❌ Error deleting vectors: {str(e)}")
            return {"success": False, "error": str(e)}
