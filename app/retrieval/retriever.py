"""
Retrieval and Reranking Module
Handles vector similarity search and cross-encoder reranking
"""

from typing import List, Dict, Tuple
import logging

from sentence_transformers import CrossEncoder, util
from app.config.settings import settings
from app.embedding.embedding_service import EmbeddingService, VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Handles document retrieval from vector store"""
    
    def __init__(self):
        """Initialize retriever"""
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.top_k = settings.top_k_results
        self.similarity_threshold = settings.similarity_threshold
    
    async def retrieve(
        self,
        query: str,
        top_k: int = None,
        filters: Dict = None
    ) -> List[Dict]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: User query
            top_k: Number of results to retrieve
            filters: Metadata filters
            
        Returns:
            List of relevant chunks with metadata
        """
        try:
            top_k = top_k or self.top_k
            
            # Generate query embedding
            query_embedding = await self.embedding_service.embed_text(query)
            
            if not query_embedding:
                logger.warning("Failed to generate query embedding")
                return []
            
            # Search vector store
            results = await self.vector_store.search(
                query_embedding,
                top_k=top_k,
                filters=filters
            )
            
            # Format results
            retrieved_docs = []
            for chunk_id, score, metadata in results:
                if score >= self.similarity_threshold:
                    retrieved_docs.append({
                        "chunk_id": chunk_id,
                        "similarity_score": float(score),
                        **metadata
                    })
            
            logger.info(f"Retrieved {len(retrieved_docs)} relevant chunks")
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Error in retrieval: {str(e)}")
            return []


class Reranker:
    """Reranks retrieved chunks using cross-encoder models"""
    
    def __init__(self, model_name: str = None):
        """
        Initialize reranker
        
        Args:
            model_name: Cross-encoder model name
        """
        self.model_name = model_name or settings.rerank_model
        
        try:
            self.model = CrossEncoder(self.model_name)
            logger.info(f"Loaded reranker model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading reranker model: {str(e)}")
            self.model = None
    
    async def rerank_chunks(
        self,
        query: str,
        chunks: List[Dict],
        top_k: int = None
    ) -> List[Dict]:
        """
        Rerank chunks using cross-encoder
        
        Args:
            query: Original query
            chunks: List of retrieved chunks
            top_k: Number of top chunks to return after reranking
            
        Returns:
            Reranked list of chunks
        """
        try:
            if not self.model or not chunks:
                return chunks
            
            top_k = top_k or settings.top_k_results
            
            # Prepare pairs for reranking
            pairs = [[query, chunk.get("text", "")] for chunk in chunks]
            
            # Get cross-encoder scores
            scores = self.model.predict(pairs)
            
            # Sort by relevance score
            scored_chunks = [
                {**chunk, "rerank_score": float(score)}
                for chunk, score in zip(chunks, scores)
            ]
            
            reranked = sorted(
                scored_chunks,
                key=lambda x: x["rerank_score"],
                reverse=True
            )[:top_k]
            
            logger.info(f"Reranked {len(chunks)} chunks to top {len(reranked)}")
            return reranked
            
        except Exception as e:
            logger.error(f"Error in reranking: {str(e)}")
            return chunks[:top_k] if top_k else chunks
