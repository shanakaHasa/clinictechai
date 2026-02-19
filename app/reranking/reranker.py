"""
Query Reranker Module
Wrapper for cross-encoder reranking from retriever
"""

from typing import List, Dict
import logging

from app.retrieval.retriever import Reranker as BaseReranker

logger = logging.getLogger(__name__)


class QueryReranker:
    """Reranks query results using cross-encoder models"""
    
    def __init__(self):
        """Initialize query reranker"""
        self._reranker = BaseReranker()
    
    async def rerank(self, query: str, texts: List[str], top_k: int = 5) -> List[float]:
        """
        Rerank texts based on relevance to query
        
        Args:
            query: User query
            texts: List of texts to rerank
            top_k: Number of top results
            
        Returns:
            List of relevance scores
        """
        try:
            if not self._reranker.model or not texts:
                # Return equal scores if model not available
                return [0.5] * len(texts)
            
            # Create query-text pairs
            pairs = [[query, text] for text in texts]
            
            # Get cross-encoder scores
            scores = self._reranker.model.predict(pairs)
            
            logger.debug(f"Reranked {len(texts)} texts - Top score: {max(scores):.4f}")
            return list(scores)
            
        except Exception as e:
            logger.error(f"Error in reranking: {str(e)}")
            # Return neutral scores if reranking fails
            return [0.5] * len(texts)
