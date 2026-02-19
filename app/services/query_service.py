"""
Query Service Layer
Handles RAG query workflow with 5-step processing pipeline
Includes chat memory management for multi-turn conversations
"""

import logging
from typing import Dict, List, Optional
from fastapi import HTTPException

from app.embedding.embedding_service import EmbeddingService, VectorStore
from app.reranking.reranker import QueryReranker
from app.llm.llm_service import LLMService
from app.verification.verifier import AnswerVerifier
from app.services.chat_memory import ChatMemory, memory_manager
from app.safety.content_moderator import ContentModerator
from app.schemas.models import SourceEvidence

logger = logging.getLogger(__name__)


class QueryService:
    """Service for RAG query processing"""
    
    def __init__(self):
        """Initialize query service with dependencies"""
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.reranker = QueryReranker()
        self.llm_service = LLMService()
        self.answer_verifier = AnswerVerifier()
        self.content_moderator = ContentModerator()
    
    async def process_query(self, user_query: str) -> Dict:
        """
        Process user query with 5-step RAG pipeline
        
        Args:
            user_query: User's question
            
        Returns:
            Dictionary with RAG response
        """
        return await self.process_query_with_history(user_query, session_id=None)
    
    async def process_query_with_history(
        self,
        user_query: str,
        session_id: Optional[str] = None
    ) -> Dict:
        """
        Process user query with chat history support
        
        Args:
            user_query: User's question
            session_id: Conversation session ID (for chat history)
            
        Returns:
            Dictionary with RAG response
        """
        logger.info("=" * 60)
        logger.info("[QUERY] Starting RAG query processing")
        logger.info(f"   User Query: '{user_query}'")
        logger.info(f"   Query Length: {len(user_query)} characters")
        if session_id:
            logger.info(f"   Session ID: {session_id}")
        
        # Step 0: Check input for harmful content (Content Moderation)
        logger.info("[0/5] Checking input for policy violations...")
        is_safe_input, mod_result = await self.content_moderator.check_input(user_query)
        
        if not is_safe_input:
            logger.warning("[MODERATION] ❌ Input rejected due to policy violation")
            violation_message = self.content_moderator.get_violation_message(mod_result, stage="input")
            return {
                "query": user_query,
                "answer": violation_message,
                "evidence": [],
                "confidence_score": 0.0,
                "tokens_used": 0,
                "moderation_flagged": True,
                "moderation_reason": "input_policy_violation"
            }
        
        logger.info("[MODERATION] ✅ Input passed safety check")
        
        # Initialize or get chat memory if session_id provided
        chat_memory: Optional[ChatMemory] = None
        if session_id:
            chat_memory = memory_manager.get_or_create_session(session_id)
            chat_memory.add_user_message(user_query)
            logger.info(f"[CHAT HISTORY] Added user message to session. Total messages: {len(chat_memory.messages)}")
        
        try:
            # Step 1: Generate query embedding
            query_embedding = await self._step_embed_query(user_query)
            
            # Step 2: Retrieve relevant chunks
            retrieved_chunks = await self._step_retrieve(user_query, query_embedding)
            
            if len(retrieved_chunks) == 0:
                logger.warning("   [WARNING] No relevant chunks found")
                response = {
                    "query": user_query,
                    "answer": "I could not find any relevant information in the documents to answer your question.",
                    "evidence": [],
                    "confidence_score": 0.0,
                    "tokens_used": 0
                }
                
                if chat_memory:
                    chat_memory.add_assistant_message(response["answer"])
                    memory_manager.save_session(session_id)
                
                return response
            
            # Step 3: Rerank results
            reranked_chunks = await self._step_rerank(user_query, retrieved_chunks)
            
            # Step 4: Generate grounded answer (with optional chat history)
            chat_history_text = None
            if chat_memory and len(chat_memory.messages) > 1:
                chat_history_text = chat_memory.get_context_for_llm(last_n=4)
                logger.info(f"[CHAT HISTORY] Including {len(chat_memory.messages) - 1} previous messages in context")
            
            rag_result = await self._step_generate_answer(
                user_query,
                reranked_chunks,
                chat_history=chat_history_text
            )
            
            # Step 5: Verify answer
            verified_result = await self._step_verify_answer(user_query, rag_result)
            
            logger.info("=" * 60)
            logger.info("[SUCCESS] RAG query processed successfully")
            logger.info("=" * 60)
            
            # Extract page numbers from evidence
            page_numbers = list(set([chunk.get("page_number") for chunk in rag_result.get("context_chunks", [])]))
            
            # Build sources from evidence
            sources = []
            for chunk in rag_result.get("context_chunks", []):
                sources.append({
                    "document": chunk.get("source_document", "Unknown"),
                    "page_number": chunk.get("page_number", 0),
                    "chunk_id": chunk.get("chunk_id", "")
                })
            
            final_result = {
                **verified_result,
                "sources": sources,
                "page_numbers": sorted(page_numbers),
                "context_used": len(rag_result.get("context_chunks", []))
            }
            
            # Save to chat memory if session exists
            if chat_memory:
                chat_memory.add_assistant_message(
                    verified_result.get("answer", ""),
                    metadata={
                        "documents_used": [s["document"] for s in sources],
                        "confidence_score": verified_result.get("confidence_score", 0)
                    }
                )
                memory_manager.save_session(session_id)
                logger.info(f"[CHAT HISTORY] Saved response to session {session_id}")
            
            return final_result
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("=" * 60)
            logger.error(f"[ERROR] Error processing query: {str(e)}")
            logger.error("=" * 60)
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _step_embed_query(self, user_query: str) -> List[float]:
        """Step 1: Generate query embedding"""
        logger.info("[1/5] Generating query embedding...")
        
        try:
            query_embedding = await self.embedding_service.embed_text(user_query)
            logger.info(f"   [OK] Query embedded: {len(query_embedding)} dimensions")
            logger.debug(f"   Embedding preview: {query_embedding[:3]}...")
            return query_embedding
        except Exception as e:
            logger.error(f"   [ERROR] Failed to embed query: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")
    
    async def _step_retrieve(self, user_query: str, query_embedding: List[float]) -> List[Dict]:
        """Step 2: Retrieve relevant chunks from vector store"""
        logger.info("[2/5] Retrieving relevant chunks...")
        logger.debug(f"[RETRIEVE DEBUG] Query embedding size: {len(query_embedding)}")
        logger.debug(f"[RETRIEVE DEBUG] Vector store backend: {self.vector_store.backend if hasattr(self.vector_store, 'backend') else 'unknown'}")
        
        try:
            # Check collection before search
            if hasattr(self.vector_store, 'collection') and self.vector_store.collection:
                try:
                    collection_count = self.vector_store.collection.count()
                    logger.debug(f"[RETRIEVE DEBUG] Collection has {collection_count} total vectors")
                except Exception as e:
                    logger.debug(f"[RETRIEVE DEBUG] Could not get collection count: {e}")
            
            logger.debug(f"[RETRIEVE DEBUG] Starting search with top_k=10...")
            # Search in ChromaDB
            search_results = await self.vector_store.search(query_embedding, top_k=10)
            
            logger.debug(f"[RETRIEVE DEBUG] Search returned {len(search_results) if search_results else 0} results")
            
            if not search_results or len(search_results) == 0:
                logger.warning("[RETRIEVE WARNING] Search returned no results or empty list")
                return []
            
            logger.info(f"   [OK] Retrieved {len(search_results)} chunks")
            
            # Extract and format results
            # Results are tuples: (chunk_id, similarity_score, metadata)
            retrieved_chunks = []
            for i, (chunk_id, similarity, metadata) in enumerate(search_results):
                chunk_data = {
                    "chunk_id": chunk_id,
                    "text": metadata.get("text", ""),
                    "page_number": metadata.get("page_number", 0),
                    "document_id": metadata.get("document_id", ""),
                    "source_document": metadata.get("source_document", ""),
                    "distance": similarity,
                    "bbox": metadata.get("bbox", None)
                }
                retrieved_chunks.append(chunk_data)
                logger.debug(f"   [{i+1}] Chunk: {chunk_id} (Distance: {similarity:.4f}, Page: {chunk_data['page_number']}, Doc: {chunk_data['document_id']})")
            
            logger.info(f"[RETRIEVE SUCCESS] Formatted {len(retrieved_chunks)} chunks for processing")
            return retrieved_chunks
        except Exception as e:
            logger.error(f"   [ERROR] Failed to retrieve chunks: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")
    
    async def _step_rerank(self, user_query: str, retrieved_chunks: List[Dict]) -> List[Dict]:
        """Step 3: Rerank retrieved chunks"""
        logger.info("[3/5] Reranking results...")
        
        try:
            chunk_texts = [chunk["text"] for chunk in retrieved_chunks]
            rerank_scores = await self.reranker.rerank(user_query, chunk_texts)
            
            if not rerank_scores or len(rerank_scores) == 0:
                logger.warning("   [WARNING] Reranking returned no scores")
                return retrieved_chunks
            
            logger.info(f"   [OK] Reranked {len(rerank_scores)} results")
            
            # Add rerank scores to chunks
            for i, chunk in enumerate(retrieved_chunks):
                if i < len(rerank_scores):
                    chunk["rerank_score"] = rerank_scores[i]
                    logger.debug(f"   [{i+1}] Rerank score: {rerank_scores[i]:.4f}")
            
            # Sort by rerank score (descending)
            reranked = sorted(retrieved_chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)
            
            # Keep top 5
            top_chunks = reranked[:5]
            logger.info(f"   [OK] Selected top {len(top_chunks)} chunks for answer generation")
            
            return top_chunks
        except Exception as e:
            logger.error(f"   [ERROR] Failed to rerank: {str(e)}")
            # Continue with original order if reranking fails
            logger.warning("   [WARNING] Proceeding with original retrieval order")
            return retrieved_chunks[:5]
    
    async def _step_generate_answer(
        self,
        user_query: str,
        context_chunks: List[Dict],
        chat_history: Optional[str] = None
    ) -> Dict:
        """Step 4: Generate grounded answer using LLM"""
        logger.info("[4/5] Generating grounded answer...")
        
        try:
            # Generate answer with LLM
            rag_result = await self.llm_service.generate_grounded_answer(
                user_query,
                context_chunks,
                chat_history=chat_history
            )
            
            logger.info(f"   [OK] Answer generated")
            logger.debug(f"   Answer length: {len(rag_result.get('answer', ''))} characters")
            
            # Check output for harmful content (Content Moderation)
            logger.info("[OUTPUT MODERATION] Checking answer for policy violations...")
            answer_text = rag_result.get('answer', '')
            is_safe_output, mod_result = await self.content_moderator.check_output(answer_text)
            
            if not is_safe_output:
                logger.warning("[MODERATION] ❌ Output flagged for policy violation")
                violation_message = self.content_moderator.get_violation_message(mod_result, stage="output")
                rag_result["answer"] = violation_message
                rag_result["moderation_flagged"] = True
                rag_result["moderation_reason"] = "output_policy_violation"
            else:
                logger.info("[OUTPUT MODERATION] ✅ Answer passed safety check")
                rag_result["moderation_flagged"] = False
            
            # Add context chunks to result
            rag_result["context_chunks"] = context_chunks
            
            return rag_result
        except Exception as e:
            logger.error(f"   [ERROR] Failed to generate answer: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Answer generation failed: {str(e)}")
    
    async def _step_verify_answer(self, user_query: str, rag_result: Dict) -> Dict:
        """Step 5: Verify answer against context"""
        logger.info("[5/5] Verifying answer...")
        
        try:
            context_chunks = rag_result.get("context_chunks", [])
            answer = rag_result.get("answer", "")
            
            verification = await self.answer_verifier.verify_answer(
                answer,
                context_chunks,
                user_query
            )
            
            logger.info(f"   [OK] Verification complete")
            logger.debug(f"   Is grounded: {verification.get('is_grounded', False)}")
            logger.debug(f"   Confidence: {verification.get('confidence_score', 0):.2f}")
            
            # Build evidence
            evidence = []
            for chunk in context_chunks:
                evidence.append(SourceEvidence(
                    page_number=chunk["page_number"],
                    document=chunk["source_document"],
                    exact_chunk=chunk["text"],
                    bbox=chunk.get("bbox"),
                    chunk_id=chunk["chunk_id"],
                    highlighted=chunk["text"][:100] + "..." if len(chunk["text"]) > 100 else chunk["text"]
                ))
            
            return {
                "query": user_query,
                "answer": answer,
                "evidence": evidence,
                "confidence_score": verification.get("confidence_score", 0),
                "tokens_used": rag_result.get("tokens_used", 0)
            }
        except Exception as e:
            logger.error(f"   [ERROR] Failed to verify answer: {str(e)}")
            # Return answer even if verification fails
            logger.warning("   [WARNING] Returning unverified answer")
            
            evidence = []
            for chunk in rag_result.get("context_chunks", []):
                evidence.append(SourceEvidence(
                    page_number=chunk["page_number"],
                    document=chunk["source_document"],
                    exact_chunk=chunk["text"],
                    bbox=chunk.get("bbox"),
                    chunk_id=chunk["chunk_id"],
                    highlighted=chunk["text"][:100] + "..." if len(chunk["text"]) > 100 else chunk["text"]
                ))
            
            return {
                "query": user_query,
                "answer": rag_result.get("answer", ""),
                "evidence": evidence,
                "confidence_score": 0.0,
                "tokens_used": rag_result.get("tokens_used", 0)
            }
