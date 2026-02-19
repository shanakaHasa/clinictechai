"""
LLM Service - Grounded Answer Generation
Generates answers strictly grounded in retrieved context
"""

from typing import List, Dict, Optional
import logging

from app.config.settings import settings
from app.llm.prompts import prompt_manager

logger = logging.getLogger(__name__)


class LLMService:
    """
    Handles LLM interaction in strict grounded mode
    Ensures answers are based only on provided context
    """
    
    def __init__(self):
        """Initialize LLM service"""
        self.provider = settings.llm_provider
        self.model = settings.llm_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize LLM client based on provider"""
        try:
            if self.provider == "openai":
                from openai import OpenAI
                self.client = OpenAI(api_key=settings.llm_api_key)
                logger.info(f"✅ Initialized OpenAI LLM client (v1.0.0+)")
            elif self.provider == "anthropic":
                import anthropic
                self.client = anthropic.Anthropic(api_key=settings.llm_api_key)
                logger.info(f"✅ Initialized Anthropic LLM client")
            else:
                logger.warning(f"Unknown LLM provider: {self.provider}")
                self.client = None
                
            logger.info(f"   Provider: {self.provider}")
            logger.info(f"   Model: {self.model}")
            
        except Exception as e:
            logger.error(f"❌ Error initializing LLM client: {str(e)}")
            self.client = None
    
    async def generate_grounded_answer(
        self,
        query: str,
        context_chunks: List[Dict],
        chat_history: Optional[str] = None
    ) -> Dict:
        """
        Generate answer strictly grounded in provided context
        
        Args:
            query: User query
            context_chunks: Retrieved and reranked context chunks
            
        Returns:
            Dictionary with answer and supporting information
        """
        try:
            logger.info("=" * 60)
            logger.info("🔍 LLM SERVICE: Starting answer generation")
            logger.info(f"   Query: {query[:100]}...")
            logger.info(f"   Context chunks available: {len(context_chunks)}")
            
            if not self.client:
                logger.error("❌ LLM client not initialized")
                return self._error_response("LLM client not initialized")
            
            logger.debug(f"   LLM Provider: {self.provider}")
            logger.debug(f"   Model: {self.model}")
            logger.debug(f"   Temperature: {self.temperature}")
            logger.debug(f"   Max tokens: {self.max_tokens}")
            
            # Build context string from chunks
            logger.info("📋 Building context from chunks...")
            context_text = self._build_context(context_chunks)
            logger.debug(f"   Context length: {len(context_text)} characters")
            
            if len(context_text) < 50:
                logger.warning(f"⚠️  Context is very short ({len(context_text)} chars) - may not have enough information")
                logger.debug(f"   Context preview: {context_text[:200]}")
            else:
                logger.debug(f"   Context preview (first 300 chars): {context_text[:300]}...")
            
            # Create grounding prompt (with or without chat history)
            logger.info("✍️  Creating grounding prompt...")
            if chat_history:
                logger.debug("[CHAT HISTORY] Including conversation history in prompt")
                prompt = self._create_chat_prompt(query, context_text, chat_history)
            else:
                prompt = self._create_grounding_prompt(query, context_text)
            logger.debug(f"   Prompt length: {len(prompt)} characters")
            
            # Generate answer
            logger.info(f"🤖 Calling {self.provider.upper()} API for answer generation...")
            if self.provider == "openai":
                response = await self._call_openai(prompt)
            elif self.provider == "anthropic":
                response = await self._call_anthropic(prompt)
            else:
                logger.error(f"❌ Unsupported LLM provider: {self.provider}")
                return self._error_response("Unsupported LLM provider")
            
            logger.info(f"✅ LLM Response received: {len(response)} characters")
            logger.debug(f"   Response preview: {response[:200]}...")
            
            # Extract source information
            logger.info("📚 Extracting source information...")
            sources = self._extract_sources(context_chunks)
            logger.info(f"   Sources extracted: {len(sources)}")
            
            result = {
                "success": True,
                "answer": response,
                "sources": sources,
                "context_used": len(context_chunks),
                "model": self.model
            }
            
            logger.info("✅ Answer generation complete")
            logger.info("=" * 60)
            
            return result
            
        except Exception as e:
            logger.error("=" * 60)
            logger.error(f"❌ Error generating answer: {str(e)}")
            logger.error("=" * 60)
            return self._error_response(str(e))
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """Build context string from chunks with clear separation"""
        context_parts = []
        
        logger.debug(f"[BUILD_CONTEXT] Processing {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks, 1):
            page_num = chunk.get('page_number', 'N/A')
            doc_name = chunk.get('source_document', 'Unknown')
            text = chunk.get("text", "")
            
            logger.debug(f"[BUILD_CONTEXT] Chunk {i}: doc={doc_name}, page={page_num}, text_len={len(text)}")
            
            if text.strip():
                context_parts.append(f"\n[Source {i}: {doc_name} (Page {page_num})]")
                context_parts.append(text)
                logger.debug(f"[BUILD_CONTEXT]   ✓ Added {len(text)} chars from chunk {i}")
            else:
                logger.debug(f"[BUILD_CONTEXT]   ⚠️  Skipped chunk {i} - no text content")
        
        if not context_parts:
            logger.warning("[BUILD_CONTEXT] No text content found in any chunks!")
        else:
            logger.debug(f"[BUILD_CONTEXT] Successfully built context from {len(context_parts)//2} chunks")
        
        return "\n---\n".join(context_parts)
    
    def _create_grounding_prompt(self, query: str, context: str) -> str:
        """
        Create a prompt that enforces grounding in context and extracts specific medical values
        Uses centralized prompt management from prompts.py
        """
        return prompt_manager.format_medical_extraction_prompt(query, context)
    
    def _create_chat_prompt(self, query: str, context: str, chat_history: str) -> str:
        """
        Create a prompt that includes conversation history
        """
        return prompt_manager.format_chat_with_history_prompt(query, context, chat_history)
    
    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API using v1.0.0+ client"""
        try:
            logger.debug("🌐 Preparing OpenAI API call...")
            logger.debug(f"   Prompt preview: {prompt[:150]}...")
            
            # Use proper system prompt from prompt manager
            system_prompt = prompt_manager.get_system_prompt("medical_assistant")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            logger.debug(f"✅ OpenAI API call successful")
            logger.debug(f"   Response length: {len(response.choices[0].message.content)}")
            logger.debug(f"   Tokens used: {response.usage.total_tokens if hasattr(response, 'usage') else 'N/A'}")
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"❌ Error calling OpenAI: {str(e)}")
            raise
    
    async def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic Claude API"""
        try:
            # Get proper system prompt
            system_prompt = prompt_manager.get_system_prompt("medical_assistant")
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Error calling Anthropic: {str(e)}")
            raise
    
    def _extract_sources(self, chunks: List[Dict]) -> List[Dict]:
        """Extract source information from chunks"""
        sources = []
        
        for chunk in chunks:
            source = {
                "document": chunk.get("source_document", "Unknown"),
                "page_number": chunk.get("page_number", "N/A"),
                "chunk_id": chunk.get("chunk_id", ""),
                "similarity_score": chunk.get("similarity_score", 0),
                "rerank_score": chunk.get("rerank_score", None)
            }
            sources.append(source)
        
        return sources
    
    def _error_response(self, error: str) -> Dict:
        """Create error response"""
        return {
            "success": False,
            "answer": None,
            "error": error,
            "sources": []
        }
