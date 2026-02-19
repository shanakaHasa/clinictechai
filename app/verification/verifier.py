"""
Post-Answer Verification Layer
Validates answer grounding, consistency, and confidence
"""

from typing import Dict, List, Tuple
import logging
from difflib import SequenceMatcher

from app.config.settings import settings

logger = logging.getLogger(__name__)


class AnswerVerifier:
    """Verifies answer quality, grounding, and confidence"""
    
    def __init__(self):
        """Initialize answer verifier"""
        self.confidence_threshold = settings.confidence_threshold
        self.verification_enabled = settings.verification_enabled
    
    async def verify_answer(
        self,
        answer: str,
        context_chunks: List[Dict],
        query: str
    ) -> Dict:
        """
        Verify answer against context and calculate confidence
        
        Args:
            answer: Generated answer
            context_chunks: Source context chunks
            query: Original query
            
        Returns:
            Verification result with confidence score
        """
        try:
            if not self.verification_enabled:
                return self._default_verification(answer, context_chunks)
            
            # Run all verification checks
            grounding_score = self._verify_grounding(answer, context_chunks)
            consistency_score = self._verify_consistency(answer, context_chunks)
            relevance_score = self._verify_relevance(answer, query)
            
            # Calculate overall confidence
            overall_confidence = (grounding_score + consistency_score + relevance_score) / 3
            
            # Extract evidence
            evidence = self._extract_evidence(answer, context_chunks)
            
            return {
                "verified": True,
                "confidence_score": overall_confidence,
                "meets_threshold": overall_confidence >= self.confidence_threshold,
                "grounding_score": grounding_score,
                "consistency_score": consistency_score,
                "relevance_score": relevance_score,
                "evidence": evidence,
                "checks": {
                    "grounding": grounding_score >= 0.7,
                    "consistency": consistency_score >= 0.7,
                    "relevance": relevance_score >= 0.7
                }
            }
            
        except Exception as e:
            logger.error(f"Error in verification: {str(e)}")
            return {
                "verified": False,
                "error": str(e),
                "confidence_score": 0
            }
    
    def _verify_grounding(self, answer: str, chunks: List[Dict]) -> float:
        """
        Verify that answer is grounded in provided context
        
        Returns:
            Confidence score 0-1
        """
        try:
            if not chunks or not isinstance(chunks, list):
                return 0.0
            
            # Filter out non-dict chunks
            valid_chunks = [c for c in chunks if isinstance(c, dict)]
            if not valid_chunks:
                return 0.0
            
            # Check how much of answer is supported by context
            answer_sentences = answer.split(". ") if isinstance(answer, str) else [""]
            answer_sentences = [s for s in answer_sentences if s.strip()]
            supported_sentences = 0
            
            for sentence in answer_sentences:
                for chunk in valid_chunks:
                    chunk_text = chunk.get("text", "")
                    if not isinstance(chunk_text, str):
                        continue
                    
                    chunk_text = chunk_text.lower()
                    sentence_lower = sentence.lower()
                    
                    # Check if key parts of sentence appear in chunks
                    if self._sentence_similarity(sentence_lower, chunk_text) > 0.6:
                        supported_sentences += 1
                        break
            
            if answer_sentences:
                grounding_score = min(1.0, supported_sentences / len(answer_sentences))
            else:
                grounding_score = 0.0
            
            logger.info(f"Grounding score: {grounding_score:.2f}")
            return grounding_score
            
        except Exception as e:
            logger.error(f"Error in grounding verification: {str(e)}", exc_info=True)
            return 0.5
    
    def _verify_consistency(self, answer: str, chunks: List[Dict]) -> float:
        """
        Verify consistency with source material
        
        Returns:
            Confidence score 0-1
        """
        try:
            if not chunks or not isinstance(chunks, list):
                return 0.5
            
            # Filter valid chunks and extract text
            valid_chunks = [c for c in chunks if isinstance(c, dict)]
            context_parts = []
            
            for chunk in valid_chunks:
                text = chunk.get("text", "")
                if isinstance(text, str) and text.strip():
                    context_parts.append(text)
            
            if not context_parts:
                return 0.5
            
            context_text = " ".join(context_parts)
            
            # Check for contradictions
            contradictions = self._find_contradictions(answer, context_text)
            
            # Score based on absence of contradictions
            consistency_score = 1.0 - min(1.0, len(contradictions) * 0.2)
            
            logger.info(f"Consistency score: {consistency_score:.2f}")
            return consistency_score
            
        except Exception as e:
            logger.error(f"Error in consistency verification: {str(e)}", exc_info=True)
            return 0.5
    
    def _verify_relevance(self, answer: str, query: str) -> float:
        """
        Verify answer relevance to query
        
        Returns:
            Confidence score 0-1
        """
        try:
            # Ensure both are strings
            if not isinstance(answer, str):
                answer = str(answer) if answer else ""
            if not isinstance(query, str):
                query = str(query) if query else ""
            
            # Calculate similarity between answer and query
            relevance = self._sentence_similarity(answer, query)
            
            logger.info(f"Relevance score: {relevance:.2f}")
            return relevance
            
        except Exception as e:
            logger.error(f"Error in relevance verification: {str(e)}", exc_info=True)
            return 0.5
    
    def _sentence_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity between two strings"""
        s1 = s1.lower().strip()
        s2 = s2.lower().strip()
        
        matcher = SequenceMatcher(None, s1, s2)
        return matcher.ratio()
    
    def _find_contradictions(self, answer: str, context: str) -> List[str]:
        """Find potential contradictions between answer and context"""
        contradictions = []
        
        # Ensure both are strings
        if not isinstance(answer, str):
            answer = str(answer) if answer else ""
        if not isinstance(context, str):
            context = str(context) if context else ""
        
        # Check for explicit negations
        negation_words = ["no ", "not ", "never ", "cannot"]
        
        for word in negation_words:
            if word in answer.lower() and word not in context.lower():
                contradictions.append(f"Possible contradiction: '{word}' in answer not in context")
        
        return contradictions
    
    def _extract_evidence(self, answer: str, chunks: List[Dict]) -> List[Dict]:
        """Extract specific evidence from chunks supporting the answer"""
        evidence = []
        
        if not chunks or not isinstance(chunks, list):
            return evidence
        
        # Filter valid chunks
        valid_chunks = [c for c in chunks if isinstance(c, dict)]
        
        answer_sentences = answer.split(". ") if isinstance(answer, str) else []
        
        for chunk in valid_chunks[:3]:  # Top 3 chunks as evidence
            try:
                evidence.append({
                    "page_number": chunk.get("page_number"),
                    "document": chunk.get("source_document"),
                    "exact_chunk": chunk.get("text", "")[:200] if isinstance(chunk.get("text", ""), str) else "",
                    "bbox": chunk.get("bbox"),
                    "chunk_id": chunk.get("chunk_id"),
                    "highlighted": self._highlight_evidence(chunk.get("text", ""), answer_sentences)
                })
            except Exception as e:
                logger.debug(f"Error extracting evidence from chunk: {e}")
                continue
        
        return evidence
    
    def _highlight_evidence(self, chunk_text: str, answer_sentences: List[str]) -> str:
        """Highlight matching text in chunk"""
        if not isinstance(chunk_text, str):
            chunk_text = str(chunk_text) if chunk_text else ""
        
        highlighted = chunk_text
        
        for sentence in answer_sentences:
            if not isinstance(sentence, str):
                continue
            
            words = sentence.split()
            for word in words:
                if len(word) > 4 and word.lower() in chunk_text.lower():
                    highlighted = highlighted.replace(word, f"**{word}**")
        
        return highlighted
    
    def _default_verification(self, answer: str, chunks: List[Dict]) -> Dict:
        """Default verification when verification is disabled"""
        if not chunks or not isinstance(chunks, list):
            evidence_list = []
        else:
            valid_chunks = [c for c in chunks if isinstance(c, dict)]
            evidence_list = [
                {
                    "page_number": c.get("page_number"),
                    "document": c.get("source_document"),
                    "exact_chunk": c.get("text", "")[:200] if isinstance(c.get("text", ""), str) else ""
                }
                for c in valid_chunks[:3]
            ]
        
        return {
            "verified": True,
            "confidence_score": 0.8,
            "meets_threshold": True,
            "verification_enabled": False,
            "evidence": evidence_list
        }
