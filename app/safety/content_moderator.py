"""
Content Moderation Service
Uses OpenAI's moderation API to detect and filter harmful content
Checks for hate speech, violence, and other policy violations
"""

import logging
from typing import Dict, Tuple
from app.config.settings import settings

logger = logging.getLogger(__name__)


class ContentModerator:
    """Handles content moderation using OpenAI's moderation API"""
    
    def __init__(self):
        """Initialize content moderator"""
        self.enabled = True
        self._initialize_client()
        self.threshold = 0.5  # Confidence threshold for flagging content
        
        # Categories we care about
        self.categories_to_check = [
            "hate",  # Hate speech: content that promotes hate based on protected characteristics
            "hate/threatening",  # Hate speech that involves violence
            "violence",  # Content that promotes, encourages, or depicts acts of violence
            "harassment",  # Content that is meant to harass or bully an individual
        ]
    
    def _initialize_client(self):
        """Initialize OpenAI client for moderation"""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=settings.llm_api_key)
            logger.info("✅ Initialized OpenAI moderation client")
        except Exception as e:
            logger.error(f"❌ Error initializing moderation client: {str(e)}")
            self.client = None
            self.enabled = False
    
    async def check_input(self, text: str) -> Tuple[bool, Dict]:
        """
        Check user input for harmful content
        
        Args:
            text: User input text to check
            
        Returns:
            Tuple of (is_safe, moderation_details)
            is_safe: True if content is safe, False if flagged
            moderation_details: Dict with violation details
        """
        if not self.enabled or not self.client or len(text.strip()) < 2:
            return True, {"flagged": False, "reason": "moderation_disabled"}
        
        try:
            logger.info("[MODERATION] Checking user input for harmful content...")
            logger.debug(f"   Input length: {len(text)} characters")
            
            result = await self._moderate_text(text, stage="input")
            
            if result["flagged"]:
                logger.warning(f"[MODERATION] ⚠️  Input flagged for: {result['violated_categories']}")
                return False, result
            
            logger.debug("[MODERATION] ✅ Input passed safety check")
            return True, {"flagged": False, "reason": "safe"}
            
        except Exception as e:
            logger.error(f"[MODERATION] Error checking input: {str(e)}")
            # On error, allow content to pass (fail-open mode)
            return True, {"flagged": False, "error": str(e), "reason": "check_error"}
    
    async def check_output(self, text: str) -> Tuple[bool, Dict]:
        """
        Check LLM output for harmful content
        
        Args:
            text: LLM-generated output to check
            
        Returns:
            Tuple of (is_safe, moderation_details)
            is_safe: True if content is safe, False if flagged
            moderation_details: Dict with violation details
        """
        if not self.enabled or not self.client or len(text.strip()) < 2:
            return True, {"flagged": False, "reason": "moderation_disabled"}
        
        try:
            logger.info("[MODERATION] Checking LLM output for harmful content...")
            logger.debug(f"   Output length: {len(text)} characters")
            
            result = await self._moderate_text(text, stage="output")
            
            if result["flagged"]:
                logger.warning(f"[MODERATION] ⚠️  Output flagged for: {result['violated_categories']}")
                return False, result
            
            logger.debug("[MODERATION] ✅ Output passed safety check")
            return True, {"flagged": False, "reason": "safe"}
            
        except Exception as e:
            logger.error(f"[MODERATION] Error checking output: {str(e)}")
            # On error, allow content to pass (fail-open mode)
            return True, {"flagged": False, "error": str(e), "reason": "check_error"}
    
    async def _moderate_text(self, text: str, stage: str = "input") -> Dict:
        """
        Run OpenAI moderation API on text
        
        Args:
            text: Text to moderate
            stage: 'input' or 'output' for logging
            
        Returns:
            Dictionary with moderation results
        """
        try:
            logger.debug(f"[MODERATION] Calling OpenAI moderation API ({stage})...")
            
            # Call OpenAI moderation API
            response = self.client.moderations.create(input=text)
            
            logger.debug(f"[MODERATION] API response received")
            
            # Check if any category is flagged
            result = response.results[0]
            
            if result.flagged:
                # Extract violated categories
                violated = []
                violations_detail = {}
                
                for category in self.categories_to_check:
                    if hasattr(result.categories, category):
                        category_flagged = getattr(result.categories, category)
                        if category_flagged:
                            violated.append(category)
                            category_score = getattr(result.category_scores, category, 0.0)
                            violations_detail[category] = category_score
                            logger.debug(f"     - {category}: {category_score:.2%}")
                
                return {
                    "flagged": True,
                    "violated_categories": violated,
                    "violation_scores": violations_detail,
                    "reason": "policy_violation"
                }
            
            return {
                "flagged": False,
                "reason": "safe",
                "checked_categories": self.categories_to_check
            }
            
        except Exception as e:
            logger.error(f"[MODERATION] Error in moderation API call: {str(e)}")
            raise
    
    def get_violation_message(self, moderation_result: Dict, stage: str = "input") -> str:
        """
        Generate user-friendly message for policy violation
        
        Args:
            moderation_result: Result from check_input or check_output
            stage: 'input' or 'output' to customize message
            
        Returns:
            User-friendly error message
        """
        if not moderation_result.get("flagged"):
            return ""
        
        categories = moderation_result.get("violated_categories", [])
        
        if stage == "input":
            return (
                f"Your message violates our safety policy ({', '.join(categories)}). "
                "Please rephrase your question without hate speech, violence, or harassment."
            )
        else:  # output
            return (
                f"The generated response contained policy violations ({', '.join(categories)}). "
                "Please ask a different question or try again."
            )
