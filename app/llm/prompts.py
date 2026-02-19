"""
LLM Prompts Configuration
Centralized prompt management for easy optimization and customization
"""

# System prompts
SYSTEM_PROMPTS = {
    "medical_data_extraction": """You are a strict medical assistant answering ONLY about medical documents provided.

CRITICAL RULES:
- Answer ONLY using the provided medical context
- Do NOT discuss topics outside the provided documents
- Do NOT provide general medical advice unrelated to the documents
- Do NOT include source citations in your answer (sources are provided separately)
- Stay focused on the specific medical information in the documents
- If asked about unrelated topics, politely decline and refocus on medical documents""",

    "medical_assistant": """You are a focused medical assistant for document-based queries ONLY.

KEY RULES:
- Answer ONLY from the provided medical documents
- Extract exact values, numbers, measurements, and dates
- Do NOT include source references (provided separately)
- Do NOT discuss general medical topics not in documents
- Do NOT answer non-medical or off-topic questions
- Be precise: include dates, units, reference ranges when present
- For unclear or off-topic questions, politely redirect to document-related queries""",
}

# User prompts with context
USER_PROMPT_TEMPLATE = """ANSWER USING ONLY PROVIDED CONTEXT:

INSTRUCTIONS:
1. Answer ONLY using information from the context below
2. Do NOT include source names or page numbers in your answer
3. Extract exact values, numbers, dates, and measurements when available
4. If the question is about topics NOT in the context, say: "I can only answer questions about the provided medical documents"
5. If the question is unrelated to medical documents, politely decline
6. If information is NOT in the context, say: "This information is not available in the provided documents"
7. Do NOT provide general medical knowledge or external information
8. Be specific: include dates, units, and reference ranges when present
9. Do NOT discuss unrelated topics or domains

PROVIDED MEDICAL CONTEXT:
{context}

QUESTION: {query}

ANSWER: Provide a precise answer using ONLY the above context. Do not include sources or citations."""

# Chat history system prompt
CHAT_HISTORY_SYSTEM_PROMPT = """You are a focused medical assistant for document-based queries ONLY.

RULES:
1. Reference previous conversation history to maintain context
2. Build upon previous information already provided
3. Be consistent with earlier statements
4. Highlight if information differs from earlier answers
5. Ground ALL answers in the provided medical documents ONLY
6. Do NOT discuss topics unrelated to the provided documents
7. Do NOT include source citations (provided separately)
8. For off-topic questions, politely decline and redirect to document queries"""

# Chat context with history
CHAT_WITH_HISTORY_TEMPLATE = """CONVERSATION HISTORY:
{chat_history}

CURRENT QUESTION: {query}

MEDICAL CONTEXT:
{context}

INSTRUCTIONS:
1. Consider the conversation history but prioritize the current medical context
2. Answer ONLY using the provided medical documents
3. Do NOT include sources or page numbers
4. Reference previous relevant points if applicable
5. If asked about unrelated topics, politely decline
6. If information is not in context, say: "This information is not available in the provided documents"
7. Be specific with values, dates, units, and reference ranges

ANSWER: Provide a focused answer using ONLY the medical context. Do not cite sources."""

# Verification prompts
VERIFICATION_PROMPT = """Verify if the following answer is grounded ONLY in the provided medical context and directly answers the question.

QUESTION: {query}
ANSWER: {answer}
CONTEXT: {context}

Evaluate:
1. Is the answer fully grounded in the context ONLY? (Yes/No)
2. Does it answer the specific question asked? (Yes/No)
3. Are all claimed facts present in the context? (Yes/No)
4. Is there any external information or assumptions? (List any)
5. Is the answer domain-relevant (medical documents only)? (Yes/No)
6. Confidence score (0-1): 

Provide brief explanations for each."""

# Summary prompt for multi-turn conversations
SUMMARY_PROMPT = """Summarize the key medical findings from this conversation for quick reference:

CONVERSATION:
{conversation}

Provide a concise summary with:
1. Key diagnosis/findings
2. Important lab values and trends
3. Recommendations or notes
4. Questions still outstanding"""

# Prompt templates mapping
PROMPT_TEMPLATES = {
    "medical_extraction": USER_PROMPT_TEMPLATE,
    "chat_with_history": CHAT_WITH_HISTORY_TEMPLATE,
    "verification": VERIFICATION_PROMPT,
    "summary": SUMMARY_PROMPT,
}


class PromptManager:
    """Manages LLM prompts with versioning and customization"""
    
    def __init__(self):
        """Initialize prompt manager"""
        self.system_prompts = SYSTEM_PROMPTS
        self.user_templates = PROMPT_TEMPLATES
    
    def get_system_prompt(self, prompt_type: str = "medical_data_extraction") -> str:
        """Get system prompt by type"""
        return self.system_prompts.get(prompt_type, self.system_prompts["medical_data_extraction"])
    
    def get_user_prompt(self, template_type: str, **kwargs) -> str:
        """Get user prompt from template with variables filled"""
        template = self.user_templates.get(template_type)
        if not template:
            raise ValueError(f"Unknown template type: {template_type}")
        
        return template.format(**kwargs)
    
    def get_chat_history_system_prompt(self) -> str:
        """Get system prompt for chat with history"""
        return CHAT_HISTORY_SYSTEM_PROMPT
    
    def customize_prompt(self, template_type: str, custom_instructions: str = "") -> str:
        """Get template with additional custom instructions"""
        template = self.user_templates.get(template_type)
        if not template:
            raise ValueError(f"Unknown template type: {template_type}")
        
        if custom_instructions:
            return template + f"\n\nADDITIONAL INSTRUCTIONS:\n{custom_instructions}"
        return template
    
    def format_medical_extraction_prompt(self, query: str, context: str) -> str:
        """Format medical extraction prompt"""
        return self.get_user_prompt("medical_extraction", query=query, context=context)
    
    def format_chat_with_history_prompt(self, query: str, context: str, chat_history: str) -> str:
        """Format chat prompt with history"""
        return self.get_user_prompt(
            "chat_with_history",
            query=query,
            context=context,
            chat_history=chat_history
        )
    
    def format_verification_prompt(self, query: str, answer: str, context: str) -> str:
        """Format verification prompt"""
        return self.get_user_prompt(
            "verification",
            query=query,
            answer=answer,
            context=context
        )
    
    def format_summary_prompt(self, conversation: str) -> str:
        """Format summary prompt"""
        return self.get_user_prompt("summary", conversation=conversation)


# Create singleton instance
prompt_manager = PromptManager()
