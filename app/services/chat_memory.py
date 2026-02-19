"""
Chat Memory Management
Handles conversation history, context management, and memory optimization
"""

import logging
import json
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ConversationMessage:
    """Represents a single message in conversation"""
    
    def __init__(
        self,
        role: str,
        content: str,
        timestamp: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Initialize conversation message
        
        Args:
            role: "user" or "assistant"
            content: Message content
            timestamp: ISO format timestamp
            metadata: Additional metadata (used chunks, documents, etc.)
        """
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.utcnow().isoformat()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict:
        """Convert message to dictionary"""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ConversationMessage":
        """Create message from dictionary"""
        return cls(
            role=data.get("role"),
            content=data.get("content"),
            timestamp=data.get("timestamp"),
            metadata=data.get("metadata", {})
        )


class ChatMemory:
    """Manages conversation history and context"""
    
    def __init__(self, session_id: str, max_messages: int = 20, max_tokens: int = 4000):
        """
        Initialize chat memory
        
        Args:
            session_id: Unique conversation session ID
            max_messages: Maximum messages to keep in memory
            max_tokens: Maximum tokens to keep (rough estimate: 1 message ≈ 150 tokens)
        """
        self.session_id = session_id
        self.messages: List[ConversationMessage] = []
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.created_at = datetime.utcnow()
        self.last_updated = datetime.utcnow()
        self.documents_used = set()
    
    def add_user_message(self, content: str, metadata: Optional[Dict] = None) -> None:
        """Add user message to memory"""
        message = ConversationMessage("user", content, metadata=metadata)
        self.messages.append(message)
        self.last_updated = datetime.utcnow()
        self._manage_memory()
        logger.debug(f"[MEMORY] Added user message. Total messages: {len(self.messages)}")
    
    def add_assistant_message(
        self,
        content: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """Add assistant message to memory"""
        message = ConversationMessage("assistant", content, metadata=metadata)
        self.messages.append(message)
        self.last_updated = datetime.utcnow()
        
        # Track documents used
        if metadata and "documents_used" in metadata:
            self.documents_used.update(metadata["documents_used"])
        
        self._manage_memory()
        logger.debug(f"[MEMORY] Added assistant message. Total messages: {len(self.messages)}")
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict]:
        """Get conversation history"""
        messages = self.messages
        if limit:
            messages = messages[-limit:]
        return [msg.to_dict() for msg in messages]
    
    def get_history_text(self, limit: Optional[int] = None) -> str:
        """Get conversation history as formatted text for LLM context"""
        messages = self.messages
        if limit:
            messages = messages[-limit:]
        
        history_text = []
        for msg in messages:
            role_label = "User" if msg.role == "user" else "Assistant"
            history_text.append(f"{role_label}: {msg.content}")
        
        return "\n\n".join(history_text)
    
    def get_context_for_llm(self, last_n: int = 6) -> str:
        """Get recent conversation context for LLM (formatted for prompt inclusion)"""
        recent_messages = self.messages[-last_n:] if len(self.messages) > last_n else self.messages
        
        if not recent_messages:
            return "No previous conversation history."
        
        context_parts = ["## Previous Conversation Context:"]
        
        for msg in recent_messages:
            role = "👤 User" if msg.role == "user" else "🤖 Assistant"
            context_parts.append(f"\n{role}:\n{msg.content}")
        
        return "\n".join(context_parts)
    
    def get_last_user_query(self) -> Optional[str]:
        """Get last user query"""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return None
    
    def get_last_assistant_response(self) -> Optional[str]:
        """Get last assistant response"""
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg.content
        return None
    
    def clear_history(self) -> None:
        """Clear conversation history"""
        self.messages = []
        self.documents_used = set()
        logger.info(f"[MEMORY] Cleared conversation history for session {self.session_id}")
    
    def _manage_memory(self) -> None:
        """Manage memory by removing old messages if limit exceeded"""
        # Keep messages within both message count and token limits
        while len(self.messages) > self.max_messages:
            removed = self.messages.pop(0)
            logger.debug(f"[MEMORY] Removed oldest message (max_messages: {self.max_messages})")
        
        # Rough token management (estimate 150 tokens per message)
        estimated_tokens = len(self.messages) * 150
        if estimated_tokens > self.max_tokens:
            # Remove oldest messages until under token limit
            while len(self.messages) > 1 and (len(self.messages) * 150) > self.max_tokens:
                removed = self.messages.pop(0)
                logger.debug(f"[MEMORY] Removed message (token limit: {self.max_tokens})")
    
    def get_summary_stats(self) -> Dict:
        """Get conversation statistics"""
        user_messages = [m for m in self.messages if m.role == "user"]
        assistant_messages = [m for m in self.messages if m.role == "assistant"]
        
        return {
            "session_id": self.session_id,
            "total_messages": len(self.messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "documents_used": list(self.documents_used),
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "duration_seconds": (self.last_updated - self.created_at).total_seconds()
        }
    
    def to_dict(self) -> Dict:
        """Convert entire conversation to dictionary"""
        return {
            "session_id": self.session_id,
            "messages": self.get_history(),
            "documents_used": list(self.documents_used),
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict, max_messages: int = 20) -> "ChatMemory":
        """Create ChatMemory from dictionary"""
        memory = cls(data.get("session_id", ""), max_messages=max_messages)
        memory.created_at = datetime.fromisoformat(data["created_at"])
        memory.last_updated = datetime.fromisoformat(data["last_updated"])
        memory.documents_used = set(data.get("documents_used", []))
        
        for msg_data in data.get("messages", []):
            memory.messages.append(ConversationMessage.from_dict(msg_data))
        
        return memory


class ChatMemoryManager:
    """Manages multiple chat sessions"""
    
    def __init__(self, storage_path: str = "./storage/chat_sessions"):
        """
        Initialize chat memory manager
        
        Args:
            storage_path: Path to store chat sessions
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.sessions: Dict[str, ChatMemory] = {}
        logger.info(f"[MEMORY MANAGER] Initialized with storage path: {self.storage_path}")
    
    def create_session(self, session_id: str) -> ChatMemory:
        """Create new chat session"""
        if session_id in self.sessions:
            logger.warning(f"[MEMORY MANAGER] Session {session_id} already exists")
            return self.sessions[session_id]
        
        memory = ChatMemory(session_id)
        self.sessions[session_id] = memory
        logger.info(f"[MEMORY MANAGER] Created new session: {session_id}")
        return memory
    
    def get_session(self, session_id: str) -> Optional[ChatMemory]:
        """Get existing session"""
        return self.sessions.get(session_id)
    
    def get_or_create_session(self, session_id: str) -> ChatMemory:
        """Get existing session or create new one"""
        if session_id not in self.sessions:
            return self.create_session(session_id)
        return self.sessions[session_id]
    
    def save_session(self, session_id: str) -> bool:
        """Save chat session to disk"""
        try:
            if session_id not in self.sessions:
                logger.warning(f"[MEMORY MANAGER] Session {session_id} not found")
                return False
            
            memory = self.sessions[session_id]
            file_path = self.storage_path / f"{session_id}.json"
            
            with open(file_path, 'w') as f:
                json.dump(memory.to_dict(), f, indent=2, default=str)
            
            logger.info(f"[MEMORY MANAGER] Saved session {session_id} to {file_path}")
            return True
        except Exception as e:
            logger.error(f"[MEMORY MANAGER] Error saving session {session_id}: {e}")
            return False
    
    def load_session(self, session_id: str) -> Optional[ChatMemory]:
        """Load chat session from disk"""
        try:
            file_path = self.storage_path / f"{session_id}.json"
            
            if not file_path.exists():
                logger.warning(f"[MEMORY MANAGER] Session file not found: {file_path}")
                return None
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            memory = ChatMemory.from_dict(data)
            self.sessions[session_id] = memory
            logger.info(f"[MEMORY MANAGER] Loaded session {session_id}")
            return memory
        except Exception as e:
            logger.error(f"[MEMORY MANAGER] Error loading session {session_id}: {e}")
            return None
    
    def delete_session(self, session_id: str) -> bool:
        """Delete chat session"""
        try:
            file_path = self.storage_path / f"{session_id}.json"
            
            if session_id in self.sessions:
                del self.sessions[session_id]
            
            if file_path.exists():
                file_path.unlink()
                logger.info(f"[MEMORY MANAGER] Deleted session {session_id}")
            
            return True
        except Exception as e:
            logger.error(f"[MEMORY MANAGER] Error deleting session {session_id}: {e}")
            return False
    
    def get_all_sessions(self) -> List[str]:
        """Get all session IDs"""
        return list(self.sessions.keys())
    
    def get_session_stats(self, session_id: str) -> Optional[Dict]:
        """Get session statistics"""
        memory = self.get_session(session_id)
        if not memory:
            return None
        return memory.get_summary_stats()


# Global memory manager instance
memory_manager = ChatMemoryManager()
