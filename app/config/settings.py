"""
Application Configuration Settings
Centralized configuration management using pydantic-settings
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Main Application Settings"""
    
    # FastAPI
    fastapi_env: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True
    
    # Database
    database_url: str = "postgresql://user:password@localhost:5432/rag_db"
    vector_db_type: str = "chroma"
    vector_db_url: Optional[str] = None
    vector_db_path: str = "./storage/chroma_db"
    
    # LLM Configuration
    llm_provider: str = "openai"
    llm_model: str = "gpt-4-turbo"
    llm_api_key: str
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2000
    
    # Embedding Configuration
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: Optional[str] = None
    embedding_dimension: int = 1536
    
    # PDF Processing
    ocr_provider: str = "tesseract"
    tesseract_path: Optional[str] = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    
    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 100
    max_metadata_length: int = 200
    
    # Retrieval
    top_k_results: int = 5
    rerank_model: str = "cross-encoder/mmarco-MiniLMv2-L12-H384-v1"
    similarity_threshold: float = 0.5
    
    # Verification
    verification_enabled: bool = True
    confidence_threshold: float = 0.7
    
    # Storage
    storage_path: str = "./storage"
    log_path: str = "./logs"
    upload_max_size: int = 52428800  # 50MB
    
    # Redis Cache
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl: int = 3600
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Allow extra fields from .env


settings = Settings()
