# import os
# from dotenv import load_dotenv

# load_dotenv()

# class Settings:
#     GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
#     EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
#     GROQ_MODEL: str = "llama-3.1-8b-instant"

# settings = Settings()



"""
app/core/config.py - Enhanced Configuration Management
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow"
    )
    
    # API Settings
    API_TITLE: str = "Advanced Agentic RAG System"
    API_VERSION: str = "2.0.0"
    API_DESCRIPTION: str = "Multi-Agent RAG with ReAct, Graph RAG, and Adaptive Strategies"
    
    # LLM Configuration
    GROQ_API_KEY: str
    OPENAI_API_KEY: Optional[str] = None
    LLM_MODEL: str = "llama-3.1-70b-versatile"
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 2000
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # Vector Store Configuration
    VECTOR_STORE_TYPE: str = "chroma"  # chroma or faiss
    CHROMA_PERSIST_DIR: str = "./data/vectors"
    COLLECTION_NAME: str = "advanced_rag"
    TOP_K_RESULTS: int = 5
    
    # Graph RAG Configuration
    ENABLE_GRAPH_RAG: bool = True
    GRAPH_STORE_PATH: str = "./data/graphs/knowledge_graph.pkl"
    MAX_GRAPH_NODES: int = 1000
    ENTITY_EXTRACTION_THRESHOLD: float = 0.7
    
    # Agent Configuration
    ENABLE_MULTI_AGENT: bool = True
    MAX_AGENT_ITERATIONS: int = 5
    AGENT_TIMEOUT: int = 60  # seconds
    
    # ReAct Configuration
    MAX_REACT_STEPS: int = 5
    ENABLE_REACT_LOGGING: bool = True
    
    # Text Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_DOCUMENT_SIZE: int = 10_000_000  # 10MB
    
    # Memory Configuration
    MEMORY_TYPE: str = "buffer"  # buffer, summary, or window
    MAX_MEMORY_MESSAGES: int = 10
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"
    AGENT_LOG_FILE: str = "./logs/agents.log"
    
    # File Upload
    UPLOAD_DIR: str = "./uploads"
    ALLOWED_EXTENSIONS: list = [".pdf", ".txt", ".docx", ".md"]
    
    # Performance
    ENABLE_CACHING: bool = True
    CACHE_TTL: int = 3600  # seconds
    
    # CORS
    CORS_ORIGINS: list = ["*"]
    CORS_CREDENTIALS: bool = True
    CORS_METHODS: list = ["*"]
    CORS_HEADERS: list = ["*"]


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


settings = get_settings()