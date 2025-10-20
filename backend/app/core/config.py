# import os
# from dotenv import load_dotenv

# load_dotenv()

# class Settings:
#     GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
#     EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
#     GROQ_MODEL: str = "llama-3.1-8b-instant"

# settings = Settings()



"""
===================================================================
app/core/config.py - Enhanced Configuration Management
===================================================================
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Optional, Generator
from functools import lru_cache
import os
import logging

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow"
    )
    
    # Database Configuration
    SQLALCHEMY_DATABASE_URL: str = "postgresql+psycopg2://postgres:admin@localhost:5432/FastApi_Agentice_RAG_with_ClaudAi"
    
    # Authentication
    SECRET_KEY: str = "this-is-a-very-secure-32-char-secret-key-1234"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # API Settings
    API_TITLE: str = "Advanced Agentic RAG System"
    API_VERSION: str = "2.0.0"
    API_DESCRIPTION: str = "Multi-Agent RAG with ReAct, Graph RAG, and Adaptive Strategies"
    
    # LLM Configuration
    GROQ_API_KEY: str
    OPENAI_API_KEY: Optional[str] = None
    LLM_MODEL: str = "llama-3.1-8b-instant"
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 2000
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # Vector Store Configuration
    VECTOR_STORE_TYPE: str = "chroma"
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
    AGENT_TIMEOUT: int = 60
    MAX_REACT_STEPS: int = 5
    
    # ReAct Configuration
    ENABLE_REACT_LOGGING: bool = True
    
    # Text Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_DOCUMENT_SIZE: int = 10_000_000
    
    # Memory Configuration
    MEMORY_TYPE: str = "buffer"
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
    CACHE_TTL: int = 3600
    
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


# ============================================
# Database Configuration
# ============================================

# Create SQLAlchemy engine
engine = create_engine(
    settings.SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=False
)

# Create SessionLocal class
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

logger.info(f"Database engine created")


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for database session
    
    Usage in FastAPI:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialize database tables
    Creates all tables defined in models
    """
    from app.models.rag_model import Base
    
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {str(e)}")
        raise


def drop_all_tables():
    """
    Drop all tables (USE WITH CAUTION!)
    Only use in development/testing
    """
    from app.models.rag_model import Base
    
    try:
        Base.metadata.drop_all(bind=engine)
        logger.warning("All database tables dropped")
    except Exception as e:
        logger.error(f"Failed to drop tables: {str(e)}")
        raise


def reset_database():
    """
    Reset database (drop and recreate all tables)
    USE WITH CAUTION - This will delete all data!
    """
    logger.warning("Resetting database - all data will be lost!")
    drop_all_tables()
    init_db()
    logger.info("Database reset complete")