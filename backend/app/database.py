# from sqlalchemy import create_engine
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker
# from app.config import settings
# import logging

# # Configure logging
# logging.basicConfig()
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

# # Create PostgreSQL engine
# engine = create_engine(
#     settings.SQLALCHEMY_DATABASE_URL,
#     echo=True,
#     pool_pre_ping=True,
#     pool_size=10,
#     max_overflow=20
# )

# SessionLocal = sessionmaker(
#     autocommit=False,
#     autoflush=False,
#     bind=engine
# )

# Base = declarative_base()

# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# # Verify connection
# try:
#     with engine.connect() as conn:
#         logger.info(f"✅ Connected to PostgreSQL at: {settings.SQLALCHEMY_DATABASE_URL}")
# except Exception as e:
#     logger.error(f"❌ Failed to connect to PostgreSQL: {e}")
#     raise




# app/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from contextlib import contextmanager
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# Define Base here only
Base = declarative_base()

# Database URL (PostgreSQL or SQLite fallback)
DATABASE_URL = getattr(settings, "DATABASE_URL", "sqlite:///./data/rag_system.db")

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise


def get_db() -> Session:
    """Get database session (dependency)"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context():
    """Get database session as context manager"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
