from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Float, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    content_type = Column(String)
    size = Column(Integer)
    status = Column(String, default="processing")
    chunks_count = Column(Integer, default=0)
    entities_count = Column(Integer, default=0)
    relationships_count = Column(Integer, default=0)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)
    meta_data = Column(JSON, default={})  # ✅ renamed

    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    queries = relationship("Query", back_populates="document")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"))
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer)
    embedding = Column(JSON)
    meta_data = Column(JSON, default={})  # ✅ renamed
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="chunks")


class Query(Base):
    __tablename__ = "queries"
    id = Column(String, primary_key=True)
    query_text = Column(Text, nullable=False)
    answer = Column(Text)
    strategy_used = Column(String)
    query_type = Column(String)
    session_id = Column(String, ForeignKey("sessions.id"))
    document_id = Column(String, ForeignKey("documents.id"))
    processing_time = Column(Float)
    confidence_score = Column(Float)
    retrieved_chunks_count = Column(Integer, default=0)
    agent_steps_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    meta_data = Column(JSON, default={})  # ✅ renamed

    session = relationship("Session", back_populates="queries")
    document = relationship("Document", back_populates="queries")


class Session(Base):
    __tablename__ = "sessions"
    id = Column(String, primary_key=True)
    user_id = Column(String)
    started_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    message_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    meta_data = Column(JSON, default={})  # ✅ renamed

    queries = relationship("Query", back_populates="session")


class GraphEntity(Base):
    __tablename__ = "graph_entities"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)
    description = Column(Text)
    properties = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class GraphRelationship(Base):
    __tablename__ = "graph_relationships"
    id = Column(String, primary_key=True)
    source_id = Column(String, ForeignKey("graph_entities.id", ondelete="CASCADE"))
    target_id = Column(String, ForeignKey("graph_entities.id", ondelete="CASCADE"))
    relation_type = Column(String, nullable=False)
    properties = Column(JSON, default={})
    weight = Column(Float, default=1.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class AgentExecution(Base):
    __tablename__ = "agent_executions"
    id = Column(String, primary_key=True)
    agent_type = Column(String, nullable=False)
    query_id = Column(String, ForeignKey("queries.id"))
    status = Column(String)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    execution_time = Column(Float)
    steps_count = Column(Integer, default=0)
    tools_used = Column(JSON, default=[])
    output = Column(Text)
    confidence = Column(Float)
    errors = Column(JSON, default=[])
    meta_data = Column(JSON, default={})  # ✅ renamed
