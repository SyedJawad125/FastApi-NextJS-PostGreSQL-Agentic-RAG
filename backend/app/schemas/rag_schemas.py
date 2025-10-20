"""
===================================================================
app/schemas/rag_schemas.py - Complete RAG Schemas (FIXED)
===================================================================
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ============================================================
# QUERY SCHEMAS
# ============================================================

class RAGQueryRequest(BaseModel):
    """RAG query request schema"""
    query: str = Field(..., min_length=1, description="User query")
    strategy: str = Field(default="simple", description="RAG strategy: simple, agentic, auto")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to retrieve")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation context")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "strategy": "simple",
                "top_k": 5,
                "session_id": None
            }
        }


class RAGQueryResponse(BaseModel):
    """RAG query response schema"""
    query: str
    answer: str
    strategy_used: str
    processing_time: float
    retrieved_chunks: List[Dict[str, Any]] = []
    confidence_score: Optional[float] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is AI?",
                "answer": "Artificial Intelligence is a branch of computer science...",
                "strategy_used": "simple",
                "processing_time": 1.23,
                "retrieved_chunks": [],
                "confidence_score": 0.85
            }
        }


# ============================================================
# MULTI-AGENT SCHEMAS
# ============================================================

class MultiAgentQuery(BaseModel):
    """Multi-agent query request"""
    query: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    include_research: bool = Field(default=True)
    include_evaluation: bool = Field(default=True)
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Explain quantum computing",
                "session_id": None,
                "include_research": True,
                "include_evaluation": True
            }
        }


class MultiAgentResponse(BaseModel):
    """Multi-agent query response"""
    query: str
    answer: str
    research: Optional[str] = None
    evaluation: Optional[str] = None
    score: Optional[float] = None
    verdict: Optional[str] = None
    processing_time: float
    agents_used: List[str] = []
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Explain quantum computing",
                "answer": "Quantum computing is...",
                "research": "Research findings...",
                "evaluation": "The answer is comprehensive...",
                "score": 0.92,
                "verdict": "approved",
                "processing_time": 3.45,
                "agents_used": ["researcher", "writer", "critic"]
            }
        }


# ============================================================
# GRAPH SCHEMAS
# ============================================================

class GraphQuery(BaseModel):
    """Graph-based RAG query"""
    query: str = Field(..., min_length=1)
    max_depth: int = Field(default=2, ge=1, le=5)
    include_relationships: bool = Field(default=True)
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "max_depth": 2,
                "include_relationships": True
            }
        }


class GraphQueryResponse(BaseModel):
    """Graph query response"""
    query: str
    answer: str
    entities_found: List[Dict[str, Any]] = []
    relationships: List[Dict[str, Any]] = []
    processing_time: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "answer": "Machine learning is...",
                "entities_found": [],
                "relationships": [],
                "processing_time": 1.5
            }
        }


# ============================================================
# DOCUMENT SCHEMAS
# ============================================================

class DocumentUpload(BaseModel):
    """Document upload request (used for additional metadata)"""
    tags: Optional[List[str]] = []
    metadata: Optional[Dict[str, Any]] = {}
    
    class Config:
        json_schema_extra = {
            "example": {
                "tags": ["ml", "ai"],
                "metadata": {"author": "John Doe"}
            }
        }


class DocumentUploadResponse(BaseModel):
    """Document upload response"""
    document_id: str
    filename: str
    status: str
    chunks_created: int
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc-123",
                "filename": "machine_learning.pdf",
                "status": "success",
                "chunks_created": 45,
                "message": "Document processed successfully"
            }
        }


class DocumentResponse(BaseModel):
    """Single document response"""
    id: str
    filename: str
    content_type: str
    size: int
    status: str
    chunks_count: int
    entities_count: Optional[int] = 0
    relationships_count: Optional[int] = 0
    uploaded_at: Optional[str] = None
    processed_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "doc-123",
                "filename": "document.pdf",
                "content_type": "application/pdf",
                "size": 1024000,
                "status": "completed",
                "chunks_count": 45,
                "entities_count": 12,
                "relationships_count": 8,
                "uploaded_at": "2025-10-20T10:00:00",
                "processed_at": "2025-10-20T10:01:00",
                "metadata": {}
            }
        }


class DocumentList(BaseModel):
    """List of documents response"""
    total: int
    count: int
    skip: int
    limit: int
    documents: List[DocumentResponse]
    
    class Config:
        json_schema_extra = {
            "example": {
                "total": 100,
                "count": 10,
                "skip": 0,
                "limit": 10,
                "documents": []
            }
        }


# ============================================================
# HEALTH & STATS SCHEMAS
# ============================================================

class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    version: str
    components: Dict[str, str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-10-20T10:00:00",
                "version": "1.0.0",
                "components": {
                    "llm_service": "operational",
                    "vector_store": "operational",
                    "database": "operational"
                }
            }
        }


class HealthCheckResponse(HealthCheck):
    """Alias for HealthCheck"""
    pass


class SystemStats(BaseModel):
    """System statistics response"""
    total_documents: int
    total_queries: int
    total_chunks: int
    average_processing_time: float
    strategy_distribution: Dict[str, int]
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_documents": 150,
                "total_queries": 1250,
                "total_chunks": 6750,
                "average_processing_time": 1.45,
                "strategy_distribution": {
                    "simple": 800,
                    "agentic": 300,
                    "auto": 150
                }
            }
        }


# ============================================================
# BACKWARD COMPATIBILITY ALIASES
# ============================================================

# These aliases ensure backward compatibility
RAGQuery = RAGQueryRequest
RAGResponse = RAGQueryResponse
DocumentUpload = DocumentUpload  # Already correct