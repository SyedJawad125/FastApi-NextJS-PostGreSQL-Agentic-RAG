"""
===================================================================
2. app/schemas/rag_schemas.py - Pydantic Schemas
===================================================================
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class RAGQueryRequest(BaseModel):
    """RAG query request schema"""
    query: str = Field(..., min_length=1, description="User query")
    strategy: str = Field(default="simple", description="RAG strategy: simple, agentic, auto")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")
    session_id: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "strategy": "simple",
                "top_k": 5
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
                "answer": "Artificial Intelligence is...",
                "strategy_used": "simple",
                "processing_time": 1.23,
                "retrieved_chunks": []
            }
        }


class DocumentUploadResponse(BaseModel):
    """Document upload response"""
    document_id: int
    filename: str
    status: str
    chunks_created: int
    message: str


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    version: str
    components: Dict[str, str]
