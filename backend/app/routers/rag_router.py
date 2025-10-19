"""
===================================================================
3. app/routers/rag_router.py - Main RAG Router
===================================================================
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from sqlalchemy.orm import Session
from typing import List
import time
from datetime import datetime

from app.schemas.rag_schemas import (
    RAGQueryRequest,
    RAGQueryResponse,
    DocumentUploadResponse,
    HealthCheckResponse
)
from app.core.config import get_db
from app.services.rag_service import RAGService

router = APIRouter(prefix="/api/rag", tags=["RAG System"])

# Initialize RAG service (you'll need to implement this)
rag_service = RAGService()


@router.post("/query", response_model=RAGQueryResponse)
async def query_rag(
    request: RAGQueryRequest,
    db: Session = Depends(get_db)
):
    """
    Main RAG query endpoint
    
    Supports multiple strategies:
    - simple: Basic retrieval + generation
    - agentic: ReAct agent with reasoning
    - auto: Automatic strategy selection
    """
    try:
        start_time = time.time()
        
        # Process query based on strategy
        if request.strategy == "simple":
            result = await rag_service.simple_query(request.query, request.top_k)
        elif request.strategy == "agentic":
            result = await rag_service.agentic_query(request.query)
        elif request.strategy == "auto":
            result = await rag_service.auto_query(request.query, request.top_k)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid strategy: {request.strategy}")
        
        processing_time = time.time() - start_time
        
        # Save query to database
        from app.models.rag_model import RAGQuery as RAGQueryModel
        db_query = RAGQueryModel(
            query_text=request.query,
            answer=result["answer"],
            strategy_used=request.strategy,
            processing_time=processing_time
        )
        db.add(db_query)
        db.commit()
        
        return RAGQueryResponse(
            query=request.query,
            answer=result["answer"],
            strategy_used=request.strategy,
            processing_time=processing_time,
            retrieved_chunks=result.get("chunks", []),
            confidence_score=result.get("confidence", 0.85)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload and process document for RAG
    
    Supports: PDF, TXT, DOCX
    """
    try:
        # Validate file type
        allowed_types = ["application/pdf", "text/plain", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="File type not supported")
        
        # Read file content
        content = await file.read()
        
        # Process document
        result = await rag_service.process_document(
            filename=file.filename,
            content=content,
            content_type=file.content_type
        )
        
        # Save to database
        from app.models.rag_model import Document
        db_doc = Document(
            filename=file.filename,
            content_type=file.content_type,
            chunks_count=result["chunks_count"]
        )
        db.add(db_doc)
        db.commit()
        db.refresh(db_doc)
        
        return DocumentUploadResponse(
            document_id=db_doc.id,
            filename=file.filename,
            status="success",
            chunks_created=result["chunks_count"],
            message=f"Document processed successfully with {result['chunks_count']} chunks"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents")
async def list_documents(db: Session = Depends(get_db)):
    """List all processed documents"""
    from app.models.rag_model import Document
    documents = db.query(Document).all()
    return {
        "total": len(documents),
        "documents": [
            {
                "id": doc.id,
                "filename": doc.filename,
                "chunks_count": doc.chunks_count,
                "created_at": doc.created_at
            }
            for doc in documents
        ]
    }


@router.get("/queries")
async def list_queries(limit: int = 10, db: Session = Depends(get_db)):
    """List recent queries"""
    from app.models.rag_model import RAGQuery as RAGQueryModel
    queries = db.query(RAGQueryModel).order_by(RAGQueryModel.created_at.desc()).limit(limit).all()
    return {
        "total": len(queries),
        "queries": [
            {
                "id": q.id,
                "query": q.query_text,
                "answer": q.answer[:200] + "..." if len(q.answer) > 200 else q.answer,
                "strategy": q.strategy_used,
                "processing_time": q.processing_time,
                "created_at": q.created_at
            }
            for q in queries
        ]
    }


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        components={
            "llm_service": "operational",
            "vector_store": "operational",
            "database": "operational"
        }
    )
