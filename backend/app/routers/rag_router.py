"""
===================================================================
app/routers/rag_router.py - FIXED (No prefix in router)
===================================================================
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from sqlalchemy.orm import Session
from typing import List, Dict, Any
import time
from datetime import datetime
import uuid
import logging

# Correct imports
from app.schemas.rag_schemas import RAGQueryRequest, RAGQueryResponse, HealthCheckResponse, DocumentUploadResponse
from app.core.config import get_db
from app.models.rag_model import (
    Document,
    Query as QueryModel,
    Session as SessionModel,
    AgentExecution
)

# Import RAG service - will be initialized on first use
from app.core.dependencies import get_rag_service

logger = logging.getLogger(__name__)

# ❌ REMOVE PREFIX HERE - it's added in main.py
router = APIRouter(tags=["RAG System"])


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
        logger.info(f"Received query: {request.query[:50]}...")
        start_time = time.time()
        
        # Get RAG service
        rag_service = get_rag_service()
        
        # Process query based on strategy
        if request.strategy == "simple":
            result = await rag_service.simple_query(request.query, request.top_k)
        elif request.strategy == "agentic":
            result = await rag_service.agentic_query(request.query)
        elif request.strategy == "auto":
            result = await rag_service.auto_query(request.query, request.top_k)
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid strategy: {request.strategy}. Use: simple, agentic, or auto"
            )
        
        processing_time = time.time() - start_time
        logger.info(f"Query processed in {processing_time:.2f}s")
        
        # Save query to database
        db_query = QueryModel(
            id=str(uuid.uuid4()),
            query_text=request.query,
            answer=result["answer"],
            strategy_used=request.strategy,
            processing_time=processing_time,
            confidence_score=result.get("confidence", 0.85),
            retrieved_chunks_count=len(result.get("chunks", [])),
            session_id=request.session_id,
            metadata={
                "top_k": request.top_k,
                "chunk_count": len(result.get("chunks", []))
            }
        )
        db.add(db_query)
        db.commit()
        db.refresh(db_query)
        
        return RAGQueryResponse(
            query=request.query,
            answer=result["answer"],
            strategy_used=request.strategy,
            processing_time=processing_time,
            retrieved_chunks=result.get("chunks", []),
            confidence_score=result.get("confidence", 0.85)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


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
        logger.info(f"Received file upload: {file.filename}")
        
        # Validate file type
        allowed_types = [
            "application/pdf", 
            "text/plain", 
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ]
        
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"File type not supported. Allowed: PDF, TXT, DOCX. Got: {file.content_type}"
            )
        
        # Read file content
        logger.info("Reading file content...")
        content = await file.read()
        logger.info(f"File read: {len(content)} bytes")
        
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        # Get RAG service
        rag_service = get_rag_service()
        
        # Process document
        logger.info("Processing document...")
        result = await rag_service.process_document(
            filename=file.filename,
            content=content,
            content_type=file.content_type
        )
        logger.info(f"Document processed: {result['chunks_count']} chunks created")
        
        # Save to database
        db_doc = Document(
            id=str(uuid.uuid4()),
            filename=file.filename,
            content_type=file.content_type,
            size=len(content),
            status="completed",
            chunks_count=result["chunks_count"],
            uploaded_at=datetime.utcnow(),
            processed_at=datetime.utcnow(),
            metadata={
                "original_size": len(content),
                "processing_method": result.get("method", "default")
            }
        )
        db.add(db_doc)
        db.commit()
        db.refresh(db_doc)
        
        logger.info(f"Document saved to database: {db_doc.id}")
        
        return DocumentUploadResponse(
            document_id=db_doc.id,
            filename=file.filename,
            status="success",
            chunks_created=result["chunks_count"],
            message=f"Document '{file.filename}' processed successfully with {result['chunks_count']} chunks"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload failed: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Document upload failed: {str(e)}")


@router.get("/documents")
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """List all processed documents"""
    try:
        documents = db.query(Document).offset(skip).limit(limit).all()
        total_count = db.query(Document).count()
        
        return {
            "total": total_count,
            "count": len(documents),
            "skip": skip,
            "limit": limit,
            "documents": [
                {
                    "id": doc.id,
                    "filename": doc.filename,
                    "content_type": doc.content_type,
                    "size": doc.size,
                    "status": doc.status,
                    "chunks_count": doc.chunks_count,
                    "uploaded_at": doc.uploaded_at.isoformat() if doc.uploaded_at else None,
                    "processed_at": doc.processed_at.isoformat() if doc.processed_at else None
                }
                for doc in documents
            ]
        }
    except Exception as e:
        logger.error(f"Failed to fetch documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch documents: {str(e)}")


@router.get("/documents/{document_id}")
async def get_document(
    document_id: str,
    db: Session = Depends(get_db)
):
    """Get specific document details"""
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        
        if not document:
            raise HTTPException(status_code=404, detail=f"Document with ID {document_id} not found")
        
        return {
            "id": document.id,
            "filename": document.filename,
            "content_type": document.content_type,
            "size": document.size,
            "status": document.status,
            "chunks_count": document.chunks_count,
            "entities_count": document.entities_count,
            "relationships_count": document.relationships_count,
            "uploaded_at": document.uploaded_at.isoformat() if document.uploaded_at else None,
            "processed_at": document.processed_at.isoformat() if document.processed_at else None,
            "metadata": document.metadata
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch document: {str(e)}")


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    db: Session = Depends(get_db)
):
    """Delete a document and its chunks"""
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        
        if not document:
            raise HTTPException(status_code=404, detail=f"Document with ID {document_id} not found")
        
        filename = document.filename
        db.delete(document)
        db.commit()
        
        return {
            "status": "success",
            "message": f"Document '{filename}' deleted successfully",
            "document_id": document_id
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to delete document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@router.get("/queries")
async def list_queries(
    limit: int = 10,
    skip: int = 0,
    db: Session = Depends(get_db)
):
    """List recent queries"""
    try:
        queries = db.query(QueryModel).order_by(
            QueryModel.created_at.desc()
        ).offset(skip).limit(limit).all()
        
        total_count = db.query(QueryModel).count()
        
        return {
            "total": total_count,
            "count": len(queries),
            "skip": skip,
            "limit": limit,
            "queries": [
                {
                    "id": q.id,
                    "query": q.query_text,
                    "answer": q.answer[:200] + "..." if q.answer and len(q.answer) > 200 else q.answer,
                    "strategy": q.strategy_used,
                    "processing_time": q.processing_time,
                    "confidence_score": q.confidence_score,
                    "retrieved_chunks_count": q.retrieved_chunks_count,
                    "created_at": q.created_at.isoformat() if q.created_at else None
                }
                for q in queries
            ]
        }
    except Exception as e:
        logger.error(f"Failed to fetch queries: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch queries: {str(e)}")


@router.get("/queries/{query_id}")
async def get_query(
    query_id: str,
    db: Session = Depends(get_db)
):
    """Get specific query details"""
    try:
        query = db.query(QueryModel).filter(QueryModel.id == query_id).first()
        
        if not query:
            raise HTTPException(status_code=404, detail=f"Query with ID {query_id} not found")
        
        return {
            "id": query.id,
            "query_text": query.query_text,
            "answer": query.answer,
            "strategy_used": query.strategy_used,
            "query_type": query.query_type,
            "processing_time": query.processing_time,
            "confidence_score": query.confidence_score,
            "retrieved_chunks_count": query.retrieved_chunks_count,
            "agent_steps_count": query.agent_steps_count,
            "created_at": query.created_at.isoformat() if query.created_at else None,
            "metadata": query.metadata
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch query: {str(e)}")


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint with database verification"""
    try:
        from sqlalchemy import text
        # Test database connection
        db.execute(text("SELECT 1"))
        db_status = "operational"
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        db_status = "error"
    
    return HealthCheckResponse(
        status="healthy" if db_status == "operational" else "degraded",
        timestamp=datetime.now(),
        version="1.0.0",
        components={
            "llm_service": "operational",
            "vector_store": "operational",
            "database": db_status
        }
    )


@router.get("/stats")
async def get_statistics(db: Session = Depends(get_db)):
    """Get system statistics"""
    try:
        from sqlalchemy import func
        
        total_documents = db.query(Document).count()
        total_queries = db.query(QueryModel).count()
        total_chunks = db.query(func.sum(Document.chunks_count)).scalar() or 0
        
        # Get average processing time
        avg_processing_time = db.query(func.avg(QueryModel.processing_time)).scalar() or 0
        
        # Get strategy distribution
        strategy_stats = db.query(
            QueryModel.strategy_used,
            func.count(QueryModel.id)
        ).group_by(QueryModel.strategy_used).all()
        
        return {
            "total_documents": total_documents,
            "total_queries": total_queries,
            "total_chunks": int(total_chunks),
            "average_processing_time": float(avg_processing_time),
            "strategy_distribution": {
                strategy: count for strategy, count in strategy_stats
            }
        }
    except Exception as e:
        logger.error(f"Failed to fetch statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch statistics: {str(e)}")


@router.post("/sessions")
async def create_session(
    user_id: str = None,
    db: Session = Depends(get_db)
):
    """Create a new chat session"""
    try:
        session = SessionModel(
            id=str(uuid.uuid4()),
            user_id=user_id,
            started_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            is_active=True
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        
        return {
            "session_id": session.id,
            "user_id": session.user_id,
            "started_at": session.started_at.isoformat(),
            "status": "active"
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")


@router.get("/sessions/{session_id}")
async def get_session(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Get session details with query history"""
    try:
        session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
        
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        queries = db.query(QueryModel).filter(
            QueryModel.session_id == session_id
        ).order_by(QueryModel.created_at.asc()).all()
        
        return {
            "session_id": session.id,
            "user_id": session.user_id,
            "started_at": session.started_at.isoformat() if session.started_at else None,
            "last_activity": session.last_activity.isoformat() if session.last_activity else None,
            "message_count": session.message_count,
            "is_active": session.is_active,
            "queries": [
                {
                    "id": q.id,
                    "query": q.query_text,
                    "answer": q.answer,
                    "timestamp": q.created_at.isoformat() if q.created_at else None
                }
                for q in queries
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch session: {str(e)}")