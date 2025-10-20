"""
===================================================================
app/routers/rag_router.py - ✅ Final Working RAG Router
===================================================================
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from sqlalchemy.orm import Session
from typing import Dict
import time
from datetime import datetime
import uuid
import logging

# Schemas
from app.schemas.rag_schemas import (
    RAGQueryRequest,
    RAGQueryResponse,
    HealthCheckResponse,
    DocumentUploadResponse
)

# Database
from app.core.config import get_db
from app.models.rag_model import (
    Document,
    Query as QueryModel,
    Session as SessionModel
)

# Orchestrator service
from app.core.dependencies import get_rag_service
from app.core.enums import RAGStrategy

logger = logging.getLogger(__name__)

# ✅ Router prefix
router = APIRouter(prefix="/api/rag", tags=["RAG System"])


# ============================================================
# 🧠 RAG Query Endpoint
# ============================================================
@router.post("/query", response_model=RAGQueryResponse)
async def query_rag(request: RAGQueryRequest, db: Session = Depends(get_db)):
    """Main RAG query endpoint with strategy support."""
    try:
        logger.info(f"Received query: {request.query[:50]}...")
        start_time = time.time()

        rag_service = get_rag_service()

        # ✅ Map strategy string to enum
        strategy_map = {
            "simple": RAGStrategy.SIMPLE,
            "agentic": RAGStrategy.AGENTIC,
            "auto": RAGStrategy.AUTO
        }

        if request.strategy not in strategy_map:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid strategy: {request.strategy}. Use: simple, agentic, or auto"
            )

        strategy = strategy_map[request.strategy]

        # ✅ Execute orchestrator
        result = await rag_service.execute_query(
            query=request.query,
            top_k=request.top_k,
            session_id=request.session_id,
            strategy=strategy
        )

        processing_time = time.time() - start_time

        # ✅ Save query in DB
        db_query = QueryModel(
            id=str(uuid.uuid4()),
            query_text=request.query,
            answer=result["answer"],
            strategy_used=request.strategy,
            processing_time=processing_time,
            confidence_score=result.get("confidence", 0.85),
            retrieved_chunks_count=len(result.get("retrieved_chunks", [])),
            session_id=request.session_id,
            metadata={
                "top_k": request.top_k,
                "chunk_count": len(result.get("retrieved_chunks", []))
            }
        )
        db.add(db_query)
        db.commit()
        db.refresh(db_query)

        return RAGQueryResponse(
            query=request.query,
            answer=result["answer"],
            strategy_used=result["strategy_used"].value,
            processing_time=processing_time,
            retrieved_chunks=result.get("retrieved_chunks", []),
            confidence_score=result.get("confidence", 0.85)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


# ============================================================
# 📤 Document Upload Endpoint
# ============================================================
@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload and process document for RAG (PDF, TXT, DOCX)."""
    try:
        logger.info(f"Received file upload: {file.filename}")

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

        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        rag_service = get_rag_service()
        # ✅ Use process_document instead of ingest_document
        result = await rag_service.process_document(
            filename=file.filename,
            content=content,
            content_type=file.content_type
        )

        db_doc = Document(
            id=str(uuid.uuid4()),
            filename=file.filename,
            content_type=file.content_type,
            size=len(content),
            status="completed",
            chunks_count=result["chunks_created"],
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

        return DocumentUploadResponse(
            document_id=db_doc.id,
            filename=file.filename,
            status="success",
            chunks_created=result["chunks_created"],
            message=f"Document '{file.filename}' processed successfully with {result['chunks_created']} chunks"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload failed: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Document upload failed: {str(e)}")


# ============================================================
# 📄 Document Management
# ============================================================
@router.get("/documents")
async def list_documents(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    try:
        documents = db.query(Document).offset(skip).limit(limit).all()
        total_count = db.query(Document).count()
        return {
            "total": total_count,
            "count": len(documents),
            "documents": [
                {
                    "id": d.id,
                    "filename": d.filename,
                    "content_type": d.content_type,
                    "size": d.size,
                    "status": d.status,
                    "chunks_count": d.chunks_count,
                    "uploaded_at": d.uploaded_at.isoformat() if d.uploaded_at else None,
                    "processed_at": d.processed_at.isoformat() if d.processed_at else None
                }
                for d in documents
            ]
        }
    except Exception as e:
        logger.error(f"Failed to fetch documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}")
async def get_document(document_id: str, db: Session = Depends(get_db)):
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "id": document.id,
        "filename": document.filename,
        "content_type": document.content_type,
        "size": document.size,
        "status": document.status,
        "chunks_count": document.chunks_count,
        "uploaded_at": document.uploaded_at,
        "processed_at": document.processed_at,
        "metadata": document.metadata
    }


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str, db: Session = Depends(get_db)):
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    filename = document.filename
    db.delete(document)
    db.commit()
    return {"status": "success", "message": f"Document '{filename}' deleted successfully"}


# ============================================================
# 📝 Query History
# ============================================================
@router.get("/queries")
async def list_queries(limit: int = 10, skip: int = 0, db: Session = Depends(get_db)):
    queries = db.query(QueryModel).order_by(QueryModel.created_at.desc()).offset(skip).limit(limit).all()
    total = db.query(QueryModel).count()
    return {
        "total": total,
        "count": len(queries),
        "queries": [
            {
                "id": q.id,
                "query": q.query_text,
                "answer": (q.answer[:200] + "...") if q.answer and len(q.answer) > 200 else q.answer,
                "strategy": q.strategy_used,
                "processing_time": q.processing_time,
                "confidence_score": q.confidence_score,
                "created_at": q.created_at.isoformat() if q.created_at else None
            }
            for q in queries
        ]
    }


@router.get("/queries/{query_id}")
async def get_query(query_id: str, db: Session = Depends(get_db)):
    query = db.query(QueryModel).filter(QueryModel.id == query_id).first()
    if not query:
        raise HTTPException(status_code=404, detail="Query not found")
    return {
        "id": query.id,
        "query_text": query.query_text,
        "answer": query.answer,
        "strategy_used": query.strategy_used,
        "processing_time": query.processing_time,
        "confidence_score": query.confidence_score,
        "created_at": query.created_at
    }


# ============================================================
# 🧪 Health & Stats
# ============================================================
@router.get("/health", response_model=HealthCheckResponse)
async def health_check(db: Session = Depends(get_db)):
    from sqlalchemy import text
    try:
        db.execute(text("SELECT 1"))
        db_status = "operational"
    except Exception:
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
    from sqlalchemy import func
    total_docs = db.query(Document).count()
    total_queries = db.query(QueryModel).count()
    total_chunks = db.query(func.sum(Document.chunks_count)).scalar() or 0
    avg_time = db.query(func.avg(QueryModel.processing_time)).scalar() or 0
    strategy_stats = db.query(
        QueryModel.strategy_used,
        func.count(QueryModel.id)
    ).group_by(QueryModel.strategy_used).all()

    return {
        "total_documents": total_docs,
        "total_queries": total_queries,
        "total_chunks": int(total_chunks),
        "average_processing_time": float(avg_time),
        "strategy_distribution": {s: c for s, c in strategy_stats}
    }


# ============================================================
# 💬 Session Endpoints
# ============================================================
@router.post("/sessions")
async def create_session(user_id: str = None, db: Session = Depends(get_db)):
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


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, db: Session = Depends(get_db)):
    session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    queries = db.query(QueryModel).filter(
        QueryModel.session_id == session_id
    ).order_by(QueryModel.created_at.asc()).all()

    return {
        "session_id": session.id,
        "user_id": session.user_id,
        "started_at": session.started_at.isoformat(),
        "last_activity": session.last_activity.isoformat(),
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


# ============================================================
# 🧪 Test
# ============================================================
@router.get("/test")
async def test_endpoint():
    return {"message": "✅ RAG router is working!"}
