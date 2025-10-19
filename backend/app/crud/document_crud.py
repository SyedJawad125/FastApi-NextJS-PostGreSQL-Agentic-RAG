"""
app/crud/document_crud.py - Document CRUD operations
"""
from sqlalchemy.orm import Session
from typing import List, Optional
from app.models.rag_model import Document, DocumentChunk
from datetime import datetime
import uuid


class DocumentCRUD:
    """CRUD operations for documents"""
    
    @staticmethod
    def create_document(
        db: Session,
        filename: str,
        content_type: str,
        size: int,
        metadata: dict = None
    ) -> Document:
        """Create new document"""
        doc = Document(
            id=str(uuid.uuid4()),
            filename=filename,
            content_type=content_type,
            size=size,
            metadata=metadata or {}
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        return doc
    
    @staticmethod
    def get_document(db: Session, document_id: str) -> Optional[Document]:
        """Get document by ID"""
        return db.query(Document).filter(Document.id == document_id).first()
    
    @staticmethod
    def list_documents(
        db: Session,
        skip: int = 0,
        limit: int = 100
    ) -> List[Document]:
        """List all documents"""
        return db.query(Document).offset(skip).limit(limit).all()
    
    @staticmethod
    def update_document_status(
        db: Session,
        document_id: str,
        status: str,
        chunks_count: int = 0,
        entities_count: int = 0,
        relationships_count: int = 0
    ):
        """Update document processing status"""
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.status = status
            doc.chunks_count = chunks_count
            doc.entities_count = entities_count
            doc.relationships_count = relationships_count
            doc.processed_at = datetime.utcnow()
            db.commit()
    
    @staticmethod
    def delete_document(db: Session, document_id: str) -> bool:
        """Delete document"""
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            db.delete(doc)
            db.commit()
            return True
        return False
    
    @staticmethod
    def create_chunk(
        db: Session,
        document_id: str,
        content: str,
        chunk_index: int,
        metadata: dict = None
    ) -> DocumentChunk:
        """Create document chunk"""
        chunk = DocumentChunk(
            id=f"{document_id}_chunk_{chunk_index}",
            document_id=document_id,
            content=content,
            chunk_index=chunk_index,
            metadata=metadata or {}
        )
        db.add(chunk)
        db.commit()
        db.refresh(chunk)
        return chunk