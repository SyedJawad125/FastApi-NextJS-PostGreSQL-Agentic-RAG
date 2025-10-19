"""
app/crud/query_crud.py - Query CRUD operations
"""
from typing import List, Optional
import uuid
from app.models.rag_model import Query, Session


class QueryCRUD:
    """CRUD operations for queries"""
    
    @staticmethod
    def create_query(
        db: Session,
        query_text: str,
        answer: str,
        strategy_used: str,
        query_type: str = None,
        session_id: str = None,
        processing_time: float = 0.0,
        confidence_score: float = None,
        metadata: dict = None
    ) -> Query:
        """Create query record"""
        query = Query(
            id=str(uuid.uuid4()),
            query_text=query_text,
            answer=answer,
            strategy_used=strategy_used,
            query_type=query_type,
            session_id=session_id,
            processing_time=processing_time,
            confidence_score=confidence_score,
            metadata=metadata or {}
        )
        db.add(query)
        db.commit()
        db.refresh(query)
        return query
    
    @staticmethod
    def get_query(db: Session, query_id: str) -> Optional[Query]:
        """Get query by ID"""
        return db.query(Query).filter(Query.id == query_id).first()
    
    @staticmethod
    def list_queries(
        db: Session,
        session_id: str = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Query]:
        """List queries"""
        query = db.query(Query)
        if session_id:
            query = query.filter(Query.session_id == session_id)
        return query.offset(skip).limit(limit).all()
    
    @staticmethod
    def get_query_stats(db: Session) -> dict:
        """Get query statistics"""
        from sqlalchemy import func
        
        total_queries = db.query(func.count(Query.id)).scalar()
        avg_processing_time = db.query(func.avg(Query.processing_time)).scalar()
        avg_confidence = db.query(func.avg(Query.confidence_score)).scalar()
        
        strategy_counts = db.query(
            Query.strategy_used,
            func.count(Query.id)
        ).group_by(Query.strategy_used).all()
        
        return {
            "total_queries": total_queries,
            "average_processing_time": avg_processing_time or 0.0,
            "average_confidence": avg_confidence or 0.0,
            "strategy_usage": dict(strategy_counts)
        }
