"""
app/crud/session_crud.py - Session CRUD operations
"""
from datetime import datetime
from typing import List, Optional
import uuid
from app.models.rag_model import Session as SessionModel
from app.models.rag_model import Session


class SessionCRUD:
    """CRUD operations for sessions"""
    
    @staticmethod
    def create_session(
        db: Session,
        session_id: str = None,
        user_id: str = None,
        metadata: dict = None
    ) -> SessionModel:
        """Create new session"""
        session = SessionModel(
            id=session_id or str(uuid.uuid4()),
            user_id=user_id,
            metadata=metadata or {}
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        return session
    
    @staticmethod
    def get_session(db: Session, session_id: str) -> Optional[SessionModel]:
        """Get session by ID"""
        return db.query(SessionModel).filter(SessionModel.id == session_id).first()
    
    @staticmethod
    def update_session_activity(db: Session, session_id: str):
        """Update session last activity"""
        session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
        if session:
            session.last_activity = datetime.utcnow()
            session.message_count += 1
            db.commit()
    
    @staticmethod
    def list_active_sessions(db: Session) -> List[SessionModel]:
        """List active sessions"""
        return db.query(SessionModel).filter(SessionModel.is_active == True).all()
    
    @staticmethod
    def deactivate_session(db: Session, session_id: str):
        """Deactivate session"""
        session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
        if session:
            session.is_active = False
            db.commit()
