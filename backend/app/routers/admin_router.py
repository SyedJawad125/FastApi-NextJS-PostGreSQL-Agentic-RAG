"""
COMPLETE app/routers/admin_router.py
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List
import logging
import sys
from datetime import datetime

from app.core.dependencies import get_rag_orchestrator
from app.schemas.rag_schemas import SystemStats

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["Admin"])


@router.get("/stats", response_model=SystemStats)
async def get_system_stats(orchestrator = Depends(get_rag_orchestrator)):
    """Get comprehensive system statistics"""
    
    try:
        vector_count = orchestrator.vectorstore.get_count()
        graph_stats = orchestrator.graph_service.get_stats()
        total_sessions = len(orchestrator.memory_store.sessions)
        
        memory_usage = sys.getsizeof(orchestrator) / (1024 * 1024)
        uptime = 0.0
        
        return SystemStats(
            total_documents=len(orchestrator.documents),
            total_chunks=vector_count,
            total_graph_nodes=graph_stats["total_nodes"],
            total_graph_edges=graph_stats["total_edges"],
            memory_usage=memory_usage,
            uptime=uptime
        )
    
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear-memory")
async def clear_all_memory(orchestrator = Depends(get_rag_orchestrator)):
    """Clear all conversation memory"""
    
    try:
        sessions = orchestrator.memory_store.get_all_sessions()
        for session_id in sessions:
            orchestrator.memory_store.clear_session(session_id)
        
        return {
            "message": f"Cleared memory for {len(sessions)} sessions",
            "sessions_cleared": len(sessions)
        }
    
    except Exception as e:
        logger.error(f"Error clearing memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear-vectorstore")
async def clear_vectorstore(orchestrator = Depends(get_rag_orchestrator)):
    """Clear all documents from vector store"""
    
    try:
        orchestrator.vectorstore.clear()
        orchestrator.documents.clear()
        
        return {"message": "Vector store cleared successfully"}
    
    except Exception as e:
        logger.error(f"Error clearing vector store: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reload-graph")
async def reload_graph(orchestrator = Depends(get_rag_orchestrator)):
    """Reload knowledge graph from disk"""
    
    try:
        from app.core.config import settings
        orchestrator.graph_builder.load(settings.GRAPH_STORE_PATH)
        
        stats = orchestrator.graph_service.get_stats()
        
        return {
            "message": "Graph reloaded successfully",
            "nodes": stats["total_nodes"],
            "edges": stats["total_edges"]
        }
    
    except Exception as e:
        logger.error(f"Error reloading graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save-graph")
async def save_graph(orchestrator = Depends(get_rag_orchestrator)):
    """Save knowledge graph to disk"""
    
    try:
        from app.core.config import settings
        orchestrator.graph_builder.save(settings.GRAPH_STORE_PATH)
        
        stats = orchestrator.graph_service.get_stats()
        
        return {
            "message": "Graph saved successfully",
            "nodes": stats["total_nodes"],
            "edges": stats["total_edges"]
        }
    
    except Exception as e:
        logger.error(f"Error saving graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions")
async def list_sessions(orchestrator = Depends(get_rag_orchestrator)):
    """List all active sessions"""
    
    try:
        sessions = orchestrator.memory_store.get_all_sessions()
        
        session_info = []
        for session_id in sessions:
            history = orchestrator.memory_store.get_history(session_id)
            session_info.append({
                "session_id": session_id,
                "message_count": len(history),
                "last_message": history[-1]["content"][:100] if history else ""
            })
        
        return {
            "total_sessions": len(sessions),
            "sessions": session_info
        }
    
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}")
async def delete_session(
    session_id: str,
    orchestrator = Depends(get_rag_orchestrator)
):
    """Delete a specific session"""
    
    try:
        orchestrator.memory_store.clear_session(session_id)
        return {"message": f"Session {session_id} deleted successfully"}
    
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_metrics():
    """Get system performance metrics"""
    
    return {
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "queries_processed": 0,
            "average_response_time": 0.0,
            "cache_hit_rate": 0.0,
            "error_rate": 0.0
        }
    }


@router.post("/reset-system")
async def reset_system(orchestrator = Depends(get_rag_orchestrator)):
    """Reset entire system (use with caution)"""
    
    try:
        # Clear vector store
        orchestrator.vectorstore.clear()
        
        # Clear memory
        sessions = orchestrator.memory_store.get_all_sessions()
        for session_id in sessions:
            orchestrator.memory_store.clear_session(session_id)
        
        # Clear documents
        orchestrator.documents.clear()
        
        # Reset agents
        for agent in orchestrator.agents.values():
            if hasattr(agent, 'clear_history'):
                agent.clear_history()
        
        return {
            "message": "System reset successfully",
            "warning": "All data has been cleared"
        }
    
    except Exception as e:
        logger.error(f"Error resetting system: {e}")
        raise HTTPException(status_code=500, detail=str(e))
