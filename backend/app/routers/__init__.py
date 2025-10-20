"""
app/routers/__init__.py
"""
from app.routers.rag_router import router as rag_router
# from app.routers.agent_router import agent_router
# from app.routers.graph_router import graph_router
# from app.routers.document_router import document_router

__all__ = [
    "rag_router",
    "agent_router",
    "graph_router",
    "document_router"
]
