# # ============================================================================
# # MAIN APPLICATION
# # ============================================================================

# """
# main.py - FastAPI Application Entry Point
# """
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from contextlib import asynccontextmanager
# import logging.config

# from app.core.config import settings
# from app.routers.rag_router import router as rag_router
# from app.routers.agent_router import agent_router
# from app.routers.graph_router import graph_router
# from app.routers.document_router import document_router

# # Configure logging
# logging.config.dictConfig({
#     "version": 1,
#     "disable_existing_loggers": False,
#     "formatters": {
#         "default": {
#             "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
#         }
#     },
#     "handlers": {
#         "console": {
#             "class": "logging.StreamHandler",
#             "formatter": "default"
#         },
#         "file": {
#             "class": "logging.FileHandler",
#             "filename": settings.LOG_FILE,
#             "formatter": "default"
#         }
#     },
#     "root": {
#         "level": settings.LOG_LEVEL,
#         "handlers": ["console", "file"]
#     }
# })

# logger = logging.getLogger(__name__)


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Application lifespan events"""
#     # Startup
#     logger.info("Starting Advanced Agentic RAG System...")
    
#     # Initialize orchestrator
#     from app.core.dependencies import initialize_orchestrator
#     initialize_orchestrator()
    
#     logger.info("System ready!")
    
#     yield
    
#     # Shutdown
#     logger.info("Shutting down...")


# # Create FastAPI app
# app = FastAPI(
#     title=settings.API_TITLE,
#     version=settings.API_VERSION,
#     description=settings.API_DESCRIPTION,
#     lifespan=lifespan
# )

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=settings.CORS_ORIGINS,
#     allow_credentials=settings.CORS_CREDENTIALS,
#     allow_methods=settings.CORS_METHODS,
#     allow_headers=settings.CORS_HEADERS
# )

# # Include routers
# app.include_router(rag_router)
# app.include_router(agent_router)
# app.include_router(graph_router)
# app.include_router(document_router)


# @app.get("/")
# async def root():
#     """Root endpoint"""
#     return {
#         "message": "Advanced Agentic RAG System",
#         "version": settings.API_VERSION,
#         "features": [
#             "Adaptive RAG Strategy Selection",
#             "ReAct Agent (Reasoning + Acting)",
#             "Multi-Agent Collaboration (Researcher + Writer + Critic)",
#             "Knowledge Graph RAG",
#             "Conversation Memory",
#             "Document Processing"
#         ],
#         "docs": "/docs",
#         "health": "/health"
#     }


# @app.get("/health")
# async def health_check():
#     """Health check endpoint"""
#     from app.core.dependencies import get_orchestrator
    
#     orchestrator = get_orchestrator()
    
#     return {
#         "status": "healthy",
#         "timestamp": datetime.now().isoformat(),
#         "version": settings.API_VERSION,
#         "components": {
#             "llm": "operational",
#             "vectorstore": "operational",
#             "graph": "operational",
#             "agents": "operational"
#         },
#         "stats": {
#             "documents": orchestrator.vectorstore.get_count(),
#             "graph_nodes": orchestrator.graph_builder.graph.number_of_nodes(),
#             "graph_edges": orchestrator.graph_builder.graph.number_of_edges()
#         }
#     }


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(
#         "main:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=True
#     )




"""
COMPLETE main.py with all routers
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging.config
from pathlib import Path

from app.core.config import settings
from app.routers.rag_router import router as rag_router
from app.routers.agent_router import agent_router
from app.routers.graph_router import graph_router
from app.routers.document_router import document_router
from app.routers.admin_router import router as admin_router

# Configure logging
Path(settings.LOG_FILE).parent.mkdir(parents=True, exist_ok=True)

logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": settings.LOG_FILE,
            "formatter": "default",
            "mode": "a"
        }
    },
    "root": {
        "level": settings.LOG_LEVEL,
        "handlers": ["console", "file"]
    }
})

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    logger.info("=" * 60)
    logger.info("Starting Advanced Agentic RAG System...")
    logger.info("=" * 60)
    
    # Create necessary directories
    Path("data/vectors").mkdir(parents=True, exist_ok=True)
    Path("data/graphs").mkdir(parents=True, exist_ok=True)
    Path("data/documents").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("uploads").mkdir(parents=True, exist_ok=True)
    
    # Initialize orchestrator
    from app.core.dependencies import initialize_orchestrator
    initialize_orchestrator()
    
    logger.info("✓ Orchestrator initialized")
    logger.info("✓ All services ready")
    logger.info(f"✓ API Documentation: http://localhost:8000/docs")
    logger.info("=" * 60)
    
    yield
    
    logger.info("Shutting down Advanced Agentic RAG System...")


# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_CREDENTIALS,
    allow_methods=settings.CORS_METHODS,
    allow_headers=settings.CORS_HEADERS
)

# Include routers
app.include_router(rag_router)
app.include_router(agent_router)
app.include_router(graph_router)
app.include_router(document_router)
app.include_router(admin_router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Advanced Agentic RAG System",
        "version": settings.API_VERSION,
        "description": "Multi-agent RAG system with adaptive strategies",
        "features": [
            "✓ Multi-Agent Collaboration (Researcher + Writer + Critic)",
            "✓ ReAct Pattern (Reasoning + Acting)",
            "✓ Adaptive RAG Strategy Selection",
            "✓ Knowledge Graph RAG",
            "✓ Multiple RAG Strategies",
            "✓ Conversation Memory",
            "✓ Document Processing"
        ],
        "endpoints": {
            "documentation": "/docs",
            "health": "/health",
            "rag_query": "/api/rag/query",
            "multi_agent": "/api/agents/collaborate",
            "graph_query": "/api/graph/query",
            "upload_document": "/api/documents/upload"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    from app.core.dependencies import get_orchestrator
    
    try:
        orchestrator = get_orchestrator()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": settings.API_VERSION,
            "components": {
                "llm_service": "operational",
                "vector_store": "operational",
                "knowledge_graph": "operational",
                "agents": "operational",
                "memory": "operational"
            },
            "stats": {
                "total_documents": len(orchestrator.documents),
                "total_chunks": orchestrator.vectorstore.get_count(),
                "graph_nodes": orchestrator.graph_builder.graph.number_of_nodes(),
                "graph_edges": orchestrator.graph_builder.graph.number_of_edges(),
                "active_sessions": len(orchestrator.memory_store.sessions)
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )