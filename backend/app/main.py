"""
===================================================================
app/main.py - Fixed FastAPI Application Entry Point
===================================================================
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging.config
from pathlib import Path
from datetime import datetime
from app.routers.rag_router import router as rag_router
from app.core.config import settings, init_db, engine

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
    logger.info(f"Starting {settings.API_TITLE} v{settings.API_VERSION}")
    logger.info("=" * 60)
    
    # Create necessary directories
    directories = [
        "data/vectors",
        "data/graphs",
        "data/documents",
        "logs",
        "uploads"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory created: {directory}/")
    
    # Initialize database
    try:
        init_db()
        logger.info("[OK] Database initialized successfully")
    except Exception as e:
        logger.error(f"[ERROR] Database initialization failed: {str(e)}")
        raise
    
    # Initialize orchestrator
    try:
        from app.core.dependencies import initialize_orchestrator
        initialize_orchestrator()
        logger.info("[OK] Orchestrator initialized")
    except Exception as e:
        logger.error(f"[ERROR] Orchestrator initialization failed: {str(e)}")
        # Continue anyway - orchestrator is optional
    
    # Check dependencies health
    try:
        from app.core.dependencies import check_dependencies_health
        health = check_dependencies_health()
        for service, status in health.items():
            if status == "operational":
                logger.info(f"[OK] {service}: {status}")
            elif status == "disabled":
                logger.info(f"[DISABLED] {service}: {status}")
            else:
                logger.warning(f"[ERROR] {service}: {status}")
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
    
    logger.info("[OK] All services ready")
    logger.info(f"[INFO] API Documentation: http://localhost:8000/docs")
    logger.info("=" * 60)
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    try:
        engine.dispose()
        logger.info("[OK] Database connections closed")
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")


# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="""
    Advanced Agentic RAG System with Multi-Agent Architecture

    ## Features
    * Document Processing: PDF, TXT, DOCX, MD
    * Semantic Search: ChromaDB vector store with sentence-transformers
    * Multiple RAG Strategies: Simple, Agentic, Multi-Agent, Auto
    * Multi-Agent System: Researcher, Writer, Critic agents
    * ReAct Pattern: Reasoning + Acting for intelligent responses
    * Knowledge Graph: Optional graph-based RAG
    * Session Management: Track conversations and context
    * PostgreSQL: Persistent storage with full history

    ## Quick Start
    1. Upload a document: POST /api/rag/upload
    2. Query the system: POST /api/rag/query
    3. Multi-agent query: POST /api/rag/multi-agent-query
    4. Check health: GET /api/rag/health
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_CREDENTIALS,
    allow_methods=settings.CORS_METHODS,
    allow_headers=settings.CORS_HEADERS
)

# Import and include routers - CORRECTED ORDER
logger.info("Loading routers...")

# RAG Router - ADD PREFIX HERE
try:
    
    # ✅ CORRECTED: Include WITH prefix
    app.include_router(rag_router)
    logger.info("[OK] RAG router loaded")
    
    # Debug: Print registered routes
    for route in app.routes:
        if hasattr(route, 'path') and '/api/rag' in route.path:
            logger.info(f"  - {route.methods} {route.path}")
            
except ImportError as e:
    logger.error(f"[ERROR] RAG router not found: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    logger.error(f"[ERROR] Failed to load RAG router: {e}")
    import traceback
    traceback.print_exc()

# Agent Router
try:
    from app.routers.agent_router import agent_router
    app.include_router(agent_router, prefix="/api/agent", tags=["Agent"])
    logger.info("[OK] Agent router loaded")
except ImportError as e:
    logger.warning(f"Agent router not found: {e}")
except Exception as e:
    logger.warning(f"Failed to load agent router: {e}")

# Graph Router
try:
    from app.routers.graph_router import graph_router
    app.include_router(graph_router, prefix="/api/graph", tags=["Graph"])
    logger.info("[OK] Graph router loaded")
except ImportError as e:
    logger.warning(f"Graph router not found: {e}")
except Exception as e:
    logger.warning(f"Failed to load graph router: {e}")

# Document Router
try:
    from app.routers.document_router import document_router
    app.include_router(document_router, prefix="/api/documents", tags=["Documents"])
    logger.info("[OK] Document router loaded")
except ImportError as e:
    logger.warning(f"Document router not found: {e}")
except Exception as e:
    logger.warning(f"Failed to load document router: {e}")

# Admin Router
try:
    from app.routers.admin_router import router as admin_router
    app.include_router(admin_router, prefix="/api/admin", tags=["Admin"])
    logger.info("[OK] Admin router loaded")
except ImportError as e:
    logger.warning(f"Admin router not found: {e}")
except Exception as e:
    logger.warning(f"Failed to load admin router: {e}")

logger.info("All routers loaded")


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": settings.API_TITLE,
        "version": settings.API_VERSION,
        "description": "Multi-agent RAG system with adaptive strategies",
        "status": "operational",
        "features": [
            "Multi-Agent Collaboration (Researcher + Writer + Critic)",
            "ReAct Pattern (Reasoning + Acting)",
            "Adaptive RAG Strategy Selection",
            "Knowledge Graph RAG",
            "Multiple RAG Strategies",
            "Conversation Memory",
            "Document Processing (PDF, TXT, DOCX, MD)",
            "PostgreSQL Storage",
            "ChromaDB Vector Store"
        ],
        "endpoints": {
            "documentation": "/docs",
            "alternative_docs": "/redoc",
            "health": "/health",
            "rag_query": "/api/rag/query",
            "multi_agent": "/api/rag/multi-agent-query",
            "upload_document": "/api/rag/upload",
            "list_documents": "/api/rag/documents",
            "statistics": "/api/rag/stats"
        },
        "powered_by": {
            "llm": f"Groq ({settings.LLM_MODEL})",
            "embeddings": settings.EMBEDDING_MODEL,
            "vector_store": settings.VECTOR_STORE_TYPE.upper(),
            "database": "PostgreSQL",
            "framework": "FastAPI"
        }
    }


# Health check endpoint
@app.get("/health", tags=["Root"])
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        from app.core.dependencies import get_orchestrator
        from sqlalchemy import text
        
        # Check database
        db_status = "operational"
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as e:
            db_status = f"error: {str(e)}"
        
        # Get orchestrator stats
        try:
            orchestrator = get_orchestrator()
            stats = orchestrator.get_stats()
        except Exception as e:
            logger.error(f"Failed to get orchestrator stats: {e}")
            stats = {
                "total_documents": 0,
                "total_chunks": 0,
                "graph_nodes": 0,
                "graph_edges": 0,
                "active_sessions": 0
            }
        
        return {
            "status": "healthy" if db_status == "operational" else "degraded",
            "timestamp": datetime.now().isoformat(),
            "version": settings.API_VERSION,
            "components": {
                "llm_service": "operational",
                "embedding_service": "operational",
                "vector_store": "operational",
                "knowledge_graph": "operational" if settings.ENABLE_GRAPH_RAG else "disabled",
                "database": db_status,
                "agents": "operational" if settings.ENABLE_MULTI_AGENT else "disabled",
                "memory": "operational"
            },
            "stats": stats,
            "configuration": {
                "llm_model": settings.LLM_MODEL,
                "embedding_model": settings.EMBEDDING_MODEL,
                "vector_store": settings.VECTOR_STORE_TYPE,
                "graph_enabled": settings.ENABLE_GRAPH_RAG,
                "multi_agent_enabled": settings.ENABLE_MULTI_AGENT
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle all unhandled exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "path": str(request.url),
            "timestamp": datetime.now().isoformat()
        }
    )


# Debug endpoint to list all routes
@app.get("/debug/routes", tags=["Debug"])
async def list_routes():
    """List all registered routes (for debugging)"""
    routes = []
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            routes.append({
                "path": route.path,
                "methods": list(route.methods),
                "name": route.name
            })
    return {"total_routes": len(routes), "routes": routes}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )