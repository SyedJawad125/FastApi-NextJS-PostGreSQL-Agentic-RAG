# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware

# # Enable SQLAlchemy logging
# import logging
# logging.basicConfig()
# logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# # Import Base and engine from database (use absolute import)
# from app.database import engine, Base

# # Import all models explicitly
# from app.models.user import User
# from app.models.role import Role
# from app.models.permission import Permission
# from app.models.image_category import ImageCategory
# from app.models.image import Image

# # Import routers
# from app.routers import (
#     employee, auth, user, 
#     role, permission,image_category, image, 
#     house_price_model, rag_router
# )

# app = FastAPI(
#     title="HRM System with House Price Prediction",
#     version="1.0.0",
#     description="An API for managing HRM features with machine learning capabilities",
#     openapi_tags=[
        
#         {
#             "name": "Employees",
#             "description": "Employee profile management"
#         },
#         {
#             "name": "Users",
#             "description": "User login and registration"
#         },
#         {
#             "name": "Roles",
#             "description": "Roles profile management"
#         },
#         {
#             "name": "Image Categories",
#             "description": "Image categories management"
#         },
#         {
#             "name": "Machine Learning",
#             "description": "House price prediction model"
#         }
#     ]
# )

# # CORS settings to allow requests from frontend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],  # Next.js frontend origin
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# from fastapi.staticfiles import StaticFiles
# import os
# # Mount static files
# app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# @app.get("/api/hello")
# def read_root():
#     return {"message": "Hello from FastAPI!"}

# # Include all routers
# app.include_router(auth.router)
# app.include_router(user.router)
# app.include_router(employee.router)
# app.include_router(role.router)
# app.include_router(permission.router)
# app.include_router(image_category.router)
# app.include_router(image.router)
# app.include_router(house_price_model.router)  # Added ML router
# app.include_router(rag_router.router)

# @app.on_event("startup")
# def startup_event():
#     print("Creating database tables...")
#     print(f"Engine URL: {engine.url}")
#     print(f"Tables in metadata: {Base.metadata.tables.keys()}")
#     Base.metadata.create_all(bind=engine)
#     print("Database tables created.")

#     # Load ML model on startup
#     try:
#         from app.models.house_price_model import train_and_save_model
#         print("Initializing ML model...")
#         train_and_save_model()
#         print("ML model ready")
#     except Exception as e:
#         print(f"Error loading ML model: {str(e)}")


# @app.get("/")
# def root():
#     return {"message": "Welcome to HRM API with Machine Learning capabilities"}

# @app.get("/")
# def read_root():
#     return {"message": "Customer Churn Prediction API"}

# @app.get("/ping")
# async def health_check():
#     return {"status": "healthy"}

# @app.get("/test")
# def test():
#     return {"status": "working"}

# @app.get("/")
# def root():
#     return {"message": "RAG Application with Groq is running 🚀"}





# ============================================================================
# MAIN APPLICATION
# ============================================================================

"""
main.py - FastAPI Application Entry Point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging.config

from app.core.config import settings
from app.routers.rag_router import router as rag_router
from app.routers.agent_router import agent_router
from app.routers.graph_router import graph_router
from app.routers.document_router import document_router

# Configure logging
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
            "formatter": "default"
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": settings.LOG_FILE,
            "formatter": "default"
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
    # Startup
    logger.info("Starting Advanced Agentic RAG System...")
    
    # Initialize orchestrator
    from app.core.dependencies import initialize_orchestrator
    initialize_orchestrator()
    
    logger.info("System ready!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")


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


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Advanced Agentic RAG System",
        "version": settings.API_VERSION,
        "features": [
            "Adaptive RAG Strategy Selection",
            "ReAct Agent (Reasoning + Acting)",
            "Multi-Agent Collaboration (Researcher + Writer + Critic)",
            "Knowledge Graph RAG",
            "Conversation Memory",
            "Document Processing"
        ],
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    from app.core.dependencies import get_orchestrator
    
    orchestrator = get_orchestrator()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": settings.API_VERSION,
        "components": {
            "llm": "operational",
            "vectorstore": "operational",
            "graph": "operational",
            "agents": "operational"
        },
        "stats": {
            "documents": orchestrator.vectorstore.get_count(),
            "graph_nodes": orchestrator.graph_builder.graph.number_of_nodes(),
            "graph_edges": orchestrator.graph_builder.graph.number_of_edges()
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )