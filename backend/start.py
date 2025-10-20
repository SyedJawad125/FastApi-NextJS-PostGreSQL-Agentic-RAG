"""
Start script for FastAPI application with proper exclusions
"""
import uvicorn
import os
import sys
from dotenv import load_dotenv

# Fix Windows console encoding for emojis
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║        Advanced Agentic RAG System                        ║
║        Starting FastAPI Server...                         ║
╚═══════════════════════════════════════════════════════════╝

    Server: http://{host}:{port}
    Docs:   http://{host}:{port}/docs
    Reload: {reload}
    """)
    
    # Configure uvicorn with proper exclusions
    config = uvicorn.Config(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        reload_excludes=[
            "data/*",           # Exclude data directory (ChromaDB, graphs)
            "*.db",             # Exclude database files
            "*.db-shm",         # Exclude SQLite shared memory
            "*.db-wal",         # Exclude SQLite write-ahead log
            "logs/*",           # Exclude logs
            "uploads/*",        # Exclude uploads
            "*.pyc",            # Exclude Python cache
            "__pycache__/*",    # Exclude pycache
            ".pytest_cache/*",  # Exclude pytest cache
            "*.pkl",            # Exclude pickle files
        ]
    )
    
    server = uvicorn.Server(config)
    server.run()