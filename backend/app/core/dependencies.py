"""
===================================================================
app/core/dependencies.py - Dependency Injection for Services
===================================================================
Compatible with your .env configuration
"""
from functools import lru_cache
from typing import Optional, List
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================
# LLM Service (GROQ)
# ============================================

class LLMService:
    """
    LLM Service using GROQ API
    Configured from your .env file
    """
    
    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("LLM_MODEL", "llama-3.1-70b-versatile")
        self.temperature = temperature or float(os.getenv("TEMPERATURE", "0.7"))
        self.max_tokens = max_tokens or int(os.getenv("MAX_TOKENS", "2000"))
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        # Initialize GROQ client
        self.client = self._initialize_client()
        logger.info(f"LLM Service initialized with model: {self.model}")
    
    def _initialize_client(self):
        """Initialize GROQ client"""
        try:
            from groq import Groq
            return Groq(api_key=self.api_key)
        except ImportError:
            raise ImportError("Groq package not installed. Run: pip install groq")
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate text from prompt using GROQ
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            **kwargs: Additional parameters (temperature, max_tokens)
        
        Returns:
            Generated text
        """
        try:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens)
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            raise Exception(f"LLM generation failed: {str(e)}")
    
    def generate_sync(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Synchronous version of generate (for non-async contexts)"""
        try:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens)
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            raise Exception(f"LLM generation failed: {str(e)}")


# ============================================
# Embedding Service
# ============================================

class EmbeddingService:
    """
    Embedding Service using sentence-transformers
    Configured from your .env file
    """
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL", 
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        logger.info("Embedding model loaded successfully")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for single text
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector as list of floats
        """
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise Exception(f"Embedding generation failed: {str(e)}")
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {str(e)}")
            raise Exception(f"Batch embedding generation failed: {str(e)}")
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()


# ============================================
# Vector Store (ChromaDB)
# ============================================

class VectorStore:
    """
    Vector Store using ChromaDB
    Configured from your .env file
    """
    
    def __init__(
        self,
        persist_directory: str = None,
        collection_name: str = None
    ):
        self.persist_directory = persist_directory or os.getenv(
            "CHROMA_PERSIST_DIR", 
            "./data/vectors"
        )
        self.collection_name = collection_name or os.getenv(
            "COLLECTION_NAME", 
            "advanced_rag"
        )
        
        # Ensure directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"VectorStore initialized with collection: {self.collection_name}")
        logger.info(f"Current documents count: {self.collection.count()}")
    
    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadata: List[dict] = None,
        ids: List[str] = None
    ):
        """
        Add documents to vector store
        
        Args:
            documents: List of document texts
            embeddings: List of embedding vectors
            metadata: List of metadata dicts (optional)
            ids: List of document IDs (optional, will generate if not provided)
        """
        try:
            if metadata is None:
                metadata = [{}] * len(documents)
            
            if ids is None:
                import uuid
                ids = [str(uuid.uuid4()) for _ in range(len(documents))]
            
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadata,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise Exception(f"Failed to add documents: {str(e)}")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = None
    ) -> List[dict]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
        
        Returns:
            List of results with documents, metadata, and distances
        """
        try:
            top_k = top_k or int(os.getenv("TOP_K_RESULTS", "5"))
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results["documents"][0])):
                formatted_results.append({
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "id": results["ids"][0][i]
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise Exception(f"Search failed: {str(e)}")
    
    def delete_collection(self):
        """Delete the entire collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {str(e)}")
            raise Exception(f"Failed to delete collection: {str(e)}")
    
    def reset_collection(self):
        """Reset collection (delete and recreate)"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Reset collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to reset collection: {str(e)}")
            raise Exception(f"Failed to reset collection: {str(e)}")
    
    def get_count(self) -> int:
        """Get number of documents in collection"""
        return self.collection.count()
    
    def get_collection_info(self) -> dict:
        """Get collection information"""
        return {
            "name": self.collection_name,
            "count": self.collection.count(),
            "persist_directory": self.persist_directory
        }


# ============================================
# Graph Store (if enabled)
# ============================================

class GraphStore:
    """
    Knowledge Graph Store
    Used when ENABLE_GRAPH_RAG=true
    """
    
    def __init__(self, graph_path: str = None):
        self.graph_path = graph_path or os.getenv(
            "GRAPH_STORE_PATH",
            "./data/graphs/knowledge_graph.pkl"
        )
        self.max_nodes = int(os.getenv("MAX_GRAPH_NODES", "1000"))
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)
        
        # Initialize graph
        try:
            import networkx as nx
            import pickle
            
            if os.path.exists(self.graph_path):
                with open(self.graph_path, 'rb') as f:
                    self.graph = pickle.load(f)
                logger.info(f"Loaded existing graph with {len(self.graph.nodes)} nodes")
            else:
                self.graph = nx.DiGraph()
                logger.info("Created new knowledge graph")
                
        except ImportError:
            raise ImportError("networkx not installed. Run: pip install networkx")
    
    def add_entity(self, entity_id: str, entity_type: str, properties: dict = None):
        """Add entity to graph"""
        self.graph.add_node(
            entity_id,
            type=entity_type,
            properties=properties or {}
        )
    
    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        properties: dict = None
    ):
        """Add relationship between entities"""
        self.graph.add_edge(
            source_id,
            target_id,
            type=relation_type,
            properties=properties or {}
        )
    
    def save(self):
        """Save graph to disk"""
        import pickle
        with open(self.graph_path, 'wb') as f:
            pickle.dump(self.graph, f)
        logger.info(f"Saved graph with {len(self.graph.nodes)} nodes")
    
    def get_neighbors(self, entity_id: str) -> list:
        """Get neighbors of an entity"""
        if entity_id in self.graph:
            return list(self.graph.neighbors(entity_id))
        return []


# ============================================
# Dependency Factory Functions
# ============================================

@lru_cache()
def get_llm_service() -> LLMService:
    """
    Get singleton LLM service instance
    Uses GROQ with configuration from .env
    """
    return LLMService()


@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """
    Get singleton embedding service instance
    Uses sentence-transformers with configuration from .env
    """
    return EmbeddingService()


@lru_cache()
def get_vectorstore() -> VectorStore:
    """
    Get singleton vector store instance
    Uses ChromaDB with configuration from .env
    """
    return VectorStore()


def get_graph_store() -> Optional[GraphStore]:
    """
    Get graph store instance if enabled
    Returns None if ENABLE_GRAPH_RAG is not true
    """
    if os.getenv("ENABLE_GRAPH_RAG", "false").lower() == "true":
        return GraphStore()
    return None


# ============================================
# Helper Functions
# ============================================

def reset_dependencies():
    """
    Reset all cached dependencies
    Useful for testing or when configuration changes
    """
    get_llm_service.cache_clear()
    get_embedding_service.cache_clear()
    get_vectorstore.cache_clear()
    logger.info("Cleared all dependency caches")


def get_agent_config() -> dict:
    """Get agent configuration from environment"""
    return {
        "enable_multi_agent": os.getenv("ENABLE_MULTI_AGENT", "true").lower() == "true",
        "max_agent_iterations": int(os.getenv("MAX_AGENT_ITERATIONS", "5")),
        "max_react_steps": int(os.getenv("MAX_REACT_STEPS", "5"))
    }


def get_text_processing_config() -> dict:
    """Get text processing configuration"""
    return {
        "chunk_size": int(os.getenv("CHUNK_SIZE", "1000")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "200")),
        "max_memory_messages": int(os.getenv("MAX_MEMORY_MESSAGES", "10"))
    }


def get_upload_config() -> dict:
    """Get file upload configuration"""
    import json
    
    allowed_extensions = os.getenv("ALLOWED_EXTENSIONS", '[".pdf", ".txt", ".docx", ".md"]')
    
    return {
        "upload_dir": os.getenv("UPLOAD_DIR", "./uploads"),
        "allowed_extensions": json.loads(allowed_extensions)
    }


# ============================================
# Orchestrator (Main System Controller)
# ============================================

class RAGOrchestrator:
    """
    Main orchestrator for the RAG system
    Coordinates all services and agents
    """
    
    def __init__(self):
        self.llm_service = get_llm_service()
        self.embedding_service = get_embedding_service()
        self.vectorstore = get_vectorstore()
        self.graph_store = get_graph_store()
        
        # Initialize components
        self.documents = []
        self.memory_store = {"sessions": {}}
        
        # Initialize graph builder if enabled
        if self.graph_store:
            try:
                import networkx as nx
                self.graph_builder = type('GraphBuilder', (), {
                    'graph': self.graph_store.graph if hasattr(self.graph_store, 'graph') else nx.DiGraph()
                })()
            except:
                self.graph_builder = None
        else:
            self.graph_builder = None
        
        logger.info("RAGOrchestrator initialized")
    
    def get_stats(self) -> dict:
        """Get system statistics"""
        stats = {
            "total_documents": len(self.documents),
            "total_chunks": self.vectorstore.get_count(),
            "active_sessions": len(self.memory_store.get("sessions", {}))
        }
        
        if self.graph_builder and hasattr(self.graph_builder, 'graph'):
            stats["graph_nodes"] = self.graph_builder.graph.number_of_nodes()
            stats["graph_edges"] = self.graph_builder.graph.number_of_edges()
        else:
            stats["graph_nodes"] = 0
            stats["graph_edges"] = 0
        
        return stats


# Global orchestrator instance
_orchestrator = None


def initialize_orchestrator():
    """Initialize the global orchestrator"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = RAGOrchestrator()
        logger.info("Global orchestrator initialized")


def get_orchestrator() -> RAGOrchestrator:
    """Get the global orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        initialize_orchestrator()
    return _orchestrator


# ============================================
# Health Check
# ============================================

def check_dependencies_health() -> dict:
    """
    Check health of all dependencies
    Returns status dict
    """
    health = {
        "llm_service": "unknown",
        "embedding_service": "unknown",
        "vector_store": "unknown",
        "graph_store": "unknown"
    }
    
    try:
        llm = get_llm_service()
        health["llm_service"] = "operational"
    except Exception as e:
        health["llm_service"] = f"error: {str(e)}"
    
    try:
        embedder = get_embedding_service()
        health["embedding_service"] = "operational"
    except Exception as e:
        health["embedding_service"] = f"error: {str(e)}"
    
    try:
        vs = get_vectorstore()
        health["vector_store"] = "operational"
    except Exception as e:
        health["vector_store"] = f"error: {str(e)}"
    
    try:
        if os.getenv("ENABLE_GRAPH_RAG", "false").lower() == "true":
            gs = get_graph_store()
            health["graph_store"] = "operational"
        else:
            health["graph_store"] = "disabled"
    except Exception as e:
        health["graph_store"] = f"error: {str(e)}"
    
    return health