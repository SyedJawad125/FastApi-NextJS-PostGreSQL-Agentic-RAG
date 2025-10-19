"""
app/services/orchestrator.py - Main RAG Orchestrator
Coordinates all strategies, agents, and services
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

from app.core.enums import RAGStrategy
from app.core.config import settings

# Services
from app.services.groq_service import GroqService
from app.services.embedding_service import EmbeddingService
from app.services.vectorstore import VectorStoreService
from app.services.memory_store import MemoryStore

# Agents
from app.agents.researcher_agent import ResearcherAgent
from app.agents.writer_agent import WriterAgent
from app.agents.critic_agent import CriticAgent
from app.agents.react_agent import ReActAgent
from app.agents.coordinator import CoordinatorAgent

# Tools
from app.tools.search_tool import VectorSearchTool
from app.tools.graph_query_tool import GraphQueryTool
from app.tools.calculator_tool import CalculatorTool
from app.tools.summarizer_tool import SummarizerTool

# Graph
from app.graph.graph_builder import GraphBuilder
from app.graph.graph_service import GraphService

# Strategies
from app.strategies.simple_rag import SimpleRAGStrategy
from app.strategies.agentic_rag import AgenticRAGStrategy
from app.strategies.graph_rag import GraphRAGStrategy
from app.strategies.multi_agent_rag import MultiAgentRAGStrategy
from app.strategies.hybrid_rag import HybridRAGStrategy
from app.strategies.strategy_selector import StrategySelector

logger = logging.getLogger(__name__)


class RAGOrchestrator:
    """
    Main orchestrator for the entire RAG system
    Manages all strategies, agents, and services
    """
    
    def __init__(self):
        logger.info("Initializing RAG Orchestrator...")
        
        # Initialize core services
        self.llm_service = GroqService()
        self.embedding_service = EmbeddingService()
        self.vectorstore = VectorStoreService(self.embedding_service)
        self.memory_store = MemoryStore()
        
        # Initialize graph components
        self.graph_builder = GraphBuilder()
        self.graph_service = GraphService(self.llm_service, self.graph_builder)
        
        # Initialize tools
        self.tools = self._initialize_tools()
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Initialize strategies
        self.strategies = self._initialize_strategies()
        
        # Initialize strategy selector
        self.strategy_selector = StrategySelector(self.llm_service)
        
        # Document tracking
        self.documents: Dict[str, Dict] = {}
        
        logger.info("RAG Orchestrator initialized successfully")
    
    def _initialize_tools(self) -> Dict[str, Any]:
        """Initialize all agent tools"""
        return {
            "vector_search": VectorSearchTool(self.vectorstore),
            "graph_query": GraphQueryTool(self.graph_service),
            "calculator": CalculatorTool(),
            "summarizer": SummarizerTool(self.llm_service)
        }
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all agents"""
        
        # ReAct agent with all tools
        react_agent = ReActAgent(
            llm_service=self.llm_service,
            tools=list(self.tools.values())
        )
        
        # Researcher agent
        researcher = ResearcherAgent(
            llm_service=self.llm_service,
            tools=[self.tools["vector_search"], self.tools["graph_query"]]
        )
        
        # Writer agent
        writer = WriterAgent(
            llm_service=self.llm_service,
            tools=[self.tools["summarizer"]]
        )
        
        # Critic agent
        critic = CriticAgent(
            llm_service=self.llm_service
        )
        
        # Coordinator agent
        coordinator = CoordinatorAgent(
            researcher=researcher,
            writer=writer,
            critic=critic
        )
        
        return {
            "react": react_agent,
            "researcher": researcher,
            "writer": writer,
            "critic": critic,
            "coordinator": coordinator
        }
    
    def _initialize_strategies(self) -> Dict[RAGStrategy, Any]:
        """Initialize all RAG strategies"""
        return {
            RAGStrategy.SIMPLE: SimpleRAGStrategy(
                self.vectorstore,
                self.llm_service
            ),
            RAGStrategy.AGENTIC: AgenticRAGStrategy(
                self.agents["react"]
            ),
            RAGStrategy.GRAPH: GraphRAGStrategy(
                self.graph_service,
                self.llm_service
            ),
            RAGStrategy.MULTI_AGENT: MultiAgentRAGStrategy(
                self.agents["coordinator"]
            ),
            RAGStrategy.HYBRID: HybridRAGStrategy(
                self.vectorstore,
                self.graph_service,
                self.llm_service
            )
        }
    
    async def execute_query(
        self,
        query: str,
        strategy: RAGStrategy = RAGStrategy.AUTO,
        session_id: Optional[str] = None,
        use_memory: bool = True,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Execute RAG query with specified or auto-selected strategy
        """
        start_time = datetime.now()
        
        # Auto-select strategy if needed
        if strategy == RAGStrategy.AUTO:
            strategy = await self.strategy_selector.select_strategy(query)
            logger.info(f"Auto-selected strategy: {strategy.value}")
        
        # Build context
        context = {
            "top_k": top_k,
            "session_id": session_id
        }
        
        # Add memory if enabled
        if use_memory and session_id:
            context["memory"] = self.memory_store.format_history(session_id, limit=5)
        
        # Execute strategy
        strategy_impl = self.strategies.get(strategy)
        if not strategy_impl:
            raise ValueError(f"Strategy {strategy.value} not implemented")
        
        result = await strategy_impl.execute(query, context)
        
        # Save to memory
        if use_memory and session_id:
            self.memory_store.add_message(session_id, "user", query)
            self.memory_store.add_message(session_id, "assistant", result["answer"])
        
        # Format response
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "query": query,
            "answer": result["answer"],
            "strategy_used": strategy,
            "retrieved_chunks": result.get("retrieved_chunks", []),
            "agent_steps": result.get("agent_steps", []),
            "agents_involved": result.get("agents_involved", []),
            "processing_time": processing_time,
            "metadata": result.get("metadata", {})
        }
    
    async def execute_multi_agent(
        self,
        query: str,
        enable_researcher: bool = True,
        enable_writer: bool = True,
        enable_critic: bool = True,
        max_iterations: int = 3,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute multi-agent collaboration"""
        
        coordinator = self.agents["coordinator"]
        coordinator.max_iterations = max_iterations
        
        context = {"session_id": session_id}
        result = await coordinator.execute(query, context)
        
        return {
            "query": query,
            "final_answer": result["final_answer"],
            "agent_executions": result["agent_executions"],
            "collaboration_summary": result["collaboration_summary"],
            "total_iterations": result["total_iterations"],
            "processing_time": result["processing_time"]
        }
    
    async def execute_graph_query(
        self,
        query: str,
        max_depth: int = 2,
        include_neighbors: bool = True
    ) -> Dict[str, Any]:
        """Execute graph RAG query"""
        start_time = datetime.now()
        
        result = await self.graph_service.query(query, max_depth=max_depth)
        
        # Generate answer using graph information
        entities_text = "\n".join([
            f"- {e['name']} ({e['type']})"
            for e in result.get("entities", [])[:10]
        ])
        
        prompt = f"""Based on the knowledge graph, answer this question:

Question: {query}

Related Entities:
{entities_text}

Answer:"""
        
        answer = await self.llm_service.generate(prompt)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "query": query,
            "answer": answer,
            "entities": result.get("entities", []),
            "relationships": result.get("relationships", []),
            "subgraph_size": len(result.get("entities", [])),
            "processing_time": processing_time
        }
    
    async def process_document(
        self,
        filename: str,
        content: bytes,
        content_type: str,
        extract_graph: bool = True
    ) -> Dict[str, Any]:
        """Process and index document"""
        start_time = datetime.now()
        document_id = str(uuid.uuid4())
        
        # Extract text based on content type
        from app.utils.pdf_reader import PDFReader
        pdf_reader = PDFReader()
        
        if content_type == "application/pdf" or filename.endswith(".pdf"):
            text = pdf_reader.extract_text_from_bytes(content)
        else:
            text = content.decode("utf-8")
        
        # Chunk text
        from app.utils.text_splitter import TextSplitter
        splitter = TextSplitter()
        chunks = splitter.split_text(text)
        
        # Add to vector store
        chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": filename,
                "document_id": document_id,
                "chunk_index": i
            }
            for i in range(len(chunks))
        ]
        
        await self.vectorstore.add_documents(chunks, metadatas, chunk_ids)
        
        # Extract graph if enabled
        entities_count = 0
        relationships_count = 0
        
        if extract_graph and settings.ENABLE_GRAPH_RAG:
            graph_result = await self.graph_service.process_document(text[:5000])
            entities_count = graph_result["entities_extracted"]
            relationships_count = graph_result["relationships_extracted"]
        
        # Track document
        self.documents[document_id] = {
            "id": document_id,
            "filename": filename,
            "chunks": len(chunks),
            "entities": entities_count,
            "relationships": relationships_count,
            "size": len(content),
            "uploaded_at": datetime.now().isoformat()
        }
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "document_id": document_id,
            "filename": filename,
            "status": "success",
            "chunks_created": len(chunks),
            "entities_extracted": entities_count,
            "relationships_extracted": relationships_count,
            "processing_time": processing_time,
            "message": f"Document processed successfully: {len(chunks)} chunks created"
        }
    
    def list_documents(self) -> Dict[str, Any]:
        """List all processed documents"""
        return {
            "documents": list(self.documents.values()),
            "total_count": len(self.documents)
        }
    
    def delete_document(self, document_id: str):
        """Delete document from system"""
        if document_id in self.documents:
            del self.documents[document_id]
            # Note: In production, also delete from vector store
            return True
        return False
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        stats = self.graph_service.get_stats()
        return {
            "total_nodes": stats["total_nodes"],
            "total_edges": stats["total_edges"],
            "connected_components": stats["connected_components"]
        }
    
    def get_graph_visualization(self) -> Dict[str, Any]:
        """Get graph data for visualization"""
        graph = self.graph_builder.graph
        
        nodes = []
        for node_id, node_data in graph.nodes(data=True):
            nodes.append({
                "id": node_id,
                "label": node_data.get("name", node_id),
                "type": node_data.get("type", "unknown")
            })
        
        edges = []
        for source, target, edge_data in graph.edges(data=True):
            edges.append({
                "source": source,
                "target": target,
                "label": edge_data.get("relation_type", "related_to")
            })
        
        return {
            "nodes": nodes[:100],  # Limit for performance
            "edges": edges[:200]
        }


