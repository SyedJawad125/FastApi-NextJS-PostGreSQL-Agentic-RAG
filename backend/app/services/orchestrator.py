"""
Simplified RAG Orchestrator - Windows Compatible (No Emojis)
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

from app.core.enums import RAGStrategy

logger = logging.getLogger(__name__)


class RAGOrchestrator:
    """
    Simplified orchestrator that works with current dependencies
    """
    
    def __init__(self):
        logger.info("Initializing RAG Orchestrator...")
        
        # Initialize core services from dependencies
        from app.core.dependencies import get_llm_service, get_embedding_service, get_vectorstore
        
        self.llm_service = get_llm_service()
        self.embedding_service = get_embedding_service() 
        self.vectorstore = get_vectorstore()
        
        # Strategy implementations
        self.strategies = {
            RAGStrategy.SIMPLE: self._simple_rag_strategy,
            RAGStrategy.AGENTIC: self._agentic_rag_strategy,
            RAGStrategy.AUTO: self._auto_rag_strategy
        }
        
        logger.info("[OK] RAG Orchestrator initialized successfully")
    
    async def _simple_rag_strategy(self, query: str, context: Dict) -> Dict[str, Any]:
        """Simple RAG implementation"""
        try:
            # Generate embedding for query
            query_embedding = self.embedding_service.embed_text(query)
            
            # Search vector store
            search_results = self.vectorstore.search(
                query_embedding=query_embedding,
                top_k=context.get("top_k", 5)
            )
            
            # Build context from retrieved chunks
            context_text = "\n\n".join([
                f"Document {i+1}: {result['document']}"
                for i, result in enumerate(search_results[:3])  # Use top 3
            ])
            
            # Generate answer using context
            if context_text.strip():
                prompt = f"""Based on the following context, answer the question clearly and concisely.

Context:
{context_text}

Question: {query}

Provide a direct answer based on the context:"""
            else:
                prompt = f"""Answer the following question based on your knowledge:

Question: {query}

Provide a clear and helpful answer:"""
            
            answer = await self.llm_service.generate(prompt)
            
            return {
                "answer": answer,
                "retrieved_chunks": [
                    {
                        "content": result["document"][:200] + "..." if len(result["document"]) > 200 else result["document"],
                        "metadata": result.get("metadata", {}),
                        "score": 1 - result.get("distance", 0)
                    }
                    for result in search_results
                ],
                "confidence": min(0.95, 0.7 + (len(search_results) * 0.05))
            }
            
        except Exception as e:
            logger.error(f"Simple RAG strategy failed: {str(e)}")
            answer = await self.llm_service.generate(query)
            return {
                "answer": f"{answer}\n\nNote: Used fallback mode due to processing error.",
                "retrieved_chunks": [],
                "confidence": 0.5
            }
    
    async def _agentic_rag_strategy(self, query: str, context: Dict) -> Dict[str, Any]:
        """Agentic RAG implementation (enhanced)"""
        try:
            # Get relevant context
            query_embedding = self.embedding_service.embed_text(query)
            search_results = self.vectorstore.search(
                query_embedding=query_embedding,
                top_k=context.get("top_k", 5)
            )
            
            context_text = "\n".join([
                f"- {result['document']}"
                for result in search_results
            ])
            
            # Enhanced prompt for agentic approach
            prompt = f"""You are an AI research assistant. Analyze the question and provided context thoroughly.

CONTEXT FROM KNOWLEDGE BASE:
{context_text if context_text.strip() else "No specific context available from knowledge base."}

USER QUESTION: {query}

Please provide a comprehensive answer that:
1. Uses the context if relevant and available
2. Acknowledges what information comes from the context vs. general knowledge
3. Provides a well-structured, detailed response
4. Notes any limitations or missing information

ANSWER:"""
            
            answer = await self.llm_service.generate(prompt)
            
            return {
                "answer": answer,
                "retrieved_chunks": [
                    {
                        "content": result["document"][:150] + "..." if len(result["document"]) > 150 else result["document"],
                        "metadata": result.get("metadata", {}),
                        "score": 1 - result.get("distance", 0)
                    }
                    for result in search_results
                ],
                "confidence": min(0.9, 0.6 + (len(search_results) * 0.06)),
                "agent_steps": ["context_analysis", "synthesis", "quality_assurance"]
            }
            
        except Exception as e:
            logger.error(f"Agentic RAG strategy failed: {str(e)}")
            answer = await self.llm_service.generate(query)
            return {
                "answer": f"{answer}\n\nNote: Agentic processing unavailable, used standard response.",
                "retrieved_chunks": [],
                "confidence": 0.4,
                "agent_steps": ["fallback"]
            }
    
    async def _auto_rag_strategy(self, query: str, context: Dict) -> Dict[str, Any]:
        """Auto strategy selector based on query complexity"""
        query_lower = query.lower()
        
        # Complex queries - use agentic
        complex_indicators = [
            "analyze", "compare", "evaluate", "critique", "explain in detail",
            "what are the advantages", "what are the disadvantages", "pros and cons"
        ]
        
        # Simple queries - use simple RAG
        simple_indicators = [
            "what is", "who is", "when was", "where is", "define"
        ]
        
        if any(indicator in query_lower for indicator in complex_indicators):
            logger.info("Auto-selected: agentic strategy (complex query)")
            return await self._agentic_rag_strategy(query, context)
        elif any(indicator in query_lower for indicator in simple_indicators):
            logger.info("Auto-selected: simple strategy (simple query)")
            return await self._simple_rag_strategy(query, context)
        else:
            logger.info("Auto-selected: agentic strategy (default)")
            return await self._agentic_rag_strategy(query, context)
    
    async def execute_query(
        self,
        query: str,
        top_k: int = 5,
        session_id: Optional[str] = None,
        strategy: RAGStrategy = RAGStrategy.AUTO
    ) -> Dict[str, Any]:
        """Execute RAG query with specified strategy"""
        start_time = datetime.now()
        
        logger.info(f"Executing query: '{query[:50]}...' with strategy: {strategy.value}")
        
        context = {
            "top_k": top_k,
            "session_id": session_id,
            "query": query
        }
        
        strategy_func = self.strategies.get(strategy, self._auto_rag_strategy)
        result = await strategy_func(query, context)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        response = {
            "answer": result["answer"],
            "strategy_used": strategy,
            "retrieved_chunks": result.get("retrieved_chunks", []),
            "confidence": result.get("confidence", 0.7),
            "processing_time": processing_time
        }
        
        if "agent_steps" in result:
            response["agent_steps"] = result["agent_steps"]
            
        logger.info(f"Query completed in {processing_time:.2f}s with {len(response['retrieved_chunks'])} chunks")
        
        return response
    
    async def process_document(
        self,
        filename: str,
        content: bytes,
        content_type: str
    ) -> Dict[str, Any]:
        """Process and index document"""
        start_time = datetime.now()
        
        try:
            logger.info(f"Processing document: {filename}")
            
            # Extract text
            text = ""
            if content_type == "application/pdf" or filename.endswith(".pdf"):
                text = self._extract_text_from_pdf(content)
            elif content_type == "text/plain" or filename.endswith(".txt"):
                text = content.decode("utf-8")
            else:
                try:
                    text = content.decode("utf-8")
                except:
                    text = "Binary content - text extraction not supported"
            
            # Chunk text
            chunks = self._chunk_text(text)
            
            if chunks:
                # Generate embeddings
                embeddings = self.embedding_service.embed_texts(chunks)
                
                # Prepare metadata
                metadatas = [
                    {
                        "source": filename,
                        "content_type": content_type,
                        "chunk_index": i,
                        "size": len(chunk)
                    }
                    for i, chunk in enumerate(chunks)
                ]
                
                # Generate IDs
                doc_id = str(uuid.uuid4())
                chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
                
                # Add to vector store
                self.vectorstore.add_documents(
                    documents=chunks,
                    embeddings=embeddings,
                    metadata=metadatas,
                    ids=chunk_ids
                )
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                return {
                    "chunks_created": len(chunks),
                    "document_id": doc_id,
                    "processing_time": processing_time,
                    "method": "vector_store",
                    "message": f"Successfully processed {len(chunks)} chunks"
                }
            else:
                return {
                    "chunks_created": 0,
                    "document_id": str(uuid.uuid4()),
                    "processing_time": (datetime.now() - start_time).total_seconds(),
                    "method": "skip",
                    "message": "No text content extracted"
                }
                
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            return {
                "chunks_created": 0,
                "document_id": str(uuid.uuid4()),
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "method": "error",
                "message": f"Processing failed: {str(e)}"
            }
    
    def _extract_text_from_pdf(self, content: bytes) -> str:
        """Extract text from PDF content"""
        try:
            import PyPDF2
            from io import BytesIO
            
            pdf_file = BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except ImportError:
            return "PDF text extraction requires PyPDF2. Install with: pip install PyPDF2"
        except Exception as e:
            return f"PDF extraction error: {str(e)}"
    
    def _chunk_text(self, text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]:
        """Simple text chunking implementation"""
        if not text or len(text.strip()) == 0:
            return []
            
        words = text.split()
        chunks = []
        
        if len(words) <= chunk_size:
            return [" ".join(words)]
        
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            
            if end >= len(words):
                break
                
            start = end - chunk_overlap
            
        return chunks


# Global instance
_rag_orchestrator = None

def get_rag_orchestrator() -> RAGOrchestrator:
    """Get the global RAG orchestrator instance"""
    global _rag_orchestrator
    if _rag_orchestrator is None:
        _rag_orchestrator = RAGOrchestrator()
    return _rag_orchestrator